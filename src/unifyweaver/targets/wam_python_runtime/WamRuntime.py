"""
WamRuntime.py — Python WAM runtime library
Trail-based mutable state. No external dependencies.
"""

from __future__ import annotations
import copy
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Optional, Callable, Dict, List, Tuple


# -- Term representation ----------------------------------------------------

@dataclass
class Atom:
    name: str
    def __repr__(self): return self.name

@dataclass
class Compound:
    functor: str
    args: List['Term']
    def __repr__(self): return f"{self.functor}({', '.join(map(repr,self.args))})"

@dataclass
class Var:
    """Unbound logic variable -- mutable ref cell."""
    ref: List  # [None] when unbound, [value] when bound
    id: int    # unique id for display
    def __repr__(self):
        if self.ref[0] is None: return f"_V{self.id}"
        return repr(self.ref[0])

@dataclass
class Int:
    n: int
    def __repr__(self): return str(self.n)

@dataclass
class Float:
    f: float
    def __repr__(self): return str(self.f)

@dataclass
class Ref:
    """Heap address reference."""
    addr: int

@dataclass
class StreamHandle:
    """Host stream wrapper that survives WAM state snapshots by identity."""
    handle: Any
    def __deepcopy__(self, memo): return self

Term = Atom | Compound | Var | Int | Float | Ref | StreamHandle

# -- Atom interning cache -----------------------------------------------------
# Ensures `make_atom("foo") is make_atom("foo")` — pointer-equality for equal
# atom names, eliminating redundant string hashing in hot loops (unify, deref).

_atom_cache: Dict[str, Atom] = {}

def make_atom(name: str) -> Atom:
    """Return a cached Atom instance for the given name."""
    a = _atom_cache.get(name)
    if a is None:
        a = Atom(name)
        _atom_cache[name] = a
    return a


# -- WAM State --------------------------------------------------------------

@dataclass
class ChoicePoint:
    n_args: int
    saved_regs: List
    saved_e: int
    saved_cp: Any          # label/callable
    next_clause: Any       # label/callable for retry
    trail_top: int         # trail mark for undo
    heap_top: int          # heap mark for trimming
    saved_b: int           # parent choice point index


@dataclass
class Environment:
    saved_e: int
    saved_cp: Any
    cut_b: int
    perm_vars: List        # Y registers


@dataclass
class CatcherFrame:
    """Side-stack frame for Prolog catch/3."""
    catcher_term: Term
    recovery_term: Term
    saved_cp: Any
    snapshot: Any


class WamState:
    def __init__(self):
        self.regs: List = [None] * 512   # A1->1, X1->129, Y1->301 (1-indexed, same as all targets)
        self.heap: Dict[int, Term] = {}  # O(1) put/get/trim via dict keyed by int addr
        self.heap_len: int = 0           # cached length — avoids len() on hot path
        self.stack: List = []            # environments + choice points (typed)
        self.trail: List[int] = []       # heap addresses to unbind
        self.trail_len: int = 0          # cached trail length
        self.pdl: List = []              # unification push-down list
        self.mode: str = 'write'
        self.s: int = 0                  # structure pointer (read mode)
        self.cp: Any = None              # continuation pointer
        self.b: int = -1                 # top choice point index in stack
        self.e: int = -1                 # top environment index in stack
        self.cut_b: int = -1
        self._var_counter: int = 0
        # Structure write context: (compound_ref, next_arg_idx) or None
        self.write_ctx: Any = None
        self.write_stack: List = []
        # Structure read context: (compound_ref, next_arg_idx) or None
        self.read_ctx: Any = None
        self.read_stack: List = []
        # Indexed fact tables — Rust-parity O(1) fact lookups keyed by predicate.
        # indexed_atom_fact2: pred_key -> { key_atom -> [value_atom, ...] }
        self.indexed_atom_fact2: Dict[str, Dict[str, List[str]]] = {}
        # indexed_weighted_edge: pred_key -> { key_atom -> [(value_atom, weight), ...] }
        self.indexed_weighted_edge: Dict[str, Dict[str, List[Tuple[str, float]]]] = {}
        # Side stack for Prolog catch/3 frames. Python exceptions provide the
        # unwind mechanism; these frames carry WAM-level state snapshots.
        self.catcher_frames: List[CatcherFrame] = []
        self.temp_y_regs: Optional[List] = None
        self.input_pushback: List[str] = []
        self.stream_pushback: Dict[int, List[str]] = {}

    def fresh_var(self) -> Var:
        v = Var(ref=[None], id=self._var_counter)
        self._var_counter += 1
        return v

    def b_trail_top(self) -> int:
        """Trail top of current choice point (for conditional trailing)."""
        if self.b < 0:
            return 0
        cp = self.stack[self.b]
        if isinstance(cp, ChoicePoint):
            return cp.trail_top
        return 0

    def b_heap_top(self) -> int:
        """Heap top at time of current choice point creation."""
        if self.b < 0:
            return 0
        cp = self.stack[self.b]
        if isinstance(cp, ChoicePoint):
            return cp.heap_top
        return 0


# -- Register helpers -------------------------------------------------------

_A_MAX = 128
_Y_BASE = 301  # Y1 = 301, Y2 = 302, ...

def get_reg(state: WamState, n: int) -> Term:
    if n >= _Y_BASE:
        yi = n - _Y_BASE  # 0-based index into perm_vars
        if state.temp_y_regs is not None and 0 <= yi < len(state.temp_y_regs):
            return state.temp_y_regs[yi]
        if state.e >= 0:
            env = state.stack[state.e]
            if isinstance(env, Environment) and 0 <= yi < len(env.perm_vars):
                return env.perm_vars[yi]
    return state.regs[n]

def set_reg(state: WamState, n: int, val: Term) -> None:
    if n >= _Y_BASE and state.e >= 0:
        env = state.stack[state.e]
        if isinstance(env, Environment):
            yi = n - _Y_BASE
            if yi >= len(env.perm_vars):
                env.perm_vars.extend([None] * (yi - len(env.perm_vars) + 1))
            env.perm_vars[yi] = val
            return
    state.regs[n] = val


# -- Heap helpers -----------------------------------------------------------

def heap_put(state: WamState, term: Term) -> int:
    addr = state.heap_len
    state.heap[addr] = term
    state.heap_len += 1
    return addr

def heap_get(state: WamState, addr: int) -> Term:
    return state.heap[addr]


def heap_trim(state: WamState, mark: int) -> None:
    """Trim heap back to mark — O(k) where k = entries removed."""
    for i in range(mark, state.heap_len):
        state.heap.pop(i, None)
    state.heap_len = mark


# -- Indexed fact tables ----------------------------------------------------
# Rust-parity O(1) fact lookups. Populated at program-build time, consumed by
# the call_indexed_atom_fact2 / base_category_ancestor* instructions.

def register_indexed_atom_fact2_pairs(state: WamState, pred_key: str,
                                      pairs: List[Tuple[str, str]]) -> None:
    """Build indexed_atom_fact2[pred_key][left] -> [right, ...] from binary atom pairs."""
    table = state.indexed_atom_fact2.setdefault(pred_key, {})
    for left, right in pairs:
        table.setdefault(left, []).append(right)


def register_indexed_weighted_edge_triples(state: WamState, pred_key: str,
                                           triples: List[Tuple[str, str, float]]) -> None:
    """Build indexed_weighted_edge[pred_key][left] -> [(right, weight), ...]."""
    table = state.indexed_weighted_edge.setdefault(pred_key, {})
    for left, right, weight in triples:
        table.setdefault(left, []).append((right, float(weight)))


def _atom_in_cons_list(needle: 'Term', list_term: 'Term', state: WamState) -> bool:
    """Walk a cons-list (Compound('.', [head, tail])) and check if any head equals needle.
    Equality compared by atom name, integer value, or float value (deref'd).
    """
    cur = list_term
    while True:
        cur = deref(cur, state)
        if isinstance(cur, Ref):
            cur = state.heap.get(cur.addr)
            if cur is None:
                return False
            continue
        if isinstance(cur, Atom) and cur.name == '[]':
            return False
        if isinstance(cur, Compound) and cur.functor == '.' and len(cur.args) == 2:
            head = deref(cur.args[0], state)
            if isinstance(head, Ref):
                h2 = state.heap.get(head.addr)
                if h2 is not None:
                    head = deref(h2, state)
            if _terms_equal(head, needle):
                return True
            cur = cur.args[1]
            continue
        return False


def _terms_equal(a: 'Term', b: 'Term') -> bool:
    """Structural equality on dereferenced terms (used for visited-list checks)."""
    if isinstance(a, Atom) and isinstance(b, Atom):
        return a.name == b.name
    if isinstance(a, Int) and isinstance(b, Int):
        return a.n == b.n
    if isinstance(a, Float) and isinstance(b, Float):
        return a.f == b.f
    return False

def _term_identical(a: 'Term', b: 'Term', state: WamState) -> bool:
    """Prolog ==/2-style identity after dereferencing bound variables."""
    a = deref(a, state)
    b = deref(b, state)
    if isinstance(a, Var) or isinstance(b, Var):
        return a is b
    if isinstance(a, Atom) and isinstance(b, Atom):
        return a.name == b.name
    if isinstance(a, Int) and isinstance(b, Int):
        return a.n == b.n
    if isinstance(a, Float) and isinstance(b, Float):
        return a.f == b.f
    if isinstance(a, Compound) and isinstance(b, Compound):
        return (a.functor == b.functor and len(a.args) == len(b.args)
                and all(_term_identical(x, y, state) for x, y in zip(a.args, b.args)))
    return False


def _term_ground(term: 'Term', state: WamState) -> bool:
    term = deref(term, state)
    if isinstance(term, Var):
        return False
    if isinstance(term, Compound):
        return all(_term_ground(arg, state) for arg in term.args)
    if isinstance(term, Ref):
        return _term_ground(deref(term, state), state)
    return True


def _make_cons(head: 'Term', tail: 'Term') -> 'Compound':
    """Construct a cons cell Compound('.', [head, tail])."""
    return Compound('.', [head, tail])


# -- Trail ------------------------------------------------------------------

def trail_if_needed(addr: int, state: WamState) -> None:
    """Trail a heap address if it might need to be undone on backtrack."""
    if addr < state.b_heap_top() or addr < state.b_trail_top():
        state.trail.append(addr)
        state.trail_len += 1

def undo_trail(state: WamState, mark: int) -> None:
    """Unbind all trailed variables since mark."""
    while state.trail_len > mark:
        item = state.trail.pop()
        state.trail_len -= 1
        if isinstance(item, Var):
            item.ref[0] = None  # unbind directly-trailed Var
        else:
            v = state.heap.get(item)
            if isinstance(v, Var):
                v.ref[0] = None


# -- Unification ------------------------------------------------------------

def deref(term: Term, state: WamState) -> Term:
    """Dereference a term through variable chains."""
    while isinstance(term, Var) and term.ref[0] is not None:
        term = term.ref[0]
    if isinstance(term, Ref):
        h = state.heap.get(term.addr)
        if h is not None:
            return deref(h, state)
    return term

def bind(v: Var, val: Term, state: WamState) -> None:
    """Bind a variable, trailing if a choice point exists (conditional trailing)."""
    if state.b >= 0:
        # Unconditional trailing when inside a choice point scope
        state.trail.append(v)
        state.trail_len += 1
    v.ref[0] = val

def unify(a: Term, b: Term, state: WamState) -> bool:
    """Robinson unification without occurs check."""
    state.pdl.append((a, b))
    while state.pdl:
        t1, t2 = state.pdl.pop()
        t1 = deref(t1, state)
        t2 = deref(t2, state)

        if t1 is t2:
            continue

        if isinstance(t1, Var):
            bind(t1, t2, state)
            continue
        if isinstance(t2, Var):
            bind(t2, t1, state)
            continue

        if isinstance(t1, Atom) and isinstance(t2, Atom):
            if t1.name != t2.name:
                state.pdl.clear()
                return False
            continue

        if isinstance(t1, Int) and isinstance(t2, Int):
            if t1.n != t2.n:
                state.pdl.clear()
                return False
            continue

        if isinstance(t1, Float) and isinstance(t2, Float):
            if t1.f != t2.f:
                state.pdl.clear()
                return False
            continue

        if isinstance(t1, Compound) and isinstance(t2, Compound):
            if t1.functor != t2.functor or len(t1.args) != len(t2.args):
                state.pdl.clear()
                return False
            for a1, a2 in zip(t1.args, t2.args):
                state.pdl.append((a1, a2))
            continue

        state.pdl.clear()
        return False

    return True


# -- Choice points ----------------------------------------------------------

def push_choice_point(state: WamState, n_args: int, next_clause: Any) -> None:
    """Create a new choice point, saving argument registers and trail mark."""
    # Registers are 1-indexed; include index n_args so A1..An are restored.
    saved_regs = state.regs[:n_args + 1].copy()
    cp = ChoicePoint(
        n_args=n_args,
        saved_regs=saved_regs,
        saved_e=state.e,
        saved_cp=state.cp,
        next_clause=next_clause,
        trail_top=state.trail_len,
        heap_top=state.heap_len,
        saved_b=state.b,
    )
    state.stack.append(cp)
    state.b = len(state.stack) - 1

def restore_choice_point(state: WamState, next_clause: Any = None) -> None:
    """Backtrack: restore state from current choice point (retry_me_else)."""
    cp_index = state.b
    cp = state.stack[cp_index]
    assert isinstance(cp, ChoicePoint)
    # Undo trail
    undo_trail(state, cp.trail_top)
    # Trim heap (dict-based — remove entries >= mark, reset heap_len)
    heap_trim(state, cp.heap_top)
    # Discard environments/frames younger than the restored choice point.
    del state.stack[cp_index + 1:]
    # Restore registers
    state.regs[:cp.n_args + 1] = cp.saved_regs.copy()
    state.e = cp.saved_e
    state.cp = cp.saved_cp
    if next_clause is not None:
        cp.next_clause = next_clause

def pop_choice_point(state: WamState) -> None:
    """Trust: restore from choice point and remove it (last clause)."""
    cp_index = state.b
    cp = state.stack[cp_index]
    assert isinstance(cp, ChoicePoint)
    restore_choice_point(state)
    # Remove the choice point and any frames above it.
    del state.stack[cp_index:]
    state.b = cp.saved_b


# -- Environments -----------------------------------------------------------

def push_environment(state: WamState, n_perm: int) -> None:
    """Allocate an environment frame for permanent variables."""
    cut_b = state.b
    if cut_b >= 0:
        cp = state.stack[cut_b]
        if isinstance(cp, ChoicePoint):
            cut_b = cp.saved_b
    env = Environment(
        saved_e=state.e,
        saved_cp=state.cp,
        cut_b=cut_b,
        perm_vars=[None] * n_perm,
    )
    state.stack.append(env)
    state.e = len(state.stack) - 1

def pop_environment(state: WamState) -> None:
    """Deallocate current environment."""
    env = state.stack[state.e]
    assert isinstance(env, Environment)
    state.temp_y_regs = env.perm_vars.copy()
    state.cp = env.saved_cp
    state.e = env.saved_e


def _cut_to(state: WamState, target_b: int) -> None:
    # Choice point and environment indices are stored as stack offsets.
    # Do not delete frames here: pruning by B is enough, and compacting the
    # stack can make saved environment offsets point at the wrong frame.
    state.b = target_b
    state.cut_b = target_b


# -- Arithmetic -------------------------------------------------------------

def _strip_arity(functor: str) -> str:
    """Strip WAM arity suffix while preserving slash-named arithmetic ops."""
    slash_ops = {'//2': '/', '///2': '//', '\\//2': '\\/'}
    if functor in slash_ops:
        return slash_ops[functor]
    if '/' in functor:
        return functor.rsplit('/', 1)[0]
    return functor

def _float_divide_lax(a, b):
    if b == 0 and (isinstance(a, float) or isinstance(b, float)):
        if a == 0:
            return float('nan')
        return float('inf') if a > 0 else float('-inf')
    return a / b


def eval_arith(term: Term, state: WamState):
    """Evaluate an arithmetic expression, return int or float."""
    t = deref(term, state)
    if isinstance(t, Int):    return t.n
    if isinstance(t, Float):  return t.f
    if isinstance(t, Compound):
        args = [eval_arith(a, state) for a in t.args]
        # Normalize functor: '+/2' -> '+', '-/1' -> '-', etc.
        fn = _strip_arity(t.functor)
        # Unary minus: -(X)
        if fn == '-' and len(t.args) == 1:
            return -args[0]
        ops = {
            # Basic arithmetic
            '+': lambda a, b: a + b,
            '-': lambda a, b: a - b,
            '*': lambda a, b: a * b,
            '/': _float_divide_lax,
            # Integer division and modulo
            '//': lambda a, b: int(a) // int(b),
            'mod': lambda a, b: int(a) % int(b),
            # Power
            '**': lambda a, b: a ** b,
            '^': lambda a, b: a ** b,
            # Unary
            'abs': lambda a: abs(a),
            'sign': lambda a: (1 if a > 0 else (-1 if a < 0 else 0)),
            'float': lambda a: float(a),
            'integer': lambda a: int(a),
            'truncate': lambda a: int(a),
            'round': lambda a: round(a),
            'ceiling': lambda a: int(a + 0.9999999),
            'floor': lambda a: int(a),
            'sqrt': lambda a: a ** 0.5,
            'exp': lambda a: __import__('math').exp(a),
            'log': lambda a: __import__('math').log(a),
            # Min/max
            'max': lambda a, b: max(a, b),
            'min': lambda a, b: min(a, b),
            # Bitwise (operate on int truncations)
            '/\\': lambda a, b: int(a) & int(b),
            '\\/': lambda a, b: int(a) | int(b),
            'xor': lambda a, b: int(a) ^ int(b),
            '\\': lambda a: ~int(a),
            '>>': lambda a, b: int(a) >> int(b),
            '<<': lambda a, b: int(a) << int(b),
        }
        if fn in ops:
            return ops[fn](*args)
    raise WAMError(f"Cannot evaluate: {term}")


def iso_term_contains_unbound(state: WamState, term: Term) -> bool:
    """Return True when an arithmetic expression tree contains an unbound var."""
    t = deref(term, state)
    if isinstance(t, Var):
        return True
    if isinstance(t, Ref):
        cell = state.heap.get(t.addr)
        return cell is not None and iso_term_contains_unbound(state, cell)
    if isinstance(t, Compound):
        return any(iso_term_contains_unbound(state, arg) for arg in t.args)
    return False


def iso_term_has_zero_divide(state: WamState, term: Term) -> bool:
    """Return True for literal /, //, mod, or rem by zero in an expression tree."""
    t = deref(term, state)
    if isinstance(t, Ref):
        cell = state.heap.get(t.addr)
        return cell is not None and iso_term_has_zero_divide(state, cell)
    if not isinstance(t, Compound):
        return False
    fn = _strip_arity(t.functor)
    if fn in ('/', '//', 'mod', 'rem') and len(t.args) == 2:
        rhs = deref(t.args[1], state)
        if isinstance(rhs, Ref):
            rhs = state.heap.get(rhs.addr, rhs)
        if (isinstance(rhs, Int) and rhs.n == 0) or (isinstance(rhs, Float) and rhs.f == 0.0):
            return True
    return any(iso_term_has_zero_divide(state, arg) for arg in t.args)


def iso_arith_culprit(state: WamState, term: Term) -> Term:
    """Build the Name/Arity culprit term used by type_error(evaluable, Culprit)."""
    t = deref(term, state)
    if isinstance(t, Ref):
        cell = state.heap.get(t.addr)
        if cell is not None:
            t = deref(cell, state)
    if isinstance(t, Atom):
        return Compound('//2', [t, Int(0)])
    if isinstance(t, Compound):
        name = _strip_arity(t.functor)
        return Compound('//2', [make_atom(name), Int(len(t.args))])
    return Compound('//2', [make_atom('unknown'), Int(0)])


def _execute_is_lax(state: WamState) -> bool:
    dst = deref(get_reg(state, 1), state)
    expr = deref(get_reg(state, 2), state)
    try:
        result = eval_arith(expr, state)
    except WAMError:
        return False
    except (ZeroDivisionError, ValueError, TypeError, OverflowError):
        return False
    r = Int(result) if isinstance(result, int) else Float(result)
    return unify(dst, r, state)


def _execute_is_iso(state: WamState) -> bool:
    dst = deref(get_reg(state, 1), state)
    expr = get_reg(state, 2)
    result = _eval_arith_iso(state, expr)
    if isinstance(result, bool):
        return result
    r = Int(result) if isinstance(result, int) else Float(result)
    return unify(dst, r, state)


def _eval_arith_iso(state: WamState, expr: Term):
    if iso_term_contains_unbound(state, expr):
        return throw_iso_error(state, make_instantiation_error(state))
    if iso_term_has_zero_divide(state, expr):
        return throw_iso_error(state, make_evaluation_error(state, 'zero_divisor'))
    try:
        return eval_arith(expr, state)
    except WAMThrow:
        raise
    except Exception:
        return throw_iso_error(state, make_type_error(state, 'evaluable', iso_arith_culprit(state, expr)))


def _compare_arith(op: str, a, b) -> bool:
    if op == '<':
        return a < b
    if op == '>':
        return a > b
    if op == '=<':
        return a <= b
    if op == '>=':
        return a >= b
    if op == '=:=':
        return a == b
    if op == '=\\=':
        return a != b
    return False


def _execute_compare_lax(state: WamState, op: str) -> bool:
    a = deref(get_reg(state, 1), state)
    b = deref(get_reg(state, 2), state)
    try:
        return _compare_arith(op, eval_arith(a, state), eval_arith(b, state))
    except WAMError:
        return False
    except (ZeroDivisionError, ValueError, TypeError, OverflowError):
        return False


def _execute_compare_iso(state: WamState, op: str) -> bool:
    a_expr = get_reg(state, 1)
    b_expr = get_reg(state, 2)
    a = _eval_arith_iso(state, a_expr)
    if isinstance(a, bool):
        return a
    b = _eval_arith_iso(state, b_expr)
    if isinstance(b, bool):
        return b
    return _compare_arith(op, a, b)


def _execute_succ_lax(state: WamState) -> bool:
    a = deref(get_reg(state, 1), state)
    b = deref(get_reg(state, 2), state)
    if isinstance(a, Int):
        if a.n < 0:
            return False
        return unify(get_reg(state, 2), Int(a.n + 1), state)
    if isinstance(b, Int):
        if b.n <= 0:
            return False
        return unify(get_reg(state, 1), Int(b.n - 1), state)
    return False


def _execute_succ_iso(state: WamState) -> bool:
    a = deref(get_reg(state, 1), state)
    b = deref(get_reg(state, 2), state)
    a_unbound = isinstance(a, Var)
    b_unbound = isinstance(b, Var)
    if a_unbound and b_unbound:
        return throw_iso_error(state, make_instantiation_error(state))
    if not a_unbound and not isinstance(a, Int):
        return throw_iso_error(state, make_type_error(state, 'integer', a))
    if not b_unbound and not isinstance(b, Int):
        return throw_iso_error(state, make_type_error(state, 'integer', b))
    if isinstance(a, Int):
        if a.n < 0:
            return throw_iso_error(state, make_type_error(state, 'not_less_than_zero', a))
        return unify(get_reg(state, 2), Int(a.n + 1), state)
    if isinstance(b, Int):
        if b.n <= 0:
            return throw_iso_error(state, make_domain_error(state, 'not_less_than_zero', b))
        return unify(get_reg(state, 1), Int(b.n - 1), state)
    return False


def _codes_from_list(term: 'Term', state: WamState) -> Optional[List[int]]:
    items = _term_to_list(term, state)
    if items is None:
        return None
    codes: List[int] = []
    for item in items:
        value = deref(item, state)
        if not isinstance(value, Int):
            return None
        codes.append(value.n)
    return codes


def _list_from_codes(codes: List[int]) -> 'Term':
    return _list_from_terms([Int(code) for code in codes])


def _chars_from_list(term: 'Term', state: WamState) -> Optional[List[str]]:
    items = _term_to_list(term, state)
    if items is None:
        return None
    chars: List[str] = []
    for item in items:
        value = deref(item, state)
        if not isinstance(value, Atom):
            return None
        text = _runtime_atom_text(value.name)
        if len(text) != 1:
            return None
        chars.append(text)
    return chars


def _list_from_chars(chars: List[str]) -> 'Term':
    return _list_from_terms([make_atom(ch) for ch in chars])


def _parse_number_text(text: str) -> Optional[Term]:
    if text == '':
        return None
    try:
        return Int(int(text, 10))
    except ValueError:
        pass
    try:
        return Float(float(text))
    except ValueError:
        return None


def _number_text(value: Term, state: WamState) -> Optional[str]:
    value = deref(value, state)
    if isinstance(value, Int):
        return str(value.n)
    if isinstance(value, Float):
        return _format_value(value, state)
    return None


def _execute_atom_codes(state: WamState) -> bool:
    atom_term = deref(get_reg(state, 1), state)
    codes_term = deref(get_reg(state, 2), state)
    if isinstance(atom_term, Atom):
        return unify(get_reg(state, 2), _list_from_codes([ord(ch) for ch in atom_term.name]), state)
    codes = _codes_from_list(codes_term, state)
    if codes is None:
        return False
    try:
        atom = ''.join(chr(code) for code in codes)
    except (OverflowError, ValueError):
        return False
    return unify(get_reg(state, 1), make_atom(atom), state)


def _execute_number_codes(state: WamState) -> bool:
    number_term = deref(get_reg(state, 1), state)
    codes_term = deref(get_reg(state, 2), state)
    if isinstance(number_term, Int):
        return unify(get_reg(state, 2), _list_from_codes([ord(ch) for ch in str(number_term.n)]), state)
    if isinstance(number_term, Float):
        return unify(get_reg(state, 2), _list_from_codes([ord(ch) for ch in repr(number_term.f)]), state)
    codes = _codes_from_list(codes_term, state)
    if codes is None:
        return False
    try:
        text = ''.join(chr(code) for code in codes)
        if any(ch in text for ch in '.eE'):
            return unify(get_reg(state, 1), Float(float(text)), state)
        return unify(get_reg(state, 1), Int(int(text)), state)
    except (OverflowError, ValueError):
        return False


def _text_coerce(value: Term, state: WamState, allow_float: bool = True) -> Optional[str]:
    value = deref(value, state)
    if isinstance(value, Atom):
        return _runtime_atom_text(value.name)
    if isinstance(value, Int):
        return str(value.n)
    if allow_float and isinstance(value, Float):
        return _format_value(value, state)
    return None


def _execute_atom_concat(state: WamState) -> bool:
    left = _text_coerce(get_reg(state, 1), state)
    right = _text_coerce(get_reg(state, 2), state)
    if left is None or right is None:
        return False
    return unify(get_reg(state, 3), make_atom(left + right), state)


def _execute_atom_length(state: WamState) -> bool:
    text = _text_coerce(get_reg(state, 1), state, allow_float=False)
    if text is None:
        return False
    return unify(get_reg(state, 2), Int(len(text)), state)


def _execute_atom_string(state: WamState) -> bool:
    left = deref(get_reg(state, 1), state)
    right = deref(get_reg(state, 2), state)
    if not isinstance(left, Var):
        text = _text_coerce(left, state)
        if text is None:
            return False
        return unify(get_reg(state, 2), make_atom(text), state)
    if isinstance(right, Var):
        return False
    text = _text_coerce(right, state)
    if text is None:
        return False
    return unify(get_reg(state, 1), make_atom(text), state)


def _execute_number_chars(state: WamState) -> bool:
    number_text = _number_text(get_reg(state, 1), state)
    if number_text is not None:
        return unify(get_reg(state, 2), _list_from_chars(list(number_text)), state)
    chars = _chars_from_list(get_reg(state, 2), state)
    if chars is None:
        return False
    parsed = _parse_number_text(''.join(chars))
    if parsed is None:
        return False
    return unify(get_reg(state, 1), parsed, state)


def _execute_atom_number(state: WamState) -> bool:
    atom_term = deref(get_reg(state, 1), state)
    if not isinstance(atom_term, Var):
        if not isinstance(atom_term, Atom):
            return False
        parsed = _parse_number_text(_runtime_atom_text(atom_term.name))
        if parsed is None:
            return False
        return unify(get_reg(state, 2), parsed, state)
    text = _number_text(get_reg(state, 2), state)
    if text is None:
        return False
    return unify(get_reg(state, 1), make_atom(text), state)


def _execute_char_code(state: WamState) -> bool:
    char_term = deref(get_reg(state, 1), state)
    code_term = deref(get_reg(state, 2), state)
    if isinstance(char_term, Atom):
        text = _runtime_atom_text(char_term.name)
        if len(text) != 1:
            return False
        return unify(get_reg(state, 2), Int(ord(text)), state)
    if isinstance(code_term, Int) and 0 <= code_term.n <= 255:
        return unify(get_reg(state, 1), make_atom(chr(code_term.n)), state)
    return False


def _execute_string_code(state: WamState) -> bool:
    index = deref(get_reg(state, 1), state)
    text_term = deref(get_reg(state, 2), state)
    if not isinstance(index, Int) or not isinstance(text_term, Atom):
        return False
    text = _runtime_atom_text(text_term.name)
    if index.n < 1 or index.n > len(text):
        return False
    return unify(get_reg(state, 3), Int(ord(text[index.n - 1])), state)


def _execute_atomic_list_concat(state: WamState, arity: int) -> bool:
    if arity == 2:
        items = _term_to_list(get_reg(state, 1), state)
        if items is None:
            return False
        parts: List[str] = []
        for item in items:
            text = _text_coerce(item, state)
            if text is None:
                return False
            parts.append(text)
        return unify(get_reg(state, 2), make_atom(''.join(parts)), state)

    separator = _text_coerce(get_reg(state, 2), state, allow_float=False)
    if separator is None:
        return False
    items = _term_to_list(get_reg(state, 1), state)
    if items is not None:
        parts: List[str] = []
        for item in items:
            text = _text_coerce(item, state)
            if text is None:
                return False
            parts.append(text)
        return unify(get_reg(state, 3), make_atom(separator.join(parts)), state)

    source = _text_coerce(get_reg(state, 3), state, allow_float=False)
    if source is None or separator == '':
        return False
    return unify(get_reg(state, 1), _list_from_terms([make_atom(part) for part in source.split(separator)]), state)


def _execute_split_string(state: WamState) -> bool:
    text = _text_coerce(get_reg(state, 1), state)
    separators = _text_coerce(get_reg(state, 2), state, allow_float=False)
    pads = _text_coerce(get_reg(state, 3), state, allow_float=False)
    if text is None or separators is None or pads is None:
        return False

    parts: List[str] = []
    current: List[str] = []
    for ch in text:
        if ch in separators:
            parts.append(''.join(current).strip(pads))
            current = []
        else:
            current.append(ch)
    parts.append(''.join(current).strip(pads))
    return unify(get_reg(state, 4), _list_from_terms([make_atom(part) for part in parts]), state)


def _execute_append(state: WamState) -> bool:
    left = _term_to_list(get_reg(state, 1), state)
    right = _term_to_list(get_reg(state, 2), state)
    if left is not None and right is not None:
        return unify(get_reg(state, 3), _list_from_terms(left + right), state)
    left = _term_to_list(get_reg(state, 1), state)
    whole = _term_to_list(get_reg(state, 3), state)
    if left is not None and whole is not None and len(whole) >= len(left):
        if all(_term_identical(left[i], whole[i], state) for i in range(len(left))):
            return unify(get_reg(state, 2), _list_from_terms(whole[len(left):]), state)
    return False


def _execute_reverse(state: WamState) -> bool:
    left = _term_to_list(get_reg(state, 1), state)
    if left is not None:
        return unify(get_reg(state, 2), _list_from_terms(list(reversed(left))), state)
    right = _term_to_list(get_reg(state, 2), state)
    if right is not None:
        return unify(get_reg(state, 1), _list_from_terms(list(reversed(right))), state)
    return False


class WAMThrow(Exception):
    """Internal carrier for Prolog throw/1 terms."""
    def __init__(self, term: Term):
        super().__init__("Prolog throw/1")
        self.term = term


class WAMError(Exception):
    pass


# -- Foreign predicate registry ---------------------------------------------

_foreign_predicates: Dict[Tuple[str, int], Callable] = {}

def register_foreign(name: str, arity: int, fn: Callable) -> None:
    """Register a Python function as a Prolog foreign predicate.

    Used to hook in ML kernels:
        register_foreign('numpy_dot', 3, my_dot_fn)

    The function receives (args: list[Term], state: WamState) -> bool
    """
    _foreign_predicates[(name, arity)] = fn

def execute_foreign(functor: str, arity: int, args: List[Term], state: WamState,
                    resume_ip: int = -1) -> bool:
    """Dispatch to a registered foreign predicate.

    resume_ip: the IP to resume from after this call_foreign (used by FFI
    predicates that want to push backtracking choice points).
    Returns True on success, False on failure (triggers backtracking).
    """
    key = (functor, arity)
    if key not in _foreign_predicates:
        raise WAMError(f"Unknown foreign predicate: {functor}/{arity}")
    fn = _foreign_predicates[key]
    return fn(args, state, resume_ip)


# -- Program loading --------------------------------------------------------

def load_program(raw_program: dict) -> Tuple[List[tuple], Dict[str, int]]:
    """
    Flatten program dict into a code array + label index.
    Pre-resolves all label references in call/execute/try_me_else/retry_me_else/
    switch_on_term instructions so the hot loop never does string lookups.

    Returns (code: list[tuple], labels: dict[str, int])
    """
    code = []
    labels = {}
    # First pass: build flat code array and extract __label__ markers
    for pred_label, instrs in raw_program.items():
        labels[pred_label] = len(code)
        if isinstance(instrs, (list, tuple)):
            for instr in instrs:
                if isinstance(instr, tuple) and len(instr) == 2 and instr[0] == '__label__':
                    # Register sub-label at current PC (don't add to code)
                    labels[instr[1]] = len(code)
                else:
                    code.append(instr)
        else:
            code.append(instrs)
    # Second pass: pre-resolve label references
    resolved = []
    for instr in code:
        resolved.append(_resolve_instr(instr, labels))
    return resolved, labels


def _resolve_instr(instr: tuple, labels: Dict[str, int]) -> tuple:
    """Replace string labels with PC indices in branch/call instructions."""
    op = instr[0]
    if op == 'call':
        _, label, arity = instr
        pc = labels.get(label, -1)
        return ('call_pc', pc, arity, label)
    if op == 'execute':
        _, label = instr
        pc = labels.get(label, -1)
        return ('execute_pc', pc, label)
    if op == 'try_me_else':
        if len(instr) >= 3:
            _, next_label, n_args = instr
        else:
            _, next_label = instr
            n_args = 8  # default: save all standard argument registers A1..A8
        pc = labels.get(next_label, -1)
        return ('try_me_else_pc', pc, n_args)
    if op == 'retry_me_else':
        _, next_label = instr
        pc = labels.get(next_label, -1)
        return ('retry_me_else_pc', pc)
    if op == 'switch_on_term':
        _, lv, lc, ll, ls = instr
        return ('switch_on_term_pc',
                labels.get(lv, -1), labels.get(lc, -1),
                labels.get(ll, -1), labels.get(ls, -1))
    if op == 'switch_on_constant':
        _, table = instr
        # table is a dict {key: label_str, ...} — resolve labels to PCs
        resolved_table = {}
        for k, lbl in (table.items() if isinstance(table, dict) else table):
            pc = labels.get(lbl, -1)
            resolved_table[str(k) if not isinstance(k, str) else k] = pc
        return ('switch_on_constant_pc', resolved_table)
    if op == 'recurse_category_ancestor':
        # recurse_category_ancestor mid_reg, root_reg, child_hops_reg,
        #                           visited_reg, pred_label, skip
        _, mid_reg, root_reg, child_hops_reg, visited_reg, pred_label, skip = instr
        pc = labels.get(pred_label, -1)
        return ('recurse_category_ancestor_pc',
                mid_reg, root_reg, child_hops_reg, visited_reg, pc, skip)
    return instr


# -- Builtin predicate dispatch --------------------------------------------

def _parse_constant(name: str) -> 'Term':
    """Parse a constant name string to the appropriate Term type.
    The WAM text format uses get_constant for both atom and numeric constants.
    '1' -> Int(1), '3.14' -> Float(3.14), everything else -> Atom(name).
    """
    try:
        return Int(int(name))
    except (ValueError, TypeError):
        pass
    try:
        return Float(float(name))
    except (ValueError, TypeError):
        pass
    return make_atom(name)


def _constant_term(atom_arg: 'Term') -> 'Term':
    # Preserve already-typed terms verbatim.  The previous behaviour
    # was to re-parse Atom("42").name through _parse_constant which
    # promotes any numeric-looking string to Int / Float -- so a
    # source-level quoted-numeric atom like `'42'` (correctly emitted
    # by the codegen as Atom("42")) silently became Int(42) at
    # put_constant time, breaking read_term_from_atom('42', _) and
    # related uses of quoted-numeric atoms.  The _parse_constant
    # fallback below is reserved for the WAM-text load path where
    # atom_arg arrives as a raw string with no type information.
    if isinstance(atom_arg, (Int, Float, Atom)):
        return atom_arg
    return _parse_constant(str(atom_arg))


def _constant_matches(val: 'Term', constant: 'Term') -> bool:
    """Check if a term matches a constant term literal."""
    return _term_identical(val, constant, WamState())


def _predicate_arity(label: str) -> int:
    try:
        return int(label.rsplit('/', 1)[1])
    except (IndexError, ValueError):
        return 0


def _begin_write_ctx(state: WamState, compound: 'Compound') -> None:
    if state.mode == 'write' and state.write_ctx is not None:
        state.write_stack.append(state.write_ctx)
    state.mode = 'write'
    state.write_ctx = [compound, 0]


def _finish_write_ctx(state: WamState) -> None:
    while state.write_ctx is not None and state.write_ctx[1] >= len(state.write_ctx[0].args):
        state.write_ctx = state.write_stack.pop() if state.write_stack else None


def _write_ctx_put(state: WamState, value: 'Term') -> None:
    wc = state.write_ctx
    if wc and wc[1] < len(wc[0].args):
        wc[0].args[wc[1]] = value
        wc[1] += 1
        _finish_write_ctx(state)


def _begin_read_ctx(state: WamState, compound: 'Compound') -> None:
    if state.mode == 'read' and state.read_ctx is not None:
        state.read_stack.append(state.read_ctx)
    state.mode = 'read'
    state.read_ctx = [compound, 0]


def _finish_read_ctx(state: WamState) -> None:
    while state.read_ctx is not None and state.read_ctx[1] >= len(state.read_ctx[0].args):
        state.read_ctx = state.read_stack.pop() if state.read_stack else None


def _read_ctx_get(state: WamState) -> 'Term':
    rc = state.read_ctx
    if rc is not None and rc[1] < len(rc[0].args):
        value = rc[0].args[rc[1]]
        rc[1] += 1
        _finish_read_ctx(state)
        return value if value is not None else state.fresh_var()
    return state.fresh_var()


def _read_ctx_skip(state: WamState, n: int) -> None:
    for _ in range(n):
        _read_ctx_get(state)


def _clear_structure_context(state: WamState) -> None:
    state.mode = 'write'
    state.write_ctx = None
    state.write_stack = []
    state.read_ctx = None
    state.read_stack = []


def _is_cons_functor(functor: str) -> bool:
    return functor in ('.', './2', '[|]/2')


def _term_to_list(term: 'Term', state: WamState) -> Optional[List['Term']]:
    """Convert a proper Prolog list to a Python list of terms."""
    items = []
    cur = deref(term, state)
    while isinstance(cur, Compound) and _is_cons_functor(cur.functor) and len(cur.args) == 2:
        items.append(deref(cur.args[0], state))
        cur = deref(cur.args[1], state)
    if isinstance(cur, Atom) and cur.name == '[]':
        return items
    return None


def _unify_would_succeed(left: 'Term', right: 'Term', state: WamState) -> bool:
    """Check unification against an isolated state before mutating the live one."""
    sub = copy.deepcopy(state)
    return unify(copy.deepcopy(left), copy.deepcopy(right), sub)


def _execute_member_from_index(elem: 'Term', items: List['Term'], start_idx: int,
                               resume_ip: int, state: WamState) -> bool:
    for idx in range(start_idx, len(items)):
        candidate = items[idx]
        if not _unify_would_succeed(elem, candidate, state):
            continue

        if resume_ip >= 0 and idx + 1 < len(items):
            def make_cont(next_idx, rip):
                def cont(s):
                    if s.b >= 0 and isinstance(s.stack[s.b], ChoicePoint):
                        cp_index = s.b
                        parent_b = s.stack[cp_index].saved_b
                        del s.stack[cp_index:]
                        s.b = parent_b
                    resumed_items = _term_to_list(get_reg(s, 2), s)
                    if resumed_items is None:
                        return -1
                    if _execute_member_from_index(get_reg(s, 1), resumed_items, next_idx, rip, s):
                        return rip
                    return -1
                return cont
            push_choice_point(state, 2, make_cont(idx + 1, resume_ip))

        return unify(elem, candidate, state)
    return False


def _list_from_terms(items: List['Term']) -> 'Term':
    """Build a proper Prolog list from Python terms."""
    result: Term = make_atom('[]')
    for item in reversed(items):
        result = Compound('.', [item, result])
    return result


def _runtime_functor_name(functor: str, arity: int) -> str:
    if functor in ('.', './2', '[|]/2'):
        return '.'
    if '/' in functor:
        return functor
    return f"{functor}/{arity}"


def _display_functor_name(functor: str, arity: int) -> str:
    suffix = f"/{arity}"
    if functor.endswith(suffix) and len(functor) > len(suffix):
        return _strip_arity(functor)
    return functor


def _format_value(val: 'Term', state: WamState) -> str:
    """Format a WAM term using compact Prolog-style syntax."""
    val = deref(val, state)
    if isinstance(val, Atom):
        return val.name
    if isinstance(val, Int):
        return str(val.n)
    if isinstance(val, Float):
        return str(val.f)
    if isinstance(val, Var):
        return f"_V{val.id}"
    if isinstance(val, Compound):
        if _is_cons_functor(val.functor) and len(val.args) == 2:
            items = _term_to_list(val, state)
            if items is not None:
                return '[' + ', '.join(_format_value(item, state) for item in items) + ']'
        functor = _display_functor_name(val.functor, len(val.args))
        return f"{functor}(" + ', '.join(_format_value(arg, state) for arg in val.args) + ')'
    if isinstance(val, Ref):
        return _format_value(deref(val, state), state)
    return str(val)


def _deep_copy_term(val: 'Term', var_map: Dict[int, Var], state: WamState) -> 'Term':
    """Copy a term, giving each source variable one fresh target variable."""
    val = deref(val, state)
    if isinstance(val, Var):
        key = id(val)
        if key not in var_map:
            var_map[key] = state.fresh_var()
        return var_map[key]
    if isinstance(val, Compound):
        return Compound(val.functor, [_deep_copy_term(arg, var_map, state) for arg in val.args])
    return val


def _copy_term_between_states(val: 'Term', source: WamState, target: WamState,
                              var_map: Dict[int, Var]) -> 'Term':
    """Copy a dereferenceable term from one WamState heap into another."""
    val = deref(val, source)
    if isinstance(val, Var):
        key = id(val.ref)
        if key not in var_map:
            var_map[key] = target.fresh_var()
        return var_map[key]
    if isinstance(val, Ref):
        return _copy_term_between_states(deref(val, source), source, target, var_map)
    if isinstance(val, Compound):
        args = [
            _copy_term_between_states(arg, source, target, var_map)
            for arg in val.args
        ]
        return Compound(_runtime_functor_name(val.functor, len(args)), args)
    return val


def _runtime_atom_text(atom_text: str) -> str:
    if len(atom_text) >= 2 and atom_text[0] == "'" and atom_text[-1] == "'":
        inner = atom_text[1:-1]
        return inner.replace("\\'", "'").replace("\\\\", "\\")
    return atom_text


def _read_option_arg(options: 'Term', name: str, state: WamState) -> Optional['Term']:
    items = _term_to_list(options, state)
    if items is None:
        return None
    wanted = f"{name}/1"
    for item in items:
        item = deref(item, state)
        if isinstance(item, Compound) and len(item.args) == 1:
            if item.functor == wanted or item.functor == name:
                return item.args[0]
    return None


def _variable_names_from_env(env_term: 'Term', source: WamState, target: WamState,
                             var_map: Dict[int, Var]) -> 'Term':
    pairs: List[Term] = []
    for pair in _term_to_list(env_term, source) or []:
        pair = deref(pair, source)
        if not isinstance(pair, Compound) or len(pair.args) != 2:
            continue
        name = deref(pair.args[0], source)
        if not isinstance(name, Atom) or name.name == '_':
            continue
        var = _copy_term_between_states(pair.args[1], source, target, var_map)
        pairs.append(Compound('=/2', [make_atom(name.name), var]))
    return _list_from_terms(list(reversed(pairs)))


def _walk_term_vars(term: 'Term', state: WamState) -> List[Var]:
    variables: List[Var] = []

    def walk(value: 'Term') -> None:
        value = deref(value, state)
        if isinstance(value, Var):
            variables.append(value)
            return
        if isinstance(value, Ref):
            heap_value = state.heap.get(value.addr)
            if heap_value is not None:
                walk(heap_value)
            return
        if isinstance(value, Compound):
            for arg in value.args:
                walk(arg)

    walk(term)
    return variables


def _variables_from_term(term: 'Term', state: WamState) -> 'Term':
    seen: set[int] = set()
    variables: List[Term] = []
    for var in _walk_term_vars(term, state):
        key = id(var.ref)
        if key not in seen:
            seen.add(key)
            variables.append(var)
    return _list_from_terms(variables)


def _variable_name_map_from_env(env_term: 'Term', source: WamState, target: WamState,
                                var_map: Dict[int, Var]) -> Dict[int, str]:
    names: Dict[int, str] = {}
    for pair in _term_to_list(env_term, source) or []:
        pair = deref(pair, source)
        if not isinstance(pair, Compound) or len(pair.args) != 2:
            continue
        name = deref(pair.args[0], source)
        if not isinstance(name, Atom):
            continue
        var = _copy_term_between_states(pair.args[1], source, target, var_map)
        if isinstance(var, Var):
            names[id(var.ref)] = name.name
    return names


def _singletons_from_term(term: 'Term', env_term: 'Term', source: WamState,
                          target: WamState, var_map: Dict[int, Var]) -> 'Term':
    occurrences = _walk_term_vars(term, target)
    counts: Dict[int, int] = {}
    for var in occurrences:
        key = id(var.ref)
        counts[key] = counts.get(key, 0) + 1

    names = _variable_name_map_from_env(env_term, source, target, var_map)
    emitted: set[int] = set()
    pairs: List[Term] = []
    for var in occurrences:
        key = id(var.ref)
        if key in emitted or counts.get(key) != 1:
            continue
        name = names.get(key, '_')
        if name != '_' and name.startswith('_'):
            continue
        emitted.add(key)
        pairs.append(Compound('=/2', [make_atom(name), var]))
    return _list_from_terms(pairs)


def _syntax_errors_mode(options: Optional['Term'], state: WamState, default: str) -> str:
    if options is None:
        return default
    opt = _read_option_arg(options, 'syntax_errors', state)
    opt = deref(opt, state) if opt is not None else None
    if isinstance(opt, Atom) and opt.name in ('error', 'fail', 'quiet'):
        return opt.name
    return default


def make_syntax_error(state: WamState, kind: str = 'end_of_clause') -> Term:
    return Compound('syntax_error/1', [make_atom(kind)])


def _execute_compiled_parse_atom(state: WamState, atom_text: str, target: 'Term',
                                 options: Optional['Term'] = None,
                                 syntax_default: str = 'fail') -> bool:
    atom_text = _runtime_atom_text(atom_text)
    code = getattr(state, '_code', None)
    labels = getattr(state, '_labels', None)
    syntax_mode = _syntax_errors_mode(options, state, syntax_default)
    if not code or not labels:
        if syntax_mode == 'error':
            return throw_iso_error(state, make_syntax_error(state))
        return False
    if 'canonical_op_table/1' not in labels:
        if syntax_mode == 'error':
            return throw_iso_error(state, make_syntax_error(state))
        return False

    want_var_names = None
    want_variables = None
    want_singletons = None
    if options is not None:
        want_var_names = _read_option_arg(options, 'variable_names', state)
        want_variables = _read_option_arg(options, 'variables', state)
        want_singletons = _read_option_arg(options, 'singletons', state)

    parser_entry = 'parse_term_from_atom/3'
    wants_env = (want_var_names is not None or want_variables is not None or
                 want_singletons is not None)
    if wants_env and 'parse_term_from_atom/4' in labels:
        parser_entry = 'parse_term_from_atom/4'
    elif parser_entry not in labels:
        return False

    parser_state = WamState()
    ops = parser_state.fresh_var()
    set_reg(parser_state, 1, ops)
    if not run_wam(code, labels, 'canonical_op_table/1', parser_state):
        return False

    parsed = parser_state.fresh_var()
    var_env = parser_state.fresh_var()
    set_reg(parser_state, 1, make_atom(atom_text))
    set_reg(parser_state, 2, deref(ops, parser_state))
    set_reg(parser_state, 3, parsed)
    if parser_entry == 'parse_term_from_atom/4':
        set_reg(parser_state, 4, var_env)
    if not run_wam(code, labels, parser_entry, parser_state):
        if syntax_mode == 'error':
            return throw_iso_error(state, make_syntax_error(state))
        return False

    var_map: Dict[int, Var] = {}
    copied = _copy_term_between_states(parsed, parser_state, state, var_map)
    if not unify(target, copied, state):
        return False
    if want_var_names is not None:
        names = _variable_names_from_env(var_env, parser_state, state, var_map)
        if not unify(want_var_names, names, state):
            return False
    if want_variables is not None:
        variables = _variables_from_term(copied, state)
        if not unify(want_variables, variables, state):
            return False
    if want_singletons is not None:
        singletons = _singletons_from_term(copied, var_env, parser_state, state, var_map)
        if not unify(want_singletons, singletons, state):
            return False
    return True


def _execute_read_term_from_atom(state: WamState, arity: int = 2,
                                 syntax_default: str = 'fail') -> bool:
    atom_term = deref(get_reg(state, 1), state)
    if not isinstance(atom_term, Atom):
        return False
    options = get_reg(state, 3) if arity == 3 else None
    return _execute_compiled_parse_atom(state, atom_term.name, get_reg(state, 2),
                                        options, syntax_default)


def _unwrap_stream(stream: Any) -> Any:
    return stream.handle if isinstance(stream, StreamHandle) else stream


def _stream_open_mode(mode_term: Term, state: WamState) -> Optional[str]:
    mode_term = deref(mode_term, state)
    if not isinstance(mode_term, Atom):
        return None
    mode = _runtime_atom_text(mode_term.name)
    if mode == 'read':
        return 'r'
    if mode == 'write':
        return 'w'
    if mode == 'append':
        return 'a'
    return None


def _stream_path_text(path_term: Term, state: WamState) -> Optional[str]:
    path_term = deref(path_term, state)
    if not isinstance(path_term, Atom):
        return None
    return _runtime_atom_text(path_term.name)


def _execute_open(state: WamState) -> bool:
    path = _stream_path_text(get_reg(state, 1), state)
    mode = _stream_open_mode(get_reg(state, 2), state)
    if path is None or mode is None:
        return False
    try:
        handle = open(path, mode, encoding='utf-8')
    except OSError:
        return False
    if not unify(get_reg(state, 3), StreamHandle(handle), state):
        try:
            handle.close()
        except OSError:
            pass
        return False
    return True


def _execute_exists_file(state: WamState) -> bool:
    path = _stream_path_text(get_reg(state, 1), state)
    return path is not None and os.path.isfile(path)


def _execute_exists_directory(state: WamState) -> bool:
    path = _stream_path_text(get_reg(state, 1), state)
    return path is not None and os.path.isdir(path)


def _execute_directory_files(state: WamState) -> bool:
    path = _stream_path_text(get_reg(state, 1), state)
    if path is None or not os.path.isdir(path):
        return False
    try:
        names = ['.', '..'] + sorted(os.listdir(path))
    except OSError:
        return False
    return unify(get_reg(state, 2), _list_from_terms([make_atom(name) for name in names]), state)


def _execute_make_directory(state: WamState) -> bool:
    path = _stream_path_text(get_reg(state, 1), state)
    if path is None:
        return False
    try:
        os.mkdir(path)
    except OSError:
        return False
    return True


def _execute_delete_file(state: WamState) -> bool:
    path = _stream_path_text(get_reg(state, 1), state)
    if path is None:
        return False
    try:
        os.remove(path)
    except OSError:
        return False
    return True


def _execute_close(state: WamState) -> bool:
    stream = _unwrap_stream(deref(get_reg(state, 1), state))
    if stream is None or not hasattr(stream, 'close'):
        return False
    try:
        stream.close()
    except OSError:
        return False
    return True


def _read_stream_char(stream: Any) -> str:
    stream = _unwrap_stream(stream)
    if hasattr(stream, 'read'):
        chunk = stream.read(1)
        return chunk or ''
    return ''


def _read_default_char(state: WamState) -> str:
    if state.input_pushback:
        return state.input_pushback.pop()
    return _read_stream_char(sys.stdin)


def _peek_default_char(state: WamState) -> str:
    ch = _read_default_char(state)
    if ch != '':
        state.input_pushback.append(ch)
    return ch


def _stream_pushback_key(stream: Any) -> int:
    return id(_unwrap_stream(stream))


def _read_handle_char(state: WamState, stream: Any) -> Optional[str]:
    raw = _unwrap_stream(stream)
    if raw is None or not hasattr(raw, 'read'):
        return None
    key = id(raw)
    chars = state.stream_pushback.get(key)
    if chars:
        return chars.pop()
    return _read_stream_char(raw)


def _peek_handle_char(state: WamState, stream: Any) -> Optional[str]:
    ch = _read_handle_char(state, stream)
    if ch is not None and ch != '':
        state.stream_pushback.setdefault(_stream_pushback_key(stream), []).append(ch)
    return ch


def _char_atom_value(ch: str) -> Atom:
    return make_atom('end_of_file') if ch == '' else make_atom(ch)


def _code_value(ch: str) -> Int:
    return Int(-1) if ch == '' else Int(ord(ch))


def _output_atom_char_text(value: Term, state: WamState) -> Optional[str]:
    value = deref(value, state)
    if not isinstance(value, Atom):
        return None
    text = _runtime_atom_text(value.name)
    return text if len(text) == 1 else None


def _output_code_text(value: Term, state: WamState) -> Optional[str]:
    value = deref(value, state)
    if not isinstance(value, Int) or value.n < 0:
        return None
    try:
        return chr(value.n)
    except (OverflowError, ValueError):
        return None


def _write_stream_char(stream: Any, text: str) -> bool:
    stream = _unwrap_stream(stream)
    if stream is None or not hasattr(stream, 'write'):
        return False
    try:
        stream.write(text)
    except OSError:
        return False
    return True


def _execute_get_char(state: WamState) -> bool:
    ch = _read_default_char(state)
    return unify(get_reg(state, 1), _char_atom_value(ch), state)


def _execute_peek_char(state: WamState) -> bool:
    ch = _peek_default_char(state)
    return unify(get_reg(state, 1), _char_atom_value(ch), state)


def _execute_get_code(state: WamState) -> bool:
    ch = _read_default_char(state)
    return unify(get_reg(state, 1), _code_value(ch), state)


def _execute_get_char_stream(state: WamState) -> bool:
    stream = deref(get_reg(state, 1), state)
    ch = _read_handle_char(state, stream)
    if ch is None:
        return False
    return unify(get_reg(state, 2), _char_atom_value(ch), state)


def _execute_peek_char_stream(state: WamState) -> bool:
    stream = deref(get_reg(state, 1), state)
    ch = _peek_handle_char(state, stream)
    if ch is None:
        return False
    return unify(get_reg(state, 2), _char_atom_value(ch), state)


def _execute_get_code_stream(state: WamState) -> bool:
    stream = deref(get_reg(state, 1), state)
    ch = _read_handle_char(state, stream)
    if ch is None:
        return False
    return unify(get_reg(state, 2), _code_value(ch), state)


def _execute_put_char(state: WamState) -> bool:
    text = _output_atom_char_text(get_reg(state, 1), state)
    if text is None:
        return False
    sys.stdout.write(text)
    return True


def _execute_put_code(state: WamState) -> bool:
    text = _output_code_text(get_reg(state, 1), state)
    if text is None:
        return False
    sys.stdout.write(text)
    return True


def _execute_put_char_stream(state: WamState) -> bool:
    text = _output_atom_char_text(get_reg(state, 2), state)
    if text is None:
        return False
    return _write_stream_char(deref(get_reg(state, 1), state), text)


def _execute_put_code_stream(state: WamState) -> bool:
    text = _output_code_text(get_reg(state, 2), state)
    if text is None:
        return False
    return _write_stream_char(deref(get_reg(state, 1), state), text)


def _execute_read_line_to_string(state: WamState) -> bool:
    stream = deref(get_reg(state, 1), state)
    raw = _unwrap_stream(stream)
    if raw is None or not hasattr(raw, 'read'):
        return False

    chars: List[str] = []
    while True:
        ch = _read_handle_char(state, stream)
        if ch is None:
            return False
        if ch == '':
            if not chars:
                return unify(get_reg(state, 2), make_atom('end_of_file'), state)
            return unify(get_reg(state, 2), make_atom(''.join(chars)), state)
        if ch == '\n':
            return unify(get_reg(state, 2), make_atom(''.join(chars)), state)
        if ch == '\r':
            next_ch = _read_handle_char(state, stream)
            if next_ch is None:
                return False
            if next_ch not in ('', '\n'):
                state.stream_pushback.setdefault(_stream_pushback_key(stream), []).append(next_ch)
            return unify(get_reg(state, 2), make_atom(''.join(chars)), state)
        chars.append(ch)


def _execute_read_string(state: WamState) -> bool:
    stream = deref(get_reg(state, 1), state)
    raw = _unwrap_stream(stream)
    length = deref(get_reg(state, 2), state)
    if raw is None or not hasattr(raw, 'read'):
        return False
    if not isinstance(length, Int) or length.n < 0:
        return False

    chars: List[str] = []
    for _ in range(length.n):
        ch = _read_handle_char(state, stream)
        if ch is None:
            return False
        if ch == '':
            break
        chars.append(ch)

    text = ''.join(chars)
    if not unify(get_reg(state, 3), Int(len(text)), state):
        return False
    return unify(get_reg(state, 5), make_atom(text), state)


def _execute_at_end_of_stream(state: WamState) -> bool:
    stream = deref(get_reg(state, 1), state)
    raw = _unwrap_stream(stream)
    if raw is None or not hasattr(raw, 'read'):
        return False
    if state.stream_pushback.get(_stream_pushback_key(stream)):
        return False
    try:
        if hasattr(raw, 'tell') and hasattr(raw, 'seek'):
            pos = raw.tell()
            ch = raw.read(1)
            raw.seek(pos)
            return ch == ''
        ch = _peek_handle_char(state, stream)
    except (OSError, ValueError):
        return False
    return ch == ''


def _execute_write_to_stream(state: WamState) -> bool:
    stream = deref(get_reg(state, 1), state)
    raw = _unwrap_stream(stream)
    if raw is None or not hasattr(raw, 'write'):
        return False
    try:
        raw.write(_format_value(get_reg(state, 2), state))
    except (OSError, ValueError):
        return False
    return True


def _execute_nl_to_stream(state: WamState) -> bool:
    stream = deref(get_reg(state, 1), state)
    raw = _unwrap_stream(stream)
    if raw is None or not hasattr(raw, 'write'):
        return False
    try:
        raw.write('\n')
    except (OSError, ValueError):
        return False
    return True


def _execute_read_from_stream(state: WamState, stream: Any, target: Term,
                              syntax_default: str = 'fail',
                              read_char: Optional[Callable[[], str]] = None) -> bool:
    if stream is None or isinstance(stream, (Atom, Compound, Var, Int, Float, Ref)):
        return False

    buffer = ''
    while True:
        ch = read_char() if read_char is not None else _read_stream_char(stream)
        if ch == '':
            if not buffer.strip():
                return unify(target, make_atom('end_of_file'), state)
            return _execute_compiled_parse_atom(state, buffer.strip(), target, None, syntax_default)

        buffer += ch
        stripped = buffer.strip()
        if not stripped.endswith('.'):
            continue

        candidate = stripped[:-1].strip()
        if not candidate:
            continue
        if _execute_compiled_parse_atom(state, candidate, target, None, 'fail'):
            return True


def _execute_read(state: WamState, syntax_default: str = 'fail') -> bool:
    stream = deref(get_reg(state, 1), state)
    raw = _unwrap_stream(stream)
    if raw is None or not hasattr(raw, 'read'):
        return False
    return _execute_read_from_stream(
        state,
        stream,
        get_reg(state, 2),
        syntax_default,
        lambda: _read_handle_char(state, stream) or '',
    )


def _execute_read_default_stream(state: WamState,
                                 syntax_default: str = 'fail') -> bool:
    return _execute_read_from_stream(
        state,
        sys.stdin,
        get_reg(state, 1),
        syntax_default,
        lambda: _read_default_char(state),
    )


def _execute_term_to_atom(state: WamState) -> bool:
    term = deref(get_reg(state, 1), state)
    atom_term = deref(get_reg(state, 2), state)
    if isinstance(term, Var):
        if not isinstance(atom_term, Atom):
            return False
        return _execute_compiled_parse_atom(state, atom_term.name, get_reg(state, 1))
    return unify(get_reg(state, 2), make_atom(_format_value(term, state)), state)


def _goal_succeeds_once(goal: 'Term', state: WamState) -> bool:
    """Evaluate a callable goal against an isolated copy of the current state."""
    isolated_goal = copy.deepcopy(deref(goal, state))
    sub = copy.deepcopy(state)
    goal = deref(isolated_goal, sub)
    if isinstance(goal, Atom):
        return _execute_builtin(goal.name, 0, sub)
    if not isinstance(goal, Compound):
        return False
    functor = goal.functor
    arity = len(goal.args)
    pred_key = functor if '/' in functor else f"{functor}/{arity}"
    for i, arg in enumerate(goal.args, start=1):
        set_reg(sub, i, deref(arg, sub))
    if _execute_builtin(pred_key, arity, sub):
        return True
    code = getattr(state, '_code', None)
    labels = getattr(state, '_labels', None)
    if code is not None and labels is not None and pred_key in labels:
        return run_wam(code, labels, pred_key, sub)
    return False


def _restore_state_from_snapshot(state: WamState, snapshot: WamState) -> None:
    """Restore a WamState object in place from a catch/3 snapshot."""
    state.__dict__.clear()
    state.__dict__.update(snapshot.__dict__)


def _execute_goal_in_state(goal: 'Term', state: WamState) -> bool:
    """Execute a callable Prolog goal term against the current WAM state."""
    goal = deref(goal, state)
    if isinstance(goal, Atom):
        return _execute_builtin(goal.name, 0, state)
    if not isinstance(goal, Compound):
        return False
    functor = goal.functor
    arity = len(goal.args)
    pred_key = functor if '/' in functor else f"{functor}/{arity}"
    for i, arg in enumerate(goal.args, start=1):
        set_reg(state, i, deref(arg, state))
    if _execute_builtin(pred_key, arity, state):
        return True
    code = getattr(state, '_code', None)
    labels = getattr(state, '_labels', None)
    if code is not None and labels is not None and pred_key in labels:
        return run_wam(code, labels, pred_key, state)
    return False


def _execute_catch(state: WamState) -> bool:
    """Execute catch(Goal, Catcher, Recovery)."""
    goal = deref(get_reg(state, 1), state)
    snapshot = copy.deepcopy(state)
    frame = CatcherFrame(
        catcher_term=deref(get_reg(snapshot, 2), snapshot),
        recovery_term=deref(get_reg(snapshot, 3), snapshot),
        saved_cp=snapshot.cp,
        snapshot=snapshot,
    )
    state.catcher_frames.append(frame)
    try:
        ok = _execute_goal_in_state(goal, state)
        if state.catcher_frames and state.catcher_frames[-1] is frame:
            state.catcher_frames.pop()
        return ok
    except WAMThrow as thrown:
        if state.catcher_frames and state.catcher_frames[-1] is frame:
            state.catcher_frames.pop()
        _restore_state_from_snapshot(state, frame.snapshot)
        if unify(frame.catcher_term, thrown.term, state):
            return _execute_goal_in_state(frame.recovery_term, state)
        raise


def _execute_throw(state: WamState) -> bool:
    """Execute throw(Term), unwinding to the nearest active catch/3."""
    thrown = _deep_copy_term(get_reg(state, 1), {}, state)
    if not state.catcher_frames:
        print(f"Uncaught Prolog throw: {_format_value(thrown, state)}", file=__import__('sys').stderr)
        return False
    raise WAMThrow(thrown)


def make_instantiation_error(state: WamState) -> Term:
    """Build the inner ISO instantiation_error term."""
    return make_atom('instantiation_error')


def make_type_error(state: WamState, expected: str, culprit: Term) -> Term:
    """Build the inner ISO type_error(Expected, Culprit) term."""
    return Compound('type_error/2', [make_atom(expected), deref(culprit, state)])


def make_domain_error(state: WamState, domain: str, culprit: Term) -> Term:
    """Build the inner ISO domain_error(Domain, Culprit) term."""
    return Compound('domain_error/2', [make_atom(domain), deref(culprit, state)])


def make_evaluation_error(state: WamState, kind: str) -> Term:
    """Build the inner ISO evaluation_error(Kind) term."""
    return Compound('evaluation_error/1', [make_atom(kind)])


def throw_iso_error(state: WamState, err_term: Term) -> bool:
    """Wrap an ISO error term as error(ErrorTerm, Context) and throw it."""
    set_reg(state, 1, Compound('error/2', [err_term, state.fresh_var()]))
    return _execute_throw(state)


def _execute_builtin(builtin: str, arity: int, state: 'WamState', resume_ip: int = -1) -> bool:
    """Execute a WAM builtin_call instruction."""
    if builtin in ('catch/3', 'catch') and arity == 3:
        return _execute_catch(state)
    if builtin in ('throw/1', 'throw') and arity == 1:
        return _execute_throw(state)
    if builtin == '!/0' or builtin == '!':  # cut
        target_b = state.cut_b
        if state.e >= 0:
            env = state.stack[state.e]
            if isinstance(env, Environment):
                target_b = env.cut_b
        _cut_to(state, target_b)
        return True
    if builtin in ('=/2', '=') and arity == 2:
        return unify(get_reg(state, 1), get_reg(state, 2), state)
    if builtin in ('\\=/2', '\\=') and arity == 2:
        sub = copy.deepcopy(state)
        return not unify(get_reg(sub, 1), get_reg(sub, 2), sub)
    if builtin in ('==/2', '==') and arity == 2:
        return _term_identical(get_reg(state, 1), get_reg(state, 2), state)
    if builtin in ('\\==/2', '\\==') and arity == 2:
        return not _term_identical(get_reg(state, 1), get_reg(state, 2), state)
    if builtin in ('is/2', 'is_lax/2', 'is', 'is_lax') and arity == 2:
        return _execute_is_lax(state)
    if builtin in ('is_iso/2', 'is_iso') and arity == 2:
        return _execute_is_iso(state)
    compare_lax = {
        '<': ('<//2', '</2', '<_lax/2', '<_lax'),
        '>': ('>/2', '>_lax/2', '>', '>_lax'),
        '=<': ('=</2', '=<_lax/2', '=<', '=<_lax'),
        '>=': ('>=/2', '>=_lax/2', '>=', '>=_lax'),
        '=:=': ('=:=/2', '=:=_lax/2', '=:=', '=:=_lax'),
        '=\\=': ('=\\=/2', '=\\=_lax/2', '=\\=', '=\\=_lax'),
    }
    for op, keys in compare_lax.items():
        if builtin in keys and arity == 2:
            return _execute_compare_lax(state, op)
    compare_iso = {
        '<': ('<_iso/2', '<_iso'),
        '>': ('>_iso/2', '>_iso'),
        '=<': ('=<_iso/2', '=<_iso'),
        '>=': ('>=_iso/2', '>=_iso'),
        '=:=': ('=:=_iso/2', '=:=_iso'),
        '=\\=': ('=\\=_iso/2', '=\\=_iso'),
    }
    for op, keys in compare_iso.items():
        if builtin in keys and arity == 2:
            return _execute_compare_iso(state, op)
    if builtin in ('succ/2', 'succ_lax/2', 'succ', 'succ_lax') and arity == 2:
        return _execute_succ_lax(state)
    if builtin in ('succ_iso/2', 'succ_iso') and arity == 2:
        return _execute_succ_iso(state)
    if builtin in ('\\+/1', '\\+'):  # negation as failure
        goal = deref(get_reg(state, 1), state)
        return not _goal_succeeds_once(goal, state)
    if builtin in ('atom_codes/2', 'atom_codes', 'string_codes/2', 'string_codes') and arity == 2:
        return _execute_atom_codes(state)
    if builtin in ('number_codes/2', 'number_codes') and arity == 2:
        return _execute_number_codes(state)
    if builtin in ('atom_concat/3', 'atom_concat', 'string_concat/3', 'string_concat') and arity == 3:
        return _execute_atom_concat(state)
    if builtin in ('atom_length/2', 'atom_length', 'string_length/2', 'string_length') and arity == 2:
        return _execute_atom_length(state)
    if builtin in ('atom_string/2', 'atom_string', 'string_to_atom/2', 'string_to_atom') and arity == 2:
        return _execute_atom_string(state)
    if builtin in ('number_chars/2', 'number_chars') and arity == 2:
        return _execute_number_chars(state)
    if builtin in ('atom_number/2', 'atom_number') and arity == 2:
        return _execute_atom_number(state)
    if builtin in ('char_code/2', 'char_code') and arity == 2:
        return _execute_char_code(state)
    if builtin in ('string_code/3', 'string_code') and arity == 3:
        return _execute_string_code(state)
    if builtin in ('atomic_list_concat/2', 'atomic_list_concat') and arity == 2:
        return _execute_atomic_list_concat(state, 2)
    if builtin in ('atomic_list_concat/3',) and arity == 3:
        return _execute_atomic_list_concat(state, 3)
    if builtin in ('split_string/4', 'split_string') and arity == 4:
        return _execute_split_string(state)
    if builtin in ('read_term_from_atom/2', 'read_term_from_atom_lax/2',
                   'read_term_from_atom', 'read_term_from_atom_lax') and arity == 2:
        return _execute_read_term_from_atom(state, 2, 'fail')
    if builtin in ('read_term_from_atom_iso/2', 'read_term_from_atom_iso') and arity == 2:
        return _execute_read_term_from_atom(state, 2, 'error')
    if builtin in ('read_term_from_atom/3', 'read_term_from_atom_lax/3') and arity == 3:
        return _execute_read_term_from_atom(state, 3, 'fail')
    if builtin in ('read_term_from_atom_iso/3',) and arity == 3:
        return _execute_read_term_from_atom(state, 3, 'error')
    if builtin in ('open/3', 'open') and arity == 3:
        return _execute_open(state)
    if builtin in ('exists_file/1', 'exists_file') and arity == 1:
        return _execute_exists_file(state)
    if builtin in ('exists_directory/1', 'exists_directory') and arity == 1:
        return _execute_exists_directory(state)
    if builtin in ('directory_files/2', 'directory_files') and arity == 2:
        return _execute_directory_files(state)
    if builtin in ('make_directory/1', 'make_directory') and arity == 1:
        return _execute_make_directory(state)
    if builtin in ('delete_file/1', 'delete_file') and arity == 1:
        return _execute_delete_file(state)
    if builtin in ('close/1', 'close') and arity == 1:
        return _execute_close(state)
    if builtin in ('get_char/1', 'get_char') and arity == 1:
        return _execute_get_char(state)
    if builtin in ('get_char/2',) and arity == 2:
        return _execute_get_char_stream(state)
    if builtin in ('peek_char/1', 'peek_char') and arity == 1:
        return _execute_peek_char(state)
    if builtin in ('peek_char/2',) and arity == 2:
        return _execute_peek_char_stream(state)
    if builtin in ('get_code/1', 'get_code') and arity == 1:
        return _execute_get_code(state)
    if builtin in ('get_code/2',) and arity == 2:
        return _execute_get_code_stream(state)
    if builtin in ('put_char/1', 'put_char') and arity == 1:
        return _execute_put_char(state)
    if builtin in ('put_char/2',) and arity == 2:
        return _execute_put_char_stream(state)
    if builtin in ('put_code/1', 'put_code') and arity == 1:
        return _execute_put_code(state)
    if builtin in ('put_code/2',) and arity == 2:
        return _execute_put_code_stream(state)
    if builtin in ('read_line_to_string/2', 'read_line_to_string') and arity == 2:
        return _execute_read_line_to_string(state)
    if builtin in ('read_string/5', 'read_string') and arity == 5:
        return _execute_read_string(state)
    if builtin in ('at_end_of_stream/1', 'at_end_of_stream') and arity == 1:
        return _execute_at_end_of_stream(state)
    if builtin in ('write_to_stream/2', 'write_to_stream') and arity == 2:
        return _execute_write_to_stream(state)
    if builtin in ('nl_to_stream/1', 'nl_to_stream') and arity == 1:
        return _execute_nl_to_stream(state)
    if builtin in ('read/1', 'read_lax/1', 'read_term/1', 'read_term_lax/1',
                   'read', 'read_lax', 'read_term', 'read_term_lax') and arity == 1:
        return _execute_read_default_stream(state, 'fail')
    if builtin in ('read_iso/1', 'read_iso',
                   'read_term_iso/1', 'read_term_iso') and arity == 1:
        return _execute_read_default_stream(state, 'error')
    if builtin in ('read/2', 'read_lax/2', 'read', 'read_lax') and arity == 2:
        return _execute_read(state, 'fail')
    if builtin in ('read_iso/2', 'read_iso') and arity == 2:
        return _execute_read(state, 'error')
    if builtin in ('term_to_atom/2', 'term_to_atom') and arity == 2:
        return _execute_term_to_atom(state)
    if builtin in ('append/3', 'append') and arity == 3:
        return _execute_append(state)
    if builtin in ('reverse/2', 'reverse') and arity == 2:
        return _execute_reverse(state)
    if builtin in ('length/2', 'length'):
        items = _term_to_list(get_reg(state, 1), state)
        if items is not None:
            return unify(get_reg(state, 2), Int(len(items)), state)
        n_reg = deref(get_reg(state, 2), state)
        if isinstance(n_reg, Var):
            return unify(get_reg(state, 2), Int(0), state)
        return False
    if builtin in ('nonvar/1', 'nonvar'):
        val = deref(get_reg(state, 1), state)
        return not isinstance(val, Var)
    if builtin in ('var/1', 'var'):
        val = deref(get_reg(state, 1), state)
        return isinstance(val, Var)
    if builtin in ('number/1', 'number'):
        val = deref(get_reg(state, 1), state)
        return isinstance(val, (Int, Float))
    if builtin in ('float/1', 'float'):
        val = deref(get_reg(state, 1), state)
        return isinstance(val, Float)
    if builtin in ('atom/1', 'atom'):
        val = deref(get_reg(state, 1), state)
        return isinstance(val, Atom)
    if builtin in ('integer/1', 'integer'):
        val = deref(get_reg(state, 1), state)
        return isinstance(val, Int)
    if builtin in ('compound/1', 'compound'):
        val = deref(get_reg(state, 1), state)
        return isinstance(val, Compound)
    if builtin in ('atomic/1', 'atomic'):
        val = deref(get_reg(state, 1), state)
        return isinstance(val, (Atom, Int, Float))
    if builtin in ('ground/1', 'ground'):
        return _term_ground(get_reg(state, 1), state)
    if builtin in ('is_list/1', 'is_list'):
        val = deref(get_reg(state, 1), state)
        return _term_to_list(val, state) is not None
    if builtin in ('true/0', 'true'):
        return True
    if builtin in ('fail/0', 'fail', 'false/0', 'false'):
        return False
    if builtin in ('member/2', 'member'):
        # member(Elem, List): bind the first solution and leave choice points
        # for later list elements so aggregate/backtracking consumers enumerate.
        items = _term_to_list(get_reg(state, 2), state)
        if items is None:
            return False
        return _execute_member_from_index(get_reg(state, 1), items, 0, resume_ip, state)
    if builtin in ('functor/3', 'functor') and arity == 3:
        term = deref(get_reg(state, 1), state)
        name = deref(get_reg(state, 2), state)
        arity_term = deref(get_reg(state, 3), state)
        if isinstance(term, Compound):
            return (unify(name, make_atom(term.functor), state)
                    and unify(arity_term, Int(len(term.args)), state))
        if isinstance(term, (Atom, Int, Float)):
            return unify(name, term, state) and unify(arity_term, Int(0), state)
        if isinstance(term, Var) and isinstance(name, Atom) and isinstance(arity_term, Int) and arity_term.n >= 0:
            built = name if arity_term.n == 0 else Compound(name.name, [state.fresh_var() for _ in range(arity_term.n)])
            return unify(term, built, state)
        return False
    if builtin in ('arg/3', 'arg') and arity == 3:
        idx = deref(get_reg(state, 1), state)
        term = deref(get_reg(state, 2), state)
        out = deref(get_reg(state, 3), state)
        if not isinstance(idx, Int) or not isinstance(term, Compound):
            return False
        if idx.n < 1 or idx.n > len(term.args):
            return False
        return unify(out, term.args[idx.n - 1], state)
    if builtin in ('=../2', '=..') and arity == 2:
        term = deref(get_reg(state, 1), state)
        list_term = deref(get_reg(state, 2), state)
        if isinstance(term, Var):
            items = _term_to_list(list_term, state)
            if not items:
                return False
            head = deref(items[0], state)
            args = [deref(arg, state) for arg in items[1:]]
            if not args:
                return unify(term, head, state)
            if not isinstance(head, Atom):
                return False
            return unify(term, Compound(head.name, args), state)
        if isinstance(term, Compound):
            return unify(list_term, _list_from_terms([make_atom(term.functor)] + term.args), state)
        if isinstance(term, (Atom, Int, Float)):
            return unify(list_term, _list_from_terms([term]), state)
        return False
    if builtin in ('copy_term/2', 'copy_term') and arity == 2:
        original = get_reg(state, 1)
        target = get_reg(state, 2)
        return unify(target, _deep_copy_term(original, {}, state), state)
    if builtin in ('write/1', 'display/1', 'print/1'):
        print(_format_value(get_reg(state, 1), state), end='')
        return True
    if builtin in ('writeln/1',):
        print(_format_value(get_reg(state, 1), state))
        return True
    if builtin in ('nl/0', 'nl'):
        print()
        return True
    return False


# -- Main WAM interpreter loop ---------------------------------------------

def run_wam(code: list, labels: dict, entry: str, state: WamState) -> bool:
    """
    Execute WAM instructions from a flat code array.

    code:    flat list of (opcode, *args) tuples (pre-resolved by load_program)
    labels:  dict mapping label string -> index into code array
    entry:   label to start execution
    state:   WamState instance

    Returns True if query succeeded, False if failed.
    """
    ip = labels.get(entry, -1)
    if ip < 0:
        return False
    state._code = code
    state._labels = labels
    code_len = len(code)
    # Argument register snapshot: taken at each clause entry so that
    # get_* instructions read the original argument values even after
    # get_variable clobbers a register that aliases an argument register.
    # (e.g. get_variable X3, A1 sets reg[3]=reg[1], clobbering A3=reg[3]
    #  before get_constant 1, A3 can read A3)
    arg_snapshot: list = list(state.regs)  # snapshot at initial entry

    def fail():
        # An exhausted FFI continuation (result < 0) means the builtin
        # whose CP this is has no more solutions — but the WAM might still
        # have OLDER choice points stacked behind it that we need to try
        # before reporting "no more". The continuation has typically
        # already shrunk state.b to the next fallback CP; loop and retry
        # so we don't drop those by returning False outright.
        nonlocal ip
        while True:
            if state.b < 0:
                return False
            cp = state.stack[state.b]
            restore_choice_point(state)
            next_ip = cp.next_clause
            if callable(next_ip):
                # Callable: FFI continuation — call with state, returns new IP.
                result = next_ip(state)
                if result < 0:
                    # Exhausted; the continuation updated state.b. Try again.
                    continue
                ip = result
                return True
            elif isinstance(next_ip, int) and next_ip >= 0:
                ip = next_ip
                return True
            else:
                # Fallback: try string label resolution.
                resolved = labels.get(cp.next_clause, -1)
                if resolved < 0:
                    # Unresolved label; pop this CP and try the next one.
                    continue
                ip = resolved
                return True

    while ip < code_len:
        instr = code[ip]
        op = instr[0]
        ip += 1

        if op == 'put_variable':
            _, reg, hreg = instr
            v = state.fresh_var()
            heap_put(state, v)
            set_reg(state, reg, v)
            set_reg(state, hreg, v)

        elif op == 'put_value':
            _, src, dst = instr
            set_reg(state, dst, get_reg(state, src))

        elif op == 'put_constant':
            _, atom_arg, reg = instr
            set_reg(state, reg, _constant_term(atom_arg))

        elif op == 'put_nil':
            _, reg = instr
            set_reg(state, reg, make_atom('[]'))

        elif op == 'put_integer':
            _, n, reg = instr
            set_reg(state, reg, Int(n))

        elif op == 'put_float':
            _, f, reg = instr
            set_reg(state, reg, Float(f))

        elif op == 'put_structure':
            _, functor, arity, reg = instr
            functor = _runtime_functor_name(functor, arity)
            old = deref(get_reg(state, reg), state)
            c = Compound(functor, [None]*arity)
            addr = heap_put(state, c)
            ref = Ref(addr)
            if isinstance(old, Var):
                bind(old, ref, state)
            set_reg(state, reg, ref)
            state.s = addr
            _begin_write_ctx(state, c)

        elif op == 'put_list':
            _, reg = instr
            old = deref(get_reg(state, reg), state)
            c = Compound('.', [None, None])
            addr = heap_put(state, c)
            ref = Ref(addr)
            if isinstance(old, Var):
                bind(old, ref, state)
            set_reg(state, reg, ref)
            state.s = addr
            _begin_write_ctx(state, c)

        elif op == 'get_variable':
            # get_variable Xn, Ai: copy Ai (argument register) into Xn (variable register)
            # Instruction format: ('get_variable', Xn, Ai) — dest=Xn, src=Ai
            # Read Ai from arg_snapshot to avoid aliasing clobber (e.g. X3=A3).
            _, xn, ai = instr
            src = arg_snapshot[ai] if (ai < len(arg_snapshot) and ai <= _A_MAX) else get_reg(state, ai)
            set_reg(state, xn, src)

        elif op == 'get_value':
            _, reg1, reg2 = instr
            # reg2 is an argument register — read from snapshot
            snap_val = arg_snapshot[reg2] if (reg2 < len(arg_snapshot) and reg2 <= _A_MAX) else get_reg(state, reg2)
            if not unify(get_reg(state, reg1), snap_val, state):
                if not fail(): return False
                continue

        elif op == 'get_constant':
            _, atom_arg, reg = instr
            constant = _constant_term(atom_arg)
            snap_val = arg_snapshot[reg] if (reg < len(arg_snapshot) and reg <= _A_MAX) else get_reg(state, reg)
            val = deref(snap_val, state)
            if isinstance(val, Var):
                bind(val, constant, state)
            elif not _constant_matches(val, constant):
                if not fail(): return False
                continue

        elif op == 'get_nil':
            _, reg = instr
            snap_val = arg_snapshot[reg] if (reg < len(arg_snapshot) and reg <= _A_MAX) else get_reg(state, reg)
            val = deref(snap_val, state)
            if isinstance(val, Var):
                bind(val, make_atom('[]'), state)
            elif not (isinstance(val, Atom) and val.name == '[]'):
                if not fail(): return False
                continue

        elif op == 'get_integer':
            _, n, reg = instr
            snap_val = arg_snapshot[reg] if (reg < len(arg_snapshot) and reg <= _A_MAX) else get_reg(state, reg)
            val = deref(snap_val, state)
            if isinstance(val, Var):
                bind(val, Int(n), state)
            elif not (isinstance(val, Int) and val.n == n):
                if not fail(): return False
                continue

        elif op == 'get_structure':
            _, functor, arity, reg = instr
            functor = _runtime_functor_name(functor, arity)
            snap_val = arg_snapshot[reg] if (reg < len(arg_snapshot) and reg <= _A_MAX) else get_reg(state, reg)
            val = deref(snap_val, state)
            if isinstance(val, Var):
                c = Compound(functor, [None]*arity)
                addr = heap_put(state, c)
                bind(val, Ref(addr), state)
                state.s = addr
                _begin_write_ctx(state, c)
            elif isinstance(val, Ref):
                h = state.heap[val.addr]
                if isinstance(h, Compound) and h.functor == functor and len(h.args) == arity:
                    state.s = val.addr
                    _begin_read_ctx(state, h)
                else:
                    if not fail(): return False
                    continue
            elif isinstance(val, Compound) and val.functor == functor and len(val.args) == arity:
                _begin_read_ctx(state, val)
            else:
                if not fail(): return False
                continue

        elif op == 'get_list':
            _, reg = instr
            snap_val = arg_snapshot[reg] if (reg < len(arg_snapshot) and reg <= _A_MAX) else get_reg(state, reg)
            val = deref(snap_val, state)
            if isinstance(val, Var):
                c = Compound('.', [None, None])
                addr = heap_put(state, c)
                bind(val, Ref(addr), state)
                state.s = addr
                _begin_write_ctx(state, c)
            elif isinstance(val, Ref):
                h = state.heap[val.addr]
                if isinstance(h, Compound) and _is_cons_functor(h.functor) and len(h.args) == 2:
                    state.s = val.addr
                    _begin_read_ctx(state, h)
                else:
                    if not fail(): return False
                    continue
            elif isinstance(val, Compound) and _is_cons_functor(val.functor) and len(val.args) == 2:
                _begin_read_ctx(state, val)
            else:
                if not fail(): return False
                continue

        elif op == 'unify_variable':
            _, reg = instr
            if state.mode == 'read':
                set_reg(state, reg, _read_ctx_get(state))
            else:
                v = state.fresh_var()
                addr = heap_put(state, v)
                set_reg(state, reg, v)
                _write_ctx_put(state, v)

        elif op == 'unify_value':
            _, reg = instr
            if state.mode == 'read':
                h = _read_ctx_get(state)
                if not unify(get_reg(state, reg), deref(h, state), state):
                    if not fail(): return False
                    continue
            else:
                v = get_reg(state, reg)
                _write_ctx_put(state, v)

        elif op == 'unify_constant':
            _, atom_arg = instr
            constant = _constant_term(atom_arg)
            if state.mode == 'read':
                h = deref(_read_ctx_get(state), state)
                if isinstance(h, Var):
                    bind(h, constant, state)
                elif not _constant_matches(h, constant):
                    if not fail(): return False
                    continue
            else:
                v = constant
                _write_ctx_put(state, v)

        elif op == 'unify_nil':
            if state.mode == 'read':
                h = deref(_read_ctx_get(state), state)
                if isinstance(h, Var):
                    bind(h, make_atom('[]'), state)
                elif not (isinstance(h, Atom) and h.name == '[]'):
                    if not fail(): return False
                    continue
            else:
                v = make_atom('[]')
                _write_ctx_put(state, v)

        elif op == 'unify_void':
            _, n = instr
            if state.mode == 'read':
                _read_ctx_skip(state, n)
            else:
                for _ in range(n):
                    _write_ctx_put(state, state.fresh_var())

        # --- set_* instructions: always WRITE mode, used after put_structure/put_list ---

        elif op == 'set_variable':
            _, xn = instr
            v = state.fresh_var()
            _write_ctx_put(state, v)
            set_reg(state, xn, v)

        elif op == 'set_value':
            _, xn = instr
            v = get_reg(state, xn)
            _write_ctx_put(state, v)

        elif op == 'set_local_value':
            _, xn = instr
            v = deref(get_reg(state, xn), state)
            _write_ctx_put(state, v)

        elif op == 'set_constant':
            _, atom_arg = instr
            v = _constant_term(atom_arg)
            _write_ctx_put(state, v)

        elif op == 'set_nil':
            v = make_atom('[]')
            _write_ctx_put(state, v)

        elif op == 'set_integer':
            _, n = instr
            v = Int(n)
            _write_ctx_put(state, v)

        elif op == 'set_void':
            _, n = instr
            for _ in range(n):
                _write_ctx_put(state, state.fresh_var())

        elif op == 'call_pc':
            _, target_ip, _arity, _label = instr
            if _label == 'catch/3':
                state.cp = ip
                if not _execute_catch(state):
                    if not fail(): return False
                    continue
                arg_snapshot = list(state.regs)
                continue
            if _label == 'throw/1':
                state.cp = ip
                if not _execute_throw(state):
                    if not fail(): return False
                    continue
                arg_snapshot = list(state.regs)
                continue
            state.cp = ip   # save continuation (current ip = next instr)
            if target_ip >= 0:
                ip = target_ip
                arg_snapshot = list(state.regs)  # snapshot at callee entry
                state.temp_y_regs = None
                _clear_structure_context(state)
            else:
                if not _execute_builtin(_label, _arity, state, resume_ip=ip):
                    if not fail(): return False
                    continue
                arg_snapshot = list(state.regs)
                continue

        elif op == 'execute_pc':
            _, target_ip, _label = instr
            if _label == 'catch/3':
                if not _execute_catch(state):
                    if not fail(): return False
                    continue
                arg_snapshot = list(state.regs)
                continue
            if _label == 'throw/1':
                if not _execute_throw(state):
                    if not fail(): return False
                    continue
                arg_snapshot = list(state.regs)
                continue
            if target_ip >= 0:
                ip = target_ip
                arg_snapshot = list(state.regs)  # snapshot at callee entry
                state.temp_y_regs = None
                _clear_structure_context(state)
            else:
                if not _execute_builtin(_label, _predicate_arity(_label), state):
                    if not fail(): return False
                    continue
                if isinstance(state.cp, int):
                    ip = state.cp
                    state.cp = None
                    state.temp_y_regs = None
                    _clear_structure_context(state)
                    arg_snapshot = list(state.regs)
                    continue
                return True

        elif op == 'call':
            # Legacy unresolved call (fallback)
            _, label, _arity = instr
            if label == 'catch/3':
                state.cp = ip
                if not _execute_catch(state):
                    if not fail(): return False
                    continue
                arg_snapshot = list(state.regs)
                continue
            if label == 'throw/1':
                state.cp = ip
                if not _execute_throw(state):
                    if not fail(): return False
                    continue
                arg_snapshot = list(state.regs)
                continue
            state.cp = ip
            target = labels.get(label, -1)
            if target >= 0:
                ip = target
                arg_snapshot = list(state.regs)  # snapshot at callee entry
                state.temp_y_regs = None
                _clear_structure_context(state)
            else:
                if not _execute_builtin(label, _arity, state, resume_ip=ip):
                    if not fail(): return False
                    continue
                arg_snapshot = list(state.regs)
                continue

        elif op == 'execute':
            # Legacy unresolved execute (fallback)
            _, label = instr
            if label == 'catch/3':
                if not _execute_catch(state):
                    if not fail(): return False
                    continue
                arg_snapshot = list(state.regs)
                continue
            if label == 'throw/1':
                if not _execute_throw(state):
                    if not fail(): return False
                    continue
                arg_snapshot = list(state.regs)
                continue
            target = labels.get(label, -1)
            if target >= 0:
                ip = target
                arg_snapshot = list(state.regs)  # snapshot at callee entry
                state.temp_y_regs = None
                _clear_structure_context(state)
            else:
                if not _execute_builtin(label, _predicate_arity(label), state):
                    if not fail(): return False
                    continue
                if isinstance(state.cp, int):
                    ip = state.cp
                    state.cp = None
                    state.temp_y_regs = None
                    _clear_structure_context(state)
                    arg_snapshot = list(state.regs)
                    continue
                return True

        elif op == 'call_lowered':
            _, fn, _arity = instr
            try:
                ok = fn(state)
            except WAMError:
                ok = False
            if not ok:
                if not fail(): return False
                continue
            arg_snapshot = list(state.regs)

        elif op == 'proceed':
            if state.cp is None:
                return True
            if isinstance(state.cp, int):
                ip = state.cp
                state.cp = None
                state.temp_y_regs = None
                _clear_structure_context(state)
                # Returning to caller: snapshot is now stale; it will be
                # refreshed at the next call_pc/execute_pc if needed.
                # Set to None so accidental reads fall back to live regs.
                arg_snapshot = list(state.regs)
            else:
                # Legacy tuple-based cp from old-style run_wam
                return True

        elif op == 'fail':
            if not fail(): return False
            continue

        elif op == 'halt':
            return True

        elif op == 'try_me_else_pc':
            _, next_pc, n_args = instr
            push_choice_point(state, n_args, next_pc)
            arg_snapshot = list(state.regs)  # snapshot at clause entry

        elif op == 'retry_me_else_pc':
            _, next_pc = instr
            restore_choice_point(state, next_clause=next_pc)
            arg_snapshot = list(state.regs)  # snapshot after choice point restore

        elif op == 'try_me_else':
            # Legacy unresolved
            _, next_label, n_args = instr
            push_choice_point(state, n_args, labels.get(next_label, -1))
            arg_snapshot = list(state.regs)  # snapshot at clause entry

        elif op == 'retry_me_else':
            # Legacy unresolved
            _, next_label = instr
            restore_choice_point(state, next_clause=labels.get(next_label, -1))
            arg_snapshot = list(state.regs)  # snapshot after choice point restore

        elif op == 'trust_me':
            pop_choice_point(state)
            arg_snapshot = list(state.regs)  # snapshot at last clause entry

        elif op == 'neck_cut':
            _cut_to(state, state.cut_b)

        elif op == 'get_level':
            _, reg = instr
            set_reg(state, reg, state.b)

        elif op == 'cut':
            _, reg = instr
            level = get_reg(state, reg)
            _cut_to(state, level)

        elif op == 'allocate':
            n_perm = instr[1] if len(instr) > 1 else 16  # default perm vars
            push_environment(state, n_perm)

        elif op == 'deallocate':
            pop_environment(state)

        elif op == 'is':
            _, dst, expr_reg = instr
            expr = deref(get_reg(state, expr_reg), state)
            try:
                result = eval_arith(expr, state)
            except WAMError:
                if not fail(): return False
                continue
            r = Int(result) if isinstance(result, int) else Float(result)
            if not unify(get_reg(state, dst), r, state):
                if not fail(): return False
                continue

        elif op == 'call_foreign':
            _, functor, arity = instr
            args = [deref(get_reg(state, i+1), state) for i in range(arity)]
            try:
                ok = execute_foreign(functor, arity, args, state, resume_ip=ip)
            except WAMError:
                ok = False
            if not ok:
                if not fail(): return False
                continue

        elif op == 'call_indexed_atom_fact2':
            # call_indexed_atom_fact2 pred_key:
            #   Lookup state.indexed_atom_fact2[pred_key][A1] → [v0, v1, ...].
            #   Push a choice point per extra value (closure resumes at this ip
            #   binding A2 to that value). Unify A2 with values[0].
            _, pred_key = instr
            key_term = deref(get_reg(state, 1), state)
            if not isinstance(key_term, Atom):
                if not fail(): return False
                continue
            table = state.indexed_atom_fact2.get(pred_key)
            values = table.get(key_term.name) if table is not None else None
            if not values:
                if not fail(): return False
                continue
            resume_ip = ip
            for v in reversed(values[1:]):
                value_atom = make_atom(v)
                def make_cont(va, rip):
                    def cont(s):
                        target = deref(get_reg(s, 2), s)
                        if unify(target, va, s):
                            return rip
                        return -1
                    return cont
                push_choice_point(state, 2, make_cont(value_atom, resume_ip))
            if not unify(get_reg(state, 2), make_atom(values[0]), state):
                if not fail(): return False
                continue

        elif op == 'base_category_ancestor':
            # base_category_ancestor cat_reg, target_reg, visited_reg:
            #   succeeds if (cat, target) is a category_parent/2 fact AND
            #   target is not already in visited; then pops env & returns to caller.
            _, cat_reg, target_reg, visited_reg = instr
            cat_val = deref(get_reg(state, cat_reg), state)
            if not isinstance(cat_val, Atom):
                if not fail(): return False
                continue
            target_val = deref(get_reg(state, target_reg), state)
            if not isinstance(target_val, Atom):
                if not fail(): return False
                continue
            visited_val = deref(get_reg(state, visited_reg), state)
            if _atom_in_cons_list(target_val, visited_val, state):
                if not fail(): return False
                continue
            table = state.indexed_atom_fact2.get('category_parent/2')
            parents = table.get(cat_val.name) if table is not None else None
            if not parents or target_val.name not in parents:
                if not fail(): return False
                continue
            if state.e < 0:
                if not fail(): return False
                continue
            pop_environment(state)
            ret = state.cp
            state.cp = None
            if isinstance(ret, int) and ret >= 0:
                ip = ret
                arg_snapshot = list(state.regs)
            else:
                return True

        elif op == 'base_category_ancestor_bind':
            # base_category_ancestor_bind cat_reg, target_reg, hops_reg, visited_reg:
            #   Same as base_category_ancestor but additionally binds hops_reg to Int(1).
            _, cat_reg, target_reg, hops_reg, visited_reg = instr
            cat_val = deref(get_reg(state, cat_reg), state)
            if not isinstance(cat_val, Atom):
                if not fail(): return False
                continue
            target_val = deref(get_reg(state, target_reg), state)
            if not isinstance(target_val, Atom):
                if not fail(): return False
                continue
            visited_val = deref(get_reg(state, visited_reg), state)
            if _atom_in_cons_list(target_val, visited_val, state):
                if not fail(): return False
                continue
            table = state.indexed_atom_fact2.get('category_parent/2')
            parents = table.get(cat_val.name) if table is not None else None
            if not parents or target_val.name not in parents:
                if not fail(): return False
                continue
            hops_val = deref(get_reg(state, hops_reg), state)
            one = Int(1)
            if isinstance(hops_val, Var):
                bind(hops_val, one, state)
            elif isinstance(hops_val, Int) and hops_val.n == 1:
                pass
            elif isinstance(hops_val, Atom) and hops_val.name == '1':
                pass
            elif not unify(hops_val, one, state):
                if not fail(): return False
                continue
            if state.e < 0:
                if not fail(): return False
                continue
            pop_environment(state)
            ret = state.cp
            state.cp = None
            if isinstance(ret, int) and ret >= 0:
                ip = ret
                arg_snapshot = list(state.regs)
            else:
                return True

        elif op == 'recurse_category_ancestor_pc':
            # recurse_category_ancestor_pc mid_reg, root_reg, child_hops_reg,
            #                              visited_reg, target_pc, skip:
            #   Tail-call to category_ancestor with A1=mid, A2=root,
            #   A3=fresh child_hops var, A4=[mid|visited]; cp=current+skip.
            _, mid_reg, root_reg, child_hops_reg, visited_reg, target_pc, skip = instr
            mid_val = deref(get_reg(state, mid_reg), state)
            root_val = deref(get_reg(state, root_reg), state)
            visited_val = deref(get_reg(state, visited_reg), state)
            child_hops = state.fresh_var()
            next_visited = _make_cons(mid_val, visited_val)
            set_reg(state, child_hops_reg, child_hops)
            set_reg(state, 1, mid_val)
            set_reg(state, 2, root_val)
            set_reg(state, 3, child_hops)
            set_reg(state, 4, next_visited)
            state.cp = ip - 1 + skip  # ip already advanced; -1 corrects to current pc
            if target_pc < 0:
                if not fail(): return False
                continue
            ip = target_pc
            arg_snapshot = list(state.regs)

        elif op == 'recurse_category_ancestor':
            # Legacy unresolved label form (load_program normally rewrites to _pc).
            _, mid_reg, root_reg, child_hops_reg, visited_reg, pred_label, skip = instr
            target_pc = labels.get(pred_label, -1)
            mid_val = deref(get_reg(state, mid_reg), state)
            root_val = deref(get_reg(state, root_reg), state)
            visited_val = deref(get_reg(state, visited_reg), state)
            child_hops = state.fresh_var()
            next_visited = _make_cons(mid_val, visited_val)
            set_reg(state, child_hops_reg, child_hops)
            set_reg(state, 1, mid_val)
            set_reg(state, 2, root_val)
            set_reg(state, 3, child_hops)
            set_reg(state, 4, next_visited)
            state.cp = ip - 1 + skip
            if target_pc < 0:
                if not fail(): return False
                continue
            ip = target_pc
            arg_snapshot = list(state.regs)

        elif op == 'return_add1':
            # return_add1 out_reg, in_reg:
            #   Compute in_reg+1, bind/unify into out_reg, then pop env & return.
            _, out_reg, in_reg = instr
            in_val = deref(get_reg(state, in_reg), state)
            if isinstance(in_val, Int):
                result = Int(in_val.n + 1)
            elif isinstance(in_val, Float):
                nxt = in_val.f + 1.0
                result = Int(int(round(nxt))) if abs(round(nxt) - nxt) < 1e-12 else Float(nxt)
            elif isinstance(in_val, Atom):
                try:
                    f = float(in_val.name)
                except (TypeError, ValueError):
                    if not fail(): return False
                    continue
                nxt = f + 1.0
                result = Int(int(round(nxt))) if abs(round(nxt) - nxt) < 1e-12 else Float(nxt)
            else:
                if not fail(): return False
                continue
            out_val = deref(get_reg(state, out_reg), state)
            if isinstance(out_val, Var):
                bind(out_val, result, state)
            elif isinstance(out_val, Int) and isinstance(result, Int) and out_val.n == result.n:
                pass
            elif isinstance(out_val, Float) and isinstance(result, Float) and out_val.f == result.f:
                pass
            elif isinstance(out_val, Atom) and isinstance(result, (Int, Float)) \
                    and out_val.name == (str(result.n) if isinstance(result, Int) else str(result.f)):
                pass
            elif not unify(out_val, result, state):
                if not fail(): return False
                continue
            if state.e < 0:
                if not fail(): return False
                continue
            pop_environment(state)
            ret = state.cp
            state.cp = None
            if isinstance(ret, int) and ret >= 0:
                ip = ret
                arg_snapshot = list(state.regs)
            else:
                return True

        elif op == 'switch_on_term_pc':
            _, lv_pc, lc_pc, ll_pc, ls_pc = instr
            val = deref(get_reg(state, 1), state)
            if isinstance(val, Var):
                target = lv_pc
            elif isinstance(val, Atom) and val.name == '[]':
                target = ll_pc
            elif isinstance(val, Atom):
                target = lc_pc
            elif isinstance(val, Int) or isinstance(val, Float):
                target = lc_pc
            elif isinstance(val, Compound) and _is_cons_functor(val.functor):
                target = ll_pc
            else:
                target = ls_pc
            if target >= 0:
                ip = target

        elif op == 'switch_on_term':
            # Legacy unresolved
            _, lv, lc, ll, ls = instr
            val = deref(get_reg(state, 1), state)
            if isinstance(val, Var):
                label = lv
            elif isinstance(val, Atom) and val.name == '[]':
                label = ll
            elif isinstance(val, Atom):
                label = lc
            elif isinstance(val, Int) or isinstance(val, Float):
                label = lc
            elif isinstance(val, Compound) and _is_cons_functor(val.functor):
                label = ll
            else:
                label = ls
            target = labels.get(label, -1)
            if target >= 0:
                ip = target

        elif op == 'switch_on_constant_pc':
            # instr: ('switch_on_constant_pc', table_dict)
            # table_dict maps string keys to pre-resolved PC ints
            _, table = instr
            val = deref(get_reg(state, 1), state)
            if isinstance(val, Var):
                pass  # unbound — fall through, let retry handle it
            else:
                if isinstance(val, Atom):
                    key = val.name
                elif isinstance(val, Int):
                    key = str(val.n)
                elif isinstance(val, Float):
                    key = str(val.f)
                else:
                    if not fail(): return False
                    continue
                target = table.get(key, -1)
                if target >= 0:
                    ip = target
                else:
                    if not fail(): return False
                    continue

        elif op == 'switch_on_constant':
            # Legacy unresolved label-based version
            _, table = instr
            val = deref(get_reg(state, 1), state)
            if isinstance(val, Var):
                pass  # unbound — fall through
            else:
                if isinstance(val, Atom):
                    key = val.name
                elif isinstance(val, Int):
                    key = str(val.n)
                elif isinstance(val, Float):
                    key = str(val.f)
                else:
                    if not fail(): return False
                    continue
                if isinstance(table, dict):
                    label = table.get(key)
                else:
                    label = next((lbl for k, lbl in table if str(k) == key), None)
                if label is not None:
                    target = labels.get(label, -1)
                    if target >= 0:
                        ip = target
                    else:
                        if not fail(): return False
                        continue
                else:
                    if not fail(): return False
                    continue

        elif op == 'builtin_call':
            _, builtin, arity = instr
            ok = _execute_builtin(builtin, arity, state, resume_ip=ip)
            if not ok:
                if not fail(): return False
                continue

        elif op == 'begin_aggregate':
            # begin_aggregate agg_type, value_reg, result_reg
            # Runs the body (up to matching end_aggregate) via backtracking,
            # collecting value_reg at each solution, then aggregates and binds result_reg.
            _, agg_type, value_reg, result_reg = instr
            # Find matching end_aggregate (ip is already past begin_aggregate)
            end_pc = _find_aggregate_end(code, ip)
            if end_pc < 0:
                if not fail(): return False
                continue
            # Run inner body in a sub-state, collecting all solutions
            collected = _run_aggregate_body(code, labels, ip, end_pc,
                                             value_reg, state, state.b)
            # Compute aggregate result
            agg_result = _compute_aggregate(agg_type, collected)
            if agg_result is None:
                # aggregate of 0 elements fails for sum/min/max, returns [] for collect
                if agg_type == 'collect':
                    agg_result = make_atom('[]')
                elif agg_type == 'count':
                    agg_result = Int(0)
                else:
                    if not fail(): return False
                    continue
            # Bind result_reg to agg_result
            existing = deref(get_reg(state, result_reg), state)
            if isinstance(existing, Var):
                bind(existing, agg_result, state)
            elif not unify(existing, agg_result, state):
                if not fail(): return False
                continue
            ip = end_pc + 1  # skip past end_aggregate
            arg_snapshot = list(state.regs)

        elif op == 'end_aggregate':
            # Should never be reached in normal execution flow —
            # begin_aggregate jumps past it. If we get here it's an error.
            raise WAMError("end_aggregate reached outside begin_aggregate scope")

        elif op == 'cut_ite':
            # Cut for if-then-else: remove the choice point created by try_me_else
            # (the else branch) so we don't backtrack into it.
            if state.b >= 0:
                cp = state.stack[state.b]
                if isinstance(cp, ChoicePoint):
                    # Pop the topmost choice point
                    state.stack.pop()
                    state.b = cp.saved_b

        elif op == 'jump':
            # Unconditional jump to label
            _, label = instr
            target = labels.get(label, -1)
            if target >= 0:
                ip = target
            # else: silently continue (label may be next instruction)

        else:
            # Unknown opcode — skip with a debug trace (don't crash)
            pass  # raise WAMError(f"Unknown WAM opcode: {op}")

    return True


def _find_aggregate_end(code: list, start_ip: int) -> int:
    """Scan forward from start_ip to find the matching end_aggregate instruction.
    Returns the PC of end_aggregate, or -1 if not found."""
    depth = 1
    for pc in range(start_ip, len(code)):
        op = code[pc][0] if code[pc] else ''
        if op == 'begin_aggregate':
            depth += 1
        elif op == 'end_aggregate':
            depth -= 1
            if depth == 0:
                return pc
    return -1


def _run_aggregate_body(code: list, labels: dict, body_start: int, end_pc: int,
                         value_reg: int, outer_state: WamState, _base_b: int) -> list:
    """
    Run the WAM body from body_start up to end_pc repeatedly via backtracking.
    Collect the value in value_reg at each solution (when ip reaches end_pc).
    Returns list of collected Term values.
    """
    sub = copy.deepcopy(outer_state)
    base_b = sub.b  # choice points above this are "inner"
    sub.cp = None
    # Patch end_aggregate to be a no-op sentinel — handled below
    collected = []
    sub_arg_snap = list(sub.regs)

    def sub_fail():
        nonlocal sub_ip
        if sub.b < 0 or sub.b <= base_b:
            return False
        cp = sub.stack[sub.b]
        restore_choice_point(sub)
        next_ip = cp.next_clause
        if callable(next_ip):
            result = next_ip(sub)
            if result < 0:
                return False
            sub_ip = result
        elif isinstance(next_ip, int) and next_ip >= 0:
            sub_ip = next_ip
        else:
            nxt = labels.get(cp.next_clause, -1) if isinstance(cp.next_clause, str) else -1
            if nxt < 0:
                return False
            sub_ip = nxt
        return True

    sub_ip = body_start
    code_len = len(code)
    MAX_SOLUTIONS = 10000
    iterations = 0

    while sub_ip < code_len and iterations < MAX_SOLUTIONS:
        if sub_ip == end_pc:
            # Hit end_aggregate: collect value and backtrack for more
            val = deref(get_reg(sub, value_reg), sub)
            collected.append(val)
            iterations += 1
            if not sub_fail():
                break
            sub_arg_snap = list(sub.regs)
            continue

        instr = code[sub_ip]
        op = instr[0]
        sub_ip += 1

        if op == 'put_variable':
            _, reg, hreg = instr
            v = sub.fresh_var()
            heap_put(sub, v)
            set_reg(sub, reg, v)
            set_reg(sub, hreg, v)
        elif op == 'put_value':
            _, src, dst = instr
            set_reg(sub, dst, get_reg(sub, src))
        elif op == 'put_constant':
            _, atom_arg, reg = instr
            set_reg(sub, reg, _constant_term(atom_arg))
        elif op == 'put_nil':
            _, reg = instr
            set_reg(sub, reg, make_atom('[]'))
        elif op == 'put_integer':
            _, n, reg = instr
            set_reg(sub, reg, Int(n))
        elif op == 'put_float':
            _, f, reg = instr
            set_reg(sub, reg, Float(f))
        elif op == 'put_structure':
            _, functor, arity, reg = instr
            old = deref(get_reg(sub, reg), sub)
            c = Compound(functor, [None]*arity)
            addr = heap_put(sub, c)
            ref = Ref(addr)
            if isinstance(old, Var):
                bind(old, ref, sub)
            set_reg(sub, reg, ref)
            sub.s = addr
            _begin_write_ctx(sub, c)
        elif op == 'put_list':
            _, reg = instr
            old = deref(get_reg(sub, reg), sub)
            c = Compound('.', [None, None])
            addr = heap_put(sub, c)
            ref = Ref(addr)
            if isinstance(old, Var):
                bind(old, ref, sub)
            set_reg(sub, reg, ref)
            sub.s = addr
            _begin_write_ctx(sub, c)
        elif op == 'get_variable':
            _, xn, ai = instr
            src = sub_arg_snap[ai] if (ai < len(sub_arg_snap) and ai <= _A_MAX) else get_reg(sub, ai)
            set_reg(sub, xn, src)
        elif op == 'get_value':
            _, reg1, reg2 = instr
            snap_val = sub_arg_snap[reg2] if (reg2 < len(sub_arg_snap) and reg2 <= _A_MAX) else get_reg(sub, reg2)
            if not unify(get_reg(sub, reg1), snap_val, sub):
                if not sub_fail(): break
                sub_arg_snap = list(sub.regs)
                continue
        elif op == 'get_constant':
            _, atom_arg, reg = instr
            constant = _constant_term(atom_arg)
            snap_val = sub_arg_snap[reg] if (reg < len(sub_arg_snap) and reg <= _A_MAX) else get_reg(sub, reg)
            val = deref(snap_val, sub)
            if isinstance(val, Var):
                bind(val, constant, sub)
            elif not _constant_matches(val, constant):
                if not sub_fail(): break
                sub_arg_snap = list(sub.regs)
                continue
        elif op == 'get_nil':
            _, reg = instr
            snap_val = sub_arg_snap[reg] if (reg < len(sub_arg_snap) and reg <= _A_MAX) else get_reg(sub, reg)
            val = deref(snap_val, sub)
            if isinstance(val, Var):
                bind(val, make_atom('[]'), sub)
            elif not (isinstance(val, Atom) and val.name == '[]'):
                if not sub_fail(): break
                sub_arg_snap = list(sub.regs)
                continue
        elif op == 'get_integer':
            _, n, reg = instr
            snap_val = sub_arg_snap[reg] if (reg < len(sub_arg_snap) and reg <= _A_MAX) else get_reg(sub, reg)
            val = deref(snap_val, sub)
            if isinstance(val, Var):
                bind(val, Int(n), sub)
            elif not (isinstance(val, Int) and val.n == n):
                if not sub_fail(): break
                sub_arg_snap = list(sub.regs)
                continue
        elif op == 'get_structure':
            _, functor, arity, reg = instr
            snap_val = sub_arg_snap[reg] if (reg < len(sub_arg_snap) and reg <= _A_MAX) else get_reg(sub, reg)
            val = deref(snap_val, sub)
            if isinstance(val, Var):
                c = Compound(functor, [None]*arity)
                addr = heap_put(sub, c)
                bind(val, Ref(addr), sub)
                sub.s = addr
                _begin_write_ctx(sub, c)
            elif isinstance(val, Ref):
                h = sub.heap.get(val.addr)
                if isinstance(h, Compound) and h.functor == functor and len(h.args) == arity:
                    sub.s = val.addr
                    _begin_read_ctx(sub, h)
                else:
                    if not sub_fail(): break
                    sub_arg_snap = list(sub.regs); continue
            elif isinstance(val, Compound) and val.functor == functor and len(val.args) == arity:
                _begin_read_ctx(sub, val)
            else:
                if not sub_fail(): break
                sub_arg_snap = list(sub.regs); continue
        elif op == 'get_list':
            _, reg = instr
            snap_val = sub_arg_snap[reg] if (reg < len(sub_arg_snap) and reg <= _A_MAX) else get_reg(sub, reg)
            val = deref(snap_val, sub)
            if isinstance(val, Var):
                c = Compound('.', [None, None])
                addr = heap_put(sub, c)
                bind(val, Ref(addr), sub)
                sub.s = addr
                _begin_write_ctx(sub, c)
            elif isinstance(val, Ref):
                h = sub.heap.get(val.addr)
                if isinstance(h, Compound) and _is_cons_functor(h.functor) and len(h.args) == 2:
                    sub.s = val.addr
                    _begin_read_ctx(sub, h)
                else:
                    if not sub_fail(): break
                    sub_arg_snap = list(sub.regs); continue
            elif isinstance(val, Compound) and _is_cons_functor(val.functor) and len(val.args) == 2:
                _begin_read_ctx(sub, val)
            else:
                if not sub_fail(): break
                sub_arg_snap = list(sub.regs); continue
        elif op == 'set_variable':
            _, xn = instr
            v = sub.fresh_var()
            _write_ctx_put(sub, v)
            set_reg(sub, xn, v)
        elif op == 'unify_variable':
            _, xn = instr
            if sub.mode == 'read':
                set_reg(sub, xn, _read_ctx_get(sub))
            else:
                v = sub.fresh_var()
                _write_ctx_put(sub, v)
                set_reg(sub, xn, v)
        elif op == 'set_value':
            _, xn = instr
            v = get_reg(sub, xn)
            _write_ctx_put(sub, v)
        elif op == 'unify_value':
            _, reg = instr
            if sub.mode == 'read':
                h = _read_ctx_get(sub)
                if not unify(get_reg(sub, reg), deref(h, sub), sub):
                    if not sub_fail(): break
                    sub_arg_snap = list(sub.regs); continue
            else:
                v = get_reg(sub, reg)
                _write_ctx_put(sub, v)
        elif op == 'set_constant':
            _, atom_arg = instr
            v = _constant_term(atom_arg)
            _write_ctx_put(sub, v)
        elif op == 'unify_constant':
            _, atom_arg = instr
            constant = _constant_term(atom_arg)
            if sub.mode == 'read':
                h = deref(_read_ctx_get(sub), sub)
                if isinstance(h, Var): bind(h, constant, sub)
                elif not _constant_matches(h, constant):
                    if not sub_fail(): break
                    sub_arg_snap = list(sub.regs); continue
            else:
                v = constant
                _write_ctx_put(sub, v)
        elif op in ('set_nil', 'unify_nil'):
            v = make_atom('[]')
            if op == 'unify_nil' and sub.mode == 'read':
                h = deref(_read_ctx_get(sub), sub)
                if isinstance(h, Var): bind(h, v, sub)
                elif not (isinstance(h, Atom) and h.name == '[]'):
                    if not sub_fail(): break
                    sub_arg_snap = list(sub.regs); continue
            else:
                _write_ctx_put(sub, v)
        elif op == 'set_integer':
            _, n = instr
            _write_ctx_put(sub, Int(n))
        elif op == 'unify_void':
            _, n = instr
            if sub.mode == 'read':
                _read_ctx_skip(sub, n)
            else:
                for _ in range(n):
                    _write_ctx_put(sub, Atom('_'))
        elif op == 'call_pc':
            _, target_ip, _arity, _label = instr
            sub.cp = sub_ip
            if target_ip >= 0:
                sub_ip = target_ip
                sub_arg_snap = list(sub.regs)
                sub.temp_y_regs = None
                _clear_structure_context(sub)
            else:
                if not _execute_builtin(_label, _arity, sub, resume_ip=sub_ip):
                    if not sub_fail(): break
                    sub_arg_snap = list(sub.regs)
                    continue
                sub_arg_snap = list(sub.regs)
        elif op == 'execute_pc':
            _, target_ip, _label = instr
            if target_ip >= 0:
                sub_ip = target_ip
                sub_arg_snap = list(sub.regs)
                sub.temp_y_regs = None
                _clear_structure_context(sub)
            else:
                if not _execute_builtin(_label, _predicate_arity(_label), sub):
                    if not sub_fail(): break
                    sub_arg_snap = list(sub.regs)
                    continue
                if isinstance(sub.cp, int):
                    sub_ip = sub.cp
                    sub.cp = None
                    sub_arg_snap = list(sub.regs)
                    continue
                break
        elif op == 'call':
            _, label, _arity = instr
            sub.cp = sub_ip
            target = labels.get(label, -1)
            if target >= 0:
                sub_ip = target
                sub_arg_snap = list(sub.regs)
                sub.temp_y_regs = None
                _clear_structure_context(sub)
            else:
                if not _execute_builtin(label, _arity, sub, resume_ip=sub_ip):
                    if not sub_fail(): break
                    sub_arg_snap = list(sub.regs)
                    continue
                sub_arg_snap = list(sub.regs)
        elif op == 'execute':
            _, label = instr
            target = labels.get(label, -1)
            if target >= 0:
                sub_ip = target
                sub_arg_snap = list(sub.regs)
                sub.temp_y_regs = None
                _clear_structure_context(sub)
            else:
                if not _execute_builtin(label, _predicate_arity(label), sub):
                    if not sub_fail(): break
                    sub_arg_snap = list(sub.regs)
                    continue
                if isinstance(sub.cp, int):
                    sub_ip = sub.cp
                    sub.cp = None
                    sub_arg_snap = list(sub.regs)
                    continue
                break
        elif op == 'call_lowered':
            _, fn, _arity = instr
            try:
                ok = fn(sub)
            except WAMError:
                ok = False
            if not ok:
                if not sub_fail(): break
                sub_arg_snap = list(sub.regs)
                continue
            sub_arg_snap = list(sub.regs)
        elif op == 'proceed':
            if sub.cp is None:
                break  # done
            if isinstance(sub.cp, int):
                sub_ip = sub.cp
                sub.cp = None
                sub.temp_y_regs = None
                _clear_structure_context(sub)
                sub_arg_snap = list(sub.regs)
            else:
                break
        elif op == 'fail':
            if not sub_fail(): break
            sub_arg_snap = list(sub.regs)
            continue
        elif op == 'halt':
            break
        elif op == 'try_me_else_pc':
            _, next_pc, n_args = instr
            push_choice_point(sub, n_args, next_pc)
            sub_arg_snap = list(sub.regs)
        elif op == 'retry_me_else_pc':
            _, next_pc = instr
            restore_choice_point(sub, next_clause=next_pc)
            sub_arg_snap = list(sub.regs)
        elif op == 'try_me_else':
            _, next_label, n_args = instr
            push_choice_point(sub, n_args, labels.get(next_label, -1))
            sub_arg_snap = list(sub.regs)
        elif op == 'retry_me_else':
            _, next_label = instr
            restore_choice_point(sub, next_clause=labels.get(next_label, -1))
            sub_arg_snap = list(sub.regs)
        elif op == 'trust_me':
            pop_choice_point(sub)
            sub_arg_snap = list(sub.regs)
        elif op == 'neck_cut':
            _cut_to(sub, sub.cut_b)
        elif op == 'cut_ite':
            target_b = sub.stack[sub.b].saved_b if sub.b >= 0 and isinstance(sub.stack[sub.b], ChoicePoint) else sub.b
            _cut_to(sub, target_b)
        elif op == 'get_level':
            _, reg = instr
            set_reg(sub, reg, sub.b)
        elif op == 'cut':
            _, reg = instr
            _cut_to(sub, get_reg(sub, reg))
        elif op == 'allocate':
            n_perm = instr[1] if len(instr) > 1 else 16
            push_environment(sub, n_perm)
        elif op == 'deallocate':
            pop_environment(sub)
        elif op == 'is':
            _, dst, expr_reg = instr
            expr = deref(get_reg(sub, expr_reg), sub)
            try:
                result_val = eval_arith(expr, sub)
            except WAMError:
                if not sub_fail(): break
                sub_arg_snap = list(sub.regs)
                continue
            r = Int(result_val) if isinstance(result_val, int) else Float(result_val)
            if not unify(get_reg(sub, dst), r, sub):
                if not sub_fail(): break
                sub_arg_snap = list(sub.regs)
                continue
        elif op == 'call_foreign':
            _, functor, arity = instr
            args = [deref(get_reg(sub, i+1), sub) for i in range(arity)]
            try:
                ok = execute_foreign(functor, arity, args, sub, resume_ip=sub_ip)
            except WAMError:
                ok = False
            if not ok:
                if not sub_fail(): break
                sub_arg_snap = list(sub.regs)
                continue
        elif op == 'builtin_call':
            _, builtin, arity = instr
            ok = _execute_builtin(builtin, arity, sub, resume_ip=sub_ip)
            if not ok:
                if not sub_fail(): break
                sub_arg_snap = list(sub.regs)
                continue
        elif op == 'switch_on_term_pc':
            _, lv_pc, lc_pc, ll_pc, ls_pc = instr
            val = deref(get_reg(sub, 1), sub)
            if isinstance(val, Var):
                tgt = lv_pc
            elif isinstance(val, Atom) and val.name == '[]':
                tgt = ll_pc
            elif isinstance(val, Atom):
                tgt = lc_pc
            elif isinstance(val, (Int, Float)):
                tgt = lc_pc
            elif isinstance(val, Ref):
                h = sub.heap.get(val.addr)
                if isinstance(h, Compound) and _is_cons_functor(h.functor) and len(h.args) == 2:
                    tgt = ll_pc
                else:
                    tgt = ls_pc
            elif isinstance(val, Compound) and _is_cons_functor(val.functor):
                tgt = ll_pc
            elif isinstance(val, Compound):
                tgt = ls_pc
            else:
                tgt = lv_pc
            if tgt < 0:
                if not sub_fail(): break
                sub_arg_snap = list(sub.regs)
                continue
            sub_ip = tgt
        elif op == 'jump':
            _, label = instr
            target = labels.get(label, -1)
            if target >= 0:
                sub_ip = target
        elif op == 'begin_aggregate':
            # Nested aggregate — recursively handle
            _, inner_agg_type, inner_value_reg, inner_result_reg = instr
            inner_end_pc = _find_aggregate_end(code, sub_ip)
            if inner_end_pc < 0:
                if not sub_fail(): break
                sub_arg_snap = list(sub.regs)
                continue
            inner_collected = _run_aggregate_body(code, labels, sub_ip, inner_end_pc,
                                                   inner_value_reg, sub, sub.b)
            inner_result = _compute_aggregate(inner_agg_type, inner_collected)
            if inner_result is None:
                if inner_agg_type == 'collect':
                    inner_result = make_atom('[]')
                elif inner_agg_type == 'count':
                    inner_result = Int(0)
                else:
                    if not sub_fail(): break
                    sub_arg_snap = list(sub.regs)
                    continue
            existing = deref(get_reg(sub, inner_result_reg), sub)
            if isinstance(existing, Var):
                bind(existing, inner_result, sub)
            elif not unify(existing, inner_result, sub):
                if not sub_fail(): break
                sub_arg_snap = list(sub.regs)
                continue
            sub_ip = inner_end_pc + 1
            sub_arg_snap = list(sub.regs)
        else:
            pass  # skip unknown

    return collected


def _compute_aggregate(agg_type: str, values: list):
    """Compute the aggregate result from a list of collected Term values."""
    if agg_type == 'count':
        return Int(len(values))
    if agg_type == 'collect':
        # Build a Prolog list
        result = make_atom('[]')
        for v in reversed(values):
            result = Compound('.', [v, result])
        return result
    if agg_type == 'sum':
        if not values:
            return None
        total = 0.0
        for v in values:
            if isinstance(v, Int):
                total += v.n
            elif isinstance(v, Float):
                total += v.f
            else:
                return None
        return Int(int(total)) if total == int(total) else Float(total)
    if agg_type == 'min':
        if not values:
            return None
        best = None
        for v in values:
            n = v.n if isinstance(v, Int) else (v.f if isinstance(v, Float) else None)
            if n is None:
                return None
            if best is None or n < best:
                best = n
        return Int(int(best)) if isinstance(best, int) or best == int(best) else Float(best)
    if agg_type == 'max':
        if not values:
            return None
        best = None
        for v in values:
            n = v.n if isinstance(v, Int) else (v.f if isinstance(v, Float) else None)
            if n is None:
                return None
            if best is None or n > best:
                best = n
        return Int(int(best)) if isinstance(best, int) or best == int(best) else Float(best)
    return None


# -- Parallel execution ------------------------------------------------------

def _run_single_seed(args):
    """Worker function for parallel execution.

    Receives (code, labels, entry, seed_regs) and runs a fresh WamState.
    Returns a list of result values (registers 1..10) or None on failure.
    """
    code, labels, entry, seed_regs = args
    state = WamState()
    for i, val in enumerate(seed_regs):
        set_reg(state, i, val)
    if run_wam(code, labels, entry, state):
        results = []
        for i in range(1, 11):
            val = get_reg(state, i)
            if val is not None:
                results.append(val)
        return results
    return None


def run_parallel(code: list, labels: dict, entry: str,
                 seeds: list, max_workers: int = 0) -> list:
    """Execute multiple seeds in parallel, each with their own WamState.

    code:        flat instruction list (from load_program)
    labels:      label -> PC mapping (from load_program)
    entry:       label name to start execution
    seeds:       list of lists — each inner list is initial register values
    max_workers: max parallelism (0 = cpu_count)

    Returns list of result lists (one per seed), None for failed seeds.

    Uses ProcessPoolExecutor to avoid GIL for CPU-bound WAM execution.
    Falls back to ThreadPoolExecutor if pickling fails.
    """
    import os
    if max_workers <= 0:
        max_workers = os.cpu_count() or 1

    tasks = [(code, labels, entry, seed) for seed in seeds]

    try:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            results = list(pool.map(_run_single_seed, tasks))
    except (TypeError, AttributeError):
        # Pickling failed — fall back to threads (GIL-limited but functional)
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            results = list(pool.map(_run_single_seed, tasks))

    return results


def run_parallel_query(raw_program: dict, entry: str,
                       seeds: list, max_workers: int = 0) -> list:
    """Convenience wrapper: load_program + run_parallel in one call."""
    code, labels = load_program(raw_program)
    return run_parallel(code, labels, entry, seeds, max_workers)
