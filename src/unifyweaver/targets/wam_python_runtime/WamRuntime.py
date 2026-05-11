"""
WamRuntime.py — Python WAM runtime library
Trail-based mutable state. No external dependencies.
"""

from __future__ import annotations
import copy
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

Term = Atom | Compound | Var | Int | Float | Ref

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
    perm_vars: List        # Y registers


class WamState:
    def __init__(self):
        self.regs: List = [None] * 512   # A1->1, X1->101, Y1->201 (1-indexed, same as all targets)
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
        # Structure read context: (compound_ref, next_arg_idx) or None
        self.read_ctx: Any = None
        # Indexed fact tables — Rust-parity O(1) fact lookups keyed by predicate.
        # indexed_atom_fact2: pred_key -> { key_atom -> [value_atom, ...] }
        self.indexed_atom_fact2: Dict[str, Dict[str, List[str]]] = {}
        # indexed_weighted_edge: pred_key -> { key_atom -> [(value_atom, weight), ...] }
        self.indexed_weighted_edge: Dict[str, Dict[str, List[Tuple[str, float]]]] = {}

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

_Y_BASE = 201  # Y1 = 201, Y2 = 202, ...

def get_reg(state: WamState, n: int) -> Term:
    if n >= _Y_BASE and state.e >= 0:
        env = state.stack[state.e]
        if isinstance(env, Environment):
            yi = n - _Y_BASE  # 0-based index into perm_vars
            if yi < len(env.perm_vars):
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
    env = Environment(
        saved_e=state.e,
        saved_cp=state.cp,
        perm_vars=[None] * n_perm,
    )
    state.stack.append(env)
    state.e = len(state.stack) - 1

def pop_environment(state: WamState) -> None:
    """Deallocate current environment."""
    env = state.stack[state.e]
    assert isinstance(env, Environment)
    state.cp = env.saved_cp
    state.e = env.saved_e


# -- Arithmetic -------------------------------------------------------------

def _strip_arity(functor: str) -> str:
    """Strip arity suffix from functor name: '+/2' -> '+', 'mod/2' -> 'mod'."""
    if '/' in functor:
        return functor.rsplit('/', 1)[0]
    return functor

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
            '/': lambda a, b: a / b,
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


def _constant_matches(val: 'Term', name: str) -> bool:
    """Check if a term matches a constant specified by name string.
    Handles integer, float, and atom constants.
    """
    if isinstance(val, Atom):
        return val.name == name
    if isinstance(val, Int):
        try:
            return val.n == int(name)
        except (ValueError, TypeError):
            return False
    if isinstance(val, Float):
        try:
            return val.f == float(name)
        except (ValueError, TypeError):
            return False
    return False


def _term_to_list(term: 'Term', state: WamState) -> Optional[List['Term']]:
    """Convert a proper Prolog list to a Python list of terms."""
    items = []
    cur = deref(term, state)
    while isinstance(cur, Compound) and cur.functor == '.' and len(cur.args) == 2:
        items.append(deref(cur.args[0], state))
        cur = deref(cur.args[1], state)
    if isinstance(cur, Atom) and cur.name == '[]':
        return items
    return None


def _list_from_terms(items: List['Term']) -> 'Term':
    """Build a proper Prolog list from Python terms."""
    result: Term = make_atom('[]')
    for item in reversed(items):
        result = Compound('.', [item, result])
    return result


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
        if val.functor == '.' and len(val.args) == 2:
            items = _term_to_list(val, state)
            if items is not None:
                return '[' + ', '.join(_format_value(item, state) for item in items) + ']'
        return f"{val.functor}(" + ', '.join(_format_value(arg, state) for arg in val.args) + ')'
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


def _execute_builtin(builtin: str, arity: int, state: 'WamState') -> bool:
    """Execute a WAM builtin_call instruction."""
    if builtin == '!/0' or builtin == '!':  # cut
        state.cut_b = state.b
        return True
    if builtin in ('=/2', '=') and arity == 2:
        return unify(get_reg(state, 1), get_reg(state, 2), state)
    if builtin in ('\\=/2', '\\=') and arity == 2:
        sub = copy.deepcopy(state)
        return not unify(get_reg(sub, 1), get_reg(sub, 2), sub)
    if builtin in ('==/2', '==') and arity == 2:
        return _term_identical(get_reg(state, 1), get_reg(state, 2), state)
    if builtin in ('is/2', 'is'):
        dst = deref(get_reg(state, 1), state)
        expr = deref(get_reg(state, 2), state)
        try:
            result = eval_arith(expr, state)
        except WAMError:
            return False
        r = Int(result) if isinstance(result, int) else Float(result)
        return unify(dst, r, state)
    if builtin in ('<//2', '</2', '<'):
        a = deref(get_reg(state, 1), state)
        b = deref(get_reg(state, 2), state)
        try:
            return eval_arith(a, state) < eval_arith(b, state)
        except WAMError:
            return False
    if builtin in ('>/2', '>'):
        a = deref(get_reg(state, 1), state)
        b = deref(get_reg(state, 2), state)
        try:
            return eval_arith(a, state) > eval_arith(b, state)
        except WAMError:
            return False
    if builtin in ('=</2', '=<'):
        a = deref(get_reg(state, 1), state)
        b = deref(get_reg(state, 2), state)
        try:
            return eval_arith(a, state) <= eval_arith(b, state)
        except WAMError:
            return False
    if builtin in ('>=/2', '>='):
        a = deref(get_reg(state, 1), state)
        b = deref(get_reg(state, 2), state)
        try:
            return eval_arith(a, state) >= eval_arith(b, state)
        except WAMError:
            return False
    if builtin in ('=:=/2', '=:='):
        a = deref(get_reg(state, 1), state)
        b = deref(get_reg(state, 2), state)
        try:
            return eval_arith(a, state) == eval_arith(b, state)
        except WAMError:
            return False
    if builtin in ('=\\=/2', '=\\='):
        a = deref(get_reg(state, 1), state)
        b = deref(get_reg(state, 2), state)
        try:
            return eval_arith(a, state) != eval_arith(b, state)
        except WAMError:
            return False
    if builtin in ('\\+/1', '\\+'):  # negation as failure
        goal = deref(get_reg(state, 1), state)
        return not _goal_succeeds_once(goal, state)
    if builtin in ('length/2', 'length'):
        lst = deref(get_reg(state, 1), state)
        n_reg = deref(get_reg(state, 2), state)
        length = 0
        cur = lst
        while isinstance(cur, Ref):
            cur = deref(cur, state)
        while isinstance(cur, Compound) and cur.functor == '.':
            length += 1
            cur = deref(cur.args[1], state) if cur.args[1] is not None else make_atom('[]')
        if isinstance(cur, Atom) and cur.name == '[]':
            return unify(n_reg, Int(length), state)
        if isinstance(n_reg, Var):
            return unify(n_reg, Int(length), state)
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
    if builtin in ('is_list/1', 'is_list'):
        val = deref(get_reg(state, 1), state)
        return _term_to_list(val, state) is not None
    if builtin in ('true/0', 'true'):
        return True
    if builtin in ('fail/0', 'fail', 'false/0', 'false'):
        return False
    if builtin in ('member/2', 'member'):
        # member(Elem, List): succeeds if Elem is in List (only first solution)
        elem = deref(get_reg(state, 1), state)
        lst = deref(get_reg(state, 2), state)
        def member_find(e, l):
            if isinstance(l, Atom) and l.name == '[]': return False
            if isinstance(l, Compound) and _strip_arity(l.functor) == '.':
                head = deref(l.args[0], state) if l.args[0] is not None else None
                tail = deref(l.args[1], state) if l.args[1] is not None else None
                if head is not None and unify(e, head, state): return True
                return member_find(e, tail)
            return False
        return member_find(elem, lst)
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
        nonlocal ip
        if state.b < 0:
            return False
        cp = state.stack[state.b]
        restore_choice_point(state)
        # next_clause is now a pre-resolved PC int, string label, or callable
        next_ip = cp.next_clause
        if callable(next_ip):
            # Callable: FFI continuation — call with state, returns new IP
            result = next_ip(state)
            if result < 0:
                return False
            ip = result
        elif isinstance(next_ip, int) and next_ip >= 0:
            ip = next_ip
        else:
            # Fallback: try string label resolution
            ip = labels.get(cp.next_clause, -1)
            if ip < 0:
                return False
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
            atom_name = atom_arg.name if isinstance(atom_arg, Atom) else str(atom_arg)
            set_reg(state, reg, _parse_constant(atom_name))

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
            c = Compound(functor, [None]*arity)
            addr = heap_put(state, c)
            set_reg(state, reg, Ref(addr))
            state.mode = 'write'
            state.s = addr
            state.write_ctx = [c, 0]  # [compound, next_arg_idx]

        elif op == 'put_list':
            _, reg = instr
            c = Compound('.', [None, None])
            addr = heap_put(state, c)
            set_reg(state, reg, Ref(addr))
            state.mode = 'write'
            state.s = addr
            state.write_ctx = [c, 0]

        elif op == 'get_variable':
            # get_variable Xn, Ai: copy Ai (argument register) into Xn (variable register)
            # Instruction format: ('get_variable', Xn, Ai) — dest=Xn, src=Ai
            # Read Ai from arg_snapshot to avoid aliasing clobber (e.g. X3=A3).
            _, xn, ai = instr
            src = arg_snapshot[ai] if (ai < len(arg_snapshot) and ai < _Y_BASE) else get_reg(state, ai)
            set_reg(state, xn, src)

        elif op == 'get_value':
            _, reg1, reg2 = instr
            # reg2 is an argument register — read from snapshot
            snap_val = arg_snapshot[reg2] if (reg2 < len(arg_snapshot) and reg2 < _Y_BASE) else get_reg(state, reg2)
            if not unify(get_reg(state, reg1), snap_val, state):
                if not fail(): return False
                continue

        elif op == 'get_constant':
            _, atom_arg, reg = instr
            atom_name = atom_arg.name if isinstance(atom_arg, Atom) else str(atom_arg)
            snap_val = arg_snapshot[reg] if (reg < len(arg_snapshot) and reg < _Y_BASE) else get_reg(state, reg)
            val = deref(snap_val, state)
            if isinstance(val, Var):
                bind(val, _parse_constant(atom_name), state)
            elif not _constant_matches(val, atom_name):
                if not fail(): return False
                continue

        elif op == 'get_nil':
            _, reg = instr
            snap_val = arg_snapshot[reg] if (reg < len(arg_snapshot) and reg < _Y_BASE) else get_reg(state, reg)
            val = deref(snap_val, state)
            if isinstance(val, Var):
                bind(val, make_atom('[]'), state)
            elif not (isinstance(val, Atom) and val.name == '[]'):
                if not fail(): return False
                continue

        elif op == 'get_integer':
            _, n, reg = instr
            snap_val = arg_snapshot[reg] if (reg < len(arg_snapshot) and reg < _Y_BASE) else get_reg(state, reg)
            val = deref(snap_val, state)
            if isinstance(val, Var):
                bind(val, Int(n), state)
            elif not (isinstance(val, Int) and val.n == n):
                if not fail(): return False
                continue

        elif op == 'get_structure':
            _, functor, arity, reg = instr
            snap_val = arg_snapshot[reg] if (reg < len(arg_snapshot) and reg < _Y_BASE) else get_reg(state, reg)
            val = deref(snap_val, state)
            if isinstance(val, Var):
                c = Compound(functor, [None]*arity)
                addr = heap_put(state, c)
                bind(val, Ref(addr), state)
                state.mode = 'write'
                state.s = addr
                state.write_ctx = [c, 0]
            elif isinstance(val, Ref):
                h = state.heap[val.addr]
                if isinstance(h, Compound) and h.functor == functor and len(h.args) == arity:
                    state.mode = 'read'
                    state.s = val.addr
                    state.read_ctx = [h, 0]
                else:
                    if not fail(): return False
                    continue
            elif isinstance(val, Compound) and val.functor == functor and len(val.args) == arity:
                state.mode = 'read'
                state.read_ctx = [val, 0]
            else:
                if not fail(): return False
                continue

        elif op == 'get_list':
            _, reg = instr
            snap_val = arg_snapshot[reg] if (reg < len(arg_snapshot) and reg < _Y_BASE) else get_reg(state, reg)
            val = deref(snap_val, state)
            if isinstance(val, Var):
                c = Compound('.', [None, None])
                addr = heap_put(state, c)
                bind(val, Ref(addr), state)
                state.mode = 'write'
                state.s = addr
                state.write_ctx = [c, 0]
            elif isinstance(val, Ref):
                h = state.heap[val.addr]
                if isinstance(h, Compound) and h.functor == '.' and len(h.args) == 2:
                    state.mode = 'read'
                    state.s = val.addr
                    state.read_ctx = [h, 0]
                else:
                    if not fail(): return False
                    continue
            else:
                if not fail(): return False
                continue

        elif op == 'unify_variable':
            _, reg = instr
            if state.mode == 'read':
                rc = state.read_ctx
                if rc is not None and rc[1] < len(rc[0].args):
                    arg_val = rc[0].args[rc[1]]
                    rc[1] += 1
                    set_reg(state, reg, arg_val if arg_val is not None else state.fresh_var())
                else:
                    set_reg(state, reg, state.fresh_var())
            else:
                v = state.fresh_var()
                addr = heap_put(state, v)
                set_reg(state, reg, v)

        elif op == 'unify_value':
            _, reg = instr
            if state.mode == 'read':
                rc = state.read_ctx
                h = rc[0].args[rc[1]] if rc and rc[1] < len(rc[0].args) else None
                if rc: rc[1] += 1
                if h is None: h = state.fresh_var()
                if not unify(get_reg(state, reg), deref(h, state), state):
                    if not fail(): return False
                    continue
            else:
                wc = state.write_ctx
                v = get_reg(state, reg)
                if wc and wc[1] < len(wc[0].args):
                    wc[0].args[wc[1]] = v; wc[1] += 1

        elif op == 'unify_constant':
            _, atom_arg = instr
            atom_name = atom_arg.name if isinstance(atom_arg, Atom) else str(atom_arg)
            if state.mode == 'read':
                rc = state.read_ctx
                h = rc[0].args[rc[1]] if rc and rc[1] < len(rc[0].args) else None
                if rc: rc[1] += 1
                h = deref(h, state) if h is not None else state.fresh_var()
                if isinstance(h, Var):
                    bind(h, _parse_constant(atom_name), state)
                elif not _constant_matches(h, atom_name):
                    if not fail(): return False
                    continue
            else:
                wc = state.write_ctx
                v = _parse_constant(atom_name)
                if wc and wc[1] < len(wc[0].args):
                    wc[0].args[wc[1]] = v; wc[1] += 1

        elif op == 'unify_nil':
            if state.mode == 'read':
                rc = state.read_ctx
                h = rc[0].args[rc[1]] if rc and rc[1] < len(rc[0].args) else None
                if rc: rc[1] += 1
                h = deref(h, state) if h is not None else state.fresh_var()
                if isinstance(h, Var):
                    bind(h, make_atom('[]'), state)
                elif not (isinstance(h, Atom) and h.name == '[]'):
                    if not fail(): return False
                    continue
            else:
                wc = state.write_ctx
                v = make_atom('[]')
                if wc and wc[1] < len(wc[0].args):
                    wc[0].args[wc[1]] = v; wc[1] += 1

        elif op == 'unify_void':
            _, n = instr
            if state.mode == 'read':
                rc = state.read_ctx
                if rc: rc[1] += n
            else:
                wc = state.write_ctx
                if wc:
                    for _ in range(n):
                        if wc[1] < len(wc[0].args):
                            wc[0].args[wc[1]] = state.fresh_var(); wc[1] += 1

        # --- set_* instructions: always WRITE mode, used after put_structure/put_list ---

        elif op == 'set_variable':
            _, xn = instr
            v = state.fresh_var()
            wc = state.write_ctx
            if wc and wc[1] < len(wc[0].args):
                wc[0].args[wc[1]] = v; wc[1] += 1
            set_reg(state, xn, v)

        elif op == 'set_value':
            _, xn = instr
            v = get_reg(state, xn)
            wc = state.write_ctx
            if wc and wc[1] < len(wc[0].args):
                wc[0].args[wc[1]] = v; wc[1] += 1

        elif op == 'set_local_value':
            _, xn = instr
            v = deref(get_reg(state, xn), state)
            wc = state.write_ctx
            if wc and wc[1] < len(wc[0].args):
                wc[0].args[wc[1]] = v; wc[1] += 1

        elif op == 'set_constant':
            _, atom_arg = instr
            atom_name = atom_arg.name if isinstance(atom_arg, Atom) else str(atom_arg)
            v = _parse_constant(atom_name)
            wc = state.write_ctx
            if wc and wc[1] < len(wc[0].args):
                wc[0].args[wc[1]] = v; wc[1] += 1

        elif op == 'set_nil':
            wc = state.write_ctx
            v = make_atom('[]')
            if wc and wc[1] < len(wc[0].args):
                wc[0].args[wc[1]] = v; wc[1] += 1

        elif op == 'set_integer':
            _, n = instr
            wc = state.write_ctx
            v = Int(n)
            if wc and wc[1] < len(wc[0].args):
                wc[0].args[wc[1]] = v; wc[1] += 1

        elif op == 'set_void':
            _, n = instr
            wc = state.write_ctx
            if wc:
                for _ in range(n):
                    if wc[1] < len(wc[0].args):
                        wc[0].args[wc[1]] = state.fresh_var(); wc[1] += 1

        elif op == 'call_pc':
            _, target_ip, _arity, _label = instr
            state.cp = ip   # save continuation (current ip = next instr)
            if target_ip >= 0:
                ip = target_ip
                arg_snapshot = list(state.regs)  # snapshot at callee entry
            else:
                if not fail(): return False
                continue

        elif op == 'execute_pc':
            _, target_ip, _label = instr
            if target_ip >= 0:
                ip = target_ip
                arg_snapshot = list(state.regs)  # snapshot at callee entry
            else:
                if not fail(): return False
                continue

        elif op == 'call':
            # Legacy unresolved call (fallback)
            _, label, _arity = instr
            state.cp = ip
            target = labels.get(label, -1)
            if target >= 0:
                ip = target
                arg_snapshot = list(state.regs)  # snapshot at callee entry
            else:
                if not fail(): return False
                continue

        elif op == 'execute':
            # Legacy unresolved execute (fallback)
            _, label = instr
            target = labels.get(label, -1)
            if target >= 0:
                ip = target
                arg_snapshot = list(state.regs)  # snapshot at callee entry
            else:
                if not fail(): return False
                continue

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
            state.b = state.cut_b

        elif op == 'get_level':
            _, reg = instr
            set_reg(state, reg, state.b)

        elif op == 'cut':
            _, reg = instr
            level = get_reg(state, reg)
            state.b = level

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
            elif isinstance(val, Compound) and val.functor == '.':
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
            elif isinstance(val, Compound) and val.functor == '.':
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
            ok = _execute_builtin(builtin, arity, state)
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
            atom_name = atom_arg.name if isinstance(atom_arg, Atom) else str(atom_arg)
            set_reg(sub, reg, _parse_constant(atom_name))
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
            c = Compound(functor, [None]*arity)
            addr = heap_put(sub, c)
            set_reg(sub, reg, Ref(addr))
            sub.mode = 'write'; sub.s = addr
            sub.write_ctx = [c, 0]
        elif op == 'put_list':
            _, reg = instr
            c = Compound('.', [None, None])
            addr = heap_put(sub, c)
            set_reg(sub, reg, Ref(addr))
            sub.mode = 'write'; sub.s = addr
            sub.write_ctx = [c, 0]
        elif op == 'get_variable':
            _, xn, ai = instr
            src = sub_arg_snap[ai] if (ai < len(sub_arg_snap) and ai < _Y_BASE) else get_reg(sub, ai)
            set_reg(sub, xn, src)
        elif op == 'get_value':
            _, reg1, reg2 = instr
            snap_val = sub_arg_snap[reg2] if (reg2 < len(sub_arg_snap) and reg2 < _Y_BASE) else get_reg(sub, reg2)
            if not unify(get_reg(sub, reg1), snap_val, sub):
                if not sub_fail(): break
                sub_arg_snap = list(sub.regs)
                continue
        elif op == 'get_constant':
            _, atom_arg, reg = instr
            atom_name = atom_arg.name if isinstance(atom_arg, Atom) else str(atom_arg)
            snap_val = sub_arg_snap[reg] if (reg < len(sub_arg_snap) and reg < _Y_BASE) else get_reg(sub, reg)
            val = deref(snap_val, sub)
            if isinstance(val, Var):
                bind(val, _parse_constant(atom_name), sub)
            elif not _constant_matches(val, atom_name):
                if not sub_fail(): break
                sub_arg_snap = list(sub.regs)
                continue
        elif op == 'get_nil':
            _, reg = instr
            snap_val = sub_arg_snap[reg] if (reg < len(sub_arg_snap) and reg < _Y_BASE) else get_reg(sub, reg)
            val = deref(snap_val, sub)
            if isinstance(val, Var):
                bind(val, make_atom('[]'), sub)
            elif not (isinstance(val, Atom) and val.name == '[]'):
                if not sub_fail(): break
                sub_arg_snap = list(sub.regs)
                continue
        elif op == 'get_integer':
            _, n, reg = instr
            snap_val = sub_arg_snap[reg] if (reg < len(sub_arg_snap) and reg < _Y_BASE) else get_reg(sub, reg)
            val = deref(snap_val, sub)
            if isinstance(val, Var):
                bind(val, Int(n), sub)
            elif not (isinstance(val, Int) and val.n == n):
                if not sub_fail(): break
                sub_arg_snap = list(sub.regs)
                continue
        elif op == 'get_structure':
            _, functor, arity, reg = instr
            snap_val = sub_arg_snap[reg] if (reg < len(sub_arg_snap) and reg < _Y_BASE) else get_reg(sub, reg)
            val = deref(snap_val, sub)
            if isinstance(val, Var):
                c = Compound(functor, [None]*arity)
                addr = heap_put(sub, c)
                bind(val, Ref(addr), sub)
                sub.mode = 'write'; sub.s = addr; sub.write_ctx = [c, 0]
            elif isinstance(val, Ref):
                h = sub.heap.get(val.addr)
                if isinstance(h, Compound) and h.functor == functor and len(h.args) == arity:
                    sub.mode = 'read'; sub.s = val.addr; sub.read_ctx = [h, 0]
                else:
                    if not sub_fail(): break
                    sub_arg_snap = list(sub.regs); continue
            elif isinstance(val, Compound) and val.functor == functor and len(val.args) == arity:
                sub.mode = 'read'; sub.read_ctx = [val, 0]
            else:
                if not sub_fail(): break
                sub_arg_snap = list(sub.regs); continue
        elif op == 'get_list':
            _, reg = instr
            snap_val = sub_arg_snap[reg] if (reg < len(sub_arg_snap) and reg < _Y_BASE) else get_reg(sub, reg)
            val = deref(snap_val, sub)
            if isinstance(val, Var):
                c = Compound('.', [None, None])
                addr = heap_put(sub, c)
                bind(val, Ref(addr), sub)
                sub.mode = 'write'; sub.s = addr; sub.write_ctx = [c, 0]
            elif isinstance(val, Ref):
                h = sub.heap.get(val.addr)
                if isinstance(h, Compound) and h.functor == '.' and len(h.args) == 2:
                    sub.mode = 'read'; sub.s = val.addr; sub.read_ctx = [h, 0]
                else:
                    if not sub_fail(): break
                    sub_arg_snap = list(sub.regs); continue
            else:
                if not sub_fail(): break
                sub_arg_snap = list(sub.regs); continue
        elif op == 'set_variable':
            _, xn = instr
            v = sub.fresh_var()
            wc = sub.write_ctx
            if wc and wc[1] < len(wc[0].args): wc[0].args[wc[1]] = v; wc[1] += 1
            set_reg(sub, xn, v)
        elif op == 'unify_variable':
            _, xn = instr
            if sub.mode == 'read':
                rc = sub.read_ctx
                arg_val = rc[0].args[rc[1]] if rc and rc[1] < len(rc[0].args) else None
                if rc: rc[1] += 1
                set_reg(sub, xn, arg_val if arg_val is not None else sub.fresh_var())
            else:
                v = sub.fresh_var()
                wc = sub.write_ctx
                if wc and wc[1] < len(wc[0].args): wc[0].args[wc[1]] = v; wc[1] += 1
                set_reg(sub, xn, v)
        elif op == 'set_value':
            _, xn = instr
            v = get_reg(sub, xn)
            wc = sub.write_ctx
            if wc and wc[1] < len(wc[0].args): wc[0].args[wc[1]] = v; wc[1] += 1
        elif op == 'unify_value':
            _, reg = instr
            if sub.mode == 'read':
                rc = sub.read_ctx
                h = rc[0].args[rc[1]] if rc and rc[1] < len(rc[0].args) else None
                if rc: rc[1] += 1
                if h is None: h = sub.fresh_var()
                if not unify(get_reg(sub, reg), deref(h, sub), sub):
                    if not sub_fail(): break
                    sub_arg_snap = list(sub.regs); continue
            else:
                v = get_reg(sub, reg)
                wc = sub.write_ctx
                if wc and wc[1] < len(wc[0].args): wc[0].args[wc[1]] = v; wc[1] += 1
        elif op == 'set_constant':
            _, atom_arg = instr
            v = _parse_constant(atom_arg.name if isinstance(atom_arg, Atom) else str(atom_arg))
            wc = sub.write_ctx
            if wc and wc[1] < len(wc[0].args): wc[0].args[wc[1]] = v; wc[1] += 1
        elif op == 'unify_constant':
            _, atom_arg = instr
            atom_name = atom_arg.name if isinstance(atom_arg, Atom) else str(atom_arg)
            if sub.mode == 'read':
                rc = sub.read_ctx
                h = rc[0].args[rc[1]] if rc and rc[1] < len(rc[0].args) else None
                if rc: rc[1] += 1
                h = deref(h, sub) if h is not None else sub.fresh_var()
                if isinstance(h, Var): bind(h, _parse_constant(atom_name), sub)
                elif not _constant_matches(h, atom_name):
                    if not sub_fail(): break
                    sub_arg_snap = list(sub.regs); continue
            else:
                v = _parse_constant(atom_name)
                wc = sub.write_ctx
                if wc and wc[1] < len(wc[0].args): wc[0].args[wc[1]] = v; wc[1] += 1
        elif op in ('set_nil', 'unify_nil'):
            v = make_atom('[]')
            if op == 'unify_nil' and sub.mode == 'read':
                rc = sub.read_ctx
                h = rc[0].args[rc[1]] if rc and rc[1] < len(rc[0].args) else None
                if rc: rc[1] += 1
                h = deref(h, sub) if h is not None else sub.fresh_var()
                if isinstance(h, Var): bind(h, v, sub)
                elif not (isinstance(h, Atom) and h.name == '[]'):
                    if not sub_fail(): break
                    sub_arg_snap = list(sub.regs); continue
            else:
                wc = sub.write_ctx
                if wc and wc[1] < len(wc[0].args): wc[0].args[wc[1]] = v; wc[1] += 1
        elif op == 'set_integer':
            _, n = instr
            wc = sub.write_ctx
            if wc and wc[1] < len(wc[0].args): wc[0].args[wc[1]] = Int(n); wc[1] += 1
        elif op == 'unify_void':
            _, n = instr
            if sub.mode == 'read':
                rc = sub.read_ctx
                if rc: rc[1] += n
            else:
                wc = sub.write_ctx
                if wc:
                    for _ in range(n):
                        if wc[1] < len(wc[0].args): wc[0].args[wc[1]] = Atom('_'); wc[1] += 1
        elif op == 'call_pc':
            _, target_ip, _arity, _label = instr
            sub.cp = sub_ip
            if target_ip >= 0:
                sub_ip = target_ip
                sub_arg_snap = list(sub.regs)
            else:
                if not sub_fail(): break
                sub_arg_snap = list(sub.regs)
                continue
        elif op == 'execute_pc':
            _, target_ip, _label = instr
            if target_ip >= 0:
                sub_ip = target_ip
                sub_arg_snap = list(sub.regs)
            else:
                if not sub_fail(): break
                sub_arg_snap = list(sub.regs)
                continue
        elif op == 'call':
            _, label, _arity = instr
            sub.cp = sub_ip
            target = labels.get(label, -1)
            if target >= 0:
                sub_ip = target
                sub_arg_snap = list(sub.regs)
            else:
                if not sub_fail(): break
                sub_arg_snap = list(sub.regs)
                continue
        elif op == 'execute':
            _, label = instr
            target = labels.get(label, -1)
            if target >= 0:
                sub_ip = target
                sub_arg_snap = list(sub.regs)
            else:
                if not sub_fail(): break
                sub_arg_snap = list(sub.regs)
                continue
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
        elif op in ('neck_cut', 'cut_ite'):
            sub.b = sub.cut_b if op == 'neck_cut' else (
                sub.stack[sub.b].saved_b if sub.b >= 0 and isinstance(sub.stack[sub.b], ChoicePoint) else sub.b
            )
        elif op == 'get_level':
            _, reg = instr
            set_reg(sub, reg, sub.b)
        elif op == 'cut':
            _, reg = instr
            sub.b = get_reg(sub, reg)
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
            ok = _execute_builtin(builtin, arity, sub)
            if not ok:
                if not sub_fail(): break
                sub_arg_snap = list(sub.regs)
                continue
        elif op == 'switch_on_term_pc':
            _, lv_pc, lc_pc, ll_pc, ls_pc = instr
            val = deref(get_reg(sub, 1), sub)
            if isinstance(val, Var):
                tgt = lv_pc
            elif isinstance(val, Atom):
                tgt = lc_pc
            elif isinstance(val, (Int, Float)):
                tgt = lc_pc
            elif isinstance(val, Ref):
                h = sub.heap.get(val.addr)
                if isinstance(h, Compound) and h.functor == '.' and len(h.args) == 2:
                    tgt = ll_pc
                else:
                    tgt = ls_pc
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
