"""
WamRuntime.py — Python WAM runtime library
Trail-based mutable state. No external dependencies.
"""

from __future__ import annotations
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

def get_reg(state: WamState, n: int) -> Term:
    return state.regs[n]

def set_reg(state: WamState, n: int, val: Term) -> None:
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


# -- Trail ------------------------------------------------------------------

def trail_if_needed(addr: int, state: WamState) -> None:
    """Trail a heap address if it might need to be undone on backtrack."""
    if addr < state.b_heap_top() or addr < state.b_trail_top():
        state.trail.append(addr)
        state.trail_len += 1

def undo_trail(state: WamState, mark: int) -> None:
    """Unbind all trailed variables since mark."""
    while state.trail_len > mark:
        addr = state.trail.pop()
        state.trail_len -= 1
        v = state.heap.get(addr)
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
    """Bind a variable, trailing if necessary."""
    # Find heap address of v for trailing
    # Simple approach: trail the var directly via id lookup
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
    saved_regs = state.regs[:n_args].copy()
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
    cp = state.stack[state.b]
    assert isinstance(cp, ChoicePoint)
    # Undo trail
    undo_trail(state, cp.trail_top)
    # Trim heap (dict-based — remove entries >= mark, reset heap_len)
    heap_trim(state, cp.heap_top)
    # Restore registers
    state.regs[:cp.n_args] = cp.saved_regs.copy()
    state.e = cp.saved_e
    state.cp = cp.saved_cp
    if next_clause is not None:
        cp.next_clause = next_clause

def pop_choice_point(state: WamState) -> None:
    """Trust: restore from choice point and remove it (last clause)."""
    cp = state.stack[state.b]
    assert isinstance(cp, ChoicePoint)
    restore_choice_point(state)
    # Remove the choice point
    state.stack.pop()
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

def eval_arith(term: Term, state: WamState):
    """Evaluate an arithmetic expression, return int or float."""
    t = deref(term, state)
    if isinstance(t, Int):    return t.n
    if isinstance(t, Float):  return t.f
    if isinstance(t, Compound):
        args = [eval_arith(a, state) for a in t.args]
        # Unary minus: -(X)
        if t.functor == '-' and len(t.args) == 1:
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
            # Unary
            'abs': lambda a: abs(a),
            'sign': lambda a: (1 if a > 0 else (-1 if a < 0 else 0)),
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
        if t.functor in ops:
            return ops[t.functor](*args)
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

def execute_foreign(functor: str, arity: int, args: List[Term], state: WamState) -> bool:
    """Dispatch to a registered foreign predicate.

    Returns True on success, False on failure (triggers backtracking).
    """
    key = (functor, arity)
    if key not in _foreign_predicates:
        raise WAMError(f"Unknown foreign predicate: {functor}/{arity}")
    fn = _foreign_predicates[key]
    return fn(args, state)


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
    for label, instrs in raw_program.items():
        labels[label] = len(code)
        if isinstance(instrs, (list, tuple)):
            code.extend(instrs)
        else:
            code.append(instrs)
    # Pre-resolve label references
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
        _, next_label, n_args = instr
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
    return instr


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
    code_len = len(code)

    def fail():
        nonlocal ip
        if state.b < 0:
            return False
        cp = state.stack[state.b]
        restore_choice_point(state)
        # next_clause is now a pre-resolved PC int
        next_ip = cp.next_clause
        if isinstance(next_ip, int) and next_ip >= 0:
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
            _, atom, reg = instr
            set_reg(state, reg, Atom(atom))

        elif op == 'put_nil':
            _, reg = instr
            set_reg(state, reg, Atom('[]'))

        elif op == 'put_integer':
            _, n, reg = instr
            set_reg(state, reg, Int(n))

        elif op == 'put_float':
            _, f, reg = instr
            set_reg(state, reg, Float(f))

        elif op == 'put_structure':
            _, functor, arity, reg = instr
            addr = heap_put(state, Compound(functor, [None]*arity))
            set_reg(state, reg, Ref(addr))
            state.mode = 'write'
            state.s = addr

        elif op == 'put_list':
            _, reg = instr
            addr = heap_put(state, Compound('.', [None, None]))
            set_reg(state, reg, Ref(addr))
            state.mode = 'write'
            state.s = addr

        elif op == 'get_variable':
            _, reg, hreg = instr
            set_reg(state, hreg, get_reg(state, reg))

        elif op == 'get_value':
            _, reg1, reg2 = instr
            if not unify(get_reg(state, reg1), get_reg(state, reg2), state):
                if not fail(): return False
                continue

        elif op == 'get_constant':
            _, atom, reg = instr
            val = deref(get_reg(state, reg), state)
            if isinstance(val, Var):
                bind(val, Atom(atom), state)
            elif not (isinstance(val, Atom) and val.name == atom):
                if not fail(): return False
                continue

        elif op == 'get_nil':
            _, reg = instr
            val = deref(get_reg(state, reg), state)
            if isinstance(val, Var):
                bind(val, Atom('[]'), state)
            elif not (isinstance(val, Atom) and val.name == '[]'):
                if not fail(): return False
                continue

        elif op == 'get_integer':
            _, n, reg = instr
            val = deref(get_reg(state, reg), state)
            if isinstance(val, Var):
                bind(val, Int(n), state)
            elif not (isinstance(val, Int) and val.n == n):
                if not fail(): return False
                continue

        elif op == 'get_structure':
            _, functor, arity, reg = instr
            val = deref(get_reg(state, reg), state)
            if isinstance(val, Var):
                addr = heap_put(state, Compound(functor, [None]*arity))
                bind(val, Ref(addr), state)
                state.mode = 'write'
                state.s = addr
            elif isinstance(val, Ref):
                h = state.heap[val.addr]
                if isinstance(h, Compound) and h.functor == functor and len(h.args) == arity:
                    state.mode = 'read'
                    state.s = val.addr
                else:
                    if not fail(): return False
                    continue
            elif isinstance(val, Compound) and val.functor == functor and len(val.args) == arity:
                state.mode = 'read'
            else:
                if not fail(): return False
                continue

        elif op == 'get_list':
            _, reg = instr
            val = deref(get_reg(state, reg), state)
            if isinstance(val, Var):
                addr = heap_put(state, Compound('.', [None, None]))
                bind(val, Ref(addr), state)
                state.mode = 'write'
                state.s = addr
            elif isinstance(val, Ref):
                h = state.heap[val.addr]
                if isinstance(h, Compound) and h.functor == '.' and len(h.args) == 2:
                    state.mode = 'read'
                    state.s = val.addr
                else:
                    if not fail(): return False
                    continue
            else:
                if not fail(): return False
                continue

        elif op == 'unify_variable':
            _, reg = instr
            if state.mode == 'read':
                h = state.heap[state.s]
                if isinstance(h, Compound):
                    set_reg(state, reg, h.args[0])
                    state.s += 1
                else:
                    set_reg(state, reg, h)
            else:
                v = state.fresh_var()
                addr = heap_put(state, v)
                set_reg(state, reg, v)

        elif op == 'unify_value':
            _, reg = instr
            if state.mode == 'read':
                h = state.heap[state.s]
                if not unify(get_reg(state, reg), h, state):
                    if not fail(): return False
                    continue
            else:
                heap_put(state, get_reg(state, reg))

        elif op == 'unify_constant':
            _, atom = instr
            if state.mode == 'read':
                h = deref(state.heap[state.s], state)
                if isinstance(h, Var):
                    bind(h, Atom(atom), state)
                elif not (isinstance(h, Atom) and h.name == atom):
                    if not fail(): return False
                    continue
            else:
                heap_put(state, Atom(atom))

        elif op == 'unify_nil':
            if state.mode == 'read':
                h = deref(state.heap[state.s], state)
                if isinstance(h, Var):
                    bind(h, Atom('[]'), state)
                elif not (isinstance(h, Atom) and h.name == '[]'):
                    if not fail(): return False
                    continue
            else:
                heap_put(state, Atom('[]'))

        elif op == 'unify_void':
            _, n = instr
            if state.mode == 'write':
                for _ in range(n):
                    v = state.fresh_var()
                    heap_put(state, v)

        elif op == 'call_pc':
            _, target_ip, _arity, _label = instr
            state.cp = ip   # save continuation (current ip = next instr)
            if target_ip >= 0:
                ip = target_ip
            else:
                if not fail(): return False
                continue

        elif op == 'execute_pc':
            _, target_ip, _label = instr
            if target_ip >= 0:
                ip = target_ip
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
            else:
                if not fail(): return False
                continue

        elif op == 'execute':
            # Legacy unresolved execute (fallback)
            _, label = instr
            target = labels.get(label, -1)
            if target >= 0:
                ip = target
            else:
                if not fail(): return False
                continue

        elif op == 'proceed':
            if state.cp is None:
                return True
            if isinstance(state.cp, int):
                ip = state.cp
                state.cp = None
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

        elif op == 'retry_me_else_pc':
            _, next_pc = instr
            restore_choice_point(state, next_clause=next_pc)

        elif op == 'try_me_else':
            # Legacy unresolved
            _, next_label, n_args = instr
            push_choice_point(state, n_args, labels.get(next_label, -1))

        elif op == 'retry_me_else':
            # Legacy unresolved
            _, next_label = instr
            restore_choice_point(state, next_clause=labels.get(next_label, -1))

        elif op == 'trust_me':
            pop_choice_point(state)

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
            _, n_perm = instr
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
                ok = execute_foreign(functor, arity, args, state)
            except WAMError:
                ok = False
            if not ok:
                if not fail(): return False
                continue

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

        else:
            raise WAMError(f"Unknown WAM opcode: {op}")

    return True
