:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% python_wam_bindings.pl - WAM-specific Python bindings for UnifyWeaver
%
% Provides the register-level and state-management bindings used by the
% WAM-to-Python transpilation target. The Python design uses trail-based
% mutable state (classical WAM), not persistent/immutable data structures.
%
% Key design decisions:
%   - Trail-based mutable state: choice points save a trail mark; backtrack
%     undoes bindings since that mark. Mirrors classical WAM and the Rust target.
%   - No extra Python dependencies: pure stdlib + dataclasses.
%   - Register encoding: A1→1, X1→101, Y1→201 (same as all other targets).
%   - Value representation: simple class hierarchy (Atom, Compound, Var, etc.)
%   - WamState: single mutable object with regs, heap, stack, trail, pdl.
%   - ChoicePoint: saves trail_top and heap_top for O(1) undo.
%
% Performance notes:
%   - Use list-based heap and trail (append is O(1) amortized in CPython)
%   - Regs are a pre-allocated list of 512 slots (indexed by int)
%   - Avoid dict overhead for hot-path register access
%   - Trail-based undo is O(k) where k = bindings since last choice point
%
% See: docs/design/WAM_PERF_OPTIMIZATION_LOG.md

:- module(python_wam_bindings, [
	init_python_wam_bindings/0,
	python_wam_value_type/0,          % Emit the Value class hierarchy
	python_wam_state_type/0,          % Emit WamState class
	python_wam_choicepoint_type/0,    % Emit ChoicePoint class
	python_wam_env_frame_type/0,      % Emit EnvFrame class
	python_wam_helpers/0,             % Emit deref, get_reg, set_reg, eval_arith
	python_wam_type_header/1,         % -Code: full type + helper preamble
	python_wam_reg_default/2,         % +Type, -PythonExpr
	python_wam_result_wrap/2,         % +Type, -PythonExpr  (wraps `rv`)
	python_wam_result_wrap_rv/3       % +Type, +Index, -PythonExpr (wraps `rv_I`)
]).

:- use_module('../core/binding_registry').

%% init_python_wam_bindings
init_python_wam_bindings.

% ============================================================================
% Value class hierarchy — tagged classes for WAM terms
% ============================================================================
%
% Trail-based design: Var uses a mutable ref cell (one-element list).
% Unification binds by mutating the ref cell; trail records the address
% so backtrack can reset it.

python_wam_value_type :-
	writeln(
"class Atom:
    __slots__ = ('name',)
    def __init__(self, name):
        self.name = name
    def __eq__(self, other):
        return isinstance(other, Atom) and self.name == other.name
    def __hash__(self):
        return hash(('Atom', self.name))
    def __repr__(self):
        return f'Atom({self.name!r})'

class Compound:
    __slots__ = ('functor', 'args')
    def __init__(self, functor, args):
        self.functor = functor
        self.args = list(args)
    def __eq__(self, other):
        return (isinstance(other, Compound) and self.functor == other.functor
                and self.args == other.args)
    def __hash__(self):
        return hash(('Compound', self.functor, tuple(self.args)))
    def __repr__(self):
        return f'Compound({self.functor!r}, {self.args!r})'

class Var:
    __slots__ = ('ref',)
    def __init__(self):
        self.ref = None  # None = unbound, otherwise bound value
    def __repr__(self):
        return f'Var({self.ref!r})'

class Int:
    __slots__ = ('n',)
    def __init__(self, n):
        self.n = n
    def __eq__(self, other):
        return isinstance(other, Int) and self.n == other.n
    def __hash__(self):
        return hash(('Int', self.n))
    def __repr__(self):
        return f'Int({self.n})'

class Float:
    __slots__ = ('f',)
    def __init__(self, f):
        self.f = f
    def __eq__(self, other):
        return isinstance(other, Float) and self.f == other.f
    def __hash__(self):
        return hash(('Float', self.f))
    def __repr__(self):
        return f'Float({self.f})'

class Ref:
    __slots__ = ('addr',)
    def __init__(self, addr):
        self.addr = addr
    def __eq__(self, other):
        return isinstance(other, Ref) and self.addr == other.addr
    def __hash__(self):
        return hash(('Ref', self.addr))
    def __repr__(self):
        return f'Ref({self.addr})'").

% ============================================================================
% EnvFrame — environment frame for Allocate/Deallocate
% ============================================================================

python_wam_env_frame_type :-
	writeln(
"class EnvFrame:
    __slots__ = ('saved_cp', 'y_regs')
    def __init__(self, saved_cp, y_regs=None):
        self.saved_cp = saved_cp
        self.y_regs = y_regs if y_regs is not None else {}").

% ============================================================================
% ChoicePoint — saves trail mark + heap top for O(1) undo
% ============================================================================
%
% Trail-based backtracking: on backtrack, unwind trail from current position
% back to trail_top, resetting each heap cell. Then truncate heap to heap_top.

python_wam_choicepoint_type :-
	writeln(
"class ChoicePoint:
    __slots__ = ('n_args', 'saved_regs', 'saved_e', 'saved_cp',
                 'next_clause', 'trail_top', 'heap_top',
                 'saved_b', 'cut_b')
    def __init__(self, n_args, saved_regs, saved_e, saved_cp,
                 next_clause, trail_top, heap_top, saved_b, cut_b):
        self.n_args = n_args
        self.saved_regs = saved_regs
        self.saved_e = saved_e
        self.saved_cp = saved_cp
        self.next_clause = next_clause
        self.trail_top = trail_top
        self.heap_top = heap_top
        self.saved_b = saved_b
        self.cut_b = cut_b").

% ============================================================================
% WamState — mutable WAM machine state (trail-based)
% ============================================================================

python_wam_state_type :-
	writeln(
"class WamState:
    __slots__ = ('regs', 'heap', 'stack', 'trail', 'pdl',
                 'mode', 's', 'cp', 'b', 'e', 'cut_b',
                 'fail', 'halt', 'labels', 'code',
                 'var_counter', 'step_count', 'step_limit',
                 'foreign_predicates')
    def __init__(self):
        self.regs = [None] * 512       # A1→1, X1→101, Y1→201
        self.heap = []                  # list of terms
        self.stack = []                 # environments + choice points
        self.trail = []                 # heap addresses to unbind on backtrack
        self.pdl = []                   # unification push-down list
        self.mode = 'write'             # 'read' | 'write'
        self.s = 0                      # structure pointer
        self.cp = None                  # continuation pointer (label or PC)
        self.b = -1                     # choice point index into stack
        self.e = -1                     # environment index into stack
        self.cut_b = -1                 # cut barrier
        self.fail = False               # failure flag
        self.halt = False               # halt flag
        self.labels = {}                # label → PC mapping
        self.code = []                  # instruction list
        self.var_counter = 0            # fresh variable counter
        self.step_count = 0             # execution step counter
        self.step_limit = 0             # 0 = unlimited
        self.foreign_predicates = {}    # name → callable").

% ============================================================================
% Helper functions — deref, get_reg, set_reg, unify, trail, eval_arith
% ============================================================================

python_wam_helpers :-
	writeln(
"# ============================================================================
# Register access helpers
# ============================================================================

def get_reg(state, n):
    \"\"\"Look up register n, dereference if it's a Var.\"\"\"
    val = state.regs[n]
    return deref(val, state) if val is not None else None

def set_reg(state, n, val):
    \"\"\"Set register n to val.\"\"\"
    state.regs[n] = val

# ============================================================================
# Dereferencing
# ============================================================================

def deref(val, state):
    \"\"\"Dereference through Var ref chain and heap Ref pointers.\"\"\"
    while True:
        if isinstance(val, Var):
            if val.ref is None:
                return val
            val = val.ref
        elif isinstance(val, Ref):
            if val.addr < len(state.heap):
                val = state.heap[val.addr]
            else:
                return val
        else:
            return val

# ============================================================================
# Trail-based binding
# ============================================================================

def trail_if_needed(addr, state):
    \"\"\"Conditionally trail a heap address for backtracking.\"\"\"
    if state.b >= 0 and state.b < len(state.stack):
        cp = state.stack[state.b]
        if isinstance(cp, ChoicePoint) and addr < cp.heap_top:
            state.trail.append(addr)
    else:
        state.trail.append(addr)

def bind(v, val, state):
    \"\"\"Bind Var v to val, with trailing.\"\"\"
    if isinstance(v, Var) and v.ref is None:
        v.ref = val

# ============================================================================
# Heap operations
# ============================================================================

def heap_put(state, term):
    \"\"\"Append term to heap, return its address.\"\"\"
    addr = len(state.heap)
    state.heap.append(term)
    return addr

def heap_get(state, addr):
    \"\"\"Read heap cell at addr.\"\"\"
    if 0 <= addr < len(state.heap):
        return state.heap[addr]
    return None

# ============================================================================
# Unification (Robinson's algorithm, iterative with PDL)
# ============================================================================

def unify(a, b, state):
    \"\"\"Unify two terms. Returns True on success, False on failure.\"\"\"
    state.pdl.clear()
    state.pdl.append((a, b))
    while state.pdl:
        t1, t2 = state.pdl.pop()
        d1 = deref(t1, state)
        d2 = deref(t2, state)
        if d1 is d2:
            continue
        if isinstance(d1, Var):
            bind(d1, d2, state)
            continue
        if isinstance(d2, Var):
            bind(d2, d1, state)
            continue
        if isinstance(d1, Atom) and isinstance(d2, Atom):
            if d1.name != d2.name:
                return False
            continue
        if isinstance(d1, Int) and isinstance(d2, Int):
            if d1.n != d2.n:
                return False
            continue
        if isinstance(d1, Float) and isinstance(d2, Float):
            if d1.f != d2.f:
                return False
            continue
        if isinstance(d1, Compound) and isinstance(d2, Compound):
            if d1.functor != d2.functor or len(d1.args) != len(d2.args):
                return False
            for i in range(len(d1.args)):
                state.pdl.append((d1.args[i], d2.args[i]))
            continue
        if isinstance(d1, Ref) and isinstance(d2, Ref):
            if d1.addr == d2.addr:
                continue
            h1 = heap_get(state, d1.addr)
            h2 = heap_get(state, d2.addr)
            if h1 is not None and h2 is not None:
                state.pdl.append((h1, h2))
                continue
        return False
    return True

# ============================================================================
# Choice point operations (trail-based)
# ============================================================================

def push_choice_point(state, n_args, next_clause):
    \"\"\"Create a new choice point saving current state.\"\"\"
    saved_regs = state.regs[:n_args + 1]  # save A1..An
    cp = ChoicePoint(
        n_args=n_args,
        saved_regs=list(saved_regs),
        saved_e=state.e,
        saved_cp=state.cp,
        next_clause=next_clause,
        trail_top=len(state.trail),
        heap_top=len(state.heap),
        saved_b=state.b,
        cut_b=state.cut_b
    )
    state.stack.append(cp)
    state.b = len(state.stack) - 1

def restore_choice_point(state):
    \"\"\"Backtrack: undo trail, restore regs from top choice point.\"\"\"
    if state.b < 0 or state.b >= len(state.stack):
        return False
    cp = state.stack[state.b]
    if not isinstance(cp, ChoicePoint):
        return False
    # Undo trail bindings since trail_top
    unwind_trail(state, cp.trail_top)
    # Truncate heap
    del state.heap[cp.heap_top:]
    # Restore registers
    for i in range(len(cp.saved_regs)):
        state.regs[i] = cp.saved_regs[i]
    # Restore control state
    state.e = cp.saved_e
    state.cp = cp.saved_cp
    state.cut_b = cp.cut_b
    return True

def pop_choice_point(state):
    \"\"\"Trust: restore and remove top choice point.\"\"\"
    if not restore_choice_point(state):
        return False
    if state.b >= 0 and state.b < len(state.stack):
        cp = state.stack[state.b]
        state.b = cp.saved_b
        del state.stack[state.b + 1:]
    return True

def unwind_trail(state, saved_top):
    \"\"\"Undo variable bindings recorded on the trail since saved_top.\"\"\"
    while len(state.trail) > saved_top:
        addr = state.trail.pop()
        if isinstance(addr, Var):
            addr.ref = None
        elif isinstance(addr, int) and addr < len(state.heap):
            cell = state.heap[addr]
            if isinstance(cell, Var):
                cell.ref = None

# ============================================================================
# Arithmetic evaluation
# ============================================================================

def eval_arith(val, state):
    \"\"\"Evaluate arithmetic expression, return numeric result or None.\"\"\"
    d = deref(val, state)
    if isinstance(d, Int):
        return float(d.n)
    if isinstance(d, Float):
        return d.f
    if isinstance(d, Compound):
        if d.functor == '+' and len(d.args) == 2:
            a = eval_arith(d.args[0], state)
            b = eval_arith(d.args[1], state)
            if a is not None and b is not None:
                return a + b
        elif d.functor == '-' and len(d.args) == 2:
            a = eval_arith(d.args[0], state)
            b = eval_arith(d.args[1], state)
            if a is not None and b is not None:
                return a - b
        elif d.functor == '*' and len(d.args) == 2:
            a = eval_arith(d.args[0], state)
            b = eval_arith(d.args[1], state)
            if a is not None and b is not None:
                return a * b
        elif d.functor == '/' and len(d.args) == 2:
            a = eval_arith(d.args[0], state)
            b = eval_arith(d.args[1], state)
            if a is not None and b is not None and b != 0.0:
                return a / b
        elif d.functor == 'mod' and len(d.args) == 2:
            a = eval_arith(d.args[0], state)
            b = eval_arith(d.args[1], state)
            if a is not None and b is not None and b != 0.0:
                return float(int(a) % int(b))
    return None

# ============================================================================
# Foreign predicate registry
# ============================================================================

_foreign_predicates = {}

def register_foreign(name, arity, fn):
    \"\"\"Register a foreign (Python) predicate for call_foreign dispatch.\"\"\"
    _foreign_predicates[f'{name}/{arity}'] = fn

def execute_foreign(functor, arity, args, state):
    \"\"\"Execute a registered foreign predicate. Returns True/False.\"\"\"
    key = f'{functor}/{arity}'
    fn = _foreign_predicates.get(key) or state.foreign_predicates.get(key)
    if fn is not None:
        return fn(args, state)
    return False").

% ============================================================================
% Combined preamble emitter
% ============================================================================

python_wam_type_header(Code) :-
	with_output_to(string(Code), (
		writeln("# WAM Runtime Types — Generated by UnifyWeaver"),
		writeln("# Trail-based mutable state, no external dependencies"),
		nl,
		python_wam_value_type,         nl,
		python_wam_env_frame_type,     nl,
		python_wam_choicepoint_type,   nl,
		python_wam_state_type,         nl,
		python_wam_helpers
	)).

% ============================================================================
% Type-driven codegen helpers (used by wam_python_target.pl)
% ============================================================================

%% python_wam_reg_default(+Type, -PythonExpr)
%  Default Value constructor when a register lookup returns None.
python_wam_reg_default(atom,        'Atom("")').
python_wam_reg_default(integer,     'Int(0)').
python_wam_reg_default(float,       'Float(0.0)').
python_wam_reg_default(vlist_atoms, '[]').

%% python_wam_result_wrap(+Type, -PythonExpr)
%  Wrap kernel result `rv` into a Value.
python_wam_result_wrap(integer, 'Int(rv)').
python_wam_result_wrap(atom,    'Atom(str(rv))').
python_wam_result_wrap(float,   'Float(rv)').

%% python_wam_result_wrap_rv(+Type, +Index, -PythonExpr)
%  Like python_wam_result_wrap but for rv_<Index> tuple components.
python_wam_result_wrap_rv(integer, I, Expr) :-
	format(atom(Expr), 'Int(rv_~w)', [I]).
python_wam_result_wrap_rv(atom, I, Expr) :-
	format(atom(Expr), 'Atom(str(rv_~w))', [I]).
python_wam_result_wrap_rv(float, I, Expr) :-
	format(atom(Expr), 'Float(rv_~w)', [I]).
