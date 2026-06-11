% test_wam_python_is_binding.pl
%
% Regression test for arithmetic / term construction through the Python WAM
% bytecode interpreter.
%
% Root cause (fixed): `put_structure F, Ai` unconditionally bound whatever var
% the target register previously held. For `X is Expr` the compiler emits
%   get_variable X1, A2   % save the result var R (head arg 2) into X1
%   put_value   X1, A1    % A1 = R   (is/2 arg 1 = the result)
%   put_structure +/2, A2 % A2 = the expression structure
%   set_constant ...      % fill the structure
%   builtin_call is/2, 2
% so A1 and A2 BOTH point at R when put_structure runs. Binding R to the freshly
% built expression structure clobbered A1 — is/2's result target — so every
% `X is Expr` with an UNBOUND X failed (it only "worked" when X was already
% bound, i.e. a ground check like `3 is 1+2`).
%
% Fix: put_structure/put_list bind the old var ONLY for X (temporary) registers
% (reg > _A_MAX) — a nested sub-term SLOT created by an earlier
% set_variable/unify_variable, where binding links the inner term into its
% parent (e.g. error(type_error(..),..)). For A (argument) registers the prior
% contents are a dead call-output slot that may alias a live argument, so it is
% overwritten without binding. Unifying a structure with an existing var is
% get_structure's job, not put_structure's.
%
% Skipped automatically when python3 is unavailable.

:- use_module('../src/unifyweaver/targets/wam_python_target').

:- dynamic user:gg/2, user:dbl/2, user:acc/3, user:mk/1, user:nest/1, user:gchk/0.

user:gg(a, R)    :- R is 1 + 0.            % unbound result via builtin is/2
user:dbl(N, M)   :- M is N + N.            % set_value fills (variable operands)
user:acc([], A, A).                        % list-recursive accumulator: the
user:acc([_|T], A, S) :- A1 is A + 1, acc(T, A1, S).
user:mk(X)       :- X = a + b.             % get_structure unify (must still work)
user:nest(Z)     :- Z = f(g(1), h(2, 3)).  % nested construction (must still work)
user:gchk        :- 3 is 1 + 2.            % ground check (already worked)

python3_available :-
    catch(( process_create(path(python3), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

:- begin_tests(wam_python_is_binding, [condition(python3_available)]).

test(is_binding_and_construction) :-
    Preds = [user:gg/2, user:dbl/2, user:acc/3, user:mk/1, user:nest/1, user:gchk/0],
    Dir = 'output/test_wam_python_is_binding',
    ( exists_directory(Dir) -> delete_directory_and_contents(Dir) ; true ),
    write_wam_python_project(Preds, [module_name(isproj)], Dir),
    harness_source(Src),
    atomic_list_concat([Dir, '/h.py'], HPath),
    setup_call_cleanup(open(HPath, write, S, [encoding(utf8)]),
                       write(S, Src), close(S)),
    format(atom(Cmd), 'cd ~w && python3 h.py 2>&1', [Dir]),
    process_create(path(sh), ['-c', Cmd],
                   [stdout(pipe(Out)), stderr(std), process(Pid)]),
    read_string(Out, _, OutStr), close(Out),
    process_wait(Pid, Status),
    ( Status == exit(0), sub_string(OutStr, _, _, _, "ALL PASS")
    ->  true
    ;   format(user_error, "~n[is-binding harness output]~n~w~n", [OutStr]),
        throw(wam_python_is_binding_failed(Status)) ),
    ( exists_directory(Dir) -> delete_directory_and_contents(Dir) ; true ).

:- end_tests(wam_python_is_binding).

harness_source(
"import sys
sys.path.insert(0, '.')
from wam_runtime import *
from predicates import build_program
_code, _labels = load_program(build_program())

def resolve(t, s):
    t = deref(t, s)
    if isinstance(t, Ref):
        t = s.heap[t.addr]
    return t

def run(entry, args):
    s = WamState(); outs = []
    for i, a in enumerate(args, 1):
        if a is None:
            v = s.fresh_var(); set_reg(s, i, v); outs.append(v)
        else:
            set_reg(s, i, a)
    ok = run_wam(_code, _labels, entry, s)
    return bool(ok), [resolve(v, s) for v in outs]

fails = 0
def chk(name, got, want):
    global fails
    if got != want:
        fails += 1
        print('FAIL', name, 'got', got, 'want', want)

# X is 1+0 with X unbound -> X = 1
ok, outs = run('gg/2', [Atom('a'), None])
chk('gg ok', ok, True)
chk('gg X', isinstance(outs[0], Int) and outs[0].n, 1)

# M is N+N with N=5 -> M = 10
ok, outs = run('dbl/2', [Int(5), None])
chk('dbl ok', ok, True)
chk('dbl M', isinstance(outs[0], Int) and outs[0].n, 10)

# list-recursive accumulator: acc([x,x,x], 0, S) -> S = 3
lst = Atom('[]')
for _ in range(3):
    lst = Compound('.', [Atom('x'), lst])
ok, outs = run('acc/3', [lst, Int(0), None])
chk('acc ok', ok, True)
chk('acc S', isinstance(outs[0], Int) and outs[0].n, 3)

# X = a+b (get_structure unify) still binds correctly (functor stored as +/2)
ok, outs = run('mk/1', [None])
x = outs[0] if ok else None
chk('mk ok', ok, True)
chk('mk X', isinstance(x, Compound) and x.functor.startswith('+') and len(x.args) == 2, True)

# nested construction Z = f(g(1), h(2,3)) still binds correctly
ok, outs = run('nest/1', [None])
z = outs[0] if ok else None
chk('nest ok', ok, True)
chk('nest Z', isinstance(z, Compound) and z.functor.startswith('f') and len(z.args) == 2, True)

# ground check still works
ok, _ = run('gchk/0', [])
chk('gchk', ok, True)

print('ALL PASS' if fails == 0 else ('FAILURES: %d' % fails))
").
