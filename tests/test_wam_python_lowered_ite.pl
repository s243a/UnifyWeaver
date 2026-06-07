% test_wam_python_lowered_ite.pl
%
% End-to-end execution test for WAM-Python if-then-else / negation / once
% lowering (emit_mode(lowered)). Counterpart to the Go / Rust / C++ /
% Haskell / F# / Clojure / LLVM / Elixir lowered-ITE exec tests. Pins:
%
%   * simple ITE         — pite/2;
%   * negation (\+)       — pneg/1 (commit is the !/0 builtin: then = fail/0,
%                          else = true/0, run after a trail rollback);
%   * sequential ITEs    — pseqite/3 (two sibling blocks);
%   * nested ITEs         — pnestite/2 (inner block in the then-arm).
%
% Python previously kept every predicate with a choice point (including the
% internal try_me_else of a soft-cut block) in interpreter mode — sound but
% not lowered. Folding clause 1 through wam_ite_structurer lets these lower
% to native Python if/else. This test ALSO guards a prerequisite fix: the
% lowered `builtin_call` instruction now dispatches through execute_builtin
% (the runtime's standard-builtin dispatcher) instead of execute_foreign,
% which only knew user-registered foreign predicates and raised
% "Unknown foreign predicate" for =/2, >/2, true/0, fail/0, ... — so any
% lowered predicate using a builtin (every ITE here) used to crash.
%
% Generates a lowered Python project and calls each lowered function with
% the A-registers set, asserting the boolean outcome. Skipped unless
% `python3` is on PATH.

:- use_module(library(plunit)).
:- use_module(library(filesex)).
:- use_module(library(process)).
:- use_module('../src/unifyweaver/targets/wam_python_target').

:- dynamic user:pite/2.
:- dynamic user:pneg/1.
:- dynamic user:pseqite/3.
:- dynamic user:pnestite/2.

user:pite(X, Y)       :- ( X > 0 -> Y = pos ; Y = nonpos ).
user:pneg(X)          :- \+ X > 0.
user:pseqite(X, Y, Z) :- ( X > 0 -> Y = pos ; Y = nonpos ),
                         ( X > 5 -> Z = big ; Z = small ).
user:pnestite(X, Y)   :- ( X > 0 -> ( X > 10 -> Y = big ; Y = small ) ; Y = neg ).

python3_available :-
    catch(( process_create(path(python3), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

:- begin_tests(wam_python_lowered_ite, [condition(python3_available)]).

test(ite_exec_parity) :-
    Dir = 'output/test_wam_python_ite_exec',
    ( exists_directory(Dir) -> delete_directory_and_contents(Dir) ; true ),
    % 1. Generate the lowered Python project.
    write_wam_python_project(
        [user:pite/2, user:pneg/1, user:pseqite/3, user:pnestite/2],
        [module_name('iteproj'), emit_mode(lowered)], Dir),
    % Sanity: the predicates must actually be lowered (native if/else),
    % else the test would pass vacuously.
    atomic_list_concat([Dir, '/predicates.py'], PredsPath),
    read_file_to_string(PredsPath, PredsSrc, []),
    forall(member(F, ["def pred_pite_2", "def pred_pneg_1",
                      "def pred_pseqite_3", "def pred_pnestite_2"]),
           assertion(sub_string(PredsSrc, _, _, _, F))),
    assertion(sub_string(PredsSrc, _, _, _, "execute_builtin")),
    % 2. Write the harness.
    atomic_list_concat([Dir, '/harness.py'], HPath),
    harness_source(Src),
    setup_call_cleanup(open(HPath, write, S, [encoding(utf8)]),
                       write(S, Src), close(S)),
    % 3. Run it.
    format(atom(Cmd), 'cd ~w && python3 harness.py 2>&1', [Dir]),
    process_create(path(sh), ['-c', Cmd],
                   [stdout(pipe(Out)), stderr(std), process(Pid)]),
    read_string(Out, _, OutStr), close(Out),
    process_wait(Pid, Status),
    ( Status == exit(0), sub_string(OutStr, _, _, _, "ALL 15 PASS")
    ->  true
    ;   format(user_error, "~n[python ite test output]~n~w~n", [OutStr]),
        throw(python_ite_test_failed(Status))
    ),
    ( exists_directory(Dir) -> delete_directory_and_contents(Dir) ; true ).

:- end_tests(wam_python_lowered_ite).

% Builds a fresh WamState, sets the A-registers, calls each lowered function
% and checks the boolean result. Atoms are plain Atom(name) values
% (structural equality), so no interning coordination is needed.
harness_source(
"import sys
sys.path.insert(0, '.')
from wam_runtime import *
from predicates import pred_pite_2, pred_pneg_1, pred_pseqite_3, pred_pnestite_2

def fresh():
    return WamState()

def run1(fn, regs):
    s = fresh()
    for i, v in regs:
        set_reg(s, i, v)
    return bool(fn(s))

I = Int
A = Atom
fails = 0
total = 0
def chk(name, got, want):
    global fails, total
    total += 1
    if got != want:
        fails += 1
        print('FAIL', name, 'got', got, 'want', want)

chk('pite(5,pos)',        run1(pred_pite_2, [(1, I(5)), (2, A('pos'))]),        True)
chk('pite(5,nonpos)',     run1(pred_pite_2, [(1, I(5)), (2, A('nonpos'))]),     False)
chk('pite(-1,nonpos)',    run1(pred_pite_2, [(1, I(-1)), (2, A('nonpos'))]),    True)
chk('pite(-1,pos)',       run1(pred_pite_2, [(1, I(-1)), (2, A('pos'))]),       False)
chk('pneg(5)',            run1(pred_pneg_1, [(1, I(5))]),                       False)
chk('pneg(-1)',           run1(pred_pneg_1, [(1, I(-1))]),                      True)
chk('pneg(0)',            run1(pred_pneg_1, [(1, I(0))]),                       True)
chk('pseqite(10,pos,big)',      run1(pred_pseqite_3, [(1, I(10)), (2, A('pos')), (3, A('big'))]),     True)
chk('pseqite(10,pos,small)',    run1(pred_pseqite_3, [(1, I(10)), (2, A('pos')), (3, A('small'))]),   False)
chk('pseqite(3,pos,small)',     run1(pred_pseqite_3, [(1, I(3)), (2, A('pos')), (3, A('small'))]),    True)
chk('pseqite(-1,nonpos,small)', run1(pred_pseqite_3, [(1, I(-1)), (2, A('nonpos')), (3, A('small'))]), True)
chk('pnestite(20,big)',   run1(pred_pnestite_2, [(1, I(20)), (2, A('big'))]),   True)
chk('pnestite(5,small)',  run1(pred_pnestite_2, [(1, I(5)), (2, A('small'))]),  True)
chk('pnestite(-1,neg)',   run1(pred_pnestite_2, [(1, I(-1)), (2, A('neg'))]),   True)
chk('pnestite(20,small)', run1(pred_pnestite_2, [(1, I(20)), (2, A('small'))]), False)

if fails == 0:
    print('ALL', total, 'PASS')
else:
    print(fails, 'FAILURES')
").
