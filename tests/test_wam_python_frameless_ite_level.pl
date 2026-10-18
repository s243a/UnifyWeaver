:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (s243a)
%
% test_wam_python_frameless_ite_level.pl
%
% Probe for the WAM_FLEET_GAPS gap-A2 hazard in its *frameless-Y-write* form
% (the shape the fleet audit had dismissed as unreachable, proven reachable on
% wam_rust in ledger row D50).
%
% The shape
% ---------
% `compile_if_then_else/7` in the shared emitter (`wam_target.pl`) reserves a
% permanent Y register for the if-then-else barrier AFTER it has decided
% whether the clause needs an environment. So a clause that needs no
% environment still gets `get_level Yn` ... `cut Yn` -- with NO `allocate`.
% `sat/2` clause 2 below is exactly that clause (`\+ G` inlines to
% `(G -> fail ; true)` under `ite_use_y_level(true)`, which the Python target
% always enables), and `pick_a/4` / `pick_b/4` are callers that DO hold an
% environment with live Y registers across the call.
%
% Before the fix the Python runtime routed every Y write (reg >= _Y_BASE) into
% the CURRENT environment frame -- the CALLER's frame for an `allocate`-less
% callee. `get_level Y1` therefore overwrote the caller's permanent variable
% Y1 with a choice-point index:
%
%   pick_a(3, gte(1), tagX, Out)  ->  FAILED          (SWI: Out = tagX)
%   pick_b(3, gte(1), tagX, Out)  ->  Out = -1        (SWI: Out = tagX)
%
% i.e. a silent wrong answer, not a crash. The fix keeps ITE barrier levels on
% the if-then-else's own choice point (`ChoicePoint.levels`,
% `record_ite_level/3` + `lookup_ite_level/2` in WamRuntime.py), so the level
% never touches a register and is per-activation for free.
%
% This suite pins BOTH halves:
%   * emission -- `sat/2` really does carry `get_level` in an `allocate`-less
%     clause (if the shared emitter ever stops doing that the probe would go
%     vacuously green, so we assert the shape);
%   * behaviour -- the generated Python project agrees with SWI as oracle,
%     including a recursive caller that keeps two activations of the ITE live.
%
% Skipped automatically when python3 is unavailable.
%
%   swipl -q -g run_tests -t halt tests/test_wam_python_frameless_ite_level.pl

:- module(test_wam_python_frameless_ite_level,
          [test_wam_python_frameless_ite_level/0]).

:- use_module(library(plunit)).
:- use_module(library(lists)).
:- use_module(library(process)).
:- use_module('../src/unifyweaver/targets/wam_python_target',
              [write_wam_python_project/3]).
:- use_module('../src/unifyweaver/targets/wam_target',
              [compile_predicate_to_wam_text/3]).

:- dynamic user:lt/2, user:sat/2, user:pick_a/4, user:pick_b/4, user:wpick/3.

% --- the probe program (also the SWI oracle) -------------------------------

user:lt(A, B) :- A < B.

% Multi-clause callee. Clause 2 needs NO environment: its only permanent
% would be the if-then-else barrier the emitter reserves for the inlined
% negation -- so it emits `get_level Y1` with no `allocate`.
user:sat(_V, any).
user:sat(V, gte(G)) :- \+ user:lt(V, G).

% Callers that DO hold an environment across the call. The Y numbering
% follows first use in the body, so pick_a parks the (unbound) output in Y1
% and pick_b parks the (bound) input tag in Y1 -- the first shape used to
% FAIL, the second used to return the clobbering choice-point index.
user:pick_a(Ver, C, Tag, Out) :- user:sat(Ver, C), Out = Tag.
user:pick_b(Ver, C, Tag, Out) :- user:sat(Ver, C), Tag = Out.

% Recursive caller: several activations of sat/2's if-then-else are live at
% once, each with its own barrier. A register-held level is not
% per-activation; a level on the guard choice point is.
user:wpick(0, Tag, Tag).
user:wpick(N, Tag, Out) :-
    N > 0,
    user:sat(N, gte(1)),
    M is N - 1,
    user:wpick(M, Tag, Out).

probe_preds([user:lt/2, user:sat/2, user:pick_a/4, user:pick_b/4,
             user:wpick/3]).

% --- oracle cases ----------------------------------------------------------
%
% probe_case(Id, PythonEntry, PythonArgs, Goal, OutVarPos)
%   OutVarPos = 0 -> the goal has no output, only success/failure.

probe_case(sat_any,     'sat/2',    [int(3), atom(any)],
           user:sat(3, any),                                   0).
probe_case(sat_gte_t,   'sat/2',    [int(3), gte(1)],
           user:sat(3, gte(1)),                                0).
probe_case(sat_gte_f,   'sat/2',    [int(0), gte(1)],
           user:sat(0, gte(1)),                                0).
probe_case(pick_a_t,    'pick_a/4', [int(3), gte(1), atom(tagX), out],
           user:pick_a(3, gte(1), tagX, O), O).
probe_case(pick_b_t,    'pick_b/4', [int(3), gte(1), atom(tagX), out],
           user:pick_b(3, gte(1), tagX, O), O).
probe_case(pick_a_f,    'pick_a/4', [int(0), gte(1), atom(tagX), out],
           user:pick_a(0, gte(1), tagX, _),                    0).
probe_case(pick_b_f,    'pick_b/4', [int(0), gte(1), atom(tagX), out],
           user:pick_b(0, gte(1), tagX, _),                    0).
probe_case(wpick_deep,  'wpick/3',  [int(4), atom(tagY), out],
           user:wpick(4, tagY, O), O).

python3_available :-
    catch(( process_create(path(python3), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

test_wam_python_frameless_ite_level :-
    run_tests(wam_python_frameless_ite_level).

:- begin_tests(wam_python_frameless_ite_level).

% The probe is only meaningful while the shared emitter still produces the
% hazard shape. Assert it directly on the emitted text.
test(sat_clause_has_get_level_without_allocate) :-
    compile_predicate_to_wam_text(user:sat/2, [ite_use_y_level(true)], Text),
    atom_string(Text, S),
    assertion(sub_string(S, _, _, _, "get_level Y")),
    assertion(\+ sub_string(S, _, _, _, "allocate")).

test(matches_swi, [condition(python3_available)]) :-
    once(run_frameless_probe).

:- end_tests(wam_python_frameless_ite_level).

% --- harness ---------------------------------------------------------------

run_frameless_probe :-
    probe_preds(Preds),
    Dir = 'output/test_wam_python_frameless_ite_level',
    ( exists_directory(Dir) -> delete_directory_and_contents(Dir) ; true ),
    write_wam_python_project(Preds, [module_name(fprobe)], Dir),
    findall(Line, case_check_line(Line), Lines),
    atomic_list_concat(Lines, '\n', ChecksBlock),
    harness_prelude(Prelude),
    harness_epilogue(Epilogue),
    atomic_list_concat([Prelude, ChecksBlock, Epilogue], '\n', Src),
    atomic_list_concat([Dir, '/h.py'], HPath),
    setup_call_cleanup(open(HPath, write, S, [encoding(utf8)]),
                       write(S, Src), close(S)),
    format(atom(Cmd), 'cd ~w && python3 h.py 2>&1', [Dir]),
    process_create(path(sh), ['-c', Cmd],
                   [stdout(pipe(Out)), stderr(std), process(Pid)]),
    read_string(Out, _, OutStr), close(Out),
    process_wait(Pid, Status),
    (   Status == exit(0), sub_string(OutStr, _, _, _, "ALL PASS")
    ->  true
    ;   format(user_error,
               "~n[frameless-ITE-level harness output]~n~w~n", [OutStr]),
        throw(wam_python_frameless_ite_level_failed(Status))
    ),
    ( exists_directory(Dir) -> delete_directory_and_contents(Dir) ; true ).

% One `chk(...)` line per case, with SWI supplying the expectation.
case_check_line(Line) :-
    probe_case(Id, Entry, Args, Goal, OutVar),
    maplist(python_arg, Args, ArgStrs),
    atomic_list_concat(ArgStrs, ', ', ArgList),
    (   catch(Goal, _, fail)
    ->  (   OutVar == 0
        ->  Expect = "True", ExpectOut = "None"
        ;   Expect = "True",
            python_literal(OutVar, ExpectOut)
        )
    ;   Expect = "False", ExpectOut = "None"
    ),
    format(atom(Line),
           "chk('~w', run('~w', [~w]), (~w, ~w))",
           [Id, Entry, ArgList, Expect, ExpectOut]).

% The harness reduces an answer to a bare Python str (atom) or int, so the
% expectation SWI computes is spelled the same way.
python_literal(V, S) :- number(V), !, format(atom(S), '~w', [V]).
python_literal(V, S) :- format(atom(S), "'~w'", [V]).

python_arg(int(N), S)  :- format(atom(S), 'Int(~w)', [N]).
python_arg(atom(A), S) :- format(atom(S), "Atom('~w')", [A]).
python_arg(gte(N), S)  :- format(atom(S), "Compound('gte/1', [Int(~w)])", [N]).
python_arg(out, 'None').

harness_prelude(
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
    ok = bool(run_wam(_code, _labels, entry, s))
    if not ok or not outs:
        return (ok, None)
    out = resolve(outs[0], s)
    if isinstance(out, Atom):
        return (ok, out.name)
    if isinstance(out, Int):
        return (ok, out.n)
    return (ok, repr(out))

fails = 0
def chk(name, got, want):
    global fails
    if got != want:
        fails += 1
        print('FAIL', name, 'got', got, 'want', want)
").

harness_epilogue(
"
print('ALL PASS' if fails == 0 else ('FAILURES: %d' % fails))
").
