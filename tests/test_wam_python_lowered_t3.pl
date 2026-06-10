% test_wam_python_lowered_t3.pl
%
% End-to-end test for the Python T3 lowering — multi-clause clause-1 fast path
% (lowering type T3 in docs/proposals/WAM_LOWERING_TAXONOMY_AND_MATRIX.md).
% Closes the last ✗ in the T3 column: python was the only target that lowered
% no multi-clause predicate (it rejected any try_me_else; its only multi-clause
% lowering was the T5 if/elif/else clause-chain).
%
% Mechanism (emitter-only — no runtime change): py_multi_clause_1 extracts a
% genuine multi-clause predicate's clause-1 slice and emits it as a lowered
% pred_* function; the registrar keeps the FULL predicate bytecode but replaces
% clause 1's body with a call_lowered to that function, retaining the leading
% try_me_else and clauses 2+ verbatim. At run time the try_me_else pushes the
% choice point onto clause 2, then:
%   - clause-1 SUCCESS  -> call_lowered falls through to proceed (the clause-2
%     choice point is left for backtracking — correct);
%   - clause-1 FAILURE  -> the runtime's call_lowered handler calls fail(),
%     which pops that choice point (restoring trail + registers) and resumes
%     the interpreter at clause 2.
% So clauses 2+ are reached exactly as the bytecode interpreter would reach
% them — this is verified below by a lowered-vs-interpreter parity battery.
%
% Skipped automatically when python3 is unavailable.

:- use_module(library(lists)).
:- use_module('../src/unifyweaver/targets/wam_target').
:- use_module('../src/unifyweaver/targets/wam_python_target').
:- use_module('../src/unifyweaver/targets/wam_python_lowered_emitter').

:- dynamic user:color/1, user:pq/1, user:classify/2.

% facts (first-argument indexed)
user:color(red).
user:color(green).
user:color(blue).
% rules, no first-arg index (head var)
user:pq(X) :- X = a.
user:pq(X) :- X = b.
% mixed: a fact, a rule with a guard goal in clause 2, a catch-all clause 3
user:classify(0, zero).
user:classify(N, pos) :- N > 0.
user:classify(_, neg).

preds([user:color/1, user:pq/1, user:classify/2]).

python3_available :-
    catch(( process_create(path(python3), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

:- begin_tests(wam_python_lowered_t3, [condition(python3_available)]).

% Each multi-clause predicate must be recognised as a T3 clause-1 candidate,
% with clause 1 being a clean deterministic slice.
test(gate_picks_multi_clause_1) :-
    forall(member(_:PI, [user:color/1, user:pq/1, user:classify/2]),
           ( wam_target:compile_predicate_to_wam(PI, [], W),
             ( py_multi_clause_1(W, C1)
             -> assertion(\+ member(try_me_else(_), C1)),
                assertion(last(C1, proceed))
             ;  throw(not_multi_clause_1(PI)) ) )).

% The emitted code: a lowered pred_* function, a call_lowered into clause 1,
% and the clause-2+ bytecode retained (so the interpreter can run them).
test(emits_call_lowered_and_retains_later_clauses) :-
    wam_target:compile_predicate_to_wam(color/1, [], W),
    wam_python_target:compile_lowered_wam_predicate_to_python(color/1, W, [emit_mode(lowered)], Code),
    assertion(sub_string(Code, _, _, _, "def pred_color_1(state)")),
    assertion(sub_string(Code, _, _, _, '("call_lowered", pred_color_1, 1)')),
    assertion(sub_string(Code, _, _, _, "try_me_else")),
    % clause 2 / clause 3 bytecode retained (green & blue still matched in interp)
    assertion(sub_string(Code, _, _, _, '("get_constant", Atom("green"), A1)')),
    assertion(sub_string(Code, _, _, _, '("get_constant", Atom("blue"), A1)')),
    % clause-1 body collapsed: red is matched by the lowered fn, not the bytecode
    assertion(\+ sub_string(Code, _, _, _, '("get_constant", Atom("red")')).

% Correctness through the interpreter (lowered mode): clause-1 fast path,
% clause-2 fallback, clause-3 fallback, and no-match.
test(t3_exec_lowered) :-
    run_battery(lowered, Results),
    assert_expected(Results).

% Parity: the lowered fast path agrees with the pure bytecode interpreter on
% every query (the defining property of a faithful lowering).
test(t3_parity_lowered_vs_interpreter) :-
    run_battery(lowered, Lowered),
    run_battery(interpreter, Interp),
    ( Lowered == Interp
    -> true
    ;  format(user_error, "~n[python t3 lowered vs interpreter]~nlowered=~q~ninterp =~q~n",
              [Lowered, Interp]),
       throw(python_t3_parity_failed) ).

:- end_tests(wam_python_lowered_t3).

% Name-Args-Expected battery. Args is a list of reg(Index, PyValue).
battery([
    'color/1'-[reg(1, 'A("red")')]-true,
    'color/1'-[reg(1, 'A("green")')]-true,
    'color/1'-[reg(1, 'A("blue")')]-true,
    'color/1'-[reg(1, 'A("pink")')]-false,
    'pq/1'-[reg(1, 'A("a")')]-true,
    'pq/1'-[reg(1, 'A("b")')]-true,
    'pq/1'-[reg(1, 'A("c")')]-false,
    'classify/2'-[reg(1, 'I(0)'), reg(2, 'A("zero")')]-true,
    'classify/2'-[reg(1, 'I(0)'), reg(2, 'A("wrong")')]-false,
    'classify/2'-[reg(1, 'I(5)'), reg(2, 'A("pos")')]-true,
    'classify/2'-[reg(1, 'I(5)'), reg(2, 'A("neg")')]-true,
    'classify/2'-[reg(1, 'I(-2)'), reg(2, 'A("neg")')]-true,
    'classify/2'-[reg(1, 'I(-2)'), reg(2, 'A("pos")')]-false
]).

assert_expected(Results) :-
    battery(B),
    findall(Name-Got-Want,
            ( nth0(I, B, _Name-_Args-Want), nth0(I, Results, Got), Got \== Want,
              nth0(I, B, Name-_-_) ),
            Bad),
    ( Bad == [] -> true
    ; format(user_error, "~n[python t3 wrong results] ~q~n", [Bad]),
      throw(python_t3_wrong(Bad)) ).

%% run_battery(+Mode, -Results) — generate a project in Mode, run the battery
%  through the interpreter, unify Results with the list of booleans (one per
%  battery entry, in order).
run_battery(Mode, Results) :-
    preds(Preds),
    battery(B),
    format(atom(Dir), 'output/test_wam_python_t3_~w', [Mode]),
    ( exists_directory(Dir) -> delete_directory_and_contents(Dir) ; true ),
    ( Mode == lowered -> Opts = [module_name(t3proj), emit_mode(lowered)]
    ;                    Opts = [module_name(t3proj)] ),
    write_wam_python_project(Preds, Opts, Dir),
    build_harness(B, HarnessSrc),
    atomic_list_concat([Dir, '/t3_harness.py'], HPath),
    setup_call_cleanup(open(HPath, write, S, [encoding(utf8)]),
                       write(S, HarnessSrc), close(S)),
    format(atom(Cmd), 'cd ~w && python3 t3_harness.py 2>&1', [Dir]),
    process_create(path(sh), ['-c', Cmd],
                   [stdout(pipe(Out)), stderr(std), process(Pid)]),
    read_string(Out, _, OutStr), close(Out),
    process_wait(Pid, Status),
    ( Status == exit(0), split_string(OutStr, "\n", " \n", LinesAll),
      ( member(RLine, LinesAll), sub_string(RLine, 0, _, _, "RESULTS ")
      -> sub_string(RLine, 8, _, 0, CSV),
         split_string(CSV, ",", "", Toks),
         maplist(tok_bool, Toks, Results)
      ;  throw(python_t3_no_results(OutStr)) )
    ; format(user_error, "~n[python t3 harness ~w output]~n~w~n", [Mode, OutStr]),
      throw(python_t3_harness_failed(Mode, Status)) ).

tok_bool("1", true).
tok_bool("0", false).

%% build_harness(+Battery, -Src) — emit a python harness that runs each query
%  through the interpreter and prints "RESULTS b,b,b,..." (1=success,0=fail).
build_harness(Battery, Src) :-
    findall(Line, ( member(Name-Args-_, Battery), query_line(Name, Args, Line) ), Lines),
    atomic_list_concat(Lines, '\n', Body),
    format(string(Src),
"import sys\n\c
sys.path.insert(0, '.')\n\c
from wam_runtime import *\n\c
from predicates import build_program\n\c
A = Atom\n\c
I = Int\n\c
_raw = build_program()\n\c
_code, _labels = load_program(_raw)\n\c
def q(entry, regs):\n\c
\s\s\s\ss = WamState()\n\c
\s\s\s\sfor i, v in regs:\n\c
\s\s\s\s\s\s\s\sset_reg(s, i, v)\n\c
\s\s\s\sreturn 1 if run_wam(_code, _labels, entry, s) else 0\n\c
_res = []\n\c
~w\n\c
print('RESULTS ' + ','.join(str(x) for x in _res))\n",
        [Body]).

query_line(Name, Args, Line) :-
    findall(RegStr, ( member(reg(I, V), Args), format(atom(RegStr), '(~w, ~w)', [I, V]) ), RegStrs),
    atomic_list_concat(RegStrs, ', ', RegList),
    format(atom(Line), '_res.append(q("~w", [~w]))', [Name, RegList]).
