% test_wam_python_lowered_t4.pl
%
% End-to-end test for the Python T4 lowering — multi-clause, ALL clauses
% (lowering type T4 / multi_clause_n in
% docs/proposals/WAM_LOWERING_TAXONOMY_AND_MATRIX.md).
%
% Python's runtime is a genuine backtracking WAM, so — unlike the imperative
% first-solution T4 targets (lua/rust/…) which commit to the first matching
% clause — Python T4 must preserve intra-query backtracking into later clauses
% or it would diverge from the bytecode interpreter. We use the HYBRID
% realisation: every clause BODY is lowered to a native pred_*_cK function, but
% the try_me_else / retry_me_else / trust_me dispatch scaffold is kept in the
% bytecode so the runtime's proven choice-point machinery drives clause
% dispatch and backtracking. Each clause's body is replaced by a call_lowered
% to its native function.
%
% The decisive checks below are bt_green/1 and bt_blue/1: a conjunctive query
% `color(X), want_green(X)` only succeeds if, after the lowered clause 1 binds
% X=red and want_green(red) fails, the engine backtracks into color's lowered
% clause 2 (green). A first-solution T4 would commit to red and fail. We also
% run a full lowered-vs-interpreter parity battery.
%
% Skipped automatically when python3 is unavailable.

:- use_module(library(lists)).
:- use_module('../src/unifyweaver/targets/wam_target').
:- use_module('../src/unifyweaver/targets/wam_python_target').
:- use_module('../src/unifyweaver/targets/wam_python_lowered_emitter').

:- dynamic user:color/1, user:want_green/1, user:want_blue/1,
           user:bt_green/1, user:bt_blue/1, user:classify/2.

user:color(red).
user:color(green).
user:color(blue).
user:want_green(green).
user:want_blue(blue).
% conjunctive predicates that force backtracking into a non-first clause of color/1
user:bt_green(X) :- user:color(X), user:want_green(X).
user:bt_blue(X)  :- user:color(X), user:want_blue(X).
% mixed: fact, guarded rule (clause 2 runs a builtin), catch-all
user:classify(0, zero).
user:classify(N, pos) :- N > 0.
user:classify(_, neg).

preds([user:color/1, user:want_green/1, user:want_blue/1,
       user:bt_green/1, user:bt_blue/1, user:classify/2]).

python3_available :-
    catch(( process_create(path(python3), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

:- begin_tests(wam_python_lowered_t4, [condition(python3_available)]).

% Every multi-clause predicate where all clauses are clean deterministic bodies
% must be recognised as a T4 candidate with the right clause count.
test(gate_picks_multi_clause_n) :-
    wam_target:compile_predicate_to_wam(color/1, [], Wc),
    assertion(( py_multi_clause_n(Wc, Cs), length(Cs, 3),
                forall(member(C, Cs), last(C, proceed)) )),
    wam_target:compile_predicate_to_wam(classify/2, [], Wcl),
    assertion(( py_multi_clause_n(Wcl, Cs2), length(Cs2, 3) )).

% The emitted code: one native pred_*_cK per clause, plus a registrar that keeps
% the try/retry/trust dispatch scaffold and replaces each clause body with a
% call_lowered to its clause function.
test(emits_per_clause_functions_and_scaffold) :-
    wam_target:compile_predicate_to_wam(color/1, [], W),
    wam_python_target:compile_lowered_wam_predicate_to_python(color/1, W, [emit_mode(lowered)], Code),
    forall(member(F, ["def pred_color_1_c1(state)",
                      "def pred_color_1_c2(state)",
                      "def pred_color_1_c3(state)"]),
           assertion(sub_string(Code, _, _, _, F))),
    % dispatch scaffold retained
    assertion(sub_string(Code, _, _, _, '("try_me_else", "L_color_1_2")')),
    assertion(sub_string(Code, _, _, _, '("retry_me_else", "L_color_1_3")')),
    assertion(sub_string(Code, _, _, _, '("trust_me",)')),
    % every clause body is a call_lowered, none left as a get_constant in bytecode
    assertion(sub_string(Code, _, _, _, '("call_lowered", pred_color_1_c2, 1)')),
    assertion(sub_string(Code, _, _, _, '("call_lowered", pred_color_1_c3, 1)')),
    assertion(\+ sub_string(Code, _, _, _, '("get_constant", Atom("green"), A1)')),
    assertion(\+ sub_string(Code, _, _, _, '("get_constant", Atom("blue"), A1)')).

% Correctness through the interpreter (lowered mode), including the
% backtracking-critical conjunctions bt_green/bt_blue.
test(t4_exec_lowered) :-
    run_battery(lowered, Results),
    assert_expected(Results).

% Parity: the all-clauses-native lowering agrees with the pure bytecode
% interpreter on every query — the defining property of a faithful lowering,
% and the guard that the native clause bodies preserve backtracking.
test(t4_parity_lowered_vs_interpreter) :-
    run_battery(lowered, Lowered),
    run_battery(interpreter, Interp),
    ( Lowered == Interp
    -> true
    ;  format(user_error, "~n[python t4 lowered vs interpreter]~nlowered=~q~ninterp =~q~n",
              [Lowered, Interp]),
       throw(python_t4_parity_failed) ).

:- end_tests(wam_python_lowered_t4).

% Name-Args-Expected battery. Args is a list of py-value strings, or the atom
% `var` for a fresh unbound argument register.
battery([
    'color/1'-['A("red")']-true,
    'color/1'-['A("green")']-true,
    'color/1'-['A("blue")']-true,
    'color/1'-['A("pink")']-false,
    % backtracking-critical: must redo color into clause 2 (green) / clause 3 (blue)
    'bt_green/1'-[var]-true,
    'bt_blue/1'-[var]-true,
    'classify/2'-['I(0)', 'A("zero")']-true,
    'classify/2'-['I(0)', 'A("wrong")']-false,
    'classify/2'-['I(5)', 'A("pos")']-true,
    'classify/2'-['I(5)', 'A("neg")']-true,
    'classify/2'-['I(-2)', 'A("neg")']-true,
    'classify/2'-['I(-2)', 'A("pos")']-false
]).

assert_expected(Results) :-
    battery(B),
    findall(Name-Got-Want,
            ( nth0(I, B, Name-_Args-Want), nth0(I, Results, Got), Got \== Want ),
            Bad),
    ( Bad == [] -> true
    ; format(user_error, "~n[python t4 wrong results] ~q~n", [Bad]),
      throw(python_t4_wrong(Bad)) ).

%% run_battery(+Mode, -Results) — generate a project in Mode, run the battery
%  through the interpreter, unify Results with the list of booleans (in order).
run_battery(Mode, Results) :-
    preds(Preds),
    battery(B),
    format(atom(Dir), 'output/test_wam_python_t4_~w', [Mode]),
    ( exists_directory(Dir) -> delete_directory_and_contents(Dir) ; true ),
    ( Mode == lowered -> Opts = [module_name(t4proj), emit_mode(lowered)]
    ;                    Opts = [module_name(t4proj)] ),
    write_wam_python_project(Preds, Opts, Dir),
    build_harness(B, HarnessSrc),
    atomic_list_concat([Dir, '/t4_harness.py'], HPath),
    setup_call_cleanup(open(HPath, write, S, [encoding(utf8)]),
                       write(S, HarnessSrc), close(S)),
    format(atom(Cmd), 'cd ~w && python3 t4_harness.py 2>&1', [Dir]),
    process_create(path(sh), ['-c', Cmd],
                   [stdout(pipe(Out)), stderr(std), process(Pid)]),
    read_string(Out, _, OutStr), close(Out),
    process_wait(Pid, Status),
    ( Status == exit(0), split_string(OutStr, "\n", " \n", LinesAll),
      ( member(RLine, LinesAll), sub_string(RLine, 0, _, _, "RESULTS ")
      -> sub_string(RLine, 8, _, 0, CSV),
         split_string(CSV, ",", "", Toks),
         maplist(tok_bool, Toks, Results)
      ;  throw(python_t4_no_results(OutStr)) )
    ; format(user_error, "~n[python t4 harness ~w output]~n~w~n", [Mode, OutStr]),
      throw(python_t4_harness_failed(Mode, Status)) ).

tok_bool("1", true).
tok_bool("0", false).

%% build_harness(+Battery, -Src) — python harness that runs each query through
%  the interpreter and prints "RESULTS b,b,b,..." (1=success, 0=fail). Unbound
%  argument positions (`var`) become fresh logic variables.
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
_code, _labels = load_program(build_program())\n\c
def q(entry, vals):\n\c
\s\s\s\ss = WamState()\n\c
\s\s\s\sfor i, v in enumerate(vals, 1):\n\c
\s\s\s\s\s\s\s\sset_reg(s, i, v if v is not None else s.fresh_var())\n\c
\s\s\s\sreturn 1 if run_wam(_code, _labels, entry, s) else 0\n\c
_res = []\n\c
~w\n\c
print('RESULTS ' + ','.join(str(x) for x in _res))\n",
        [Body]).

query_line(Name, Args, Line) :-
    maplist(arg_py, Args, ArgStrs),
    atomic_list_concat(ArgStrs, ', ', ArgList),
    format(atom(Line), '_res.append(q("~w", [~w]))', [Name, ArgList]).

arg_py(var, "None") :- !.
arg_py(V, V).
