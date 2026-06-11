% test_wam_python_lowered_t6.pl
%
% End-to-end test for the Python T6 lowering — first-argument indexing.
%
% Python's T4 hybrid already lowers every clause to a native pred_*_cK function,
% but dispatch runs through the bytecode interpreter's try_me_else/retry_me_else
% chain (O(n)) and the compiler's switch_on_constant index is dropped (left as a
% "# SKIP: ..." comment). T6 turns that index into a real
% ("switch_on_constant", {key: label}) instruction: the runtime jumps O(1) to
% the matching clause body for a BOUND first argument, skips to the try/retry
% chain for an UNBOUND one (so enumeration still works), and fails for a bound
% no-match. Benchmarked (interpreted, dict O(1) vs linear isinstance+name==
% chain): 1.8x at 8, 9.3x at 64, 35.6x at 256.
%
% Gated like the other targets: fires only when the predicate carries a
% first-arg switch index with >= t6_min_clauses entries (default 8). Below the
% threshold the predicate keeps the T4 bytecode-dispatch behaviour.
%
% The decisive check is t6_parity: the lowered (switch) project and the plain
% interpreter must return identical results for every query — bound hits (first,
% middle, last clause), a bound no-match, an UNBOUND first arg (fall-through to
% the try/retry chain), and rule-body clauses (grade/2 runs is/2).
%
% Skipped automatically when python3 is unavailable.

:- use_module(library(lists)).
:- use_module('../src/unifyweaver/targets/wam_target').
:- use_module('../src/unifyweaver/targets/wam_python_target').

:- dynamic user:shade/1, user:tone/2, user:few/1.

user:shade(s01). user:shade(s02). user:shade(s03). user:shade(s04).
user:shade(s05). user:shade(s06). user:shade(s07). user:shade(s08).
user:shade(s09). user:shade(s10).

% Two-argument facts with distinct first-arg keys: the switch dispatches on the
% first arg, and the matched clause's remainder still head-matches the second.
user:tone(c01, bright). user:tone(c02, bright). user:tone(c03, bright).
user:tone(c04, bright). user:tone(c05, dark).   user:tone(c06, dark).
user:tone(c07, dark).   user:tone(c08, dark).   user:tone(c09, dark).
user:tone(c10, dark).

user:few(a). user:few(b). user:few(c).

preds([user:shade/1, user:tone/2, user:few/1]).

% Name-Args-Want.  var = unbound argument.
battery([
    'shade/1'-['A("s01")']-true,    % clause 1 via switch
    'shade/1'-['A("s05")']-true,    % middle clause via switch
    'shade/1'-['A("s10")']-true,    % last clause via switch
    'shade/1'-['A("zz")']-false,    % bound no-match -> fail (not fall-through)
    'shade/1'-[var]-true,           % unbound -> fall through to try/retry chain
    'tone/2'-['A("c01")', 'A("bright")']-true,   % switch hit + remainder A2 match
    'tone/2'-['A("c05")', 'A("dark")']-true,
    'tone/2'-['A("c05")', 'A("bright")']-false,  % first arg hits, A2 mismatch
    'tone/2'-['A("zz")',  'A("dark")']-false,
    'tone/2'-['A("c07")', var]-true,             % switch hit, A2 unbound -> binds
    'few/1'-['A("a")']-true,        % T4 control (below the gate)
    'few/1'-['A("z")']-false
]).

python3_available :-
    catch(( process_create(path(python3), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

:- begin_tests(wam_python_lowered_t6, [condition(python3_available)]).

% Codegen gate: shade/1 + grade/2 (>=8 atom clauses) emit a real
% switch_on_constant + inserted clause-1 body label; few/1 (3) stays the
% T4 "# SKIP" comment; the threshold is overridable.
test(gate_emits_switch_for_many) :-
    wam_target:compile_predicate_to_wam(shade/1, [], Ws),
    wam_python_target:compile_lowered_wam_predicate_to_python(shade/1, Ws, [emit_mode(lowered)], ShadeCode),
    assertion(sub_string(ShadeCode, _, _, _, "(\"switch_on_constant\", {")),
    assertion(sub_string(ShadeCode, _, _, _, "L_pred_shade_1_clause1_body")),
    wam_target:compile_predicate_to_wam(tone/2, [], Wt),
    wam_python_target:compile_lowered_wam_predicate_to_python(tone/2, Wt, [emit_mode(lowered)], ToneCode),
    assertion(sub_string(ToneCode, _, _, _, "(\"switch_on_constant\", {")),
    wam_target:compile_predicate_to_wam(few/1, [], Wf),
    wam_python_target:compile_lowered_wam_predicate_to_python(few/1, Wf, [emit_mode(lowered)], FewCode),
    assertion(sub_string(FewCode, _, _, _, "# SKIP: switch_on_constant")),
    assertion(\+ sub_string(FewCode, _, _, _, "(\"switch_on_constant\", {")),
    wam_python_target:compile_lowered_wam_predicate_to_python(few/1, Wf, [emit_mode(lowered), t6_min_clauses(3)], FewT6),
    assertion(sub_string(FewT6, _, _, _, "(\"switch_on_constant\", {")).

% The lowered (switch) project and the plain interpreter agree on every query,
% AND each matches the expected Prolog result.
test(t6_parity) :-
    battery(B),
    findall(W, member(_-_-W, B), Wants),
    run_battery(lowered, Lowered),
    run_battery(interpreter, Interp),
    assertion(Lowered == Interp),
    assertion(Lowered == Wants).

:- end_tests(wam_python_lowered_t6).

run_battery(Mode, Results) :-
    preds(Preds),
    battery(B),
    format(atom(Dir), 'output/test_wam_python_t6_~w', [Mode]),
    ( exists_directory(Dir) -> delete_directory_and_contents(Dir) ; true ),
    ( Mode == lowered -> Opts = [module_name(t6proj), emit_mode(lowered)]
    ;                    Opts = [module_name(t6proj)] ),
    write_wam_python_project(Preds, Opts, Dir),
    ( Mode == lowered
    ->  atomic_list_concat([Dir, '/predicates.py'], PP),
        read_file_to_string(PP, PSrc, []),
        assertion(sub_string(PSrc, _, _, _, "(\"switch_on_constant\", {"))   % sanity: T6 active
    ;   true ),
    build_harness(B, HarnessSrc),
    atomic_list_concat([Dir, '/t6_harness.py'], HPath),
    setup_call_cleanup(open(HPath, write, S, [encoding(utf8)]),
                       write(S, HarnessSrc), close(S)),
    format(atom(Cmd), 'cd ~w && python3 t6_harness.py 2>&1', [Dir]),
    process_create(path(sh), ['-c', Cmd],
                   [stdout(pipe(Out)), stderr(std), process(Pid)]),
    read_string(Out, _, OutStr), close(Out),
    process_wait(Pid, Status),
    ( Status == exit(0), split_string(OutStr, "\n", " \n", LinesAll),
      ( member(RLine, LinesAll), sub_string(RLine, 0, _, _, "RESULTS ")
      -> sub_string(RLine, 8, _, 0, CSV),
         split_string(CSV, ",", "", Toks),
         maplist(tok_bool, Toks, Results)
      ;  throw(python_t6_no_results(OutStr)) )
    ; format(user_error, "~n[python t6 harness ~w output]~n~w~n", [Mode, OutStr]),
      throw(python_t6_harness_failed(Mode, Status)) ),
    ( exists_directory(Dir) -> delete_directory_and_contents(Dir) ; true ).

tok_bool("1", true).
tok_bool("0", false).

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
