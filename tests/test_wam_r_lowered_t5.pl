% test_wam_r_lowered_t5.pl
%
% End-to-end execution test for the R T5 lowering — "multi-clause as a
% first-argument dispatch" (lowering type T5 in
% docs/proposals/WAM_LOWERING_TAXONOMY_AND_MATRIX.md). R is line-based, so the
% shared wam_clause_chain front-end is fed a minimal term view of the
% tokenized lines (separators + head get_constant + opaque line(Parts) leaves).
%
% R already lowers ALL clauses (multi_clause_n: a try-all loop with a choice
% point per success). T5 adds an O(1) bound-first-argument fast path: a bound
% A1 dispatches straight to the matching clause via try_clause_(k) — no
% try-all loop, no choice point — while an unbound A1 falls back to the
% multi_clause_n loop, which enumerates every clause. The per-clause bodies
% (try_clause_) are shared between both paths.
%
% Pins (the harness preloads a BOUND first arg, exercising every clause incl.
% the non-first ones — the T5 payoff):
%   * color/1 — fact chain, atom discriminators;
%   * sz/2    — fact chain with a second head match in each remainder;
%   * op/2    — RULE chain (each remainder runs an is/2 builtin).
%
% Skipped automatically when `Rscript` is not on PATH.

:- use_module(library(plunit)).
:- use_module(library(filesex)).
:- use_module('../src/unifyweaver/targets/wam_target').
:- use_module('../src/unifyweaver/targets/wam_r_target').
:- use_module('../src/unifyweaver/targets/wam_r_lowered_emitter').

:- dynamic user:color/1.
:- dynamic user:sz/2.
:- dynamic user:op/2.

user:color(red).
user:color(green).
user:color(blue).

user:sz(small, 1).
user:sz(medium, 2).
user:sz(large, 3).

user:op(add, R) :- R is 1 + 1.
user:op(mul, R) :- R is 2 * 3.
user:op(neg, R) :- R is 0 - 1.

rscript_available :-
    catch(( process_create(path('Rscript'), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

:- begin_tests(wam_r_lowered_t5, [condition(rscript_available)]).

test(gate_picks_clause_chain) :-
    forall(member(PI, [color/1, sz/2, op/2]),
           ( wam_target:compile_predicate_to_wam(PI, [], W),
             wam_r_lowerable(PI, W, Reason),
             assertion(Reason == clause_chain) )).

test(t5_exec_parity) :-
    Dir = 'output/test_wam_r_t5_exec',
    ( exists_directory(Dir) -> delete_directory_and_contents(Dir) ; true ),
    write_wam_r_project(
        [user:color/1, user:sz/2, user:op/2],
        [module_name('t5proj'), emit_mode(functions)], Dir),
    atomic_list_concat([Dir, '/R/generated_program.R'], ProgPath),
    read_file_to_string(ProgPath, ProgSrc, []),
    forall(member(F, ["lowered_color_1", "lowered_sz_2", "lowered_op_2",
                      "T5 first-argument dispatch"]),
           assertion(sub_string(ProgSrc, _, _, _, F))),
    atomic_list_concat([Dir, '/R/harness.R'], HPath),
    harness_source(Src),
    setup_call_cleanup(open(HPath, write, S, [encoding(utf8)]),
                       write(S, Src), close(S)),
    atomic_list_concat([Dir, '/R'], RDir),
    format(atom(Cmd), 'cd ~w && Rscript harness.R 2>&1', [RDir]),
    process_create(path(sh), ['-c', Cmd],
                   [stdout(pipe(Out)), stderr(std), process(Pid)]),
    read_string(Out, _, OutStr), close(Out),
    process_wait(Pid, Status),
    ( Status == exit(0), sub_string(OutStr, _, _, _, "ALL 14 PASS")
    ->  true
    ;   format(user_error, "~n[r t5 test output]~n~w~n", [OutStr]),
        throw(r_t5_test_failed(Status))
    ),
    ( exists_directory(Dir) -> delete_directory_and_contents(Dir) ; true ).

:- end_tests(wam_r_lowered_t5).

harness_source(
"source(\"generated_program.R\")
A <- function(s) Atom(WamRuntime$intern(intern_table, s))
I <- function(n) IntTerm(n)
fails <- 0L; total <- 0L
chk <- function(name, got, want) {
  total <<- total + 1L
  if (!identical(isTRUE(got), want)) {
    fails <<- fails + 1L
    cat(\"FAIL\", name, \"got\", isTRUE(got), \"want\", want, \"\\n\")
  }
}
chk(\"color(red)\",    pred_color(A(\"red\")),    TRUE)
chk(\"color(green)\",  pred_color(A(\"green\")),  TRUE)
chk(\"color(blue)\",   pred_color(A(\"blue\")),   TRUE)
chk(\"color(yellow)\", pred_color(A(\"yellow\")), FALSE)
chk(\"sz(small,1)\",  pred_sz(A(\"small\"),  I(1)), TRUE)
chk(\"sz(medium,2)\", pred_sz(A(\"medium\"), I(2)), TRUE)
chk(\"sz(large,3)\",  pred_sz(A(\"large\"),  I(3)), TRUE)
chk(\"sz(small,2)\",  pred_sz(A(\"small\"),  I(2)), FALSE)
chk(\"sz(big,1)\",    pred_sz(A(\"big\"),    I(1)), FALSE)
chk(\"op(add,2)\",  pred_op(A(\"add\"), I(2)),  TRUE)
chk(\"op(mul,6)\",  pred_op(A(\"mul\"), I(6)),  TRUE)
chk(\"op(neg,-1)\", pred_op(A(\"neg\"), I(-1)), TRUE)
chk(\"op(add,3)\",  pred_op(A(\"add\"), I(3)),  FALSE)
chk(\"op(div,1)\",  pred_op(A(\"div\"), I(1)),  FALSE)
if (fails == 0L) cat(\"ALL\", total, \"PASS\\n\") else cat(fails, \"FAILURES\\n\")
").
