:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
%
% test_wam_r_lowered_ite.pl
%
% End-to-end execution test for WAM-R if-then-else / negation / once
% lowering (emit_mode(functions)). Final backend of the lowered-ITE parity
% sweep; counterpart to the Go / Rust / C++ / Haskell / F# / Clojure / LLVM
% / Elixir / Python / Lua exec tests. Pins:
%
%   * simple ITE         — rite/2;
%   * negation (\+)       — rneg/1 (commit is the !/0 builtin: then = fail/0,
%                          else = true/0, run after a trail rollback);
%   * sequential ITEs    — rseqite/3 (two sibling blocks);
%   * nested ITEs         — rnestite/2 (inner block in the then-arm).
%
% R previously declined any predicate whose lowered body contained the
% soft-cut block's try_me_else / cut_ite / jump / trust_me (not in
% parts_supported/1), so such predicates fell back to the WAM interpreter —
% sound but not lowered. Folding clause 1 through wam_ite_structurer emits
% native R if/else. R's bind always trails, so undoing the trail to the
% pre-condition mark before the else branch restores the condition's
% bindings (WamRuntime$undo_trail_to).
%
% This also guards a prerequisite fix: the lowered `allocate` emit created
% the call frame with a `locals` field instead of the `ys` environment that
% WamRuntime$put_reg/get_reg use for Y registers, so any lowered predicate
% with an environment frame (every ITE here) crashed on the first
% Y-register access. Now it mirrors the interpreter's frame (ys +
% cps_barrier), and deallocate retains the popped frame as shadow_frame.
%
% Generates a lowered R project and calls the per-predicate entry points
% (pred_rite, pred_rneg, ...) asserting the boolean outcome. Skipped unless
% `Rscript` is on PATH.

:- use_module(library(plunit)).
:- use_module(library(filesex)).
:- use_module(library(process)).
:- use_module('../src/unifyweaver/targets/wam_r_target').

:- dynamic user:rite/2.
:- dynamic user:rneg/1.
:- dynamic user:rseqite/3.
:- dynamic user:rnestite/2.

user:rite(X, Y)       :- ( X > 0 -> Y = pos ; Y = nonpos ).
user:rneg(X)          :- \+ X > 0.
user:rseqite(X, Y, Z) :- ( X > 0 -> Y = pos ; Y = nonpos ),
                         ( X > 5 -> Z = big ; Z = small ).
user:rnestite(X, Y)   :- ( X > 0 -> ( X > 10 -> Y = big ; Y = small ) ; Y = neg ).

rscript_available :-
    catch(( process_create(path('Rscript'), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

:- begin_tests(wam_r_lowered_ite, [condition(rscript_available)]).

test(ite_exec_parity) :-
    Dir = 'output/test_wam_r_ite_exec',
    ( exists_directory(Dir) -> delete_directory_and_contents(Dir) ; true ),
    % 1. Generate the lowered R project.
    write_wam_r_project(
        [user:rite/2, user:rneg/1, user:rseqite/3, user:rnestite/2],
        [module_name('iteproj'), emit_mode(functions)], Dir),
    % Sanity: predicates must be lowered as native if/else (not the
    % interpreter fallback), else the test would pass vacuously.
    atomic_list_concat([Dir, '/R/generated_program.R'], ProgPath),
    read_file_to_string(ProgPath, ProgSrc, []),
    forall(member(F, ["lowered_rite_2", "lowered_rneg_1",
                      "lowered_rseqite_3", "lowered_rnestite_2"]),
           assertion(sub_string(ProgSrc, _, _, _, F))),
    assertion(sub_string(ProgSrc, _, _, _, "if-then-else / negation / once")),
    % 2. Write the harness next to the generated module.
    atomic_list_concat([Dir, '/R/harness.R'], HPath),
    harness_source(Src),
    setup_call_cleanup(open(HPath, write, S, [encoding(utf8)]),
                       write(S, Src), close(S)),
    % 3. Run it.
    atomic_list_concat([Dir, '/R'], RDir),
    format(atom(Cmd), 'cd ~w && Rscript harness.R 2>&1', [RDir]),
    process_create(path(sh), ['-c', Cmd],
                   [stdout(pipe(Out)), stderr(std), process(Pid)]),
    read_string(Out, _, OutStr), close(Out),
    process_wait(Pid, Status),
    ( Status == exit(0), sub_string(OutStr, _, _, _, "ALL 15 PASS")
    ->  true
    ;   format(user_error, "~n[r ite test output]~n~w~n", [OutStr]),
        throw(r_ite_test_failed(Status))
    ),
    ( exists_directory(Dir) -> delete_directory_and_contents(Dir) ; true ).

:- end_tests(wam_r_lowered_ite).

% Sources the generated program (its CLI driver is guarded by
% sys.nframe() == 0L, so source/1 skips it) and calls each per-predicate
% entry. Atoms are built through the shared intern table so the ids match
% the constants the lowered functions compare against.
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
chk(\"rite(5,pos)\",     pred_rite(I(5),  A(\"pos\")),    TRUE)
chk(\"rite(5,nonpos)\",  pred_rite(I(5),  A(\"nonpos\")), FALSE)
chk(\"rite(-1,nonpos)\", pred_rite(I(-1), A(\"nonpos\")), TRUE)
chk(\"rite(-1,pos)\",    pred_rite(I(-1), A(\"pos\")),    FALSE)
chk(\"rneg(5)\",  pred_rneg(I(5)),  FALSE)
chk(\"rneg(-1)\", pred_rneg(I(-1)), TRUE)
chk(\"rneg(0)\",  pred_rneg(I(0)),  TRUE)
chk(\"rseqite(10,pos,big)\",      pred_rseqite(I(10), A(\"pos\"),    A(\"big\")),   TRUE)
chk(\"rseqite(10,pos,small)\",    pred_rseqite(I(10), A(\"pos\"),    A(\"small\")), FALSE)
chk(\"rseqite(3,pos,small)\",     pred_rseqite(I(3),  A(\"pos\"),    A(\"small\")), TRUE)
chk(\"rseqite(-1,nonpos,small)\", pred_rseqite(I(-1), A(\"nonpos\"), A(\"small\")), TRUE)
chk(\"rnestite(20,big)\",   pred_rnestite(I(20), A(\"big\")),   TRUE)
chk(\"rnestite(5,small)\",  pred_rnestite(I(5),  A(\"small\")), TRUE)
chk(\"rnestite(-1,neg)\",   pred_rnestite(I(-1), A(\"neg\")),   TRUE)
chk(\"rnestite(20,small)\", pred_rnestite(I(20), A(\"small\")), FALSE)
if (fails == 0L) cat(\"ALL\", total, \"PASS\\n\") else cat(fails, \"FAILURES\\n\")
").
