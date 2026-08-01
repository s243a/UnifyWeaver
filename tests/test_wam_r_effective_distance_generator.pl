:- begin_tests(wam_r_effective_distance_generator).

:- use_module('../examples/benchmark/generate_wam_r_effective_distance_benchmark').
:- use_module(library(filesex), [delete_directory_and_contents/1, make_directory_path/1]).
:- use_module(library(readutil), [read_file_to_string/3]).

% ------------------------------------------------------------------
% Structural / contract tests for the R effective-distance generator.
% ------------------------------------------------------------------

test(generate_kernels_on_functions_emits_runner_kernel_and_factsource) :-
    setup_call_cleanup(
        unique_tmp_dir('tmp_wam_r_ed_on', TmpDir),
        (   once(generate('data/benchmark/dev/facts.pl', TmpDir, kernels_on, functions)),
            directory_file_path(TmpDir, 'R/run_effective_distance.R', RunnerPath),
            directory_file_path(TmpDir, 'R/generated_program.R', ProgPath),
            directory_file_path(TmpDir, 'data/category_parent_by_child.tsv', FactPath),
            assertion(exists_file(RunnerPath)),
            assertion(exists_file(ProgPath)),
            assertion(exists_file(FactPath)),
            read_file_to_string(RunnerPath, Runner, []),
            read_file_to_string(ProgPath, Prog, []),
            assertion(once(sub_string(Runner, _, _, _, 'mode=r_wam_effective_distance'))),
            assertion(once(sub_string(Runner, _, _, _, 'root_category'))),
            assertion(once(sub_string(Runner, _, _, _, 'query_ms='))),
            assertion(once(sub_string(Runner, _, _, _, 'total_ms='))),
            assertion(once(sub_string(Runner, _, _, _,
                'proc.time()[["elapsed"]] * 1000'))),
            assertion(\+ sub_string(Runner, _, _, _, 'total_t0')),
            assertion(once(sub_string(Runner, _, _, _, 'KERNEL_MODE <- "kernels_on"'))),
            assertion(once(sub_string(Runner, _, _, _, 'EMIT_MODE'))),
            assertion(once(sub_string(Runner, _, _, _, 'functions'))),
            assertion(once(sub_string(Runner, _, _, _, 'effective_distance_sum_selected'))),
            assertion(once(sub_string(Prog, _, _, _, 'pred_category_ancestor_kernel_ca'))),
            assertion(once(sub_string(Prog, _, _, _, 'source, root, hops, visited'))),
            assertion(once(sub_string(Prog, _, _, _, 'category_ancestor/4'))),
            assertion(once(sub_string(Prog, _, _, _, 'read_facts_grouped_tsv_atoms'))),
            assertion(once(sub_string(Prog, _, _, _, 'category_parent/2'))),
            assertion(once(sub_string(Prog, _, _, _,
                'register_indexed_arg1_parent_lookup(shared_program, "category_parent/2"'))),
            assertion(once(sub_string(Prog, _, _, _, 'arg1_lookups = new.env'))),
            % Runner enumerates via WAM helpers only — no host-side graph walk.
            assertion(\+ sub_string(Runner, _, _, _, 'lookup_parents')),
            assertion(\+ sub_string(Runner, _, _, _, 'category_ancestor_hops')),
            % PERF-R-WAM-STEP: is_lax power/unary fast path is present in runtime.
            directory_file_path(TmpDir, 'R/wam_runtime.R', RtPath),
            read_file_to_string(RtPath, Rt, []),
            assertion(once(sub_string(Rt, _, _, _, 'table$is_lax_fids'))),
            assertion(once(sub_string(Rt, _, _, _, 'PERF-R-WAM-STEP')))
        ),
        cleanup_tmp_dir(TmpDir)).

% Runtime: is_lax **/2 and unary -/1 match eval_arith (power-sum hot path).
test(r_is_lax_power_unary_fast_path_parity) :-
    once((rscript_available -> r_is_lax_power_unary_proof ; true)).

r_is_lax_power_unary_proof :-
    setup_call_cleanup(
        unique_tmp_dir('tmp_wam_r_is_lax', TmpDir),
        (   once(generate('data/benchmark/dev/facts.pl', TmpDir, kernels_on, functions)),
            directory_file_path(TmpDir, 'R', RDir),
            directory_file_path(RDir, 'is_lax_proof.R', Script),
            setup_call_cleanup(
                open(Script, write, S),
                write(S,
'source("generated_program.R")
call_is_lax <- function(program, expr_term, target_term = Unbound("T")) {
  state <- WamRuntime$new_state()
  WamRuntime$promote_regs(state)
  WamRuntime$put_reg(state, 1L, target_term)
  WamRuntime$put_reg(state, 2L, expr_term)
  ok <- WamRuntime$builtin_is_lax(program, state)
  list(ok = ok, value = WamRuntime$deref(state, target_term))
}
plus <- as.integer(WamRuntime$intern(intern_table, "+"))
hat <- as.integer(WamRuntime$intern(intern_table, "^"))
pow <- as.integer(WamRuntime$intern(intern_table, "**"))
minus <- as.integer(WamRuntime$intern(intern_table, "-"))
# Hops+1
e_plus <- StructTerm(plus, list(IntTerm(3L), IntTerm(1L)))
r_plus <- call_is_lax(shared_program, e_plus)
stopifnot(isTRUE(r_plus$ok), identical(r_plus$value$tag, "int"),
          identical(as.integer(r_plus$value$val), 4L))
# unary -N for integer and float inputs
e_neg <- StructTerm(minus, list(IntTerm(5L)))
r_neg <- call_is_lax(shared_program, e_neg)
stopifnot(isTRUE(r_neg$ok), identical(r_neg$value$tag, "int"),
          identical(as.integer(r_neg$value$val), -5L))
e_neg_float <- StructTerm(minus, list(FloatTerm(1.25)))
r_neg_float <- call_is_lax(shared_program, e_neg_float)
stopifnot(isTRUE(r_neg_float$ok), identical(r_neg_float$value$tag, "float"),
          identical(as.numeric(r_neg_float$value$val), -1.25))
# (Hops+1) ** (-N), plus the ^ alias
e_pow <- StructTerm(pow, list(IntTerm(4L), IntTerm(-5L)))
r_pow <- call_is_lax(shared_program, e_pow)
stopifnot(isTRUE(r_pow$ok), identical(r_pow$value$tag, "float"),
          abs(as.numeric(r_pow$value$val) - (4 ^ -5)) < 1e-12)
e_hat <- StructTerm(hat, list(IntTerm(3L), IntTerm(4L)))
r_hat <- call_is_lax(shared_program, e_hat)
stopifnot(isTRUE(r_hat$ok), identical(as.numeric(r_hat$value$val), 81))
# Cross-check the recursive evaluator and bound-target unification.
n <- WamRuntime$eval_arith(WamRuntime$new_state(), e_pow, intern_table)
stopifnot(abs(n - as.numeric(r_pow$value$val)) < 1e-12)
stopifnot(isTRUE(call_is_lax(shared_program, e_neg, IntTerm(-5L))$ok))
stopifnot(!isTRUE(call_is_lax(shared_program, e_neg, IntTerm(-4L))$ok))
# Bad/unbound arithmetic retains the lax silent-failure contract.
e_unbound <- StructTerm(pow, list(IntTerm(2L), Unbound("Exponent")))
stopifnot(!isTRUE(call_is_lax(shared_program, e_unbound)$ok))
# Functor ids belong to one intern table. Exercise a second program whose
# atom ordering deliberately gives ** a different id after the first cache
# is warm; a WamRuntime-global cache would misdispatch this call.
pads <- sprintf("cache_pad_%d", seq_len(as.integer(pow) + 3L))
table2 <- WamRuntime$new_intern_table(c(pads, "-", "+", "^", "**"))
program2 <- list(intern_table = table2)
pow2 <- as.integer(WamRuntime$intern(table2, "**"))
stopifnot(!identical(pow2, pow))
e_pow2 <- StructTerm(pow2, list(IntTerm(2L), IntTerm(-3L)))
r_pow2 <- call_is_lax(program2, e_pow2)
stopifnot(isTRUE(r_pow2$ok), identical(r_pow2$value$tag, "float"),
          abs(as.numeric(r_pow2$value$val) - 0.125) < 1e-12)
stopifnot(!is.null(intern_table$is_lax_fids),
          !is.null(table2$is_lax_fids),
          !identical(intern_table$is_lax_fids$pow, table2$is_lax_fids$pow))
cat("OK is_lax power/unary parity\\n")
'),
                close(S)
            ),
            process_create(path('Rscript'), ['is_lax_proof.R'],
                           [cwd(RDir), stdout(pipe(Out)), stderr(pipe(Err)),
                            process(PID)]),
            read_string(Out, _, OutTxt), close(Out),
            read_string(Err, _, ErrTxt), close(Err),
            process_wait(PID, exit(0)),
            assertion(once(sub_string(OutTxt, _, _, _, 'OK is_lax power/unary parity')))
        ),
        cleanup_tmp_dir(TmpDir)).

test(generate_kernels_off_disables_ca_kernel) :-
    setup_call_cleanup(
        unique_tmp_dir('tmp_wam_r_ed_off', TmpDir),
        (   once(generate('data/benchmark/dev/facts.pl', TmpDir, kernels_off, functions)),
            directory_file_path(TmpDir, 'R/generated_program.R', ProgPath),
            directory_file_path(TmpDir, 'R/run_effective_distance.R', RunnerPath),
            read_file_to_string(ProgPath, Prog, []),
            read_file_to_string(RunnerPath, Runner, []),
            assertion(once(sub_string(Runner, _, _, _, 'KERNEL_MODE <- "kernels_off"'))),
            assertion(\+ sub_string(Prog, _, _, _, 'pred_category_ancestor_kernel_ca')),
            assertion(once(sub_string(Prog, _, _, _, 'read_facts_grouped_tsv_atoms')))
        ),
        cleanup_tmp_dir(TmpDir)).

test(generate_interpreter_mode_runner_branch) :-
    setup_call_cleanup(
        unique_tmp_dir('tmp_wam_r_ed_interp', TmpDir),
        (   once(generate('data/benchmark/dev/facts.pl', TmpDir, kernels_on, interpreter)),
            directory_file_path(TmpDir, 'R/run_effective_distance.R', RunnerPath),
            read_file_to_string(RunnerPath, Runner, []),
            assertion(once(sub_string(Runner, _, _, _, 'interpreter'))),
            assertion(\+ sub_string(Runner, _, _, _,
                'lowered_category_ancestor_effective_distance_sum_selected_3'))
        ),
        cleanup_tmp_dir(TmpDir)).

% ------------------------------------------------------------------
% Rscript e2e: full article x root enumeration on dev facts.
% Skips cleanly when Rscript is unavailable. No performance thresholds.
% ------------------------------------------------------------------

test(r_effective_distance_e2e_functions_dev) :-
    once((
        rscript_available
    ->  e2e_r_effective_distance(functions)
    ;   true
    )).

test(r_effective_distance_e2e_interpreter_dev) :-
    once((
        rscript_available
    ->  e2e_r_effective_distance(interpreter)
    ;   true
    )).

e2e_r_effective_distance(EmitMode) :-
    setup_call_cleanup(
        unique_tmp_dir('tmp_wam_r_ed_e2e', TmpDir),
        (   once(generate('data/benchmark/dev/facts.pl', TmpDir, kernels_on, EmitMode)),
            directory_file_path(TmpDir, 'R', RDir),
            absolute_file_name('data/benchmark/dev', DataDir, [file_type(directory)]),
            run_r_ed_runner(RDir, DataDir, Out, Err),
            assertion(once(sub_string(Err, _, _, _, 'mode=r_wam_effective_distance'))),
            assertion(once(sub_string(Err, _, _, _, 'query_ms='))),
            assertion(once(sub_string(Err, _, _, _, 'total_ms='))),
            assertion(once(sub_string(Err, _, _, _, 'row_count='))),
            metric_value(Err, "query_ms", QueryMs),
            metric_value(Err, "total_ms", TotalMs),
            assertion(QueryMs >= 0),
            assertion(TotalMs >= QueryMs),
            assertion(once(sub_string(Out, _, _, _, 'root_category'))),
            assertion(once(sub_string(Out, _, _, _, 'Special_relativity'))),
            assertion(once(sub_string(Out, _, _, _, '0.993073'))),
            assertion(once(sub_string(Out, _, _, _, 'Bose-Einstein_statistics'))),
            assertion(once(sub_string(Out, _, _, _, '0.999179'))),
            count_tsv_data_rows(Out, N),
            assertion(N =:= 19)
        ),
        cleanup_tmp_dir(TmpDir)).

run_r_ed_runner(RDir, DataDir, Out, Err) :-
    process_create(path('Rscript'),
                   ['run_effective_distance.R', DataDir, '1'],
                   [ cwd(RDir),
                     stdout(pipe(OutStream)),
                     stderr(pipe(ErrStream)),
                     process(PID)
                   ]),
    read_string(OutStream, _, Out), close(OutStream),
    read_string(ErrStream, _, Err), close(ErrStream),
    process_wait(PID, exit(0)).

count_tsv_data_rows(Out, N) :-
    split_string(Out, "\n", "\r\n", Lines0),
    exclude(=(""), Lines0, Lines),
    (   Lines = [_Header|Rows]
    ->  length(Rows, N)
    ;   N = 0
    ).

metric_value(Text, Key, Value) :-
    split_string(Text, "\n", "\r\n", Lines),
    string_concat(Key, "=", Prefix),
    member(Line, Lines),
    string_concat(Prefix, Raw, Line),
    number_string(Value, Raw),
    !.

rscript_available :-
    catch((
        process_create(path('Rscript'), ['--version'],
                       [ stdout(null), stderr(null), process(PID) ]),
        process_wait(PID, exit(0))
    ), _, fail).

unique_tmp_dir(Prefix, TmpDir) :-
    tmp_file(Prefix, TmpDir),
    catch(delete_directory_and_contents(TmpDir), _, true),
    make_directory_path(TmpDir).

cleanup_tmp_dir(TmpDir) :-
    catch(delete_directory_and_contents(TmpDir), _, true).

:- end_tests(wam_r_effective_distance_generator).
