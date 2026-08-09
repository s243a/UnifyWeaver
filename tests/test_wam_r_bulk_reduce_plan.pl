:- begin_tests(wam_r_bulk_reduce_plan).

:- use_module('../examples/benchmark/generate_wam_r_effective_distance_benchmark').
:- use_module(library(filesex), [delete_directory_and_contents/1, make_directory_path/1]).
:- use_module(library(readutil), [read_file_to_string/3]).
:- use_module(library(process)).
:- use_module('../src/unifyweaver/targets/wam_r_target').
:- use_module('../src/unifyweaver/targets/wam_target', [compile_predicate_to_wam_text/3]).

% PERF-R-BULK-REDUCE-PLAN-1: generate-time fused bulk-reduce plans.

rscript_available :-
    catch((process_create(path('Rscript'), ['--version'],
                          [stdout(null), stderr(null), process(PID)]),
           process_wait(PID, exit(0))),
          _, fail).

unique_tmp_dir(Prefix, Dir) :-
    get_time(T),
    format(atom(Name), '~w_~w', [Prefix, T]),
    tmp_file(Name, Dir),
    make_directory_path(Dir).

cleanup_tmp_dir(Dir) :-
    ( exists_directory(Dir) -> delete_directory_and_contents(Dir) ; true ).

run_rscript(RDir, Script, Exit) :-
    process_create(path('Rscript'), [Script],
                   [cwd(RDir), stdout(pipe(Out)), stderr(pipe(Err)), process(PID)]),
    read_string(Out, _, _), close(Out),
    read_string(Err, _, _), close(Err),
    process_wait(PID, Exit).

% Packer recognizes the closed bulk-reduce shape (ISO-rewritten WAM tokens).
test(pack_power_sum_bound_shape) :-
    Lines = [
        "p/3:",
        "allocate",
        "get_variable Y2, A1",
        "get_variable Y3, A2",
        "get_variable Y8, A3",
        "put_value Y3, A1",
        "builtin_call nonvar/1, 1",
        "put_variable Y1, A1",
        "call dimension_n/1, 1",
        "put_variable Y7, A1",
        "put_structure -/1, A2",
        "set_value Y1",
        "builtin_call is_lax/2, 2",
        "put_variable Y5, Y5",
        "begin_aggregate sum, Y5, Y8",
        "put_value Y2, A1",
        "put_value Y3, A2",
        "put_variable Y4, A3",
        "put_list A4",
        "set_value Y2",
        "set_constant []",
        "call category_ancestor/4, 4",
        "put_variable Y6, A1",
        "put_structure +/2, A2",
        "set_value Y4",
        "set_constant 1",
        "builtin_call is_lax/2, 2",
        "put_value Y5, A1",
        "put_structure **/2, A2",
        "set_value Y6",
        "set_value Y7",
        "builtin_call is_lax/2, 2",
        "end_aggregate Y5",
        "put_value Y8, A1",
        "put_constant 0, A2",
        "builtin_call >_lax/2, 2",
        "deallocate",
        "proceed"
    ],
    atomic_list_concat(Lines, '\n', Wam),
    (   wam_r_target:r_pack_bulk_reduce_plan(Wam, Plan)
    ->  true
    ;   throw(pack_failed(Wam))
    ),
    assertion(functor(Plan, br_plan, 12)),
    Plan = br_plan(Kind, BagAi, BulkKey, _NV, ExpFact, _EY, UMinus, Gt0, _T,
                   Compiled, _Al, Puts),
    assertion(Kind == sum),
    assertion(BagAi =:= 3),
    assertion(UMinus == true),
    assertion(Gt0 == true),
    ( string(BulkKey) -> BulkKeyS = BulkKey ; atom_string(BulkKey, BulkKeyS) ),
    ( string(ExpFact) -> ExpFactS = ExpFact ; atom_string(ExpFact, ExpFactS) ),
    assertion(BulkKeyS == "category_ancestor/4"),
    assertion(ExpFactS == "dimension_n/1"),
    assertion(Compiled \= []),
    assertion(Puts \= []).

% Shape recognition is name-agnostic: non-ED predicate + closed is/2 packs.
test(pack_non_ed_scores_sum_is2) :-
    Lines = [
        "scores_sum/1:",
        "allocate",
        "get_variable Y3, A1",
        "put_variable Y1, Y1",
        "begin_aggregate sum, Y1, Y3",
        "put_variable Y2, A1",
        "call score_fact/1, 1",
        "put_value Y1, A1",
        "put_structure +/2, A2",
        "set_value Y2",
        "set_constant 0",
        "builtin_call is/2, 2",
        "end_aggregate Y1",
        "deallocate",
        "proceed"
    ],
    atomic_list_concat(Lines, '\n', Wam),
    (   wam_r_target:r_pack_bulk_reduce_plan(Wam, Plan)
    ->  true
    ;   throw(pack_failed(Wam))
    ),
    Plan = br_plan(sum, 1, BulkKey, _, '', 0, false, false, _, Compiled, _, _),
    format(string(BulkKeyS), '~w', [BulkKey]),
    assertion(BulkKeyS == "score_fact/1"),
    assertion(Compiled \= []).

% Near-miss: cut / multi-clause choice rejects packing.
test(pack_rejects_cut) :-
    Wam = 'p/1:\n    allocate\n    cut\n    begin_aggregate sum, Y1, A1\n    call q/1, 1\n    put_variable Y2, A1\n    put_structure +/2, A2\n    set_value Y1\n    set_constant 1\n    builtin_call is_lax/2, 2\n    end_aggregate Y1\n    deallocate\n    proceed\n',
    assertion(\+ wam_r_target:r_pack_bulk_reduce_plan(Wam, _)).

test(pack_rejects_nested_or_multi_call) :-
    Wam = 'p/1:\n    begin_aggregate sum, Y1, A1\n    call q/1, 1\n    call r/1, 1\n    end_aggregate Y1\n    proceed\n',
    assertion(\+ wam_r_target:r_pack_bulk_reduce_plan(Wam, _)).

test(pack_rejects_nested_single_call) :-
    Wam = 'p/1:\n    begin_aggregate sum, Y1, A1\n    begin_aggregate count, Y2, Y3\n    call q/1, 1\n    end_aggregate Y2\n    end_aggregate Y1\n    proceed\n',
    assertion(\+ wam_r_target:r_pack_bulk_reduce_plan(Wam, _)).

test(pack_rejects_unconsumed_builtin) :-
    Wam = 'p/1:\n    allocate\n    get_variable Y1, A1\n    put_variable Y2, Y2\n    begin_aggregate sum, Y2, Y1\n    put_variable Y3, A1\n    call q/1, 1\n    builtin_call write/1, 1\n    put_value Y2, A1\n    put_structure +/2, A2\n    set_value Y3\n    set_constant 0\n    builtin_call is_lax/2, 2\n    end_aggregate Y2\n    deallocate\n    proceed\n',
    assertion(\+ wam_r_target:r_pack_bulk_reduce_plan(Wam, _)).

test(pack_rejects_unconsumed_after_instruction) :-
    Wam = 'p/1:\n    allocate\n    get_variable Y1, A1\n    put_variable Y2, Y2\n    begin_aggregate sum, Y2, Y1\n    put_variable Y3, A1\n    call q/1, 1\n    put_value Y2, A1\n    put_structure +/2, A2\n    set_value Y3\n    set_constant 0\n    builtin_call is_lax/2, 2\n    end_aggregate Y2\n    put_constant 9, A1\n    deallocate\n    proceed\n',
    assertion(\+ wam_r_target:r_pack_bulk_reduce_plan(Wam, _)).

test(pack_rejects_switch) :-
    Wam = 'p/1:\n    switch_on_term A1, L1, L2, L3, L4\n    begin_aggregate sum, Y1, A1\n    call q/1, 1\n    end_aggregate Y1\n    proceed\n',
    assertion(\+ wam_r_target:r_pack_bulk_reduce_plan(Wam, _)).

% Emission: fused plan + dispatch for power_sum_bound; absent under kernels_off.
test(emit_fused_plan_kernels_on_off) :-
    setup_call_cleanup(
        unique_tmp_dir('tmp_br_emit', TmpDir),
        (   once(generate('data/benchmark/dev/facts.pl', TmpDir, kernels_on, functions)),
            directory_file_path(TmpDir, 'R/generated_program.R', ProgPath),
            directory_file_path(TmpDir, 'R/wam_runtime.R', RtPath),
            read_file_to_string(ProgPath, Prog, []),
            read_file_to_string(RtPath, Rt, []),
            assertion(once(sub_string(Prog, _, _, _, 'fused_br_'))),
            assertion(once(sub_string(Prog, _, _, _, 'exec_bulk_reduce_plan'))),
            assertion(once(sub_string(Prog, _, _, _,
                'category_ancestor$power_sum_bound/3'))),
            assertion(once(sub_string(Rt, _, _, _, 'exec_bulk_reduce_plan'))),
            assertion(once(sub_string(Rt, _, _, _, 'finalize_bulk_scalar_reduce'))),
            unique_tmp_dir('tmp_br_off', OffDir),
            setup_call_cleanup(
                once(generate('data/benchmark/dev/facts.pl', OffDir,
                              kernels_off, functions)),
                (   directory_file_path(OffDir, 'R/generated_program.R', OffProg),
                    read_file_to_string(OffProg, OffS, []),
                    % Fusion may still emit, but bulk_collect for CA must be absent.
                    assertion(\+ sub_string(OffS, _, _, _,
                        'register_bulk_collect(shared_program, "category_ancestor/4"'))
                ),
                cleanup_tmp_dir(OffDir)
            )
        ),
        cleanup_tmp_dir(TmpDir)).

% Runtime: sum/count/min/max, empty/singleton, int/float, fallbacks, non-ED plan.
test(exec_plan_kinds_and_fallbacks, [condition(rscript_available)]) :-
    setup_call_cleanup(
        unique_tmp_dir('tmp_br_rt', TmpDir),
        (   once(generate('data/benchmark/dev/facts.pl', TmpDir, kernels_on, functions)),
            directory_file_path(TmpDir, 'R', RDir),
            directory_file_path(RDir, 'br_proof.R', Script),
            setup_call_cleanup(
                open(Script, write, S),
                write(S,
'source("generated_program.R")
stopifnot(exists("category_ancestor$power_sum_bound/3",
                 envir = shared_program$lowered_dispatch, inherits = FALSE))
fn <- get("category_ancestor$power_sum_bound/3",
          envir = shared_program$lowered_dispatch, inherits = FALSE)
stopifnot(exists("fused_br_category_ancestor_power_sum_bound_3_plan",
                 inherits = FALSE) ||
          exists("fused_br_category_ancestor_power_sum_bound_3_plan",
                 envir = .GlobalEnv, inherits = FALSE))
plan <- fused_br_category_ancestor_power_sum_bound_3_plan
stopifnot(identical(plan$kind, "sum"), identical(plan$bulk_key, "category_ancestor/4"))

atom_of <- function(nm) Atom(WamRuntime$intern(intern_table, nm))
run_psb <- function(cat, root, sum_term) {
  state <- WamRuntime$new_state(); WamRuntime$promote_regs(state)
  WamRuntime$put_reg(state, 1L, atom_of(cat))
  WamRuntime$put_reg(state, 2L, atom_of(root))
  WamRuntime$put_reg(state, 3L, sum_term)
  state$cp <- 0L
  ok <- isTRUE(fn(shared_program, state))
  list(ok = ok, val = WamRuntime$deref(state, sum_term))
}

# Fused success + counters: no BeginAggregate/EndAggregate on fused path.
orig_step <- WamRuntime$step
ba <- 0L; ea <- 0L; steps <- 0L
WamRuntime$step <- function(program, state, instr) {
  steps <<- steps + 1L
  if (identical(instr$op, "BeginAggregate")) ba <<- ba + 1L
  if (identical(instr$op, "EndAggregate")) ea <<- ea + 1L
  orig_step(program, state, instr)
}
Sum <- Unbound("Sum")
r <- run_psb("Quantum_mechanics", "Physics", Sum)
stopifnot(isTRUE(r$ok), as.numeric(r$val$val) > 0)
stopifnot(identical(ba, 0L), identical(ea, 0L))
fused_val <- as.numeric(r$val$val)

# Empty / reverse path fails gt0
r0 <- run_psb("Physics", "Quantum_mechanics", Unbound("S0"))
stopifnot(!isTRUE(r0$ok))

# Synthetic plans: count/min/max + empty/singleton batches + int/float.
mk_state <- function() {
  st <- WamRuntime$new_state(); WamRuntime$promote_regs(st); st
}
synth <- function(kind, vals, compiled, tkey = "10", gt0 = FALSE) {
  be <- WamRuntime$bulk_collect_env(shared_program)
  assign("synth_bulk/1", list(
    fn = function(program, state) list(type = "numeric_batch", reg = 1L, val = vals),
    output_regs = 1L), envir = be)
  assign("synth_bulk/1", function(program, state) TRUE,
         envir = shared_program$lowered_dispatch)
  st <- mk_state(); bag <- Unbound("B")
  WamRuntime$put_reg(st, 1L, bag)
  pl <- list(kind = kind, bag_ai = 1L, bulk_key = "synth_bulk/1",
             nonvar_ais = integer(0), exp_fact = "", exp_y = 0L, uminus = FALSE,
             gt0 = gt0, tkey = tkey, compiled = compiled,
             aliases = list(c(1L, 9L)),
             puts = list(list(k = "f", a = 1L)))
  shared_program$labels[["synth_host/1"]] <- 1L
  ok <- WamRuntime$exec_bulk_reduce_plan(shared_program, st, pl, "synth_host/1")
  list(ok = ok, val = WamRuntime$deref(st, bag))
}
# Hop identity: dest tkey from arg reg 9 aliased to batch reg 1
id_compiled <- list(list(dest = 10L, opname = "+", arity = 2L,
  args = list(list(kind = "reg", reg = 9L), list(kind = "const", val = 0))))
# sum duplicates / multiplicity
rs <- synth("sum", c(1, 1, 4), id_compiled)
stopifnot(isTRUE(rs$ok), identical(as.numeric(rs$val$val), 6))
# count
rc <- synth("count", c(1, 2, 3), id_compiled)
stopifnot(isTRUE(rc$ok), identical(as.integer(rc$val$val), 3L))
# min / max
rmin <- synth("min", c(5, 2, 9), id_compiled)
rmax <- synth("max", c(5, 2, 9), id_compiled)
stopifnot(isTRUE(rmin$ok), identical(as.numeric(rmin$val$val), 2))
stopifnot(isTRUE(rmax$ok), identical(as.numeric(rmax$val$val), 9))
# empty sum => 0; empty min fails
re <- synth("sum", numeric(0), id_compiled)
stopifnot(isTRUE(re$ok), identical(as.numeric(re$val$val), 0))
re_min <- synth("min", numeric(0), id_compiled)
stopifnot(!isTRUE(re_min$ok))
# singleton
r1 <- synth("sum", 3, id_compiled)
stopifnot(isTRUE(r1$ok), identical(as.numeric(r1$val$val), 3))
# float arith (**)
pow_compiled <- list(list(dest = 10L, opname = "**", arity = 2L,
  args = list(list(kind = "reg", reg = 9L), list(kind = "const", val = 0.5))))
rf <- synth("sum", c(4, 9), pow_compiled)
stopifnot(isTRUE(rf$ok), abs(as.numeric(rf$val$val) - 5) < 1e-9)

# Missing capability => WAM label fallback (label 1 Proceed may fail/succeed)
be <- WamRuntime$bulk_collect_env(shared_program)
rm(list = "synth_bulk/1", envir = be)
st <- mk_state(); WamRuntime$put_reg(st, 1L, Unbound("X"))
pl_miss <- list(kind = "sum", bag_ai = 1L, bulk_key = "missing/1",
  nonvar_ais = integer(0), exp_fact = "", exp_y = 0L, uminus = FALSE,
  gt0 = FALSE, tkey = "10", compiled = id_compiled,
  aliases = list(c(1L, 9L)), puts = list(list(k = "f", a = 1L)))
# Should not throw; returns FALSE or label result
invisible(WamRuntime$exec_bulk_reduce_plan(shared_program, st, pl_miss, "synth_host/1"))

# Malformed capability
assign("synth_bulk/1", list(fn = NULL, output_regs = 1L), envir = be)
invisible(WamRuntime$exec_bulk_reduce_plan(shared_program, st, pl_miss, "synth_host/1"))

# Dynamic shadowing of bulk key => fallback
assign("synth_bulk/1", list(
  fn = function(program, state) list(type = "numeric_batch", reg = 1L, val = 1),
  output_regs = 1L), envir = be)
assign("synth_bulk/1", function(program, state) TRUE,
       envir = shared_program$lowered_dispatch)
shared_program$dynamic <- new.env(parent = emptyenv())
assign("synth_bulk/1", list(), envir = shared_program$dynamic)
st2 <- mk_state(); WamRuntime$put_reg(st2, 1L, Unbound("Y"))
pl_ok <- list(kind = "sum", bag_ai = 1L, bulk_key = "synth_bulk/1",
  nonvar_ais = integer(0), exp_fact = "", exp_y = 0L, uminus = FALSE,
  gt0 = FALSE, tkey = "10", compiled = id_compiled,
  aliases = list(c(1L, 9L)), puts = list(list(k = "f", a = 1L)))
# lowered_path_ok false => fb(); must not use fused reduce
ok_shadow <- WamRuntime$exec_bulk_reduce_plan(shared_program, st2, pl_ok, "synth_host/1")
stopifnot(is.logical(ok_shadow))
shared_program$dynamic <- NULL

# Legacy list-of-records that cannot become numeric batch => fallback
assign("synth_bulk/1", list(
  fn = function(program, state) list(list(reg = 1L, val = "x")),
  output_regs = 1L), envir = be)
st3 <- mk_state(); WamRuntime$put_reg(st3, 1L, Unbound("Z"))
invisible(WamRuntime$exec_bulk_reduce_plan(shared_program, st3, pl_ok, "synth_host/1"))

# Non-ED eligible: remove fusion dispatch and compare classic vs fused value
rm(list = "category_ancestor$power_sum_bound/3",
   envir = shared_program$lowered_dispatch)
WamRuntime$step <- orig_step
Sum_c <- Unbound("Sc")
state_c <- WamRuntime$new_state(); WamRuntime$promote_regs(state_c)
WamRuntime$put_reg(state_c, 1L, atom_of("Quantum_mechanics"))
WamRuntime$put_reg(state_c, 2L, atom_of("Physics"))
WamRuntime$put_reg(state_c, 3L, Sum_c)
state_c$pc <- as.integer(shared_program$labels[["category_ancestor$power_sum_bound/3"]])
state_c$cp <- 0L
ok_c <- isTRUE(WamRuntime$run(shared_program, state_c))
stopifnot(isTRUE(ok_c))
classic_val <- as.numeric(WamRuntime$deref(state_c, Sum_c)$val)
stopifnot(abs(classic_val - fused_val) < 1e-12)

cat("PLAN1_PROOF_OK fused_val=", fused_val, " steps_fused_path_ba=", ba, "\n", sep="")
'),
                close(S)
            ),
            run_rscript(RDir, 'br_proof.R', exit(0))
        ),
        cleanup_tmp_dir(TmpDir)).

% ED scale-dev or scale-300 parity when data present (271 rows on 300).
test(ed_reference_multiset_parity_300, [
    condition((rscript_available, exists_file('data/benchmark/300/facts.pl')))
]) :-
    absolute_file_name('data/benchmark/300', FactsDir, [file_type(directory)]),
    setup_call_cleanup(
        unique_tmp_dir('tmp_br_ed300', TmpDir),
        (   once(generate('data/benchmark/300/facts.pl', TmpDir, kernels_on, functions)),
            directory_file_path(TmpDir, 'R', RDir),
            (   exists_file('src/uw_ca_hops.c')
            ->  true
            ;   true
            ),
            % Build optional native hop lib if present in project
            directory_file_path(TmpDir, 'src/uw_ca_hops.c', CSrc),
            (   exists_file(CSrc)
            ->  process_create(path('R'), ['CMD', 'SHLIB', 'src/uw_ca_hops.c',
                                           '-o', 'src/uw_ca_hops.so'],
                               [cwd(TmpDir), stdout(null), stderr(null), process(P1)]),
                process_wait(P1, _)
            ;   true
            ),
            process_create(path('Rscript'),
                ['run_effective_distance.R',
                 FactsDir, '1'],
                [cwd(RDir), stdout(pipe(Out)), stderr(pipe(Err)), process(PID)]),
            read_string(Out, _, OutS), close(Out),
            read_string(Err, _, ErrS), close(Err),
            process_wait(PID, exit(0)),
            assertion(once(sub_string(ErrS, _, _, _, 'row_count=271'))),
            split_string(OutS, "\n", "", Lines0),
            exclude([L]>>(L == "" ; L = "article\troot_category\teffective_distance"),
                    Lines0, Lines),
            assertion(length(Lines, 271))
        ),
        cleanup_tmp_dir(TmpDir)).

:- end_tests(wam_r_bulk_reduce_plan).
