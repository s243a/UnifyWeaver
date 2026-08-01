:- begin_tests(wam_r_aggregate_lower).

:- use_module('../examples/benchmark/generate_wam_r_effective_distance_benchmark').
:- use_module(library(filesex), [delete_directory_and_contents/1, make_directory_path/1]).
:- use_module(library(readutil), [read_file_to_string/3]).
:- use_module(library(process)).

% PERF-R-AGGREGATE-LOWER: capability-gated scalar aggregate region tests.

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
    (   exists_directory(Dir)
    ->  delete_directory_and_contents(Dir)
    ;   true
    ).

test(aggregate_lower_emits_bulk_collect_for_ca_kernel) :-
    setup_call_cleanup(
        unique_tmp_dir('tmp_wam_r_agg_emit', TmpDir),
        (   once(generate('data/benchmark/dev/facts.pl', TmpDir, kernels_on, functions)),
            directory_file_path(TmpDir, 'R/generated_program.R', ProgPath),
            directory_file_path(TmpDir, 'R/wam_runtime.R', RtPath),
            read_file_to_string(ProgPath, Prog, []),
            read_file_to_string(RtPath, Rt, []),
            assertion(once(sub_string(Prog, _, _, _, 'register_bulk_collect'))),
            assertion(once(sub_string(Prog, _, _, _,
                'category_ancestor_bulk_collect'))),
            assertion(once(sub_string(Rt, _, _, _,
                'try_scalar_aggregate_fastpath'))),
            assertion(once(sub_string(Rt, _, _, _,
                'compile_scalar_aggregate_arith'))),
            assertion(once(sub_string(Rt, _, _, _,
                'PERF-R-AGGREGATE-LOWER'))),
            assertion(once(sub_string(Rt, _, _, _,
                'PERF-R-AGG-BATCH'))),
            assertion(once(sub_string(Rt, _, _, _,
                'as_bulk_numeric_batch'))),
            assertion(once(sub_string(Rt, _, _, _,
                'numeric_batch'))),
            % kernels_off must not register bulk_collect for CA
            unique_tmp_dir('tmp_wam_r_agg_off', OffDir),
            setup_call_cleanup(
                once(generate('data/benchmark/dev/facts.pl', OffDir,
                              kernels_off, functions)),
                (   directory_file_path(OffDir, 'R/generated_program.R', OffProg),
                    read_file_to_string(OffProg, OffS, []),
                    assertion(\+ sub_string(OffS, _, _, _,
                        'register_bulk_collect(shared_program, "category_ancestor/4"'))
                ),
                cleanup_tmp_dir(OffDir)
            )
        ),
        cleanup_tmp_dir(TmpDir)).

test(aggregate_lower_runtime_semantics, [condition(rscript_available)]) :-
    setup_call_cleanup(
        unique_tmp_dir('tmp_wam_r_agg_rt', TmpDir),
        (   once(generate('data/benchmark/dev/facts.pl', TmpDir, kernels_on, functions)),
            directory_file_path(TmpDir, 'R', RDir),
            directory_file_path(RDir, 'agg_proof.R', Script),
            setup_call_cleanup(
                open(Script, write, S),
                write(S,
'source("generated_program.R")
stopifnot(exists("__bulk_collect__", envir = shared_program$lowered_dispatch, inherits = FALSE))
be <- get("__bulk_collect__", envir = shared_program$lowered_dispatch, inherits = FALSE)
stopifnot(exists("category_ancestor/4", envir = be, inherits = FALSE))

atom_of <- function(nm) Atom(WamRuntime$intern(intern_table, nm))
run_bound <- function(cat, root, sum_term) {
  state <- WamRuntime$new_state(); WamRuntime$promote_regs(state)
  WamRuntime$put_reg(state, 1L, atom_of(cat))
  WamRuntime$put_reg(state, 2L, atom_of(root))
  WamRuntime$put_reg(state, 3L, sum_term)
  state$cp <- 0L
  key <- "category_ancestor$power_sum_bound/3"
  state$pc <- as.integer(shared_program$labels[[key]])
  ok <- isTRUE(tryCatch(WamRuntime$run(shared_program, state),
                        prolog_throw = function(cond) FALSE))
  list(ok = ok, state = state, val = WamRuntime$deref(state, sum_term))
}

# Count EndAggregate during a successful bound query (fast path => 0).
orig_step <- WamRuntime$step
ea <- 0L; ba <- 0L
WamRuntime$step <- function(program, state, instr) {
  if (identical(instr$op, "BeginAggregate")) ba <<- ba + 1L
  if (identical(instr$op, "EndAggregate")) ea <<- ea + 1L
  orig_step(program, state, instr)
}
Sum <- Unbound("Sum")
r <- run_bound("Quantum_mechanics", "Physics", Sum)
stopifnot(isTRUE(r$ok), !is.null(r$val$tag), as.numeric(r$val$val) > 0)
stopifnot(identical(ba, 1L), identical(ea, 0L))

# Failure: wrong bound result
rfail <- run_bound("Quantum_mechanics", "Physics", IntTerm(-1L))
stopifnot(!isTRUE(rfail$ok))

# Empty-ish / no-path: category with no path to root fails C>0 after sum 0
Sum0 <- Unbound("S0")
r0 <- run_bound("Physics", "Quantum_mechanics", Sum0)
# may fail the trailing >/2; either fail or non-positive
if (isTRUE(r0$ok)) stopifnot(!(as.numeric(r0$val$val) > 0))

# Variable bindings: unbound Sum receives float/int term
Sum2 <- Unbound("S2")
r2 <- run_bound("Quantum_mechanics", "Physics", Sum2)
stopifnot(isTRUE(r2$ok), r2$val$tag %in% c("int", "float"))

# Fallback when bulk_collect removed: classic EndAggregate path still works
rm(list = ls(be, all.names = TRUE), envir = be)
ea <<- 0L; ba <<- 0L
Sum3 <- Unbound("S3")
r3 <- run_bound("Quantum_mechanics", "Physics", Sum3)
stopifnot(isTRUE(r3$ok), as.numeric(r3$val$val) > 0)
stopifnot(ba >= 1L, ea >= 1L)
# Parity vs fast-path value
stopifnot(abs(as.numeric(r3$val$val) - as.numeric(r2$val$val)) < 1e-12)

# Unsupported shape: try_scalar returns NULL for bag/collect kinds
state <- WamRuntime$new_state(); WamRuntime$promote_regs(state)
# Synthetic BeginAggregate(collect) instruction
instr <- BeginAggregate("collect", 205L, 208L)
# Need a matching EndAggregate in program — use a throwaway program copy
prog <- shared_program
# Append a tiny collect region at end
n0 <- length(prog$instructions)
prog$instructions <- c(prog$instructions,
  list(BeginAggregate("collect", 205L, 208L),
       Proceed(),
       EndAggregate(205L)))
state$pc <- as.integer(n0 + 1L)
handled <- WamRuntime$try_scalar_aggregate_fastpath(
  prog, state, prog$instructions[[state$pc]])
stopifnot(is.null(handled))

# Nested BeginAggregate in body is ineligible
prog2 <- shared_program
n1 <- length(prog2$instructions)
prog2$instructions <- c(prog2$instructions,
  list(BeginAggregate("sum", 205L, 208L),
       BeginAggregate("sum", 206L, 209L),
       EndAggregate(206L),
       EndAggregate(205L)))
state2 <- WamRuntime$new_state(); WamRuntime$promote_regs(state2)
state2$pc <- as.integer(n1 + 1L)
handled2 <- WamRuntime$try_scalar_aggregate_fastpath(
  prog2, state2, prog2$instructions[[state2$pc]])
stopifnot(is.null(handled2))

# compile helper: zero-length body, strict is/2, SetVariable arithmetic
# inputs, and invalid operator arities are outside the closed lax shape.
stopifnot(is.null(WamRuntime$compile_scalar_aggregate_arith(
  shared_program, list(), 0L, 210L)))
plus_fid <- as.integer(WamRuntime$intern(intern_table, "+"))
strict_body <- list(
  PutVariable(210L, 1L), PutStructure(plus_fid, 2L, 2L),
  SetConstant(IntTerm(1L)), SetConstant(IntTerm(2L)),
  BuiltinCall("is/2", 2L))
stopifnot(is.null(WamRuntime$compile_scalar_aggregate_arith(
  shared_program, strict_body, 0L, 210L)))
var_body <- list(
  PutVariable(210L, 1L), PutStructure(plus_fid, 2L, 2L),
  SetVariable(211L), SetConstant(IntTerm(2L)),
  BuiltinCall("is_lax/2", 2L))
stopifnot(is.null(WamRuntime$compile_scalar_aggregate_arith(
  shared_program, var_body, 0L, 210L)))
unary_plus_body <- list(
  PutVariable(210L, 1L), PutStructure(plus_fid, 2L, 1L),
  SetConstant(IntTerm(1L)), BuiltinCall("is_lax/2", 2L))
stopifnot(is.null(WamRuntime$compile_scalar_aggregate_arith(
  shared_program, unary_plus_body, 0L, 210L)))

# Preserve aggregate term tags: min/max returns the selected item type,
# not a float merely because some non-selected item was a float.
min_i <- WamRuntime$scalar_aggregate_bag_term(
  "min", 2L, 0, FALSE, 1, FALSE)
min_f <- WamRuntime$scalar_aggregate_bag_term(
  "min", 2L, 0, FALSE, 1, TRUE)
sum_inf <- WamRuntime$scalar_aggregate_bag_term(
  "sum", 1L, Inf, TRUE, NULL, FALSE)
stopifnot(identical(min_i$tag, "int"),
          identical(min_f$tag, "float"),
          identical(sum_inf$tag, "float"))

# Dynamic clauses shadow the bulk capability exactly as they shadow the
# ordinary lowered Call/Execute path.
prog_dyn <- shared_program
prog_dyn$dynamic <- new.env(parent = emptyenv())
assign("category_ancestor/4", list(), envir = prog_dyn$dynamic)
psb <- as.integer(prog_dyn$labels[["category_ancestor$power_sum_bound/3"]])
rel_begin <- which(vapply(
  prog_dyn$instructions[psb:length(prog_dyn$instructions)],
  function(x) identical(x$op, "BeginAggregate"), logical(1)))[[1]]
state_dyn <- WamRuntime$new_state()
state_dyn$pc <- as.integer(psb + rel_begin - 1L)
stopifnot(is.null(WamRuntime$try_scalar_aggregate_fastpath(
  prog_dyn, state_dyn, prog_dyn$instructions[[state_dyn$pc]])))

# ---- PERF-R-AGG-BATCH: typed numeric batch + vector reduce ----
# Re-register CA bulk collect (removed above for classic-fallback check).
WamRuntime$register_bulk_collect(
  shared_program, "category_ancestor/4",
  function(program, state) {
    WamRuntime$category_ancestor_bulk_collect(
      program, state, "category_parent", "category_parent/2", 10L)
  }, 3L)
be <- get("__bulk_collect__", envir = shared_program$lowered_dispatch,
          inherits = FALSE)

# CA bulk_collect returns typed numeric_batch (not per-hop records).
st_b <- WamRuntime$new_state(); WamRuntime$promote_regs(st_b)
WamRuntime$put_reg(st_b, 1L, atom_of("Quantum_mechanics"))
WamRuntime$put_reg(st_b, 2L, atom_of("Physics"))
WamRuntime$put_reg(st_b, 3L, Unbound("H"))
WamRuntime$put_reg(st_b, 4L, WamRuntime$wam_list_build(list(), intern_table))
cap <- get("category_ancestor/4", envir = be, inherits = FALSE)
batch <- cap$fn(shared_program, st_b)
stopifnot(is.list(batch), identical(batch$type, "numeric_batch"),
          identical(as.integer(batch$reg), 3L), is.numeric(batch$val),
          length(batch$val) >= 1L)

# as_bulk_numeric_batch: empty list, one/many records, typed batch, rejects junk.
stopifnot(identical(
  WamRuntime$as_bulk_numeric_batch(list(), 3L)$val, numeric(0)))
one <- WamRuntime$as_bulk_numeric_batch(list(list(reg = 3L, val = 2)), 3L)
many <- WamRuntime$as_bulk_numeric_batch(
  list(list(reg = 3L, val = 1), list(reg = 3L, val = 1),
       list(reg = 3L, val = 4)), 3L)
stopifnot(identical(one$val, 2), identical(many$val, c(1, 1, 4)))
stopifnot(identical(
  WamRuntime$as_bulk_numeric_batch(
    list(type = "numeric_batch", reg = 3L, val = c(1, 2)), 3L)$val, c(1, 2)))
stopifnot(is.null(WamRuntime$as_bulk_numeric_batch(
  list(list(reg = 3L, val = 1), list(reg = 4L, val = 2)), 3L)))
stopifnot(is.null(WamRuntime$as_bulk_numeric_batch(
  list(type = "numeric_batch", reg = 3L, val = "x"), 3L)))

# Vector arith shared with scalar eval: integer, whole float, fractional,
# Inf, NaN, and unary/binary ops via compiled-op helper.
stopifnot(identical(
  WamRuntime$eval_scalar_aggregate_arith_op("+", 2L, list(1, 2)), 3))
stopifnot(identical(
  WamRuntime$eval_scalar_aggregate_arith_op("-", 1L, list(4)), -4))
pow_v <- WamRuntime$eval_scalar_aggregate_arith_op(
  "**", 2L, list(c(1, 2, 3), 2))
stopifnot(identical(as.numeric(pow_v), c(1, 4, 9)))
stopifnot(isTRUE(WamRuntime$arith_num_is_float_tag(1.5)),
          isTRUE(WamRuntime$arith_num_is_float_tag(Inf)),
          isTRUE(WamRuntime$arith_num_is_float_tag(NaN)),
          !isTRUE(WamRuntime$arith_num_is_float_tag(3)),
          !isTRUE(WamRuntime$arith_num_is_float_tag(4.0)))

# Empty / singleton aggregate bags for sum|count|min|max.
stopifnot(identical(
  WamRuntime$scalar_aggregate_bag_term("sum", 0L, 0, FALSE, NULL, FALSE)$tag,
  "int"),
  identical(
    as.integer(WamRuntime$scalar_aggregate_bag_term(
      "sum", 0L, 0, FALSE, NULL, FALSE)$val), 0L),
  identical(
    as.integer(WamRuntime$scalar_aggregate_bag_term(
      "count", 0L, 0, FALSE, NULL, FALSE)$val), 0L),
  identical(
    as.integer(WamRuntime$scalar_aggregate_bag_term(
      "count", 1L, 0, FALSE, NULL, FALSE)$val), 1L),
  is.null(WamRuntime$scalar_aggregate_bag_term(
    "min", 0L, 0, FALSE, NULL, FALSE)),
  is.null(WamRuntime$scalar_aggregate_bag_term(
    "max", 0L, 0, FALSE, NULL, FALSE)),
  identical(
    WamRuntime$scalar_aggregate_bag_term(
      "min", 1L, 0, FALSE, 7, FALSE)$tag, "int"))

# Legacy list-record callback still accepted (coerce -> batch) and matches
# typed-batch result for the bound power-sum query.
WamRuntime$register_bulk_collect(
  shared_program, "category_ancestor/4",
  function(program, state) {
    b <- WamRuntime$category_ancestor_bulk_collect(
      program, state, "category_parent", "category_parent/2", 10L)
    if (is.list(b) && identical(b$type, "numeric_batch"))
      return(lapply(b$val, function(h) list(reg = 3L, val = h)))
    b
  }, 3L)
Sum_list <- Unbound("SL")
r_list <- run_bound("Quantum_mechanics", "Physics", Sum_list)
stopifnot(isTRUE(r_list$ok))
stopifnot(abs(as.numeric(r_list$val$val) - as.numeric(r2$val$val)) < 1e-12)

# Restore typed-batch CA publisher and confirm parity again.
WamRuntime$register_bulk_collect(
  shared_program, "category_ancestor/4",
  function(program, state) {
    WamRuntime$category_ancestor_bulk_collect(
      program, state, "category_parent", "category_parent/2", 10L)
  }, 3L)
Sum_batch <- Unbound("SB")
r_batch <- run_bound("Quantum_mechanics", "Physics", Sum_batch)
stopifnot(isTRUE(r_batch$ok))
stopifnot(abs(as.numeric(r_batch$val$val) - as.numeric(r2$val$val)) < 1e-12)

cat("OK aggregate_lower runtime semantics\\n")
'),
                close(S)
            ),
            process_create(path('Rscript'), ['agg_proof.R'],
                           [cwd(RDir), stdout(pipe(POut)), stderr(pipe(PErr)),
                            process(PID)]),
            read_string(POut, _, OutTxt), close(POut),
            read_string(PErr, _, ErrTxt), close(PErr),
            process_wait(PID, exit(Status)),
            (   Status =:= 0
            ->  assertion(once(sub_string(OutTxt, _, _, _,
                                          'OK aggregate_lower runtime semantics')))
            ;   format(user_error, 'agg_proof failed (~w):~n~w~n~w~n',
                       [Status, OutTxt, ErrTxt]),
                fail
            )
        ),
        cleanup_tmp_dir(TmpDir)).

:- end_tests(wam_r_aggregate_lower).
