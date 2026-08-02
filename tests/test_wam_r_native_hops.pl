:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% PERF-R-NATIVE-HOPS-0: optional native category_ancestor hop kernel.

:- use_module(library(plunit)).
:- use_module(library(filesex)).
:- use_module(library(process)).
:- use_module(library(readutil)).
:- use_module('../src/unifyweaver/targets/wam_r_target').
:- use_module('../examples/benchmark/generate_wam_r_effective_distance_benchmark').

:- begin_tests(wam_r_native_hops).

rscript_available :-
    catch((process_create(path('Rscript'), ['--version'],
                          [stdout(null), stderr(null), process(PID)]),
           process_wait(PID, exit(0))),
          _, fail).

unique_tmp(Prefix, Dir) :-
    get_time(T),
    format(atom(Name), '~w_~w', [Prefix, T]),
    tmp_file(Name, Dir),
    make_directory_path(Dir).

cleanup_tmp(Dir) :-
    (   exists_directory(Dir)
    ->  delete_directory_and_contents(Dir)
    ;   true
    ).

test(native_hops_emits_c_source_and_soft_compile) :-
    setup_call_cleanup(
        unique_tmp('tmp_wam_r_native_emit', TmpDir),
        (   once(generate('data/benchmark/dev/facts.pl', TmpDir,
                          kernels_on, functions)),
            directory_file_path(TmpDir, 'src/uw_ca_hops.c', CPath),
            directory_file_path(TmpDir, 'R/wam_runtime.R', RtPath),
            directory_file_path(TmpDir, 'R/generated_program.R', ProgPath),
            assertion(exists_file(CPath)),
            read_file_to_string(CPath, CSrc, []),
            assertion(once(sub_string(CSrc, _, _, _, 'uw_ca_hops_ids'))),
            assertion(once(sub_string(CSrc, _, _, _, 'PERF-R-NATIVE-HOPS-0'))),
            read_file_to_string(RtPath, Rt, []),
            assertion(once(sub_string(Rt, _, _, _,
                'configure_native_ca_hops'))),
            assertion(once(sub_string(Rt, _, _, _,
                'try_native_ca_hops_ids'))),
            assertion(once(sub_string(Rt, _, _, _,
                'category_ancestor_hops_ids'))),
            read_file_to_string(ProgPath, Prog, []),
            assertion(once(sub_string(Prog, _, _, _,
                'configure_native_ca_hops'))),
            % Soft-compile: .so present when toolchain works; absence is OK.
            directory_file_path(TmpDir, 'src/uw_ca_hops.so', SoPath),
            (   exists_file(SoPath)
            ->  true
            ;   true
            )
        ),
        cleanup_tmp(TmpDir)).

test(native_hops_parity_and_fallback, [condition(rscript_available)]) :-
    setup_call_cleanup(
        unique_tmp('tmp_wam_r_native_rt', TmpDir),
        (   once(generate('data/benchmark/dev/facts.pl', TmpDir,
                          kernels_on, functions)),
            directory_file_path(TmpDir, 'R', RDir),
            directory_file_path(TmpDir, 'src/uw_ca_hops.so', SoPath),
            directory_file_path(RDir, 'native_proof.R', Script),
            setup_call_cleanup(
                open(Script, write, S),
                write(S,
'source("generated_program.R")
atom_of <- function(nm) Atom(WamRuntime$intern(intern_table, nm))
cap <- WamRuntime$get_arg1_capability(shared_program, "category_parent/2")
id_table <- WamRuntime$resolve_arg1_parent_id_table(cap$id_table)
stopifnot(!is.null(id_table))

run_hops <- function(force_r = FALSE) {
  if (force_r) WamRuntime$configure_native_ca_hops(NULL)
  state <- WamRuntime$new_state(); WamRuntime$promote_regs(state)
  WamRuntime$put_reg(state, 1L, atom_of("Quantum_mechanics"))
  WamRuntime$put_reg(state, 2L, atom_of("Physics"))
  WamRuntime$put_reg(state, 3L, Unbound("H"))
  WamRuntime$put_reg(state, 4L, WamRuntime$wam_list_build(
    list(atom_of("Quantum_mechanics")), intern_table))
  batch <- WamRuntime$category_ancestor_bulk_collect(
    shared_program, state, "category_parent", "category_parent/2", 10L)
  batch
}

# Pure-R baseline (disable native)
b_r <- run_hops(TRUE)
stopifnot(is.list(b_r), identical(b_r$type, "numeric_batch"),
          is.numeric(b_r$val))

# Reconfigure to project .so if present
so <- normalizePath(file.path("..", "src", "uw_ca_hops.so"), mustWork = FALSE)
WamRuntime$configure_native_ca_hops(so)
b_n <- run_hops(FALSE)
stopifnot(identical(as.numeric(b_r$val), as.numeric(b_n$val)))

# Missing shared library falls back without error
WamRuntime$configure_native_ca_hops("/nonexistent/uw_ca_hops.so")
b_fb <- run_hops(FALSE)
stopifnot(identical(as.numeric(b_r$val), as.numeric(b_fb$val)))

# Empty / singleton style via hops_ids helper
out <- new.env(parent = emptyenv()); out$buf <- integer(8L); out$n <- 0L
WamRuntime$configure_native_ca_hops(so)
WamRuntime$category_ancestor_hops_ids(
  NULL, 0L, 0L, integer(0), 0L, 10L, out, id_table)
stopifnot(identical(out$n, 0L))

cat("OK native_hops parity and fallback\\n")
'),
                close(S)
            ),
            process_create(path('Rscript'), ['native_proof.R'],
                           [cwd(RDir), stdout(pipe(POut)), stderr(pipe(PErr)),
                            process(PID)]),
            read_string(POut, _, OutTxt), close(POut),
            read_string(PErr, _, ErrTxt), close(PErr),
            process_wait(PID, exit(Status)),
            (   Status =:= 0
            ->  assertion(once(sub_string(OutTxt, _, _, _,
                                          'OK native_hops parity and fallback')))
            ;   format(user_error, 'native_proof failed (~w):~n~w~n~w~n',
                       [Status, OutTxt, ErrTxt]),
                fail
            ),
            % Hide .so if it was built: generated program must still run.
            (   exists_file(SoPath)
            ->  directory_file_path(TmpDir, 'src/uw_ca_hops.so.bak', Bak),
                rename_file(SoPath, Bak),
                process_create(path('Rscript'), ['native_proof.R'],
                               [cwd(RDir), stdout(pipe(P2)), stderr(pipe(E2)),
                                process(PID2)]),
                read_string(P2, _, _), close(P2),
                read_string(E2, _, _), close(E2),
                process_wait(PID2, exit(St2)),
                assertion(St2 =:= 0),
                rename_file(Bak, SoPath)
            ;   true
            )
        ),
        cleanup_tmp(TmpDir)).

test(native_hops_kernels_off_has_no_bulk_native_req) :-
    setup_call_cleanup(
        unique_tmp('tmp_wam_r_native_off', TmpDir),
        (   once(generate('data/benchmark/dev/facts.pl', TmpDir,
                          kernels_off, functions)),
            directory_file_path(TmpDir, 'R/generated_program.R', ProgPath),
            read_file_to_string(ProgPath, Prog, []),
            assertion(\+ sub_string(Prog, _, _, _,
                'register_bulk_collect(shared_program, "category_ancestor/4"')),
            % C source may still be emitted (harmless); runtime stays optional.
            directory_file_path(TmpDir, 'src/uw_ca_hops.c', CPath),
            assertion(exists_file(CPath))
        ),
        cleanup_tmp(TmpDir)).

:- end_tests(wam_r_native_hops).
