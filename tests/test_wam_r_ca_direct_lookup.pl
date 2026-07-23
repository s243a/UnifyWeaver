:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% PERF-R-CA-DIRECT: capability-gated bound-arg1 parent lookup for the
% R category_ancestor/4 recursive kernel.
%
% Usage: swipl -g run_tests -t halt tests/test_wam_r_ca_direct_lookup.pl

:- use_module(library(plunit)).
:- use_module(library(filesex)).
:- use_module(library(process)).
:- use_module(library(readutil)).
:- use_module('../src/unifyweaver/targets/wam_r_target').

:- begin_tests(wam_r_ca_direct_lookup).

% ------------------------------------------------------------------
% Structural: grouped_by_first_atoms registers indexed arg1 lookup.
% ------------------------------------------------------------------
test(grouped_by_first_atoms_registers_indexed_lookup, [nondet]) :-
    setup_call_cleanup(
        (   unique_tmp('tmp_ca_direct_gbf', TmpDir),
            write_edge_tsv(TmpDir, 'edges.tsv',
                           ['a\tb', 'b\tc']),
            directory_file_path(TmpDir, 'edges.tsv', TsvPath),
            atom_string(TsvPath, TsvPathStr),
            install_ca_kernel(gbf_edge, 4)
        ),
        (   write_wam_r_project(
                [user:gbf_edge/2, user:cat_anc/4, user:max_depth/1],
                [r_fact_sources([source(gbf_edge/2,
                                        grouped_by_first_atoms(TsvPathStr))])],
                TmpDir),
            directory_file_path(TmpDir, 'R/generated_program.R', ProgPath),
            read_file_to_string(ProgPath, Code, []),
            assertion(once(sub_string(Code, _, _, _,
                'read_facts_grouped_tsv_atoms'))),
            assertion(once(sub_string(Code, _, _, _,
                'register_indexed_arg1_parent_lookup(shared_program, "gbf_edge/2"'))),
            assertion(once(sub_string(Code, _, _, _,
                'arg1_lookups = new.env'))),
            assertion(once(sub_string(Code, _, _, _,
                'pred_cat_anc_kernel_ca'))),
            directory_file_path(TmpDir, 'R/wam_runtime.R', RtPath),
            read_file_to_string(RtPath, Rt, []),
            assertion(once(sub_string(Rt, _, _, _,
                'make_indexed_arg1_parent_lookup_ids'))),
            assertion(once(sub_string(Rt, _, _, _,
                'category_ancestor_hops_rec_ids')))
        ),
        (   cleanup_tmp(TmpDir),
            uninstall_ca_kernel
        )).

% ------------------------------------------------------------------
% Structural: lazy LMDB gets on-demand capability (no silent stream).
% ------------------------------------------------------------------
test(lmdb_lazy_registers_ondemand_lookup_not_materialize) :-
    setup_call_cleanup(
        unique_tmp('tmp_ca_direct_lmdb_lazy', TmpDir),
        (   write_wam_r_project(
                [user:lzedge/2],
                [r_fact_sources([source(lzedge/2,
                                        lmdb_arg1_v1('/tmp/lz_ca.lmdb'))])],
                TmpDir),
            directory_file_path(TmpDir, 'R/generated_program.R', ProgPath),
            read_file_to_string(ProgPath, Code, []),
            assertion(once(sub_string(Code, _, _, _,
                'lmdb_arg1_v1_dispatch('))),
            assertion(once(sub_string(Code, _, _, _,
                'register_lmdb_arg1_parent_lookup(shared_program, "lzedge/2"'))),
            assertion(\+ sub_string(Code, _, _, _,
                'register_indexed_arg1_parent_lookup(shared_program, "lzedge/2"')),
            assertion(\+ sub_string(Code, _, _, _,
                'lmdb_arg1_v1_stream('))
        ),
        cleanup_tmp(TmpDir)).

test(lmdb_cached_registers_cached_ondemand_lookup) :-
    setup_call_cleanup(
        unique_tmp('tmp_ca_direct_lmdb_cached', TmpDir),
        (   write_wam_r_project(
                [user:chedge/2],
                [r_fact_sources([source(chedge/2,
                                        lmdb_arg1_v1('/tmp/ch_ca.lmdb'))]),
                 lmdb_materialisation(cached)],
                TmpDir),
            directory_file_path(TmpDir, 'R/generated_program.R', ProgPath),
            read_file_to_string(ProgPath, Code, []),
            assertion(once(sub_string(Code, _, _, _,
                'register_lmdb_cached_arg1_parent_lookup(shared_program, "chedge/2"'))),
            assertion(once(sub_string(Code, _, _, _,
                'lmdb_arg1_v1_cached_dispatch('))),
            assertion(\+ sub_string(Code, _, _, _,
                'lmdb_arg1_v1_stream('))
        ),
        cleanup_tmp(TmpDir)).

test(lmdb_eager_registers_indexed_lookup) :-
    setup_call_cleanup(
        unique_tmp('tmp_ca_direct_lmdb_eager', TmpDir),
        (   write_wam_r_project(
                [user:egedge/2],
                [r_fact_sources([source(egedge/2,
                                        lmdb_arg1_v1('/tmp/eg_ca.lmdb'))]),
                 lmdb_materialisation(eager)],
                TmpDir),
            directory_file_path(TmpDir, 'R/generated_program.R', ProgPath),
            read_file_to_string(ProgPath, Code, []),
            assertion(once(sub_string(Code, _, _, _,
                'lmdb_arg1_v1_stream('))),
            assertion(once(sub_string(Code, _, _, _,
                'register_indexed_arg1_parent_lookup(shared_program, "egedge/2"')))
        ),
        cleanup_tmp(TmpDir)).

% ------------------------------------------------------------------
% Runtime: direct selected, fallback when cleared, result parity,
% multipath, cycles, visited, depth boundary, numeric atoms.
% ------------------------------------------------------------------
test(ca_direct_fallback_parity_and_contract_rscript) :-
    once((rscript_available -> ca_direct_runtime_proof ; true)).

ca_direct_runtime_proof :-
    setup_call_cleanup(
        ca_direct_setup_program(TmpDir, RDir),
        (   directory_file_path(RDir, 'ca_direct_proof.R', Script),
            setup_call_cleanup(
                open(Script, write, S),
                write(S,
'source("generated_program.R")
stopifnot(!is.null(shared_program$arg1_lookups))
stopifnot(exists("cparent/2", envir = shared_program$arg1_lookups, inherits = FALSE))
cap <- WamRuntime$get_arg1_capability(shared_program, "cparent/2")
stopifnot(!is.null(cap$lookup), !is.null(cap$lookup_ids))
lk <- WamRuntime$get_arg1_parent_lookup(shared_program, "cparent/2")
stopifnot(!is.null(lk))
lk_ids <- WamRuntime$get_arg1_parent_lookup_ids(shared_program, "cparent/2")
stopifnot(!is.null(lk_ids))

collect_hops <- function(cat, root, visited_chars, sorted = TRUE) {
  state <- WamRuntime$new_state()
  WamRuntime$promote_regs(state)
  hops <- Unbound("H")
  vis_elems <- lapply(visited_chars, function(nm)
    Atom(WamRuntime$intern(intern_table, nm)))
  vis <- WamRuntime$wam_list_build(vis_elems, intern_table)
  WamRuntime$put_reg(state, 1L, Atom(WamRuntime$intern(intern_table, cat)))
  WamRuntime$put_reg(state, 2L, Atom(WamRuntime$intern(intern_table, root)))
  WamRuntime$put_reg(state, 3L, hops)
  WamRuntime$put_reg(state, 4L, vis)
  state$cp <- 0L
  fn <- get("cat_anc/4", envir = shared_program$lowered_dispatch, inherits = FALSE)
  out <- integer(0)
  ok <- isTRUE(fn(shared_program, state))
  while (isTRUE(ok)) {
    v <- WamRuntime$deref(state, hops)
    stopifnot(!is.null(v), identical(v$tag, "int"))
    out <- c(out, as.integer(v$val))
    ok <- isTRUE(WamRuntime$backtrack(state))
  }
  if (isTRUE(sorted)) sort(out) else out
}

# ID-native path (lookup_ids present)
d_multi_eq  <- collect_hops("a", "d", character(0))
d_multi_une <- collect_hops("a", "z", character(0))
d_cycle     <- collect_hops("loop_a", "loop_b", character(0))
d_visited   <- collect_hops("a", "d", c("b"))
d_root_vis  <- collect_hops("a", "d", c("d"))
d_depth     <- collect_hops("deep0", "deep3", character(0))
d_num       <- collect_hops("1827", "9001", character(0))
d_num_miss  <- collect_hops("1827", "nope", character(0))
d_order     <- collect_hops("a", "d", character(0), sorted = FALSE)

stopifnot(identical(d_multi_eq, c(2L, 2L)))
stopifnot(identical(d_multi_une, c(1L, 3L)))
stopifnot(identical(d_cycle, 1L))
stopifnot(identical(d_visited, 2L))
stopifnot(length(d_root_vis) == 0L)              # root already in Visited
stopifnot(identical(d_depth, 3L))
stopifnot(identical(d_num, 2L))
stopifnot(length(d_num_miss) == 0L)
# DFS order: a->b->d before a->c->d (edge order in TSV)
stopifnot(identical(d_order, c(2L, 2L)))

# Forced TermValue-capability fallback (lookup only, no lookup_ids)
facts <- pred_cparent_facts; indexes <- pred_cparent_indexes
WamRuntime$register_arg1_capability(shared_program, "cparent/2", list(
  lookup = WamRuntime$make_indexed_arg1_parent_lookup(facts, indexes),
  lookup_ids = NULL
))
stopifnot(is.null(WamRuntime$get_arg1_parent_lookup_ids(shared_program, "cparent/2")))
stopifnot(!is.null(WamRuntime$get_arg1_parent_lookup(shared_program, "cparent/2")))
t_multi_eq  <- collect_hops("a", "d", character(0))
t_multi_une <- collect_hops("a", "z", character(0))
t_cycle     <- collect_hops("loop_a", "loop_b", character(0))
t_visited   <- collect_hops("a", "d", c("b"))
t_root_vis  <- collect_hops("a", "d", c("d"))
t_depth     <- collect_hops("deep0", "deep3", character(0))
t_num       <- collect_hops("1827", "9001", character(0))
t_order     <- collect_hops("a", "d", character(0), sorted = FALSE)

# Forced iterate_goal fallback (no capability)
rm(list = "cparent/2", envir = shared_program$arg1_lookups)
stopifnot(is.null(WamRuntime$get_arg1_parent_lookup(shared_program, "cparent/2")))
f_multi_eq  <- collect_hops("a", "d", character(0))
f_multi_une <- collect_hops("a", "z", character(0))
f_cycle     <- collect_hops("loop_a", "loop_b", character(0))
f_visited   <- collect_hops("a", "d", c("b"))
f_root_vis  <- collect_hops("a", "d", c("d"))
f_depth     <- collect_hops("deep0", "deep3", character(0))
f_num       <- collect_hops("1827", "9001", character(0))
f_order     <- collect_hops("a", "d", character(0), sorted = FALSE)

# Three-way parity (sorted values)
stopifnot(identical(d_multi_eq, t_multi_eq), identical(d_multi_eq, f_multi_eq))
stopifnot(identical(d_multi_une, t_multi_une), identical(d_multi_une, f_multi_une))
stopifnot(identical(d_cycle, t_cycle), identical(d_cycle, f_cycle))
stopifnot(identical(d_visited, t_visited), identical(d_visited, f_visited))
stopifnot(identical(d_root_vis, t_root_vis), identical(d_root_vis, f_root_vis))
stopifnot(identical(d_depth, t_depth), identical(d_depth, f_depth))
stopifnot(identical(d_num, t_num), identical(d_num, f_num))
# DFS emission order parity across all three paths
stopifnot(identical(d_order, t_order), identical(d_order, f_order))
cat("ok\n")
'),
                close(S)),
            process_create(path('Rscript'), ['ca_direct_proof.R'],
                           [ cwd(RDir),
                             stdout(pipe(OutS)),
                             stderr(pipe(ErrS)),
                             process(PID)
                           ]),
            read_string(OutS, _, Out), close(OutS),
            read_string(ErrS, _, Err), close(ErrS),
            process_wait(PID, Status),
            (   Status = exit(0),
                once(sub_string(Out, _, _, _, "ok"))
            ->  true
            ;   format(user_error, 'ca_direct_proof failed: ~w~n~w~n~w~n',
                       [Status, Out, Err]),
                fail
            )
        ),
        ca_direct_cleanup(TmpDir)).

% functions + interpreter modes (FactSource + CA kernel queries)
test(ca_direct_functions_mode_e2e) :-
    once((rscript_available -> ca_mode_e2e(functions) ; true)).

test(ca_direct_interpreter_mode_e2e) :-
    once((rscript_available -> ca_mode_e2e(interpreter) ; true)).

ca_mode_e2e(EmitMode) :-
    setup_call_cleanup(
        (   unique_tmp('tmp_ca_direct_mode', TmpDir),
            write_edge_tsv(TmpDir, 'edges.tsv',
                           ['a\tb', 'a\tc', 'b\td', 'c\td',
                            '1827\tmid', 'mid\t9001']),
            directory_file_path(TmpDir, 'edges.tsv', TsvPath),
            atom_string(TsvPath, TsvPathStr),
            install_ca_kernel(cparent, 4),
            retractall(user:ca_ok),
            retractall(user:ca_paths),
            retractall(user:ca_num),
            assertz((user:ca_ok :- cat_anc(a, d, 2, []))),
            assertz((user:ca_paths :-
                findall(H, cat_anc(a, d, H, []), L0),
                msort(L0, L),
                L == [2, 2])),
            assertz((user:ca_num :- cat_anc('1827', '9001', 2, [])))
        ),
        (   write_wam_r_project(
                [user:cparent/2, user:cat_anc/4, user:max_depth/1,
                 user:ca_ok/0, user:ca_paths/0, user:ca_num/0],
                [emit_mode(EmitMode),
                 r_fact_sources([source(cparent/2,
                                        grouped_by_first_atoms(TsvPathStr))])],
                TmpDir),
            directory_file_path(TmpDir, 'R', RDir),
            directory_file_path(TmpDir, 'R/generated_program.R', ProgPath),
            read_file_to_string(ProgPath, Code, []),
            assertion(once(sub_string(Code, _, _, _,
                'register_indexed_arg1_parent_lookup(shared_program, "cparent/2"'))),
            run_rscript_query(RDir, 'ca_ok/0', Out1),
            assertion(once(sub_string(Out1, _, _, _, "true"))),
            run_rscript_query(RDir, 'ca_paths/0', Out2),
            assertion(once(sub_string(Out2, _, _, _, "true"))),
            run_rscript_query(RDir, 'ca_num/0', Out3),
            assertion(once(sub_string(Out3, _, _, _, "true")))
        ),
        (   cleanup_tmp(TmpDir),
            uninstall_ca_kernel,
            retractall(user:ca_ok),
            retractall(user:ca_paths),
            retractall(user:ca_num)
        )).

:- end_tests(wam_r_ca_direct_lookup).

% ------------------------------------------------------------------
% Helpers
% ------------------------------------------------------------------
% Kernel detector requires a direct body call to EdgePred/2 (not call/N).
install_ca_kernel(EdgePred, MaxDepth) :-
    uninstall_ca_kernel,
    assertz(user:max_depth(MaxDepth)),
    Head1 = user:cat_anc(Cat, Root, 1, Visited),
    Edge1 =.. [EdgePred, Cat, Root],
    assertz((Head1 :- user:Edge1, \+ member(Root, Visited))),
    Head2 = user:cat_anc(Cat, Root, Hops, Visited),
    Edge2 =.. [EdgePred, Cat, Mid],
    assertz((Head2 :-
        max_depth(MaxD),
        length(Visited, Depth),
        Depth < MaxD, !,
        user:Edge2,
        \+ member(Mid, Visited),
        user:cat_anc(Mid, Root, H1, [Mid|Visited]),
        Hops is H1 + 1)).

uninstall_ca_kernel :-
    retractall(user:max_depth(_)),
    retractall(user:cat_anc(_,_,_,_)).

ca_direct_setup_program(TmpDir, RDir) :-
    unique_tmp('tmp_ca_direct_rt', TmpDir),
    % multipath equal: a->b->d, a->c->d
    % multipath unequal: a->z (1), a->p->q->z (3)
    % cycle, depth boundary, numeric-looking atoms
    write_edge_tsv(TmpDir, 'edges.tsv',
                   ['a\tb', 'a\tc', 'b\td', 'c\td',
                    'a\tz', 'a\tp', 'p\tq', 'q\tz',
                    'loop_a\tloop_b', 'loop_b\tloop_a',
                    'deep0\tdeep1', 'deep1\tdeep2', 'deep2\tdeep3',
                    '1827\tmid', 'mid\t9001']),
    directory_file_path(TmpDir, 'edges.tsv', TsvPath),
    atom_string(TsvPath, TsvPathStr),
    install_ca_kernel(cparent, 3),
    write_wam_r_project(
        [user:cparent/2, user:cat_anc/4, user:max_depth/1],
        [emit_mode(functions),
         r_fact_sources([source(cparent/2,
                                grouped_by_first_atoms(TsvPathStr))])],
        TmpDir),
    directory_file_path(TmpDir, 'R', RDir).

ca_direct_cleanup(TmpDir) :-
    cleanup_tmp(TmpDir),
    uninstall_ca_kernel.

write_edge_tsv(Dir, File, Lines) :-
    directory_file_path(Dir, File, Path),
    setup_call_cleanup(
        open(Path, write, S),
        forall(member(Line, Lines), format(S, '~w~n', [Line])),
        close(S)).

run_rscript_query(RDir, Query, Out) :-
    process_create(path('Rscript'),
                   ['generated_program.R', Query],
                   [ cwd(RDir),
                     stdout(pipe(OutStream)),
                     stderr(pipe(ErrStream)),
                     process(PID)
                   ]),
    read_string(OutStream, _, Out), close(OutStream),
    read_string(ErrStream, _, _), close(ErrStream),
    process_wait(PID, _).

rscript_available :-
    catch((
        process_create(path('Rscript'), ['--version'],
                       [ stdout(null), stderr(null), process(PID) ]),
        process_wait(PID, exit(0))
    ), _, fail).

unique_tmp(Prefix, Dir) :-
    tmp_file(Prefix, Dir),
    catch(delete_directory_and_contents(Dir), _, true),
    make_directory_path(Dir).

cleanup_tmp(Dir) :-
    catch(delete_directory_and_contents(Dir), _, true).
