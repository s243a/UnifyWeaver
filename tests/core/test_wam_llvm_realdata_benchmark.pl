:- encoding(utf8).
% test_wam_llvm_realdata_benchmark.pl
%
% First real-workload LLVM benchmark: runs the `category_ancestor`
% foreign kernel against the dev-scale Wikipedia category fixture
% (`data/benchmark/dev/category_parent.tsv`). Previously the only
% LLVM benchmark was a synthetic 100/500-node chain
% (tests/core/test_wam_llvm_benchmark.pl); this one exercises the
% same path on real category graph data so the M-series perf work
% has at least one number comparable to the Haskell/Rust effective-
% distance benchmark in
% `benchmarks/wam_effective_distance_cross_target.md`.
%
% What this measures:
%   - Compile time: writing the .ll module + llc + clang
%   - Run time: 100 iterations of category_ancestor on a known seed
%     pair (Quantum_mechanics → Physics, 2 hops via Subfields_of_physics)
%
% Not measured here (out of scope):
%   - The outer enumeration over (article, root) pairs that the full
%     effective_distance computation requires — that needs setof/3 +
%     findall/3 in the bytecode interpreter, which is the next
%     planned direction.
%   - Larger fixture scales (300/1k/10k). The dev fixture is ~200
%     edges; running on larger scales is a follow-up once the
%     infrastructure proves out.

:- use_module('../../src/unifyweaver/targets/wam_llvm_target',
    [write_wam_llvm_project/3,
     clear_llvm_foreign_kernel_specs/0]).
:- use_module(library(process)).
:- use_module(library(readutil)).
:- use_module(library(pcre)).

% Edge facts loaded from the TSV fixture.
:- dynamic dev_cat_parent/2.

% Wrapper predicate: dispatched via foreign_predicates to the
% category_ancestor kernel, which scans dev_cat_parent/2 facts.
:- dynamic dev_cat_ancestor/4.
dev_cat_ancestor(_, _, _, _) :- fail.

load_dev_fixture :-
    retractall(dev_cat_parent(_, _)),
    source_file(load_dev_fixture, ThisFile),
    file_directory_name(ThisFile, ThisDir),
    directory_file_path(ThisDir,
        '../../data/benchmark/dev/category_parent.tsv', TsvPath),
    setup_call_cleanup(
        open(TsvPath, read, S),
        load_dev_lines(S, 0, Loaded),
        close(S)),
    format(user_error, '  loaded ~w category_parent edges~n', [Loaded]).

load_dev_lines(S, Acc, Out) :-
    read_line_to_string(S, Line),
    ( Line == end_of_file
    -> Out = Acc
    ;  ( split_string(Line, "\t", "", [ChildStr, ParentStr]),
         ChildStr \== "child"   % skip header row
       -> atom_string(Child, ChildStr),
          atom_string(Parent, ParentStr),
          assertz(dev_cat_parent(Child, Parent)),
          Acc1 is Acc + 1
       ;  Acc1 = Acc
       ),
       load_dev_lines(S, Acc1, Out)
    ).

host_target_triple(Triple) :-
    ( catch(
        ( process_create(path(clang), ['-print-target-triple'],
              [stdout(pipe(Out)), stderr(null), process(PID)]),
          read_string(Out, _, Raw), close(Out),
          process_wait(PID, exit(0))
        ), _, fail)
    -> split_string(Raw, "", "\n\r\t ", [S]), atom_string(Triple, S)
    ;  Triple = 'x86_64-pc-linux-gnu'
    ).

extract_instr_count(Src, _Pred, C) :-
    re_matchsub(
        "@module_code = private constant \\[(?<n>\\d+) x %Instruction\\]",
        Src, M, []),
    get_dict(n, M, NS), number_string(C, NS).
extract_label_count(Src, _Pred, C) :-
    re_matchsub(
        "@module_labels = private constant \\[(?<n>\\d+) x i32\\]",
        Src, M, []),
    get_dict(n, M, NS), number_string(C, NS).

bench_devscale(StartAtom, CompileMs, RunMs, AncestorCount) :-
    clear_llvm_foreign_kernel_specs,
    tmp_file_stream(text, LLPath, Stream), close(Stream),
    host_target_triple(Triple),
    get_time(T0),
    write_wam_llvm_project(
        [user:dev_cat_ancestor/4],
        [ module_name('m_realdata'),
          target_triple(Triple),
          target_datalayout(''),
          foreign_predicates([
              dev_cat_ancestor/4 - category_ancestor -
                  [edge_pred(dev_cat_parent/2), max_depth(10)]
          ])
        ],
        LLPath),
    read_file_to_string(LLPath, Src, []),
    extract_instr_count(Src, dev_cat_ancestor, IC),
    extract_label_count(Src, dev_cat_ancestor, LC),
    % The foreign-kernel codegen runs intern_atom over every atom in
    % dev_cat_parent during write_wam_llvm_project. Look up the ID
    % for our seed so the driver IR can pass the right i64 payload.
    wam_llvm_target:atom_table_entry(StartAtom, StartId),
    % Driver: run dev_cat_ancestor(Start, _Target, _Hops, []) in
    % streaming mode (A2..A4 unbound) — the kernel walks the entire
    % BFS frontier and yields each (Target, Hops) pair through the
    % multi-result iterator. We loop 100 times and, on the LAST
    % iteration, count results via backtrack to sanity-check.
    format(atom(DriverIR),
'define i32 @main() {
entry:
  br label %loop
loop:
  %i = phi i32 [0, %entry], [%i_next, %loop_continue]
  %done = icmp uge i32 %i, 99
  br i1 %done, label %final, label %loop_continue
loop_continue:
  ; Plain timed iteration — discard result, free state.
  %a1_0 = insertvalue %Value undef, i32 0, 0
  %a1 = insertvalue %Value %a1_0, i64 ~w, 1
  %a2_0 = insertvalue %Value undef, i32 6, 0
  %a2 = insertvalue %Value %a2_0, i64 0, 1
  %vm = call %WamState* @wam_state_new(
      %Instruction* getelementptr ([~w x %Instruction], [~w x %Instruction]* @module_code, i32 0, i32 0),
      i32 ~w,
      i32* getelementptr ([~w x i32], [~w x i32]* @module_labels, i32 0, i32 0),
      i32 ~w)
  call void @wam_set_reg(%WamState* %vm, i32 0, %Value %a1)
  call void @wam_set_reg(%WamState* %vm, i32 1, %Value %a2)
  call void @wam_set_reg(%WamState* %vm, i32 2, %Value %a2)
  call void @wam_set_reg(%WamState* %vm, i32 3, %Value %a2)
  %ok = call i1 @run_loop(%WamState* %vm)
  call void @wam_state_free(%WamState* %vm)
  %i_next = add i32 %i, 1
  br label %loop
final:
  ; Final iteration: count results via backtracking, return count.
  %f_a1_0 = insertvalue %Value undef, i32 0, 0
  %f_a1 = insertvalue %Value %f_a1_0, i64 ~w, 1
  %f_a2_0 = insertvalue %Value undef, i32 6, 0
  %f_a2 = insertvalue %Value %f_a2_0, i64 0, 1
  %f_vm = call %WamState* @wam_state_new(
      %Instruction* getelementptr ([~w x %Instruction], [~w x %Instruction]* @module_code, i32 0, i32 0),
      i32 ~w,
      i32* getelementptr ([~w x i32], [~w x i32]* @module_labels, i32 0, i32 0),
      i32 ~w)
  call void @wam_set_reg(%WamState* %f_vm, i32 0, %Value %f_a1)
  call void @wam_set_reg(%WamState* %f_vm, i32 1, %Value %f_a2)
  call void @wam_set_reg(%WamState* %f_vm, i32 2, %Value %f_a2)
  call void @wam_set_reg(%WamState* %f_vm, i32 3, %Value %f_a2)
  %f_ok = call i1 @run_loop(%WamState* %f_vm)
  br i1 %f_ok, label %count_entry, label %no_results

count_entry:
  br label %count_loop

count_loop:
  %count = phi i32 [1, %count_entry], [%count_inc, %got_next]
  %bt_ok = call i1 @backtrack(%WamState* %f_vm)
  br i1 %bt_ok, label %got_next, label %count_done

got_next:
  %count_inc = add i32 %count, 1
  br label %count_loop

count_done:
  call void @wam_state_free(%WamState* %f_vm)
  ret i32 %count

no_results:
  call void @wam_state_free(%WamState* %f_vm)
  ret i32 0
}
', [StartId,
    IC, IC, IC, LC, LC, LC,
    StartId,
    IC, IC, IC, LC, LC, LC]),
    setup_call_cleanup(
        open(LLPath, append, Out),
        ( write(Out, '\n'), write(Out, DriverIR) ),
        close(Out)),
    atom_concat(LLPath, '.o', OPath),
    atom_concat(LLPath, '.out', BinPath),
    format(atom(LlcCmd),
        'llc -O2 -filetype=obj -relocation-model=pic ~w -o ~w 2>/dev/null',
        [LLPath, OPath]),
    shell(LlcCmd, _),
    format(atom(ClangCmd), 'clang -O2 ~w -o ~w -lm 2>/dev/null',
        [OPath, BinPath]),
    shell(ClangCmd, _),
    get_time(T1),
    CompileMs is (T1 - T0) * 1000,
    get_time(T2),
    shell(BinPath, ExitCode),
    get_time(T3),
    RunMs is (T3 - T2) * 1000,
    AncestorCount = ExitCode,
    catch(delete_file(LLPath), _, true),
    catch(delete_file(OPath), _, true),
    catch(delete_file(BinPath), _, true),
    clear_llvm_foreign_kernel_specs.

% --- Main ---

run_benchmark :-
    format('~n=== LLVM real-data benchmark (dev-scale Wikipedia) ===~n'),
    format('  category_ancestor kernel over data/benchmark/dev/~n'),
    format('  100 iterations per scenario, llc -O2 + clang -O2~n~n'),
    load_dev_fixture,
    % Streaming mode: kernel enumerates all reachable ancestors of
    % Quantum_mechanics in the dev fixture's category_parent graph.
    % The exact count depends on the graph structure (should be >0
    % since Quantum_mechanics has parents in the data).
    Start = 'Quantum_mechanics',
    bench_devscale(Start, CompileMs, RunMs, AncestorCount),
    format('  ~w (streaming all ancestors):~n', [Start]),
    format('     compile=~3fms  run(x100)=~3fms  ancestors_found=~w~n',
        [CompileMs, RunMs, AncestorCount]),
    ( AncestorCount > 0
    -> format('  PASS: kernel returned ~w ancestors~n', [AncestorCount])
    ;  format('  FAIL: kernel returned 0 ancestors (expected > 0)~n'),
       throw(no_ancestors)
    ),
    PerIter is RunMs / 100.0,
    format('  per-iter: ~3fms~n', [PerIter]).

test_all :-
    ( process_which('clang'), process_which('llc')
    -> catch(run_benchmark, E,
           ( format(user_error, '  ERROR: ~w~n', [E]), halt(1) ))
    ;  format('  SKIP: clang or llc not found~n')
    ).

process_which(Tool) :-
    catch(
        ( process_create(path(which), [Tool],
              [stdout(pipe(Out)), stderr(null), process(PID)]),
          read_string(Out, _, _), close(Out),
          process_wait(PID, exit(0))
        ), _, fail).

:- initialization(test_all, main).
