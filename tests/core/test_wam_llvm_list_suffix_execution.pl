:- encoding(utf8).
% test_wam_llvm_list_suffix_execution.pl
% End-to-end execution test for list_suffix2 kernel.
% When A2 is unbound (tag=6), the kernel enumerates all suffixes
% of the list in A1 via the multi-result foreign dispatch iterator.
%
% Tests:
%   1. Auto-detect: my_suffix/2 recognized as list_suffix2.
%   2. [a,b,c] has 4 suffixes → exit code 4.
%   3. [x] has 2 suffixes → exit code 2.
%   4. [] has 1 suffix (itself, the empty list) → exit code 1.

:- use_module('../../src/unifyweaver/targets/wam_llvm_target',
    [write_wam_llvm_project/3,
     llvm_foreign_kernel_spec/3,
     clear_llvm_foreign_kernel_specs/0]).
:- use_module(library(process)).
:- use_module(library(readutil)).
:- use_module(library(pcre)).

% A predicate matching the list_suffix2 clause shape.
:- dynamic my_suffix/2.
my_suffix(X, X).
my_suffix([_|Tail], Suffix) :- my_suffix(Tail, Suffix).

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

extract_instr_count(Src, P, C) :-
    Pat = "@module_code = private constant \\[(?<n>\\d+) x %Instruction\\]",
    re_matchsub(Pat, Src, M, []), get_dict(n, M, NS), number_string(C, NS).
extract_label_count(Src, P, C) :-
    Pat = "@module_labels = private constant \\[(?<n>\\d+) x i32\\]",
    re_matchsub(Pat, Src, M, []), get_dict(n, M, NS), number_string(C, NS).

test_autodetect :-
    format('--- list_suffix2 auto-detect ---~n'),
    clear_llvm_foreign_kernel_specs,
    tmp_file_stream(text, LLPath, Stream), close(Stream),
    host_target_triple(Triple),
    write_wam_llvm_project(
        [user:my_suffix/2],
        [ module_name('ls2_autodetect'),
          target_triple(Triple),
          target_datalayout(''),
          foreign_lowering(true)
        ],
        LLPath),
    ( llvm_foreign_kernel_spec(my_suffix/2, list_suffix2, _)
    -> format('  PASS: auto-detect registered my_suffix/2 as list_suffix2~n')
    ;  format('  FAIL: auto-detect did not match my_suffix/2~n'),
       throw(autodetect_failed)
    ),
    catch(delete_file(LLPath), _, true),
    clear_llvm_foreign_kernel_specs.

% Build a driver that creates a Prolog list of N atoms in LLVM IR,
% calls the WAM predicate with A2 unbound, counts results via backtracking.
run_suffix_case(Label, NumElems, Expected) :-
    format('  testing ~w: my_suffix(list_of_~w, X) expected ~w suffixes~n',
        [Label, NumElems, Expected]),
    clear_llvm_foreign_kernel_specs,
    tmp_file_stream(text, LLPath, Stream), close(Stream),
    host_target_triple(Triple),
    write_wam_llvm_project(
        [user:my_suffix/2],
        [ module_name('ls2_exec'),
          target_triple(Triple),
          target_datalayout(''),
          foreign_lowering(true)
        ],
        LLPath),
    read_file_to_string(LLPath, Src, []),
    extract_instr_count(Src, my_suffix, IC),
    extract_label_count(Src, my_suffix, LC),
    % Build the driver IR.
    % The driver allocates a %List struct with NumElems elements (all atom 0),
    % sets A1 to the list and A2 to unbound, runs the WAM loop,
    % then counts results via backtracking.
    build_list_driver_ir(NumElems, IC, LC, DriverIR),
    setup_call_cleanup(
        open(LLPath, append, Out),
        ( write(Out, '\n'), write(Out, DriverIR) ),
        close(Out)),
    atom_concat(LLPath, '.o', OPath),
    atom_concat(LLPath, '.out', BinPath),
    format(atom(LlcCmd),
        'llc -filetype=obj -relocation-model=pic ~w -o ~w 2>~w.llc.err',
        [LLPath, OPath, LLPath]),
    shell(LlcCmd, LlcExit),
    ( LlcExit =\= 0
    -> format('    FAIL: llc exit=~w~n', [LlcExit]), ExitCode = -1
    ;  format(atom(ClangCmd), 'clang ~w -o ~w 2>~w.clang.err',
           [OPath, BinPath, LLPath]),
       shell(ClangCmd, ClangExit),
       ( ClangExit =\= 0
       -> format('    FAIL: clang exit=~w~n', [ClangExit]), ExitCode = -1
       ;  shell(BinPath, ExitCode)
       )
    ),
    ( ExitCode =:= Expected
    -> format('    PASS: ~w returned ~w~n', [Label, ExitCode])
    ;  format('    FAIL: ~w returned ~w (expected ~w)~n', [Label, ExitCode, Expected])
    ),
    catch(delete_file(LLPath), _, true),
    catch(delete_file(OPath), _, true),
    catch(delete_file(BinPath), _, true),
    clear_llvm_foreign_kernel_specs,
    assertion(ExitCode =:= Expected).

% Generate LLVM IR for a driver that builds a list of NumElems atoms,
% calls the WAM loop, and counts suffixes via backtracking.
build_list_driver_ir(NumElems, IC, LC, DriverIR) :-
    % Build element initializers: all Atom(id=0).
    build_elem_stores(NumElems, 0, ElemStores),
    ElemBytes0 is NumElems * 16,
    ( ElemBytes0 =:= 0 -> ElemBytes = 16 ; ElemBytes = ElemBytes0 ),
    format(atom(DriverIR),
'define i32 @main() {
entry:
  ; Allocate elements array: ~w x %Value.
  %elems_mem = call i8* @malloc(i64 ~w)
  %elems = bitcast i8* %elems_mem to %Value*
~w
  ; Build %List struct.
  %list_struct_size_p = getelementptr %List, %List* null, i32 1
  %list_struct_size = ptrtoint %List* %list_struct_size_p to i64
  %list_mem = call i8* @malloc(i64 %list_struct_size)
  %list_ptr = bitcast i8* %list_mem to %List*
  %list_len_ptr = getelementptr %List, %List* %list_ptr, i32 0, i32 0
  store i32 ~w, i32* %list_len_ptr
  %list_elems_ptr = getelementptr %List, %List* %list_ptr, i32 0, i32 1
  store %Value* %elems, %Value** %list_elems_ptr

  ; A1 = List value (tag=4, payload=ptr).
  %list_i64 = ptrtoint %List* %list_ptr to i64
  %a1_0 = insertvalue %Value undef, i32 4, 0
  %a1 = insertvalue %Value %a1_0, i64 %list_i64, 1

  ; A2 = unbound (tag=6).
  %a2_0 = insertvalue %Value undef, i32 6, 0
  %a2 = insertvalue %Value %a2_0, i64 0, 1

  %vm = call %WamState* @wam_state_new(
      %Instruction* getelementptr ([~w x %Instruction], [~w x %Instruction]* @module_code, i32 0, i32 0),
      i32 ~w,
      i32* getelementptr ([~w x i32], [~w x i32]* @module_labels, i32 0, i32 0),
      i32 0)
  call void @wam_set_reg(%WamState* %vm, i32 0, %Value %a1)
  call void @wam_set_reg(%WamState* %vm, i32 1, %Value %a2)
  %ok = call i1 @run_loop(%WamState* %vm)
  br i1 %ok, label %count_entry, label %no_results

count_entry:
  br label %count_loop

count_loop:
  %count = phi i32 [1, %count_entry], [%count_inc, %got_next]
  %bt_ok = call i1 @backtrack(%WamState* %vm)
  br i1 %bt_ok, label %got_next, label %done

got_next:
  %count_inc = add i32 %count, 1
  br label %count_loop

done:
  ret i32 %count

no_results:
  ret i32 0
}
',
        [NumElems, ElemBytes, ElemStores,
         NumElems,
         IC, IC, IC, LC, LC]).

build_elem_stores(0, _, "") :- !.
build_elem_stores(N, Idx, Stores) :-
    N > 0,
    format(atom(Store),
'  %e~w = getelementptr %Value, %Value* %elems, i64 ~w
  %e~w_tag = getelementptr %Value, %Value* %e~w, i32 0, i32 0
  store i32 0, i32* %e~w_tag
  %e~w_pay = getelementptr %Value, %Value* %e~w, i32 0, i32 1
  store i64 ~w, i64* %e~w_pay
',
        [Idx, Idx, Idx, Idx, Idx, Idx, Idx, Idx, Idx]),
    NextN is N - 1,
    NextIdx is Idx + 1,
    build_elem_stores(NextN, NextIdx, RestStores),
    atom_concat(Store, RestStores, Stores).

test_execution :-
    format('--- list_suffix2 execution ---~n'),
    ( process_which('clang'), process_which('llc')
    -> run_suffix_case('3-elem list', 3, 4),
       run_suffix_case('1-elem list', 1, 2),
       run_suffix_case('empty list',  0, 1)
    ;  format('  SKIP: clang or llc not found~n')
    ).

process_which(Tool) :-
    catch(
        ( process_create(path(which), [Tool],
              [stdout(pipe(Out)), stderr(null), process(PID)]),
          read_string(Out, _, _), close(Out),
          process_wait(PID, exit(0))
        ), _, fail).

test_all :-
    catch(test_autodetect, E1,
        format('  ERROR: ~w~n', [E1])),
    catch(test_execution, E2,
        format('  ERROR: ~w~n', [E2])).

:- initialization(test_all, main).
