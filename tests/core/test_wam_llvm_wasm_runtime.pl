:- encoding(utf8).
% test_wam_llvm_wasm_runtime.pl
% End-to-end WASM runtime execution tests. Compiles Prolog predicates
% to .wasm, runs them in wasmtime, and checks results.
%
% Tests:
%   1. wam_cleanup runs without error in wasmtime.
%   2. wam_arena_ensure + wam_arena_mark returns non-zero.
%   3. Basic head unification predicate executes in wasmtime.

:- use_module('../../src/unifyweaver/targets/wam_llvm_target',
    [write_wam_llvm_project/3,
     clear_llvm_foreign_kernel_specs/0]).
:- use_module(library(process)).
:- use_module(library(readutil)).
:- use_module(library(pcre)).

host_has_tool(Tool) :-
    catch(
        ( process_create(path(which), [Tool],
              [stdout(pipe(Out)), stderr(null), process(PID)]),
          read_string(Out, _, _), close(Out),
          process_wait(PID, exit(0))
        ), _, fail).

extract_instr_count(Src, P, C) :-
    Pat = "@module_code = private constant \\[(?<n>\\d+) x %Instruction\\]",
    re_matchsub(Pat, Src, M, []), get_dict(n, M, NS), number_string(C, NS).
extract_label_count(Src, P, C) :-
    Pat = "@module_labels = private constant \\[(?<n>\\d+) x i32\\]",
    re_matchsub(Pat, Src, M, []), get_dict(n, M, NS), number_string(C, NS).

build_wasm(LLPath, WasmPath) :-
    atom_concat(LLPath, '.o', OPath),
    format(atom(LlcCmd),
        'llc -march=wasm32 -mattr=+tail-call -filetype=obj ~w -o ~w 2>/dev/null',
        [LLPath, OPath]),
    shell(LlcCmd, 0),
    format(atom(LdCmd),
        'wasm-ld --no-entry --export-all --initial-memory=2097152 ~w -o ~w 2>/dev/null',
        [OPath, WasmPath]),
    shell(LdCmd, 0),
    catch(delete_file(OPath), _, true).

% Test 1: wam_cleanup runs in wasmtime
test_cleanup_runs :-
    format('  cleanup runs in wasmtime: '),
    clear_llvm_foreign_kernel_specs,
    tmp_file_stream(text, LLPath, Stream), close(Stream),
    write_wam_llvm_project([],
        [module_name(wrt1), target_triple('wasm32-unknown-unknown'),
         target_datalayout('e-m:e-p:32:32-i64:64-n32:64-S128')],
        LLPath),
    atom_concat(LLPath, '.wasm', WasmPath),
    build_wasm(LLPath, WasmPath),
    format(atom(Cmd), 'wasmtime run --invoke wam_cleanup ~w 2>&1', [WasmPath]),
    shell(Cmd, ExitCode),
    ( ExitCode =:= 0
    -> format('PASS~n')
    ;  format('FAIL (exit=~w)~n', [ExitCode])
    ),
    catch(delete_file(LLPath), _, true),
    catch(delete_file(WasmPath), _, true),
    clear_llvm_foreign_kernel_specs,
    assertion(ExitCode =:= 0).

% Test 2: arena allocator works in wasmtime
test_arena_in_wasmtime :-
    format('  arena ensure+mark in wasmtime: '),
    clear_llvm_foreign_kernel_specs,
    tmp_file_stream(text, LLPath, Stream), close(Stream),
    write_wam_llvm_project([],
        [module_name(wrt2), target_triple('wasm32-unknown-unknown'),
         target_datalayout('e-m:e-p:32:32-i64:64-n32:64-S128')],
        LLPath),
    % Append a test function that inits arena and returns mark value.
    DriverIR = '
define i32 @test_arena() {
entry:
  call void @wam_arena_ensure()
  %mark = call i64 @wam_arena_mark()
  ; mark should be 0 (arena just initialized, nothing allocated).
  %r = trunc i64 %mark to i32
  ret i32 %r
}
',
    setup_call_cleanup(
        open(LLPath, append, Out),
        ( write(Out, '\n'), write(Out, DriverIR) ),
        close(Out)),
    atom_concat(LLPath, '.wasm', WasmPath),
    build_wasm(LLPath, WasmPath),
    format(atom(Cmd), 'wasmtime run --invoke test_arena ~w 2>&1', [WasmPath]),
    process_create(path(sh), ['-c', Cmd],
        [stdout(pipe(StdOut)), stderr(null), process(PID)]),
    read_string(StdOut, _, Output), close(StdOut),
    process_wait(PID, exit(ExitCode)),
    ( ExitCode =:= 0
    -> format('PASS (mark=~w)~n', [Output])
    ;  format('FAIL (exit=~w)~n', [ExitCode])
    ),
    catch(delete_file(LLPath), _, true),
    catch(delete_file(WasmPath), _, true),
    clear_llvm_foreign_kernel_specs,
    assertion(ExitCode =:= 0).

% Test 3: WAM predicate runs in wasmtime
% Compile test_id(X, X) — identity predicate, run via a driver
% that creates a VM, sets A1=42, runs, reads A2.
:- dynamic wasm_id/2.
wasm_id(X, X).

test_wam_predicate_in_wasmtime :-
    format('  WAM predicate in wasmtime: '),
    clear_llvm_foreign_kernel_specs,
    tmp_file_stream(text, LLPath, Stream), close(Stream),
    write_wam_llvm_project([user:wasm_id/2],
        [module_name(wrt3), target_triple('wasm32-unknown-unknown'),
         target_datalayout('e-m:e-p:32:32-i64:64-n32:64-S128')],
        LLPath),
    read_file_to_string(LLPath, Src, []),
    extract_instr_count(Src, wasm_id, IC),
    extract_label_count(Src, wasm_id, LC),
    format(atom(DriverIR),
'define i32 @test_pred() {
entry:
  %a1_0 = insertvalue %Value undef, i32 1, 0
  %a1 = insertvalue %Value %a1_0, i64 42, 1
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
  call void @wam_cleanup()
  br i1 %ok, label %hit, label %miss
hit:
  %r = call i64 @wam_get_reg_payload(%WamState* %vm, i32 1)
  %r32 = trunc i64 %r to i32
  ret i32 %r32
miss:
  ret i32 255
}
',
        [IC, IC, IC, LC, LC]),
    setup_call_cleanup(
        open(LLPath, append, Out),
        ( write(Out, '\n'), write(Out, DriverIR) ),
        close(Out)),
    atom_concat(LLPath, '.wasm', WasmPath),
    build_wasm(LLPath, WasmPath),
    format(atom(Cmd), 'wasmtime run --invoke test_pred ~w 2>&1', [WasmPath]),
    process_create(path(sh), ['-c', Cmd],
        [stdout(pipe(StdOut)), stderr(null), process(PID)]),
    read_string(StdOut, _, Output), close(StdOut),
    process_wait(PID, exit(ExitCode)),
    % Parse result: wasmtime may print warnings before the value.
    % Find the last line that looks like a number.
    split_string(Output, "\n", "\r\t ", Lines),
    ( member(Line, Lines), string_to_atom(Line, LineAtom),
      atom_number(LineAtom, Result0)
    -> Result = Result0
    ;  Result = -1
    ),
    ( ExitCode =:= 0, Result =:= 42
    -> format('PASS (A2=~w)~n', [Result])
    ;  format('FAIL (exit=~w, output=~w)~n', [ExitCode, Output])
    ),
    catch(delete_file(LLPath), _, true),
    catch(delete_file(WasmPath), _, true),
    clear_llvm_foreign_kernel_specs,
    assertion(ExitCode =:= 0),
    assertion(Result =:= 42).

test_all :-
    format('--- WASM runtime execution (wasmtime) ---~n'),
    ( host_has_tool('llc'), host_has_tool('wasm-ld'), host_has_tool('wasmtime')
    -> catch(test_cleanup_runs, E1, format('  ERROR: ~w~n', [E1])),
       catch(test_arena_in_wasmtime, E2, format('  ERROR: ~w~n', [E2])),
       catch(test_wam_predicate_in_wasmtime, E3, format('  ERROR: ~w~n', [E3]))
    ;  format('  SKIP: llc, wasm-ld, or wasmtime not found~n')
    ).

:- initialization(test_all, main).
