:- encoding(utf8).
% test_wam_llvm_wasm_validation.pl
% Validates the full WASM pipeline: Prolog → LLVM IR → wasm32 object → .wasm binary.
%
% Tests:
%   1. @wam_cleanup present in the generated IR.
%   2. llvm-as accepts the module with wasm32 triple.
%   3. llc produces a wasm32 object file.
%   4. wasm-ld links a standalone .wasm binary.
%   5. WASM module contains malloc/free definitions (not declarations).
%   6. Foreign kernel BFS compiles to wasm32 end-to-end.

:- use_module('../../src/unifyweaver/targets/wam_llvm_target',
    [write_wam_llvm_project/3,
     clear_llvm_foreign_kernel_specs/0]).
:- use_module(library(process)).
:- use_module(library(readutil)).

:- dynamic wasm_dummy/1.
wasm_dummy(x).

:- dynamic wasm_edge/2.
wasm_edge(a, b).
wasm_edge(b, c).

:- dynamic wasm_reach/3.
wasm_reach(_, _, _) :- fail.

host_has_tool(Tool) :-
    catch(
        ( process_create(path(which), [Tool],
              [stdout(pipe(Out)), stderr(null), process(PID)]),
          read_string(Out, _, _), close(Out),
          process_wait(PID, exit(0))
        ), _, fail).

sub_atom_or_string(Atom, B, L, A, Sub) :-
    ( atom(Atom) -> sub_atom(Atom, B, L, A, Sub)
    ; string(Atom) -> sub_string(Atom, B, L, A, Sub)
    ).

test_basic_wasm_pipeline :-
    format('--- WASM basic pipeline ---~n'),
    clear_llvm_foreign_kernel_specs,
    tmp_file_stream(text, LLPath, Stream), close(Stream),
    write_wam_llvm_project(
        [user:wasm_dummy/1],
        [ module_name('wasm_basic'),
          target_triple('wasm32-unknown-unknown'),
          target_datalayout('e-m:e-p:32:32-i64:64-n32:64-S128')
        ],
        LLPath),
    read_file_to_string(LLPath, Src, []),

    % Test 1: @wam_cleanup present
    ( sub_atom_or_string(Src, _, _, _, '@wam_cleanup')
    -> format('  PASS: @wam_cleanup present~n')
    ;  format('  FAIL: @wam_cleanup missing~n'), throw(fail)
    ),

    % Test 2: malloc is defined (not just declared)
    ( sub_atom_or_string(Src, _, _, _, 'define i8* @malloc')
    -> format('  PASS: @malloc defined (not declared)~n')
    ;  format('  FAIL: @malloc not defined for wasm32~n'), throw(fail)
    ),

    % Test 3: llvm-as
    format(atom(AsmCmd), 'llvm-as ~w -o /dev/null 2>&1', [LLPath]),
    shell(AsmCmd, AsmExit),
    ( AsmExit =:= 0
    -> format('  PASS: llvm-as accepted wasm32 module~n')
    ;  format('  FAIL: llvm-as rejected (exit=~w)~n', [AsmExit]), throw(fail)
    ),

    % Test 4: llc to wasm32 object
    atom_concat(LLPath, '.o', OPath),
    format(atom(LlcCmd),
        'llc -march=wasm32 -mattr=+tail-call -filetype=obj ~w -o ~w 2>~w.llc.err',
        [LLPath, OPath, LLPath]),
    shell(LlcCmd, LlcExit),
    ( LlcExit =:= 0
    -> format('  PASS: llc produced wasm32 object~n')
    ;  format('  FAIL: llc wasm32 failed (exit=~w)~n', [LlcExit]), throw(fail)
    ),

    % Test 5: wasm-ld to standalone .wasm
    atom_concat(LLPath, '.wasm', WasmPath),
    format(atom(LdCmd),
        'wasm-ld --no-entry --export-all ~w -o ~w 2>~w.ld.err',
        [OPath, WasmPath, LLPath]),
    shell(LdCmd, LdExit),
    ( LdExit =:= 0
    -> format('  PASS: wasm-ld linked standalone .wasm~n')
    ;  format('  FAIL: wasm-ld failed (exit=~w)~n', [LdExit]), throw(fail)
    ),

    % Check .wasm file exists and has reasonable size.
    size_file(WasmPath, WasmSize),
    format('  INFO: .wasm binary size = ~w bytes~n', [WasmSize]),
    ( WasmSize > 1000
    -> format('  PASS: .wasm has reasonable size~n')
    ;  format('  FAIL: .wasm too small~n'), throw(fail)
    ),

    catch(delete_file(LLPath), _, true),
    catch(delete_file(OPath), _, true),
    catch(delete_file(WasmPath), _, true),
    clear_llvm_foreign_kernel_specs.

test_foreign_kernel_wasm :-
    format('--- WASM foreign kernel (BFS) pipeline ---~n'),
    clear_llvm_foreign_kernel_specs,
    tmp_file_stream(text, LLPath, Stream), close(Stream),
    write_wam_llvm_project(
        [user:wasm_reach/3],
        [ module_name('wasm_bfs'),
          target_triple('wasm32-unknown-unknown'),
          target_datalayout('e-m:e-p:32:32-i64:64-n32:64-S128'),
          foreign_predicates([
              wasm_reach/3 - transitive_distance3 - [edge_pred(wasm_edge/2)]
          ])
        ],
        LLPath),
    read_file_to_string(LLPath, Src, []),

    % Verify BFS kernel is in the module.
    ( sub_atom_or_string(Src, _, _, _, '@wam_bfs_atom_distance')
    -> format('  PASS: BFS kernel present in IR~n')
    ;  format('  FAIL: BFS kernel missing~n'), throw(fail)
    ),

    % Verify edge table present.
    ( sub_atom_or_string(Src, _, _, _, 'td3_inst_wasm_reach_0_edges')
    -> format('  PASS: edge table emitted~n')
    ;  format('  FAIL: edge table missing~n'), throw(fail)
    ),

    % Full pipeline: llvm-as → llc → wasm-ld
    atom_concat(LLPath, '.o', OPath),
    atom_concat(LLPath, '.wasm', WasmPath),
    format(atom(LlcCmd),
        'llc -march=wasm32 -mattr=+tail-call -filetype=obj ~w -o ~w 2>/dev/null',
        [LLPath, OPath]),
    shell(LlcCmd, LlcExit),
    ( LlcExit =:= 0
    -> format('  PASS: llc compiled BFS module to wasm32~n')
    ;  format('  FAIL: llc failed (exit=~w)~n', [LlcExit]), throw(fail)
    ),

    format(atom(LdCmd),
        'wasm-ld --no-entry --export-all ~w -o ~w 2>/dev/null',
        [OPath, WasmPath]),
    shell(LdCmd, LdExit),
    ( LdExit =:= 0
    -> format('  PASS: wasm-ld linked BFS .wasm binary~n')
    ;  format('  FAIL: wasm-ld failed (exit=~w)~n', [LdExit]), throw(fail)
    ),

    size_file(WasmPath, WasmSize),
    format('  INFO: BFS .wasm size = ~w bytes~n', [WasmSize]),

    catch(delete_file(LLPath), _, true),
    catch(delete_file(OPath), _, true),
    catch(delete_file(WasmPath), _, true),
    clear_llvm_foreign_kernel_specs.

test_all :-
    ( host_has_tool('llc'), host_has_tool('wasm-ld')
    -> catch(test_basic_wasm_pipeline, E1,
           format('  ERROR: ~w~n', [E1])),
       catch(test_foreign_kernel_wasm, E2,
           format('  ERROR: ~w~n', [E2]))
    ;  format('  SKIP: llc or wasm-ld not found~n')
    ).

:- initialization(test_all, main).
