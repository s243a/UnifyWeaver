:- encoding(utf8).
% test_wam_llvm_wasm_validation.pl
% Validates that the native LLVM IR module (with wasm32 triple) passes
% llvm-as and can compile to a wasm32 object via llc. This verifies
% IR correctness for WASM without requiring the full WASM type templates
% (which diverged from the native templates during M2+).
%
% Tests:
%   1. llvm-as accepts the module with wasm32-unknown-unknown triple.
%   2. llc --march=wasm32 produces a .o object file.
%   3. @wam_cleanup is present in the generated IR.

:- use_module('../../src/unifyweaver/targets/wam_llvm_target',
    [write_wam_llvm_project/3,
     clear_llvm_foreign_kernel_specs/0]).
:- use_module(library(process)).
:- use_module(library(readutil)).

:- dynamic wasm_dummy/1.
wasm_dummy(x).

test_wasm_validation :-
    format('--- WASM IR validation ---~n'),
    clear_llvm_foreign_kernel_specs,
    tmp_file_stream(text, LLPath, Stream), close(Stream),
    write_wam_llvm_project(
        [user:wasm_dummy/1],
        [ module_name('wasm_val'),
          target_triple('wasm32-unknown-unknown'),
          target_datalayout('e-m:e-p:32:32-i64:64-n32:64-S128')
        ],
        LLPath),
    read_file_to_string(LLPath, Src, []),

    % Test 1: @wam_cleanup present
    ( sub_string(Src, _, _, _, '@wam_cleanup')
    -> format('  PASS: @wam_cleanup present in IR~n')
    ;  format('  FAIL: @wam_cleanup missing~n'),
       throw(missing_cleanup)
    ),

    % Test 2: llvm-as accepts the module
    format(atom(AsmCmd), 'llvm-as ~w -o /dev/null 2>&1', [LLPath]),
    shell(AsmCmd, AsmExit),
    ( AsmExit =:= 0
    -> format('  PASS: llvm-as accepted wasm32 module~n')
    ;  format('  FAIL: llvm-as rejected wasm32 module (exit=~w)~n', [AsmExit]),
       throw(llvm_as_failed)
    ),

    % Test 3: llc can produce wasm32 object
    atom_concat(LLPath, '.o', OPath),
    format(atom(LlcCmd),
        'llc -march=wasm32 -mattr=+tail-call -filetype=obj ~w -o ~w 2>~w.llc.err',
        [LLPath, OPath, LLPath]),
    shell(LlcCmd, LlcExit),
    ( LlcExit =:= 0
    -> format('  PASS: llc produced wasm32 object~n')
    ;  atom_concat(LLPath, '.llc.err', ErrPath),
       ( catch(read_file_to_string(ErrPath, ErrStr, []), _, ErrStr = "")
       -> true ; true ),
       format('  FAIL: llc wasm32 failed (exit=~w)~n', [LlcExit]),
       ( ErrStr \== "" -> format('    ~w~n', [ErrStr]) ; true ),
       throw(llc_wasm_failed)
    ),

    catch(delete_file(LLPath), _, true),
    catch(delete_file(OPath), _, true),
    clear_llvm_foreign_kernel_specs.

test_all :-
    catch(test_wasm_validation, E,
        format('  ERROR: ~w~n', [E])).

:- initialization(test_all, main).
