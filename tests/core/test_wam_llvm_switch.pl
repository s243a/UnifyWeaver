:- encoding(utf8).
% test_wam_llvm_switch.pl
% Verifies that the LLVM WAM target can compile predicates using
% switch_on_constant indexing without producing invalid IR.

:- use_module('../../src/unifyweaver/targets/wam_target', [compile_predicate_to_wam/3]).
:- use_module('../../src/unifyweaver/targets/wam_llvm_target',
    [compile_wam_predicate_to_llvm/4, write_wam_llvm_project/3]).

% --- Test fixture: an indexed predicate ---
:- dynamic color/2.
color(red, 1).
color(green, 2).
color(blue, 3).

test_switch_on_constant_compiles :-
    format('--- switch_on_constant compilation ---~n'),
    % Compile the predicate to WAM text first
    compile_predicate_to_wam(user:color/2, [], WamCode),
    format('  WAM code generated (~w chars)~n', [WamCode]),
    % Ensure WAM actually contains a switch_on_constant
    ( sub_atom_or_string(WamCode, _, _, _, 'switch_on_constant')
    -> format('  PASS: WAM contains switch_on_constant~n')
    ;  format('  FAIL: WAM does not contain switch_on_constant~n'),
       throw(missing_switch_on_constant)
    ),
    % Now compile the WAM to LLVM IR
    compile_wam_predicate_to_llvm(color/2, WamCode, [], LLVMCode),
    format('  LLVM code generated (~w chars)~n', [LLVMCode]),
    % Check: no stub comments should remain
    ( sub_atom_or_string(LLVMCode, _, _, _, 'handled via labels')
    -> format('  FAIL: LLVM output still contains stub comment~n'),
       throw(unresolved_stub)
    ;  format('  PASS: no stub comments in LLVM output~n')
    ),
    % Check: the switch table global is emitted
    ( sub_atom_or_string(LLVMCode, _, _, _, 'SwitchEntry')
    -> format('  PASS: LLVM output contains %SwitchEntry references~n')
    ;  format('  FAIL: LLVM output missing %SwitchEntry~n'),
       throw(missing_switch_entry)
    ),
    % Check: the switch_on_constant instruction (tag 25) is emitted
    ( sub_atom_or_string(LLVMCode, _, _, _, 'i32 25')
    -> format('  PASS: LLVM output contains tag 25 (switch_on_constant)~n')
    ;  format('  FAIL: LLVM output missing tag 25~n'),
       throw(missing_switch_tag)
    ),
    % Check: uses ptrtoint to reference the switch table
    ( sub_atom_or_string(LLVMCode, _, _, _, 'ptrtoint')
    -> format('  PASS: LLVM output uses ptrtoint for table reference~n')
    ;  format('  FAIL: LLVM output missing ptrtoint~n'),
       throw(missing_ptrtoint)
    ),
    % Check: the table global definition is present
    ( sub_atom_or_string(LLVMCode, _, _, _, 'color_switch_0')
    -> format('  PASS: switch table global @color_switch_0 emitted~n')
    ;  format('  FAIL: switch table global missing~n'),
       throw(missing_table_global)
    ).

% Portable substring check (works on both atoms and strings).
sub_atom_or_string(Haystack, Before, Length, After, Needle) :-
    ( atom(Haystack) -> sub_atom(Haystack, Before, Length, After, Needle)
    ; string(Haystack) -> sub_string(Haystack, Before, Length, After, Needle)
    ; atom_string(Atom, Haystack), sub_atom(Atom, Before, Length, After, Needle)
    ).

% Optional: run the full module through llvm-as if the tool is available.
test_full_module_llvm_as :-
    format('--- Full module validates with llvm-as ---~n'),
    ( process_which(llvm_as_path)
    -> test_full_module_validates
    ;  format('  SKIP: llvm-as not found on PATH~n')
    ).

process_which(_) :-
    catch(
        ( process_create(path(which), ['llvm-as'],
              [stdout(pipe(Out)), stderr(null), process(PID)]),
          read_string(Out, _, _),
          close(Out),
          process_wait(PID, exit(0))
        ),
        _,
        fail).

test_full_module_validates :-
    tmp_file_stream(text, LLPath, Stream), close(Stream),
    write_wam_llvm_project([user:color/2], [module_name('test_mod')], LLPath),
    format('  Wrote: ~w~n', [LLPath]),
    atom_concat(LLPath, '.bc', BCPath),
    format(atom(Cmd), 'llvm-as ~w -o ~w 2>&1', [LLPath, BCPath]),
    shell(Cmd, Exit),
    ( Exit == 0
    -> format('  PASS: llvm-as accepted the generated IR~n')
    ;  format('  FAIL: llvm-as exit=~w~n', [Exit])
    ),
    catch(delete_file(LLPath), _, true),
    catch(delete_file(BCPath), _, true).

:- use_module(library(process)).

test_all :-
    test_switch_on_constant_compiles,
    catch(test_full_module_llvm_as, E,
        format('  ERROR running llvm-as test: ~w~n', [E])).

:- initialization(test_all, main).
