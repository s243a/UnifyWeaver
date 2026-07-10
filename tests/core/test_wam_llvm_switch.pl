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
:- use_module(library(pcre)).
:- use_module(library(readutil)).

% --- Quoted-key dispatch EXECUTION regression ------------------------------
% The switch-entry parser used to split "key:label" at the FIRST colon and
% never unquoted writeq-style keys, so table entries for quoted atoms like
% '=:=' or '\==' sheared in half or interned their quote characters into
% the key -- the entry never matched the runtime atom and a strict switch
% silently FAILED the call (this binary exits 255 instead of 33 on the
% broken parser). Fixed by switch_entry_split/3 + switch_entry_unquote/2.
:- dynamic qk/2.
qk('=:=', 5).
qk('\\==', 21).
qk(plainkey, 7).
qk(other1, 1).
qk(other2, 2).
qk(other3, 3).
qk(other4, 4).
qk(other5, 6).
:- dynamic qkmain/1.
qkmain(R) :-
    qk('=:=', A),
    qk('\\==', B),
    qk(plainkey, C),
    R is A + B + C.                          % 5 + 21 + 7 = 33

test_quoted_key_dispatch_executes :-
    format('--- quoted-key switch dispatch executes ---~n'),
    ( clang_on_path
    -> run_qk_case
    ;  format('  SKIP: clang not found~n')
    ).

clang_on_path :-
    catch(
        ( process_create(path(clang), ['--version'],
              [stdout(null), stderr(null), process(PID)]),
          process_wait(PID, exit(0))
        ),
        _, fail).

qk_instr_count(Src, C) :-
    re_matchsub("@module_code = private constant \\[(?<n>\\d+) x %Instruction\\]",
                Src, M, []),
    get_dict(n, M, NS), number_string(C, NS).
qk_label_count(Src, C) :-
    re_matchsub("@module_labels = private constant \\[(?<n>\\d+) x i32\\]",
                Src, M, []),
    get_dict(n, M, NS), number_string(C, NS).

run_qk_case :-
    tmp_file_stream(text, LLPath, Stream), close(Stream),
    % qkmain first in the list -> its code starts at PC 0
    write_wam_llvm_project([user:qkmain/1, user:qk/2],
        [module_name('qk_exec')], LLPath),
    read_file_to_string(LLPath, Src, []),
    ( sub_atom_or_string(Src, _, _, _, 'qk_switch_')
    -> format('  PASS: qk switch table emitted~n')
    ;  format('  FAIL: qk switch table missing~n'),
       throw(missing_qk_switch_table)
    ),
    qk_instr_count(Src, IC),
    qk_label_count(Src, LC),
    % A1 = unbound result slot (tag 6); entry starts at PC 0 (qkmain).
    format(atom(DriverIR),
'define i32 @main() {
entry:
  %a1_0 = insertvalue %Value undef, i32 6, 0
  %a1 = insertvalue %Value %a1_0, i64 0, 1
  %vm = call %WamState* @wam_state_new(
      %Instruction* getelementptr ([~w x %Instruction], [~w x %Instruction]* @module_code, i32 0, i32 0),
      i32 ~w,
      i32* getelementptr ([~w x i32], [~w x i32]* @module_labels, i32 0, i32 0),
      i32 ~w)
  call void @wam_set_reg(%WamState* %vm, i32 0, %Value %a1)
  %ok = call i1 @run_loop(%WamState* %vm)
  br i1 %ok, label %hit, label %miss
hit:
  %r = call i64 @wam_get_reg_payload(%WamState* %vm, i32 0)
  %r32 = trunc i64 %r to i32
  ret i32 %r32
miss:
  ret i32 255
}
',
        [IC, IC, IC, LC, LC, LC]),
    setup_call_cleanup(
        open(LLPath, append, Out),
        ( write(Out, '\n'), write(Out, DriverIR) ),
        close(Out)),
    atom_concat(LLPath, '.out', BinPath),
    % -x ir: the tmp file has no .ll extension, so clang needs telling
    format(atom(ClangCmd), 'clang -w -O0 -x ir ~w -o ~w -lm 2>~w.clang.err',
        [LLPath, BinPath, LLPath]),
    shell(ClangCmd, ClangExit),
    ( ClangExit =\= 0
    -> format('  FAIL: clang exit=~w (see ~w.clang.err)~n', [ClangExit, LLPath]),
       throw(qk_clang_failed)
    ;  true
    ),
    shell(BinPath, ExitCode),
    catch(delete_file(LLPath), _, true),
    catch(delete_file(BinPath), _, true),
    ( ExitCode =:= 33
    -> format('  PASS: quoted-key dispatch returned 33~n')
    ;  format('  FAIL: quoted-key dispatch returned ~w (expected 33; 255 = the entry FAILED, the pre-fix symptom)~n',
           [ExitCode]),
       throw(quoted_key_dispatch_failed(ExitCode))
    ).

test_all :-
    test_switch_on_constant_compiles,
    test_quoted_key_dispatch_executes,
    catch(test_full_module_llvm_as, E,
        format('  ERROR running llvm-as test: ~w~n', [E])).

:- initialization(test_all, main).
