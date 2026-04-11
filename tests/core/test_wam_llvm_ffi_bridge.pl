:- encoding(utf8).
% test_wam_llvm_ffi_bridge.pl
% Verifies M5.1: FFI bridge helpers for native kernels.
%
% M5.1 adds @wam_get_reg_tag, @wam_get_reg_payload, @wam_get_reg_double,
% @wam_set_reg_atom_id, @wam_set_reg_int, @wam_set_reg_double to the
% runtime template (state.ll.mustache). These are the primitives that
% foreign kernels will use to read and write WAM registers without
% constructing %Value structs inline.
%
% This test generates a full WAM LLVM module and a small native-style
% caller that exercises every helper, then pipes the result through
% llvm-as to validate the IR parses cleanly.

:- use_module('../../src/unifyweaver/targets/wam_llvm_target',
    [write_wam_llvm_project/3]).
:- use_module(library(process)).

:- dynamic color/1.
color(red).

test_helpers_defined_in_template :-
    format('--- bridge helpers are defined in state template ---~n'),
    tmp_file_stream(text, LLPath, Stream), close(Stream),
    write_wam_llvm_project([user:color/1], [module_name('ffi_test')], LLPath),
    read_file_to_string(LLPath, Src, []),
    forall(
        member(Name, [
            'define i32 @wam_get_reg_tag',
            'define i64 @wam_get_reg_payload',
            'define double @wam_get_reg_double',
            'define void @wam_set_reg_atom_id',
            'define void @wam_set_reg_int',
            'define void @wam_set_reg_double'
        ]),
        ( sub_string(Src, _, _, _, Name)
        -> format('  PASS: ~w present~n', [Name])
        ;  format('  FAIL: ~w missing~n', [Name]),
           throw(helper_missing(Name))
        )),
    catch(delete_file(LLPath), _, true).

test_bridge_module_validates :-
    format('--- llvm-as accepts module using bridge helpers ---~n'),
    ( process_which('llvm-as')
    -> test_bridge_llvm_as
    ;  format('  SKIP: llvm-as not found on PATH~n')
    ).

test_bridge_llvm_as :-
    tmp_file_stream(text, LLPath, Stream), close(Stream),
    write_wam_llvm_project([user:color/1], [module_name('ffi_test')], LLPath),
    % Append a small kernel-style function that exercises each helper.
    % This is the shape a real M5 native kernel will take.
    CallerIR = '
define i1 @demo_kernel(%WamState* %vm) {
entry:
  ; Read register 0 tag + payload.
  %tag0 = call i32 @wam_get_reg_tag(%WamState* %vm, i32 0)
  %pay0 = call i64 @wam_get_reg_payload(%WamState* %vm, i32 0)

  ; Read register 1 as a double.
  %d1 = call double @wam_get_reg_double(%WamState* %vm, i32 1)

  ; Write atom ID 42 into register 2.
  call void @wam_set_reg_atom_id(%WamState* %vm, i32 2, i64 42)

  ; Write integer 17 into register 3.
  call void @wam_set_reg_int(%WamState* %vm, i32 3, i64 17)

  ; Write double 3.14 into register 4.
  call void @wam_set_reg_double(%WamState* %vm, i32 4, double 3.14)

  ret i1 true
}
',
    setup_call_cleanup(
        open(LLPath, append, Out),
        write(Out, CallerIR),
        close(Out)),
    format('  Wrote module: ~w~n', [LLPath]),
    atom_concat(LLPath, '.bc', BCPath),
    format(atom(Cmd), 'llvm-as ~w -o ~w 2>&1', [LLPath, BCPath]),
    shell(Cmd, Exit),
    ( Exit == 0
    -> format('  PASS: llvm-as accepted module with bridge-helper caller~n')
    ;  format('  FAIL: llvm-as exit=~w~n', [Exit])
    ),
    catch(delete_file(LLPath), _, true),
    catch(delete_file(BCPath), _, true).

process_which(Tool) :-
    catch(
        ( process_create(path(which), [Tool],
              [stdout(pipe(Out)), stderr(null), process(PID)]),
          read_string(Out, _, _),
          close(Out),
          process_wait(PID, exit(0))
        ),
        _,
        fail).

test_all :-
    test_helpers_defined_in_template,
    catch(test_bridge_module_validates, E,
        format('  ERROR in llvm-as test: ~w~n', [E])).

:- initialization(test_all, main).
