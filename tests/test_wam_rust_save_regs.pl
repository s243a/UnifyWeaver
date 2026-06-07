:- encoding(utf8).
% Regression test for save_regs bug.
%
% Before this fix, save_regs only saved A registers (indices 0..99)
% but restore_regs cleared both A and X registers (0..199). This
% destroyed X-register state across choice points like BeginAggregate
% pushed mid-body, causing aggregate results to never be bound to
% their result register.
%
% Usage: swipl -q -g run_tests -t halt tests/test_wam_rust_save_regs.pl

:- use_module('../src/unifyweaver/targets/wam_rust_target').
:- use_module('../src/unifyweaver/targets/wam_target').

:- dynamic test_failed/0.

pass(Test) :-
    format('[PASS] ~w~n', [Test]).

fail_test(Test, Reason) :-
    format('[FAIL] ~w: ~w~n', [Test, Reason]),
    (test_failed -> true ; assert(test_failed)).

test_save_regs_includes_x_registers :-
    Test = 'WAM-Rust: save_regs saves X registers (0..199) for mid-body CPs',
    read_file_to_string('templates/targets/rust_wam/state.rs.mustache', Code, []),
    (   sub_string(Code, _, _, _, "take(200)")
    ->  pass(Test)
    ;   fail_test(Test, 'save_regs still uses take(100); X registers will be lost on backtrack')
    ).

run_tests :-
    retractall(test_failed),
    format('~n=== WAM-Rust save_regs Regression Test ===~n', []),
    test_save_regs_includes_x_registers,
    format('~n', []),
    (   test_failed
    ->  format('=== FAILED ===~n', []), halt(1)
    ;   format('=== All save_regs tests passed ===~n', [])
    ).
