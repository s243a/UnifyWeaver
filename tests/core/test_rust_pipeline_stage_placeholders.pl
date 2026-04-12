:- module(test_rust_pipeline_stage_placeholders, [test_rust_pipeline_stage_placeholders/0]).

:- use_module(library(plunit)).
:- use_module('../../src/unifyweaver/targets/rust_target').

:- begin_tests(rust_pipeline_stage_placeholders).

test(unsupported_standard_stage_fails_explicitly) :-
    once(compile_rust_pipeline([unknown_stage/1], [pipeline_mode(generator)], Code)),
    once(sub_string(Code, _, _, _, 'fn stage_unknown_stage')),
    \+ sub_string(Code, _, _, _, 'TODO: Implement stage logic'),
    once(sub_string(Code, _, _, _, 'panic!("unsupported Rust pipeline stage: unknown_stage")')).

test(unsupported_enhanced_stage_fails_explicitly) :-
    once(compile_rust_enhanced_pipeline([unknown_stage/1], [validate(false)], Code)),
    once(sub_string(Code, _, _, _, 'fn unknown_stage')),
    \+ sub_string(Code, _, _, _, 'TODO: Implement based on predicate bindings'),
    once(sub_string(Code, _, _, _, 'panic!("unsupported Rust enhanced pipeline stage: unknown_stage/1")')).

test(standard_pipeline_can_wrap_stage_panics_for_fallback) :-
    once(compile_rust_pipeline([unknown_stage/1],
        [pipeline_name(test_pipe), pipeline_mode(generator), panic_fallback(input)],
        Code)),
    once(sub_string(Code, _, _, _, 'fn test_pipe')),
    once(sub_string(Code, _, _, _, 'fn safe_test_pipe')),
    once(sub_string(Code, _, _, _, 'std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| test_pipe(input)))')),
    once(sub_string(Code, _, _, _, 'Err(_) => fallback')),
    once(sub_string(Code, _, _, _, 'let output = safe_test_pipe(input);')).

:- end_tests(rust_pipeline_stage_placeholders).

test_rust_pipeline_stage_placeholders :-
    run_tests([rust_pipeline_stage_placeholders]).
