:- encoding(utf8).
:- use_module(library(plunit)).
:- use_module(library(filesex), [make_directory_path/1]).
:- use_module('../src/unifyweaver/targets/wam_kotlin_target').
:- use_module('../src/unifyweaver/core/target_registry').

:- dynamic kt_guard/2.
:- dynamic kt_fact/2.

:- begin_tests(wam_kotlin_target).

test(safe_identifier_sanitizes_symbols) :-
    wam_kotlin_target:kotlin_safe_identifier('foo-bar/baz', Safe),
    assertion(Safe == 'foo_bar_baz').

test(atom_literal_preserves_quoted_numeric_atom) :-
    wam_kotlin_target:wam_instruction_to_kotlin_literal(get_constant("'5'", "A1"), Lit),
    assertion(sub_string(Lit, _, _, _, 'Value.Atom("5")')),
    assertion(sub_string(Lit, _, _, _, '"A1"')).

test(integer_literal_uses_value_intval) :-
    wam_kotlin_target:wam_instruction_to_kotlin_literal(put_integer("42", "A2"), Lit),
    assertion(sub_string(Lit, _, _, _, 'Value.IntVal(42L)')),
    assertion(sub_string(Lit, _, _, _, 'Instruction("put_integer"')).

test(wam_predicate_registrar_skips_labels) :-
    WamText = "demo/1:\nput_integer 42, A1\nproceed\n",
    wam_kotlin_target:compile_wam_predicate_to_kotlin(demo/1, WamText, [], Code),
    assertion(sub_string(Code, _, _, _, 'fun register_demo(program: WamProgram)')),
    assertion(sub_string(Code, _, _, _, 'program.register("demo/1"')),
    assertion(sub_string(Code, _, _, _, 'Instruction("put_integer"')),
    assertion(\+ sub_string(Code, _, _, _, 'demo/1:')).

test(runtime_template_exposes_wam_abi) :-
    wam_kotlin_target:compile_wam_runtime_to_kotlin([], Code),
    assertion(sub_string(Code, _, _, _, 'sealed class Value')),
    assertion(sub_string(Code, _, _, _, 'data class Instruction')),
    assertion(sub_string(Code, _, _, _, 'class WamProgram')),
    assertion(sub_string(Code, _, _, _, 'class WamRuntime')).

test(interpreter_project_writes_gradle_and_sources) :-
    TmpDir = 'output/test_wam_kotlin_project',
    make_directory_path('output'),
    retractall(user:kt_fact(_, _)),
    assertz(user:kt_fact(alpha, beta)),
    wam_kotlin_target:write_wam_kotlin_project([user:kt_fact/2], [emit_mode(interpreter)], TmpDir),
    assertion(exists_file('output/test_wam_kotlin_project/settings.gradle')),
    assertion(exists_file('output/test_wam_kotlin_project/build.gradle')),
    assertion(exists_file('output/test_wam_kotlin_project/src/main/kotlin/generated/wam/WamRuntime.kt')),
    assertion(exists_file('output/test_wam_kotlin_project/src/main/kotlin/generated/wam/Main.kt')),
    read_file_to_string('output/test_wam_kotlin_project/src/main/kotlin/generated/wam/Main.kt', Main, []),
    assertion(sub_string(Main, _, _, _, 'register_kt_fact(program)')),
    retractall(user:kt_fact(_, _)).

test(functions_mode_partitions_native_when_available) :-
    retractall(user:kt_guard(_, _)),
    assertz(user:(kt_guard(X, yes) :- X > 0)),
    wam_kotlin_target:wam_kotlin_partition_predicates(functions, [user:kt_guard/2], Native, Wam, Failed),
    assertion(Native = [native(kt_guard/2, _)]),
    assertion(Wam == []),
    assertion(Failed == []),
    retractall(user:kt_guard(_, _)).

test(registry_exposes_wam_kotlin_target) :-
    target_registry:target_exists(wam_kotlin),
    target_registry:target_family(wam_kotlin, jvm),
    target_registry:target_has_capability(wam_kotlin, wam),
    target_registry:target_module(wam_kotlin, wam_kotlin_target).

:- end_tests(wam_kotlin_target).
