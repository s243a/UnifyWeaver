:- encoding(utf8).
:- use_module(library(plunit)).
:- use_module(library(filesex), [make_directory_path/1, delete_directory_and_contents/1]).
:- use_module(library(process)).
:- use_module('../src/unifyweaver/targets/wam_kotlin_target').
:- use_module('../src/unifyweaver/core/target_registry').

:- dynamic kt_guard/2.
:- dynamic kt_fact/2.
:- dynamic kt_same/2.
:- dynamic kt_eq/2.
:- dynamic kt_wrap/2.
:- dynamic kt_make_list/3.
:- dynamic kt_match_pair/3.
:- dynamic kt_color/1.
:- dynamic kt_parent/2.
:- dynamic kt_grandparent/2.
:- dynamic kt_edge/2.
:- dynamic kt_two_step/2.
:- dynamic kt_chain/2.

:- begin_tests(wam_kotlin_target).

gradle_available :-
    \+ current_prolog_flag(windows, true),
    absolute_file_name(path(gradle), _Path, [access(execute), file_errors(fail)]).

clean_dir(Dir) :-
    (   exists_directory(Dir)
    ->  delete_directory_and_contents(Dir)
    ;   true
    ).

has_substring(Haystack, Needle) :-
    once(sub_string(Haystack, _, _, _, Needle)).

run_gradle(ProjectDir, Args, Stdout, Stderr, Status) :-
    setup_call_cleanup(
        process_create(path(gradle), Args,
                       [cwd(ProjectDir), stdout(pipe(Out)), stderr(pipe(Err)), process(PID)]),
        (   read_string(Out, _, Stdout),
            read_string(Err, _, Stderr),
            process_wait(PID, Status)
        ),
        (   close(Out), close(Err)
        )
    ).

test(safe_identifier_sanitizes_symbols) :-
    wam_kotlin_target:kotlin_safe_identifier('foo-bar/baz', Safe),
    assertion(Safe == 'foo_bar_baz').

test(atom_literal_preserves_quoted_numeric_atom) :-
    wam_kotlin_target:wam_instruction_to_kotlin_literal(get_constant("'5'", "A1"), Lit),
    assertion(has_substring(Lit, 'Value.Atom("5")')),
    assertion(has_substring(Lit, '"A1"')).

test(integer_literal_uses_value_intval) :-
    wam_kotlin_target:wam_instruction_to_kotlin_literal(put_integer("42", "A2"), Lit),
    assertion(has_substring(Lit, 'Value.IntVal(42L)')),
    assertion(has_substring(Lit, "Instruction(\"put_integer\",")).

test(wam_predicate_registrar_converts_labels_to_metadata, [nondet]) :-
    WamText = "demo/1:\nput_integer 42, A1\nproceed\n",
    wam_kotlin_target:compile_wam_predicate_to_kotlin(demo/1, WamText, [], Code),
    assertion(has_substring(Code, 'fun register_demo(program: WamProgram)')),
    assertion(has_substring(Code, "program.register(\"demo/1\",")),
    assertion(has_substring(Code, "Instruction(\"label\",")),
    assertion(has_substring(Code, "Instruction(\"put_integer\",")),
    assertion(\+ sub_string(Code, _, _, _, 'demo/1:')).

test(runtime_template_exposes_executable_wam_abi) :-
    wam_kotlin_target:compile_wam_runtime_to_kotlin([], Code),
    assertion(has_substring(Code, 'sealed class Value')),
    assertion(has_substring(Code, 'sealed class WamContext')),
    assertion(has_substring(Code, 'data class PredicateCode')),
    assertion(has_substring(Code, 'data class CallFrame')),
    assertion(has_substring(Code, 'data class EnvironmentFrame')),
    assertion(has_substring(Code, 'val environmentStack: MutableList<EnvironmentFrame>')),
    assertion(has_substring(Code, 'fun restoreChoicePoint(): Boolean')),
    assertion(has_substring(Code, 'fun enterPredicate(state: WamState, predicate: String')),
    assertion(has_substring(Code, 'fun unify(left: Value?, right: Value?): Boolean')),
    assertion(has_substring(Code, 'fun beginStructure(functor: String, register: String): Boolean')),
    assertion(has_substring(Code, '"get_structure"')),
    assertion(has_substring(Code, '"unify_variable"')),
    assertion(has_substring(Code, '"put_value", "put_unsafe_value"')),
    assertion(has_substring(Code, 'fun registerNative(key: String, fn: (WamState) -> Boolean)')),
    assertion(has_substring(Code, 'fun snapshotForNative(): WamNativeSnapshot')),
    assertion(has_substring(Code, 'fun kotlinLoGetConstant(state: WamState')),
    assertion(has_substring(Code, 'fun stateFromCliArgs(values: List<String>): WamState')).

test(interpreter_project_writes_gradle_and_sources, [nondet]) :-
    TmpDir = 'output/test_wam_kotlin_project',
    make_directory_path('output'),
    clean_dir(TmpDir),
    setup_call_cleanup(
        (   retractall(user:kt_fact(_, _)),
            assertz(user:kt_fact(alpha, beta))
        ),
        (   wam_kotlin_target:write_wam_kotlin_project([user:kt_fact/2], [emit_mode(interpreter)], TmpDir),
            assertion(exists_file('output/test_wam_kotlin_project/settings.gradle')),
            assertion(exists_file('output/test_wam_kotlin_project/build.gradle')),
            assertion(exists_file('output/test_wam_kotlin_project/src/main/kotlin/generated/wam/WamRuntime.kt')),
            assertion(exists_file('output/test_wam_kotlin_project/src/main/kotlin/generated/wam/Main.kt')),
            read_file_to_string('output/test_wam_kotlin_project/src/main/kotlin/generated/wam/Main.kt', Main, []),
            assertion(has_substring(Main, 'register_kt_fact(program)'))
        ),
        retractall(user:kt_fact(_, _))
    ).

test(generated_project_compiles_and_runs_fact_variable_and_terms, [condition(gradle_available), nondet]) :-
    TmpDir = 'output/test_wam_kotlin_gradle_fact',
    make_directory_path('output'),
    clean_dir(TmpDir),
    setup_call_cleanup(
        (   retractall(user:kt_fact(_, _)),
            retractall(user:kt_same(_, _)),
            retractall(user:kt_eq(_, _)),
            retractall(user:kt_wrap(_, _)),
            retractall(user:kt_make_list(_, _, _)),
            retractall(user:kt_match_pair(_, _, _)),
            retractall(user:kt_color(_)),
            retractall(user:kt_parent(_, _)),
            retractall(user:kt_grandparent(_, _)),
            retractall(user:kt_edge(_, _)),
            retractall(user:kt_two_step(_, _)),
            retractall(user:kt_chain(_, _)),
            assertz(user:kt_fact(alpha, beta)),
            assertz(user:kt_same(X, X)),
            assertz(user:(kt_eq(X, Y) :- Y = X)),
            assertz(user:kt_wrap(X, box(X))),
            assertz(user:kt_make_list(X, Y, [X, Y])),
            assertz(user:kt_match_pair(pair(X, Y), X, Y)),
            assertz(user:kt_color(red)),
            assertz(user:kt_color(blue)),
            assertz(user:kt_parent(alice, bob)),
            assertz(user:kt_parent(bob, carol)),
            assertz(user:(kt_grandparent(X, Z) :- kt_parent(X, Y), kt_parent(Y, Z))),
            assertz(user:kt_edge(a, b)),
            assertz(user:kt_edge(b, c)),
            assertz(user:kt_edge(c, d)),
            assertz(user:(kt_two_step(X, Z) :- kt_edge(X, Y), kt_edge(Y, Z))),
            assertz(user:(kt_chain(X, Z) :- kt_two_step(X, Y), kt_edge(Y, Z)))
        ),
        (   wam_kotlin_target:write_wam_kotlin_project(
                [ user:kt_fact/2,
                  user:kt_same/2,
                  user:kt_eq/2,
                  user:kt_wrap/2,
                  user:kt_make_list/3,
                  user:kt_match_pair/3,
                  user:kt_color/1,
                  user:kt_parent/2,
                  user:kt_grandparent/2,
                  user:kt_edge/2,
                  user:kt_two_step/2,
                  user:kt_chain/2
                ],
                [emit_mode(interpreter)], TmpDir),
            run_gradle(TmpDir, ['-q', 'compileKotlin'], _CompileOut, CompileErr, CompileStatus),
            assertion(CompileStatus == exit(0)),
            assertion(CompileErr == ""),
            run_gradle(TmpDir, ['-q', 'run', '--args=kt_fact/2 alpha beta'], FactOut, _FactErr, FactStatus),
            assertion(FactStatus == exit(0)),
            assertion(has_substring(FactOut, 'Ran kt_fact/2')),
            assertion(has_substring(FactOut, 'A1=Atom(name=alpha)')),
            assertion(has_substring(FactOut, 'A2=Atom(name=beta)')),
            run_gradle(TmpDir, ['-q', 'run', '--args=kt_same/2 alpha alpha'], SameOut, _SameErr, SameStatus),
            assertion(SameStatus == exit(0)),
            assertion(has_substring(SameOut, 'Ran kt_same/2')),
            run_gradle(TmpDir, ['-q', 'run', '--args=kt_eq/2 alpha alpha'], EqOut, _EqErr, EqStatus),
            assertion(EqStatus == exit(0)),
            assertion(has_substring(EqOut, 'Ran kt_eq/2')),
            run_gradle(TmpDir, ['-q', 'run', '--args=kt_wrap/2 alpha'], WrapOut, _WrapErr, WrapStatus),
            assertion(WrapStatus == exit(0)),
            assertion(has_substring(WrapOut, 'A2=Struct(functor=box/1, args=[Atom(name=alpha)])')),
            run_gradle(TmpDir, ['-q', 'run', '--args=kt_make_list/3 alpha beta'], ListOut, _ListErr, ListStatus),
            assertion(ListStatus == exit(0)),
            assertion(has_substring(ListOut, "A3=Struct(functor=[|]/2,")),
            assertion(has_substring(ListOut, 'Atom(name=beta)')),
            run_gradle(TmpDir, ['-q', 'run', '--args=kt_match_pair/3 pair(alpha,beta) alpha beta'], PairOut, _PairErr, PairStatus),
            assertion(PairStatus == exit(0)),
            assertion(has_substring(PairOut, 'Ran kt_match_pair/3')),
            run_gradle(TmpDir, ['-q', 'run', '--args=kt_color/1 blue'], ColorOut, _ColorErr, ColorStatus),
            assertion(ColorStatus == exit(0)),
            assertion(has_substring(ColorOut, 'Ran kt_color/1')),
            assertion(has_substring(ColorOut, 'A1=Atom(name=blue)')),
            run_gradle(TmpDir, ['-q', 'run', '--args=kt_grandparent/2 alice carol'], GrandOut, _GrandErr, GrandStatus),
            assertion(GrandStatus == exit(0)),
            assertion(has_substring(GrandOut, 'Ran kt_grandparent/2')),
            assertion(has_substring(GrandOut, 'X3=Atom(name=alice)')),
            assertion(has_substring(GrandOut, 'A2=Atom(name=carol)')),
            run_gradle(TmpDir, ['-q', 'run', '--args=kt_chain/2 a d'], ChainOut, _ChainErr, ChainStatus),
            assertion(ChainStatus == exit(0)),
            assertion(has_substring(ChainOut, 'Ran kt_chain/2')),
            assertion(has_substring(ChainOut, 'Atom(name=a)')),
            assertion(has_substring(ChainOut, 'Atom(name=d)'))
        ),
        (   retractall(user:kt_fact(_, _)),
            retractall(user:kt_same(_, _)),
            retractall(user:kt_eq(_, _)),
            retractall(user:kt_wrap(_, _)),
            retractall(user:kt_make_list(_, _, _)),
            retractall(user:kt_match_pair(_, _, _)),
            retractall(user:kt_color(_)),
            retractall(user:kt_parent(_, _)),
            retractall(user:kt_grandparent(_, _)),
            retractall(user:kt_edge(_, _)),
            retractall(user:kt_two_step(_, _)),
            retractall(user:kt_chain(_, _))
        )
    ).

test(functions_mode_partitions_native_when_available) :-
    setup_call_cleanup(
        (   retractall(user:kt_fact(_, _)),
            assertz(user:kt_fact(alpha, beta))
        ),
        (   wam_kotlin_target:wam_kotlin_partition_predicates(functions, [user:kt_fact/2], Native, Wam, Failed),
            assertion(Native = [native(kt_fact/2, lowered(_PredKey, lowered_kt_fact_2, _))]),
            assertion(Wam = [wam(kt_fact/2, _)]),
            assertion(Failed == [])
        ),
        retractall(user:kt_fact(_, _))
    ).

test(functions_mode_emits_lowered_fun_and_register_native, [nondet]) :-
    TmpDir = 'output/test_wam_kotlin_functions_fact',
    make_directory_path('output'),
    clean_dir(TmpDir),
    setup_call_cleanup(
        (   retractall(user:kt_fact(_, _)),
            assertz(user:kt_fact(alpha, beta))
        ),
        (   wam_kotlin_target:write_wam_kotlin_project([user:kt_fact/2], [emit_mode(functions)], TmpDir),
            read_file_to_string('output/test_wam_kotlin_functions_fact/src/main/kotlin/generated/wam/Main.kt', Main, []),
            assertion(has_substring(Main, 'fun lowered_kt_fact_2(state: WamState): Boolean')),
            assertion(has_substring(Main, 'program.registerNative("kt_fact/2", ::lowered_kt_fact_2)')),
            assertion(has_substring(Main, 'register_kt_fact(program)')),
            assertion(\+ has_substring(Main, 'Native Kotlin lowering selected'))
        ),
        retractall(user:kt_fact(_, _))
    ).

test(functions_mode_gradle_runs_lowered_fact, [condition(gradle_available), nondet]) :-
    TmpDir = 'output/test_wam_kotlin_functions_gradle_fact',
    make_directory_path('output'),
    clean_dir(TmpDir),
    setup_call_cleanup(
        (   retractall(user:kt_fact(_, _)),
            assertz(user:kt_fact(alpha, beta))
        ),
        (   wam_kotlin_target:write_wam_kotlin_project([user:kt_fact/2], [emit_mode(functions)], TmpDir),
            run_gradle(TmpDir, ['-q', 'compileKotlin'], _CompileOut, CompileErr, CompileStatus),
            assertion(CompileStatus == exit(0)),
            assertion(CompileErr == ""),
            run_gradle(TmpDir, ['-q', 'run', '--args=kt_fact/2 alpha beta'], FactOut, _FactErr, FactStatus),
            assertion(FactStatus == exit(0)),
            assertion(has_substring(FactOut, 'Ran kt_fact/2')),
            assertion(has_substring(FactOut, 'A1=Atom(name=alpha)')),
            assertion(has_substring(FactOut, 'A2=Atom(name=beta)'))
        ),
        retractall(user:kt_fact(_, _))
    ).

% Regression guard for the silent-wrong-answer bug: write-mode structure/list
% construction was lowered incorrectly (unbound vars in the result). Such
% predicates must DECLINE lowering and stay on the correct bytecode interpreter.
test(functions_mode_declines_structure_lowering) :-
    setup_call_cleanup(
        (   retractall(user:kt_wrap(_, _)),
            assertz(user:(kt_wrap(X, wrap(X))))
        ),
        (   wam_kotlin_target:wam_kotlin_partition_predicates(functions, [user:kt_wrap/2], Native, Wam, Failed),
            assertion(Native == []),
            assertion(Wam = [wam(kt_wrap/2, _)]),
            assertion(Failed == [])
        ),
        retractall(user:kt_wrap(_, _))
    ).

% End-to-end: a list-building single-clause predicate under functions mode must
% produce the SAME bindings as the interpreter (it declines lowering and runs
% via the WAM fallback). Before the narrowing fix this returned [X1,X2] with
% unbound vars instead of [alpha,beta].
test(functions_mode_gradle_list_matches_interpreter, [condition(gradle_available), nondet]) :-
    TmpDir = 'output/test_wam_kotlin_functions_list',
    make_directory_path('output'),
    clean_dir(TmpDir),
    setup_call_cleanup(
        (   retractall(user:kt_make_list(_, _, _)),
            assertz(user:(kt_make_list(A, B, [A, B])))
        ),
        (   wam_kotlin_target:write_wam_kotlin_project([user:kt_make_list/3], [emit_mode(functions)], TmpDir),
            run_gradle(TmpDir, ['-q', 'compileKotlin'], _CompileOut, _CompileErr, CompileStatus),
            assertion(CompileStatus == exit(0)),
            run_gradle(TmpDir, ['-q', 'run', '--args=kt_make_list/3 alpha beta'], ListOut, _ListErr, ListStatus),
            assertion(ListStatus == exit(0)),
            assertion(has_substring(ListOut, 'A3=Struct(functor=[|]/2, args=[Atom(name=alpha), Struct(functor=[|]/2, args=[Atom(name=beta), Atom(name=[])])])'))
        ),
        retractall(user:kt_make_list(_, _, _))
    ).

test(registry_exposes_wam_kotlin_target, [nondet]) :-
    target_registry:target_exists(wam_kotlin),
    target_registry:target_family(wam_kotlin, jvm),
    target_registry:target_has_capability(wam_kotlin, wam),
    target_registry:target_module(wam_kotlin, wam_kotlin_target).

:- end_tests(wam_kotlin_target).
