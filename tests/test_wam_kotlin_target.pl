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
:- dynamic kt_nested/3.
:- dynamic kt_match_pair/3.
:- dynamic kt_color/1.
:- dynamic kt_parent/2.
:- dynamic kt_grandparent/2.
:- dynamic kt_edge/2.
:- dynamic kt_two_step/2.
:- dynamic kt_chain/2.
:- dynamic kt_arith/1.
:- dynamic kt_member/2.
:- dynamic kt_mem_b/0.
:- dynamic kt_mem_c/0.
:- dynamic kt_mem_z/0.
:- dynamic kt_fib/2.
:- dynamic kt_fib_ok/0.
:- dynamic kt_fib_bad/0.

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
    assertion(has_substring(Code, 'fun tryRun(predicate: String, initialState: WamState')),
    assertion(has_substring(Code, 'fun functorName(functor: String): String')),
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

% Structure/list builders must LOWER under functions mode (Native partition),
% not decline. Regression guard for the silent-wrong-answer bug where a bare
% Kotlin `{ ... }` block around get_variable was a no-op lambda.
test(functions_mode_lowers_structure_builders) :-
    setup_call_cleanup(
        (   retractall(user:kt_wrap(_, _)),
            retractall(user:kt_make_list(_, _, _)),
            assertz(user:(kt_wrap(X, wrap(X)))),
            assertz(user:(kt_make_list(A, B, [A, B])))
        ),
        (   wam_kotlin_target:wam_kotlin_partition_predicates(
                functions, [user:kt_wrap/2, user:kt_make_list/3], Native, Wam, Failed),
            assertion(Failed == []),
            assertion(memberchk(native(kt_wrap/2, _), Native)),
            assertion(memberchk(native(kt_make_list/3, _), Native)),
            assertion(memberchk(wam(kt_wrap/2, _), Wam)),
            assertion(memberchk(wam(kt_make_list/3, _), Wam)),
            memberchk(native(kt_wrap/2, lowered(_, _, WrapCode)), Native),
            assertion(has_substring(WrapCode, 'fun lowered_kt_wrap_2')),
            assertion(has_substring(WrapCode, 'beginStructure')),
            memberchk(native(kt_make_list/3, lowered(_, _, ListCode)), Native),
            assertion(has_substring(ListCode, 'fun lowered_kt_make_list_3')),
            assertion(has_substring(ListCode, 'run {'))
        ),
        (   retractall(user:kt_wrap(_, _)),
            retractall(user:kt_make_list(_, _, _))
        )
    ).

% End-to-end: list builder under functions mode must LOWER and produce the
% SAME bindings as emit_mode(interpreter). Before the run{} fix this returned
% [X1,X2] unbound vars while still returning true (no interpreter fallback).
test(functions_mode_gradle_list_matches_interpreter, [condition(gradle_available), nondet]) :-
    TmpDir = 'output/test_wam_kotlin_functions_list',
    make_directory_path('output'),
    clean_dir(TmpDir),
    Expected = 'A3=Struct(functor=[|]/2, args=[Atom(name=alpha), Struct(functor=[|]/2, args=[Atom(name=beta), Atom(name=[])])])',
    setup_call_cleanup(
        (   retractall(user:kt_make_list(_, _, _)),
            assertz(user:(kt_make_list(A, B, [A, B])))
        ),
        (   wam_kotlin_target:write_wam_kotlin_project([user:kt_make_list/3], [emit_mode(interpreter)], TmpDir),
            run_gradle(TmpDir, ['-q', 'run', '--args=kt_make_list/3 alpha beta'], InterpOut, _InterpErr, InterpStatus),
            assertion(InterpStatus == exit(0)),
            assertion(has_substring(InterpOut, Expected)),
            clean_dir(TmpDir),
            wam_kotlin_target:write_wam_kotlin_project([user:kt_make_list/3], [emit_mode(functions)], TmpDir),
            assertion(exists_file('output/test_wam_kotlin_functions_list/src/main/kotlin/generated/wam/Main.kt')),
            read_file_to_string('output/test_wam_kotlin_functions_list/src/main/kotlin/generated/wam/Main.kt', Main, []),
            assertion(has_substring(Main, 'fun lowered_kt_make_list_3')),
            assertion(has_substring(Main, 'registerNative("kt_make_list/3"')),
            run_gradle(TmpDir, ['-q', 'compileKotlin'], _CompileOut, _CompileErr, CompileStatus),
            assertion(CompileStatus == exit(0)),
            run_gradle(TmpDir, ['-q', 'run', '--args=kt_make_list/3 alpha beta'], ListOut, _ListErr, ListStatus),
            assertion(ListStatus == exit(0)),
            assertion(has_substring(ListOut, Expected))
        ),
        retractall(user:kt_make_list(_, _, _))
    ).

% Structure builder: p(X, wrap(X)) lowers and matches interpreter bindings.
test(functions_mode_gradle_wrap_matches_interpreter, [condition(gradle_available), nondet]) :-
    TmpDir = 'output/test_wam_kotlin_functions_wrap',
    make_directory_path('output'),
    clean_dir(TmpDir),
    Expected = 'A2=Struct(functor=wrap/1, args=[Atom(name=alpha)])',
    setup_call_cleanup(
        (   retractall(user:kt_wrap(_, _)),
            assertz(user:(kt_wrap(X, wrap(X))))
        ),
        (   wam_kotlin_target:write_wam_kotlin_project([user:kt_wrap/2], [emit_mode(interpreter)], TmpDir),
            run_gradle(TmpDir, ['-q', 'run', '--args=kt_wrap/2 alpha'], InterpOut, _InterpErr, InterpStatus),
            assertion(InterpStatus == exit(0)),
            assertion(has_substring(InterpOut, Expected)),
            clean_dir(TmpDir),
            wam_kotlin_target:write_wam_kotlin_project([user:kt_wrap/2], [emit_mode(functions)], TmpDir),
            read_file_to_string('output/test_wam_kotlin_functions_wrap/src/main/kotlin/generated/wam/Main.kt', Main, []),
            assertion(has_substring(Main, 'fun lowered_kt_wrap_2')),
            assertion(has_substring(Main, 'registerNative("kt_wrap/2"')),
            run_gradle(TmpDir, ['-q', 'run', '--args=kt_wrap/2 alpha'], WrapOut, _WrapErr, WrapStatus),
            assertion(WrapStatus == exit(0)),
            assertion(has_substring(WrapOut, Expected))
        ),
        retractall(user:kt_wrap(_, _))
    ).

% Nested/mixed structure+list: p(X, Y, foo(X, [Y])) lowers == interpreter.
test(functions_mode_gradle_nested_matches_interpreter, [condition(gradle_available), nondet]) :-
    TmpDir = 'output/test_wam_kotlin_functions_nested',
    make_directory_path('output'),
    clean_dir(TmpDir),
    Expected = 'A3=Struct(functor=foo/2, args=[Atom(name=alpha), Struct(functor=[|]/2, args=[Atom(name=beta), Atom(name=[])])])',
    setup_call_cleanup(
        (   retractall(user:kt_nested(_, _, _)),
            assertz(user:(kt_nested(X, Y, foo(X, [Y]))))
        ),
        (   wam_kotlin_target:wam_kotlin_partition_predicates(functions, [user:kt_nested/3], Native, _Wam, Failed),
            assertion(Failed == []),
            assertion(memberchk(native(kt_nested/3, _), Native)),
            wam_kotlin_target:write_wam_kotlin_project([user:kt_nested/3], [emit_mode(interpreter)], TmpDir),
            run_gradle(TmpDir, ['-q', 'run', '--args=kt_nested/3 alpha beta'], InterpOut, _InterpErr, InterpStatus),
            assertion(InterpStatus == exit(0)),
            assertion(has_substring(InterpOut, Expected)),
            clean_dir(TmpDir),
            wam_kotlin_target:write_wam_kotlin_project([user:kt_nested/3], [emit_mode(functions)], TmpDir),
            read_file_to_string('output/test_wam_kotlin_functions_nested/src/main/kotlin/generated/wam/Main.kt', Main, []),
            assertion(has_substring(Main, 'fun lowered_kt_nested_3')),
            assertion(has_substring(Main, 'registerNative("kt_nested/3"')),
            run_gradle(TmpDir, ['-q', 'run', '--args=kt_nested/3 alpha beta'], NestedOut, _NestedErr, NestedStatus),
            assertion(NestedStatus == exit(0)),
            assertion(has_substring(NestedOut, Expected))
        ),
        retractall(user:kt_nested(_, _, _))
    ).

test(registry_exposes_wam_kotlin_target, [nondet]) :-
    target_registry:target_exists(wam_kotlin),
    target_registry:target_family(wam_kotlin, jvm),
    target_registry:target_has_capability(wam_kotlin, wam),
    target_registry:target_module(wam_kotlin, wam_kotlin_target).

test(conformance_main_prints_true_false, [condition(gradle_available), nondet]) :-
    TmpDir = 'output/test_wam_kotlin_conformance_main',
    make_directory_path('output'),
    clean_dir(TmpDir),
    setup_call_cleanup(
        (   retractall(user:kt_fact(_, _)),
            assertz(user:kt_fact(alpha, beta))
        ),
        (   wam_kotlin_target:write_wam_kotlin_project(
                [user:kt_fact/2],
                [emit_mode(functions), conformance_main(true)], TmpDir),
            read_file_to_string(
                'output/test_wam_kotlin_conformance_main/src/main/kotlin/generated/wam/Main.kt',
                Main, []),
            assertion(has_substring(Main, 'tryRun')),
            assertion(has_substring(Main, 'println(if (ok) "true" else "false")')),
            assertion(\+ has_substring(Main, 'Ran $predicate')),
            run_gradle(TmpDir, ['-q', 'run', '--args=kt_fact/2 alpha beta'],
                       OkOut, _OkErr, OkStatus),
            assertion(OkStatus == exit(0)),
            normalize_space(string(OkTrim), OkOut),
            assertion(OkTrim == "true"),
            run_gradle(TmpDir, ['-q', 'run', '--args=kt_fact/2 alpha gamma'],
                       BadOut, _BadErr, BadStatus),
            assertion(BadStatus == exit(0)),
            normalize_space(string(BadTrim), BadOut),
            assertion(BadTrim == "false")
        ),
        retractall(user:kt_fact(_, _))
    ).

% KT-ARITH-SLASH-FUNCTOR: integer division // (functor key "///2") must
% evaluate — a naive functor.split("/") yielded name="" and failed closed.
test(arith_slash_functor_integer_div, [condition(gradle_available), nondet]) :-
    TmpDir = 'output/test_wam_kotlin_arith_slash',
    make_directory_path('output'),
    clean_dir(TmpDir),
    setup_call_cleanup(
        (   retractall(user:kt_arith(_, _)),
            assertz(user:(kt_arith(R) :- A is 17 // 5, B is 17 mod 5, R is A + B))
        ),
        (   wam_kotlin_target:write_wam_kotlin_project(
                [user:kt_arith/1],
                [emit_mode(interpreter), conformance_main(true)], TmpDir),
            run_gradle(TmpDir, ['-q', 'run', '--args=kt_arith/1 5'], OkOut, _OkErr, OkStatus),
            assertion(OkStatus == exit(0)),
            normalize_space(string(OkTrim), OkOut),
            assertion(OkTrim == "true"),
            run_gradle(TmpDir, ['-q', 'run', '--args=kt_arith/1 4'], BadOut, _BadErr, BadStatus),
            assertion(BadStatus == exit(0)),
            normalize_space(string(BadTrim), BadOut),
            assertion(BadTrim == "false")
        ),
        retractall(user:kt_arith(_, _))
    ).

% KT-LIST-BACKTRACK: recursive member must survive backtracking over a
% heap-built list (CDR placeholders must not share the unify_variable Xn
% register name).
test(list_backtrack_member, [condition(gradle_available), nondet]) :-
    TmpDir = 'output/test_wam_kotlin_list_backtrack',
    make_directory_path('output'),
    clean_dir(TmpDir),
    setup_call_cleanup(
        (   retractall(user:kt_member(_, _)),
            retractall(user:kt_mem_b),
            retractall(user:kt_mem_c),
            retractall(user:kt_mem_z),
            assertz(user:kt_member(X, [X|_])),
            assertz(user:(kt_member(X, [_|T]) :- kt_member(X, T))),
            assertz(user:(kt_mem_b :- kt_member(b, [a,b,c]))),
            assertz(user:(kt_mem_c :- kt_member(c, [a,b,c]))),
            assertz(user:(kt_mem_z :- kt_member(z, [a,b,c])))
        ),
        (   wam_kotlin_target:write_wam_kotlin_project(
                [user:kt_mem_b/0, user:kt_mem_c/0, user:kt_mem_z/0, user:kt_member/2],
                [emit_mode(interpreter), conformance_main(true)], TmpDir),
            run_gradle(TmpDir, ['-q', 'run', '--args=kt_mem_b/0'], BOut, _BErr, BStatus),
            assertion(BStatus == exit(0)),
            normalize_space(string(BTrim), BOut),
            assertion(BTrim == "true"),
            run_gradle(TmpDir, ['-q', 'run', '--args=kt_mem_c/0'], COut, _CErr, CStatus),
            assertion(CStatus == exit(0)),
            normalize_space(string(CTrim), COut),
            assertion(CTrim == "true"),
            run_gradle(TmpDir, ['-q', 'run', '--args=kt_mem_z/0'], ZOut, _ZErr, ZStatus),
            assertion(ZStatus == exit(0)),
            normalize_space(string(ZTrim), ZOut),
            assertion(ZTrim == "false")
        ),
        (   retractall(user:kt_member(_, _)),
            retractall(user:kt_mem_b),
            retractall(user:kt_mem_c),
            retractall(user:kt_mem_z)
        )
    ).

% KT-Y-ENV-RECURSION: recursive arithmetic must see Y-register bindings
% after call returns (heap-identity vars, not scoped Y@E names).
test(y_env_recursion_fib, [condition(gradle_available), nondet]) :-
    TmpDir = 'output/test_wam_kotlin_y_env_fib',
    make_directory_path('output'),
    clean_dir(TmpDir),
    setup_call_cleanup(
        (   retractall(user:kt_fib(_, _)),
            retractall(user:kt_fib_ok),
            retractall(user:kt_fib_bad),
            assertz(user:kt_fib(0, 0)),
            assertz(user:kt_fib(1, 1)),
            assertz(user:(kt_fib(N, R) :- N > 1, N1 is N - 1, N2 is N - 2,
                kt_fib(N1, R1), kt_fib(N2, R2), R is R1 + R2)),
            assertz(user:(kt_fib_ok :- kt_fib(10, 55))),
            assertz(user:(kt_fib_bad :- kt_fib(10, 54)))
        ),
        (   wam_kotlin_target:write_wam_kotlin_project(
                [user:kt_fib_ok/0, user:kt_fib_bad/0, user:kt_fib/2],
                [emit_mode(interpreter), conformance_main(true)], TmpDir),
            run_gradle(TmpDir, ['-q', 'run', '--args=kt_fib_ok/0'], OkOut, _OkErr, OkStatus),
            assertion(OkStatus == exit(0)),
            normalize_space(string(OkTrim), OkOut),
            assertion(OkTrim == "true"),
            run_gradle(TmpDir, ['-q', 'run', '--args=kt_fib_bad/0'], BadOut, _BadErr, BadStatus),
            assertion(BadStatus == exit(0)),
            normalize_space(string(BadTrim), BadOut),
            assertion(BadTrim == "false")
        ),
        (   retractall(user:kt_fib(_, _)),
            retractall(user:kt_fib_ok),
            retractall(user:kt_fib_bad)
        )
    ).

:- end_tests(wam_kotlin_target).
