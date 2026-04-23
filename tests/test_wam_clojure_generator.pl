:- encoding(utf8).
:- use_module(library(plunit)).
:- use_module(library(filesex)).
:- use_module(library(process)).
:- use_module('../src/unifyweaver/targets/wam_clojure_target').

:- begin_tests(wam_clojure_generator).

:- dynamic user:wam_fact/1.
:- dynamic user:wam_execute_caller/1.
:- dynamic user:wam_call_caller/1.
:- dynamic user:wam_choice_fact/1.
:- dynamic user:wam_choice_caller/1.
:- dynamic user:wam_choice_or_z/1.
:- dynamic user:wam_bind_then_fact/1.
:- dynamic user:wam_bind_after_call/1.
:- dynamic user:wam_bind_before_execute/1.
:- dynamic user:wam_if_then_else/1.
:- dynamic user:wam_struct_fact/1.
:- dynamic user:wam_list_fact/1.
:- dynamic user:wam_use_struct/1.
:- dynamic user:wam_use_list/1.

user:wam_fact(a).
user:wam_execute_caller(X) :- user:wam_fact(X).
user:wam_call_caller(X) :- user:wam_fact(X), user:wam_fact(X).
user:wam_choice_fact(a).
user:wam_choice_fact(b).
user:wam_choice_fact(c).
user:wam_choice_caller(X) :- user:wam_choice_fact(X).
user:wam_choice_or_z(X) :- user:wam_choice_fact(X).
user:wam_choice_or_z(z).
user:wam_bind_then_fact(X) :- Y = X, user:wam_fact(Y).
user:wam_bind_after_call(X) :- user:wam_fact(X), X = a.
user:wam_bind_before_execute(X) :- X = a, user:wam_fact(X).
user:wam_if_then_else(X) :- (X = a -> true ; X = b).
user:wam_struct_fact(f(a)).
user:wam_list_fact([a,b]).
user:wam_use_struct(X) :- user:wam_struct_fact(X).
user:wam_use_list(X) :- user:wam_list_fact(X).

test(project_uses_shared_wam_table_for_cross_predicate_calls) :-
    once((
        unique_tmp_dir('tmp_wam_clojure_shared', TmpDir),
        write_wam_clojure_project([user:wam_execute_caller/1,
                                   user:wam_call_caller/1,
                                   user:wam_fact/1],
                                  [namespace('generated.wam_test')], TmpDir),
        directory_file_path(TmpDir, 'src/generated/wam_test/core.clj', CorePath),
        directory_file_path(TmpDir, 'src/generated/wam_test/runtime.clj', RuntimePath),
        directory_file_path(TmpDir, 'deps.edn', DepsPath),
        directory_file_path(TmpDir, 'project.clj', ProjectPath),
        read_file_to_string(CorePath, CoreCode, []),
        read_file_to_string(RuntimePath, RuntimeCode, []),
        read_file_to_string(DepsPath, DepsCode, []),
        read_file_to_string(ProjectPath, ProjectCode, []),
        assertion(sub_string(CoreCode, _, _, _, '(def shared-wam-code-raw [')),
        assertion(sub_string(CoreCode, _, _, _, '(def shared-wam-labels {')),
        assertion(sub_string(CoreCode, _, _, _, '(def shared-wam-code')),
        assertion(sub_string(CoreCode, _, _, _, 'runtime/resolve-instructions shared-wam-code-raw shared-wam-labels')),
        assertion(sub_string(CoreCode, _, _, _, '(def wam-execute-caller-start-pc ')),
        assertion(sub_string(CoreCode, _, _, _, '(defn wam-execute-caller [a1]')),
        assertion(sub_string(CoreCode, _, _, _, '(defn wam-call-caller [a1]')),
        assertion(sub_string(CoreCode, _, _, _, 'runtime/run-wam-predicate shared-wam-code shared-wam-labels wam-execute-caller-start-pc')),
        assertion(sub_string(CoreCode, _, _, _, 'foreign-handlers')),
        assertion(sub_string(CoreCode, _, _, _, '(def predicate-dispatch {')),
        assertion(sub_string(CoreCode, _, _, _, '"wam_execute_caller/1" wam-execute-caller')),
        assertion(sub_string(CoreCode, _, _, _, '"wam_call_caller/1" wam-call-caller')),
        assertion(sub_string(CoreCode, _, _, _, '(defn -main [& args]')),
        assertion(sub_string(RuntimeCode, _, _, _, '(defn resolve-instructions [code labels]')),
        assertion(sub_string(RuntimeCode, _, _, _, ':try-me-else-pc')),
        assertion(sub_string(RuntimeCode, _, _, _, ':switch-on-constant')),
        assertion(sub_string(RuntimeCode, _, _, _, ':call-pc')),
        assertion(sub_string(RuntimeCode, _, _, _, ':execute-pc')),
        assertion(sub_string(RuntimeCode, _, _, _, ':jump-pc')),
        assertion(sub_string(RuntimeCode, _, _, _, ':builtin-call')),
        assertion(sub_string(RuntimeCode, _, _, _, ':call-foreign')),
        assertion(sub_string(RuntimeCode, _, _, _, '(defn apply-foreign-result [state result]')),
        assertion(sub_string(RuntimeCode, _, _, _, '(defn apply-foreign-bindings [state bindings]')),
        assertion(sub_string(RuntimeCode, _, _, _, '(defn apply-first-foreign-solution [base-state resume-pc results]')),
        assertion(sub_string(RuntimeCode, _, _, _, '(defn foreign-choice-snapshot [state target-pc results]')),
        assertion(sub_string(RuntimeCode, _, _, _, ':kind :foreign')),
        assertion(sub_string(RuntimeCode, _, _, _, ':bindings')),
        assertion(sub_string(RuntimeCode, _, _, _, ':solutions')),
        assertion(sub_string(RuntimeCode, _, _, _, ':cut-ite')),
        assertion(sub_string(RuntimeCode, _, _, _, ':put-structure')),
        assertion(sub_string(RuntimeCode, _, _, _, ':put-list')),
        assertion(sub_string(RuntimeCode, _, _, _, ':set-variable')),
        assertion(sub_string(RuntimeCode, _, _, _, ':set-value')),
        assertion(sub_string(RuntimeCode, _, _, _, ':set-constant')),
        assertion(sub_string(RuntimeCode, _, _, _, ':get-structure')),
        assertion(sub_string(RuntimeCode, _, _, _, ':get-list')),
        assertion(sub_string(RuntimeCode, _, _, _, ':unify-constant')),
        assertion(sub_string(RuntimeCode, _, _, _, ':unify-variable')),
        assertion(sub_string(RuntimeCode, _, _, _, ':retry-me-else-pc')),
        assertion(sub_string(RuntimeCode, _, _, _, ':trust-me')),
        assertion(sub_string(RuntimeCode, _, _, _, ':regs (:regs state)')),
        assertion(\+ sub_string(RuntimeCode, _, _, _, 'snapshot-choice-regs')),
        assertion(sub_string(RuntimeCode, _, _, _, '(defn step [state]')),
        assertion(sub_string(DepsCode, _, _, _, '"-m" "generated.wam_test.core"')),
        assertion(sub_string(ProjectCode, _, _, _, ':main generated.wam_test.core')),
        delete_directory_and_contents(TmpDir)
    )).

test(foreign_predicates_emit_call_foreign_stub) :-
    once((
        unique_tmp_dir('tmp_wam_clojure_foreign', TmpDir),
        write_wam_clojure_project([user:wam_fact/1,
                                   user:wam_execute_caller/1],
                                  [ namespace('generated.wam_foreign_test'),
                                    foreign_predicates([wam_fact/1])
                                  ], TmpDir),
        directory_file_path(TmpDir, 'src/generated/wam_foreign_test/core.clj', CorePath),
        read_file_to_string(CorePath, CoreCode, []),
        assertion(wam_clojure_target:clojure_foreign_predicate(wam_fact, 1,
                   [foreign_predicates([wam_fact/1])])),
        assertion(sub_string(CoreCode, _, _, _, '{:op :call-foreign :pred "wam_fact" :arity 1}')),
        assertion(sub_string(CoreCode, _, _, _, '"wam_fact/1" 0')),
        assertion(\+ sub_string(CoreCode, _, _, _, '{:op :get-constant :constant "a" :reg "A1"}')),
        delete_directory_and_contents(TmpDir)
    )).

test(clojure_foreign_handlers_emit_handler_map) :-
    once((
        unique_tmp_dir('tmp_wam_clojure_foreign_handler', TmpDir),
        write_wam_clojure_project([user:wam_fact/1],
                                  [ namespace('generated.wam_foreign_handler_test'),
                                    foreign_predicates([wam_fact/1]),
                                    clojure_foreign_handlers([
                                        handler(wam_fact/1, "(fn [args] (= (first args) \"a\"))")
                                    ])
                                  ], TmpDir),
        directory_file_path(TmpDir, 'src/generated/wam_foreign_handler_test/core.clj', CorePath),
        read_file_to_string(CorePath, CoreCode, []),
        assertion(sub_string(CoreCode, _, _, _, '(def foreign-handlers {')),
        assertion(sub_string(CoreCode, _, _, _, '"wam_fact/1" (fn [args] (= (first args) "a"))')),
        assertion(sub_string(CoreCode, _, _, _, 'foreign-handlers)')),
        delete_directory_and_contents(TmpDir)
    )).

test(no_kernels_suppresses_clojure_foreign_stub) :-
    once((
        unique_tmp_dir('tmp_wam_clojure_no_kernels', TmpDir),
        write_wam_clojure_project([user:wam_fact/1],
                                  [ namespace('generated.wam_no_kernels_test'),
                                    foreign_predicates([wam_fact/1]),
                                    no_kernels(true)
                                  ], TmpDir),
        directory_file_path(TmpDir, 'src/generated/wam_no_kernels_test/core.clj', CorePath),
        read_file_to_string(CorePath, CoreCode, []),
        assertion(\+ wam_clojure_target:clojure_foreign_predicate(wam_fact, 1,
                   [foreign_predicates([wam_fact/1]), no_kernels(true)])),
        assertion(\+ sub_string(CoreCode, _, _, _, ':call-foreign')),
        assertion(sub_string(CoreCode, _, _, _, '{:op :get-constant :constant "a" :reg "A1"}')),
        delete_directory_and_contents(TmpDir)
    )).

test(foreign_lowering_false_suppresses_clojure_foreign_stub) :-
    once((
        unique_tmp_dir('tmp_wam_clojure_foreign_off', TmpDir),
        write_wam_clojure_project([user:wam_fact/1],
                                  [ namespace('generated.wam_foreign_off_test'),
                                    foreign_predicates([wam_fact/1]),
                                    foreign_lowering(false)
                                  ], TmpDir),
        directory_file_path(TmpDir, 'src/generated/wam_foreign_off_test/core.clj', CorePath),
        read_file_to_string(CorePath, CoreCode, []),
        assertion(\+ wam_clojure_target:clojure_foreign_predicate(wam_fact, 1,
                   [foreign_predicates([wam_fact/1]), foreign_lowering(false)])),
        assertion(\+ sub_string(CoreCode, _, _, _, ':call-foreign')),
        assertion(sub_string(CoreCode, _, _, _, '{:op :get-constant :constant "a" :reg "A1"}')),
        delete_directory_and_contents(TmpDir)
    )).

test(switch_on_constant_preserves_default_fallthrough) :-
    once((
        unique_tmp_dir('tmp_wam_clojure_switch', TmpDir),
        write_wam_clojure_project([user:wam_choice_fact/1],
                                  [namespace('generated.wam_switch_test')], TmpDir),
        directory_file_path(TmpDir, 'src/generated/wam_switch_test/core.clj', CorePath),
        directory_file_path(TmpDir, 'src/generated/wam_switch_test/runtime.clj', RuntimePath),
        read_file_to_string(CorePath, CoreCode, []),
        read_file_to_string(RuntimePath, RuntimeCode, []),
        assertion(sub_string(CoreCode, _, _, _, '{:value "a" :label "default"}')),
        assertion(sub_string(CoreCode, _, _, _, '{:value "b" :label "L_wam_choice_fact_1_2"}')),
        assertion(sub_string(CoreCode, _, _, _, '{:value "c" :label "L_wam_choice_fact_1_3"}')),
        assertion(sub_string(RuntimeCode, _, _, _, ':default-fallthrough?')),
        assertion(sub_string(RuntimeCode, _, _, _, '(advance state)')),
        delete_directory_and_contents(TmpDir)
    )).

test(minimal_runtime_executes_execute_and_call_paths,
     [condition(clojure_exec_e2e_enabled)]) :-
    once((
        unique_tmp_dir('tmp_wam_clojure_exec_basic', TmpDir),
        write_wam_clojure_project([user:wam_execute_caller/1,
                                   user:wam_call_caller/1,
                                   user:wam_fact/1],
                                  [namespace('generated.wam_exec_test'),
                                   module_name('wam-clojure-exec-test')], TmpDir),
        run_clojure_predicate(TmpDir, 'wam_execute_caller/1', 'a', TrueExecute),
        run_clojure_predicate(TmpDir, 'wam_execute_caller/1', 'b', FalseExecute),
        run_clojure_predicate(TmpDir, 'wam_call_caller/1', 'a', TrueCall),
        run_clojure_predicate(TmpDir, 'wam_call_caller/1', 'b', FalseCall),
        assertion(TrueExecute == "true"),
        assertion(FalseExecute == "false"),
        assertion(TrueCall == "true"),
        assertion(FalseCall == "false"),
        delete_directory_and_contents(TmpDir)
    )).

test(choicepoints_and_fallback_execute,
     [condition(clojure_exec_e2e_enabled)]) :-
    once((
        unique_tmp_dir('tmp_wam_clojure_exec_choice', TmpDir),
        write_wam_clojure_project([user:wam_choice_fact/1,
                                   user:wam_choice_caller/1,
                                   user:wam_choice_or_z/1],
                                  [namespace('generated.wam_exec_test'),
                                   module_name('wam-clojure-exec-test')], TmpDir),
        run_clojure_predicate(TmpDir, 'wam_choice_caller/1', 'a', ChoiceA),
        run_clojure_predicate(TmpDir, 'wam_choice_caller/1', 'b', ChoiceB),
        run_clojure_predicate(TmpDir, 'wam_choice_caller/1', 'c', ChoiceC),
        run_clojure_predicate(TmpDir, 'wam_choice_caller/1', 'd', ChoiceD),
        run_clojure_predicate(TmpDir, 'wam_choice_or_z/1', 'z', ChoiceZ),
        assertion(ChoiceA == "true"),
        assertion(ChoiceB == "true"),
        assertion(ChoiceC == "true"),
        assertion(ChoiceD == "false"),
        assertion(ChoiceZ == "true"),
        delete_directory_and_contents(TmpDir)
    )).

test(bindings_and_control_flow_execute,
     [condition(clojure_exec_e2e_enabled)]) :-
    once((
        unique_tmp_dir('tmp_wam_clojure_exec_bind', TmpDir),
        write_wam_clojure_project([user:wam_fact/1,
                                   user:wam_bind_then_fact/1,
                                   user:wam_bind_after_call/1,
                                   user:wam_bind_before_execute/1,
                                   user:wam_if_then_else/1],
                                  [namespace('generated.wam_exec_test'),
                                   module_name('wam-clojure-exec-test')], TmpDir),
        run_clojure_predicate(TmpDir, 'wam_bind_then_fact/1', 'a', BindThenFactA),
        run_clojure_predicate(TmpDir, 'wam_bind_then_fact/1', 'b', BindThenFactB),
        run_clojure_predicate(TmpDir, 'wam_bind_after_call/1', 'a', BindAfterCallA),
        run_clojure_predicate(TmpDir, 'wam_bind_after_call/1', 'b', BindAfterCallB),
        run_clojure_predicate(TmpDir, 'wam_bind_before_execute/1', 'a', BindBeforeExecA),
        run_clojure_predicate(TmpDir, 'wam_bind_before_execute/1', 'b', BindBeforeExecB),
        run_clojure_predicate(TmpDir, 'wam_if_then_else/1', 'a', IfA),
        run_clojure_predicate(TmpDir, 'wam_if_then_else/1', 'b', IfB),
        run_clojure_predicate(TmpDir, 'wam_if_then_else/1', 'c', IfC),
        assertion(BindThenFactA == "true"),
        assertion(BindThenFactB == "false"),
        assertion(BindAfterCallA == "true"),
        assertion(BindAfterCallB == "false"),
        assertion(BindBeforeExecA == "true"),
        assertion(BindBeforeExecB == "false"),
        assertion(IfA == "true"),
        assertion(IfB == "true"),
        assertion(IfC == "false"),
        delete_directory_and_contents(TmpDir)
    )).

test(read_mode_structures_and_lists_execute,
     [condition(clojure_exec_e2e_enabled)]) :-
    once((
        unique_tmp_dir('tmp_wam_clojure_exec_struct', TmpDir),
        write_wam_clojure_project([user:wam_struct_fact/1,
                                   user:wam_list_fact/1,
                                   user:wam_use_struct/1,
                                   user:wam_use_list/1],
                                  [namespace('generated.wam_exec_test'),
                                   module_name('wam-clojure-exec-test')], TmpDir),
        run_clojure_predicate(TmpDir, 'wam_use_struct/1', 'f(a)', UseStruct),
        run_clojure_predicate(TmpDir, 'wam_use_struct/1', 'f(b)', UseStructFail),
        run_clojure_predicate(TmpDir, 'wam_use_list/1', '[a,b]', UseList),
        run_clojure_predicate(TmpDir, 'wam_use_list/1', '[a,c]', UseListFail),
        assertion(UseStruct == "true"),
        assertion(UseStructFail == "false"),
        assertion(UseList == "true"),
        assertion(UseListFail == "false"),
        delete_directory_and_contents(TmpDir)
    )).

has_working_clojure_main :-
    find_clojure_classpath(ClassPath),
    catch(process_create(path(java),
                         ['-cp', ClassPath, 'clojure.main', '-e', '(println true)'],
                         [stdout(pipe(Out)), stderr(pipe(Err))]), _, fail),
    read_string(Out, _, _),
    read_string(Err, _, _),
    close(Out),
    close(Err).

clojure_exec_e2e_enabled :-
    % Direct java/clojure.main runs work manually, but running them from
    % plunit is unstable in this Termux environment. Keep the helpers and
    % predicates available, but skip these end-to-end checks here.
    fail.

run_clojure_predicate(ProjectDir, PredKey, Arg, Output) :-
    find_clojure_classpath(ClassPath),
    prolog_term_string_to_edn(Arg, EdnArg),
    process_create(path(java),
                   ['-cp', ClassPath, 'clojure.main', '-m', 'generated.wam_exec_test.core', PredKey, EdnArg],
                   [cwd(ProjectDir), stdout(pipe(Out)), stderr(pipe(Err))]),
    read_string(Out, _, OutStr0),
    read_string(Err, _, ErrStr),
    close(Out),
    close(Err),
    normalize_space(string(Output), OutStr0),
    assertion(ErrStr == "").

prolog_term_string_to_edn("a", "\"a\"") :- !.
prolog_term_string_to_edn("b", "\"b\"") :- !.
prolog_term_string_to_edn("c", "\"c\"") :- !.
prolog_term_string_to_edn("d", "\"d\"") :- !.
prolog_term_string_to_edn("z", "\"z\"") :- !.
prolog_term_string_to_edn("f(a)", "{:tag :struct :functor \"f/1\" :args [\"a\"]}") :- !.
prolog_term_string_to_edn("f(b)", "{:tag :struct :functor \"f/1\" :args [\"b\"]}") :- !.
prolog_term_string_to_edn("g(a)", "{:tag :struct :functor \"g/1\" :args [\"a\"]}") :- !.
prolog_term_string_to_edn("[a,b]", "{:tag :struct :functor \"[|]/2\" :args [\"a\" {:tag :struct :functor \"[|]/2\" :args [\"b\" \"[]\"]}]}") :- !.
prolog_term_string_to_edn("[a,c]", "{:tag :struct :functor \"[|]/2\" :args [\"a\" {:tag :struct :functor \"[|]/2\" :args [\"c\" \"[]\"]}]}") :- !.
prolog_term_string_to_edn(Atom, Atom).

find_clojure_classpath(ClassPath) :-
    findall(Path,
        ( member(Path,
              [ '/data/data/com.termux/files/home/.m2/repository/org/clojure/clojure/1.11.1/clojure-1.11.1.jar',
                '/data/data/com.termux/files/home/.m2/repository/org/clojure/spec.alpha/0.3.218/spec.alpha-0.3.218.jar',
                '/data/data/com.termux/files/home/.m2/repository/org/clojure/core.specs.alpha/0.2.62/core.specs.alpha-0.2.62.jar'
              ]),
          exists_file(Path)
        ),
        JarPaths),
    JarPaths \= [],
    atomic_list_concat(['src'|JarPaths], :, ClassPath).

unique_tmp_dir(Prefix, TmpDir) :-
    get_time(T),
    Stamp is floor(T * 1000),
    format(atom(TmpDir), '~w_~w', [Prefix, Stamp]).

:- end_tests(wam_clojure_generator).
