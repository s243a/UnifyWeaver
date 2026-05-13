:- encoding(utf8).
:- use_module(library(plunit)).
:- use_module('../src/unifyweaver/targets/wam_go_target').
:- use_module(library(filesex)).

:- begin_tests(wam_go_generator).

:- dynamic test_resistant/3.
:- dynamic test_resistant_helper/2.
:- dynamic test_caller/2.
:- dynamic wam_only_inner/2.
:- dynamic wam_only_caller/2.

test_resistant(X, Y, Z) :- test_resistant_helper(X, Y), test_resistant_helper(Y, Z).
test_resistant(X, _, X).
test_resistant_helper(a, b).
test_resistant_helper(b, c).
test_caller(X, Z) :- test_resistant(X, _, Z), test_resistant(Z, _, X).
test_caller(X, X).
wam_only_inner(X, Y) :- Y is X + 1, atom(foo), \+ atom(5).
wam_only_caller(X, Z) :- wam_only_inner(X, Y), wam_only_inner(Y, Z).

test(wam_get_constant) :-
    wam_instruction_to_go_literal(get_constant(atom(john), 'A1'), Literal),
    assertion(Literal == '&GetConstant{C: internAtom("john"), Ai: 0}').

test(wam_get_variable) :-
    wam_instruction_to_go_literal(get_variable('X1', 'A1'), Literal),
    assertion(Literal == '&GetVariable{Xn: 100, Ai: 0}').

test(wam_put_structure) :-
    wam_instruction_to_go_literal(put_structure('f/2', 'A1'), Literal),
    assertion(Literal == '&PutStructure{Functor: "f/2", Ai: 0}').

test(wam_switch_on_constant) :-
    once((
        Table = [john-default, jane-'L1'],
        wam_instruction_to_go_literal(switch_on_constant(Table), Literal),
        assertion(sub_string(Literal, _, _, _, '&SwitchOnConstant{Cases: []ConstCase{')),
        assertion(sub_string(Literal, _, _, _, '{Val: internAtom("john"), Label: "default"}')),
        assertion(sub_string(Literal, _, _, _, '{Val: internAtom("jane"), Label: "L1"}'))
    )).

test(parse_wam_line_label) :-
    WamCode = "L_parent_2_2:",
    compile_wam_predicate_to_go(test/1, WamCode, [], GoCode),
    assertion(sub_string(GoCode, _, _, _, '"L_parent_2_2": 0')).

test(parse_wam_line_instruction) :-
    WamCode = "    get_constant john, A1",
    compile_wam_predicate_to_go(test/1, WamCode, [], GoCode),
    assertion(sub_string(GoCode, _, _, _, '&GetConstant{C: internAtom("john"), Ai: 0}')).

test(parse_wam_line_switch) :-
    once((
        WamCode = "    switch_on_constant john:default, jane:L1",
        compile_wam_predicate_to_go(test/1, WamCode, [], GoCode),
        assertion(sub_string(GoCode, _, _, _, '&SwitchOnConstant{Cases: []ConstCase{'))
    )).

test(compiled_predicate_emits_resolved_code_and_wrapper) :-
    once((
        WamCode = "test/1:\n    call helper/1, 1\n    execute done/1\n    try_me_else L1\n    retry_me_else L2\n    switch_on_constant john:L1\n",
        compile_wam_predicate_to_go(test/1, WamCode, [], GoCode),
        assertion(sub_string(GoCode, _, _, _, 'var TestResolvedCode = resolveInstructions(TestCode, TestLabels)')),
        assertion(sub_string(GoCode, _, _, _, 'func Test(a1 Value) bool {')),
        assertion(sub_string(GoCode, _, _, _, 'vm := NewWamState(TestResolvedCode, TestLabels)'))
    )).

test(project_uses_shared_wam_table_for_cross_predicate_calls) :-
    once((
        get_time(T),
        format(atom(TmpDir), 'tmp_wam_go_shared_~w', [T]),
        write_wam_go_project([plunit_wam_go_generator:wam_only_caller/2,
                              plunit_wam_go_generator:wam_only_inner/2],
                             [module_name(go_shared_test)], TmpDir),
        directory_file_path(TmpDir, 'lib.go', LibPath),
        read_file_to_string(LibPath, LibCode, []),
        assertion(sub_string(LibCode, _, _, _, 'var sharedWamCodeRaw = []Instruction{')),
        assertion(sub_string(LibCode, _, _, _, 'var sharedWamCode = resolveInstructions(sharedWamCodeRaw, sharedWamLabels)')),
        assertion(sub_string(LibCode, _, _, _, 'var Wam_only_callerCode = sharedWamCode')),
        assertion(sub_string(LibCode, _, _, _, 'func Wam_only_caller(a1 Value, a2 Value) bool {')),
        assertion(sub_string(LibCode, _, _, _, 'vm := NewWamState(sharedWamCode, sharedWamLabels)')),
        assertion(sub_string(LibCode, _, _, _, 'vm.PC = ')),
        delete_directory_and_contents(TmpDir)
    )).

test(robust_switch_parsing) :-
    once((
        % Test malformed input robustness
        wam_line_to_go_literal(["switch_on_constant", "john"], Literal),
        assertion(sub_string(Literal, _, _, _, 'internAtom("malformed")'))
    )).

:- end_tests(wam_go_generator).
