:- encoding(utf8).
:- use_module(library(plunit)).
:- use_module('../src/unifyweaver/targets/wam_go_target').

:- begin_tests(wam_go_generator).

test(wam_get_constant) :-
    wam_instruction_to_go_literal(get_constant(atom(john), 'A1'), Literal),
    assertion(Literal == '&GetConstant{C: &Atom{Name: "john"}, Ai: "A1"}').

test(wam_get_variable) :-
    wam_instruction_to_go_literal(get_variable('X1', 'A1'), Literal),
    assertion(Literal == '&GetVariable{Xn: "X1", Ai: "A1"}').

test(wam_put_structure) :-
    wam_instruction_to_go_literal(put_structure('f/2', 'A1'), Literal),
    assertion(Literal == '&PutStructure{Functor: "f/2", Ai: "A1"}').

test(wam_switch_on_constant) :-
    Table = [john-default, jane-'L1'],
    wam_instruction_to_go_literal(switch_on_constant(Table), Literal),
    assertion(sub_string(Literal, _, _, _, '&SwitchOnConstant{Cases: []ConstCase{')),
    assertion(sub_string(Literal, _, _, _, '{Val: &Atom{Name: "john"}, Label: "default"}')),
    assertion(sub_string(Literal, _, _, _, '{Val: &Atom{Name: "jane"}, Label: "L1"}')).

test(parse_wam_line_label) :-
    WamCode = "L_parent_2_2:",
    compile_wam_predicate_to_go(test/1, WamCode, [], GoCode),
    assertion(sub_string(GoCode, _, _, _, '"L_parent_2_2": 0')).

test(parse_wam_line_instruction) :-
    WamCode = "    get_constant john, A1",
    compile_wam_predicate_to_go(test/1, WamCode, [], GoCode),
    assertion(sub_string(GoCode, _, _, _, '&GetConstant{C: &Atom{Name: "john"}, Ai: "A1"}')).

test(parse_wam_line_switch) :-
    WamCode = "    switch_on_constant john:default, jane:L1",
    compile_wam_predicate_to_go(test/1, WamCode, [], GoCode),
    assertion(sub_string(GoCode, _, _, _, '&SwitchOnConstant{Cases: []ConstCase{')).

test(robust_switch_parsing) :-
    % Test malformed input robustness
    wam_line_to_go_literal(["switch_on_constant", "john"], Literal),
    assertion(sub_string(Literal, _, _, _, '&Atom{Name: "malformed"}')).

:- end_tests(wam_go_generator).
