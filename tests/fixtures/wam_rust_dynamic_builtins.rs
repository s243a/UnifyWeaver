use dynrt::instructions::Instruction;
use dynrt::state::WamState;
use dynrt::value::Value;
use dynrt::shared_wam_program;
use std::collections::HashMap;

fn at(s: &str) -> Value { Value::Atom(s.to_string()) }
fn ub(s: &str) -> Value { Value::Unbound(s.to_string()) }
fn fact(name: &str, args: Vec<Value>) -> Value { Value::Str(name.to_string(), args) }
fn rule(head: Value, body: Value) -> Value { fact(":-", vec![head, body]) }
fn conj(left: Value, right: Value) -> Value { fact(",", vec![left, right]) }

fn assert_clause(vm: &mut WamState, op: &str, clause: Value) {
    vm.set_reg_str("A1", clause);
    assert!(vm.execute_builtin(op, 1), "{} should succeed", op);
}

fn dyn_values(vm: &mut WamState) -> Vec<Value> {
    vm.reset_query();
    vm.code = vec![Instruction::Call("dyn/1".to_string(), 1), Instruction::Proceed];
    vm.labels = HashMap::new();
    vm.set_reg_str("A1", ub("X"));
    vm.pc = 1;
    let mut out = Vec::new();
    if vm.run() {
        loop {
            let x = vm.bindings.get("X").cloned().expect("X bound");
            out.push(vm.deref_heap(&vm.deref_var(&x)));
            if !vm.backtrack() { break; }
        }
    }
    out
}

fn second_values(vm: &mut WamState, pred: &str, first: Value) -> Vec<Value> {
    vm.reset_query();
    vm.code = vec![Instruction::Call(pred.to_string(), 2), Instruction::Proceed];
    vm.labels = HashMap::new();
    vm.set_reg_str("A1", first);
    vm.set_reg_str("A2", ub("Y"));
    vm.pc = 1;
    let mut out = Vec::new();
    if vm.run() {
        loop {
            let y = vm.bindings.get("Y").cloned().expect("Y bound");
            out.push(vm.deref_heap(&vm.deref_var(&y)));
            if !vm.backtrack() { break; }
        }
    }
    out
}

#[test]
fn generated_assert_alias_appends_a_dynamic_fact() {
    let (code, labels) = shared_wam_program();
    let mut vm = WamState::new(code, labels);
    vm.pc = *vm.labels
        .get("rust_assert_alias_demo/0")
        .expect("generated assert alias label");
    assert!(vm.run());
    assert_eq!(dyn_values(&mut vm), vec![at("alias")]);
}

#[test]
fn asserted_rule_body_can_call_assert_alias() {
    let mut vm = WamState::new(vec![], HashMap::new());
    assert_clause(
        &mut vm,
        "assertz/1",
        rule(
            at("seed_alias"),
            fact("assert", vec![fact("dyn", vec![at("meta_alias")])]),
        ),
    );

    vm.reset_query();
    vm.code = vec![
        Instruction::Call("seed_alias/0".to_string(), 0),
        Instruction::Proceed,
    ];
    vm.labels = HashMap::new();
    vm.pc = 1;
    assert!(vm.run());
    assert_eq!(dyn_values(&mut vm), vec![at("meta_alias")]);
}

#[test]
fn assertz_asserta_query_order_and_retractall() {
    let mut vm = WamState::new(vec![], HashMap::new());
    assert_clause(&mut vm, "assertz/1", fact("dyn", vec![at("red")]));
    assert_clause(&mut vm, "assertz/1", fact("dyn", vec![at("blue")]));
    assert_clause(&mut vm, "asserta/1", fact("dyn", vec![at("first")]));
    assert_eq!(dyn_values(&mut vm), vec![at("first"), at("red"), at("blue")]);

    vm.reset_query();
    vm.set_reg_str("A1", fact("dyn", vec![at("red")]));
    assert!(vm.execute_builtin("retractall/1", 1));
    assert_eq!(dyn_values(&mut vm), vec![at("first"), at("blue")]);
}

#[test]
fn retract_removes_one_match_and_binds_pattern() {
    let mut vm = WamState::new(vec![Instruction::Call("retract/1".to_string(), 1), Instruction::Proceed], HashMap::new());
    assert_clause(&mut vm, "assertz/1", fact("dyn", vec![at("red")]));
    assert_clause(&mut vm, "assertz/1", fact("dyn", vec![at("blue")]));

    vm.reset_query();
    vm.set_reg_str("A1", fact("dyn", vec![ub("X")]));
    vm.pc = 1;
    assert!(vm.run());
    assert_eq!(vm.bindings.get("X"), Some(&at("red")));
    assert_eq!(dyn_values(&mut vm), vec![at("blue")]);
}

#[test]
fn retract_backtracks_through_each_matching_fact() {
    let mut vm = WamState::new(vec![Instruction::Call("retract/1".to_string(), 1), Instruction::Proceed], HashMap::new());
    assert_clause(&mut vm, "assertz/1", fact("dyn", vec![at("red")]));
    assert_clause(&mut vm, "assertz/1", fact("dyn", vec![at("blue")]));
    assert_clause(&mut vm, "assertz/1", fact("dyn", vec![at("green")]));

    vm.reset_query();
    vm.set_reg_str("A1", fact("dyn", vec![ub("X")]));
    vm.pc = 1;
    let mut removed = Vec::new();
    assert!(vm.run());
    loop {
        let x = vm.bindings.get("X").cloned().expect("X bound");
        removed.push(vm.deref_heap(&vm.deref_var(&x)));
        if !vm.backtrack() { break; }
    }
    assert_eq!(removed, vec![at("red"), at("blue"), at("green")]);
    assert!(dyn_values(&mut vm).is_empty());
}

#[test]
fn retract_distinguishes_fact_patterns_from_rule_patterns() {
    let mut vm = WamState::new(vec![Instruction::Call("retract/1".to_string(), 1), Instruction::Proceed], HashMap::new());
    assert_clause(&mut vm, "assertz/1", fact("marker", vec![at("rule")]));
    assert_clause(
        &mut vm,
        "assertz/1",
        rule(
            fact("dyn", vec![at("rule")]),
            fact("marker", vec![at("rule")]),
        ),
    );
    assert_clause(&mut vm, "assertz/1", fact("dyn", vec![at("fact")]));

    vm.reset_query();
    vm.set_reg_str("A1", fact("dyn", vec![ub("X")]));
    vm.pc = 1;
    assert!(vm.run());
    assert_eq!(vm.bindings.get("X"), Some(&at("fact")));
    assert_eq!(dyn_values(&mut vm), vec![at("rule")]);

    vm.reset_query();
    vm.code = vec![Instruction::Call("retract/1".to_string(), 1), Instruction::Proceed];
    vm.set_reg_str(
        "A1",
        rule(
            fact("dyn", vec![ub("X")]),
            fact("marker", vec![ub("X")]),
        ),
    );
    vm.pc = 1;
    assert!(vm.run());
    assert_eq!(vm.bindings.get("X"), Some(&at("rule")));
    assert!(dyn_values(&mut vm).is_empty());

    assert_clause(&mut vm, "assertz/1", fact("dyn", vec![at("normalized")]));
    vm.reset_query();
    vm.code = vec![Instruction::Call("retract/1".to_string(), 1), Instruction::Proceed];
    vm.set_reg_str(
        "A1",
        rule(
            fact("dyn", vec![ub("X")]),
            at("true"),
        ),
    );
    vm.pc = 1;
    assert!(vm.run());
    assert_eq!(vm.bindings.get("X"), Some(&at("normalized")));
    assert!(dyn_values(&mut vm).is_empty());
}

#[test]
fn asserted_rule_body_calls_dynamic_predicates() {
    let mut vm = WamState::new(vec![], HashMap::new());
    assert_clause(&mut vm, "assertz/1", fact("parent", vec![at("ann"), at("bob")]));
    assert_clause(&mut vm, "assertz/1", fact("parent", vec![at("bob"), at("cid")]));
    assert_clause(
        &mut vm,
        "assertz/1",
        rule(
            fact("ancestor", vec![ub("X"), ub("Y")]),
            fact("parent", vec![ub("X"), ub("Y")]),
        ),
    );
    assert_clause(
        &mut vm,
        "assertz/1",
        rule(
            fact("grandparent", vec![ub("X"), ub("Z")]),
            conj(
                fact("parent", vec![ub("X"), ub("Y")]),
                fact("parent", vec![ub("Y"), ub("Z")]),
            ),
        ),
    );

    assert_eq!(second_values(&mut vm, "ancestor/2", at("ann")), vec![at("bob")]);
    assert_eq!(second_values(&mut vm, "grandparent/2", at("ann")), vec![at("cid")]);
}

#[test]
fn conjunction_rule_backtracks_left_subgoal_for_more_body_solutions() {
    let mut vm = WamState::new(vec![], HashMap::new());
    assert_clause(&mut vm, "assertz/1", fact("parent", vec![at("ann"), at("bob")]));
    assert_clause(&mut vm, "assertz/1", fact("parent", vec![at("ann"), at("beth")]));
    assert_clause(&mut vm, "assertz/1", fact("likes", vec![at("bob"), at("pizza")]));
    assert_clause(&mut vm, "assertz/1", fact("likes", vec![at("beth"), at("salad")]));
    assert_clause(
        &mut vm,
        "assertz/1",
        rule(
            fact("favorite", vec![ub("X"), ub("Z")]),
            conj(
                fact("parent", vec![ub("X"), ub("Y")]),
                fact("likes", vec![ub("Y"), ub("Z")]),
            ),
        ),
    );

    assert_eq!(second_values(&mut vm, "favorite/2", at("ann")), vec![at("pizza"), at("salad")]);
}

#[test]
fn generated_read_term_two_applies_variable_names() {
    let (code, labels) = shared_wam_program();
    let mut vm = WamState::new(code, labels);
    vm.set_term_input("p(A, A, B).");
    vm.set_reg_str("A1", ub("Term"));
    vm.set_reg_str("A2", ub("Names"));
    vm.pc = *vm.labels
        .get("rust_read_term_options_demo/2")
        .expect("generated read_term/2 label");

    assert!(vm.run());
    let parsed = vm.bindings.get("Term").cloned().expect("Term bound");
    let args = match parsed {
        Value::Str(ref functor, ref args) if functor == "p" => args.clone(),
        other => panic!("unexpected parsed term: {:?}", other),
    };
    assert_eq!(args[0], args[1]);
    assert_ne!(args[0], args[2]);

    let names_raw = vm.bindings.get("Names").cloned().expect("Names bound");
    let names = vm.deref_heap(&vm.deref_var(&names_raw));
    assert_eq!(
        names,
        Value::List(vec![
            fact("=", vec![at("A"), args[0].clone()]),
            fact("=", vec![at("B"), args[2].clone()]),
        ]),
    );
}

#[test]
fn generated_atom_to_term_preserves_variables_and_bindings() {
    let (code, labels) = shared_wam_program();
    let mut vm = WamState::new(code, labels);
    vm.set_reg_str("A1", ub("Term"));
    vm.set_reg_str("A2", ub("Bindings"));
    vm.pc = *vm
        .labels
        .get("rust_atom_to_term_demo/2")
        .expect("generated atom_to_term/3 label");

    assert!(vm.run());
    let parsed = vm.bindings.get("Term").cloned().expect("Term bound");
    let args = match parsed {
        Value::Str(ref functor, ref args) if functor == "p" => args.clone(),
        other => panic!("unexpected parsed term: {:?}", other),
    };
    assert_eq!(args[0], args[1]);
    assert_ne!(args[0], args[2]);

    let bindings_raw = vm.bindings.get("Bindings").cloned().expect("Bindings bound");
    let bindings = vm.deref_heap(&vm.deref_var(&bindings_raw));
    assert_eq!(
        bindings,
        Value::List(vec![
            fact("=", vec![at("A"), args[0].clone()]),
            fact("=", vec![at("B"), args[2].clone()]),
        ]),
    );
}

#[test]
fn read_eof_binds_end_of_file() {
    let mut vm = WamState::new(vec![], HashMap::new());
    vm.set_reg_str("A1", ub("T"));
    assert!(vm.execute_builtin("read/1", 1));
    assert_eq!(vm.bindings.get("T"), Some(&at("end_of_file")));
}

#[test]
fn read_consumes_buffered_terms_before_end_of_file() {
    let (code, labels) = shared_wam_program();
    let mut vm = WamState::new(code, labels);
    vm.set_term_input("first. pair(second, 2).");

    vm.set_reg_str("A1", ub("T1"));
    assert!(vm.execute_builtin("read/1", 1));
    assert_eq!(vm.bindings.get("T1"), Some(&at("first")));

    vm.reset_query();
    vm.set_reg_str("A1", ub("T2"));
    assert!(vm.execute_builtin("read_term/1", 1));
    assert_eq!(
        vm.bindings.get("T2"),
        Some(&fact("pair", vec![at("second"), Value::Integer(2)])),
    );

    vm.reset_query();
    vm.set_reg_str("A1", ub("End"));
    assert!(vm.execute_builtin("read/1", 1));
    assert_eq!(vm.bindings.get("End"), Some(&at("end_of_file")));
}

#[test]
fn term_to_atom_quotes_and_roundtrips_non_bare_atoms() {
    let (code, labels) = shared_wam_program();
    let mut vm = WamState::new(code, labels);
    let term = fact(
        "odd functor",
        vec![
            at("hello world"),
            at("42"),
            at("can't"),
            at("a\\b"),
        ],
    );

    vm.set_reg_str("A1", term.clone());
    vm.set_reg_str("A2", ub("Text"));
    assert!(vm.execute_builtin("term_to_atom/2", 2));
    let rendered = at("'odd functor'('hello world', '42', 'can\\'t', 'a\\\\b')");
    assert_eq!(vm.bindings.get("Text"), Some(&rendered));

    for (text, expected) in [
        ("'hello world'", at("hello world")),
        ("'42'", at("42")),
        ("'can\\'t'", at("can't")),
        ("'a\\\\b'", at("a\\b")),
    ] {
        vm.reset_query();
        vm.set_reg_str("A1", ub("Quoted"));
        vm.set_reg_str("A2", at(text));
        assert!(vm.execute_builtin("term_to_atom/2", 2), "failed to parse {}", text);
        assert_eq!(vm.bindings.get("Quoted"), Some(&expected));
    }

    vm.reset_query();
    vm.set_reg_str("A1", ub("Parsed"));
    vm.set_reg_str("A2", rendered);
    assert!(vm.execute_builtin("term_to_atom/2", 2));
    assert_eq!(vm.bindings.get("Parsed"), Some(&term));
}

#[test]
fn read_term_from_atom_returns_variable_metadata_with_shared_variables() {
    let (code, labels) = shared_wam_program();
    let mut vm = WamState::new(code, labels);
    vm.set_reg_str("A1", at("p(A, B, A, _, _C, _C, D)"));
    vm.set_reg_str("A2", ub("Term"));
    vm.set_reg_str(
        "A3",
        Value::List(vec![
            fact("variables", vec![ub("Variables")]),
            fact("variable_names", vec![ub("Names")]),
            fact("singletons", vec![ub("Singletons")]),
        ]),
    );

    assert!(vm.execute_builtin("read_term_from_atom/3", 3));
    let parsed = vm.bindings.get("Term").cloned().expect("Term bound");
    let args = match parsed {
        Value::Str(ref functor, ref args) if functor == "p" => args.clone(),
        other => panic!("unexpected parsed term: {:?}", other),
    };
    assert_eq!(args[0], args[2]);
    assert_eq!(args[4], args[5]);
    assert_ne!(args[0], args[1]);
    assert_ne!(args[0], args[3]);

    let variables_raw = vm.bindings.get("Variables").cloned().expect("Variables bound");
    let variables = vm.deref_heap(&vm.deref_var(&variables_raw));
    assert_eq!(
        variables,
        Value::List(vec![
            args[0].clone(),
            args[1].clone(),
            args[3].clone(),
            args[4].clone(),
            args[6].clone(),
        ]),
    );

    let names_raw = vm.bindings.get("Names").cloned().expect("Names bound");
    let names = vm.deref_heap(&vm.deref_var(&names_raw));
    assert_eq!(
        names,
        Value::List(vec![
            fact("=", vec![at("A"), args[0].clone()]),
            fact("=", vec![at("B"), args[1].clone()]),
            fact("=", vec![at("_C"), args[4].clone()]),
            fact("=", vec![at("D"), args[6].clone()]),
        ]),
    );

    let singletons_raw = vm.bindings.get("Singletons").cloned().expect("Singletons bound");
    let singletons = vm.deref_heap(&vm.deref_var(&singletons_raw));
    assert_eq!(
        singletons,
        Value::List(vec![
            fact("=", vec![at("B"), args[1].clone()]),
            fact("=", vec![at("_"), args[3].clone()]),
            fact("=", vec![at("D"), args[6].clone()]),
        ]),
    );
}

#[test]
fn read_term_syntax_errors_error_throws_and_quiet_fails() {
    let (code, labels) = shared_wam_program();
    let mut vm = WamState::new(code.clone(), labels.clone());
    vm.set_reg_str("A1", ub("Result"));
    vm.pc = *vm
        .labels
        .get("rust_syntax_error_demo/1")
        .expect("missing syntax error demo");
    assert!(vm.run());
    assert_eq!(vm.bindings.get("Result"), Some(&at("caught")));
    assert!(vm.thrown_ball.is_none());

    let mut quiet = WamState::new(code, labels);
    quiet.set_reg_str("A1", at("p("));
    quiet.set_reg_str("A2", ub("Term"));
    quiet.set_reg_str(
        "A3",
        Value::List(vec![fact("syntax_errors", vec![at("quiet")])]),
    );
    assert!(!quiet.execute_builtin("read_term_from_atom/3", 3));
    assert!(quiet.thrown_ball.is_none());
}
