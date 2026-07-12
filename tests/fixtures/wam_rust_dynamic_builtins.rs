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
fn asserted_rule_body_can_call_retract() {
    let mut vm = WamState::new(vec![], HashMap::new());
    assert_clause(&mut vm, "assertz/1", fact("dyn", vec![at("taken")]));
    assert_clause(
        &mut vm,
        "assertz/1",
        rule(
            fact("take", vec![ub("X")]),
            fact("retract", vec![fact("dyn", vec![ub("X")])]),
        ),
    );

    vm.reset_query();
    vm.code = vec![
        Instruction::Call("take/1".to_string(), 1),
        Instruction::Proceed,
    ];
    vm.labels = HashMap::new();
    vm.set_reg_str("A1", ub("X"));
    vm.pc = 1;

    assert!(vm.run());
    let result_raw = vm.bindings.get("X").cloned().expect("X bound");
    assert_eq!(
        vm.deref_heap(&vm.deref_var(&result_raw)),
        at("taken"),
    );
    assert!(dyn_values(&mut vm).is_empty());
}

#[test]
fn asserted_rule_body_retract_backtracks_without_skipping() {
    let mut vm = WamState::new(vec![], HashMap::new());
    for value in ["red", "blue", "green"] {
        assert_clause(&mut vm, "assertz/1", fact("dyn", vec![at(value)]));
    }
    assert_clause(
        &mut vm,
        "assertz/1",
        rule(
            fact("take", vec![ub("X")]),
            fact("retract", vec![fact("dyn", vec![ub("X")])]),
        ),
    );

    vm.reset_query();
    vm.code = vec![
        Instruction::Call("take/1".to_string(), 1),
        Instruction::Proceed,
    ];
    vm.labels = HashMap::new();
    vm.set_reg_str("A1", ub("X"));
    vm.pc = 1;

    let mut removed = Vec::new();
    assert!(vm.run());
    loop {
        let result_raw = vm.bindings.get("X").cloned().expect("X bound");
        removed.push(vm.deref_heap(&vm.deref_var(&result_raw)));
        if !vm.backtrack() {
            break;
        }
    }

    assert_eq!(
        removed,
        vec![at("red"), at("blue"), at("green")],
    );
    assert!(dyn_values(&mut vm).is_empty());
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
fn clause_backtracks_over_facts_and_rules_without_removing_them() {
    let mut vm = WamState::new(
        vec![Instruction::Call("clause/2".to_string(), 2), Instruction::Proceed],
        HashMap::new(),
    );
    assert_clause(&mut vm, "assertz/1", fact("dyn", vec![at("red")]));
    assert_clause(&mut vm, "assertz/1", fact("marker", vec![at("blue")]));
    assert_clause(
        &mut vm,
        "assertz/1",
        rule(
            fact("dyn", vec![at("blue")]),
            fact("marker", vec![at("blue")]),
        ),
    );

    vm.reset_query();
    vm.set_reg_str("A1", fact("dyn", vec![ub("X")]));
    vm.set_reg_str("A2", ub("Body"));
    vm.pc = 1;

    let mut clauses = Vec::new();
    assert!(vm.run());
    loop {
        let x = vm.bindings.get("X").cloned().expect("X bound");
        let body = vm.bindings.get("Body").cloned().expect("Body bound");
        clauses.push((
            vm.deref_heap(&vm.deref_var(&x)),
            vm.deref_heap(&vm.deref_var(&body)),
        ));
        if !vm.backtrack() { break; }
    }

    assert_eq!(
        clauses,
        vec![
            (at("red"), at("true")),
            (at("blue"), fact("marker", vec![at("blue")]))
        ],
    );
    assert_eq!(dyn_values(&mut vm), vec![at("red"), at("blue")]);
}

#[test]
fn generated_tail_clause_call_reads_dynamic_facts() {
    let (code, labels) = shared_wam_program();
    let mut vm = WamState::new(code, labels);
    assert_clause(&mut vm, "assertz/1", fact("dyn", vec![at("tail")]));

    vm.reset_query();
    vm.set_reg_str("A1", ub("X"));
    vm.set_reg_str("A2", ub("Body"));
    vm.pc = *vm.labels
        .get("rust_clause_demo/2")
        .expect("generated clause demo label");

    assert!(vm.run());
    let x = vm.bindings.get("X").cloned().expect("X bound");
    let body = vm.bindings.get("Body").cloned().expect("Body bound");
    assert_eq!(vm.deref_heap(&vm.deref_var(&x)), at("tail"));
    assert_eq!(vm.deref_heap(&vm.deref_var(&body)), at("true"));
}

#[test]
fn current_predicate_backtracks_over_matching_dynamic_arities() {
    let mut vm = WamState::new(
        vec![
            Instruction::Call("current_predicate/1".to_string(), 1),
            Instruction::Proceed,
        ],
        HashMap::new(),
    );
    assert_clause(&mut vm, "assertz/1", fact("dyn", vec![at("one")]));
    assert_clause(
        &mut vm,
        "assertz/1",
        fact("dyn", vec![at("one"), at("two")]),
    );

    vm.reset_query();
    vm.set_reg_str("A1", fact("/", vec![at("dyn"), ub("Arity")]));
    vm.pc = 1;

    let mut arities = Vec::new();
    assert!(vm.run());
    loop {
        let arity = vm.bindings.get("Arity").cloned().expect("Arity bound");
        arities.push(vm.deref_heap(&vm.deref_var(&arity)));
        if !vm.backtrack() { break; }
    }
    assert_eq!(arities, vec![Value::Integer(1), Value::Integer(2)]);

    vm.reset_query();
    vm.set_reg_str("A1", fact("/", vec![at("dyn"), Value::Integer(-1)]));
    vm.pc = 1;
    assert!(!vm.run());
    assert!(vm.thrown_ball.is_none());
}

#[test]
fn generated_tail_current_predicate_call_reads_static_labels() {
    let (code, labels) = shared_wam_program();
    let mut vm = WamState::new(code, labels);
    vm.set_reg_str("A1", at("rust_clause_demo"));
    vm.set_reg_str("A2", ub("Arity"));
    vm.pc = *vm.labels
        .get("rust_current_predicate_demo/2")
        .expect("generated current_predicate demo label");

    assert!(vm.run());
    let arity = vm.bindings.get("Arity").cloned().expect("Arity bound");
    assert_eq!(vm.deref_heap(&vm.deref_var(&arity)), Value::Integer(2));
}

#[test]
fn generated_current_predicate_errors_are_catchable() {
    let (code, labels) = shared_wam_program();
    for pred in [
        "rust_current_predicate_instantiation_demo/1",
        "rust_current_predicate_type_demo/1",
        "rust_current_predicate_name_type_demo/1",
        "rust_current_predicate_arity_type_demo/1",
    ] {
        let mut vm = WamState::new(code.clone(), labels.clone());
        vm.set_reg_str("A1", ub("Result"));
        vm.pc = *vm.labels.get(pred).expect("generated error demo label");

        assert!(vm.run(), "{} should catch its error", pred);
        assert_eq!(
            vm.bindings.get("Result"),
            Some(&at("caught")),
            "{} recovery should run",
            pred,
        );
        assert!(vm.thrown_ball.is_none(), "{} should consume the error", pred);
    }
}

#[test]
fn generated_predicate_property_reads_static_labels() {
    let (code, labels) = shared_wam_program();
    let mut vm = WamState::new(code, labels);
    vm.set_reg_str("A1", fact("rust_clause_demo", vec![ub("_"), ub("_")]));
    vm.set_reg_str("A2", at("static"));
    assert!(vm.execute_builtin("predicate_property/2", 2));

    vm.reset_query();
    vm.set_reg_str("A1", fact("rust_clause_demo", vec![ub("_"), ub("_")]));
    vm.set_reg_str("A2", at("static"));
    vm.pc = *vm.labels
        .get("rust_predicate_property_demo/2")
        .expect("generated predicate_property demo label");

    assert!(vm.run());

    vm.reset_query();
    vm.set_reg_str("A1", fact("rust_clause_demo", vec![ub("_"), ub("_")]));
    vm.set_reg_str("A2", at("defined"));
    assert!(vm.execute_builtin("predicate_property/2", 2));

    vm.reset_query();
    vm.set_reg_str(
        "A1",
        fact("rust_clause_demo", vec![ub("_"), ub("_")]),
    );
    vm.set_reg_str("A2", fact("number_of_clauses", vec![ub("Count")]));
    assert!(vm.execute_builtin("predicate_property/2", 2));
    let count = vm.bindings.get("Count").cloned().expect("Count bound");
    assert_eq!(vm.deref_heap(&vm.deref_var(&count)), Value::Integer(1));

    vm.reset_query();
    vm.set_reg_str("A1", fact("missing", vec![ub("_")]));
    vm.set_reg_str("A2", at("defined"));
    assert!(!vm.execute_builtin("predicate_property/2", 2));
}

#[test]
fn generated_predicate_property_errors_are_catchable() {
    let (code, labels) = shared_wam_program();
    for pred in [
        "rust_predicate_property_head_instantiation_demo/1",
        "rust_predicate_property_head_type_demo/1",
        "rust_predicate_property_property_instantiation_demo/1",
        "rust_predicate_property_domain_demo/1",
    ] {
        let mut vm = WamState::new(code.clone(), labels.clone());
        vm.set_reg_str("A1", ub("Result"));
        vm.pc = *vm.labels.get(pred).expect("generated error demo label");

        assert!(vm.run(), "{} should catch its error", pred);
        assert_eq!(vm.bindings.get("Result"), Some(&at("caught")));
        assert!(vm.thrown_ball.is_none(), "{} should consume the error", pred);
    }
}

#[test]
fn predicate_property_reports_dynamic_status_and_clause_count() {
    let mut vm = WamState::new(vec![], HashMap::new());
    assert_clause(&mut vm, "assertz/1", fact("dyn", vec![at("red")]));
    assert_clause(&mut vm, "assertz/1", fact("dyn", vec![at("blue")]));

    vm.reset_query();
    vm.set_reg_str("A1", fact("dyn", vec![ub("_")]));
    vm.set_reg_str("A2", at("dynamic"));
    assert!(vm.execute_builtin("predicate_property/2", 2));

    vm.reset_query();
    vm.set_reg_str("A1", fact("dyn", vec![ub("_")]));
    vm.set_reg_str(
        "A2",
        fact("number_of_clauses", vec![ub("Count")]),
    );
    assert!(vm.execute_builtin("predicate_property/2", 2));
    let count = vm.bindings.get("Count").cloned().expect("Count bound");
    assert_eq!(vm.deref_heap(&vm.deref_var(&count)), Value::Integer(2));

    vm.reset_query();
    vm.set_reg_str("A1", fact("dyn", vec![ub("_")]));
    vm.set_reg_str("A2", at("static"));
    assert!(!vm.execute_builtin("predicate_property/2", 2));
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
fn asserted_rule_body_can_call_read_term_two() {
    let (code, labels) = shared_wam_program();
    let mut vm = WamState::new(code, labels);
    assert_clause(
        &mut vm,
        "assertz/1",
        rule(
            fact("parse_meta", vec![ub("T"), ub("Names")]),
            fact(
                "read_term",
                vec![
                    ub("T"),
                    Value::List(vec![fact("variable_names", vec![ub("Names")])]),
                ],
            ),
        ),
    );

    vm.reset_query();
    vm.set_term_input("p(A, A, B).");
    let query_pc = vm.code.len() + 1;
    vm.code.extend([
        Instruction::Call("parse_meta/2".to_string(), 2),
        Instruction::Proceed,
    ]);
    vm.set_reg_str("A1", ub("Term"));
    vm.set_reg_str("A2", ub("Names"));
    vm.pc = query_pc;

    assert!(vm.run());
    let parsed_raw = vm.bindings.get("Term").cloned().expect("Term bound");
    let parsed = vm.deref_heap(&vm.deref_var(&parsed_raw));
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
    vm.set_term_input(
        "/* block header . */% header\nfirst./* block between . */% between terms\n\
         pair(second, '/* kept */', /* ignored . block */ % ignored . full stop\n 2).");

    vm.set_reg_str("A1", ub("T1"));
    assert!(vm.execute_builtin("read/1", 1));
    assert_eq!(vm.bindings.get("T1"), Some(&at("first")));

    vm.reset_query();
    vm.set_reg_str("A1", ub("T2"));
    assert!(vm.execute_builtin("read_term/1", 1));
    assert_eq!(
        vm.bindings.get("T2"),
        Some(&fact(
            "pair",
            vec![at("second"), at("/* kept */"), Value::Integer(2)],
        )),
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
    vm.set_reg_str(
        "A1",
        at("p(A, /* ignored . block */ B, A, _, _C, _C, D)"),
    );
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
