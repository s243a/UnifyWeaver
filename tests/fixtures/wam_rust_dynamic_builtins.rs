use dynrt::instructions::Instruction;
use dynrt::state::WamState;
use dynrt::value::Value;
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
fn read_eof_binds_end_of_file() {
    let mut vm = WamState::new(vec![], HashMap::new());
    vm.set_reg_str("A1", ub("T"));
    assert!(vm.execute_builtin("read/1", 1));
    assert_eq!(vm.bindings.get("T"), Some(&at("end_of_file")));
}
