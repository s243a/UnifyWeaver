use dynrt::instructions::Instruction;
use dynrt::state::WamState;
use dynrt::value::Value;
use std::collections::HashMap;

fn at(s: &str) -> Value { Value::Atom(s.to_string()) }
fn ub(s: &str) -> Value { Value::Unbound(s.to_string()) }
fn fact(name: &str, args: Vec<Value>) -> Value { Value::Str(name.to_string(), args) }

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
            out.push(vm.bindings.get("X").cloned().expect("X bound"));
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
fn read_eof_binds_end_of_file() {
    let mut vm = WamState::new(vec![], HashMap::new());
    vm.set_reg_str("A1", ub("T"));
    assert!(vm.execute_builtin("read/1", 1));
    assert_eq!(vm.bindings.get("T"), Some(&at("end_of_file")));
}
