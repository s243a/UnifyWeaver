:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% wam_rust_target.pl - WAM-to-Rust Transpilation Target
%
% Transpiles WAM runtime predicates (wam_runtime.pl) to Rust code.
% Phase 2: step_wam/3 → match arms
% Phase 3: helper predicates → Rust functions
%
% See: docs/design/WAM_RUST_TRANSPILATION_IMPLEMENTATION_PLAN.md

:- module(wam_rust_target, [
    compile_step_wam_to_rust/2,          % -MatchArms
    compile_wam_helpers_to_rust/2,       % -HelperFunctions
    compile_wam_runtime_to_rust/2,       % +Options, -RustCode
    compile_wam_predicate_to_rust/4,     % +Pred/Arity, +WamCode, +Options, -RustCode
    write_wam_rust_project/3,            % +Predicates, +Options, +ProjectDir
    cargo_check_project/2,               % +ProjectDir, -Result
    detect_kernels/2,                    % +Predicates, -DetectedKernels
    generate_setup_foreign_predicates_rust/2  % +DetectedKernels, -RustCode
]).

:- use_module(library(lists)).
:- use_module(library(option)).
:- use_module('../core/template_system').
:- use_module('../bindings/rust_wam_bindings').
:- use_module('../targets/wam_target', [compile_predicate_to_wam/3]).
:- use_module('../targets/wam_rust_lowered_emitter', [
    wam_rust_lowerable/3,
    lower_predicate_to_rust/4,
    rust_lowered_func_name/2
]).
:- use_module('../core/recursive_kernel_detection', [
    detect_recursive_kernel/4,
    kernel_metadata/4,
    kernel_config/2,
    kernel_register_layout/2,
    kernel_native_call/2
]).

% ============================================================================
% PHASE 2: step_wam/3 → Rust match arms
% ============================================================================

%% compile_step_wam_to_rust(-RustCode)
%  Generates the body of the step() method as a Rust match expression.
%  Each step_wam/3 clause becomes one match arm.
compile_step_wam_to_rust(_Options, RustCode) :-
    findall(Arm, compile_step_arm(Arm), Arms),
    atomic_list_concat(Arms, '\n', ArmsCode),
    format(string(RustCode),
'    pub fn step(&mut self, instr: &Instruction) -> bool {
        match instr {
~w
            _ => false,
        }
    }', [ArmsCode]).

%% compile_step_arm(-ArmCode)
%  Each WAM instruction maps to a match arm.
compile_step_arm(ArmCode) :-
    wam_instruction_arm(InstrPattern, BodyCode),
    format(string(ArmCode),
        '            ~w => {\n~w\n            }', [InstrPattern, BodyCode]).

% --- Head Unification Instructions ---

wam_instruction_arm('Instruction::GetConstant(c, ai)', Body) :-
    Body = '                let raw_val = self.get_reg_raw(ai);
                let val = raw_val.map(|v| self.deref_var(&v));
                match val {
                    Some(v) if v == *c => { self.pc += 1; true }
                    Some(Value::Unbound(ref var_name)) => {
                        self.trail_binding(ai);
                        self.set_reg_str(ai, c.clone());
                        self.bind_var(var_name, c.clone());
                        self.pc += 1;
                        true
                    }
                    _ => false,
                }'.

wam_instruction_arm('Instruction::GetVariable(xn, ai)', Body) :-
    Body = '                if let Some(raw) = self.get_reg_raw(ai) {
                    let val = self.deref_var(&raw);
                    self.trail_binding(xn);
                    self.put_reg(xn, val);
                    self.pc += 1;
                    true
                } else { false }'.

wam_instruction_arm('Instruction::GetValue(xn, ai)', Body) :-
    Body = '                let val_a = self.get_reg_raw(ai);
                let val_x = self.get_reg(xn);
                match (val_a, val_x) {
                    (Some(a), Some(x)) if a == x => { self.pc += 1; true }
                    (Some(a), _) if a.is_unbound() => {
                        self.trail_binding(ai);
                        if let Some(x) = self.get_reg(xn) {
                            self.set_reg_str(ai, x);
                        }
                        self.pc += 1; true
                    }
                    (_, Some(x)) if x.is_unbound() => {
                        self.trail_binding(xn);
                        if let Some(a) = self.get_reg_raw(ai) {
                            self.put_reg(xn, a);
                        }
                        self.pc += 1; true
                    }
                    _ => false,
                }'.

wam_instruction_arm('Instruction::GetStructure(fn_str, ai)', Body) :-
    Body = '                if let Some(val) = self.get_reg_raw(ai) {
                    if val.is_unbound() {
                        // Write mode
                        let addr = self.heap.len();
                        self.heap.push(Value::Str(format!("str({})", fn_str), vec![]));
                        self.trail_binding(ai);
                        self.set_reg_str(ai, Value::Ref(addr));
                        let arity = fn_str.split(\'/\').nth(1)
                            .and_then(|s| s.parse::<usize>().ok()).unwrap_or(0);
                        self.smut().push(StackEntry::WriteCtx(arity));
                        self.pc += 1; true
                    } else if let Value::Ref(addr) = &val {
                        // Read mode on heap ref
                        if let Some(Value::Str(s, _)) = self.heap.get(*addr) {
                            if s == &format!("str({})", fn_str) {
                                let arity = fn_str.split(\'/\').nth(1)
                                    .and_then(|s| s.parse::<usize>().ok()).unwrap_or(0);
                                let args = self.heap_subargs(addr + 1, arity);
                                self.smut().push(StackEntry::UnifyCtx(args));
                                self.pc += 1; true
                            } else { false }
                        } else { false }
                    } else if let Some((f, args)) = val.univ() {
                        // Read mode on Prolog compound
                        let check = format!("{}/{}", f, args.len());
                        if check == *fn_str {
                            self.smut().push(StackEntry::UnifyCtx(args.to_vec()));
                            self.pc += 1; true
                        } else { false }
                    } else { false }
                } else { false }'.

wam_instruction_arm('Instruction::GetList(ai)', Body) :-
    Body = '                if let Some(val) = self.get_reg_raw(ai) {
                    if val.is_unbound() {
                        let addr = self.heap.len();
                        self.heap.push(Value::Str("str(./2)".to_string(), vec![]));
                        self.trail_binding(ai);
                        self.set_reg_str(ai, Value::Ref(addr));
                        self.smut().push(StackEntry::WriteCtx(2));
                        self.pc += 1; true
                    } else if let Value::List(items) = &val {
                        if let Some((head, tail)) = items.split_first() {
                            self.smut().push(StackEntry::UnifyCtx(
                                vec![head.clone(), Value::List(tail.to_vec())]));
                            self.pc += 1; true
                        } else { false }
                    } else if let Value::Ref(addr) = &val {
                        if let Some(Value::Str(s, _)) = self.heap.get(*addr) {
                            if s == "str(./2)" {
                                let args = self.heap_subargs(addr + 1, 2);
                                self.smut().push(StackEntry::UnifyCtx(args));
                                self.pc += 1; true
                            } else { false }
                        } else { false }
                    } else { false }
                } else { false }'.

wam_instruction_arm('Instruction::UnifyVariable(xn)', Body) :-
    Body = '                if let Some(StackEntry::UnifyCtx(args)) = self.stack.last().cloned() {
                    if let Some(arg) = args.first().cloned() {
                        let rest: Vec<Value> = args[1..].to_vec();
                        self.smut().pop();
                        if !rest.is_empty() {
                            self.smut().push(StackEntry::UnifyCtx(rest));
                        }
                        self.trail_binding(xn);
                        self.put_reg(xn, arg);
                        self.pc += 1; true
                    } else { false }
                } else if let Some(StackEntry::WriteCtx(n)) = self.stack.last().cloned() {
                    if n > 0 {
                        let addr = self.heap.len();
                        let var = Value::Unbound(format!("_H{}", addr));
                        self.heap.push(var.clone());
                        self.smut().pop();
                        if n - 1 > 0 { self.smut().push(StackEntry::WriteCtx(n - 1)); }
                        self.put_reg(xn, var);
                        self.pc += 1; true
                    } else { false }
                } else { false }'.

wam_instruction_arm('Instruction::UnifyValue(xn)', Body) :-
    Body = '                if let Some(StackEntry::UnifyCtx(args)) = self.stack.last().cloned() {
                    if let Some(arg) = args.first().cloned() {
                        let val = self.get_reg(xn);
                        let ok = match (&val, &arg) {
                            (Some(v), a) if v == a => true,
                            (Some(v), _) if v.is_unbound() => {
                                self.trail_binding(xn);
                                self.put_reg(xn, arg.clone());
                                true
                            }
                            (_, a) if a.is_unbound() => true,
                            _ => false,
                        };
                        if ok {
                            let rest: Vec<Value> = args[1..].to_vec();
                            self.smut().pop();
                            if !rest.is_empty() {
                                self.smut().push(StackEntry::UnifyCtx(rest));
                            }
                            self.pc += 1; true
                        } else { false }
                    } else { false }
                } else if let Some(StackEntry::WriteCtx(n)) = self.stack.last().cloned() {
                    if n > 0 {
                        if let Some(val) = self.get_reg(xn) {
                            self.heap.push(val);
                            self.smut().pop();
                            if n - 1 > 0 { self.smut().push(StackEntry::WriteCtx(n - 1)); }
                            self.pc += 1; true
                        } else { false }
                    } else { false }
                } else { false }'.

wam_instruction_arm('Instruction::UnifyConstant(c)', Body) :-
    Body = '                if let Some(StackEntry::UnifyCtx(args)) = self.stack.last().cloned() {
                    if let Some(arg) = args.first().cloned() {
                        if arg == *c || arg.is_unbound() {
                            let rest: Vec<Value> = args[1..].to_vec();
                            self.smut().pop();
                            if !rest.is_empty() {
                                self.smut().push(StackEntry::UnifyCtx(rest));
                            }
                            self.pc += 1; true
                        } else { false }
                    } else { false }
                } else if let Some(StackEntry::WriteCtx(n)) = self.stack.last().cloned() {
                    if n > 0 {
                        self.heap.push(c.clone());
                        self.smut().pop();
                        if n - 1 > 0 { self.smut().push(StackEntry::WriteCtx(n - 1)); }
                        self.pc += 1; true
                    } else { false }
                } else { false }'.

% --- Body Construction Instructions ---

wam_instruction_arm('Instruction::PutConstant(c, ai)', Body) :-
    Body = '                self.trail_binding(ai);
                self.set_reg_str(ai, c.clone());
                self.pc += 1; true'.

wam_instruction_arm('Instruction::PutVariable(xn, ai)', Body) :-
    Body = '                let var = Value::Unbound(format!("_V{}", self.var_counter));
                self.var_counter += 1;
                self.trail_binding(xn);
                self.trail_binding(ai);
                self.put_reg(xn, var.clone());
                self.set_reg_str(ai, var);
                self.pc += 1; true'.

wam_instruction_arm('Instruction::PutValue(xn, ai)', Body) :-
    Body = '                if let Some(val) = self.get_reg(xn) {
                    self.trail_binding(ai);
                    self.set_reg_str(ai, val);
                    self.pc += 1; true
                } else { false }'.

wam_instruction_arm('Instruction::PutStructure(fn_str, ai)', Body) :-
    Body = '                // Parse arity from functor string (e.g. "member/2" -> arity=2)
                let arity: usize = fn_str.split(''/'').last()
                    .and_then(|s| s.parse().ok()).unwrap_or(0);
                // Reserve heap slots for the structure header + args
                let addr = self.heap.len();
                self.heap.push(Value::Str(fn_str.clone(), vec![])); // placeholder
                for _ in 0..arity {
                    self.heap.push(Value::Atom("__struct_arg__".to_string()));
                }
                // Enter structure-write mode: next N SetValue/SetConstant calls fill args
                self.smut().push(StackEntry::WriteCtx(addr));
                self.set_reg_str(ai, Value::Ref(addr));
                self.pc += 1; true'.

wam_instruction_arm('Instruction::PutList(ai)', Body) :-
    Body = '                // Enter list-write mode. The next two SetValue/SetConstant
                // instructions provide the head and tail of [H|T].
                // We record which register to store the result in, and
                // use the heap as a scratch area: push a sentinel, then
                // collect two values; after the second, build the list.
                let marker = self.heap.len();
                self.heap.push(Value::Atom("__list_head__".to_string()));
                self.heap.push(Value::Atom("__list_tail__".to_string()));
                self.set_reg_str(ai, Value::Integer(marker as i64));
                self.smut().push(StackEntry::WriteCtx(marker));
                self.pc += 1; true'.

wam_instruction_arm('Instruction::SetVariable(xn)', Body) :-
    Body = '                let addr = self.heap.len();
                let var = Value::Unbound(format!("_H{}", addr));
                self.heap.push(var.clone());
                self.put_reg(xn, var);
                self.pc += 1; true'.

wam_instruction_arm('Instruction::SetValue(xn)', Body) :-
    Body = '                if let Some(val) = self.get_reg(xn) {
                    self.set_heap_or_list(val);
                    self.pc += 1; true
                } else { false }'.

wam_instruction_arm('Instruction::SetConstant(c)', Body) :-
    Body = '                self.set_heap_or_list(c.clone());
                self.pc += 1; true'.

wam_instruction_arm('Instruction::LoadRegisterConstant(c, reg, skip)', Body) :-
    Body = '                self.trail_binding(reg);
                self.put_reg(reg, c.clone());
                self.pc += *skip; true'.

wam_instruction_arm('Instruction::Cons(head_reg, tail_reg, out_reg, skip)', Body) :-
    Body = '                if let (Some(head), Some(tail)) = (self.get_reg(head_reg), self.get_reg(tail_reg)) {
                    self.heap.push(head.clone());
                    self.heap.push(tail.clone());
                    let list = match tail {
                        Value::List(mut items) => { items.insert(0, head); Value::List(items) }
                        tail_val => Value::List(vec![head, tail_val]),
                    };
                    self.set_reg_str(&out_reg, list);
                    self.pc += *skip; true
                } else { false }'.

wam_instruction_arm('Instruction::NotMember(elem_reg, list_reg, skip)', Body) :-
    Body = '                if let (Some(elem), Some(list_val)) = (self.get_reg(elem_reg), self.get_reg(list_reg)) {
                    let needle = self.deref_var(&elem);
                    let haystack = self.deref_var(&list_val);
                    let found = match &haystack {
                        Value::List(items) => items.iter().any(|item| self.deref_var(item) == needle),
                        _ => false,
                    };
                    if found { false } else { self.pc += *skip; true }
                } else { false }'.

wam_instruction_arm('Instruction::ListLengthLt(list_reg, limit_reg, skip)', Body) :-
    Body = '                if let (Some(list_val), Some(limit_val)) = (self.get_reg(list_reg), self.get_reg(limit_reg)) {
                    let len = match self.deref_var(&list_val) {
                        Value::List(items) => items.len() as i64,
                        direct => match self.deref_heap(&direct) {
                            Value::List(items) => items.len() as i64,
                            _ => return false,
                        }
                    };
                    let limit_ok = match self.deref_var(&limit_val) {
                        Value::Integer(n) => len < n,
                        Value::Float(f) => (len as f64) < f,
                        Value::Atom(s) => match s.parse::<f64>() {
                            Ok(f) => (len as f64) < f,
                            Err(_) => false,
                        },
                        _ => false,
                    };
                    if limit_ok { self.pc += *skip; true } else { false }
                } else { false }'.

wam_instruction_arm('Instruction::BaseCategoryAncestor(cat_reg, target_reg, visited_reg)', Body) :-
    Body = '                let cat = match self.get_reg(cat_reg) {
                    Some(Value::Atom(s)) => s,
                    Some(val) => match self.deref_var(&val) {
                        Value::Atom(s) => s,
                        _ => return false,
                    },
                    None => return false,
                };
                let target = match self.get_reg(target_reg) {
                    Some(val) => self.deref_var(&val),
                    None => return false,
                };
                let target_atom = match &target {
                    Value::Atom(s) => s.clone(),
                    _ => return false,
                };
                let visited = match self.get_reg(visited_reg) {
                    Some(val) => self.deref_var(&val),
                    None => return false,
                };
                let already_visited = match &visited {
                    Value::List(items) => items.iter().any(|item| self.deref_var(item) == target),
                    _ => false,
                };
                if already_visited {
                    return false;
                }
                let parent_matches = self.indexed_atom_fact2
                    .get("category_parent/2")
                    .and_then(|table| table.get(&cat))
                    .map(|values| values.iter().any(|parent| parent == &target_atom))
                    .unwrap_or(false);
                if !parent_matches {
                    return false;
                }
                if let Some(StackEntry::Env(old_cp, _)) = self.smut().pop() {
                    self.cp = old_cp;
                    let ret = self.cp;
                    self.cp = 0;
                    self.pc = ret;
                    true
                } else {
                    false
                }'.

wam_instruction_arm('Instruction::RecurseCategoryAncestor(mid_reg, root_reg, child_hops_reg, visited_reg, pred, skip)', Body) :-
    Body = '                let Some(&target_pc) = self.labels.get(pred) else { return false; };
                let instr = Instruction::RecurseCategoryAncestorPc(
                    mid_reg.clone(),
                    root_reg.clone(),
                    child_hops_reg.clone(),
                    visited_reg.clone(),
                    target_pc,
                    *skip,
                );
                self.step(&instr)'.

wam_instruction_arm('Instruction::RecurseCategoryAncestorPc(mid_reg, root_reg, child_hops_reg, visited_reg, target_pc, skip)', Body) :-
    Body = '                let mid = match self.get_reg(mid_reg) {
                    Some(val) => self.deref_var(&val),
                    None => return false,
                };
                let root = match self.get_reg(root_reg) {
                    Some(val) => self.deref_var(&val),
                    None => return false,
                };
                let visited = match self.get_reg(visited_reg) {
                    Some(val) => self.deref_var(&val),
                    None => return false,
                };
                let child_hops = Value::Unbound(format!("_V{}", self.var_counter));
                self.var_counter += 1;
                let next_visited = match visited {
                    Value::List(mut items) => {
                        items.insert(0, mid.clone());
                        Value::List(items)
                    }
                    tail => Value::List(vec![mid.clone(), tail]),
                };
                self.trail_binding(child_hops_reg);
                self.trail_binding("A1");
                self.trail_binding("A2");
                self.trail_binding("A3");
                self.trail_binding("A4");
                self.put_reg(child_hops_reg, child_hops.clone());
                self.set_reg_str("A1", mid);
                self.set_reg_str("A2", root);
                self.set_reg_str("A3", child_hops);
                self.set_reg_str("A4", next_visited);
                self.cp = self.pc + *skip;
                self.pc = *target_pc;
                true'.

wam_instruction_arm('Instruction::ReturnAdd1(out_reg, in_reg)', Body) :-
    Body = '                let in_val = match self.get_reg(in_reg) {
                    Some(val) => self.deref_var(&val),
                    None => return false,
                };
                let result = match in_val {
                    Value::Integer(n) => Value::Integer(n + 1),
                    Value::Float(f) => {
                        let next = f + 1.0;
                        if (next.round() - next).abs() < f64::EPSILON {
                            Value::Integer(next.round() as i64)
                        } else {
                            Value::Float(next)
                        }
                    }
                    Value::Atom(s) => match s.parse::<f64>() {
                        Ok(f) => {
                            let next = f + 1.0;
                            if (next.round() - next).abs() < f64::EPSILON {
                                Value::Integer(next.round() as i64)
                            } else {
                                Value::Float(next)
                            }
                        }
                        Err(_) => return false,
                    },
                    _ => return false,
                };
                let out_val = match self.get_reg(out_reg) {
                    Some(val) => val,
                    None => return false,
                };
                if !self.unify(&out_val, &result) {
                    return false;
                }
                if let Some(StackEntry::Env(old_cp, _)) = self.smut().pop() {
                    self.cp = old_cp;
                    let ret = self.cp;
                    self.cp = 0;
                    self.pc = ret;
                    true
                } else {
                    false
                }'.

% --- Control Instructions ---

wam_instruction_arm('Instruction::Allocate', Body) :-
    Body = '                use std::collections::HashMap;
                self.cut_barrier = self.choice_points.len();
                let saved_cp = self.cp;
                self.smut().push(StackEntry::Env(saved_cp, HashMap::new()));
                self.pc += 1; true'.

wam_instruction_arm('Instruction::Deallocate', Body) :-
    Body = '                if let Some(StackEntry::Env(old_cp, _)) = self.smut().pop() {
                    self.cp = old_cp;
                    self.pc += 1; true
                } else { false }'.

wam_instruction_arm('Instruction::Call(p, _arity)', Body) :-
    Body = '                if let Some(&target_pc) = self.labels.get(p) {
                    self.cp = self.pc + 1;
                    self.pc = target_pc;
                    true
                } else { false }'.

wam_instruction_arm('Instruction::CallPc(target_pc, _arity)', Body) :-
    Body = '                self.cp = self.pc + 1;
                self.pc = *target_pc;
                true'.

wam_instruction_arm('Instruction::CallForeign(pred, arity)', Body) :-
    Body = '                if self.execute_foreign_predicate(pred, *arity) {
                    self.pc += 1;
                    true
                } else { false }'.

wam_instruction_arm('Instruction::CallIndexedAtomFact2(pred)', Body) :-
    Body = '                let key = match self.get_reg_raw("A1").map(|v| self.deref_var(&v)) {
                    Some(Value::Atom(s)) => s,
                    _ => return false,
                };
                let a2 = match self.get_reg_raw("A2") {
                    Some(val) => val,
                    None => return false,
                };
                let values = match self.indexed_atom_fact2.get(pred).and_then(|table| table.get(&key)) {
                    Some(values) if !values.is_empty() => values.clone(),
                    _ => return false,
                };
                if values.len() > 1 {
                    self.choice_points.push(ChoicePoint {
                        next_pc: self.pc,
                        saved_args: self.save_regs(),
                        cp: self.cp,
                        stack: self.stack.clone(),
                        trail_len: self.trail.len(),
                        heap_len: self.heap.len(),
                        builtin_state: Some(BuiltinState {
                            name: "indexed_atom_fact2".to_string(),
                            args: vec![Value::Atom(pred.clone()), Value::Atom(key.clone())],
                            data: vec![Value::Integer(1)],
                        }),
                        cut_barrier: self.cut_barrier,
                    });
                }
                if self.unify(&a2, &Value::Atom(values[0].clone())) {
                    self.pc += 1; true
                } else { false }'.

wam_instruction_arm('Instruction::Execute(p)', Body) :-
    Body = '                if let Some(&target_pc) = self.labels.get(p) {
                    self.pc = target_pc;
                    true
                } else { false }'.

wam_instruction_arm('Instruction::ExecutePc(target_pc)', Body) :-
    Body = '                self.pc = *target_pc;
                true'.

wam_instruction_arm('Instruction::Proceed', Body) :-
    Body = '                let ret = self.cp;
                self.cp = 0; // halt
                self.pc = ret;
                true'.

wam_instruction_arm('Instruction::BuiltinCall(op, arity)', Body) :-
    Body = '                self.execute_builtin(op, *arity)'.

wam_instruction_arm('Instruction::BeginAggregate(agg_type, value_reg, result_reg)', Body) :-
    Body = '                self.aggregate_acc.clear();
                self.choice_points.push(ChoicePoint {
                    next_pc: self.pc,
                    saved_args: self.save_regs(),
                    cp: self.cp,
                    stack: self.stack.clone(),
                    trail_len: self.trail.len(),
                    heap_len: self.heap.len(),
                    builtin_state: Some(BuiltinState {
                        name: "aggregate_frame".to_string(),
                        args: vec![
                            Value::Atom(agg_type.clone()),
                            Value::Atom(value_reg.clone()),
                            Value::Atom(result_reg.clone()),
                        ],
                        data: vec![],
                    }),
                    cut_barrier: self.cut_barrier,
                });
                self.pc += 1; true'.

wam_instruction_arm('Instruction::EndAggregate(value_reg)', Body) :-
    Body = '                if let Some(val) = self.get_reg(value_reg) {
                    self.aggregate_acc.push(self.deref_var(&self.deref_heap(&val)));
                    self.aggregate_return_pc = self.pc + 1;
                    self.backtrack()
                } else { false }'.

% --- Choice Point Instructions ---

wam_instruction_arm('Instruction::TryMeElse(label)', Body) :-
    Body = '                if let Some(&next_pc) = self.labels.get(label) {
                    self.choice_points.push(ChoicePoint {
                        next_pc,
                        saved_args: self.save_regs(),
                        cp: self.cp,
                        stack: self.stack.clone(),
                        trail_len: self.trail.len(),
                        heap_len: self.heap.len(),
                        builtin_state: None,
                        cut_barrier: self.cut_barrier,
                    });
                    self.pc += 1; true
                } else { false }'.

wam_instruction_arm('Instruction::TryMeElsePc(next_pc)', Body) :-
    Body = '                self.choice_points.push(ChoicePoint {
                    next_pc: *next_pc,
                    saved_args: self.save_regs(),
                    cp: self.cp,
                    stack: self.stack.clone(),
                    trail_len: self.trail.len(),
                    heap_len: self.heap.len(),
                    builtin_state: None,
                    cut_barrier: self.cut_barrier,
                });
                self.pc += 1; true'.

wam_instruction_arm('Instruction::TrustMe', Body) :-
    Body = '                self.choice_points.pop();
                self.pc += 1; true'.

wam_instruction_arm('Instruction::RetryMeElse(label)', Body) :-
    Body = '                if let Some(&next_pc) = self.labels.get(label) {
                    if let Some(cp) = self.choice_points.last_mut() {
                        cp.next_pc = next_pc;
                    }
                    self.pc += 1; true
                } else { false }'.

wam_instruction_arm('Instruction::RetryMeElsePc(next_pc)', Body) :-
    Body = '                if let Some(cp) = self.choice_points.last_mut() {
                    cp.next_pc = *next_pc;
                }
                self.pc += 1; true'.

% --- Indexing Instructions ---

wam_instruction_arm('Instruction::SwitchOnConstant(table)', Body) :-
    Body = '                let raw = self.get_reg_raw("A1").map(|v| self.deref_var(&v));
                if let Some(val) = raw {
                    if !val.is_unbound() {
                        // Binary search for Atom keys (table is sorted from BTreeMap)
                        if let Value::Atom(ref needle) = val {
                            match table.binary_search_by_key(&needle.as_str(), |(k, _)| {
                                if let Value::Atom(s) = k { s.as_str() } else { "" }
                            }) {
                                Ok(idx) => {
                                    if let Some(&pc) = self.labels.get(&table[idx].1) {
                                        self.pc = pc; return true;
                                    }
                                }
                                Err(_) => {}
                            }
                        } else {
                            // Fallback linear scan for non-Atom values
                            for (key, label) in table {
                                if *key == val {
                                    if let Some(&pc) = self.labels.get(label) {
                                        self.pc = pc; return true;
                                    }
                                }
                            }
                        }
                        return false;
                    }
                }
                // Unbound A1: skip dispatch, advance to next instruction
                self.pc += 1; true'.

wam_instruction_arm('Instruction::SwitchOnConstantPc(table)', Body) :-
    Body = '                let raw = self.get_reg_raw("A1").map(|v| self.deref_var(&v));
                if let Some(val) = raw {
                    if !val.is_unbound() {
                        if let Value::Atom(ref needle) = val {
                            match table.binary_search_by_key(&needle.as_str(), |(k, _)| {
                                if let Value::Atom(s) = k { s.as_str() } else { "" }
                            }) {
                                Ok(idx) => {
                                    self.pc = table[idx].1;
                                    return true;
                                }
                                Err(_) => {}
                            }
                        } else {
                            for (key, target_pc) in table {
                                if *key == val {
                                    self.pc = *target_pc;
                                    return true;
                                }
                            }
                        }
                        return false;
                    }
                }
                self.pc += 1; true'.

wam_instruction_arm('Instruction::SwitchOnStructure(table)', Body) :-
    Body = '                if let Some(val) = self.get_reg_raw("A1") {
                    if let Some((f, args)) = val.univ() {
                        let key = format!("{}/{}", f, args.len());
                        for (k, label) in table {
                            if *k == key {
                                if let Some(&pc) = self.labels.get(label) {
                                    self.pc = pc; return true;
                                }
                            }
                        }
                    }
                }
                self.pc += 1; true'.

wam_instruction_arm('Instruction::SwitchOnStructurePc(table)', Body) :-
    Body = '                if let Some(val) = self.get_reg_raw("A1") {
                    if let Some((f, args)) = val.univ() {
                        let key = format!("{}/{}", f, args.len());
                        for (k, target_pc) in table {
                            if *k == key {
                                self.pc = *target_pc; return true;
                            }
                        }
                    }
                }
                self.pc += 1; true'.

wam_instruction_arm('Instruction::SwitchOnConstantA2(table)', Body) :-
    Body = '                if let Some(val) = self.get_reg_raw("A2") {
                    if !val.is_unbound() {
                        for (key, label) in table {
                            if *key == val {
                                if let Some(&pc) = self.labels.get(label) {
                                    self.pc = pc; return true;
                                }
                            }
                        }
                    }
                }
                self.pc += 1; true'.

wam_instruction_arm('Instruction::SwitchOnConstantA2Pc(table)', Body) :-
    Body = '                if let Some(val) = self.get_reg_raw("A2") {
                    if !val.is_unbound() {
                        for (key, target_pc) in table {
                            if *key == val {
                                self.pc = *target_pc; return true;
                            }
                        }
                    }
                }
                self.pc += 1; true'.

% ============================================================================
% PHASE 3: Helper predicates → Rust functions
% ============================================================================

%% compile_wam_helpers_to_rust(-RustCode)
%  Generates Rust methods for WAM runtime helpers.
compile_wam_helpers_to_rust(_Options, RustCode) :-
    compile_run_loop_to_rust(RunLoopCode),
    compile_backtrack_to_rust(BacktrackCode),
    compile_unwind_trail_to_rust(UnwindCode),
    compile_execute_builtin_to_rust(BuiltinCode),
    compile_execute_arith_builtin_to_rust(ArithBuiltinCode),
    compile_execute_io_builtin_to_rust(IoBuiltinCode),
    compile_execute_type_builtin_to_rust(TypeBuiltinCode),
    compile_execute_term_builtin_to_rust(TermBuiltinCode),
    compile_execute_meta_builtin_to_rust(MetaBuiltinCode),
    compile_execute_foreign_predicate_to_rust(ForeignCode),
    compile_resume_builtin_to_rust(ResumeCode),
    compile_foreign_result_helpers_to_rust(ForeignResultHelpersCode),
    compile_parse_foreign_tuple_layout_to_rust(ForeignTupleLayoutCode),
    compile_collect_native_category_ancestor_to_rust(NativeAncestorCode),
    compile_compute_native_countdown_sum_to_rust(NativeCountdownSumCode),
    compile_collect_native_list_suffixes_to_rust(NativeListSuffixCode),
    compile_collect_native_transitive_closure_to_rust(NativeClosureCode),
    compile_collect_native_transitive_distance_to_rust(NativeDistanceCode),
    compile_collect_native_transitive_parent_distance_to_rust(NativeParentDistanceCode),
    compile_collect_native_transitive_step_parent_distance_to_rust(NativeStepParentDistanceCode),
    compile_collect_native_weighted_shortest_path_to_rust(NativeWeightedPathCode),
    compile_collect_native_astar_shortest_path_to_rust(NativeAstarPathCode),
    compile_eval_arith_to_rust(ArithCode),
    atomic_list_concat([
        RunLoopCode, '\n\n',
        BacktrackCode, '\n\n',
        UnwindCode, '\n\n',
        BuiltinCode, '\n\n',
        ArithBuiltinCode, '\n\n',
        IoBuiltinCode, '\n\n',
        TypeBuiltinCode, '\n\n',
        TermBuiltinCode, '\n\n',
        MetaBuiltinCode, '\n\n',
        ForeignCode, '\n\n',
        ResumeCode, '\n\n',
        ForeignResultHelpersCode, '\n\n',
        ForeignTupleLayoutCode, '\n\n',
        NativeAncestorCode, '\n\n',
        NativeCountdownSumCode, '\n\n',
        NativeListSuffixCode, '\n\n',
        NativeClosureCode, '\n\n',
        NativeDistanceCode, '\n\n',
        NativeParentDistanceCode, '\n\n',
        NativeStepParentDistanceCode, '\n\n',
        NativeWeightedPathCode, '\n\n',
        NativeAstarPathCode, '\n\n',
        ArithCode
    ], RustCode).

compile_run_loop_to_rust(Code) :-
    Code = '    /// Main execution loop. Runs until halt (pc=0), failure, or step limit.
    pub fn run(&mut self) -> bool {
        loop {
            if self.pc == 0 { return true; }
            if self.step_limit > 0 && self.step_count >= self.step_limit {
                return false;
            }
            if let Some(instr) = self.fetch().cloned() {
                if !self.step(&instr) {
                    if !self.backtrack() { return false; }
                }
                self.step_count += 1;
            } else {
                return false;
            }
        }
    }'.

compile_backtrack_to_rust(Code0) :-
    Code0 = '    /// Restore state from the top choice point without popping it.
    pub fn backtrack(&mut self) -> bool {
        self.backtrack_count += 1;
        while let Some(cp) = self.choice_points.last().cloned() {
            self.pc = cp.next_pc;

            // 1. Unwind bindings from trail entries added since the CP.
            self.unwind_trail_bindings_only(cp.trail_len);

            // 2. Restore stack (full clone), truncate trail and heap.
            self.stack = cp.stack;
            self.trail.truncate(cp.trail_len);
            self.heap.truncate(cp.heap_len);

            // 3. Restore registers and control state.
            // Binding-table changes are restored by trail unwind above.
            self.restore_regs(&cp.saved_args);
            self.cp = cp.cp;
            self.cut_barrier = cp.cut_barrier;

            if let Some(state) = cp.builtin_state {
                self.choice_points.pop();
                if self.resume_builtin(state) {
                    return true;
                }
                continue;
            }
            return true;
        }
        false
    }
'.

compile_unwind_trail_to_rust(Code) :-
    Code = '    /// Undo only binding-table entries from trail entries added since saved_len.
    fn unwind_trail_bindings_only(&mut self, saved_len: usize) {
        if self.trail.len() <= saved_len { return; }
        let new_entries = self.trail.len() - saved_len;
        for entry in self.trail.iter().rev().take(new_entries) {
            if let Some(binding_key) = entry.key.strip_prefix("__binding__") {
                match &entry.old_value {
                    Some(val) => { self.bindings.insert(binding_key.to_string(), val.clone()); }
                    None => { self.bindings.remove(binding_key); }
                }
            }
        }
    }'.

compile_execute_builtin_to_rust(Code) :-
    Code = '    /// Execute a built-in predicate by name.
    pub fn execute_builtin(&mut self, op: &str, arity: usize) -> bool {
        if self.execute_arith_builtin(op, arity) { return true; }
        if self.execute_io_builtin(op, arity) { return true; }
        if self.execute_type_builtin(op, arity) { return true; }
        if self.execute_term_builtin(op, arity) { return true; }
        if self.execute_meta_builtin(op, arity) { return true; }

        match op {
            "true/0" => { self.pc += 1; true }
            "fail/0" => false,
            "!/0" => { self.choice_points.truncate(self.cut_barrier); self.pc += 1; true }
            _ => false,
        }
    }'.

compile_execute_arith_builtin_to_rust(Code) :-
    Code = '    fn execute_arith_builtin(&mut self, op: &str, _arity: usize) -> bool {
        match op {
            "is/2" => {
                let expr = self.get_reg_raw("A2").map(|v| self.deref_var(&v)).unwrap_or(Value::Integer(0));
                if let Some(result) = self.eval_arith(&expr) {
                    let lhs = self.get_reg_raw("A1").map(|v| self.deref_var(&v));
                    // Bind as integer if result is very close to a whole number
                    let final_val = if (result.round() - result).abs() < f64::EPSILON {
                        Value::Integer(result.round() as i64)
                    } else {
                        Value::Float(result)
                    };
                    match lhs {
                        Some(Value::Unbound(ref var_name)) => {
                            self.trail_binding("A1");
                            self.set_reg_str("A1", final_val.clone());
                            self.bind_var(var_name, final_val);
                            self.pc += 1; true
                        }
                        Some(Value::Integer(n)) if (n as f64) == result => {
                            self.pc += 1; true
                        }
                        Some(Value::Float(f)) if f == result => {
                            self.pc += 1; true
                        }
                        _ => false,
                    }
                } else { false }
            }
            ">/2" | "</2" | ">=/2" | "=</2" | "=:=/2" | "=\\\\=/2" => {
                let v1 = self.get_reg_raw("A1").map(|v| self.deref_var(&v)).and_then(|v| self.eval_arith(&v));
                let v2 = self.get_reg_raw("A2").map(|v| self.deref_var(&v)).and_then(|v| self.eval_arith(&v));
                if let (Some(n1), Some(n2)) = (v1, v2) {
                    let ok = match op {
                        ">/2" => n1 > n2,
                        "</2" => n1 < n2,
                        ">=/2" => n1 >= n2,
                        "=</2" => n1 <= n2,
                        "=:=/2" => (n1 - n2).abs() < f64::EPSILON,
                        "=\\\\=/2" => (n1 - n2).abs() >= f64::EPSILON,
                        _ => false,
                    };
                    if ok { self.pc += 1; true } else { false }
                } else { false }
            }
            "==/2" => {
                let v1 = self.get_reg_raw("A1");
                let v2 = self.get_reg_raw("A2");
                if v1 == v2 { self.pc += 1; true } else { false }
            }
            _ => false,
        }
    }'.

compile_execute_io_builtin_to_rust(Code) :-
    Code = '    fn execute_io_builtin(&mut self, op: &str, _arity: usize) -> bool {
        match op {
            "write/1" | "display/1" => {
                // Both use Display for now. Standard Prolog differentiates them:
                // write/1 suppresses quoting, display/1 uses functional notation.
                if let Some(val) = self.get_reg_raw("A1") {
                    let derefed = self.deref_heap(&val);
                    print!("{}", derefed);
                    self.pc += 1; true
                } else { false }
            }
            "nl/0" => { println!(); self.pc += 1; true }
            _ => false,
        }
    }'.

compile_execute_type_builtin_to_rust(Code) :-
    Code = '    fn execute_type_builtin(&mut self, op: &str, _arity: usize) -> bool {
        if let Some(val) = self.get_reg_raw("A1") {
            let ok = match op {
                "atom/1" => matches!(val, Value::Atom(_)),
                "integer/1" => matches!(val, Value::Integer(_)),
                "float/1" => matches!(val, Value::Float(_)),
                "number/1" => val.is_number(),
                "compound/1" => val.is_compound(),
                "var/1" => val.is_unbound(),
                "nonvar/1" => !val.is_unbound(),
                "is_list/1" => val.is_list(),
                _ => return false,
            };
            if ok { self.pc += 1; true } else { false }
        } else { false }
    }'.

compile_execute_term_builtin_to_rust(Code) :-
    Code = '    fn execute_term_builtin(&mut self, op: &str, _arity: usize) -> bool {
        match op {
            "member/2" => {
                if let (Some(val1), Some(val2)) = (self.get_reg_raw("A1"), self.get_reg_raw("A2")) {
                    if let Value::List(items) = self.deref_heap(&val2) {
                        if items.is_empty() { return false; }
                        
                        // Push choice point for the rest of the list
                        if items.len() > 1 {
                            self.choice_points.push(ChoicePoint {
                                next_pc: self.pc,
                                saved_args: self.save_regs(),
                                stack: self.stack.clone(),
                                cp: self.cp,
                                trail_len: self.trail.len(),
                                heap_len: self.heap.len(),
                                builtin_state: Some(BuiltinState {
                                    name: "member/2".to_string(),
                                    args: vec![val1.clone(), val2.clone()],
                                    data: vec![Value::Integer(1)],
                                }),
                                cut_barrier: self.cut_barrier,
                            });
                        }

                        // Try first element
                        if self.unify(&val1, &items[0]) {
                            self.pc += 1; true
                        } else { false }
                    } else { false }
                } else { false }
            }
            "length/2" => {
                let list_val = self.get_reg_raw("A1").unwrap_or(Value::List(vec![]));
                let derefed = self.deref_heap(&list_val);
                let len = match &derefed {
                    Value::List(items) => items.len() as i64,
                    _ => return false,
                };
                let len_val = Value::Integer(len);
                let lhs = self.get_reg_raw("A2").map(|v| self.deref_var(&v));
                match lhs {
                    Some(Value::Unbound(ref var_name)) => {
                        self.trail_binding("A2");
                        self.set_reg_str("A2", len_val.clone());
                        self.bind_var(var_name, len_val);
                        self.pc += 1; true
                    }
                    Some(Value::Integer(n)) if n == len => {
                        self.pc += 1; true
                    }
                    _ => false,
                }
            }
            "append/3" => {
                eprintln!("Warning: append/3 is not yet implemented in WAM-Rust runtime");
                false
            }
            "functor/3" => {
                let t = self.get_reg_raw("A1")
                    .map(|v| self.deref_heap(&self.deref_var(&v)));
                match t {
                    Some(Value::Unbound(ref var_name)) => {
                        // Construct mode: read N (atom) and A (integer).
                        let n = self.get_reg_raw("A2")
                            .map(|v| self.deref_heap(&self.deref_var(&v)));
                        let a = self.get_reg_raw("A3")
                            .map(|v| self.deref_heap(&self.deref_var(&v)));
                        match (n, a) {
                            (Some(name_val), Some(Value::Integer(arity))) if arity >= 0 => {
                                let built = if arity == 0 {
                                    name_val
                                } else if let Value::Atom(fname) = name_val {
                                    let args: Vec<Value> = (0..arity as usize).map(|_| {
                                        self.var_counter += 1;
                                        Value::Unbound(format!("_F{}", self.var_counter))
                                    }).collect();
                                    Value::Str(fname, args)
                                } else { return false; };
                                self.trail_binding("A1");
                                self.set_reg_str("A1", built.clone());
                                self.bind_var(var_name, built);
                                self.pc += 1; true
                            }
                            _ => false,
                        }
                    }
                    Some(t_val) => {
                        // Read mode: extract functor name and arity.
                        let (name, arity): (Value, i64) = match &t_val {
                            Value::Str(f, args) => (Value::Atom(f.clone()), args.len() as i64),
                            Value::List(items) if items.is_empty() =>
                                (Value::Atom("[]".to_string()), 0),
                            Value::List(_) => (Value::Atom(".".to_string()), 2),
                            Value::Atom(s) => (Value::Atom(s.clone()), 0),
                            Value::Integer(_) | Value::Float(_) | Value::Bool(_) =>
                                (t_val.clone(), 0),
                            _ => return false,
                        };
                        if let Some(a2) = self.get_reg_raw("A2") {
                            let derefed = self.deref_var(&self.deref_heap(&a2));
                            if !self.unify(&derefed, &name) { return false; }
                        }
                        if let Some(a3) = self.get_reg_raw("A3") {
                            let derefed = self.deref_var(&self.deref_heap(&a3));
                            if !self.unify(&derefed, &Value::Integer(arity)) { return false; }
                        }
                        self.pc += 1; true
                    }
                    None => false,
                }
            }
            "arg/3" => {
                let n_val = self.get_reg_raw("A1")
                    .map(|v| self.deref_heap(&self.deref_var(&v)));
                let t_val = self.get_reg_raw("A2")
                    .map(|v| self.deref_heap(&self.deref_var(&v)));
                match (n_val, t_val) {
                    (Some(Value::Integer(n)), Some(t)) if n >= 1 => {
                        let idx = (n - 1) as usize;
                        let arg = match &t {
                            Value::Str(_, args) => args.get(idx).cloned(),
                            Value::List(items) if n == 1 && !items.is_empty() =>
                                Some(items[0].clone()),
                            Value::List(items) if n == 2 && !items.is_empty() =>
                                Some(Value::List(items[1..].to_vec())),
                            _ => None,
                        };
                        match arg {
                            Some(a) => {
                                if let Some(a3) = self.get_reg_raw("A3") {
                                    let derefed = self.deref_var(&self.deref_heap(&a3));
                                    if self.unify(&derefed, &a) { self.pc += 1; true }
                                    else { false }
                                } else { false }
                            }
                            None => false,
                        }
                    }
                    _ => false,
                }
            }
            "=../2" => {
                let t_val = self.get_reg_raw("A1")
                    .map(|v| self.deref_heap(&self.deref_var(&v)));
                match t_val {
                    Some(Value::Unbound(ref var_name)) => {
                        // Compose mode: build T from list in A2.
                        let l_val = self.get_reg_raw("A2")
                            .map(|v| self.deref_heap(&self.deref_var(&v)));
                        if let Some(Value::List(items)) = l_val {
                            if items.is_empty() { return false; }
                            let built = if items.len() == 1 {
                                items[0].clone()
                            } else if let Value::Atom(fname) = &items[0] {
                                Value::Str(fname.clone(), items[1..].to_vec())
                            } else { return false; };
                            self.trail_binding("A1");
                            self.set_reg_str("A1", built.clone());
                            self.bind_var(var_name, built);
                            self.pc += 1; true
                        } else { false }
                    }
                    Some(t) => {
                        // Decompose mode: build list from T.
                        let list = match &t {
                            Value::Str(f, args) => {
                                let mut items = vec![Value::Atom(f.clone())];
                                items.extend(args.iter().cloned());
                                Value::List(items)
                            }
                            Value::Atom(_) | Value::Integer(_)
                            | Value::Float(_) | Value::Bool(_) => {
                                Value::List(vec![t.clone()])
                            }
                            Value::List(items) if items.is_empty() => {
                                Value::List(vec![Value::Atom("[]".to_string())])
                            }
                            Value::List(items) => Value::List(vec![
                                Value::Atom(".".to_string()),
                                items[0].clone(),
                                Value::List(items[1..].to_vec()),
                            ]),
                            _ => return false,
                        };
                        if let Some(a2) = self.get_reg_raw("A2") {
                            let derefed = self.deref_var(&self.deref_heap(&a2));
                            if self.unify(&derefed, &list) { self.pc += 1; true }
                            else { false }
                        } else { false }
                    }
                    None => false,
                }
            }
            "copy_term/2" => {
                let t_val = self.get_reg_raw("A1")
                    .map(|v| self.deref_heap(&self.deref_var(&v)));
                if let Some(t) = t_val {
                    // Sharing is preserved via a var_map from source
                    // variable name to the single fresh name used for
                    // ALL occurrences of that source variable. The
                    // counter is bumped once per distinct source var.
                    let mut var_map: std::collections::HashMap<String, String>
                        = std::collections::HashMap::new();
                    let copy = Self::copy_term_walk(
                        &mut self.var_counter, &mut var_map, &t);
                    if let Some(a2) = self.get_reg_raw("A2") {
                        let derefed = self.deref_var(&self.deref_heap(&a2));
                        if self.unify(&derefed, &copy) { self.pc += 1; true }
                        else { false }
                    } else { false }
                } else { false }
            }
            _ => false,
        }
    }

    /// Recursive walker for copy_term/2. Preserves variable sharing
    /// by mapping each source variable name to exactly one fresh
    /// destination variable name via var_map. Non-var values are
    /// rebuilt structurally; atomic values clone as-is.
    fn copy_term_walk(
        counter: &mut usize,
        var_map: &mut std::collections::HashMap<String, String>,
        v: &Value,
    ) -> Value {
        match v {
            Value::Unbound(name) => {
                if let Some(new_name) = var_map.get(name) {
                    Value::Unbound(new_name.clone())
                } else {
                    *counter += 1;
                    let new_name = format!("_C{}", counter);
                    var_map.insert(name.clone(), new_name.clone());
                    Value::Unbound(new_name)
                }
            }
            Value::Str(f, args) => {
                let new_args: Vec<Value> = args.iter()
                    .map(|a| Self::copy_term_walk(counter, var_map, a))
                    .collect();
                Value::Str(f.clone(), new_args)
            }
            Value::List(items) => {
                let new_items: Vec<Value> = items.iter()
                    .map(|i| Self::copy_term_walk(counter, var_map, i))
                    .collect();
                Value::List(new_items)
            }
            _ => v.clone(),
        }
    }'.

compile_execute_foreign_predicate_to_rust(Code) :-
    Code = '    /// Execute a registered foreign predicate by name/arity.
    pub fn execute_foreign_predicate(&mut self, pred: &str, arity: usize) -> bool {
        let pred_key = format!("{}/{}", pred, arity);
        if !self.foreign_predicates.contains(&pred_key) {
            return false;
        }
        let native_kind = match self.foreign_native_kind(&pred_key) {
            Some(kind) => kind,
            None => return false,
        };
        match native_kind {
            "category_ancestor" => {
                let cat = match self.get_reg_raw("A1").map(|v| self.deref_var(&v)) {
                    Some(Value::Atom(cat)) => cat,
                    _ => return false,
                };
                let root = match self.get_reg_raw("A2").map(|v| self.deref_var(&v)) {
                    Some(Value::Atom(root)) => root,
                    _ => return false,
                };
                let hops_reg = match self.get_reg_raw("A3") {
                    Some(val) => val,
                    None => return false,
                };
                let visited = match self.get_reg_raw("A4").map(|v| self.deref_var(&v)) {
                    Some(Value::List(items)) => items,
                    Some(_) => return false,
                    None => return false,
                };
                let max_depth = match self.foreign_usize_config(&pred_key, "max_depth") {
                    Some(limit) => limit,
                    None => return false,
                };
                let edge_pred = match self.foreign_string_config(&pred_key, "edge_pred") {
                    Some(pred) => pred.to_string(),
                    None => return false,
                };
                let cat_id = self.intern_atom(&cat);
                let root_id = self.intern_atom(&root);
                let visited_ids: Vec<u32> = visited.iter().filter_map(|item| {
                    match self.deref_var(item) {
                        Value::Atom(s) => Some(self.intern_atom(&s)),
                        _ => None,
                    }
                }).collect();
                let mut hops: Vec<i64> = Vec::new();
                self.collect_native_category_ancestor_hops(cat_id, root_id, &visited_ids, max_depth, &edge_pred, &mut hops);
                if hops.is_empty() {
                    return false;
                }
                let results: Vec<Value> = hops.into_iter().map(|hop| {
                    Value::Str("__tuple__".to_string(), vec![Value::Integer(hop)])
                }).collect();
                self.finish_foreign_results(&pred_key, vec![hops_reg], results)
            }
            "countdown_sum2" => {
                let n = match self.get_reg_raw("A1").map(|v| self.deref_var(&v)) {
                    Some(Value::Integer(n)) => n,
                    _ => return false,
                };
                let sum_reg = match self.get_reg_raw("A2") {
                    Some(val) => val,
                    None => return false,
                };
                let sum = match self.compute_native_countdown_sum(n) {
                    Some(sum) => sum,
                    None => return false,
                };
                self.finish_foreign_results(&pred_key, vec![sum_reg], vec![
                    Value::Str("__tuple__".to_string(), vec![Value::Integer(sum)])
                ])
            }
            "list_suffix2" => {
                let items = match self.get_reg_raw("A1").map(|v| self.deref_var(&v)) {
                    Some(Value::List(items)) => items,
                    _ => return false,
                };
                let suffix_reg = match self.get_reg_raw("A2") {
                    Some(val) => val,
                    None => return false,
                };
                let suffix_filter = match self.deref_var(&suffix_reg) {
                    Value::List(items) => Some(items),
                    Value::Unbound(_) => None,
                    _ => return false,
                };
                let mut suffixes: Vec<Value> = Vec::new();
                self.collect_native_list_suffixes(&items, &mut suffixes);
                if let Some(filter) = suffix_filter {
                    suffixes.retain(|suffix| matches!(suffix, Value::List(items) if *items == filter));
                }
                if suffixes.is_empty() {
                    return false;
                }
                let results: Vec<Value> = suffixes.into_iter().map(|suffix| {
                    Value::Str("__tuple__".to_string(), vec![suffix])
                }).collect();
                self.finish_foreign_results(&pred_key, vec![suffix_reg], results)
            }
            "list_suffixes2" => {
                let items = match self.get_reg_raw("A1").map(|v| self.deref_var(&v)) {
                    Some(Value::List(items)) => items,
                    _ => return false,
                };
                let suffixes_reg = match self.get_reg_raw("A2") {
                    Some(val) => val,
                    None => return false,
                };
                let mut suffixes: Vec<Value> = Vec::new();
                self.collect_native_list_suffixes(&items, &mut suffixes);
                let result = Value::Str("__tuple__".to_string(), vec![Value::List(suffixes)]);
                self.finish_foreign_results(&pred_key, vec![suffixes_reg], vec![result])
            }
            "transitive_closure2" => {
                let start = match self.get_reg_raw("A1").map(|v| self.deref_var(&v)) {
                    Some(Value::Atom(start)) => start,
                    _ => return false,
                };
                let target_reg = match self.get_reg_raw("A2") {
                    Some(val) => val,
                    None => return false,
                };
                let target_filter = match self.deref_var(&target_reg) {
                    Value::Atom(target) => Some(target),
                    Value::Unbound(_) => None,
                    _ => return false,
                };
                let edge_pred = match self.foreign_string_config(&pred_key, "edge_pred") {
                    Some(pred) => pred.to_string(),
                    None => return false,
                };
                let mut nodes: Vec<String> = Vec::new();
                self.collect_native_transitive_closure_nodes(&start, &edge_pred, &mut nodes);
                if let Some(target) = target_filter {
                    nodes.retain(|node| *node == target);
                }
                if nodes.is_empty() {
                    return false;
                }
                let results: Vec<Value> = nodes.into_iter().map(|node| {
                    Value::Str("__tuple__".to_string(), vec![Value::Atom(node)])
                }).collect();
                self.finish_foreign_results(&pred_key, vec![target_reg], results)
            }
            "transitive_distance3" => {
                let start = match self.get_reg_raw("A1").map(|v| self.deref_var(&v)) {
                    Some(Value::Atom(start)) => start,
                    _ => return false,
                };
                let target_reg = match self.get_reg_raw("A2") {
                    Some(val) => val,
                    None => return false,
                };
                let dist_reg = match self.get_reg_raw("A3") {
                    Some(val) => val,
                    None => return false,
                };
                let target_filter = match self.deref_var(&target_reg) {
                    Value::Atom(target) => Some(target),
                    Value::Unbound(_) => None,
                    _ => return false,
                };
                let edge_pred = match self.foreign_string_config(&pred_key, "edge_pred") {
                    Some(pred) => pred.to_string(),
                    None => return false,
                };
                let mut results: Vec<(String, i64)> = Vec::new();
                self.collect_native_transitive_distance_results(&start, &edge_pred, &mut results);
                if let Some(target) = target_filter {
                    results.retain(|(node, _)| *node == target);
                }
                if results.is_empty() {
                    return false;
                }
                let packed_results: Vec<Value> = results.into_iter().map(|(node, dist)| {
                    Value::Str("__tuple__".to_string(), vec![
                        Value::Atom(node),
                        Value::Integer(dist),
                    ])
                }).collect();
                self.finish_foreign_results(&pred_key, vec![target_reg, dist_reg], packed_results)
            }
            "transitive_parent_distance4" => {
                let start = match self.get_reg_raw("A1").map(|v| self.deref_var(&v)) {
                    Some(Value::Atom(start)) => start,
                    _ => return false,
                };
                let target_reg = match self.get_reg_raw("A2") {
                    Some(val) => val,
                    None => return false,
                };
                let parent_reg = match self.get_reg_raw("A3") {
                    Some(val) => val,
                    None => return false,
                };
                let dist_reg = match self.get_reg_raw("A4") {
                    Some(val) => val,
                    None => return false,
                };
                let target_filter = match self.deref_var(&target_reg) {
                    Value::Atom(target) => Some(target),
                    Value::Unbound(_) => None,
                    _ => return false,
                };
                let edge_pred = match self.foreign_string_config(&pred_key, "edge_pred") {
                    Some(pred) => pred.to_string(),
                    None => return false,
                };
                let mut results: Vec<(String, String, i64)> = Vec::new();
                self.collect_native_transitive_parent_distance_results(&start, &edge_pred, &mut results);
                if let Some(target) = target_filter {
                    results.retain(|(node, _, _)| *node == target);
                }
                if results.is_empty() {
                    return false;
                }
                let packed_results: Vec<Value> = results.into_iter().map(|(node, parent, dist)| {
                    Value::Str("__tuple__".to_string(), vec![
                        Value::Atom(node),
                        Value::Atom(parent),
                        Value::Integer(dist),
                    ])
                }).collect();
                self.finish_foreign_results(&pred_key, vec![target_reg, parent_reg, dist_reg], packed_results)
            }
            "transitive_step_parent_distance5" => {
                let start = match self.get_reg_raw("A1").map(|v| self.deref_var(&v)) {
                    Some(Value::Atom(start)) => start,
                    _ => return false,
                };
                let target_reg = match self.get_reg_raw("A2") {
                    Some(val) => val,
                    None => return false,
                };
                let step_reg = match self.get_reg_raw("A3") {
                    Some(val) => val,
                    None => return false,
                };
                let parent_reg = match self.get_reg_raw("A4") {
                    Some(val) => val,
                    None => return false,
                };
                let dist_reg = match self.get_reg_raw("A5") {
                    Some(val) => val,
                    None => return false,
                };
                let target_filter = match self.deref_var(&target_reg) {
                    Value::Atom(target) => Some(target),
                    Value::Unbound(_) => None,
                    _ => return false,
                };
                let edge_pred = match self.foreign_string_config(&pred_key, "edge_pred") {
                    Some(pred) => pred.to_string(),
                    None => return false,
                };
                let mut results: Vec<(String, String, String, i64)> = Vec::new();
                self.collect_native_transitive_step_parent_distance_results(&start, &edge_pred, &mut results);
                if let Some(target) = target_filter {
                    results.retain(|(node, _, _, _)| *node == target);
                }
                if results.is_empty() {
                    return false;
                }
                let packed_results: Vec<Value> = results.into_iter().map(|(node, step, parent, dist)| {
                    Value::Str("__tuple__".to_string(), vec![
                        Value::Atom(node),
                        Value::Atom(step),
                        Value::Atom(parent),
                        Value::Integer(dist),
                    ])
                }).collect();
                self.finish_foreign_results(&pred_key, vec![target_reg, step_reg, parent_reg, dist_reg], packed_results)
            }
            "weighted_shortest_path3" => {
                let start = match self.get_reg_raw("A1").map(|v| self.deref_var(&v)) {
                    Some(Value::Atom(start)) => start,
                    _ => return false,
                };
                let target_reg = match self.get_reg_raw("A2") {
                    Some(val) => val,
                    None => return false,
                };
                let dist_reg = match self.get_reg_raw("A3") {
                    Some(val) => val,
                    None => return false,
                };
                let target_filter = match self.deref_var(&target_reg) {
                    Value::Atom(target) => Some(target),
                    Value::Unbound(_) => None,
                    _ => return false,
                };
                let weight_pred = match self.foreign_string_config(&pred_key, "weight_pred") {
                    Some(pred) => pred.to_string(),
                    None => return false,
                };
                let mut results: Vec<(String, f64)> = Vec::new();
                self.collect_native_weighted_shortest_path_results(&start, &weight_pred, &mut results);
                if let Some(target) = target_filter {
                    results.retain(|(node, _)| *node == target);
                }
                if results.is_empty() {
                    return false;
                }
                let packed_results: Vec<Value> = results.into_iter().map(|(node, dist)| {
                    Value::Str("__tuple__".to_string(), vec![
                        Value::Atom(node),
                        Value::Float(dist),
                    ])
                }).collect();
                self.finish_foreign_results(&pred_key, vec![target_reg, dist_reg], packed_results)
            }
            "astar_shortest_path4" => {
                let start = match self.get_reg_raw("A1").map(|v| self.deref_var(&v)) {
                    Some(Value::Atom(start)) => start,
                    _ => return false,
                };
                let target_reg = match self.get_reg_raw("A2") {
                    Some(val) => val,
                    None => return false,
                };
                let dim_val = match self.get_reg_raw("A3").map(|v| self.deref_var(&v)) {
                    Some(Value::Integer(d)) => d as f64,
                    Some(Value::Float(d)) => d,
                    _ => 5.0,  // default dimensionality
                };
                let dist_reg = match self.get_reg_raw("A4") {
                    Some(val) => val,
                    None => return false,
                };
                let target_filter = match self.deref_var(&target_reg) {
                    Value::Atom(target) => Some(target),
                    Value::Unbound(_) => None,
                    _ => return false,
                };
                let weight_pred = match self.foreign_string_config(&pred_key, "weight_pred") {
                    Some(pred) => pred.to_string(),
                    None => return false,
                };
                let direct_pred = match self.foreign_string_config(&pred_key, "direct_dist_pred") {
                    Some(pred) => pred.to_string(),
                    None => return false,
                };
                let target_node = target_filter.as_deref().unwrap_or("");
                let mut results: Vec<(String, f64)> = Vec::new();
                self.collect_native_astar_shortest_path_results(
                    &start, &weight_pred, &direct_pred, target_node, dim_val, &mut results);
                if let Some(target) = target_filter {
                    results.retain(|(node, _)| *node == target);
                }
                if results.is_empty() {
                    return false;
                }
                let packed_results: Vec<Value> = results.into_iter().map(|(node, dist)| {
                    Value::Str("__tuple__".to_string(), vec![
                        Value::Atom(node),
                        Value::Float(dist),
                    ])
                }).collect();
                self.finish_foreign_results(&pred_key, vec![target_reg, dist_reg], packed_results)
            }
            _ => false,
        }
    }'.

compile_foreign_result_helpers_to_rust(Code) :-
    Code = '    pub fn finish_foreign_results(
        &mut self,
        pred_key: &str,
        result_regs: Vec<Value>,
        results: Vec<Value>,
    ) -> bool {
        let result_mode = match self.foreign_result_mode(pred_key) {
            Some(mode) => mode,
            None => "stream",
        };
        if results.is_empty() {
            return false;
        }
        match result_mode {
            "stream" => {
                if results.len() > 1 {
                    self.choice_points.push(ChoicePoint {
                        next_pc: self.pc,
                        saved_args: self.save_regs(),
                        stack: self.stack.clone(),
                        cp: self.cp,
                        trail_len: self.trail.len(),
                        heap_len: self.heap.len(),
                        builtin_state: Some(BuiltinState {
                            name: "foreign_results".to_string(),
                            args: {
                                let mut args = Vec::with_capacity(result_regs.len() + 1);
                                args.push(Value::Atom(pred_key.to_string()));
                                args.extend(result_regs.iter().cloned());
                                args
                            },
                            data: results[1..].to_vec(),
                        }),
                        cut_barrier: self.cut_barrier,
                    });
                }
                self.apply_foreign_result(pred_key, &result_regs, &results[0])
            }
            "deterministic" | "deterministic_collection" => {
                if results.len() != 1 {
                    return false;
                }
                self.apply_foreign_result(pred_key, &result_regs, &results[0])
            }
            _ => false,
        }
    }

    fn apply_foreign_result(
        &mut self,
        pred_key: &str,
        result_regs: &[Value],
        result: &Value,
    ) -> bool {
        let tuple_arity = match self.foreign_result_layout(pred_key)
            .and_then(Self::parse_foreign_tuple_layout) {
            Some(arity) => arity,
            None => return false,
        };
        if result_regs.len() != tuple_arity {
            return false;
        }
        match result {
            Value::Str(functor, args) if functor == "__tuple__" && args.len() == tuple_arity => {
                for (reg, arg) in result_regs.iter().zip(args.iter()) {
                    if !self.unify(reg, arg) {
                        return false;
                    }
                }
                true
            }
            _ => false,
        }
    }'.

compile_parse_foreign_tuple_layout_to_rust(Code) :-
    Code = '    fn parse_foreign_tuple_layout(layout: &str) -> Option<usize> {
        layout.strip_prefix("tuple:")?.parse::<usize>().ok()
    }'.

compile_collect_native_category_ancestor_to_rust(Code) :-
    Code = '    pub fn collect_native_category_ancestor_hops(
        &self,
        cat_id: u32,
        root_id: u32,
        visited: &[u32],
        max_depth: usize,
        edge_pred: &str,
        out: &mut Vec<i64>,
    ) {
        let ffi_table = self.ffi_facts.get(edge_pred);
        let root_seen = visited.contains(&root_id);
        if !root_seen {
            if let Some(values) = ffi_table.and_then(|table| table.get(&cat_id)) {
                if values.contains(&root_id) {
                    out.push(1);
                }
            }
        }

        if visited.len() >= max_depth {
            return;
        }

        if let Some(values) = ffi_table.and_then(|table| table.get(&cat_id)) {
            let values = values.clone();
            for parent_id in &values {
                if visited.contains(parent_id) {
                    continue;
                }
                let mut next_visited: Vec<u32> = Vec::with_capacity(visited.len() + 1);
                next_visited.push(*parent_id);
                next_visited.extend_from_slice(visited);
                let before = out.len();
                self.collect_native_category_ancestor_hops(*parent_id, root_id, &next_visited, max_depth, edge_pred, out);
                for hop in out.iter_mut().skip(before) {
                    *hop += 1;
                }
            }
        }
    }'.

compile_compute_native_countdown_sum_to_rust(Code) :-
    Code = '    pub fn compute_native_countdown_sum(&self, n: i64) -> Option<i64> {
        if n < 0 {
            return None;
        }
        let mut total = 0;
        let mut current = n;
        while current > 0 {
            total += current;
            current -= 1;
        }
        Some(total)
    }'.

compile_collect_native_list_suffixes_to_rust(Code) :-
    Code = '    pub fn collect_native_list_suffixes(
        &self,
        items: &[Value],
        out: &mut Vec<Value>,
    ) {
        for idx in 0..=items.len() {
            out.push(Value::List(items[idx..].to_vec()));
        }
    }'.

compile_collect_native_transitive_closure_to_rust(Code) :-
    Code = '    pub fn collect_native_transitive_closure_nodes(
        &self,
        start: &str,
        edge_pred: &str,
        out: &mut Vec<String>,
    ) {
        let mut seen: HashSet<String> = HashSet::new();
        let mut stack: Vec<String> = vec![start.to_string()];
        while let Some(node) = stack.pop() {
            if let Some(next_nodes) = self.indexed_atom_fact2.get(edge_pred).and_then(|table| table.get(&node)) {
                for next in next_nodes.iter().rev() {
                    if seen.insert(next.clone()) {
                        out.push(next.clone());
                        stack.push(next.clone());
                    }
                }
            }
        }
    }'.

compile_collect_native_transitive_distance_to_rust(Code) :-
    Code = '    pub fn collect_native_transitive_distance_results(
        &self,
        start: &str,
        edge_pred: &str,
        out: &mut Vec<(String, i64)>,
    ) {
        let mut stack: Vec<(String, i64, Vec<String>)> =
            vec![(start.to_string(), 0, vec![start.to_string()])];
        while let Some((node, depth, path)) = stack.pop() {
            if let Some(next_nodes) = self.indexed_atom_fact2.get(edge_pred).and_then(|table| table.get(&node)) {
                for next in next_nodes.iter().rev() {
                    if path.iter().any(|seen| seen == next) {
                        continue;
                    }
                    let next_depth = depth + 1;
                    out.push((next.clone(), next_depth));
                    let mut next_path = path.clone();
                    next_path.push(next.clone());
                    stack.push((next.clone(), next_depth, next_path));
                }
            }
        }
    }'.

compile_collect_native_transitive_parent_distance_to_rust(Code) :-
    Code = '    pub fn collect_native_transitive_parent_distance_results(
        &self,
        start: &str,
        edge_pred: &str,
        out: &mut Vec<(String, String, i64)>,
    ) {
        let mut stack: Vec<(String, i64)> = vec![(start.to_string(), 0)];
        while let Some((node, depth)) = stack.pop() {
            if let Some(next_nodes) = self.indexed_atom_fact2.get(edge_pred).and_then(|table| table.get(&node)) {
                for next in next_nodes.iter().rev() {
                    let next_depth = depth + 1;
                    out.push((next.clone(), node.clone(), next_depth));
                    stack.push((next.clone(), next_depth));
                }
            }
        }
    }'.

compile_collect_native_transitive_step_parent_distance_to_rust(Code) :-
    Code = '    pub fn collect_native_transitive_step_parent_distance_results(
        &self,
        start: &str,
        edge_pred: &str,
        out: &mut Vec<(String, String, String, i64)>,
    ) {
        if let Some(next_nodes) = self.indexed_atom_fact2.get(edge_pred).and_then(|table| table.get(start)) {
            for next in next_nodes {
                out.push((next.clone(), next.clone(), start.to_string(), 1));
                let mut nested: Vec<(String, String, i64)> = Vec::new();
                self.collect_native_transitive_parent_distance_results(next, edge_pred, &mut nested);
                for (target, parent, dist) in nested {
                    out.push((target, next.clone(), parent, dist + 1));
                }
            }
        }
    }'.

compile_collect_native_weighted_shortest_path_to_rust(Code) :-
    Code = '    /// Dijkstra shortest path with precomputed semantic edge weights.
    /// Returns the minimum-cost path from start to each reachable node.
    /// Edge weights are f64 (typically 1 - cosine_similarity).
    /// Uses a BinaryHeap priority queue — greedy expansion naturally
    /// follows the lowest-cost (most semantically similar) path first,
    /// with backtracking to alternatives when needed.
    pub fn collect_native_weighted_shortest_path_results(
        &self,
        start: &str,
        weight_pred: &str,
        out: &mut Vec<(String, f64)>,
    ) {
        use std::collections::BinaryHeap;
        use std::cmp::Ordering;

        // Min-heap entry (Rust BinaryHeap is max-heap, so reverse ordering)
        #[derive(PartialEq)]
        struct State { cost: f64, node: String }
        impl Eq for State {}
        impl PartialOrd for State {
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                other.cost.partial_cmp(&self.cost) // reversed for min-heap
            }
        }
        impl Ord for State {
            fn cmp(&self, other: &Self) -> Ordering {
                self.partial_cmp(other).unwrap_or(Ordering::Equal)
            }
        }

        let mut dist: std::collections::HashMap<String, f64> = std::collections::HashMap::new();
        let mut heap = BinaryHeap::new();

        dist.insert(start.to_string(), 0.0);
        heap.push(State { cost: 0.0, node: start.to_string() });

        while let Some(State { cost, node }) = heap.pop() {
            // Skip if we already found a shorter path
            if let Some(&best) = dist.get(&node) {
                if cost > best { continue; }
            }

            // Explore weighted edges from this node
            if let Some(edges) = self.indexed_weighted_edge.get(weight_pred)
                .and_then(|table| table.get(&node))
            {
                for (next, weight) in edges {
                    let next_cost = cost + weight;
                    let is_shorter = match dist.get(next) {
                        Some(&prev_best) => next_cost < prev_best,
                        None => true,
                    };
                    if is_shorter {
                        dist.insert(next.clone(), next_cost);
                        heap.push(State { cost: next_cost, node: next.clone() });
                    }
                }
            }
        }

        // Collect all reachable nodes (excluding start)
        for (node, cost) in &dist {
            if node != start {
                out.push((node.clone(), *cost));
            }
        }
    }'.

compile_collect_native_astar_shortest_path_to_rust(Code) :-
    Code = '    /// A* shortest path with dimensionality-aware heuristic.
    /// Priority: f(n) = g(n)^D + h(n)^D where D = graph dimensionality.
    /// h(n) = direct semantic distance from n to target.
    /// By Minkowski inequality this is admissible and tighter than L1 A*.
    /// The power D amplifies the effect of short remaining distances,
    /// matching the intrinsic dimensionality of the graph structure.
    pub fn collect_native_astar_shortest_path_results(
        &self,
        start: &str,
        weight_pred: &str,
        direct_pred: &str,
        target: &str,
        dim: f64,
        out: &mut Vec<(String, f64)>,
    ) {
        use std::collections::BinaryHeap;
        use std::cmp::Ordering;

        #[derive(PartialEq)]
        struct State { f_cost: f64, g_cost: f64, node: String }
        impl Eq for State {}
        impl PartialOrd for State {
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                other.f_cost.partial_cmp(&self.f_cost) // reversed for min-heap
            }
        }
        impl Ord for State {
            fn cmp(&self, other: &Self) -> Ordering {
                self.partial_cmp(other).unwrap_or(Ordering::Equal)
            }
        }

        // Heuristic: h(n) = direct semantic distance from n to target
        let heuristic = |node: &str| -> f64 {
            if target.is_empty() { return 0.0; } // no target filter = Dijkstra
            self.indexed_weighted_edge.get(direct_pred)
                .and_then(|table| table.get(node))
                .and_then(|edges| edges.iter().find(|(t, _)| t == target))
                .map(|(_, w)| *w)
                .unwrap_or(1.0) // conservative default if no direct distance
        };

        // f(n) = g^D + h^D
        let f_cost = |g: f64, h: f64| -> f64 {
            g.powf(dim) + h.powf(dim)
        };

        let mut dist: std::collections::HashMap<String, f64> = std::collections::HashMap::new();
        let mut heap = BinaryHeap::new();

        let h_start = heuristic(start);
        dist.insert(start.to_string(), 0.0);
        heap.push(State { f_cost: f_cost(0.0, h_start), g_cost: 0.0, node: start.to_string() });

        while let Some(State { f_cost: _, g_cost, node }) = heap.pop() {
            // Skip if we already found a shorter path
            if let Some(&best) = dist.get(&node) {
                if g_cost > best { continue; }
            }

            // Early termination: if we reached the target, no need to explore further
            if !target.is_empty() && node == target {
                break;
            }

            // Explore weighted edges
            if let Some(edges) = self.indexed_weighted_edge.get(weight_pred)
                .and_then(|table| table.get(&node))
            {
                for (next, weight) in edges {
                    let next_g = g_cost + weight;
                    let is_shorter = match dist.get(next) {
                        Some(&prev_best) => next_g < prev_best,
                        None => true,
                    };
                    if is_shorter {
                        dist.insert(next.clone(), next_g);
                        let h = heuristic(next);
                        heap.push(State {
                            f_cost: f_cost(next_g, h),
                            g_cost: next_g,
                            node: next.clone(),
                        });
                    }
                }
            }
        }

        // Collect results
        for (node, cost) in &dist {
            if node != start {
                out.push((node.clone(), *cost));
            }
        }
    }'.

compile_resume_builtin_to_rust(Code) :-
    Code = '    fn resume_builtin(&mut self, state: BuiltinState) -> bool {
        match state.name.as_str() {
            "aggregate_frame" => {
                if state.args.len() != 3 { return false; }
                let agg_type = match &state.args[0] {
                    Value::Atom(s) => s.as_str(),
                    _ => return false,
                };
                let result_reg = match &state.args[2] {
                    Value::Atom(s) => s.clone(),
                    _ => return false,
                };

                let result = match agg_type {
                    "sum" => {
                        let mut sum_i: i64 = 0;
                        let mut sum_f: f64 = 0.0;
                        let mut saw_float = false;
                        for val in &self.aggregate_acc {
                            match self.deref_var(&self.deref_heap(val)) {
                                Value::Integer(n) => {
                                    sum_i += n;
                                    sum_f += n as f64;
                                }
                                Value::Float(f) => {
                                    saw_float = true;
                                    sum_f += f;
                                }
                                _ => {}
                            }
                        }
                        if saw_float { Value::Float(sum_f) } else { Value::Integer(sum_i) }
                    }
                    "count" => Value::Integer(self.aggregate_acc.len() as i64),
                    "collect" => Value::List(self.aggregate_acc.clone()),
                    "max" => {
                        let mut best: Option<Value> = None;
                        for val in &self.aggregate_acc {
                            let current = self.deref_var(&self.deref_heap(val));
                            best = match best {
                                None => Some(current),
                                Some(ref prev) => {
                                    match (&current, prev) {
                                        (Value::Integer(a), Value::Integer(b)) if a > b => Some(current),
                                        (Value::Float(a), Value::Float(b)) if a > b => Some(current),
                                        (Value::Integer(a), Value::Float(b)) if (*a as f64) > *b => Some(current),
                                        (Value::Float(a), Value::Integer(b)) if *a > (*b as f64) => Some(current),
                                        _ => Some(prev.clone()),
                                    }
                                }
                            };
                        }
                        best.unwrap_or(Value::List(vec![]))
                    }
                    "min" => {
                        let mut best: Option<Value> = None;
                        for val in &self.aggregate_acc {
                            let current = self.deref_var(&self.deref_heap(val));
                            best = match best {
                                None => Some(current),
                                Some(ref prev) => {
                                    match (&current, prev) {
                                        (Value::Integer(a), Value::Integer(b)) if a < b => Some(current),
                                        (Value::Float(a), Value::Float(b)) if a < b => Some(current),
                                        (Value::Integer(a), Value::Float(b)) if (*a as f64) < *b => Some(current),
                                        (Value::Float(a), Value::Integer(b)) if *a < (*b as f64) => Some(current),
                                        _ => Some(prev.clone()),
                                    }
                                }
                            };
                        }
                        best.unwrap_or(Value::List(vec![]))
                    }
                    _ => return false,
                };

                self.aggregate_acc.clear();
                let lhs = self.get_reg_raw(&result_reg).map(|v| self.deref_var(&v));
                match lhs {
                    Some(Value::Unbound(ref var_name)) => {
                        self.trail_binding(&result_reg);
                        self.set_reg_str(&result_reg, result.clone());
                        self.bind_var(var_name, result);
                    }
                    Some(existing) if existing == result => {}
                    Some(_) => return false,
                    None => {
                        self.set_reg_str(&result_reg, result);
                    }
                }
                self.pc = self.aggregate_return_pc;
                true
            }
            "member/2" => {
                let val1 = state.args[0].clone();
                let val2 = state.args[1].clone();
                let idx = match state.data[0] {
                    Value::Integer(n) => n as usize,
                    _ => return false,
                };
                
                if let Value::List(items) = self.deref_heap(&val2) {
                    if idx >= items.len() { return false; }
                    
                    // Push choice point for the rest of the list if any
                    if idx + 1 < items.len() {
                        self.choice_points.push(ChoicePoint {
                            next_pc: self.pc,
                            saved_args: self.save_regs(),
                            stack: self.stack.clone(),
                            cp: self.cp,
                            trail_len: self.trail.len(),
                            heap_len: self.heap.len(),
                            builtin_state: Some(BuiltinState {
                                name: "member/2".to_string(),
                                args: state.args.clone(),
                                data: vec![Value::Integer((idx + 1) as i64)],
                            }),
                            cut_barrier: self.cut_barrier,
                        });
                    }
                    
                    // Try current element
                    if self.unify(&val1, &items[idx]) {
                        self.pc += 1; true
                    } else { false }
                } else { false }
            }
            "indexed_atom_fact2" => {
                let pred = match state.args.get(0) {
                    Some(Value::Atom(pred)) => pred.clone(),
                    _ => return false,
                };
                let key = match state.args.get(1) {
                    Some(Value::Atom(key)) => key.clone(),
                    _ => return false,
                };
                let idx = match state.data.get(0) {
                    Some(Value::Integer(n)) => *n as usize,
                    _ => return false,
                };
                let values = match self.indexed_atom_fact2.get(&pred).and_then(|table| table.get(&key)) {
                    Some(values) => values,
                    None => return false,
                };
                if idx >= values.len() { return false; }
                if idx + 1 < values.len() {
                    self.choice_points.push(ChoicePoint {
                        next_pc: self.pc,
                        saved_args: self.save_regs(),
                        stack: self.stack.clone(),
                        cp: self.cp,
                        trail_len: self.trail.len(),
                        heap_len: self.heap.len(),
                        builtin_state: Some(BuiltinState {
                            name: "indexed_atom_fact2".to_string(),
                            args: state.args.clone(),
                            data: vec![Value::Integer((idx + 1) as i64)],
                        }),
                        cut_barrier: self.cut_barrier,
                    });
                }
                let a2 = match self.get_reg_raw("A2") {
                    Some(val) => val,
                    None => return false,
                };
                if self.unify(&a2, &Value::Atom(values[idx].clone())) {
                    self.pc += 1; true
                } else { false }
            }
            "foreign_results" => {
                let pred_key = match state.args.get(0) {
                    Some(Value::Atom(pred_key)) => pred_key.clone(),
                    _ => return false,
                };
                let result = match state.data.first() {
                    Some(value) => value.clone(),
                    None => return false,
                };
                if state.data.len() > 1 {
                    self.choice_points.push(ChoicePoint {
                        next_pc: self.pc,
                        saved_args: self.save_regs(),
                        stack: self.stack.clone(),
                        cp: self.cp,
                        trail_len: self.trail.len(),
                        heap_len: self.heap.len(),
                        builtin_state: Some(BuiltinState {
                            name: "foreign_results".to_string(),
                            args: state.args.clone(),
                            data: state.data[1..].to_vec(),
                        }),
                        cut_barrier: self.cut_barrier,
                    });
                }
                if self.apply_foreign_result(&pred_key, &state.args[1..], &result) {
                    self.pc += 1; true
                } else { false }
            }
            _ => false,
        }
    }'.


compile_execute_meta_builtin_to_rust(Code) :-
    Code = '    /// Execute meta-predicates that require goal evaluation.
    fn execute_meta_builtin(&mut self, op: &str, _arity: usize) -> bool {
        match op {
            "\\\\+/1" => {
                // Negation-as-failure using WAM choice point mechanism.
                // Push a choice point that will succeed if the goal fails.
                // If the goal succeeds, cut and fail (negation).
                let goal = self.get_reg_raw("A1");
                let goal = goal.map(|v| self.deref_heap(&v));
                match goal {
                    Some(Value::Str(functor, args)) if functor == "member/2" && args.len() == 2 => {
                        // Fast path for \\+(member(X, List)) — direct list scan,
                        // no choice points, no builtin dispatch overhead.
                        let needle = self.deref_var(&args[0]);
                        let haystack = self.deref_var(&args[1]);
                        let found = match &haystack {
                            Value::List(items) => items.iter().any(|item| {
                                let di = self.deref_var(item);
                                di == needle
                            }),
                            _ => false,
                        };
                        // Negate: if found, \\+ fails; if not found, \\+ succeeds
                        if found { return false; }
                        self.pc += 1;
                        return true;
                    }
                    Some(Value::Str(functor, args)) => {
                        let naf_pc = self.pc;

                        // Push a NAF choice point for general goals.
                        let cp_depth = self.choice_points.len();
                        self.choice_points.push(ChoicePoint {
                            next_pc: 0,
                            stack: self.stack.clone(),
                            saved_args: self.save_regs(),
                            cp: self.cp,
                            trail_len: self.trail.len(),
                            heap_len: self.heap.len(),
                            builtin_state: Some(BuiltinState {
                                name: "naf_succeed".to_string(),
                                args: vec![Value::Integer(naf_pc as i64)],
                                data: vec![],
                            }),
                            cut_barrier: self.cut_barrier,
                        });

                        let pred_key = format!("{}", functor);
                        let arity = args.len();
                        for (i, arg) in args.iter().enumerate() {
                            self.set_reg(&format!("A{}", i + 1), arg.clone());
                        }

                        if self.execute_builtin(&pred_key, arity) {
                            self.choice_points.truncate(cp_depth);
                            return false;
                        }

                        // Try as compiled predicate via label dispatch
                        if let Some(&target_pc) = self.labels.get(&pred_key) {
                            self.pc = target_pc;
                            self.cp = 0; // halt sentinel for sub-run
                            let goal_ok = self.run();
                            if goal_ok {
                                // Goal succeeded → \\+ fails.
                                // Remove the NAF choice point and any nested ones.
                                self.choice_points.truncate(cp_depth);
                                return false;
                            }
                            // Goal failed → backtrack will reach our NAF CP
                            // (it may already have been reached by run()''s backtrack)
                        }

                        // Goal failed entirely. The NAF CP should still be on the stack
                        // (or was already consumed by backtrack). Either way, \\+ succeeds.
                        // Clean up: ensure the NAF CP is removed if still present.
                        if self.choice_points.len() > cp_depth {
                            self.choice_points.truncate(cp_depth);
                        }
                        // Restore state from the NAF CP we pushed
                        // (backtrack may have already done this, but be safe)
                        self.pc = naf_pc + 1;
                        true
                    }
                    _ => false,
                }
            }
            _ => false,
        }
    }'.

compile_resume_naf_to_rust(Code) :-
    Code = ''.

compile_eval_arith_to_rust(Code) :-
    Code = '    /// Evaluate an arithmetic expression to a float.
    /// TODO: Implement a dual integer/float arithmetic path for better performance and precision.
    fn eval_arith(&self, expr: &Value) -> Option<f64> {
        match expr {
            Value::Integer(n) => Some(*n as f64),
            Value::Float(f) => Some(*f),
            Value::Ref(addr) => {
                // Dereference heap structure
                let derefed = self.deref_heap(expr);
                if let Value::Str(op, args) = &derefed {
                    self.eval_arith_compound(op, args)
                } else {
                    self.eval_arith(&derefed)
                }
            }
            Value::Str(op, args) => self.eval_arith_compound(op, args),
            Value::Atom(name) => {
                // Try register dereference
                if name.starts_with(\'A\') || name.starts_with(\'X\') || name.starts_with(\'Y\') {
                    self.get_reg(name).and_then(|v| self.eval_arith(&v))
                } else if let Ok(n) = name.parse::<f64>() {
                    Some(n)
                } else { None }
            }
            _ => None,
        }
    }

    fn eval_arith_compound(&self, op: &str, args: &[Value]) -> Option<f64> {
        // Strip arity suffix from functor (e.g. "+/2" -> "+")
        let bare_op = if let Some(slash) = op.rfind(''/'') {
            &op[..slash]
        } else { op };
        if args.len() == 2 {
            let a = self.eval_arith(&args[0])?;
            let b = self.eval_arith(&args[1])?;
            match bare_op {
                "+" => Some(a + b),
                "-" => Some(a - b),
                "*" => Some(a * b),
                "/" if b != 0.0 => Some(a / b),
                "//" if b != 0.0 => Some((a / b).floor()),
                "mod" if b != 0.0 => Some(a % b),
                "**" => Some(a.powf(b)),
                _ => None,
            }
        } else if args.len() == 1 {
            let a = self.eval_arith(&args[0])?;
            match bare_op {
                "-" => Some(-a),
                "abs" => Some(a.abs()),
                _ => None,
            }
        } else { None }
    }'.

% ============================================================================
% ASSEMBLY: Combine Phase 2 + Phase 3 into complete runtime
% ============================================================================

%% compile_wam_runtime_to_rust(+Options, -RustCode)
%  Generates the complete impl WamState block with step(), run(),
%  backtrack(), and all helper methods.
compile_wam_runtime_to_rust(Options, RustCode) :-
    compile_step_wam_to_rust(Options, StepCode),
    compile_wam_helpers_to_rust(Options, HelpersCode),
    format(string(RustCode),
'impl WamState {
~w

~w
}', [StepCode, HelpersCode]).

% ============================================================================
% WAM PREDICATE WRAPPER: Compile a Prolog predicate via WAM to Rust
% ============================================================================

%% compile_wam_predicate_to_rust(+Pred/Arity, +WamCode, +Options, -RustCode)
%  Given WAM compiled code for a predicate, generate a Rust wrapper function
%  that creates instruction data and executes it via the WAM runtime.
compile_wam_predicate_to_rust(Pred/Arity, WamCode, Options, RustCode) :-
    atom_string(Pred, PredStr),
    build_rust_wam_arg_list(Arity, ArgList),
    build_rust_wam_arg_setup(Arity, ArgSetup),
    foreign_wrapper_setup(Pred/Arity, WamCode, Options, InstrSetup, ForeignSetup, RunExpr),
    format(string(RustCode),
'/// WAM-compiled predicate: ~w/~w
/// Compiled via WAM for predicates that resist native lowering.
pub fn ~w(~w) -> bool {
    use std::collections::HashMap;
~w
~w
~w
    ~w
}', [PredStr, Arity, PredStr, ArgList, InstrSetup, ForeignSetup, ArgSetup, RunExpr]).

foreign_wrapper_setup(Pred/Arity, WamCode, Options, InstrSetup, Setup, RunExpr) :-
    (   option(foreign_lowering(ForeignSpec), Options),
        rust_foreign_spec(Pred/Arity, ForeignSpec, SetupOps, EntryPred/EntryArity)
    ->  InstrSetup = '    vm.code = Vec::new();
    vm.labels = HashMap::new();
    vm.pc = 1;',
        rust_foreign_setup_code(SetupOps, Setup),
        format(string(RunExpr),
            'vm.execute_foreign_predicate("~w", ~w)', [EntryPred, EntryArity])
    ;   wam_code_to_rust_instructions(WamCode, Pred/Arity, Options, InstrLiterals, LabelLiterals),
        format(string(InstrSetup),
'    let code: Vec<Instruction> = vec![
~w
    ];
    let mut labels: HashMap<String, usize> = HashMap::new();
~w
    vm.code = code;
    vm.labels = labels;
    vm.pc = 1;', [InstrLiterals, LabelLiterals]),
        Setup = "",
        RunExpr = 'vm.run()'
    ).

rust_foreign_spec(Pred/Arity,
        foreign_predicate(Pred/Arity, SetupOps, RewriteCalls),
        SetupOps,
        Pred/Arity) :-
    is_list(SetupOps),
    is_list(RewriteCalls).

rust_foreign_setup_code([], "").
rust_foreign_setup_code(Ops, Setup) :-
    maplist(rust_foreign_setup_line, Ops, Lines),
    atomic_list_concat(Lines, '\n', Setup).

rust_foreign_setup_line(register_foreign_native_kind(Pred/Arity, Kind), Line) :-
    format(string(Line),
        '    vm.register_foreign_native_kind("~w/~w", "~w");', [Pred, Arity, Kind]).
rust_foreign_setup_line(register_foreign_result_layout(Pred/Arity, Layout), Line) :-
    rust_foreign_result_layout_literal(Layout, LayoutLiteral),
    format(string(Line),
        '    vm.register_foreign_result_layout("~w/~w", "~w");', [Pred, Arity, LayoutLiteral]).
rust_foreign_setup_line(register_foreign_result_mode(Pred/Arity, Mode), Line) :-
    format(string(Line),
        '    vm.register_foreign_result_mode("~w/~w", "~w");', [Pred, Arity, Mode]).
rust_foreign_setup_line(register_foreign_string_config(Pred/Arity, Key, ValuePred/ValueArity), Line) :-
    format(string(Line),
        '    vm.register_foreign_string_config("~w/~w", "~w", "~w/~w");',
        [Pred, Arity, Key, ValuePred, ValueArity]).
rust_foreign_setup_line(register_foreign_string_config(Pred/Arity, Key, Value), Line) :-
    format(string(Line),
        '    vm.register_foreign_string_config("~w/~w", "~w", "~w");',
        [Pred, Arity, Key, Value]).
rust_foreign_setup_line(register_foreign_usize_config(Pred/Arity, Key, Value), Line) :-
    format(string(Line),
        '    vm.register_foreign_usize_config("~w/~w", "~w", ~w);', [Pred, Arity, Key, Value]).
rust_foreign_setup_line(register_indexed_atom_fact2(Pred/Arity, Pairs), Line) :-
    rust_fact_pairs_literal(Pairs, PairsLiteral),
    format(string(Line),
        '    vm.register_indexed_atom_fact2_pairs("~w/~w", &~w);',
        [Pred, Arity, PairsLiteral]).
rust_foreign_setup_line(register_indexed_weighted_edge(Pred/Arity, Triples), Line) :-
    rust_fact_triples_literal(Triples, TriplesLiteral),
    format(string(Line),
        '    vm.register_indexed_weighted_edge_triples("~w/~w", &~w);',
        [Pred, Arity, TriplesLiteral]).

rust_fact_pairs_literal(Pairs, Literal) :-
    maplist(rust_fact_pair_literal, Pairs, PairLiterals),
    atomic_list_concat(PairLiterals, ', ', Joined),
    format(string(Literal), '[~w]', [Joined]).

rust_fact_pair_literal(Left-Right, Literal) :-
    escape_rust_string(Left, ELeft),
    escape_rust_string(Right, ERight),
    format(string(Literal), '("~w", "~w")', [ELeft, ERight]).

rust_fact_triples_literal(Triples, Literal) :-
    maplist(rust_fact_triple_literal, Triples, TripleLiterals),
    atomic_list_concat(TripleLiterals, ', ', Joined),
    format(string(Literal), '[~w]', [Joined]).

rust_fact_triple_literal(Left-Right-Weight, Literal) :-
    escape_rust_string(Left, ELeft),
    escape_rust_string(Right, ERight),
    format(string(Literal), '("~w", "~w", ~w)', [ELeft, ERight, Weight]).

rust_foreign_result_layout_literal(tuple(Arity), Literal) :-
    format(string(Literal), 'tuple:~w', [Arity]).
rust_foreign_result_layout_literal(Layout, Layout).

%% wam_code_to_rust_instructions(+WamCodeStr, +PredIndicator, +Options, -InstrLiterals, -LabelLiterals)
%  Parses a WAM code string and generates Rust vec![] entries and label map inserts.
wam_code_to_rust_instructions(WamCode, PredIndicator, Options, InstrLiterals, LabelLiterals) :-
    atom_string(WamCode, WamStr),
    split_string(WamStr, "\n", "", Lines),
    wam_lines_to_rust(Lines, 1, PredIndicator, Options, InstrParts, LabelParts),
    atomic_list_concat(InstrParts, '\n', InstrLiterals),
    atomic_list_concat(LabelParts, '\n', LabelLiterals).

wam_lines_to_rust([], _, _, _, [], []).
wam_lines_to_rust([Line|Rest], PC, PredIndicator, Options, Instrs, Labels) :-
    split_string(Line, " \t,", " \t,", Parts),
    delete(Parts, "", CleanParts),
    (   CleanParts == []
    ->  wam_lines_to_rust(Rest, PC, PredIndicator, Options, Instrs, Labels)
    ;   CleanParts = [First|_],
        (   % Label line: "pred/2:" or "L_pred_2_2:"
            sub_string(First, _, 1, 0, ":")
        ->  sub_string(First, 0, _, 1, LabelName),
            format(string(LabelInsert),
                '    labels.insert("~w".to_string(), ~w);', [LabelName, PC]),
            Labels = [LabelInsert|RestLabels],
            wam_lines_to_rust(Rest, PC, PredIndicator, Options, Instrs, RestLabels)
        ;   % Instruction line
            wam_line_to_rust_instr(CleanParts, PredIndicator, Options, RustInstr),
            (   sub_string(RustInstr, 0, _, _, "/*")
            ->  format(string(InstrEntry), '        ~w', [RustInstr])
            ;   format(string(InstrEntry), '        ~w,', [RustInstr])
            ),
            NPC is PC + 1,
            Instrs = [InstrEntry|RestInstrs],
            wam_lines_to_rust(Rest, NPC, PredIndicator, Options, RestInstrs, Labels)
        )
    ).

%% wam_line_to_rust_instr(+Parts, +PredIndicator, +Options, -RustExpr)
%  Converts parsed WAM instruction parts to a Rust Instruction enum literal.
wam_line_to_rust_instr(["get_constant", C, Ai], _, _, Rust) :-
    clean_comma(C, CC), clean_comma(Ai, CAi),
    format(string(Rust),
        'Instruction::GetConstant(Value::Atom("~w".to_string()), "~w".to_string())',
        [CC, CAi]).
wam_line_to_rust_instr(["get_variable", Xn, Ai], _, _, Rust) :-
    clean_comma(Xn, CXn), clean_comma(Ai, CAi),
    format(string(Rust),
        'Instruction::GetVariable("~w".to_string(), "~w".to_string())',
        [CXn, CAi]).
wam_line_to_rust_instr(["get_value", Xn, Ai], _, _, Rust) :-
    clean_comma(Xn, CXn), clean_comma(Ai, CAi),
    format(string(Rust),
        'Instruction::GetValue("~w".to_string(), "~w".to_string())',
        [CXn, CAi]).
wam_line_to_rust_instr(["get_structure", FN, Ai], _, _, Rust) :-
    clean_comma(FN, CFN), clean_comma(Ai, CAi),
    format(string(Rust),
        'Instruction::GetStructure("~w".to_string(), "~w".to_string())',
        [CFN, CAi]).
wam_line_to_rust_instr(["get_list", Ai], _, _, Rust) :-
    clean_comma(Ai, CAi),
    format(string(Rust),
        'Instruction::GetList("~w".to_string())', [CAi]).
wam_line_to_rust_instr(["unify_variable", Xn], _, _, Rust) :-
    format(string(Rust),
        'Instruction::UnifyVariable("~w".to_string())', [Xn]).
wam_line_to_rust_instr(["unify_value", Xn], _, _, Rust) :-
    format(string(Rust),
        'Instruction::UnifyValue("~w".to_string())', [Xn]).
wam_line_to_rust_instr(["unify_constant", C], _, _, Rust) :-
    (   number_string(N, C)
    ->  format(string(Rust),
            'Instruction::UnifyConstant(Value::Integer(~w))', [N])
    ;   format(string(Rust),
            'Instruction::UnifyConstant(Value::Atom("~w".to_string()))', [C])
    ).
wam_line_to_rust_instr(["put_constant", C, Ai], _, _, Rust) :-
    clean_comma(C, CC), clean_comma(Ai, CAi),
    format(string(Rust),
        'Instruction::PutConstant(Value::Atom("~w".to_string()), "~w".to_string())',
        [CC, CAi]).
wam_line_to_rust_instr(["put_variable", Xn, Ai], _, _, Rust) :-
    clean_comma(Xn, CXn), clean_comma(Ai, CAi),
    format(string(Rust),
        'Instruction::PutVariable("~w".to_string(), "~w".to_string())',
        [CXn, CAi]).
wam_line_to_rust_instr(["put_value", Xn, Ai], _, _, Rust) :-
    clean_comma(Xn, CXn), clean_comma(Ai, CAi),
    format(string(Rust),
        'Instruction::PutValue("~w".to_string(), "~w".to_string())',
        [CXn, CAi]).
wam_line_to_rust_instr(["put_structure", FN, Ai], _, _, Rust) :-
    clean_comma(FN, CFN), clean_comma(Ai, CAi),
    format(string(Rust),
        'Instruction::PutStructure("~w".to_string(), "~w".to_string())',
        [CFN, CAi]).
wam_line_to_rust_instr(["put_list", Ai], _, _, Rust) :-
    format(string(Rust),
        'Instruction::PutList("~w".to_string())', [Ai]).
wam_line_to_rust_instr(["set_variable", Xn], _, _, Rust) :-
    format(string(Rust),
        'Instruction::SetVariable("~w".to_string())', [Xn]).
wam_line_to_rust_instr(["set_value", Xn], _, _, Rust) :-
    format(string(Rust),
        'Instruction::SetValue("~w".to_string())', [Xn]).
wam_line_to_rust_instr(["set_constant", C], _, _, Rust) :-
    (   number_string(N, C)
    ->  format(string(Rust),
            'Instruction::SetConstant(Value::Integer(~w))', [N])
    ;   format(string(Rust),
            'Instruction::SetConstant(Value::Atom("~w".to_string()))', [C])
    ).
wam_line_to_rust_instr(["allocate"], _, _, "Instruction::Allocate").
wam_line_to_rust_instr(["deallocate"], _, _, "Instruction::Deallocate").
wam_line_to_rust_instr(["call", P, N], Pred/Arity, Options, Rust) :-
    clean_comma(P, CP), clean_comma(N, CN),
    escape_rust_string(CP, ECP),
    (   number_string(Num, CN) -> true ; Num = 0 ),
    (   rust_foreign_rewrite_call(Options, Pred/Arity, ECP, Num, ForeignPred, ForeignArity)
    ->  format(string(Rust),
            'Instruction::CallForeign("~w".to_string(), ~w)', [ForeignPred, ForeignArity])
    ;   format(string(Rust),
            'Instruction::Call("~w".to_string(), ~w)', [ECP, Num])
    ).
wam_line_to_rust_instr(["execute", P], Pred/Arity, Options, Rust) :-
    escape_rust_string(P, EP),
    (   rust_foreign_rewrite_execute(Options, Pred/Arity, EP, ForeignPred, ForeignArity)
    ->  format(string(Rust),
            'Instruction::CallForeign("~w".to_string(), ~w)', [ForeignPred, ForeignArity])
    ;   format(string(Rust),
            'Instruction::Execute("~w".to_string())', [EP])
    ).
wam_line_to_rust_instr(["proceed"], _, _, "Instruction::Proceed").
wam_line_to_rust_instr(["builtin_call", Op, N], _, _, Rust) :-
    clean_comma(Op, COp), clean_comma(N, CN),
    escape_rust_string(COp, ECOp),
    (   number_string(Num, CN) -> true ; Num = 0 ),
    format(string(Rust),
        'Instruction::BuiltinCall("~w".to_string(), ~w)', [ECOp, Num]).
wam_line_to_rust_instr(["begin_aggregate", Type, ValueReg, ResultReg], _, _, Rust) :-
    clean_comma(Type, CType),
    clean_comma(ValueReg, CValueReg),
    clean_comma(ResultReg, CResultReg),
    format(string(Rust),
        'Instruction::BeginAggregate("~w".to_string(), "~w".to_string(), "~w".to_string())',
        [CType, CValueReg, CResultReg]).
wam_line_to_rust_instr(["end_aggregate", ValueReg], _, _, Rust) :-
    clean_comma(ValueReg, CValueReg),
    format(string(Rust),
        'Instruction::EndAggregate("~w".to_string())', [CValueReg]).
wam_line_to_rust_instr(["try_me_else", Label], _, _, Rust) :-
    format(string(Rust),
        'Instruction::TryMeElse("~w".to_string())', [Label]).
wam_line_to_rust_instr(["trust_me"], _, _, "Instruction::TrustMe").
wam_line_to_rust_instr(["retry_me_else", Label], _, _, Rust) :-
    format(string(Rust),
        'Instruction::RetryMeElse("~w".to_string())', [Label]).
% Indexing instructions
wam_line_to_rust_instr(["switch_on_constant"|Entries], _, _, Rust) :-
    maplist(parse_index_entry_constant, Entries, RustEntries),
    atomic_list_concat(RustEntries, ', ', Joined),
    format(string(Rust), 'Instruction::SwitchOnConstant(vec![~w])', [Joined]).
wam_line_to_rust_instr(["switch_on_structure"|Entries], _, _, Rust) :-
    maplist(parse_index_entry_structure, Entries, RustEntries),
    atomic_list_concat(RustEntries, ', ', Joined),
    format(string(Rust), 'Instruction::SwitchOnStructure(vec![~w])', [Joined]).
wam_line_to_rust_instr(["switch_on_constant_a2"|Entries], _, _, Rust) :-
    maplist(parse_index_entry_constant, Entries, RustEntries),
    atomic_list_concat(RustEntries, ', ', Joined),
    format(string(Rust), 'Instruction::SwitchOnConstantA2(vec![~w])', [Joined]).
% Fallback for unknown instructions
wam_line_to_rust_instr(Parts, _, _, Rust) :-
    atomic_list_concat(Parts, ' ', Joined),
    format(string(Rust), '/* unknown: ~w */', [Joined]).

rust_foreign_rewrite_call(Options, CurrentPred, TargetPredArity, Num, ForeignPred, ForeignArity) :-
    option(foreign_lowering(ForeignSpec), Options),
    rust_foreign_spec(CurrentPred, ForeignSpec, _, RewriteCalls, ForeignPred/ForeignArity),
    member(TargetPred/TargetArity, RewriteCalls),
    format(string(ExpectedTarget), "~w/~w", [TargetPred, TargetArity]),
    TargetPredArity == ExpectedTarget,
    Num =:= ForeignArity.

rust_foreign_rewrite_execute(Options, CurrentPred, TargetPredArity, ForeignPred, ForeignArity) :-
    option(foreign_lowering(ForeignSpec), Options),
    rust_foreign_spec(CurrentPred, ForeignSpec, _, RewriteCalls, ForeignPred/ForeignArity),
    member(TargetPred/TargetArity, RewriteCalls),
    format(string(ExpectedTarget), "~w/~w", [TargetPred, TargetArity]),
    TargetPredArity == ExpectedTarget.

rust_foreign_spec(Pred/Arity,
        foreign_predicate(Pred/Arity, SetupOps, RewriteCalls),
        SetupOps,
        RewriteCalls,
        Pred/Arity) :-
    is_list(SetupOps),
    is_list(RewriteCalls).

parse_index_entry_constant(Entry, Rust) :-
    (   sub_string(Entry, Before, 1, After, ":")
    ->  sub_string(Entry, 0, Before, _, ValStr),
        sub_string(Entry, _, After, 0, Label),
        (   number_string(N, ValStr)
        ->  format(string(Rust), '(Value::Integer(~w), "~w".to_string())', [N, Label])
        ;   ValStr = "true" -> format(string(Rust), '(Value::Bool(true), "~w".to_string())', [Label])
        ;   ValStr = "false" -> format(string(Rust), '(Value::Bool(false), "~w".to_string())', [Label])
        ;   format(string(Rust), '(Value::Atom("~w".to_string()), "~w".to_string())', [ValStr, Label])
        )
    ;   format(string(Rust), '(Value::Atom("~w".to_string()), "unknown".to_string())', [Entry])
    ).

parse_index_entry_structure(Entry, Rust) :-
    (   sub_string(Entry, Before, 1, After, ":")
    ->  sub_string(Entry, 0, Before, _, Functor),
        sub_string(Entry, _, After, 0, Label),
        format(string(Rust), '("~w".to_string(), "~w".to_string())', [Functor, Label])
    ;   format(string(Rust), '("~w".to_string(), "unknown".to_string())', [Entry])
    ).

clean_comma(Str, Clean) :-
    (   sub_string(Str, _, 1, 0, ",")
    ->  sub_string(Str, 0, _, 1, Clean)
    ;   Clean = Str
    ).

%% escape_rust_string(+In, -Out)
%  Escapes backslashes and double-quotes for embedding in a Rust string literal.
escape_rust_string(In, Out) :-
    atom_string(In, S),
    split_string(S, "\\", "", Parts),
    atomics_to_string(Parts, "\\\\", Escaped1),
    split_string(Escaped1, "\"", "", Parts2),
    atomics_to_string(Parts2, "\\\"", Out).

atomics_to_string([], _, "").
atomics_to_string([X], _, X).
atomics_to_string([X, Y|Rest], Sep, Result) :-
    atomics_to_string([Y|Rest], Sep, Tail),
    string_concat(X, Sep, XSep),
    string_concat(XSep, Tail, Result).

build_rust_wam_arg_list(0, "vm: &mut WamState") :- !.
build_rust_wam_arg_list(Arity, ArgList) :-
    numlist(1, Arity, Indices),
    maplist([I, S]>>format(atom(S), "a~w: Value", [I]), Indices, Parts),
    atomic_list_concat(['vm: &mut WamState'|Parts], ', ', ArgList).

build_rust_wam_arg_setup(0, "") :- !.
build_rust_wam_arg_setup(Arity, Setup) :-
    numlist(1, Arity, Indices),
    maplist([I, S]>>format(string(S),
        '    vm.set_reg("A~w", a~w);', [I, I]), Indices, Lines),
    atomic_list_concat(Lines, '\n', Setup).

% ============================================================================
% FFI KERNEL DETECTION AND RUST REGISTRATION CODE GENERATION
% ============================================================================

%% detect_kernels(+Predicates, -DetectedKernels)
%  Run shared kernel detection on each predicate. Returns a list of
%  Key-Kernel pairs where Key is 'pred/arity' atom and Kernel is the
%  full recursive_kernel(Kind, Pred/Arity, ConfigOps) term.
detect_kernels([], []).
detect_kernels([PI|Rest], Kernels) :-
    (   PI = _Mod:Pred/Arity -> true ; PI = Pred/Arity ),
    functor(Head, Pred, Arity),
    findall(Head-Body, user:clause(Head, Body), Clauses),
    (   Clauses \= [],
        detect_recursive_kernel(Pred, Arity, Clauses, Kernel)
    ->  format(atom(Key), '~w/~w', [Pred, Arity]),
        Kernels = [Key-Kernel|RestKernels]
    ;   Kernels = RestKernels
    ),
    detect_kernels(Rest, RestKernels).

%% generate_setup_foreign_predicates_rust(+DetectedKernels, -RustCode)
%  Generates a Rust function that registers all detected FFI kernels with
%  the WamState at startup. Uses kernel_metadata/4 and kernel_config/2 to
%  produce the correct register_foreign_* calls for each kernel.
generate_setup_foreign_predicates_rust([], Code) :- !,
    Code = "pub fn setup_foreign_predicates(_vm: &mut WamState) {\n    // No kernels detected\n}\n\npub fn foreign_pred_keys() -> HashSet<String> {\n    HashSet::new()\n}".
generate_setup_foreign_predicates_rust(DetectedKernels, Code) :-
    pairs_keys(DetectedKernels, Keys),
    with_output_to(string(Body), (
        format('pub fn setup_foreign_predicates(vm: &mut WamState) {~n'),
        forall(member(KV, DetectedKernels), emit_kernel_registration(KV)),
        format('}~n~n'),
        format('pub fn foreign_pred_keys() -> HashSet<String> {~n'),
        format('    let mut s = HashSet::new();~n'),
        forall(member(K, Keys),
               format('    s.insert("~w".to_string());~n', [K])),
        format('    s~n'),
        format('}~n')
    )),
    Code = Body.

%% emit_kernel_registration(+Key-Kernel)
%  Emit Rust registration statements for a single detected kernel.
emit_kernel_registration(Key-Kernel) :-
    kernel_metadata(Kernel, NativeKind, ResultLayout, ResultMode),
    kernel_config(Kernel, ConfigOps),
    format('    vm.register_foreign_predicate("~w");~n', [Key]),
    format('    vm.register_foreign_native_kind("~w", "~w");~n', [Key, NativeKind]),
    ResultLayout =.. [tuple, N],
    format('    vm.register_foreign_result_layout("~w", "tuple(~w)");~n', [Key, N]),
    format('    vm.register_foreign_result_mode("~w", "~w");~n', [Key, ResultMode]),
    emit_kernel_config_ops(Key, ConfigOps).

%% emit_kernel_config_ops(+Key, +ConfigOps)
%  Emit register_foreign_usize_config / register_foreign_string_config
%  calls for each config operation extracted from the kernel.
emit_kernel_config_ops(_, []).
emit_kernel_config_ops(Key, [Op|Rest]) :-
    Op =.. [ConfigKey, RawValue],
    (   RawValue = EdgePred/_ ->
        % edge_pred(foo/2) -> register string config "edge_pred" = "foo"
        format('    vm.register_foreign_string_config("~w", "~w", "~w");~n',
               [Key, ConfigKey, EdgePred])
    ;   integer(RawValue) ->
        format('    vm.register_foreign_usize_config("~w", "~w", ~w);~n',
               [Key, ConfigKey, RawValue])
    ;   float(RawValue) ->
        format('    vm.register_foreign_string_config("~w", "~w", "~w");~n',
               [Key, ConfigKey, RawValue])
    ;   atom(RawValue) ->
        format('    vm.register_foreign_string_config("~w", "~w", "~w");~n',
               [Key, ConfigKey, RawValue])
    ;   true  % skip unknown config types
    ),
    emit_kernel_config_ops(Key, Rest).

% ============================================================================
% CARGO PROJECT GENERATION
% ============================================================================

%% write_wam_rust_project(+Predicates, +Options, +ProjectDir)
%  Generates a complete Cargo crate with:
%  - Cargo.toml
%  - src/value.rs      (Value enum — from template)
%  - src/instructions.rs (Instruction enum — from template)
%  - src/state.rs      (WamState struct — from template + transpiled runtime)
%  - src/lib.rs        (module layout + compiled predicates)
%
%  Predicates is a list of Module:Pred/Arity indicators to compile.
%  Each predicate is attempted via native lowering first, falling back
%  to WAM compilation.
%
%  Options:
%    module_name(Name)   — crate/module name (default: 'wam_generated')
%    wam_fallback(Bool)  — enable/disable WAM fallback (default: true)
%    include_runtime(Bool) — include transpiled WAM runtime (default: true)
%    emit_mode(Mode)     — interpreter | functions (default: interpreter)
%    parallel(Bool)      — enable Rayon parallel execution (default: false)
write_wam_rust_project(Predicates, Options, ProjectDir) :-
    option(module_name(ModuleName), Options, 'wam_generated'),
    get_time(TimeStamp),
    format_time(string(Date), "%Y-%m-%d %H:%M:%S", TimeStamp),

    % Detect recursive kernels in the predicate list. Detected kernels
    % are handled by the FFI (execute_foreign_predicate) at runtime and
    % are excluded from WAM compilation. The detected kernel list is used
    % to auto-generate setup_foreign_predicates() in lib.rs.
    (   option(no_kernels(true), Options)
    ->  DetectedKernels = [],
        format(user_error, '[WAM-Rust] kernel detection suppressed~n', [])
    ;   detect_kernels(Predicates, DetectedKernels),
        (   DetectedKernels \= []
        ->  pairs_keys(DetectedKernels, DetectedKeys),
            format(user_error, '[WAM-Rust] detected kernels: ~w~n', [DetectedKeys])
        ;   true
        )
    ),

    % Create directory structure
    make_directory_path(ProjectDir),
    directory_file_path(ProjectDir, 'src', SrcDir),
    make_directory_path(SrcDir),

    % Generate Cargo.toml
    render_named_template(rust_wam_cargo,
        [module_name=ModuleName], CargoContent),
    directory_file_path(ProjectDir, 'Cargo.toml', CargoPath),
    write_file(CargoPath, CargoContent),

    % Write value.rs from template file
    read_template_file('templates/targets/rust_wam/value.rs.mustache', ValueTemplate),
    render_template(ValueTemplate, [date=Date], ValueCode),
    directory_file_path(SrcDir, 'value.rs', ValuePath),
    write_file(ValuePath, ValueCode),

    % Write instructions.rs from template file
    read_template_file('templates/targets/rust_wam/instructions.rs.mustache', InstrTemplate),
    render_template(InstrTemplate, [date=Date], InstrCode),
    directory_file_path(SrcDir, 'instructions.rs', InstrPath),
    write_file(InstrPath, InstrCode),

    % Generate state.rs: template + transpiled runtime methods
    option(include_runtime(IncludeRuntime), Options, true),
    read_template_file('templates/targets/rust_wam/state.rs.mustache', StateTemplate),
    render_template(StateTemplate, [date=Date], StateBase),
    (   IncludeRuntime == true
    ->  compile_wam_runtime_to_rust(Options, RuntimeCode),
        format(string(StateCode), "~w\n\n~w", [StateBase, RuntimeCode])
    ;   StateCode = StateBase
    ),
    directory_file_path(SrcDir, 'state.rs', StatePath),
    write_file(StatePath, StateCode),

    % Generate setup_foreign_predicates function for detected kernels
    generate_setup_foreign_predicates_rust(DetectedKernels, SetupForeignCode),

    % Compile predicates and generate lib.rs
    pairs_keys(DetectedKernels, DetectedKeys),
    compile_predicates_for_project(Predicates, [foreign_pred_keys(DetectedKeys)|Options], PredicatesCode),
    format(string(FullPredicatesCode), "~w\n\n~w", [SetupForeignCode, PredicatesCode]),
    render_named_template(rust_wam_lib,
        [module_name=ModuleName, date=Date, predicates_code=FullPredicatesCode],
        LibContent),
    directory_file_path(SrcDir, 'lib.rs', LibPath),
    write_file(LibPath, LibContent),

    format('WAM Rust project created at: ~w~n', [ProjectDir]),
    format('  Predicates compiled: ~w~n', [Predicates]).

%% compile_predicates_for_project(+Predicates, +Options, -Code)
%  Two-pass compilation: collects all WAM-fallback predicates into a shared
%  code+labels table so cross-predicate WAM calls (Execute/Call) work.
%  Native-lowered predicates are compiled independently as before.
compile_predicates_for_project(Predicates, Options, Code) :-
    % Pass 1: classify each predicate as native, wam, or failed
    classify_predicates(Predicates, Options, Classified),
    % Pass 2: collect WAM entries with cumulative PCs, build shared table
    collect_wam_entries(Classified, 1, WamEntries, AllInstrParts, AllLabelParts),
    % Generate shared WAM table if any WAM predicates exist
    (   WamEntries \== []
    ->  atomic_list_concat(AllInstrParts, '\n', AllInstrs),
        atomic_list_concat(AllLabelParts, '\n', AllLabels),
        format(string(SharedCode),
'use std::sync::OnceLock;

static SHARED_WAM: OnceLock<(Vec<Instruction>, HashMap<String, usize>)> = OnceLock::new();

fn get_shared_wam() -> &\'static (Vec<Instruction>, HashMap<String, usize>) {
    SHARED_WAM.get_or_init(|| {
        let mut labels: HashMap<String, usize> = HashMap::new();
~w
        let code: Vec<Instruction> = vec![
~w
        ];
        (code, labels)
    })
}', [AllLabels, AllInstrs])
    ;   SharedCode = ""
    ),
    % Pass 3: generate code for each predicate
    generate_predicate_codes(Classified, WamEntries, PredCodes),
    % Combine shared table + predicate code
    (   SharedCode == ""
    ->  atomic_list_concat(PredCodes, '\n\n', Code)
    ;   atomic_list_concat(PredCodes, '\n\n', PredCodesStr),
        format(string(Code), "~w\n\n~w", [SharedCode, PredCodesStr])
    ).

%% classify_predicates(+Predicates, +Options, -Classified)
%  Returns list of classify(Module, Pred, Arity, Strategy, ExtraData) terms.
classify_predicates([], _, []).
classify_predicates([PredIndicator|Rest], Options, [Entry|RestEntries]) :-
    (   PredIndicator = Module:Pred/Arity -> true
    ;   PredIndicator = Pred/Arity, Module = user
    ),
    (   % Check for auto-detectable FFI kernel FIRST (unless suppressed)
        \+ option(no_kernels(true), Options),
        functor(KHead, Pred, Arity),
        findall(KHead-KBody, Module:clause(KHead, KBody), KClauses),
        KClauses \= [],
        detect_recursive_kernel(Pred, Arity, KClauses, Kernel)
    ->  format(user_error, '  ~w/~w: ffi kernel (~w)~n', [Pred, Arity, Kernel]),
        Entry = classified(Module, Pred, Arity, ffi_kernel, Kernel)
    ;   % Try native Rust lowering (disable WAM fallback inside rust_target
        % so we can distinguish truly-native from WAM-needing predicates)
        catch(
            rust_target:compile_predicate_to_rust(Module:Pred/Arity,
                [include_main(false), wam_fallback(false)|Options], PredCode),
            _, fail)
    ->  format(user_error, '  ~w/~w: native lowering~n', [Pred, Arity]),
        Entry = classified(Module, Pred, Arity, native, PredCode)
    ;   % Fall back to WAM compilation
        option(wam_fallback(WamFB), Options, true),
        WamFB \== false,
        wam_target:compile_predicate_to_wam(Module:Pred/Arity, Options, WamCode),
        (   option(foreign_lowering(ForeignSpec), Options),
            rust_foreign_spec(Pred/Arity, ForeignSpec, _, _)
        ->  % Foreign-lowered: compile individually (not shared WAM)
            compile_wam_predicate_to_rust(Pred/Arity, WamCode, Options, PredCode),
            format(user_error, '  ~w/~w: WAM fallback (foreign)~n', [Pred, Arity]),
            Entry = classified(Module, Pred, Arity, wam_foreign, PredCode)
        ;   % Try lowered emitter when emit_mode(functions)
            option(emit_mode(functions), Options),
            wam_rust_lowerable(Pred/Arity, WamCode, Reason)
        ->  lower_predicate_to_rust(Pred/Arity, WamCode, Options, RustLines),
            atomic_list_concat(RustLines, '\n', PredCode),
            format(user_error, '  ~w/~w: lowered (~w)~n', [Pred, Arity, Reason]),
            Entry = classified(Module, Pred, Arity, lowered, lowered_code(PredCode, WamCode))
        ;   % Standard WAM fallback: will use shared table
            format(user_error, '  ~w/~w: WAM fallback~n', [Pred, Arity]),
            Entry = classified(Module, Pred, Arity, wam, WamCode)
        )
    ;   % Neither worked — emit a stub
        format(string(PredCode),
            '// ~w/~w: compilation failed (neither native nor WAM)', [Pred, Arity]),
        Entry = classified(Module, Pred, Arity, failed, PredCode)
    ),
    classify_predicates(Rest, Options, RestEntries).

%% collect_wam_entries(+Classified, +StartPC, -WamEntries, -AllInstrParts, -AllLabelParts)
%  Iterates over classified predicates, collecting WAM code with cumulative PCs.
%  WamEntries is a list of wam_entry(Pred, Arity, StartPC) terms.
collect_wam_entries([], _, [], [], []).
collect_wam_entries([classified(_, Pred, Arity, wam, WamCode)|Rest], PC,
                    [wam_entry(Pred, Arity, PC)|RestEntries],
                    AllInstrs, AllLabels) :-
    atom_string(WamCode, WamStr),
    split_string(WamStr, "\n", "", Lines),
    wam_lines_to_rust(Lines, PC, Pred/Arity, [], InstrParts, LabelParts),
    % Count instructions to compute next PC
    length(InstrParts, InstrCount),
    NextPC is PC + InstrCount,
    collect_wam_entries(Rest, NextPC, RestEntries, RestInstrs, RestLabels),
    append(InstrParts, RestInstrs, AllInstrs),
    append(LabelParts, RestLabels, AllLabels).
collect_wam_entries([classified(_, Pred, Arity, lowered, lowered_code(_, WamCode))|Rest], PC,
                    [wam_entry(Pred, Arity, PC)|RestEntries],
                    AllInstrs, AllLabels) :-
    atom_string(WamCode, WamStr),
    split_string(WamStr, "\n", "", Lines),
    wam_lines_to_rust(Lines, PC, Pred/Arity, [], InstrParts, LabelParts),
    length(InstrParts, InstrCount),
    NextPC is PC + InstrCount,
    collect_wam_entries(Rest, NextPC, RestEntries, RestInstrs, RestLabels),
    append(InstrParts, RestInstrs, AllInstrs),
    append(LabelParts, RestLabels, AllLabels).
collect_wam_entries([_|Rest], PC, Entries, Instrs, Labels) :-
    collect_wam_entries(Rest, PC, Entries, Instrs, Labels).

%% generate_predicate_codes(+Classified, +WamEntries, -PredCodes)
%  Generates Rust code for each classified predicate.
generate_predicate_codes([], _, []).
generate_predicate_codes([classified(_, Pred, Arity, ffi_kernel, Kernel)|Rest],
                         WamEntries, [Code|RestCodes]) :-
    % FFI kernel — no WAM code needed; handled by setup_foreign_predicates
    % and execute_foreign_predicate at runtime via CallForeign dispatch.
    Kernel = recursive_kernel(Kind, _, _),
    format(string(Code),
        "// Strategy: ffi_kernel (~w)\n// ~w/~w dispatched via CallForeign → execute_foreign_predicate",
        [Kind, Pred, Arity]),
    generate_predicate_codes(Rest, WamEntries, RestCodes).
generate_predicate_codes([classified(_, _Pred, _Arity, native, PredCode)|Rest],
                         WamEntries, [Code|RestCodes]) :-
    format(string(Code), "// Strategy: native\n~w", [PredCode]),
    generate_predicate_codes(Rest, WamEntries, RestCodes).
generate_predicate_codes([classified(_, _Pred, _Arity, wam_foreign, PredCode)|Rest],
                         WamEntries, [Code|RestCodes]) :-
    format(string(Code), "// Strategy: wam\n~w", [PredCode]),
    generate_predicate_codes(Rest, WamEntries, RestCodes).
generate_predicate_codes([classified(_, _Pred, _Arity, lowered, lowered_code(PredCode, _))|Rest],
                         WamEntries, [Code|RestCodes]) :-
    format(string(Code), "// Strategy: lowered\n~w", [PredCode]),
    generate_predicate_codes(Rest, WamEntries, RestCodes).
generate_predicate_codes([classified(_, Pred, Arity, wam, _WamCode)|Rest],
                         WamEntries, [Code|RestCodes]) :-
    % Look up this predicate's start PC in the shared table
    member(wam_entry(Pred, Arity, StartPC), WamEntries),
    compile_wam_predicate_to_rust_shared(Pred/Arity, StartPC, Code),
    generate_predicate_codes(Rest, WamEntries, RestCodes).
generate_predicate_codes([classified(_, _Pred, _Arity, failed, PredCode)|Rest],
                         WamEntries, [Code|RestCodes]) :-
    format(string(Code), "// Strategy: failed\n~w", [PredCode]),
    generate_predicate_codes(Rest, WamEntries, RestCodes).

%% compile_wam_predicate_to_rust_shared(+Pred/Arity, +StartPC, -RustCode)
%  Generates a thin WAM predicate wrapper that references the shared code table.
compile_wam_predicate_to_rust_shared(Pred/Arity, StartPC, RustCode) :-
    atom_string(Pred, PredStr),
    build_rust_wam_arg_list(Arity, ArgList),
    build_rust_wam_arg_setup(Arity, ArgSetup),
    format(string(RustCode),
'// Strategy: wam
/// WAM-compiled predicate: ~w/~w (shared table, pc=~w)
/// Compiled via WAM for predicates that resist native lowering.
pub fn ~w(~w) -> bool {
    let (code, labels) = get_shared_wam();
    vm.code = code.clone();
    vm.labels = labels.clone();
    vm.pc = ~w;
~w
    vm.run()
}', [PredStr, Arity, StartPC, PredStr, ArgList, StartPC, ArgSetup]).

%% write_file(+Path, +Content)
write_file(Path, Content) :-
    setup_call_cleanup(
        open(Path, write, Stream),
        format(Stream, "~w", [Content]),
        close(Stream)
    ).

%% read_template_file(+RelativePath, -Content)
%  Reads a template file from the project directory.
read_template_file(RelativePath, Content) :-
    (   exists_file(RelativePath)
    ->  read_file_to_string(RelativePath, Content, [])
    ;   format(atom(Content),
            '// Template not found: ~w', [RelativePath])
    ).

% ============================================================================
% CARGO CHECK VALIDATION
% ============================================================================

%% cargo_check_project(+ProjectDir, -Result)
%  Runs `cargo check` on a generated Rust project to verify the code compiles.
%  Result is one of:
%    ok               — cargo check succeeded
%    error(ExitCode, Output) — cargo check failed
%    not_available     — cargo not found on PATH
cargo_check_project(ProjectDir, Result) :-
    (   % Check if cargo is available
        catch(
            (process_create(path(cargo), ['--version'],
                [stdout(pipe(S)), stderr(pipe(_)), process(Pid)]),
             read_string(S, _, _), close(S),
             process_wait(Pid, exit(0))),
            _, fail)
    ->  % Run cargo check
        format(atom(Cmd), 'cd "~w" && cargo check 2>&1', [ProjectDir]),
        catch(
            (process_create(path(sh), ['-c', Cmd],
                [stdout(pipe(Out)), stderr(pipe(Err)), process(Pid2)]),
             read_string(Out, _, OutStr), close(Out),
             read_string(Err, _, ErrStr), close(Err),
             process_wait(Pid2, exit(ExitCode))),
            E,
            (format(user_error, 'cargo check error: ~w~n', [E]),
             ExitCode = -1, OutStr = "", ErrStr = "")
        ),
        (   ExitCode == 0
        ->  Result = ok,
            format('cargo check: OK~n')
        ;   atomic_list_concat([OutStr, ErrStr], '\n', FullOutput),
            Result = error(ExitCode, FullOutput),
            format(user_error, 'cargo check failed (exit ~w):~n~w~n',
                [ExitCode, FullOutput])
        )
    ;   Result = not_available,
        format(user_error, 'cargo not found on PATH~n', [])
    ).
