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
    cargo_check_project/2                % +ProjectDir, -Result
]).

:- use_module(library(lists)).
:- use_module(library(option)).
:- use_module('../core/template_system').
:- use_module('../bindings/rust_wam_bindings').
:- use_module('../targets/wam_target', [compile_predicate_to_wam/3]).

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
    Body = '                let raw_val = self.regs.get(ai).cloned();
                let val = raw_val.map(|v| self.deref_var(&v));
                match val {
                    Some(v) if v == *c => { self.pc += 1; true }
                    Some(Value::Unbound(ref var_name)) => {
                        self.trail_binding(ai);
                        self.regs.insert(ai.clone(), c.clone());
                        self.bind_var(var_name, c.clone());
                        self.pc += 1;
                        true
                    }
                    _ => false,
                }'.

wam_instruction_arm('Instruction::GetVariable(xn, ai)', Body) :-
    Body = '                if let Some(raw) = self.regs.get(ai).cloned() {
                    let val = self.deref_var(&raw);
                    self.trail_binding(xn);
                    self.put_reg(xn, val);
                    self.pc += 1;
                    true
                } else { false }'.

wam_instruction_arm('Instruction::GetValue(xn, ai)', Body) :-
    Body = '                let val_a = self.regs.get(ai).cloned();
                let val_x = self.get_reg(xn);
                match (val_a, val_x) {
                    (Some(a), Some(x)) if a == x => { self.pc += 1; true }
                    (Some(a), _) if a.is_unbound() => {
                        self.trail_binding(ai);
                        if let Some(x) = self.get_reg(xn) {
                            self.regs.insert(ai.clone(), x);
                        }
                        self.pc += 1; true
                    }
                    (_, Some(x)) if x.is_unbound() => {
                        self.trail_binding(xn);
                        if let Some(a) = self.regs.get(ai).cloned() {
                            self.put_reg(xn, a);
                        }
                        self.pc += 1; true
                    }
                    _ => false,
                }'.

wam_instruction_arm('Instruction::GetStructure(fn_str, ai)', Body) :-
    Body = '                if let Some(val) = self.regs.get(ai).cloned() {
                    if val.is_unbound() {
                        // Write mode
                        let addr = self.heap.len();
                        self.heap.push(Value::Str(format!("str({})", fn_str), vec![]));
                        self.trail_binding(ai);
                        self.regs.insert(ai.clone(), Value::Ref(addr));
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
    Body = '                if let Some(val) = self.regs.get(ai).cloned() {
                    if val.is_unbound() {
                        let addr = self.heap.len();
                        self.heap.push(Value::Str("str(./2)".to_string(), vec![]));
                        self.trail_binding(ai);
                        self.regs.insert(ai.clone(), Value::Ref(addr));
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
                self.regs.insert(ai.clone(), c.clone());
                self.pc += 1; true'.

wam_instruction_arm('Instruction::PutVariable(xn, ai)', Body) :-
    Body = '                let var = Value::Unbound(format!("_V{}", self.var_counter));
                self.var_counter += 1;
                self.trail_binding(xn);
                self.trail_binding(ai);
                self.put_reg(xn, var.clone());
                self.regs.insert(ai.clone(), var);
                self.pc += 1; true'.

wam_instruction_arm('Instruction::PutValue(xn, ai)', Body) :-
    Body = '                if let Some(val) = self.get_reg(xn) {
                    self.trail_binding(ai);
                    self.regs.insert(ai.clone(), val);
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
                self.regs.insert(ai.clone(), Value::Ref(addr));
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
                self.regs.insert(ai.clone(), Value::Integer(marker as i64));
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
                    self.regs.insert(out_reg.clone(), list);
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
                self.regs.insert("A1".to_string(), mid);
                self.regs.insert("A2".to_string(), root);
                self.regs.insert("A3".to_string(), child_hops);
                self.regs.insert("A4".to_string(), next_visited);
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
    Body = '                self.execute_foreign_predicate(pred, *arity)'.

wam_instruction_arm('Instruction::CallIndexedAtomFact2(pred)', Body) :-
    Body = '                let key = match self.regs.get("A1").cloned().map(|v| self.deref_var(&v)) {
                    Some(Value::Atom(s)) => s,
                    _ => return false,
                };
                let a2 = match self.regs.get("A2").cloned() {
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
    Body = '                let raw = self.regs.get("A1").cloned().map(|v| self.deref_var(&v));
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
    Body = '                let raw = self.regs.get("A1").cloned().map(|v| self.deref_var(&v));
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
    Body = '                if let Some(val) = self.regs.get("A1").cloned() {
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
    Body = '                if let Some(val) = self.regs.get("A1").cloned() {
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
    Body = '                if let Some(val) = self.regs.get("A2").cloned() {
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
    Body = '                if let Some(val) = self.regs.get("A2").cloned() {
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
    compile_collect_native_category_ancestor_to_rust(NativeAncestorCode),
    compile_compute_native_countdown_sum_to_rust(NativeCountdownSumCode),
    compile_collect_native_transitive_closure_to_rust(NativeClosureCode),
    compile_collect_native_transitive_distance_to_rust(NativeDistanceCode),
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
        NativeAncestorCode, '\n\n',
        NativeCountdownSumCode, '\n\n',
        NativeClosureCode, '\n\n',
        NativeDistanceCode, '\n\n',
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
                let expr = self.regs.get("A2").cloned().map(|v| self.deref_var(&v)).unwrap_or(Value::Integer(0));
                if let Some(result) = self.eval_arith(&expr) {
                    let lhs = self.regs.get("A1").cloned().map(|v| self.deref_var(&v));
                    // Bind as integer if result is very close to a whole number
                    let final_val = if (result.round() - result).abs() < f64::EPSILON {
                        Value::Integer(result.round() as i64)
                    } else {
                        Value::Float(result)
                    };
                    match lhs {
                        Some(Value::Unbound(ref var_name)) => {
                            self.trail_binding("A1");
                            self.regs.insert("A1".to_string(), final_val.clone());
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
                let v1 = self.regs.get("A1").cloned().map(|v| self.deref_var(&v)).and_then(|v| self.eval_arith(&v));
                let v2 = self.regs.get("A2").cloned().map(|v| self.deref_var(&v)).and_then(|v| self.eval_arith(&v));
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
                let v1 = self.regs.get("A1").cloned();
                let v2 = self.regs.get("A2").cloned();
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
                if let Some(val) = self.regs.get("A1").cloned() {
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
        if let Some(val) = self.regs.get("A1").cloned() {
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
                if let (Some(val1), Some(val2)) = (self.regs.get("A1").cloned(), self.regs.get("A2").cloned()) {
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
                let list_val = self.regs.get("A1").cloned().unwrap_or(Value::List(vec![]));
                let derefed = self.deref_heap(&list_val);
                let len = match &derefed {
                    Value::List(items) => items.len() as i64,
                    _ => return false,
                };
                let len_val = Value::Integer(len);
                let lhs = self.regs.get("A2").cloned().map(|v| self.deref_var(&v));
                match lhs {
                    Some(Value::Unbound(ref var_name)) => {
                        self.trail_binding("A2");
                        self.regs.insert("A2".to_string(), len_val.clone());
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
            _ => false,
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
                let cat = match self.regs.get("A1").cloned().map(|v| self.deref_var(&v)) {
                    Some(Value::Atom(cat)) => cat,
                    _ => return false,
                };
                let root = match self.regs.get("A2").cloned().map(|v| self.deref_var(&v)) {
                    Some(Value::Atom(root)) => root,
                    _ => return false,
                };
                let hops_reg = match self.regs.get("A3").cloned() {
                    Some(val) => val,
                    None => return false,
                };
                let visited = match self.regs.get("A4").cloned().map(|v| self.deref_var(&v)) {
                    Some(Value::List(items)) => items,
                    Some(_) => return false,
                    None => return false,
                };
                let max_depth = match self.foreign_usize_config(&pred_key, "max_depth") {
                    Some(limit) => limit,
                    None => return false,
                };
                let mut hops: Vec<i64> = Vec::new();
                self.collect_native_category_ancestor_hops(&cat, &root, &visited, max_depth, &mut hops);
                if hops.is_empty() {
                    return false;
                }
                if hops.len() > 1 {
                    self.choice_points.push(ChoicePoint {
                        next_pc: self.pc,
                        saved_args: self.save_regs(),
                        stack: self.stack.clone(),
                        cp: self.cp,
                        trail_len: self.trail.len(),
                        heap_len: self.heap.len(),
                        builtin_state: Some(BuiltinState {
                            name: "foreign_results".to_string(),
                            args: vec![Value::Atom(pred_key.clone()), hops_reg.clone()],
                            data: hops[1..].iter().map(|hop| Value::Integer(*hop)).collect(),
                        }),
                        cut_barrier: self.cut_barrier,
                    });
                }
                if self.unify(&hops_reg, &Value::Integer(hops[0])) {
                    self.pc += 1; true
                } else { false }
            }
            "countdown_sum2" => {
                let n = match self.regs.get("A1").cloned().map(|v| self.deref_var(&v)) {
                    Some(Value::Integer(n)) => n,
                    _ => return false,
                };
                let sum_reg = match self.regs.get("A2").cloned() {
                    Some(val) => val,
                    None => return false,
                };
                let sum = match self.compute_native_countdown_sum(n) {
                    Some(sum) => sum,
                    None => return false,
                };
                if self.unify(&sum_reg, &Value::Integer(sum)) {
                    self.pc += 1; true
                } else { false }
            }
            "transitive_closure2" => {
                let start = match self.regs.get("A1").cloned().map(|v| self.deref_var(&v)) {
                    Some(Value::Atom(start)) => start,
                    _ => return false,
                };
                let target_reg = match self.regs.get("A2").cloned() {
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
                if nodes.len() > 1 {
                    self.choice_points.push(ChoicePoint {
                        next_pc: self.pc,
                        saved_args: self.save_regs(),
                        stack: self.stack.clone(),
                        cp: self.cp,
                        trail_len: self.trail.len(),
                        heap_len: self.heap.len(),
                        builtin_state: Some(BuiltinState {
                            name: "foreign_results".to_string(),
                            args: vec![Value::Atom(pred_key.clone()), target_reg.clone()],
                            data: nodes[1..].iter().map(|node| Value::Atom(node.clone())).collect(),
                        }),
                        cut_barrier: self.cut_barrier,
                    });
                }
                if self.unify(&target_reg, &Value::Atom(nodes[0].clone())) {
                    self.pc += 1; true
                } else { false }
            }
            "transitive_distance3" => {
                let start = match self.regs.get("A1").cloned().map(|v| self.deref_var(&v)) {
                    Some(Value::Atom(start)) => start,
                    _ => return false,
                };
                let target_reg = match self.regs.get("A2").cloned() {
                    Some(val) => val,
                    None => return false,
                };
                let dist_reg = match self.regs.get("A3").cloned() {
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
                            args: vec![Value::Atom(pred_key.clone()), target_reg.clone(), dist_reg.clone()],
                            data: results[1..].iter().map(|(node, dist)| {
                                Value::Str("__pair__".to_string(), vec![
                                    Value::Atom(node.clone()),
                                    Value::Integer(*dist),
                                ])
                            }).collect(),
                        }),
                        cut_barrier: self.cut_barrier,
                    });
                }
                if self.unify(&target_reg, &Value::Atom(results[0].0.clone()))
                    && self.unify(&dist_reg, &Value::Integer(results[0].1)) {
                    self.pc += 1; true
                } else { false }
            }
            _ => false,
        }
    }'.

compile_collect_native_category_ancestor_to_rust(Code) :-
    Code = '    pub fn collect_native_category_ancestor_hops(
        &self,
        cat: &str,
        root: &str,
        visited: &[Value],
        max_depth: usize,
        out: &mut Vec<i64>,
    ) {
        let root_val = Value::Atom(root.to_string());
        let root_seen = visited.iter().any(|item| self.deref_var(item) == root_val);
        if !root_seen {
            if let Some(values) = self.indexed_atom_fact2
                .get("category_parent/2")
                .and_then(|table| table.get(cat)) {
                if values.iter().any(|parent| parent == root) {
                    out.push(1);
                }
            }
        }

        if visited.len() >= max_depth {
            return;
        }

        if let Some(values) = self.indexed_atom_fact2
            .get("category_parent/2")
            .and_then(|table| table.get(cat)) {
            for parent in values {
                let parent_val = Value::Atom(parent.clone());
                if visited.iter().any(|item| self.deref_var(item) == parent_val) {
                    continue;
                }
                let mut next_visited: Vec<Value> = Vec::with_capacity(visited.len() + 1);
                next_visited.push(parent_val);
                next_visited.extend(visited.iter().cloned());
                let before = out.len();
                self.collect_native_category_ancestor_hops(parent, root, &next_visited, max_depth, out);
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
                let lhs = self.regs.get(&result_reg).cloned().map(|v| self.deref_var(&v));
                match lhs {
                    Some(Value::Unbound(ref var_name)) => {
                        self.trail_binding(&result_reg);
                        self.regs.insert(result_reg.clone(), result.clone());
                        self.bind_var(var_name, result);
                    }
                    Some(existing) if existing == result => {}
                    Some(_) => return false,
                    None => {
                        self.regs.insert(result_reg.clone(), result);
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
                let a2 = match self.regs.get("A2").cloned() {
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
                let native_kind = match self.foreign_native_kind(&pred_key) {
                    Some(kind) => kind,
                    None => return false,
                };
                match native_kind {
                    "category_ancestor" | "transitive_closure2" => {
                        let result_reg = match state.args.get(1) {
                            Some(val) => val.clone(),
                            None => return false,
                        };
                        if self.unify(&result_reg, &result) {
                            self.pc += 1; true
                        } else { false }
                    }
                    "transitive_distance3" => {
                        let target_reg = match state.args.get(1) {
                            Some(val) => val.clone(),
                            None => return false,
                        };
                        let dist_reg = match state.args.get(2) {
                            Some(val) => val.clone(),
                            None => return false,
                        };
                        match result {
                            Value::Str(functor, args) if functor == "__pair__" && args.len() == 2 => {
                                if self.unify(&target_reg, &args[0]) && self.unify(&dist_reg, &args[1]) {
                                    self.pc += 1; true
                                } else { false }
                            }
                            _ => false,
                        }
                    }
                    _ => false,
                }
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
                let goal = self.regs.get("A1").cloned();
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
    % Convert WAM instruction string to Rust Instruction enum literals
    wam_code_to_rust_instructions(WamCode, Pred/Arity, Options, InstrLiterals, LabelLiterals),
    foreign_wrapper_setup(Pred/Arity, Options, ForeignSetup, RunExpr),
    format(string(RustCode),
'/// WAM-compiled predicate: ~w/~w
/// Compiled via WAM for predicates that resist native lowering.
pub fn ~w(~w) -> bool {
    use std::collections::HashMap;
    let code: Vec<Instruction> = vec![
~w
    ];
    let mut labels: HashMap<String, usize> = HashMap::new();
~w
    vm.code = code;
    vm.labels = labels;
    vm.pc = 1;
~w
~w
    ~w
}', [PredStr, Arity, PredStr, ArgList, InstrLiterals, LabelLiterals, ForeignSetup, ArgSetup, RunExpr]).

foreign_wrapper_setup(Pred/Arity, Options, Setup, RunExpr) :-
    (   option(foreign_lowering(ForeignSpec), Options),
        rust_foreign_spec(Pred/Arity, ForeignSpec, SetupOps, EntryPred/EntryArity)
    ->  rust_foreign_setup_code(SetupOps, Setup),
        format(string(RunExpr),
            'vm.execute_foreign_predicate("~w", ~w)', [EntryPred, EntryArity])
    ;   Setup = "",
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

rust_fact_pairs_literal(Pairs, Literal) :-
    maplist(rust_fact_pair_literal, Pairs, PairLiterals),
    atomic_list_concat(PairLiterals, ', ', Joined),
    format(string(Literal), '[~w]', [Joined]).

rust_fact_pair_literal(Left-Right, Literal) :-
    escape_rust_string(Left, ELeft),
    escape_rust_string(Right, ERight),
    format(string(Literal), '("~w", "~w")', [ELeft, ERight]).

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
            format(string(InstrEntry), '        ~w,', [RustInstr]),
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
write_wam_rust_project(Predicates, Options, ProjectDir) :-
    option(module_name(ModuleName), Options, 'wam_generated'),
    get_time(TimeStamp),
    format_time(string(Date), "%Y-%m-%d %H:%M:%S", TimeStamp),

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

    % Compile predicates and generate lib.rs
    compile_predicates_for_project(Predicates, Options, PredicatesCode),
    render_named_template(rust_wam_lib,
        [module_name=ModuleName, date=Date, predicates_code=PredicatesCode],
        LibContent),
    directory_file_path(SrcDir, 'lib.rs', LibPath),
    write_file(LibPath, LibContent),

    format('WAM Rust project created at: ~w~n', [ProjectDir]),
    format('  Predicates compiled: ~w~n', [Predicates]).

%% compile_predicates_for_project(+Predicates, +Options, -Code)
%  Compiles each predicate, trying native lowering first, then WAM fallback.
compile_predicates_for_project([], _, "").
compile_predicates_for_project([PredIndicator|Rest], Options, Code) :-
    (   PredIndicator = Module:Pred/Arity -> true
    ;   PredIndicator = Pred/Arity, Module = user
    ),
    (   % Try native Rust lowering first
        catch(
            rust_target:compile_predicate_to_rust(Module:Pred/Arity,
                [include_main(false)|Options], PredCode),
            _, fail)
    ->  format(user_error, '  ~w/~w: native lowering~n', [Pred, Arity]),
        Strategy = native
    ;   % Fall back to WAM compilation
        option(wam_fallback(WamFB), Options, true),
        WamFB \== false,
        wam_target:compile_predicate_to_wam(Module:Pred/Arity, Options, WamCode),
        compile_wam_predicate_to_rust(Pred/Arity, WamCode, Options, PredCode)
    ->  format(user_error, '  ~w/~w: WAM fallback~n', [Pred, Arity]),
        Strategy = wam
    ;   % Neither worked — emit a stub
        format(string(PredCode),
            '// ~w/~w: compilation failed (neither native nor WAM)', [Pred, Arity]),
        Strategy = failed
    ),
    compile_predicates_for_project(Rest, Options, RestCode),
    (   RestCode == ""
    ->  format(string(Code), "// Strategy: ~w\n~w", [Strategy, PredCode])
    ;   format(string(Code), "// Strategy: ~w\n~w\n\n~w", [Strategy, PredCode, RestCode])
    ).

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
