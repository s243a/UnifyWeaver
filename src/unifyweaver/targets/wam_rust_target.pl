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
                        self.stack.push(StackEntry::WriteCtx(arity));
                        self.pc += 1; true
                    } else if let Value::Ref(addr) = &val {
                        // Read mode on heap ref
                        if let Some(Value::Str(s, _)) = self.heap.get(*addr) {
                            if s == &format!("str({})", fn_str) {
                                let arity = fn_str.split(\'/\').nth(1)
                                    .and_then(|s| s.parse::<usize>().ok()).unwrap_or(0);
                                let args = self.heap_subargs(addr + 1, arity);
                                self.stack.push(StackEntry::UnifyCtx(args));
                                self.pc += 1; true
                            } else { false }
                        } else { false }
                    } else if let Some((f, args)) = val.univ() {
                        // Read mode on Prolog compound
                        let check = format!("{}/{}", f, args.len());
                        if check == *fn_str {
                            self.stack.push(StackEntry::UnifyCtx(args.to_vec()));
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
                        self.stack.push(StackEntry::WriteCtx(2));
                        self.pc += 1; true
                    } else if let Value::List(items) = &val {
                        if let Some((head, tail)) = items.split_first() {
                            self.stack.push(StackEntry::UnifyCtx(
                                vec![head.clone(), Value::List(tail.to_vec())]));
                            self.pc += 1; true
                        } else { false }
                    } else if let Value::Ref(addr) = &val {
                        if let Some(Value::Str(s, _)) = self.heap.get(*addr) {
                            if s == "str(./2)" {
                                let args = self.heap_subargs(addr + 1, 2);
                                self.stack.push(StackEntry::UnifyCtx(args));
                                self.pc += 1; true
                            } else { false }
                        } else { false }
                    } else { false }
                } else { false }'.

wam_instruction_arm('Instruction::UnifyVariable(xn)', Body) :-
    Body = '                if let Some(StackEntry::UnifyCtx(args)) = self.stack.last().cloned() {
                    if let Some(arg) = args.first().cloned() {
                        let rest: Vec<Value> = args[1..].to_vec();
                        self.stack.pop();
                        if !rest.is_empty() {
                            self.stack.push(StackEntry::UnifyCtx(rest));
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
                        self.stack.pop();
                        if n - 1 > 0 { self.stack.push(StackEntry::WriteCtx(n - 1)); }
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
                            self.stack.pop();
                            if !rest.is_empty() {
                                self.stack.push(StackEntry::UnifyCtx(rest));
                            }
                            self.pc += 1; true
                        } else { false }
                    } else { false }
                } else if let Some(StackEntry::WriteCtx(n)) = self.stack.last().cloned() {
                    if n > 0 {
                        if let Some(val) = self.get_reg(xn) {
                            self.heap.push(val);
                            self.stack.pop();
                            if n - 1 > 0 { self.stack.push(StackEntry::WriteCtx(n - 1)); }
                            self.pc += 1; true
                        } else { false }
                    } else { false }
                } else { false }'.

wam_instruction_arm('Instruction::UnifyConstant(c)', Body) :-
    Body = '                if let Some(StackEntry::UnifyCtx(args)) = self.stack.last().cloned() {
                    if let Some(arg) = args.first().cloned() {
                        if arg == *c || arg.is_unbound() {
                            let rest: Vec<Value> = args[1..].to_vec();
                            self.stack.pop();
                            if !rest.is_empty() {
                                self.stack.push(StackEntry::UnifyCtx(rest));
                            }
                            self.pc += 1; true
                        } else { false }
                    } else { false }
                } else if let Some(StackEntry::WriteCtx(n)) = self.stack.last().cloned() {
                    if n > 0 {
                        self.heap.push(c.clone());
                        self.stack.pop();
                        if n - 1 > 0 { self.stack.push(StackEntry::WriteCtx(n - 1)); }
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
    Body = '                let addr = self.heap.len();
                let ref_val = Value::Ref(addr);
                self.heap.push(Value::Str(format!("str({})", fn_str), vec![]));
                if let Some(val) = self.regs.get(ai) {
                    if val.is_unbound() {
                        let dv = self.deref_heap(val);
                        if let Value::Unbound(name) = dv {
                            self.trail_binding(&name);
                            self.put_reg(&name, ref_val.clone());
                        }
                    }
                }
                self.regs.insert(ai.clone(), ref_val);
                self.pc += 1; true'.

wam_instruction_arm('Instruction::PutList(ai)', Body) :-
    Body = '                let addr = self.heap.len();
                let ref_val = Value::Ref(addr);
                self.heap.push(Value::Str("str(./2)".to_string(), vec![]));
                if let Some(val) = self.regs.get(ai) {
                    if val.is_unbound() {
                        let dv = self.deref_heap(val);
                        if let Value::Unbound(name) = dv {
                            self.trail_binding(&name);
                            self.put_reg(&name, ref_val.clone());
                        }
                    }
                }
                self.regs.insert(ai.clone(), ref_val);
                self.pc += 1; true'.

wam_instruction_arm('Instruction::SetVariable(xn)', Body) :-
    Body = '                let addr = self.heap.len();
                let var = Value::Unbound(format!("_H{}", addr));
                self.heap.push(var.clone());
                self.put_reg(xn, var);
                self.pc += 1; true'.

wam_instruction_arm('Instruction::SetValue(xn)', Body) :-
    Body = '                if let Some(val) = self.get_reg(xn) {
                    self.heap.push(val);
                    self.pc += 1; true
                } else { false }'.

wam_instruction_arm('Instruction::SetConstant(c)', Body) :-
    Body = '                self.heap.push(c.clone());
                self.pc += 1; true'.

% --- Control Instructions ---

wam_instruction_arm('Instruction::Allocate', Body) :-
    Body = '                use std::collections::HashMap;
                self.stack.push(StackEntry::Env(self.cp, HashMap::new()));
                self.pc += 1; true'.

wam_instruction_arm('Instruction::Deallocate', Body) :-
    Body = '                if let Some(StackEntry::Env(old_cp, _)) = self.stack.pop() {
                    self.cp = old_cp;
                    self.pc += 1; true
                } else { false }'.

wam_instruction_arm('Instruction::Call(p, _arity)', Body) :-
    Body = '                if let Some(&target_pc) = self.labels.get(p) {
                    self.cp = self.pc + 1;
                    self.pc = target_pc;
                    true
                } else { false }'.

wam_instruction_arm('Instruction::Execute(p)', Body) :-
    Body = '                if let Some(&target_pc) = self.labels.get(p) {
                    self.pc = target_pc;
                    true
                } else { false }'.

wam_instruction_arm('Instruction::Proceed', Body) :-
    Body = '                let ret = self.cp;
                self.cp = 0; // halt
                self.pc = ret;
                true'.

wam_instruction_arm('Instruction::BuiltinCall(op, arity)', Body) :-
    Body = '                self.execute_builtin(op, *arity)'.

% --- Choice Point Instructions ---

wam_instruction_arm('Instruction::TryMeElse(label)', Body) :-
    Body = '                if let Some(&next_pc) = self.labels.get(label) {
                    self.choice_points.push(ChoicePoint {
                        next_pc,
                        regs: self.regs.clone(),
                        stack: self.stack.clone(),
                        cp: self.cp,
                        trail: self.trail.clone(),
                        builtin_state: None,
                        bindings: self.bindings.clone(),
                    });
                    self.pc += 1; true
                } else { false }'.

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

% --- Indexing Instructions ---

wam_instruction_arm('Instruction::SwitchOnConstant(table)', Body) :-
    Body = '                if let Some(val) = self.regs.get("A1").cloned() {
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
    compile_resume_builtin_to_rust(ResumeCode),
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
        ResumeCode, '\n\n',
        ArithCode
    ], RustCode).

compile_run_loop_to_rust(Code) :-
    Code = '    /// Main execution loop. Runs until halt (pc=0) or failure.
    pub fn run(&mut self) -> bool {
        loop {
            if self.pc == 0 { return true; }
            if let Some(instr) = self.fetch().cloned() {
                if !self.step(&instr) {
                    if !self.backtrack() { return false; }
                }
            } else {
                return false;
            }
        }
    }'.

compile_backtrack_to_rust(Code0) :-
    Code0 = '    /// Restore state from the top choice point without popping it.
    /// In standard WAM semantics, the choice point is kept alive so that
    /// retry_me_else can update its next_pc.  Only trust_me pops it.
    pub fn backtrack(&mut self) -> bool {
        if let Some(cp) = self.choice_points.last().cloned() {
            self.pc = cp.next_pc;
            self.regs = cp.regs;
            self.stack = cp.stack;
            self.cp = cp.cp;
            // Unwind trail
            self.unwind_trail(&cp.trail);
            self.trail = cp.trail;
            self.bindings = cp.bindings;

            if let Some(state) = cp.builtin_state {
                // Pop for non-deterministic builtins (they manage their own state)
                self.choice_points.pop();
                return self.resume_builtin(state);
            }
            true
        } else {
            false
        }
    }
'.

compile_unwind_trail_to_rust(Code) :-
    Code = '    /// Undo bindings recorded since the saved trail state.
    /// Handles both register entries (key = "A1", "Y2", etc.) and
    /// binding-table entries (key = "__binding__<var_name>").
    fn unwind_trail(&mut self, saved: &[TrailEntry]) {
        let new_entries = self.trail.len() - saved.len();
        for entry in self.trail.iter().rev().take(new_entries) {
            if let Some(binding_key) = entry.key.strip_prefix("__binding__") {
                // Undo a variable binding table entry
                match &entry.old_value {
                    Some(val) => { self.bindings.insert(binding_key.to_string(), val.clone()); }
                    None => { self.bindings.remove(binding_key); }
                }
            } else {
                // Undo a register entry
                match &entry.old_value {
                    Some(val) => { self.regs.insert(entry.key.clone(), val.clone()); }
                    None => { self.regs.remove(&entry.key); }
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
            "!/0" => { self.choice_points.clear(); self.pc += 1; true }
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
                                next_pc: self.pc, // Stay on this instruction
                                regs: self.regs.clone(),
                                stack: self.stack.clone(),
                                cp: self.cp,
                                trail: self.trail.clone(),
                                builtin_state: Some(BuiltinState {
                                    name: "member/2".to_string(),
                                    args: vec![val1.clone(), val2.clone()],
                                    data: vec![Value::Integer(1)], // Start from index 1 next time
                                }),
                                bindings: self.bindings.clone(),
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

compile_resume_builtin_to_rust(Code) :-
    Code = '    fn resume_builtin(&mut self, state: BuiltinState) -> bool {
        match state.name.as_str() {
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
                            regs: self.regs.clone(),
                            stack: self.stack.clone(),
                            cp: self.cp,
                            trail: self.trail.clone(),
                            builtin_state: Some(BuiltinState {
                                name: "member/2".to_string(),
                                args: state.args.clone(),
                                data: vec![Value::Integer((idx + 1) as i64)],
                            }),
                            bindings: self.bindings.clone(),
                        });
                    }
                    
                    // Try current element
                    if self.unify(&val1, &items[idx]) {
                        self.pc += 1; true
                    } else { false }
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
                // Negation-as-failure: \\+(Goal)
                // Goal is in A1 as a Str(functor, args) or via label dispatch.
                // Save state, try the goal, negate the result.
                let goal = self.regs.get("A1").cloned();
                let goal = goal.map(|v| self.deref_heap(&v));
                match goal {
                    Some(Value::Str(functor, args)) => {
                        // Save VM state
                        let saved_regs = self.regs.clone();
                        let saved_stack = self.stack.clone();
                        let saved_trail = self.trail.clone();
                        let saved_cp = self.cp;
                        let saved_pc = self.pc;
                        let saved_choice_points = self.choice_points.clone();
                        let saved_heap = self.heap.clone();
                        let saved_bindings = self.bindings.clone();

                        // Try as builtin first
                        let pred_key = format!("{}", functor);
                        let arity = args.len();
                        for (i, arg) in args.iter().enumerate() {
                            self.set_reg(&format!("A{}", i + 1), arg.clone());
                        }

                        let goal_succeeded = if self.execute_builtin(&pred_key, arity) {
                            true
                        } else if let Some(&target_pc) = self.labels.get(&pred_key) {
                            // Try as a compiled predicate via label dispatch
                            self.pc = target_pc;
                            self.cp = 0;
                            self.run()
                        } else {
                            false
                        };

                        // Restore state regardless
                        self.regs = saved_regs;
                        self.stack = saved_stack;
                        self.trail = saved_trail;
                        self.cp = saved_cp;
                        self.choice_points = saved_choice_points;
                        self.heap = saved_heap;
                        self.bindings = saved_bindings;

                        // Negate: if goal succeeded, \\+ fails; if goal failed, \\+ succeeds
                        if goal_succeeded {
                            false
                        } else {
                            self.pc = saved_pc + 1;
                            true
                        }
                    }
                    _ => false,
                }
            }
            _ => false,
        }
    }'.

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
        if args.len() == 2 {
            let a = self.eval_arith(&args[0])?;
            let b = self.eval_arith(&args[1])?;
            match op {
                "+" => Some(a + b),
                "-" => Some(a - b),
                "*" => Some(a * b),
                "/" if b != 0.0 => Some(a / b),
                "//" if b != 0.0 => Some((a / b).floor()),
                "mod" if b != 0.0 => Some(a % b),
                _ => None,
            }
        } else if args.len() == 1 {
            let a = self.eval_arith(&args[0])?;
            match op {
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
compile_wam_predicate_to_rust(Pred/Arity, WamCode, _Options, RustCode) :-
    atom_string(Pred, PredStr),
    build_rust_wam_arg_list(Arity, ArgList),
    build_rust_wam_arg_setup(Arity, ArgSetup),
    % Convert WAM instruction string to Rust Instruction enum literals
    wam_code_to_rust_instructions(WamCode, InstrLiterals, LabelLiterals),
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
    vm.run()
}', [PredStr, Arity, PredStr, ArgList, InstrLiterals, LabelLiterals, ArgSetup]).

%% wam_code_to_rust_instructions(+WamCodeStr, -InstrLiterals, -LabelLiterals)
%  Parses a WAM code string and generates Rust vec![] entries and label map inserts.
wam_code_to_rust_instructions(WamCode, InstrLiterals, LabelLiterals) :-
    atom_string(WamCode, WamStr),
    split_string(WamStr, "\n", "", Lines),
    wam_lines_to_rust(Lines, 1, InstrParts, LabelParts),
    atomic_list_concat(InstrParts, '\n', InstrLiterals),
    atomic_list_concat(LabelParts, '\n', LabelLiterals).

wam_lines_to_rust([], _, [], []).
wam_lines_to_rust([Line|Rest], PC, Instrs, Labels) :-
    split_string(Line, " \t,", " \t,", Parts),
    delete(Parts, "", CleanParts),
    (   CleanParts == []
    ->  wam_lines_to_rust(Rest, PC, Instrs, Labels)
    ;   CleanParts = [First|_],
        (   % Label line: "pred/2:" or "L_pred_2_2:"
            sub_string(First, _, 1, 0, ":")
        ->  sub_string(First, 0, _, 1, LabelName),
            format(string(LabelInsert),
                '    labels.insert("~w".to_string(), ~w);', [LabelName, PC]),
            Labels = [LabelInsert|RestLabels],
            wam_lines_to_rust(Rest, PC, Instrs, RestLabels)
        ;   % Instruction line
            wam_line_to_rust_instr(CleanParts, RustInstr),
            format(string(InstrEntry), '        ~w,', [RustInstr]),
            NPC is PC + 1,
            Instrs = [InstrEntry|RestInstrs],
            wam_lines_to_rust(Rest, NPC, RestInstrs, Labels)
        )
    ).

%% wam_line_to_rust_instr(+Parts, -RustExpr)
%  Converts parsed WAM instruction parts to a Rust Instruction enum literal.
wam_line_to_rust_instr(["get_constant", C, Ai], Rust) :-
    clean_comma(C, CC), clean_comma(Ai, CAi),
    format(string(Rust),
        'Instruction::GetConstant(Value::Atom("~w".to_string()), "~w".to_string())',
        [CC, CAi]).
wam_line_to_rust_instr(["get_variable", Xn, Ai], Rust) :-
    clean_comma(Xn, CXn), clean_comma(Ai, CAi),
    format(string(Rust),
        'Instruction::GetVariable("~w".to_string(), "~w".to_string())',
        [CXn, CAi]).
wam_line_to_rust_instr(["get_value", Xn, Ai], Rust) :-
    clean_comma(Xn, CXn), clean_comma(Ai, CAi),
    format(string(Rust),
        'Instruction::GetValue("~w".to_string(), "~w".to_string())',
        [CXn, CAi]).
wam_line_to_rust_instr(["get_structure", FN, Ai], Rust) :-
    clean_comma(FN, CFN), clean_comma(Ai, CAi),
    format(string(Rust),
        'Instruction::GetStructure("~w".to_string(), "~w".to_string())',
        [CFN, CAi]).
wam_line_to_rust_instr(["get_list", Ai], Rust) :-
    clean_comma(Ai, CAi),
    format(string(Rust),
        'Instruction::GetList("~w".to_string())', [CAi]).
wam_line_to_rust_instr(["unify_variable", Xn], Rust) :-
    format(string(Rust),
        'Instruction::UnifyVariable("~w".to_string())', [Xn]).
wam_line_to_rust_instr(["unify_value", Xn], Rust) :-
    format(string(Rust),
        'Instruction::UnifyValue("~w".to_string())', [Xn]).
wam_line_to_rust_instr(["unify_constant", C], Rust) :-
    (   number_string(N, C)
    ->  format(string(Rust),
            'Instruction::UnifyConstant(Value::Integer(~w))', [N])
    ;   format(string(Rust),
            'Instruction::UnifyConstant(Value::Atom("~w".to_string()))', [C])
    ).
wam_line_to_rust_instr(["put_constant", C, Ai], Rust) :-
    clean_comma(C, CC), clean_comma(Ai, CAi),
    format(string(Rust),
        'Instruction::PutConstant(Value::Atom("~w".to_string()), "~w".to_string())',
        [CC, CAi]).
wam_line_to_rust_instr(["put_variable", Xn, Ai], Rust) :-
    clean_comma(Xn, CXn), clean_comma(Ai, CAi),
    format(string(Rust),
        'Instruction::PutVariable("~w".to_string(), "~w".to_string())',
        [CXn, CAi]).
wam_line_to_rust_instr(["put_value", Xn, Ai], Rust) :-
    clean_comma(Xn, CXn), clean_comma(Ai, CAi),
    format(string(Rust),
        'Instruction::PutValue("~w".to_string(), "~w".to_string())',
        [CXn, CAi]).
wam_line_to_rust_instr(["put_structure", FN, Ai], Rust) :-
    clean_comma(FN, CFN), clean_comma(Ai, CAi),
    format(string(Rust),
        'Instruction::PutStructure("~w".to_string(), "~w".to_string())',
        [CFN, CAi]).
wam_line_to_rust_instr(["put_list", Ai], Rust) :-
    format(string(Rust),
        'Instruction::PutList("~w".to_string())', [Ai]).
wam_line_to_rust_instr(["set_variable", Xn], Rust) :-
    format(string(Rust),
        'Instruction::SetVariable("~w".to_string())', [Xn]).
wam_line_to_rust_instr(["set_value", Xn], Rust) :-
    format(string(Rust),
        'Instruction::SetValue("~w".to_string())', [Xn]).
wam_line_to_rust_instr(["set_constant", C], Rust) :-
    (   number_string(N, C)
    ->  format(string(Rust),
            'Instruction::SetConstant(Value::Integer(~w))', [N])
    ;   format(string(Rust),
            'Instruction::SetConstant(Value::Atom("~w".to_string()))', [C])
    ).
wam_line_to_rust_instr(["allocate"], "Instruction::Allocate").
wam_line_to_rust_instr(["deallocate"], "Instruction::Deallocate").
wam_line_to_rust_instr(["call", P, N], Rust) :-
    clean_comma(P, CP), clean_comma(N, CN),
    escape_rust_string(CP, ECP),
    (   number_string(Num, CN) -> true ; Num = 0 ),
    format(string(Rust),
        'Instruction::Call("~w".to_string(), ~w)', [ECP, Num]).
wam_line_to_rust_instr(["execute", P], Rust) :-
    escape_rust_string(P, EP),
    format(string(Rust),
        'Instruction::Execute("~w".to_string())', [EP]).
wam_line_to_rust_instr(["proceed"], "Instruction::Proceed").
wam_line_to_rust_instr(["builtin_call", Op, N], Rust) :-
    clean_comma(Op, COp), clean_comma(N, CN),
    escape_rust_string(COp, ECOp),
    (   number_string(Num, CN) -> true ; Num = 0 ),
    format(string(Rust),
        'Instruction::BuiltinCall("~w".to_string(), ~w)', [ECOp, Num]).
wam_line_to_rust_instr(["try_me_else", Label], Rust) :-
    format(string(Rust),
        'Instruction::TryMeElse("~w".to_string())', [Label]).
wam_line_to_rust_instr(["trust_me"], "Instruction::TrustMe").
wam_line_to_rust_instr(["retry_me_else", Label], Rust) :-
    format(string(Rust),
        'Instruction::RetryMeElse("~w".to_string())', [Label]).
% Indexing instructions
wam_line_to_rust_instr(["switch_on_constant"|Entries], Rust) :-
    maplist(parse_index_entry_constant, Entries, RustEntries),
    atomic_list_concat(RustEntries, ', ', Joined),
    format(string(Rust), 'Instruction::SwitchOnConstant(vec![~w])', [Joined]).
wam_line_to_rust_instr(["switch_on_structure"|Entries], Rust) :-
    maplist(parse_index_entry_structure, Entries, RustEntries),
    atomic_list_concat(RustEntries, ', ', Joined),
    format(string(Rust), 'Instruction::SwitchOnStructure(vec![~w])', [Joined]).
wam_line_to_rust_instr(["switch_on_constant_a2"|Entries], Rust) :-
    maplist(parse_index_entry_constant, Entries, RustEntries),
    atomic_list_concat(RustEntries, ', ', Joined),
    format(string(Rust), 'Instruction::SwitchOnConstantA2(vec![~w])', [Joined]).
% Fallback for unknown instructions
wam_line_to_rust_instr(Parts, Rust) :-
    atomic_list_concat(Parts, ' ', Joined),
    format(string(Rust), '/* unknown: ~w */', [Joined]).

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
