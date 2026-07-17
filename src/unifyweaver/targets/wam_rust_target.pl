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
    graph_analysis/4,                    % +EdgePred, +Inputs, +Stages, +Queries (directive)
    graph_analysis_options/2,            % -Options, -QueryPreds (expand asserted decls)
    graph_analysis_expand/5,             % +EdgePred,+Inputs,+Stages,+Queries,-Options (pure)
    cargo_check_project/2,               % +ProjectDir, -Result
    detect_kernels/2,                    % +Predicates, -DetectedKernels
    generate_setup_foreign_predicates_rust/2, % +DetectedKernels, -RustCode
    escape_rust_string/2,
    resolve_lmdb_crate/2,                % +Spec, -Crate  (lmdb_zero|heed|auto -> concrete)
    rust_fact_table_classify/3,          % +Module:Pred/Arity, +Options, -fact_info(Arity,Rows)
    emit_fact_table_rust/4               % +Pred/Arity, +fact_info, +Options, -RustCode
]).

:- use_module(library(lists)).
:- use_module(library(option)).
:- use_module('../core/template_system').
:- use_module('../bindings/rust_wam_bindings').
:- use_module('../targets/wam_target', [compile_predicate_to_wam/3]).
% T7 compile-time parallel-aggregate gate (consumer of the cost machinery).
:- use_module('../core/parallel_gate', [forkable_aggregate/3, goal_parallel_decision/3, parallel_aggregate_transform/5, parallel_aggregate_transform/6, lift_embedded_aggregate/6, aggregate_result/2]).
:- use_module('../core/cost_analysis', [build_cost_model/2, goal_cost_tier/3]).
:- use_module('../targets/wam_text_parser', [
    wam_classify_constant_token/2,
    wam_tokenize_line/2
]).
:- use_module('../targets/wam_rust_lowered_emitter', [
    wam_rust_lowerable/3,
    lower_predicate_to_rust/4,
    rust_lowered_func_name/2
]).
:- use_module('../targets/wam_runtime_parser_capability', [
    parser_dependent_body_goal/2,
    wam_target_runtime_parser/3
]).
:- use_module('../core/prolog_term_parser', []).
:- use_module('../core/cpp_runtime_parser_wrappers', []).
:- use_module('../core/recursive_kernel_detection', [
    detect_recursive_kernel/4,
    kernel_metadata/4,
    kernel_config/2,
    kernel_register_layout/2,
    kernel_native_call/2
]).

rust_safe_function_name(Pred/Arity, FuncName) :-
    !,
    rust_safe_function_name(Pred, BaseName),
    format(atom(FuncName), '~w_~w', [BaseName, Arity]).
rust_safe_function_name(Pred, FuncName) :-
    atom_string(Pred, PredStr),
    string_codes(PredStr, Codes),
    maplist(rust_safe_identifier_code, Codes, SafeCodes),
    string_codes(FuncStr, SafeCodes),
    atom_string(FuncName, FuncStr).

rust_safe_identifier_code(C, C) :-
    (   C >= 0'a, C =< 0'z
    ;   C >= 0'A, C =< 0'Z
    ;   C >= 0'0, C =< 0'9
    ;   C =:= 0'_
    ),
    !.
rust_safe_identifier_code(_, 0'_).

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
                    // Empty-list aliasing: the [] atom and an empty Value::List
                    // are the same term. A list tail peeled down to its end is
                    // Value::List([]), and a clause head get_constant [] must
                    // match it (e.g. append/3 base case capp([],L,L)).
                    Some(Value::List(ref items)) if items.is_empty()
                        && matches!(c, Value::Atom(s) if s == "[]") => {
                        self.pc += 1; true
                    }
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
    Body = '                // get_value Xn, Ai is full unification of the two.
                // The old hand-rolled version only did raw equality plus
                // unbound-binding, so it could not unify two bound compound
                // terms or follow a heap Ref — e.g. append/3 binding the
                // accumulated result list against a tail cell. Route through
                // unify(), which derefs (incl. Ref->heap), binds vars, and
                // structurally unifies with cons-cell aliasing.
                let val_a = self.get_reg_raw(ai);
                let val_x = self.get_reg(xn);
                match (val_a, val_x) {
                    (Some(a), Some(x)) => {
                        if self.unify(&a, &x) { self.pc += 1; true } else { false }
                    }
                    _ => false,
                }'.

wam_instruction_arm('Instruction::GetStructure(fn_str, ai)', Body) :-
    Body = '                if let Some(val) = self.get_reg_raw(ai) {
                    if val.is_unbound() {
                        // Write mode
                        let addr = self.heap.len();
                        let arity = fn_str.split(\'/\').nth(1)
                            .and_then(|s| s.parse::<usize>().ok()).unwrap_or(0);
                        self.heap.push(Value::Str(fn_str.clone(), vec![]));
                        for _ in 0..arity {
                            self.heap.push(Value::Atom("__struct_arg__".to_string()));
                        }
                        self.trail_binding(ai);
                        if let Value::Unbound(ref name) = val {
                            self.bind_var(name, Value::Ref(addr));
                        }
                        self.set_reg_str(ai, Value::Ref(addr));
                        self.smut().push(StackEntry::WriteCtx(addr));
                        self.pc += 1; true
                    } else if let Value::Ref(addr) = &val {
                        // Read mode on heap ref
                        if let Some(Value::Str(s, _)) = self.heap.get(*addr) {
                            if s == fn_str || s == &format!("str({})", fn_str) {
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
    Body = '                if let Some(raw) = self.get_reg_raw(ai) {
                    // Deref BEFORE the unbound test: a bound variable is still
                    // typed Unbound, and treating a bound list tail as unbound
                    // would wrongly take the write-mode branch.
                    let val = self.deref_var(&raw);
                    if val.is_unbound() {
                        let addr = self.heap.len();
                        self.heap.push(Value::Str("./2".to_string(), vec![]));
                        self.heap.push(Value::Atom("__struct_arg__".to_string()));
                        self.heap.push(Value::Atom("__struct_arg__".to_string()));
                        self.trail_binding(ai);
                        if let Value::Unbound(ref name) = val {
                            self.bind_var(name, Value::Ref(addr));
                        }
                        self.set_reg_str(ai, Value::Ref(addr));
                        self.smut().push(StackEntry::WriteCtx(addr));
                        self.pc += 1; true
                    } else if let Value::List(items) = &val {
                        if let Some((head, tail)) = items.split_first() {
                            self.smut().push(StackEntry::UnifyCtx(
                                vec![head.clone(), Value::List(tail.to_vec())]));
                            self.pc += 1; true
                        } else { false }
                    } else if let Value::Str(s, args) = &val {
                        // Materialised cons cell, e.g. "[|]/2"/"./2".
                        if self.is_cons_functor(s) && args.len() == 2 {
                            self.smut().push(StackEntry::UnifyCtx(args.clone()));
                            self.pc += 1; true
                        } else { false }
                    } else if let Value::Ref(addr) = &val {
                        if let Some(Value::Str(s, _)) = self.heap.get(*addr) {
                            if self.is_cons_functor(s) {
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
                } else if let Some(StackEntry::WriteCtx(_marker)) = self.stack.last().cloned() {
                    let var = Value::Unbound(format!("_H{}", self.var_counter));
                    self.var_counter += 1;
                    self.set_heap_or_list(var.clone());
                    self.put_reg(xn, var);
                    self.pc += 1; true
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
                } else if let Some(StackEntry::WriteCtx(_marker)) = self.stack.last().cloned() {
                    if let Some(val) = self.get_reg(xn) {
                        self.set_heap_or_list(val);
                        self.pc += 1; true
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
                } else if let Some(StackEntry::WriteCtx(_marker)) = self.stack.last().cloned() {
                    self.set_heap_or_list(c.clone());
                    self.pc += 1; true
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
                // If this register holds an unbound placeholder (one a prior
                // set_variable embedded into an enclosing term), bind it to the
                // new structure so the embedded copy resolves here. The compiler
                // builds nested terms outer-first — +(+(A,B),C), or a list tail
                // cell — leaving a placeholder a later put_structure must fill.
                //
                // A-REGISTER EXCEPTION (M139/M140 bind-through class): A
                // registers are argument STAGING — their old occupant is an
                // unrelated variable (often a clause-head argument), and
                // binding it to the new cell creates a cyclic term
                // (X = f(X), then deref_heap recurses forever). Top-down
                // chaining placeholders only ever live in X/Y registers
                // (set_variable Xn), so the bind-through is conditioned on
                // the register class — the same design the LLVM target
                // adopted in M140 after the identical bug.
                if ai.as_bytes().first() != Some(&b''A'') {
                    if let Some(cur) = self.get_reg_raw(ai) {
                        if let Value::Unbound(name) = self.deref_var(&cur) {
                            self.bind_var(&name, Value::Ref(addr));
                        }
                    }
                }
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
    Body = '                let var = Value::Unbound(format!("_H{}", self.var_counter));
                self.var_counter += 1;
                self.put_reg(xn, var.clone());
                // Write the fresh variable into the current structure/list arg
                // slot, exactly as set_value/set_constant do. Without this the
                // placeholder never lands in the term being built, so any
                // structure/list with a variable argument (nested arithmetic
                // +(+(A,B),C); list tail cells [H|T]) had its args misaligned.
                self.set_heap_or_list(var);
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

wam_instruction_arm('Instruction::BaseCategoryAncestorBind(cat_reg, target_reg, hops_reg, visited_reg)', Body) :-
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
                let hops_val = match self.get_reg(hops_reg) {
                    Some(val) => val,
                    None => return false,
                };
                match hops_val {
                    Value::Unbound(var_name) => self.bind_var(&var_name, Value::Integer(1)),
                    Value::Integer(1) => {},
                    Value::Atom(ref raw) if raw == "1" => {},
                    _ => return false,
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
                match out_val {
                    Value::Unbound(var_name) => self.bind_var(&var_name, result),
                    Value::Integer(n) if result == Value::Integer(n) => {},
                    Value::Float(f) if result == Value::Float(f) => {},
                    Value::Atom(ref raw) if result == Value::Atom(raw.clone()) => {},
                    other => {
                        if !self.unify(&other, &result) {
                            return false;
                        }
                    }
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
                self.cut_barrier = self.pending_cut_barrier
                    .take()
                    .unwrap_or(self.choice_points.len());
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
                } else if p == "retract/1" {
                    self.dynamic_retract_call(self.pc + 1)
                } else if p == "clause/2" {
                    self.dynamic_clause_call(self.pc + 1)
                } else if p == "current_predicate/1" {
                    self.current_predicate_call(self.pc + 1)
                } else if p == "assert/1" {
                    self.execute_assert_builtin("assert/1")
                } else if p == "read_term/2" {
                    let options = self.get_reg_raw("A2");
                    self.execute_read_term_builtin(options.as_ref())
                } else if p == "atomic/1" {
                    self.execute_builtin(p, *_arity)
                } else if self.foreign_predicates.contains(p) {
                    self.cp = self.pc + 1;
                    if self.execute_foreign_predicate(p, *_arity) {
                        self.pc = self.cp;
                        true
                    } else { false }
                } else if let Some(__ftr) = crate::fact_table_call(self, p, self.pc + 1) {
                    // T9 fact table called as a deterministic-position goal:
                    // continuation is the next instruction; the fact enumerator
                    // sets pc/cp and leaves a choice point for further rows.
                    __ftr
                } else if self.dynamic_call(p, self.pc + 1) {
                    true
                } else if Self::is_iso_meta_builtin(p) {
                    // ISO meta-builtins (catch/3, throw/1, succ/2) are
                    // emitted by the shared WAM compiler as Call rather
                    // than BuiltinCall; route them through the builtin
                    // dispatch (mirrors the F# isIsoMetaBuiltin arm).
                    self.execute_builtin(p, *_arity)
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
                } else if p == "retract/1" {
                    self.dynamic_retract_call(self.cp)
                } else if p == "clause/2" {
                    self.dynamic_clause_call(self.cp)
                } else if p == "current_predicate/1" {
                    self.current_predicate_call(self.cp)
                } else if p == "assert/1" {
                    if self.execute_assert_builtin("assert/1") {
                        self.pc = self.cp;
                        true
                    } else { false }
                } else if p == "read_term/2" {
                    let options = self.get_reg_raw("A2");
                    if self.execute_read_term_builtin(options.as_ref()) {
                        self.pc = self.cp;
                        true
                    } else { false }
                } else if p == "atomic/1" {
                    if self.execute_builtin(p, 1) {
                        self.pc = self.cp;
                        true
                    } else { false }
                } else if self.foreign_predicates.contains(p) {
                    self.execute_foreign_predicate(p, 0)
                } else if let Some(__ftr) = crate::fact_table_call(self, p, self.cp) {
                    // T9 fact table in tail position: continuation is the saved
                    // cp (the caller resumes there when the fact pred succeeds).
                    __ftr
                } else if self.dynamic_call(p, self.cp) {
                    true
                } else if Self::is_iso_meta_builtin(p) {
                    // Tail-position ISO meta-builtin: dispatch, then honor
                    // return semantics by jumping to the continuation.
                    let arity: usize = p.rsplit(\'/\').next()
                        .and_then(|a| a.parse().ok()).unwrap_or(0);
                    if self.execute_builtin(p, arity) {
                        self.pc = self.cp;
                        true
                    } else { false }
                } else { false }'.

wam_instruction_arm('Instruction::ExecutePc(target_pc)', Body) :-
    Body = '                self.pc = *target_pc;
                true'.

wam_instruction_arm('Instruction::Proceed', Body) :-
    Body = '                let ret = self.cp;
                self.cp = 0; // halt
                self.pc = ret;
                true'.

wam_instruction_arm('Instruction::Jump(label)', Body) :-
    Body = '                if let Some(&target_pc) = self.labels.get(label) {
                    self.pc = target_pc;
                    true
                } else { false }'.

wam_instruction_arm('Instruction::NoOp', Body) :-
    Body = '                self.pc += 1; true'.

% M144: if-then-else soft cut. GetLevel snapshots the choice-point
% depth into a (permanent) register before the condition's try_me_else;
% CutTo restores it at the commit point, removing the ITE guard CP plus
% any CPs the condition pushed - regardless of how many that was.
wam_instruction_arm('Instruction::GetLevel(yn)', Body) :-
    Body = '                let depth = Value::Integer(self.choice_points.len() as i64);
                self.trail_binding(yn);
                self.put_reg(yn, depth);
                self.pc += 1;
                true'.

wam_instruction_arm('Instruction::CutTo(yn)', Body) :-
    Body = '                // get_reg (not get_reg_raw): Y registers live in the
                // topmost env frame, where GetLevel stored the depth.
                if let Some(v) = self.get_reg(yn) {
                    if let Value::Integer(depth) = self.deref_var(&v) {
                        self.choice_points.truncate(depth as usize);
                    }
                }
                self.pc += 1;
                true'.

wam_instruction_arm('Instruction::CutIte', Body) :-
    Body = '                self.choice_points.pop();
                self.pc += 1;
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

% T7 route 2: parallel aggregate embedded in a larger clause body. Drives the
% __par_enum/__par_body helper predicates by their entry-PC labels, reduces by
% agg_type (mirrors the aggregate_frame finalisation), binds result_reg in
% place, and advances to the next instruction (no choice point left behind, so
% the surrounding clause keeps running normally).
wam_instruction_arm('Instruction::ParAggregate(agg_type, enum_label, body_label, result_reg, input_regs)', Body) :-
    Body = '                let __base = self.clone();
                // Capture the external-input values from the container''s registers
                // (Y-aware, fully dereferenced) so the helpers run with them bound.
                let __ivals: Vec<Value> = input_regs.iter().map(|__r| {
                    let __raw = self.get_reg(__r).unwrap_or(Value::Unbound(__r.clone()));
                    self.deref_var(&self.deref_heap(&__raw))
                }).collect();
                let __vals = crate::par_aggregate::par_collect_labels(&__base, enum_label, body_label, &__ivals);
                let __result = match agg_type.as_str() {
                    "count" => Value::Integer(__vals.len() as i64),
                    "sum" => {
                        let mut sum_i: i64 = 0;
                        let mut sum_f: f64 = 0.0;
                        let mut saw_float = false;
                        for val in &__vals {
                            match self.deref_var(&self.deref_heap(val)) {
                                Value::Integer(n) => { sum_i += n; sum_f += n as f64; }
                                Value::Float(f) => { saw_float = true; sum_f += f; }
                                _ => {}
                            }
                        }
                        if saw_float { Value::Float(sum_f) } else { Value::Integer(sum_i) }
                    }
                    "max" => {
                        let mut best: Option<Value> = None;
                        for val in &__vals {
                            let current = self.deref_var(&self.deref_heap(val));
                            best = match best {
                                None => Some(current),
                                Some(ref prev) => match (&current, prev) {
                                    (Value::Integer(a), Value::Integer(b)) if a > b => Some(current),
                                    (Value::Float(a), Value::Float(b)) if a > b => Some(current),
                                    (Value::Integer(a), Value::Float(b)) if (*a as f64) > *b => Some(current),
                                    (Value::Float(a), Value::Integer(b)) if *a > (*b as f64) => Some(current),
                                    _ => Some(prev.clone()),
                                },
                            };
                        }
                        best.unwrap_or(Value::List(vec![]))
                    }
                    "min" => {
                        let mut best: Option<Value> = None;
                        for val in &__vals {
                            let current = self.deref_var(&self.deref_heap(val));
                            best = match best {
                                None => Some(current),
                                Some(ref prev) => match (&current, prev) {
                                    (Value::Integer(a), Value::Integer(b)) if a < b => Some(current),
                                    (Value::Float(a), Value::Float(b)) if a < b => Some(current),
                                    (Value::Integer(a), Value::Float(b)) if (*a as f64) < *b => Some(current),
                                    (Value::Float(a), Value::Integer(b)) if *a < (*b as f64) => Some(current),
                                    _ => Some(prev.clone()),
                                },
                            };
                        }
                        best.unwrap_or(Value::List(vec![]))
                    }
                    _ => Value::List(__vals),
                };
                // Bind through the Y-aware accessors (get_reg/put_reg): an
                // embedded aggregate''s result register is a permanent (Y)
                // variable in the environment frame, so get_reg_raw/set_reg_str
                // would miss it. Mirrors the aggregate_frame finalisation.
                let __lhs = self.get_reg(result_reg);
                let __ok = match __lhs {
                    Some(Value::Unbound(ref __vn)) => {
                        self.trail_binding(result_reg);
                        self.put_reg(result_reg, __result.clone());
                        self.bind_var(__vn, __result);
                        true
                    }
                    Some(__existing) => __existing == __result,
                    None => { self.put_reg(result_reg, __result); true }
                };
                if __ok { self.pc += 1; }
                __ok'.

% --- Choice Point Instructions ---

wam_instruction_arm('Instruction::TryMeElse(label)', Body) :-
    Body = '                if let Some(&next_pc) = self.labels.get(label) {
                    let clause_barrier = self.choice_points.len();
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
                    if label.starts_with("L_ite_else_") {
                        self.pending_cut_barrier = None;
                    } else {
                        self.pending_cut_barrier = Some(clause_barrier);
                    }
                    self.pc += 1; true
                } else { false }'.

wam_instruction_arm('Instruction::TryMeElsePc(next_pc)', Body) :-
    Body = '                let clause_barrier = self.choice_points.len();
                self.choice_points.push(ChoicePoint {
                    next_pc: *next_pc,
                    saved_args: self.save_regs(),
                    cp: self.cp,
                    stack: self.stack.clone(),
                    trail_len: self.trail.len(),
                    heap_len: self.heap.len(),
                    builtin_state: None,
                    cut_barrier: self.cut_barrier,
                });
                self.pending_cut_barrier = Some(clause_barrier);
                self.pc += 1; true'.

wam_instruction_arm('Instruction::TrustMe', Body) :-
    Body = '                self.choice_points.pop();
                self.pc += 1; true'.

wam_instruction_arm('Instruction::RetryMeElse(label)', Body) :-
    Body = '                if let Some(&next_pc) = self.labels.get(label) {
                    if let Some(cp) = self.choice_points.last_mut() {
                        cp.next_pc = next_pc;
                    }
                    self.pending_cut_barrier = Some(self.choice_points.len().saturating_sub(1));
                    self.pc += 1; true
                } else { false }'.

wam_instruction_arm('Instruction::RetryMeElsePc(next_pc)', Body) :-
    Body = '                if let Some(cp) = self.choice_points.last_mut() {
                    cp.next_pc = *next_pc;
                }
                self.pending_cut_barrier = Some(self.choice_points.len().saturating_sub(1));
                self.pc += 1; true'.

% --- Indexing Instructions ---

wam_instruction_arm('Instruction::SwitchOnConstant(table)', Body) :-
    Body = '                let raw = self.get_reg_raw("A1").map(|v| self.deref_var(&v));
                if let Some(val) = raw {
                    if !val.is_unbound() {
                        // Linear scan by value equality. (A binary search would
                        // require the table to be sorted by the search key; it is
                        // emitted in clause order — e.g. k0,k1,..,k9,k10 — which
                        // is NOT lexicographically sorted, so binary search
                        // dropped keys whose clause order != lexical order.)
                        let target: Option<&str> =
                            table.iter().find(|(k, _)| *k == val).map(|(_, l)| l.as_str());
                        match target {
                            // A key with multiple clauses maps to the "default"
                            // sentinel: fall through to the try_me_else chain that
                            // immediately follows the switch. (Treating it as a real
                            // label silently failed, dropping every such clause.)
                            Some("default") => { self.pc += 1; return true; }
                            Some(label) => {
                                if let Some(&pc) = self.labels.get(label) {
                                    self.pc = pc; return true;
                                }
                                return false;
                            }
                            None => return false, // no clause indexes this key
                        }
                    }
                }
                // Unbound A1: skip dispatch, advance to next instruction
                self.pc += 1; true'.

wam_instruction_arm('Instruction::SwitchOnConstantFallthrough(table)', Body) :-
    Body = '                let raw = self.get_reg_raw("A1").map(|v| self.deref_var(&v));
                if let Some(val) = raw {
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
                // No table key matched (or A1 unbound): fall through to the
                // next instruction (the try_me_else clause chain). Unlike
                // SwitchOnConstant this never fails — failing here skipped the
                // chain entirely and, because the dropped instruction also
                // shifted every later label by one, backtracking landed past
                // retry_me_else and looped.
                self.pc += 1; true'.

wam_instruction_arm('Instruction::SwitchOnConstantPc(table)', Body) :-
    Body = '                let raw = self.get_reg_raw("A1").map(|v| self.deref_var(&v));
                if let Some(val) = raw {
                    if !val.is_unbound() {
                        // Linear scan by value equality (see SwitchOnConstant:
                        // the table is in clause order, not lexically sorted, so a
                        // binary search dropped keys).
                        for (key, target_pc) in table {
                            if *key == val {
                                self.pc = *target_pc;
                                return true;
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
    compile_execute_ext_builtin_to_rust(ExtBuiltinCode),
    compile_execute_meta_builtin_to_rust(MetaBuiltinCode),
    compile_execute_foreign_predicate_to_rust(ForeignCode),
    compile_resume_builtin_to_rust(ResumeCode),
    compile_fact_table_attempt_to_rust(FactTableAttemptCode),
    compile_foreign_result_helpers_to_rust(ForeignResultHelpersCode),
    compile_parse_foreign_tuple_layout_to_rust(ForeignTupleLayoutCode),
    compile_collect_native_category_ancestor_to_rust(NativeAncestorCode),
    compile_collect_native_bidirectional_ancestor_to_rust(NativeBidirectionalCode),
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
        ExtBuiltinCode, '\n\n',
        MetaBuiltinCode, '\n\n',
        ForeignCode, '\n\n',
        ResumeCode, '\n\n',
        FactTableAttemptCode, '\n\n',
        ForeignResultHelpersCode, '\n\n',
        ForeignTupleLayoutCode, '\n\n',
        NativeAncestorCode, '\n\n',
        NativeBidirectionalCode, '\n\n',
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
                    // ISO throw in flight: abort instead of backtracking —
                    // alternatives are discarded until a catch/3 frame
                    // (in a caller) consumes the ball.
                    if self.thrown_ball.is_some() { return false; }
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
            self.pending_cut_barrier = None;

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
        if self.execute_ext_builtin(op, arity) { return true; }
        if self.execute_meta_builtin(op, arity) { return true; }

        match op {
            "true/0" => { self.pc += 1; true }
            "fail/0" => false,
            "!/0" => { self.choice_points.truncate(self.cut_barrier); self.pc += 1; true }
            "=/2" => {
                // Unification: the compiler emits `X = Y` as builtin_call =/2.
                // Without a handler it fell through to `_ => false`, so even
                // `a = a` failed.
                let a1 = self.get_reg_raw("A1").unwrap_or(Value::Uninit);
                let a2 = self.get_reg_raw("A2").unwrap_or(Value::Uninit);
                if self.unify(&a1, &a2) { self.pc += 1; true } else { false }
            }
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
                let v1 = self.get_reg_raw("A1")
                    .map(|v| self.deref_heap(&self.deref_var(&v)));
                let v2 = self.get_reg_raw("A2")
                    .map(|v| self.deref_heap(&self.deref_var(&v)));
                if v1 == v2 { self.pc += 1; true } else { false }
            }
            _ => false,
        }
    }'.

compile_execute_io_builtin_to_rust(Code) :-
    Code = '    fn builtin_path_arg(&self, reg: &str) -> Option<String> {
        match self.get_reg_raw(reg)
            .map(|v| self.deref_heap(&self.deref_var(&v))) {
            Some(Value::Atom(path)) => Some(path),
            _ => None,
        }
    }

    fn execute_io_builtin(&mut self, op: &str, _arity: usize) -> bool {
        match op {
            "write/1" | "display/1" | "print/1" => {
                // These share Display rendering for now. Standard Prolog
                // differentiates write/1, display/1, and print/1.
                if let Some(val) = self.get_reg_raw("A1") {
                    let derefed = self.deref_heap(&val);
                    print!("{}", derefed);
                    self.pc += 1; true
                } else { false }
            }
            "write_canonical/1" => {
                if let Some(val) = self.get_reg_raw("A1") {
                    let rendered = self.term_to_atom_text(&val);
                    let mut stdout = std::io::stdout().lock();
                    if std::io::Write::write_all(&mut stdout, rendered.as_bytes()).is_err() {
                        return false;
                    }
                    self.pc += 1; true
                } else { false }
            }
            "writeln/1" => {
                if let Some(val) = self.get_reg_raw("A1") {
                    let derefed = self.deref_heap(&val);
                    println!("{}", derefed);
                    self.pc += 1; true
                } else { false }
            }
            "tab/1" => {
                let mut remaining = match self.get_reg_raw("A1")
                    .map(|v| self.deref_heap(&self.deref_var(&v))) {
                    Some(Value::Integer(n)) if n >= 0 => match usize::try_from(n) {
                        Ok(count) => count,
                        Err(_) => return false,
                    },
                    _ => return false,
                };
                let spaces = [b'' ''; 64];
                let mut stdout = std::io::stdout().lock();
                while remaining > 0 {
                    let count = remaining.min(spaces.len());
                    if std::io::Write::write_all(&mut stdout, &spaces[..count]).is_err() {
                        return false;
                    }
                    remaining -= count;
                }
                self.pc += 1; true
            }
            "put_char/1" => {
                let text = match self.get_reg_raw("A1")
                    .map(|v| self.deref_heap(&self.deref_var(&v))) {
                    Some(Value::Atom(text)) => text,
                    _ => return false,
                };
                let mut chars = text.chars();
                let ch = match (chars.next(), chars.next()) {
                    (Some(ch), None) => ch,
                    _ => return false,
                };
                let mut encoded = [0u8; 4];
                let bytes = ch.encode_utf8(&mut encoded).as_bytes();
                let mut stdout = std::io::stdout().lock();
                if std::io::Write::write_all(&mut stdout, bytes).is_err() {
                    return false;
                }
                self.pc += 1; true
            }
            "put_code/1" => {
                let ch = match self.get_reg_raw("A1")
                    .map(|v| self.deref_heap(&self.deref_var(&v))) {
                    Some(Value::Integer(code)) if code >= 0 => {
                        match u32::try_from(code).ok().and_then(char::from_u32) {
                            Some(ch) => ch,
                            None => return false,
                        }
                    }
                    _ => return false,
                };
                let mut encoded = [0u8; 4];
                let bytes = ch.encode_utf8(&mut encoded).as_bytes();
                let mut stdout = std::io::stdout().lock();
                if std::io::Write::write_all(&mut stdout, bytes).is_err() {
                    return false;
                }
                self.pc += 1; true
            }
            "file_base_name/2" | "file_directory_name/2" => {
                let path = match self.builtin_path_arg("A1") {
                    Some(path) => path,
                    None => return false,
                };
                let component = if op == "file_base_name/2" {
                    match path.rfind("/") {
                        Some(index) => path[index + 1..].to_string(),
                        None => path,
                    }
                } else {
                    match path.rfind("/") {
                        Some(0) => "/".to_string(),
                        Some(index) => path[..index].to_string(),
                        None => ".".to_string(),
                    }
                };
                let output = self.get_reg_raw("A2").unwrap_or(Value::Uninit);
                let mark = self.trail.len();
                if self.unify(&output, &Value::Atom(component)) {
                    self.pc += 1; true
                } else {
                    self.unwind_trail_to(mark);
                    false
                }
            }
            "file_name_extension/3" => {
                let file_value = self.get_reg_raw("A3")
                    .map(|v| self.deref_heap(&self.deref_var(&v)))
                    .unwrap_or(Value::Uninit);
                if let Value::Atom(file) = file_value {
                    let basename_start = file.rfind("/")
                        .map(|index| index + 1)
                        .unwrap_or(0);
                    let extension_dot = file[basename_start..]
                        .rfind(".")
                        .map(|index| basename_start + index)
                        .filter(|index| *index > basename_start);
                    let (base, extension) = match extension_dot {
                        Some(index) => (
                            file[..index].to_string(),
                            file[index + 1..].to_string(),
                        ),
                        None => (file, String::new()),
                    };
                    let base_output = self.get_reg_raw("A1").unwrap_or(Value::Uninit);
                    let extension_output = self.get_reg_raw("A2").unwrap_or(Value::Uninit);
                    let mark = self.trail.len();
                    if !self.unify(&base_output, &Value::Atom(base)) {
                        self.unwind_trail_to(mark);
                        return false;
                    }
                    if self.unify(&extension_output, &Value::Atom(extension)) {
                        self.pc += 1; true
                    } else {
                        self.unwind_trail_to(mark);
                        false
                    }
                } else {
                    let base = match self.builtin_path_arg("A1") {
                        Some(base) => base,
                        None => return false,
                    };
                    let extension = match self.builtin_path_arg("A2") {
                        Some(extension) => extension,
                        None => return false,
                    };
                    let file = if extension.is_empty() {
                        base
                    } else {
                        format!("{}.{}", base, extension)
                    };
                    let output = self.get_reg_raw("A3").unwrap_or(Value::Uninit);
                    let mark = self.trail.len();
                    if self.unify(&output, &Value::Atom(file)) {
                        self.pc += 1; true
                    } else {
                        self.unwind_trail_to(mark);
                        false
                    }
                }
            }
            "is_absolute_file_name/1" => {
                let path = match self.builtin_path_arg("A1") {
                    Some(path) => path,
                    None => return false,
                };
                if path.starts_with("/") { self.pc += 1; true } else { false }
            }
            "path_join/3" => {
                let base = match self.builtin_path_arg("A1") {
                    Some(base) => base,
                    None => return false,
                };
                let relative = match self.builtin_path_arg("A2") {
                    Some(relative) => relative,
                    None => return false,
                };
                let full = if relative.starts_with("/") || base.is_empty() {
                    relative
                } else if relative.is_empty() {
                    base
                } else if base.ends_with("/") {
                    format!("{}{}", base, relative)
                } else {
                    format!("{}/{}", base, relative)
                };
                let output = self.get_reg_raw("A3").unwrap_or(Value::Uninit);
                let mark = self.trail.len();
                if self.unify(&output, &Value::Atom(full)) {
                    self.pc += 1; true
                } else {
                    self.unwind_trail_to(mark);
                    false
                }
            }
            "realpath/2" | "read_link/2" => {
                let path = match self.builtin_path_arg("A1") {
                    Some(path) => path,
                    None => return false,
                };
                let resolved = if op == "realpath/2" {
                    std::fs::canonicalize(path)
                } else {
                    std::fs::read_link(path)
                };
                let resolved = match resolved
                    .ok()
                    .and_then(|path| path.into_os_string().into_string().ok()) {
                    Some(path) => path,
                    None => return false,
                };
                let output = self.get_reg_raw("A2").unwrap_or(Value::Uninit);
                let mark = self.trail.len();
                if self.unify(&output, &Value::Atom(resolved)) {
                    self.pc += 1; true
                } else {
                    self.unwind_trail_to(mark);
                    false
                }
            }
            "same_file/2" => {
                let first = match self.builtin_path_arg("A1") {
                    Some(path) => path,
                    None => return false,
                };
                let second = match self.builtin_path_arg("A2") {
                    Some(path) => path,
                    None => return false,
                };
                let same = {
                    #[cfg(unix)]
                    {
                        use std::os::unix::fs::MetadataExt;
                        match (std::fs::metadata(first), std::fs::metadata(second)) {
                            (Ok(a), Ok(b)) => a.dev() == b.dev() && a.ino() == b.ino(),
                            _ => false,
                        }
                    }
                    #[cfg(not(unix))]
                    {
                        match (std::fs::canonicalize(first), std::fs::canonicalize(second)) {
                            (Ok(a), Ok(b)) => a == b,
                            _ => false,
                        }
                    }
                };
                if same { self.pc += 1; true } else { false }
            }
            "make_directory/1" | "delete_file/1" | "delete_directory/1" => {
                let path = match self.builtin_path_arg("A1") {
                    Some(path) => path,
                    None => return false,
                };
                let result = match op {
                    "make_directory/1" => std::fs::create_dir(path),
                    "delete_file/1" => std::fs::remove_file(path),
                    _ => std::fs::remove_dir(path),
                };
                if result.is_ok() { self.pc += 1; true } else { false }
            }
            "copy_file/2" | "rename_file/2" => {
                let source = match self.builtin_path_arg("A1") {
                    Some(path) => path,
                    None => return false,
                };
                let destination = match self.builtin_path_arg("A2") {
                    Some(path) => path,
                    None => return false,
                };
                let result = if op == "copy_file/2" {
                    std::fs::copy(source, destination).map(|_| ())
                } else {
                    std::fs::rename(source, destination)
                };
                if result.is_ok() { self.pc += 1; true } else { false }
            }
            "read_file_to_atom/2" => {
                let path = match self.builtin_path_arg("A1") {
                    Some(path) => path,
                    None => return false,
                };
                let content = match std::fs::read_to_string(path) {
                    Ok(content) => content,
                    Err(_) => return false,
                };
                let output = self.get_reg_raw("A2").unwrap_or(Value::Uninit);
                let mark = self.trail.len();
                if self.unify(&output, &Value::Atom(content)) {
                    self.pc += 1; true
                } else {
                    self.unwind_trail_to(mark);
                    false
                }
            }
            "write_atom_to_file/2" | "append_atom_to_file/2" => {
                let path = match self.builtin_path_arg("A1") {
                    Some(path) => path,
                    None => return false,
                };
                let content = match self.get_reg_raw("A2")
                    .map(|v| self.deref_heap(&self.deref_var(&v))) {
                    Some(Value::Atom(content)) => content,
                    _ => return false,
                };
                let result = if op == "write_atom_to_file/2" {
                    std::fs::write(path, content.as_bytes())
                } else {
                    std::fs::OpenOptions::new()
                        .create(true)
                        .append(true)
                        .open(path)
                        .and_then(|mut file| {
                            std::io::Write::write_all(&mut file, content.as_bytes())
                        })
                };
                if result.is_ok() { self.pc += 1; true } else { false }
            }
            "exists_file/1" => {
                let path = match self.builtin_path_arg("A1") {
                    Some(path) => path,
                    None => return false,
                };
                match std::fs::metadata(path) {
                    Ok(metadata) if metadata.is_file() => { self.pc += 1; true }
                    _ => false,
                }
            }
            "exists_directory/1" => {
                let path = match self.builtin_path_arg("A1") {
                    Some(path) => path,
                    None => return false,
                };
                match std::fs::metadata(path) {
                    Ok(metadata) if metadata.is_dir() => { self.pc += 1; true }
                    _ => false,
                }
            }
            "directory_files/2" => {
                let path = match self.builtin_path_arg("A1") {
                    Some(path) => path,
                    None => return false,
                };
                let entries = match std::fs::read_dir(path) {
                    Ok(entries) => entries,
                    Err(_) => return false,
                };
                let mut names = Vec::new();
                for entry in entries {
                    let entry = match entry {
                        Ok(entry) => entry,
                        Err(_) => return false,
                    };
                    let name = match entry.file_name().into_string() {
                        Ok(name) => name,
                        Err(_) => return false,
                    };
                    names.push(name);
                }
                names.sort_unstable();
                let mut files = Vec::with_capacity(names.len() + 2);
                files.push(Value::Atom(".".to_string()));
                files.push(Value::Atom("..".to_string()));
                files.extend(names.into_iter().map(Value::Atom));

                let output = self.get_reg_raw("A2").unwrap_or(Value::Uninit);
                let mark = self.trail.len();
                if self.unify(&output, &Value::List(files)) {
                    self.pc += 1; true
                } else {
                    self.unwind_trail_to(mark);
                    false
                }
            }
            "size_file/2" => {
                let path = match self.builtin_path_arg("A1") {
                    Some(path) => path,
                    None => return false,
                };
                let size = match std::fs::metadata(path)
                    .ok()
                    .and_then(|metadata| i64::try_from(metadata.len()).ok()) {
                    Some(size) => size,
                    None => return false,
                };
                let output = self.get_reg_raw("A2").unwrap_or(Value::Uninit);
                let mark = self.trail.len();
                if self.unify(&output, &Value::Integer(size)) {
                    self.pc += 1; true
                } else {
                    self.unwind_trail_to(mark);
                    false
                }
            }
            "time_file/2" => {
                let path = match self.builtin_path_arg("A1") {
                    Some(path) => path,
                    None => return false,
                };
                let modified = match std::fs::metadata(path)
                    .and_then(|metadata| metadata.modified()) {
                    Ok(modified) => modified,
                    Err(_) => return false,
                };
                let seconds = match modified.duration_since(std::time::UNIX_EPOCH) {
                    Ok(duration) => duration.as_secs_f64(),
                    Err(error) => -error.duration().as_secs_f64(),
                };
                let output = self.get_reg_raw("A2").unwrap_or(Value::Uninit);
                let mark = self.trail.len();
                if self.unify(&output, &Value::Float(seconds)) {
                    self.pc += 1; true
                } else {
                    self.unwind_trail_to(mark);
                    false
                }
            }
            "get_time/1" => {
                let now = std::time::SystemTime::now();
                let seconds = match now.duration_since(std::time::UNIX_EPOCH) {
                    Ok(duration) => duration.as_secs_f64(),
                    Err(error) => -error.duration().as_secs_f64(),
                };
                let output = self.get_reg_raw("A1").unwrap_or(Value::Uninit);
                let mark = self.trail.len();
                if self.unify(&output, &Value::Float(seconds)) {
                    self.pc += 1; true
                } else {
                    self.unwind_trail_to(mark);
                    false
                }
            }
            "getenv/2" => {
                let name = match self.get_reg_raw("A1")
                    .map(|v| self.deref_heap(&self.deref_var(&v))) {
                    Some(Value::Atom(name)) => name,
                    _ => return false,
                };
                let value = match std::env::var(name) {
                    Ok(value) => value,
                    Err(_) => return false,
                };
                let output = self.get_reg_raw("A2").unwrap_or(Value::Uninit);
                let mark = self.trail.len();
                if self.unify(&output, &Value::Atom(value)) {
                    self.pc += 1; true
                } else {
                    self.unwind_trail_to(mark);
                    false
                }
            }
            "getpid/1" => {
                let pid = Value::Integer(i64::from(std::process::id()));
                let output = self.get_reg_raw("A1").unwrap_or(Value::Uninit);
                let mark = self.trail.len();
                if self.unify(&output, &pid) {
                    self.pc += 1; true
                } else {
                    self.unwind_trail_to(mark);
                    false
                }
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
                "atomic/1" => {
                    let derefed = self.deref_heap(&self.deref_var(&val));
                    matches!(&derefed,
                        Value::Atom(_) | Value::Integer(_) | Value::Float(_) | Value::Bool(_))
                        || matches!(&derefed, Value::List(items) if items.is_empty())
                }
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
                let a1 = self.get_reg_raw("A1").unwrap_or(Value::Uninit);
                let a2 = self.get_reg_raw("A2").unwrap_or(Value::Uninit);
                let a3 = self.get_reg_raw("A3").unwrap_or(Value::Uninit);
                match (self.value_as_list(&a1), self.value_as_list(&a2)) {
                    (Some(mut left), Some(right)) => {
                        left.extend(right);
                        if self.unify(&a3, &Value::List(left)) { self.pc += 1; true }
                        else { false }
                    }
                    _ => false,
                }
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
            "reverse/2" => { self.execute_reverse_builtin() }
            "atom_codes/2" | "string_codes/2" => { self.execute_atom_codes_builtin() }
            "number_codes/2" => { self.execute_number_codes_builtin() }
            "number_chars/2" => { self.execute_number_chars_builtin() }
            "atom_to_term/3" => { self.execute_atom_to_term_builtin() }
            "term_to_atom/2" => { self.execute_term_to_atom_builtin() }
            "read_term_from_atom/2" => { self.execute_read_term_from_atom_builtin(2) }
            "read_term_from_atom/3" => { self.execute_read_term_from_atom_builtin(3) }
            "read_term/2" => {
                let options = self.get_reg_raw("A2");
                self.execute_read_term_builtin(options.as_ref())
            }
            "read/1" | "read_term/1" => { self.execute_read_term_builtin(None) }
            "assert/1" => { self.execute_assert_builtin("assert/1") }
            "assertz/1" | "asserta/1" => { self.execute_assert_builtin(op) }
            "retractall/1" => { self.execute_retractall_builtin() }
            "predicate_property/2" => { self.execute_predicate_property_builtin() }
            "term_variables/2" => {
                let term = self.get_reg_raw("A1").unwrap_or(Value::Uninit);
                let variables = self.variables_from_term(&term);
                let output = self.get_reg_raw("A2").unwrap_or(Value::Uninit);
                if self.unify(&output, &variables) { self.pc += 1; true }
                else { false }
            }
            "numbervars/3" => {
                let start = match self.get_reg_raw("A2")
                    .map(|v| self.deref_heap(&self.deref_var(&v))) {
                    Some(Value::Integer(n)) => n,
                    _ => return false,
                };
                let term = self.get_reg_raw("A1").unwrap_or(Value::Uninit);
                let variables = match self.variables_from_term(&term) {
                    Value::List(items) => items,
                    _ => return false,
                };
                let mark = self.trail.len();
                let mut next_number = start;
                for variable in variables {
                    let following = match next_number.checked_add(1) {
                        Some(n) => n,
                        None => { self.unwind_trail_to(mark); return false; }
                    };
                    let numbered = Value::Str(
                        "$VAR/1".to_string(),
                        vec![Value::Integer(next_number)],
                    );
                    if !self.unify(&variable, &numbered) {
                        self.unwind_trail_to(mark);
                        return false;
                    }
                    next_number = following;
                }
                let output = self.get_reg_raw("A3").unwrap_or(Value::Uninit);
                if self.unify(&output, &Value::Integer(next_number)) {
                    self.pc += 1; true
                } else {
                    self.unwind_trail_to(mark);
                    false
                }
            }
            "=@=/2" | "\\\\=@=/2" => {
                let left_raw = self.get_reg_raw("A1").unwrap_or(Value::Uninit);
                let right_raw = self.get_reg_raw("A2").unwrap_or(Value::Uninit);
                let left = self.deref_heap(&self.deref_var(&left_raw));
                let right = self.deref_heap(&self.deref_var(&right_raw));
                let mut left_vars = std::collections::HashMap::new();
                let mut right_vars = std::collections::HashMap::new();
                let equivalent = Self::variant_terms(
                    &left,
                    &right,
                    &mut left_vars,
                    &mut right_vars,
                );
                let succeeds = if op == "=@=/2" { equivalent } else { !equivalent };
                if succeeds { self.pc += 1; true } else { false }
            }
            "unifiable/3" => {
                let left = self.get_reg_raw("A1").unwrap_or(Value::Uninit);
                let right = self.get_reg_raw("A2").unwrap_or(Value::Uninit);
                let mark = self.trail.len();
                if !self.unify(&left, &right) {
                    self.unwind_trail_to(mark);
                    return false;
                }
                // Keep raw values so alias substitutions remain X=Y after
                // the trial bindings are unwound.
                let pairs: Vec<Value> = self.trail[mark..].iter()
                    .filter_map(|entry| {
                        let name = entry.key.strip_prefix("__binding__")?;
                        let bound = self.bindings.get(name)?.clone();
                        Some(Value::Str(
                            "=/2".to_string(),
                            vec![Value::Unbound(name.to_string()), bound],
                        ))
                    })
                    .collect();
                self.unwind_trail_to(mark);
                let output = self.get_reg_raw("A3").unwrap_or(Value::Uninit);
                if self.unify(&output, &Value::List(pairs)) {
                    self.pc += 1; true
                } else {
                    self.unwind_trail_to(mark);
                    false
                }
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
    }

    fn variant_terms(
        left: &Value,
        right: &Value,
        left_vars: &mut std::collections::HashMap<String, String>,
        right_vars: &mut std::collections::HashMap<String, String>,
    ) -> bool {
        match (left, right) {
            (Value::Unbound(a), Value::Unbound(b)) => {
                if let Some(mapped) = left_vars.get(a) {
                    return mapped == b && right_vars.get(b) == Some(a);
                }
                if right_vars.contains_key(b) { return false; }
                left_vars.insert(a.clone(), b.clone());
                right_vars.insert(b.clone(), a.clone());
                true
            }
            (Value::Atom(a), Value::Atom(b)) => a == b,
            (Value::Integer(a), Value::Integer(b)) => a == b,
            (Value::Float(a), Value::Float(b)) => a == b,
            (Value::Bool(a), Value::Bool(b)) => a == b,
            (Value::List(items), Value::Atom(atom))
            | (Value::Atom(atom), Value::List(items))
                if items.is_empty() && atom == "[]" => true,
            (Value::Str(f1, args1), Value::Str(f2, args2)) => {
                if f1 != f2 || args1.len() != args2.len() { return false; }
                for (a, b) in args1.iter().zip(args2.iter()) {
                    if !Self::variant_terms(a, b, left_vars, right_vars) {
                        return false;
                    }
                }
                true
            }
            (Value::List(items1), Value::List(items2)) => {
                if items1.len() != items2.len() { return false; }
                for (a, b) in items1.iter().zip(items2.iter()) {
                    if !Self::variant_terms(a, b, left_vars, right_vars) {
                        return false;
                    }
                }
                true
            }
            _ => false,
        }
    }

    fn execute_reverse_builtin(&mut self) -> bool {
        let src = self.get_reg_raw("A1").unwrap_or(Value::Uninit);
        let dst = self.get_reg_raw("A2").unwrap_or(Value::Uninit);
        match self.value_as_list(&src) {
            Some(mut items) => {
                items.reverse();
                if self.unify(&dst, &Value::List(items)) { self.pc += 1; true }
                else { false }
            }
            None => false,
        }
    }

    fn execute_atom_codes_builtin(&mut self) -> bool {
        let atom_raw = self.get_reg_raw("A1").unwrap_or(Value::Uninit);
        let codes_raw = self.get_reg_raw("A2").unwrap_or(Value::Uninit);
        let atom = self.deref_heap(&self.deref_var(&atom_raw));
        let codes = self.deref_heap(&self.deref_var(&codes_raw));
        match (atom, codes) {
            (Value::Atom(text), _) => {
                let list = Self::string_to_codes_value(&text);
                if self.unify(&codes_raw, &list) { self.pc += 1; true }
                else { false }
            }
            (Value::Unbound(_), Value::List(items)) => {
                match self.code_list_to_string(&items) {
                    Some(text) => {
                        if self.unify(&atom_raw, &Value::Atom(text)) { self.pc += 1; true }
                        else { false }
                    }
                    None => false,
                }
            }
            _ => false,
        }
    }

    fn execute_number_codes_builtin(&mut self) -> bool {
        let num_raw = self.get_reg_raw("A1").unwrap_or(Value::Uninit);
        let codes_raw = self.get_reg_raw("A2").unwrap_or(Value::Uninit);
        let num = self.deref_heap(&self.deref_var(&num_raw));
        let codes = self.deref_heap(&self.deref_var(&codes_raw));
        match (num, codes) {
            (Value::Integer(n), _) => {
                let list = Self::string_to_codes_value(&n.to_string());
                if self.unify(&codes_raw, &list) { self.pc += 1; true }
                else { false }
            }
            (Value::Float(f), _) => {
                let list = Self::string_to_codes_value(&f.to_string());
                if self.unify(&codes_raw, &list) { self.pc += 1; true }
                else { false }
            }
            (Value::Unbound(_), Value::List(items)) => {
                match self.code_list_to_string(&items) {
                    Some(text) => {
                        let parsed = if let Ok(n) = text.parse::<i64>() {
                            Some(Value::Integer(n))
                        } else if let Ok(f) = text.parse::<f64>() {
                            Some(Value::Float(f))
                        } else {
                            None
                        };
                        match parsed {
                            Some(v) => {
                                if self.unify(&num_raw, &v) { self.pc += 1; true }
                                else { false }
                            }
                            None => false,
                        }
                    }
                    None => false,
                }
            }
            _ => false,
        }
    }

    fn execute_number_chars_builtin(&mut self) -> bool {
        let num_raw = self.get_reg_raw("A1").unwrap_or(Value::Uninit);
        let chars_raw = self.get_reg_raw("A2").unwrap_or(Value::Uninit);
        let num = self.deref_heap(&self.deref_var(&num_raw));
        let chars = self.deref_heap(&self.deref_var(&chars_raw));
        match (num, chars) {
            (Value::Integer(n), _) => {
                let list = Value::List(
                    n.to_string()
                        .chars()
                        .map(|ch| Value::Atom(ch.to_string()))
                        .collect(),
                );
                if self.unify(&chars_raw, &list) { self.pc += 1; true }
                else { false }
            }
            (Value::Float(f), _) => {
                let list = Value::List(
                    f.to_string()
                        .chars()
                        .map(|ch| Value::Atom(ch.to_string()))
                        .collect(),
                );
                if self.unify(&chars_raw, &list) { self.pc += 1; true }
                else { false }
            }
            (Value::Unbound(_), Value::List(items)) => {
                let mut text = String::new();
                for item in &items {
                    match self.deref_heap(&self.deref_var(item)) {
                        Value::Atom(atom) if atom.chars().count() == 1 => {
                            text.push(atom.chars().next().unwrap());
                        }
                        _ => return false,
                    }
                }
                let parsed = if let Ok(n) = text.parse::<i64>() {
                    Some(Value::Integer(n))
                } else if let Ok(f) = text.parse::<f64>() {
                    Some(Value::Float(f))
                } else {
                    None
                };
                match parsed {
                    Some(value) => {
                        if self.unify(&num_raw, &value) { self.pc += 1; true }
                        else { false }
                    }
                    None => false,
                }
            }
            _ => false,
        }
    }

    fn execute_atom_to_term_builtin(&mut self) -> bool {
        let atom = self.get_reg_raw("A1")
            .map(|value| self.deref_heap(&self.deref_var(&value)));
        let bindings = self.get_reg_raw("A3").unwrap_or(Value::Uninit);
        let options = Value::List(vec![Value::Str(
            "variable_names".to_string(),
            vec![bindings],
        )]);
        match atom {
            Some(Value::Atom(text)) => {
                if self.bind_compiled_parse_atom_with_options(&text, "A2", Some(&options)) {
                    self.pc += 1;
                    true
                } else {
                    false
                }
            }
            _ => false,
        }
    }

    fn execute_term_to_atom_builtin(&mut self) -> bool {
        let term_raw = self.get_reg_raw("A1").unwrap_or(Value::Uninit);
        let atom_raw = self.get_reg_raw("A2").unwrap_or(Value::Uninit);
        let term = self.deref_heap(&self.deref_var(&term_raw));
        if matches!(term, Value::Unbound(_)) {
            let atom = self.deref_heap(&self.deref_var(&atom_raw));
            match atom {
                Value::Atom(text) => {
                    if self.bind_compiled_parse_atom(&text, "A1") { self.pc += 1; true }
                    else { false }
                }
                _ => false,
            }
        } else {
            let rendered = self.term_to_atom_text(&term);
            if self.unify(&atom_raw, &Value::Atom(rendered)) { self.pc += 1; true }
            else { false }
        }
    }

    fn execute_read_term_from_atom_builtin(&mut self, arity: usize) -> bool {
        let atom = self.get_reg_raw("A1")
            .map(|v| self.deref_heap(&self.deref_var(&v)));
        let options = if arity == 3 { self.get_reg_raw("A3") } else { None };
        match atom {
            Some(Value::Atom(text)) => {
                if self.bind_compiled_parse_atom_with_options(&text, "A2", options.as_ref()) {
                    self.pc += 1; true
                }
                else { false }
            }
            _ => false,
        }
    }

    fn bind_compiled_parse_atom(&mut self, atom_text: &str, target_reg: &str) -> bool {
        self.bind_compiled_parse_atom_with_options(atom_text, target_reg, None)
    }

    fn bind_compiled_parse_atom_with_options(
        &mut self,
        atom_text: &str,
        target_reg: &str,
        options: Option<&Value>,
    ) -> bool {
        let atom_text = Self::strip_term_comments(atom_text);
        let variable_names = options
            .and_then(|value| self.read_term_option_arg(value, "variable_names"));
        let variables = options
            .and_then(|value| self.read_term_option_arg(value, "variables"));
        let singletons = options
            .and_then(|value| self.read_term_option_arg(value, "singletons"));
        let syntax_errors_error = options
            .and_then(|value| self.read_term_option_atom(value, "syntax_errors"))
            .as_deref() == Some("error");
        let wants_env = variable_names.is_some() || variables.is_some() || singletons.is_some();
        let parser_entry = if wants_env {
            "parse_term_from_atom/4"
        } else {
            "parse_term_from_atom/3"
        };
        if !self.labels.contains_key("canonical_op_table/1") ||
           !self.labels.contains_key(parser_entry) {
            if syntax_errors_error {
                return self.raise_read_syntax_error();
            }
            return false;
        }
        let mut parser = WamState::new(self.code.clone(), self.labels.clone());

        let ops_var = Value::Unbound("_RP_ops".to_string());
        parser.set_reg_str("A1", ops_var.clone());
        if !parser.run_named_label("canonical_op_table/1") {
            if syntax_errors_error {
                return self.raise_read_syntax_error();
            }
            return false;
        }
        let ops = parser.deref_heap(&parser.deref_var(&ops_var));

        parser.reset_query();
        let parsed_var = Value::Unbound("_RP_term".to_string());
        parser.set_reg_str("A1", Value::Atom(atom_text));
        parser.set_reg_str("A2", ops);
        parser.set_reg_str("A3", parsed_var.clone());
        let var_env = Value::Unbound("_RP_env".to_string());
        if wants_env {
            parser.set_reg_str("A4", var_env.clone());
        }
        if !parser.run_named_label(parser_entry) {
            if syntax_errors_error {
                return self.raise_read_syntax_error();
            }
            return false;
        }

        let parsed = parser.deref_heap(&parser.deref_var(&parsed_var));
        let mut var_map: std::collections::HashMap<String, String> = std::collections::HashMap::new();
        let copied = self.copy_external_term_from(&parser, &parsed, &mut var_map);
        let target = match self.get_reg_raw(target_reg) {
            Some(target) => target,
            None => return false,
        };
        if !self.unify(&target, &copied) {
            return false;
        }
        if let Some(names_target) = variable_names {
            let names = self.copy_variable_names_from_env(&parser, &var_env, &mut var_map);
            if !self.unify(&names_target, &names) {
                return false;
            }
        }
        if let Some(variables_target) = variables {
            let variable_list = self.variables_from_term(&copied);
            if !self.unify(&variables_target, &variable_list) {
                return false;
            }
        }
        if let Some(singletons_target) = singletons {
            let singleton_list =
                self.singletons_from_term(&copied, &parser, &var_env, &mut var_map);
            if !self.unify(&singletons_target, &singleton_list) {
                return false;
            }
        }
        true
    }

    fn run_named_label(&mut self, label: &str) -> bool {
        match self.labels.get(label).copied() {
            Some(pc) => {
                self.pc = pc;
                self.cp = 0;
                self.step_count = 0;
                self.run()
            }
            None => false,
        }
    }

    fn copy_external_term_from(
        &mut self,
        source: &WamState,
        value: &Value,
        var_map: &mut std::collections::HashMap<String, String>,
    ) -> Value {
        let derefed_var = source.deref_var(value);
        let derefed = source.deref_heap(&derefed_var);
        match derefed {
            Value::Unbound(name) => {
                if let Some(new_name) = var_map.get(&name) {
                    Value::Unbound(new_name.clone())
                } else {
                    self.var_counter += 1;
                    let new_name = format!("_RP{}", self.var_counter);
                    var_map.insert(name, new_name.clone());
                    Value::Unbound(new_name)
                }
            }
            Value::Str(f, args) => {
                let copied_args: Vec<Value> = args.iter()
                    .map(|a| self.copy_external_term_from(source, a, var_map))
                    .collect();
                Value::Str(Self::display_functor_name(&f, copied_args.len()), copied_args)
            }
            Value::List(items) => {
                let copied_items: Vec<Value> = items.iter()
                    .map(|i| self.copy_external_term_from(source, i, var_map))
                    .collect();
                Value::List(copied_items)
            }
            other => other,
        }
    }

    fn value_as_list(&self, value: &Value) -> Option<Vec<Value>> {
        let derefed = self.deref_heap(&self.deref_var(value));
        match derefed {
            Value::List(items) => Some(items),
            Value::Atom(s) if s == "[]" => Some(Vec::new()),
            Value::Str(f, args) if self.is_cons_functor(&f) && args.len() == 2 => {
                let mut tail = self.value_as_list(&args[1])?;
                tail.insert(0, args[0].clone());
                Some(tail)
            }
            _ => None,
        }
    }

    fn pair_list_columns(
        &self,
        value: &Value,
        take_keys: bool,
        take_values: bool,
    ) -> Option<(Vec<Value>, Vec<Value>)> {
        let items = self.value_as_list(value)?;
        let mut keys = if take_keys { Vec::with_capacity(items.len()) } else { Vec::new() };
        let mut values = if take_values { Vec::with_capacity(items.len()) } else { Vec::new() };
        for item in &items {
            match self.deref_heap(&self.deref_var(item)) {
                Value::Str(functor, args)
                    if args.len() == 2
                        && Self::display_functor_name(&functor, 2) == "-" => {
                    if take_keys { keys.push(args[0].clone()); }
                    if take_values { values.push(args[1].clone()); }
                }
                _ => return None,
            }
        }
        Some((keys, values))
    }

    fn string_to_codes_value(text: &str) -> Value {
        Value::List(text.chars()
            .map(|c| Value::Integer(c as i64))
            .collect())
    }

    fn code_list_to_string(&self, items: &[Value]) -> Option<String> {
        let mut out = String::new();
        for item in items {
            let derefed = self.deref_heap(&self.deref_var(item));
            match derefed {
                Value::Integer(code) if code >= 0 => {
                    out.push(char::from_u32(code as u32)?);
                }
                _ => return None,
            }
        }
        Some(out)
    }

    fn term_to_atom_text(&self, value: &Value) -> String {
        let derefed = self.deref_heap(&self.deref_var(value));
        match derefed {
            Value::Atom(s) => Self::term_atom_text(&s),
            Value::Integer(n) => n.to_string(),
            Value::Float(f) => f.to_string(),
            Value::Bool(b) => b.to_string(),
            Value::Unbound(name) => name,
            Value::List(items) => {
                let rendered: Vec<String> = items.iter()
                    .map(|i| self.term_to_atom_text(i))
                    .collect();
                format!("[{}]", rendered.join(", "))
            }
            Value::Str(f, args) => {
                let name = Self::display_functor_name(&f, args.len());
                let rendered: Vec<String> = args.iter()
                    .map(|a| self.term_to_atom_text(a))
                    .collect();
                format!("{}({})", Self::term_atom_text(&name), rendered.join(", "))
            }
            Value::Ref(_) => format!("{}", derefed),
            Value::Uninit => "_".to_string(),
        }
    }

    fn display_functor_name(functor: &str, arity: usize) -> String {
        let mut name = functor
            .strip_prefix("str(")
            .and_then(|s| s.strip_suffix(")"))
            .unwrap_or(functor);
        if let Some((base, ar)) = name.rsplit_once(''/'') {
            if ar.parse::<usize>().ok() == Some(arity) {
                name = base;
            }
        }
        name.to_string()
    }

    '.


compile_dynamic_db_helpers_to_rust(Code) :-
    read_template_file('templates/targets/rust_wam/dynamic_db_methods.rs.mustache', Template),
    render_template(Template, [], Code).

compile_execute_foreign_predicate_to_rust(Code) :-
    Code = '    /// Execute a registered foreign predicate by name/arity.
    pub fn execute_foreign_predicate(&mut self, pred: &str, arity: usize) -> bool {
        let pred_key = if pred.contains(\'/\') && self.foreign_predicates.contains(pred) {
            pred.to_string()
        } else {
            format!("{}/{}", pred, arity)
        };
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
                let mut visited_ids: Vec<u32> = visited.iter().filter_map(|item| {
                    match self.deref_var(item) {
                        Value::Atom(s) => Some(self.intern_atom(&s)),
                        _ => None,
                    }
                }).collect();
                let mut hops: Vec<i64> = Vec::new();
                let acc = self.resolve_edge_accessor(&edge_pred);
                self.collect_native_category_ancestor_hops(cat_id, root_id, &mut visited_ids, max_depth, &acc, 0, &mut hops);
                if hops.is_empty() {
                    return false;
                }
                let results: Vec<Value> = hops.into_iter().map(|hop| {
                    Value::Str("__tuple__".to_string(), vec![Value::Integer(hop)])
                }).collect();
                self.finish_foreign_results(&pred_key, vec![hops_reg], results)
            }
            "bidirectional_ancestor" => {
                // 5-ary interface: A1 cat (in), A2 root (in), A3 total
                // hops, A4 parent hops, A5 child hops (out, streamed per
                // budget-feasible path). Costs and budget come from f64
                // configs with the F#-parity defaults 1.0 / 3.0 / 10.0.
                let cat = match self.get_reg_raw("A1").map(|v| self.deref_var(&v)) {
                    Some(Value::Atom(cat)) => cat,
                    _ => return false,
                };
                let root = match self.get_reg_raw("A2").map(|v| self.deref_var(&v)) {
                    Some(Value::Atom(root)) => root,
                    _ => return false,
                };
                let total_reg = match self.get_reg_raw("A3") {
                    Some(val) => val,
                    None => return false,
                };
                let phops_reg = match self.get_reg_raw("A4") {
                    Some(val) => val,
                    None => return false,
                };
                let chops_reg = match self.get_reg_raw("A5") {
                    Some(val) => val,
                    None => return false,
                };
                let edge_pred = match self.foreign_string_config(&pred_key, "edge_pred") {
                    Some(pred) => pred.to_string(),
                    None => return false,
                };
                let child_pred = self.foreign_string_config(&pred_key, "child_pred")
                    .unwrap_or("category_child")
                    .to_string();
                let parent_cost = self.foreign_f64_config(&pred_key, "parent_step_cost").unwrap_or(1.0);
                let child_cost = self.foreign_f64_config(&pred_key, "child_step_cost").unwrap_or(3.0);
                let budget = self.foreign_f64_config(&pred_key, "cost_budget").unwrap_or(10.0);
                let cat_id = self.intern_atom(&cat);
                let root_id = self.intern_atom(&root);
                // Derive the downward index from the parent edges when no
                // child-direction source is registered.
                self.ensure_reverse_edge_index(&edge_pred, &child_pred);
                let mut hops: Vec<(i64, i64, i64)> = Vec::new();
                {
                    let parents = self.resolve_edge_accessor(&edge_pred);
                    let children = self.resolve_edge_accessor(&child_pred);
                    let (_dim, _branch_ratio, _branch_ratio_raw, _routing_correction, min_dist) =
                        self.calibrate_bidirectional_graph(root_id, &parents, &children);
                    self.collect_native_bidirectional_ancestor_hops(
                        cat_id, root_id, parent_cost, child_cost, budget,
                        &parents, &children, &min_dist, &mut hops);
                }
                if hops.is_empty() {
                    return false;
                }
                let results: Vec<Value> = hops.into_iter().map(|(t, p, c)| {
                    Value::Str("__tuple__".to_string(), vec![
                        Value::Integer(t),
                        Value::Integer(p),
                        Value::Integer(c),
                    ])
                }).collect();
                self.finish_foreign_results(&pred_key, vec![total_reg, phops_reg, chops_reg], results)
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
                let distance_filter = match self.deref_var(&dist_reg) {
                    Value::Integer(d) if d > 0 => Some(d),
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
                if let Some(want_d) = distance_filter {
                    results.retain(|(_, d)| *d == want_d);
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
            "lazy_lmdb_lookup" => {
                // R7: lazy parent-edge lookup via the LookupSource trait.
                // A1 is the input atom key; A2 is the binding target.
                // The backend (e.g. LmdbFactSource) maps the atom to an
                // int, looks up parents, and maps each result back to
                // an atom. Mirrors Haskell EdgeLookup :: Int -> [Int]
                // but with the kernel staying in atom-space.
                let key_atom = match self.get_reg_raw("A1").map(|v| self.deref_var(&v)) {
                    Some(Value::Atom(s)) => s,
                    _ => return false,
                };
                let value_reg = match self.get_reg_raw("A2") {
                    Some(val) => val,
                    None => return false,
                };
                let source = match self.lazy_lookup(&pred_key) {
                    Some(src) => src,
                    None => return false,
                };
                let key_int = match source.lookup_key_for_atom(&key_atom) {
                    Some(k) => k,
                    None => return false,
                };
                let value_ints = source.lookup_parents(key_int);
                if value_ints.is_empty() {
                    return false;
                }
                let results: Vec<Value> = value_ints.iter().filter_map(|vid| {
                    source.atom_for_key(*vid).map(|s| {
                        Value::Str("__tuple__".to_string(), vec![Value::Atom(s)])
                    })
                }).collect();
                if results.is_empty() {
                    return false;
                }
                self.finish_foreign_results(&pred_key, vec![value_reg], results)
            }
            "category_ancestor_boundary" => {
                // P2c: the boundary-distribution optimization as a foreign kernel.
                // Runs the boundary-spliced ancestor walk (collect_native_category_
                // ancestor_boundary_hist) over the cached boundary side-table, then
                // reads the path-length measure into ONE result (deterministic mode)
                // per the configured extractor. With an empty side-table it degrades
                // to full enumeration (still correct). See
                // WAM_RUST_BOUNDARY_DISTRIBUTION_SPECIFICATION.md §5/§6.
                let cat = match self.get_reg_raw("A1").map(|v| self.deref_var(&v)) {
                    Some(Value::Atom(c)) => c,
                    _ => return false,
                };
                let root = match self.get_reg_raw("A2").map(|v| self.deref_var(&v)) {
                    Some(Value::Atom(r)) => r,
                    _ => return false,
                };
                let out_reg = match self.get_reg_raw("A3") {
                    Some(val) => val,
                    None => return false,
                };
                let max_depth = match self.foreign_usize_config(&pred_key, "max_depth") {
                    Some(limit) => limit,
                    None => return false,
                };
                let edge_pred = match self.foreign_string_config(&pred_key, "edge_pred") {
                    Some(p) => p.to_string(),
                    None => return false,
                };
                let n = self.foreign_f64_config(&pred_key, "weight_n").unwrap_or(2.0);
                let extractor = self.foreign_string_config(&pred_key, "result_extractor")
                    .map(|s| s.to_string()).unwrap_or_else(|| "scalar".to_string());
                let cat_id = self.intern_atom(&cat);
                let root_id = self.intern_atom(&root);
                let acc = self.resolve_edge_accessor(&edge_pred);
                let result: Value = if extractor == "shortest_distance" {
                    // min-plus distance read (graph-functional-semiring increment 2):
                    // the CYCLE-CORRECT shortest hop-distance to root via the distance
                    // cache (build_boundary_distances / boundary_dist), splice-pruned at
                    // cached boundaries. Degrades to a plain (correct) BFS with an empty
                    // cache, so it is safe without a precompute. Distinct from reading the
                    // histogram support: that DFS histogram is unsound on cycles.
                    match self.category_ancestor_boundary_distance(cat_id, root_id, &acc) {
                        Some(d) => Value::Integer(d as i64),
                        None => return false, // root unreachable
                    }
                } else {
                    let mut hist: Vec<u64> = Vec::new();
                    let mut visited: Vec<u32> = vec![cat_id];
                    self.collect_native_category_ancestor_boundary_hist(
                        cat_id, root_id, &mut visited, max_depth, &acc, 0, &mut hist);
                    // weighted_power read: WeightSum = sum_{L>0} H[L] * L^-n
                    let weight_sum = || -> f64 {
                        hist.iter().enumerate().filter(|(l, _)| *l > 0)
                            .map(|(l, &c)| c as f64 * (l as f64).powf(-n)).sum()
                    };
                    match extractor.as_str() {
                        "distribution" => Value::List(
                            hist.iter().map(|&c| Value::Integer(c as i64)).collect()),
                        "effective_distance" => {
                            let ws = weight_sum();
                            if ws > 0.0 { Value::Float(ws.powf(-1.0 / n)) } else { return false; }
                        }
                        // default: scalar(weighted_power)
                        _ => Value::Float(weight_sum()),
                    }
                };
                self.finish_foreign_results(
                    &pred_key, vec![out_reg],
                    vec![Value::Str("__tuple__".to_string(), vec![result])])
            }
            "category_bridge_score" => {
                // category_bridge_score(Node, Class): Node atom in (A1), Class atom out (A2). Builds
                // the bridge scores on first use from edge_pred + mu_pred + threshold (self-calibrating
                // tau_pure), then packs bridge_class_atom(class) as a Prolog atom. The mu map is a
                // Prolog fact predicate (mu_pred config, e.g. category_mu/2) loaded via register_ffi_mu.
                // See WAM_RUST_BRIDGE_CLUSTERING.md increment 1.
                let node = match self.get_reg_raw("A1").map(|v| self.deref_var(&v)) {
                    Some(Value::Atom(node)) => node,
                    _ => return false,
                };
                let class_reg = match self.get_reg_raw("A2") {
                    Some(val) => val,
                    None => return false,
                };
                let edge_pred = match self.foreign_string_config(&pred_key, "edge_pred") {
                    Some(pred) => pred.to_string(),
                    None => return false,
                };
                let mu_pred = match self.foreign_string_config(&pred_key, "mu_pred") {
                    Some(pred) => pred.to_string(),
                    None => return false,
                };
                let threshold = self.foreign_f64_config(&pred_key, "threshold").unwrap_or(0.3);
                if !self.ensure_bridge_scores(&edge_pred, &mu_pred, threshold) {
                    return false;
                }
                let node_id = self.intern_atom(&node);
                let (class, _n_eff) = match self.category_bridge_class(node_id) {
                    Some(pair) => pair,
                    None => return false,
                };
                self.finish_foreign_results(&pred_key, vec![class_reg], vec![
                    Value::Str("__tuple__".to_string(), vec![Value::Atom(class.to_string())])
                ])
            }
            "bridge" => {
                // bridge(Node, Class, Neff): enumerate all bridge candidates (ranking). Node atom (A1),
                // Class atom (A2), Neff float (A3) — streamed one solution per candidate, sorted by
                // n_eff descending. Same build-on-first-use as category_bridge_score.
                let node_reg = match self.get_reg_raw("A1") {
                    Some(val) => val,
                    None => return false,
                };
                let class_reg = match self.get_reg_raw("A2") {
                    Some(val) => val,
                    None => return false,
                };
                let neff_reg = match self.get_reg_raw("A3") {
                    Some(val) => val,
                    None => return false,
                };
                let edge_pred = match self.foreign_string_config(&pred_key, "edge_pred") {
                    Some(pred) => pred.to_string(),
                    None => return false,
                };
                let mu_pred = match self.foreign_string_config(&pred_key, "mu_pred") {
                    Some(pred) => pred.to_string(),
                    None => return false,
                };
                let threshold = self.foreign_f64_config(&pred_key, "threshold").unwrap_or(0.3);
                if !self.ensure_bridge_scores(&edge_pred, &mu_pred, threshold) {
                    return false;
                }
                let candidates = self.bridge_candidates();
                if candidates.is_empty() {
                    return false;
                }
                let results: Vec<Value> = candidates.into_iter().map(|(id, class, neff)| {
                    let name = self.atom_name(id).unwrap_or("").to_string();
                    Value::Str("__tuple__".to_string(), vec![
                        Value::Atom(name),
                        Value::Atom(class.to_string()),
                        Value::Float(neff),
                    ])
                }).collect();
                self.finish_foreign_results(&pred_key, vec![node_reg, class_reg, neff_reg], results)
            }
            "category_cluster" => {
                // category_cluster(Node, ClusterId): Node atom in (A1), cluster id integer out (A2).
                // Builds the clusters on first use from edge_pred + mu_pred + threshold (leak-removal
                // components + a bridge-split level). See WAM_RUST_BRIDGE_CLUSTERING.md increment 2.
                let node = match self.get_reg_raw("A1").map(|v| self.deref_var(&v)) {
                    Some(Value::Atom(node)) => node,
                    _ => return false,
                };
                let id_reg = match self.get_reg_raw("A2") {
                    Some(val) => val,
                    None => return false,
                };
                let edge_pred = match self.foreign_string_config(&pred_key, "edge_pred") {
                    Some(pred) => pred.to_string(),
                    None => return false,
                };
                let mu_pred = match self.foreign_string_config(&pred_key, "mu_pred") {
                    Some(pred) => pred.to_string(),
                    None => return false,
                };
                let threshold = self.foreign_f64_config(&pred_key, "threshold").unwrap_or(0.3);
                // Optional tau_pure config pins the leak/bridge purity cut; absent => self-calibrating.
                let tau_pure = self.foreign_f64_config(&pred_key, "tau_pure");
                if self.clusters.is_empty() {
                    let params = crate::boundary_cache::BridgeParams { phi_min: 2, tau_div: 1.5, tau_pure };
                    self.build_clusters_named(&edge_pred, &mu_pred, threshold, &params);
                }
                if self.clusters.is_empty() {
                    return false;
                }
                let node_id = self.intern_atom(&node);
                let cluster = match self.category_cluster(node_id) {
                    Some(c) => c,
                    None => return false,
                };
                self.finish_foreign_results(&pred_key, vec![id_reg], vec![
                    Value::Str("__tuple__".to_string(), vec![Value::Integer(cluster as i64)])
                ])
            }
            "cluster_members" => {
                // cluster_members(ClusterId, Member): cluster id integer in (A1), Member atom out (A2),
                // streamed one solution per member node. Same build-on-first-use as category_cluster.
                let cluster = match self.get_reg_raw("A1").map(|v| self.deref_var(&v)) {
                    Some(Value::Integer(c)) => c,
                    _ => return false,
                };
                let member_reg = match self.get_reg_raw("A2") {
                    Some(val) => val,
                    None => return false,
                };
                let edge_pred = match self.foreign_string_config(&pred_key, "edge_pred") {
                    Some(pred) => pred.to_string(),
                    None => return false,
                };
                let mu_pred = match self.foreign_string_config(&pred_key, "mu_pred") {
                    Some(pred) => pred.to_string(),
                    None => return false,
                };
                let threshold = self.foreign_f64_config(&pred_key, "threshold").unwrap_or(0.3);
                let tau_pure = self.foreign_f64_config(&pred_key, "tau_pure");
                if self.clusters.is_empty() {
                    let params = crate::boundary_cache::BridgeParams { phi_min: 2, tau_div: 1.5, tau_pure };
                    self.build_clusters_named(&edge_pred, &mu_pred, threshold, &params);
                }
                if self.clusters.is_empty() {
                    return false;
                }
                if cluster < 0 {
                    return false;
                }
                let members = self.cluster_members(cluster as u32);
                if members.is_empty() {
                    return false;
                }
                let results: Vec<Value> = members.into_iter().map(|id| {
                    let name = self.atom_name(id).unwrap_or("").to_string();
                    Value::Str("__tuple__".to_string(), vec![Value::Atom(name)])
                }).collect();
                self.finish_foreign_results(&pred_key, vec![member_reg], results)
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
        if let Some(inner) = layout.strip_prefix("tuple(").and_then(|s| s.strip_suffix(")")) {
            inner.parse::<usize>().ok()
        } else {
            layout.strip_prefix("tuple:")?.parse::<usize>().ok()
        }
    }'.

compile_collect_native_category_ancestor_to_rust(Code) :-
    Code = '    pub fn collect_native_category_ancestor_hops(
        &self,
        cat_id: u32,
        root_id: u32,
        visited: &mut Vec<u32>,
        max_depth: usize,
        acc: &EdgeAccessor,
        depth: i64,
        out: &mut Vec<i64>,
    ) {
        // ROOT_ANCHORED_METRICS admissible prune: when a materialised
        // min_dist_to_root table is loaded, a node whose shortest remaining
        // distance to root exceeds the remaining depth budget cannot reach
        // root within max_depth, so the whole branch is cut. A node absent
        // from the table is unreachable and likewise pruned. Disabled (empty
        // table) by default; never changes results, only avoids exploring
        // walks that provably cannot reach root in budget.
        if !self.min_dist.is_empty() {
            match self.min_dist.get(&(cat_id as i32)) {
                Some(&d) => {
                    if depth + (d as i64) > max_depth as i64 {
                        return;
                    }
                }
                None => return,
            }
        }
        // R7: route through self.edge_parents so lazy_lookups is
        // consulted before ffi_facts. This keeps the kernels_on
        // native path working under lmdb_materialisation(lazy).
        //
        // Accumulator-passing (ported from the Haskell kernel, perf-log
        // #16/#207): thread the hop count DOWN as `depth` and emit the
        // final distance (depth + 1) when the root is reached, instead of
        // pushing 1 and incrementing every emitted result on the way back
        // up. Eliminates the O(sum-of-path-depths) re-increment pass
        // (~57M ops at simplewiki). Output is byte-identical: at a level
        // with visited.len() == L the old code yielded L (push 1 + L-1
        // increments) and this yields depth + 1 == L, in the same DFS
        // emission order.
        let parents = self.edge_parents_via(cat_id, acc);
        let root_seen = visited.contains(&root_id);
        if !root_seen {
            if parents.contains(&root_id) {
                out.push(depth + 1);
            }
        }

        if visited.len() >= max_depth {
            return;
        }

        // Single reused `visited` Vec with DFS push/pop instead of a fresh
        // per-branch allocation (was ~64M Vec allocs at simplewiki). The
        // path-so-far is restored by pop() on the way back up, so contents
        // at every level are identical to the old `[parent | visited]`
        // copy — output is byte-identical.
        for parent_id in &parents {
            if visited.contains(parent_id) {
                continue;
            }
            visited.push(*parent_id);
            self.collect_native_category_ancestor_hops(*parent_id, root_id, visited, max_depth, acc, depth + 1, out);
            visited.pop();
        }
    }'.

compile_collect_native_bidirectional_ancestor_to_rust(Code) :-
    Code = '    /// Calibrate the bidirectional graph from root_id: BFS downward via
    /// child edges computing the minimum parent-hop distance to root for
    /// every reachable node (the A* lower bound), plus degree-moment
    /// statistics for the direction-weighted distance metric. Returns
    /// (dimensionality D, branch_ratio b, branch_ratio_raw,
    ///  routing_correction, min_dist). Port of calibrateGraph in the
    /// F# kernel template (kernel_bidirectional_ancestor.fs.mustache):
    ///   D     = E[d_child]                 (average child fan-out)
    ///   b_raw = E[d^2_child] / E[d^2_parent]  (second moment ratio)
    ///   b     = b_raw * routing_correction
    /// The routing correction (avg_min_dist / avg_path_hops over a probe
    /// sample) accounts for hub correlation: where paths overlap through
    /// shared hubs, the raw b overestimates the effective branching.
    pub fn calibrate_bidirectional_graph(
        &self,
        root_id: u32,
        parents: &EdgeAccessor,
        children: &EdgeAccessor,
    ) -> (f64, f64, f64, f64, FxMap<u32, i32>) {
        let mut dist: FxMap<u32, i32> = FxMap::default();
        dist.insert(root_id, 0);
        let mut frontier: Vec<u32> = vec![root_id];
        let mut depth: i32 = 0;
        let mut sum_child_deg = 0.0f64;
        let mut sum_child_deg2 = 0.0f64;
        let mut child_nodes = 0usize;
        let mut sum_parent_deg = 0.0f64;
        let mut sum_parent_deg2 = 0.0f64;
        let mut parent_nodes = 0usize;
        while !frontier.is_empty() {
            depth += 1;
            let mut next: Vec<u32> = Vec::new();
            for &nd in &frontier {
                let ch = self.edge_parents_via(nd, children);
                if !ch.is_empty() {
                    let d = ch.len() as f64;
                    sum_child_deg += d;
                    sum_child_deg2 += d * d;
                    child_nodes += 1;
                }
                let ps = self.edge_parents_via(nd, parents);
                if !ps.is_empty() {
                    let d = ps.len() as f64;
                    sum_parent_deg += d;
                    sum_parent_deg2 += d * d;
                    parent_nodes += 1;
                }
                for c in ch {
                    if !dist.contains_key(&c) {
                        dist.insert(c, depth);
                        next.push(c);
                    }
                }
            }
            frontier = next;
        }
        let dimensionality = if child_nodes > 0 {
            (sum_child_deg / child_nodes as f64).max(1.5)
        } else {
            3.0
        };
        let branch_ratio_raw = if child_nodes > 0 && parent_nodes > 0 {
            let ed2_child = sum_child_deg2 / child_nodes as f64;
            let ed2_parent = sum_parent_deg2 / parent_nodes as f64;
            if ed2_parent > 0.0 {
                (ed2_child / ed2_parent).max(1.0)
            } else {
                1.0
            }
        } else {
            1.0
        };
        // Routing correction: probe up to 20 sample nodes at depth 3..=6,
        // enumerating budget-15 paths (parent cost 1, child cost 5,
        // A*-pruned by dist) and measuring avg_min_dist / avg_path_hops.
        // Constants mirror the F# probe exactly.
        let mut seed_sample: Vec<u32> = Vec::new();
        for (k, d) in dist.iter() {
            if *d >= 3 && *d <= 6 {
                seed_sample.push(*k);
                if seed_sample.len() >= 20 {
                    break;
                }
            }
        }
        let routing_correction = if !seed_sample.is_empty() {
            let mut total_min_d = 0.0f64;
            let mut total_hops = 0.0f64;
            let mut total_paths = 0usize;
            for &s in &seed_sample {
                total_min_d += dist[&s] as f64;
                let mut stack: Vec<(u32, f64, i64, Vec<u32>)> =
                    vec![(s, 0.0, 0, vec![s])];
                while let Some((nd, co, h, visited)) = stack.pop() {
                    if nd == root_id {
                        total_hops += h as f64;
                        total_paths += 1;
                    }
                    let gcr = |x: u32| -> f64 {
                        match dist.get(&x) {
                            Some(&d) => d as f64,
                            None => f64::INFINITY,
                        }
                    };
                    if co + 1.0 <= 15.0 {
                        for p in self.edge_parents_via(nd, parents) {
                            if visited.contains(&p) {
                                continue;
                            }
                            let nc = co + 1.0;
                            if nc + gcr(p) > 15.0 {
                                continue;
                            }
                            let mut v2 = visited.clone();
                            v2.push(p);
                            stack.push((p, nc, h + 1, v2));
                        }
                    }
                    if co + 5.0 <= 15.0 {
                        for c in self.edge_parents_via(nd, children) {
                            if visited.contains(&c) {
                                continue;
                            }
                            let nc = co + 5.0;
                            if nc + gcr(c) > 15.0 {
                                continue;
                            }
                            let mut v2 = visited.clone();
                            v2.push(c);
                            stack.push((c, nc, h + 1, v2));
                        }
                    }
                }
            }
            let avg_min_d = total_min_d / seed_sample.len() as f64;
            let avg_hops = if total_paths > 0 {
                total_hops / total_paths as f64
            } else {
                avg_min_d
            };
            if avg_hops > 0.0 {
                (avg_min_d / avg_hops).min(1.0)
            } else {
                1.0
            }
        } else {
            1.0
        };
        let branch_ratio = branch_ratio_raw * routing_correction;
        (dimensionality, branch_ratio, branch_ratio_raw, routing_correction, dist)
    }

    /// Bidirectional effective-distance kernel with path-cost pruning and
    /// A*-style lower-bound elimination. Explores parent hops (cost
    /// parent_cost) and child hops (cost child_cost) from cat_id, pruning
    /// any frontier entry whose cumulative cost plus
    /// min_dist[node] * parent_cost exceeds the budget — it cannot reach
    /// root within budget even using only parent hops (the cheapest
    /// direction). Pushes (total_hops, parent_hops, child_hops) for every
    /// path that reaches root_id within budget; the caller applies the
    /// direction-weighted distance metric. With child_cost above the
    /// budget this degenerates to upward-only search. Port of
    /// nativeKernel_bidirectional_ancestor from the F# template.
    pub fn collect_native_bidirectional_ancestor_hops(
        &self,
        cat_id: u32,
        root_id: u32,
        parent_cost: f64,
        child_cost: f64,
        budget: f64,
        parents: &EdgeAccessor,
        children: &EdgeAccessor,
        min_dist: &FxMap<u32, i32>,
        out: &mut Vec<(i64, i64, i64)>,
    ) {
        let min_cost_to_root = |nd: u32| -> f64 {
            match min_dist.get(&nd) {
                Some(&d) => d as f64 * parent_cost,
                None => f64::INFINITY,
            }
        };
        let mut stack: Vec<(u32, f64, i64, i64, i64, Vec<u32>)> =
            vec![(cat_id, 0.0, 0, 0, 0, vec![cat_id])];
        while let Some((node, cost, hops, p_hops, c_hops, visited)) = stack.pop() {
            if node == root_id && hops > 0 {
                out.push((hops, p_hops, c_hops));
            }
            if cost + parent_cost <= budget {
                for p in self.edge_parents_via(node, parents) {
                    if visited.contains(&p) {
                        continue;
                    }
                    let nc = cost + parent_cost;
                    if nc + min_cost_to_root(p) > budget {
                        continue;
                    }
                    let mut v2 = visited.clone();
                    v2.push(p);
                    stack.push((p, nc, hops + 1, p_hops + 1, c_hops, v2));
                }
            }
            if cost + child_cost <= budget {
                for c in self.edge_parents_via(node, children) {
                    if visited.contains(&c) {
                        continue;
                    }
                    let nc = cost + child_cost;
                    if nc + min_cost_to_root(c) > budget {
                        continue;
                    }
                    let mut v2 = visited.clone();
                    v2.push(c);
                    stack.push((c, nc, hops + 1, p_hops, c_hops + 1, v2));
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
    % dist+ (docs/design/WAM_TRANSITIVE_DISTANCE3_CONTRACT.md): finite BFS;
    % each target once at minimum positive distance. Visited tracks nodes
    % discovered via an edge — do not seed with start (Source appears only
    % for self-loop / nonempty cycle). Inline kept (matches surrounding
    % collect_native_* helpers; not large enough to warrant Mustache).
    Code = '    /// dist+ — docs/design/WAM_TRANSITIVE_DISTANCE3_CONTRACT.md
    pub fn collect_native_transitive_distance_results(
        &self,
        start: &str,
        edge_pred: &str,
        out: &mut Vec<(String, i64)>,
    ) {
        let mut seen: HashSet<String> = HashSet::new();
        let mut queue: VecDeque<(String, i64)> = VecDeque::new();
        queue.push_back((start.to_string(), 0));
        while let Some((node, depth)) = queue.pop_front() {
            if let Some(next_nodes) = self.indexed_atom_fact2.get(edge_pred).and_then(|table| table.get(&node)) {
                for next in next_nodes {
                    if !seen.insert(next.clone()) {
                        continue;
                    }
                    let next_depth = depth + 1;
                    out.push((next.clone(), next_depth));
                    queue.push_back((next.clone(), next_depth));
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

% T9 fact-table enumeration helper (generic, predicate-independent). Each
% candidate is a Value::List of column values; unify all columns with the query
% args. On the first match leave a choice point carrying the remaining candidates
% (resumed via the "fact_table" arm of resume_builtin) so backtrack() yields the
% next matching row. Mirrors builtin_between_attempt for choice-point bookkeeping.
compile_fact_table_attempt_to_rust(Code) :-
    Code = '    /// T9: try each candidate fact row against the query args, leaving a
    /// choice point for the rest so backtrack() enumerates further matches. On
    /// success the pc is set to cont_pc (the call site continuation: pc+1 for a
    /// `call`, or the saved cp for a tail-call `execute`). The continuation is
    /// stashed as the first data slot so resume_builtin can restore it.
    pub fn fact_table_attempt(&mut self, args: Vec<Value>, cands: Vec<Value>, cont_pc: usize) -> bool {
        let mut rest = cands;
        while !rest.is_empty() {
            let row = rest.remove(0);
            let cols = match &row { Value::List(c) => c.clone(), _ => continue };
            if cols.len() != args.len() { continue; }
            // Snapshot the machine BEFORE this row binds anything, so a later
            // backtrack restores exactly here and resume_builtin retries on rest.
            let cp_trail = self.trail.len();
            let cp_heap = self.heap.len();
            let cp_regs = self.save_regs();
            let cp_stack = self.stack.clone();
            let mut ok = true;
            for (a, c) in args.iter().zip(cols.iter()) {
                if !self.unify(a, c) { ok = false; break; }
            }
            if ok {
                if !rest.is_empty() {
                    let mut data = Vec::with_capacity(rest.len() + 1);
                    data.push(Value::Integer(cont_pc as i64));
                    data.append(&mut rest);
                    self.choice_points.push(ChoicePoint {
                        next_pc: cont_pc,
                        saved_args: cp_regs,
                        stack: cp_stack,
                        cp: self.cp,
                        trail_len: cp_trail,
                        heap_len: cp_heap,
                        builtin_state: Some(BuiltinState {
                            name: "fact_table".to_string(),
                            args: args.clone(),
                            data,
                        }),
                        cut_barrier: self.cut_barrier,
                    });
                }
                self.pc = cont_pc;
                return true;
            }
            // This row failed: undo any partial bindings/heap and try the next.
            self.unwind_trail_to(cp_trail);
            self.heap.truncate(cp_heap);
        }
        false
    }'.

compile_resume_builtin_to_rust(Code) :-
    Code = '    fn resume_builtin(&mut self, state: BuiltinState) -> bool {
        match state.name.as_str() {
            "fact_table" => {
                // T9: resume a fact-table scan at the next candidate row. The
                // machine has already been restored to the pre-row snapshot. The
                // continuation pc is data[0]; the remaining candidates follow.
                if state.data.is_empty() { return false; }
                let cont_pc = match &state.data[0] { Value::Integer(n) => *n as usize, _ => return false };
                let rest: Vec<Value> = state.data[1..].to_vec();
                self.fact_table_attempt(state.args, rest, cont_pc)
            }
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
                // Bind through the Y-aware accessors. An aggregate embedded in a
                // larger clause body has a *permanent* (Y) result register, which
                // lives in the environment frame, not the flat regs array, so
                // get_reg_raw/set_reg_str would read/write a dead slot (Uninit)
                // and never surface the result to the clause variable. get_reg/
                // put_reg handle both temporary (A/X) and permanent (Y) registers.
                let lhs = self.get_reg(&result_reg);
                match lhs {
                    Some(Value::Unbound(ref var_name)) => {
                        self.trail_binding(&result_reg);
                        self.put_reg(&result_reg, result.clone());
                        self.bind_var(var_name, result);
                    }
                    Some(existing) if existing == result => {}
                    Some(_) => return false,
                    None => {
                        self.put_reg(&result_reg, result);
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
            "between/3" => {
                let x_raw = state.args[0].clone();
                let (n, high) = match (&state.data[0], &state.data[1]) {
                    (Value::Integer(n), Value::Integer(h)) => (*n, *h),
                    _ => return false,
                };
                self.builtin_between_attempt(&x_raw, n, high)
            }
            "select/3" => {
                let x_raw = state.args[0].clone();
                let items = match &state.args[1] {
                    Value::List(items) => items.clone(),
                    _ => return false,
                };
                let rest_raw = state.args[2].clone();
                let idx = match state.data[0] {
                    Value::Integer(n) => n as usize,
                    _ => return false,
                };
                self.builtin_select_attempt(&x_raw, &items, &rest_raw, idx)
            }
            "nth0/3" | "nth1/3" => {
                let base: i64 = if state.name == "nth1/3" { 1 } else { 0 };
                let n_raw = state.args[0].clone();
                let items = match &state.args[1] {
                    Value::List(items) => items.clone(),
                    _ => return false,
                };
                let elem_raw = state.args[2].clone();
                let idx = match state.data[0] {
                    Value::Integer(n) => n as usize,
                    _ => return false,
                };
                let name = state.name.clone();
                self.builtin_nth_attempt(&name, base, &n_raw, &items, &elem_raw, idx)
            }
            "atom_concat/3" => {
                let a1_raw = state.args[0].clone();
                let a2_raw = state.args[1].clone();
                let chars: Vec<char> = match &state.args[2] {
                    Value::Atom(s) => s.chars().collect(),
                    _ => return false,
                };
                let split = match state.data[0] {
                    Value::Integer(n) => n as usize,
                    _ => return false,
                };
                self.builtin_atom_concat_attempt(&a1_raw, &a2_raw, &chars, split)
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
            "dynamic_call" => {
                let key = match state.args.get(0) {
                    Some(Value::Atom(key)) => key.clone(),
                    _ => return false,
                };
                let start_idx = match state.data.get(0) {
                    Some(Value::Integer(n)) => *n as usize,
                    _ => return false,
                };
                let cont_pc = match state.data.get(1) {
                    Some(Value::Integer(n)) => *n as usize,
                    _ => return false,
                };
                self.dynamic_call_attempt(key, start_idx, cont_pc)
            }
            "dynamic_retract" => {
                let key = match state.args.get(0) {
                    Some(Value::Atom(key)) => key.clone(),
                    _ => return false,
                };
                let pattern = match state.args.get(1) {
                    Some(pattern) => pattern.clone(),
                    _ => return false,
                };
                let start_idx = match state.data.get(0) {
                    Some(Value::Integer(n)) => *n as usize,
                    _ => return false,
                };
                let cont_pc = match state.data.get(1) {
                    Some(Value::Integer(n)) => *n as usize,
                    _ => return false,
                };
                self.dynamic_retract_attempt(key, start_idx, pattern, cont_pc)
            }
            "dynamic_clause" => {
                let key = match state.args.get(0) {
                    Some(Value::Atom(key)) => key.clone(),
                    _ => return false,
                };
                let head = match state.args.get(1) {
                    Some(head) => head.clone(),
                    _ => return false,
                };
                let body = match state.args.get(2) {
                    Some(body) => body.clone(),
                    _ => return false,
                };
                let start_idx = match state.data.get(0) {
                    Some(Value::Integer(n)) => *n as usize,
                    _ => return false,
                };
                let cont_pc = match state.data.get(1) {
                    Some(Value::Integer(n)) => *n as usize,
                    _ => return false,
                };
                self.dynamic_clause_attempt(key, start_idx, head, body, cont_pc)
            }
            "current_predicate" => {
                let keys = match state.args.get(0) {
                    Some(Value::List(keys)) => keys.clone(),
                    _ => return false,
                };
                let name = match state.args.get(1) {
                    Some(name) => name.clone(),
                    _ => return false,
                };
                let arity = match state.args.get(2) {
                    Some(arity) => arity.clone(),
                    _ => return false,
                };
                let start_idx = match state.data.get(0) {
                    Some(Value::Integer(n)) => *n as usize,
                    _ => return false,
                };
                let cont_pc = match state.data.get(1) {
                    Some(Value::Integer(n)) => *n as usize,
                    _ => return false,
                };
                self.current_predicate_attempt(keys, start_idx, name, arity, cont_pc)
            }
            "dynamic_rule_body" => {
                let clause = match state.args.get(0) {
                    Some(clause) => clause.clone(),
                    _ => return false,
                };
                let solution_idx = match state.data.get(0) {
                    Some(Value::Integer(n)) => *n as usize,
                    _ => return false,
                };
                let cont_pc = match state.data.get(1) {
                    Some(Value::Integer(n)) => *n as usize,
                    _ => return false,
                };
                self.dynamic_rule_body_attempt(clause, solution_idx, cont_pc)
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


compile_execute_ext_builtin_to_rust(Code) :-
    Code = '    /// Standard order class: Var < Number < Atom < Compound.
    /// Bool orders as its atom name; the empty list as the atom [].
    fn term_order_class(v: &Value) -> u8 {
        match v {
            Value::Unbound(_) | Value::Ref(_) | Value::Uninit => 0,
            Value::Integer(_) | Value::Float(_) => 1,
            Value::Atom(_) | Value::Bool(_) => 2,
            Value::Str(_, _) => 3,
            Value::List(items) => if items.is_empty() { 2 } else { 3 },
        }
    }

    fn value_atom_name(v: &Value) -> Option<String> {
        match v {
            Value::Atom(s) => Some(s.clone()),
            Value::Bool(true) => Some("true".to_string()),
            Value::Bool(false) => Some("false".to_string()),
            Value::List(items) if items.is_empty() => Some("[]".to_string()),
            _ => None,
        }
    }

    /// Atomic-to-text for atom_concat/atom_length style builtins
    /// (atoms, numbers, booleans; not compounds or variables).
    fn value_atomic_text(v: &Value) -> Option<String> {
        match v {
            Value::Integer(n) => Some(n.to_string()),
            Value::Float(f) => Some(format!("{}", f)),
            other => Self::value_atom_name(other),
        }
    }

    /// Standard order of terms (pragmatic subset shared with sort/2,
    /// msort/2, compare/3 and the @-comparison family). Numbers compare
    /// by value with Float preceding an equal Integer; atoms textually;
    /// lists cons-wise (a strict prefix precedes its extension); other
    /// compounds by arity, then functor name, then args left to right.
    /// Unbound variables order by internal name — stable within a query.
    pub fn term_compare(&self, a: &Value, b: &Value) -> std::cmp::Ordering {
        use std::cmp::Ordering;
        let da = self.deref_heap(&self.deref_var(a));
        let db = self.deref_heap(&self.deref_var(b));
        let ca = Self::term_order_class(&da);
        let cb = Self::term_order_class(&db);
        if ca != cb {
            return ca.cmp(&cb);
        }
        match ca {
            0 => {
                let na = match &da { Value::Unbound(n) => n.clone(), _ => String::new() };
                let nb = match &db { Value::Unbound(n) => n.clone(), _ => String::new() };
                na.cmp(&nb)
            }
            1 => {
                let fa = match &da { Value::Integer(n) => *n as f64, Value::Float(f) => *f, _ => 0.0 };
                let fb = match &db { Value::Integer(n) => *n as f64, Value::Float(f) => *f, _ => 0.0 };
                match fa.partial_cmp(&fb).unwrap_or(Ordering::Equal) {
                    Ordering::Equal => {
                        let ka = matches!(da, Value::Integer(_)) as u8;
                        let kb = matches!(db, Value::Integer(_)) as u8;
                        ka.cmp(&kb)
                    }
                    o => o,
                }
            }
            2 => {
                let na = Self::value_atom_name(&da).unwrap_or_default();
                let nb = Self::value_atom_name(&db).unwrap_or_default();
                na.cmp(&nb)
            }
            _ => match (&da, &db) {
                (Value::List(l1), Value::List(l2)) => {
                    let n = l1.len().min(l2.len());
                    for i in 0..n {
                        let o = self.term_compare(&l1[i], &l2[i]);
                        if o != Ordering::Equal { return o; }
                    }
                    l1.len().cmp(&l2.len())
                }
                (Value::List(l), Value::Str(f, args))
                    if self.is_cons_functor(f) && args.len() == 2 && !l.is_empty() => {
                    let o = self.term_compare(&l[0], &args[0]);
                    if o != Ordering::Equal { return o; }
                    self.term_compare(&Value::List(l[1..].to_vec()), &args[1])
                }
                (Value::Str(f, args), Value::List(l))
                    if self.is_cons_functor(f) && args.len() == 2 && !l.is_empty() => {
                    let o = self.term_compare(&args[0], &l[0]);
                    if o != Ordering::Equal { return o; }
                    self.term_compare(&args[1], &Value::List(l[1..].to_vec()))
                }
                (Value::Str(f1, a1), Value::Str(f2, a2)) => {
                    match a1.len().cmp(&a2.len()) {
                        Ordering::Equal => {
                            let n1 = Self::display_functor_name(f1, a1.len());
                            let n2 = Self::display_functor_name(f2, a2.len());
                            match n1.cmp(&n2) {
                                Ordering::Equal => {
                                    for i in 0..a1.len() {
                                        let o = self.term_compare(&a1[i], &a2[i]);
                                        if o != Ordering::Equal { return o; }
                                    }
                                    Ordering::Equal
                                }
                                o => o,
                            }
                        }
                        o => o,
                    }
                }
                // List vs Str non-cons compound: lists are ./2, lowest arity 2
                (Value::List(_), Value::Str(_, args)) => 2usize.cmp(&args.len()),
                (Value::Str(_, args), Value::List(_)) => args.len().cmp(&2usize),
                _ => Ordering::Equal,
            },
        }
    }

    fn value_is_ground(&self, v: &Value) -> bool {
        match self.deref_heap(&self.deref_var(v)) {
            Value::Unbound(_) => false,
            Value::List(items) => items.iter().all(|i| self.value_is_ground(i)),
            Value::Str(_, args) => args.iter().all(|a| self.value_is_ground(a)),
            _ => true,
        }
    }

    /// Extract the 1-based compound argument used by sort/4. None
    /// means compare the whole term, including Key=0 and out-of-range
    /// keys. Lists expose their head and tail as arguments 1 and 2.
    fn builtin_sort_key(&self, value: &Value, position: usize) -> Option<Value> {
        if position == 0 { return None; }
        let term = self.deref_heap(&self.deref_var(value));
        match &term {
            Value::Str(_, args) if position <= args.len() => {
                Some(self.deref_heap(&self.deref_var(&args[position - 1])))
            }
            Value::List(items) if !items.is_empty() && position == 1 => {
                Some(self.deref_heap(&self.deref_var(&items[0])))
            }
            Value::List(items) if !items.is_empty() && position == 2 => {
                Some(Value::List(items[1..].to_vec()))
            }
            _ => None,
        }
    }

    /// Test list membership by unification. Failed candidates are
    /// unwound; the first successful candidate keeps its bindings.
    fn builtin_unify_member(&mut self, item: &Value, candidates: &[Value]) -> bool {
        for candidate in candidates {
            let mark = self.trail.len();
            if self.unify(item, candidate) { return true; }
            self.unwind_trail_to(mark);
        }
        false
    }

    /// One alternative of between/3 enumeration: bind X to N, leaving a
    /// choice point for N+1..High. Called from the first builtin
    /// dispatch and from resume_builtin on backtrack.
    fn builtin_between_attempt(&mut self, x_raw: &Value, n: i64, high: i64) -> bool {
        if n > high { return false; }
        if n < high {
            self.choice_points.push(ChoicePoint {
                next_pc: self.pc,
                saved_args: self.save_regs(),
                stack: self.stack.clone(),
                cp: self.cp,
                trail_len: self.trail.len(),
                heap_len: self.heap.len(),
                builtin_state: Some(BuiltinState {
                    name: "between/3".to_string(),
                    args: vec![x_raw.clone()],
                    data: vec![Value::Integer(n + 1), Value::Integer(high)],
                }),
                cut_barrier: self.cut_barrier,
            });
        }
        if self.unify(x_raw, &Value::Integer(n)) { self.pc += 1; true } else { false }
    }

    /// One alternative of select/3 over a bound list: unify X with item
    /// idx and Rest with the list minus that item.
    fn builtin_select_attempt(&mut self, x_raw: &Value, items: &[Value], rest_raw: &Value, idx: usize) -> bool {
        if idx >= items.len() { return false; }
        if idx + 1 < items.len() {
            self.choice_points.push(ChoicePoint {
                next_pc: self.pc,
                saved_args: self.save_regs(),
                stack: self.stack.clone(),
                cp: self.cp,
                trail_len: self.trail.len(),
                heap_len: self.heap.len(),
                builtin_state: Some(BuiltinState {
                    name: "select/3".to_string(),
                    args: vec![x_raw.clone(), Value::List(items.to_vec()), rest_raw.clone()],
                    data: vec![Value::Integer((idx + 1) as i64)],
                }),
                cut_barrier: self.cut_barrier,
            });
        }
        let mut rest: Vec<Value> = items.to_vec();
        rest.remove(idx);
        if self.unify(x_raw, &items[idx]) && self.unify(rest_raw, &Value::List(rest)) {
            self.pc += 1; true
        } else { false }
    }

    /// One alternative of nth0/nth1 enumeration with an unbound index.
    /// base is 0 for nth0, 1 for nth1; name carries it through resume.
    fn builtin_nth_attempt(&mut self, name: &str, base: i64, n_raw: &Value, items: &[Value], elem_raw: &Value, idx: usize) -> bool {
        if idx >= items.len() { return false; }
        if idx + 1 < items.len() {
            self.choice_points.push(ChoicePoint {
                next_pc: self.pc,
                saved_args: self.save_regs(),
                stack: self.stack.clone(),
                cp: self.cp,
                trail_len: self.trail.len(),
                heap_len: self.heap.len(),
                builtin_state: Some(BuiltinState {
                    name: name.to_string(),
                    args: vec![n_raw.clone(), Value::List(items.to_vec()), elem_raw.clone()],
                    data: vec![Value::Integer((idx + 1) as i64)],
                }),
                cut_barrier: self.cut_barrier,
            });
        }
        if self.unify(n_raw, &Value::Integer(idx as i64 + base))
            && self.unify(elem_raw, &items[idx]) {
            self.pc += 1; true
        } else { false }
    }

    /// One alternative of atom_concat/3 split enumeration over a bound
    /// third argument (chars is its full character vector).
    fn builtin_atom_concat_attempt(&mut self, a1_raw: &Value, a2_raw: &Value, chars: &[char], split: usize) -> bool {
        if split > chars.len() { return false; }
        if split < chars.len() {
            let whole: String = chars.iter().collect();
            self.choice_points.push(ChoicePoint {
                next_pc: self.pc,
                saved_args: self.save_regs(),
                stack: self.stack.clone(),
                cp: self.cp,
                trail_len: self.trail.len(),
                heap_len: self.heap.len(),
                builtin_state: Some(BuiltinState {
                    name: "atom_concat/3".to_string(),
                    args: vec![a1_raw.clone(), a2_raw.clone(), Value::Atom(whole)],
                    data: vec![Value::Integer((split + 1) as i64)],
                }),
                cut_barrier: self.cut_barrier,
            });
        }
        let prefix: String = chars[..split].iter().collect();
        let suffix: String = chars[split..].iter().collect();
        if self.unify(a1_raw, &Value::Atom(prefix)) && self.unify(a2_raw, &Value::Atom(suffix)) {
            self.pc += 1; true
        } else { false }
    }

    /// Extended builtin dispatch: term ordering, list utilities, atom
    /// and string text operations, integer relations. F#-parity sweep —
    /// every op here is already emitted as builtin_call by the shared
    /// WAM compiler (is_builtin_pred), so missing arms previously failed
    /// closed at runtime.
    fn execute_ext_builtin(&mut self, op: &str, _arity: usize) -> bool {
        use std::cmp::Ordering;
        match op {
            "\\\\=/2" => {
                // Not unifiable: trial-unify, then unwind any bindings.
                let a1 = self.get_reg_raw("A1").unwrap_or(Value::Uninit);
                let a2 = self.get_reg_raw("A2").unwrap_or(Value::Uninit);
                let mark = self.trail.len();
                let unified = self.unify(&a1, &a2);
                self.unwind_trail_to(mark);
                if unified { false } else { self.pc += 1; true }
            }
            "\\\\==/2" => {
                let v1 = self.get_reg_raw("A1").map(|v| self.deref_heap(&self.deref_var(&v)));
                let v2 = self.get_reg_raw("A2").map(|v| self.deref_heap(&self.deref_var(&v)));
                if v1 != v2 { self.pc += 1; true } else { false }
            }
            "@</2" | "@=</2" | "@>/2" | "@>=/2" => {
                let a1 = self.get_reg_raw("A1").unwrap_or(Value::Uninit);
                let a2 = self.get_reg_raw("A2").unwrap_or(Value::Uninit);
                let o = self.term_compare(&a1, &a2);
                let ok = match op {
                    "@</2" => o == Ordering::Less,
                    "@=</2" => o != Ordering::Greater,
                    "@>/2" => o == Ordering::Greater,
                    "@>=/2" => o != Ordering::Less,
                    _ => false,
                };
                if ok { self.pc += 1; true } else { false }
            }
            "compare/3" => {
                let a2 = self.get_reg_raw("A2").unwrap_or(Value::Uninit);
                let a3 = self.get_reg_raw("A3").unwrap_or(Value::Uninit);
                let sym = match self.term_compare(&a2, &a3) {
                    Ordering::Less => "<",
                    Ordering::Equal => "=",
                    Ordering::Greater => ">",
                };
                let a1 = self.get_reg_raw("A1").unwrap_or(Value::Uninit);
                if self.unify(&a1, &Value::Atom(sym.to_string())) { self.pc += 1; true } else { false }
            }
            "msort/2" | "sort/2" => {
                let list = match self.get_reg_raw("A1").map(|v| self.deref_heap(&self.deref_var(&v))) {
                    Some(Value::List(items)) => items,
                    _ => return false,
                };
                let mut sorted: Vec<Value> = list.iter()
                    .map(|v| self.deref_heap(&self.deref_var(v)))
                    .collect();
                sorted.sort_by(|a, b| self.term_compare(a, b));
                if op == "sort/2" {
                    sorted.dedup_by(|a, b| self.term_compare(a, b) == Ordering::Equal);
                }
                let a2 = self.get_reg_raw("A2").unwrap_or(Value::Uninit);
                if self.unify(&a2, &Value::List(sorted)) { self.pc += 1; true } else { false }
            }
            "sort/4" => {
                let key_position = match self.get_reg_raw("A1")
                    .map(|v| self.deref_heap(&self.deref_var(&v))) {
                    Some(Value::Integer(position)) if position >= 0 => {
                        match usize::try_from(position) {
                            Ok(position) => position,
                            Err(_) => return false,
                        }
                    }
                    _ => return false,
                };
                let order = match self.get_reg_raw("A2")
                    .map(|v| self.deref_heap(&self.deref_var(&v))) {
                    Some(Value::Atom(order)) => order,
                    _ => return false,
                };
                let (descending, deduplicate) = match order.as_str() {
                    "@<" => (false, true),
                    "@=<" => (false, false),
                    "@>" => (true, true),
                    "@>=" => (true, false),
                    _ => return false,
                };
                let list = match self.get_reg_raw("A3")
                    .map(|v| self.deref_heap(&self.deref_var(&v))) {
                    Some(Value::List(items)) => items,
                    _ => return false,
                };
                let mut keyed = Vec::with_capacity(list.len());
                for item in &list {
                    let value = self.deref_heap(&self.deref_var(item));
                    let key = self.builtin_sort_key(&value, key_position);
                    keyed.push((value, key));
                }
                keyed.sort_by(|a, b| {
                    let a_key = a.1.as_ref().unwrap_or(&a.0);
                    let b_key = b.1.as_ref().unwrap_or(&b.0);
                    let ordering = self.term_compare(a_key, b_key);
                    if descending { ordering.reverse() } else { ordering }
                });
                if deduplicate {
                    keyed.dedup_by(|a, b| {
                        let a_key = a.1.as_ref().unwrap_or(&a.0);
                        let b_key = b.1.as_ref().unwrap_or(&b.0);
                        self.term_compare(a_key, b_key) == Ordering::Equal
                    });
                }
                let sorted = keyed.into_iter()
                    .map(|(value, _)| value)
                    .collect::<Vec<_>>();
                let output = self.get_reg_raw("A4").unwrap_or(Value::Uninit);
                let mark = self.trail.len();
                if self.unify(&output, &Value::List(sorted)) {
                    self.pc += 1; true
                } else {
                    self.unwind_trail_to(mark);
                    false
                }
            }
            "keysort/2" => {
                let list = match self.get_reg_raw("A1").map(|v| self.deref_heap(&self.deref_var(&v))) {
                    Some(Value::List(items)) => items,
                    _ => return false,
                };
                let mut keyed: Vec<(Value, Value)> = Vec::with_capacity(list.len());
                for item in &list {
                    match self.deref_heap(&self.deref_var(item)) {
                        Value::Str(f, args)
                            if args.len() == 2 && Self::display_functor_name(&f, 2) == "-" =>
                            keyed.push((args[0].clone(), Value::Str(f, args))),
                        _ => return false,
                    }
                }
                keyed.sort_by(|a, b| self.term_compare(&a.0, &b.0));
                let sorted: Vec<Value> = keyed.into_iter().map(|kv| kv.1).collect();
                let a2 = self.get_reg_raw("A2").unwrap_or(Value::Uninit);
                if self.unify(&a2, &Value::List(sorted)) { self.pc += 1; true } else { false }
            }
            "pairs_keys/2" | "pairs_values/2" => {
                let pairs = self.get_reg_raw("A1").unwrap_or(Value::Uninit);
                let take_keys = op == "pairs_keys/2";
                let (keys, values) = match self.pair_list_columns(&pairs, take_keys, !take_keys) {
                    Some(parts) => parts,
                    None => return false,
                };
                let projected = if op == "pairs_keys/2" { keys } else { values };
                let output = self.get_reg_raw("A2").unwrap_or(Value::Uninit);
                let mark = self.trail.len();
                if self.unify(&output, &Value::List(projected)) {
                    self.pc += 1; true
                } else {
                    self.unwind_trail_to(mark);
                    false
                }
            }
            "pairs_keys_values/3" => {
                let pairs_raw = self.get_reg_raw("A1").unwrap_or(Value::Uninit);
                let keys_raw = self.get_reg_raw("A2").unwrap_or(Value::Uninit);
                let values_raw = self.get_reg_raw("A3").unwrap_or(Value::Uninit);
                let pairs = self.deref_heap(&self.deref_var(&pairs_raw));
                if matches!(pairs, Value::Unbound(_)) {
                    let keys = match self.value_as_list(&keys_raw) {
                        Some(items) => items,
                        None => return false,
                    };
                    let values = match self.value_as_list(&values_raw) {
                        Some(items) if items.len() == keys.len() => items,
                        _ => return false,
                    };
                    let zipped: Vec<Value> = keys.into_iter().zip(values)
                        .map(|(key, value)| Value::Str("-".to_string(), vec![key, value]))
                        .collect();
                    let mark = self.trail.len();
                    if self.unify(&pairs_raw, &Value::List(zipped)) {
                        self.pc += 1; true
                    } else {
                        self.unwind_trail_to(mark);
                        false
                    }
                } else {
                    let (keys, values) = match self.pair_list_columns(&pairs_raw, true, true) {
                        Some(parts) => parts,
                        None => return false,
                    };
                    let mark = self.trail.len();
                    if !self.unify(&keys_raw, &Value::List(keys)) {
                        self.unwind_trail_to(mark);
                        return false;
                    }
                    if self.unify(&values_raw, &Value::List(values)) {
                        self.pc += 1; true
                    } else {
                        self.unwind_trail_to(mark);
                        false
                    }
                }
            }
            "memberchk/2" => {
                // First element that unifies wins, deterministically;
                // failed attempts are unwound, the winning binding kept.
                let x = self.get_reg_raw("A1").unwrap_or(Value::Uninit);
                let list = match self.get_reg_raw("A2").map(|v| self.deref_heap(&self.deref_var(&v))) {
                    Some(Value::List(items)) => items,
                    _ => return false,
                };
                for item in &list {
                    let mark = self.trail.len();
                    if self.unify(&x, item) { self.pc += 1; return true; }
                    self.unwind_trail_to(mark);
                }
                false
            }
            "last/2" => {
                let list = match self.get_reg_raw("A1").map(|v| self.deref_heap(&self.deref_var(&v))) {
                    Some(Value::List(items)) if !items.is_empty() => items,
                    _ => return false,
                };
                let a2 = self.get_reg_raw("A2").unwrap_or(Value::Uninit);
                let last = list[list.len() - 1].clone();
                if self.unify(&a2, &last) { self.pc += 1; true } else { false }
            }
            "nth0/3" | "nth1/3" => {
                let base: i64 = if op == "nth1/3" { 1 } else { 0 };
                let n_raw = self.get_reg_raw("A1").unwrap_or(Value::Uninit);
                let list = match self.get_reg_raw("A2").map(|v| self.deref_heap(&self.deref_var(&v))) {
                    Some(Value::List(items)) => items,
                    _ => return false,
                };
                let elem_raw = self.get_reg_raw("A3").unwrap_or(Value::Uninit);
                match self.deref_var(&n_raw) {
                    Value::Integer(n) => {
                        let idx = n - base;
                        if idx < 0 || idx as usize >= list.len() { return false; }
                        let item = list[idx as usize].clone();
                        if self.unify(&elem_raw, &item) { self.pc += 1; true } else { false }
                    }
                    Value::Unbound(_) => self.builtin_nth_attempt(op, base, &n_raw, &list, &elem_raw, 0),
                    _ => false,
                }
            }
            "numlist/3" => {
                let low = match self.get_reg_raw("A1").map(|v| self.deref_var(&v)) {
                    Some(Value::Integer(n)) => n,
                    _ => return false,
                };
                let high = match self.get_reg_raw("A2").map(|v| self.deref_var(&v)) {
                    Some(Value::Integer(n)) => n,
                    _ => return false,
                };
                if low > high { return false; }
                let items: Vec<Value> = (low..=high).map(Value::Integer).collect();
                let a3 = self.get_reg_raw("A3").unwrap_or(Value::Uninit);
                if self.unify(&a3, &Value::List(items)) { self.pc += 1; true } else { false }
            }
            "delete/3" => {
                // Keep elements that do NOT unify with A2 (trial-unify,
                // always unwound — matches the no-residual-bindings use).
                let list = match self.get_reg_raw("A1").map(|v| self.deref_heap(&self.deref_var(&v))) {
                    Some(Value::List(items)) => items,
                    _ => return false,
                };
                let pat = self.get_reg_raw("A2").unwrap_or(Value::Uninit);
                let mut kept: Vec<Value> = Vec::new();
                for item in &list {
                    let mark = self.trail.len();
                    let matched = self.unify(&pat, item);
                    self.unwind_trail_to(mark);
                    if !matched { kept.push(item.clone()); }
                }
                let a3 = self.get_reg_raw("A3").unwrap_or(Value::Uninit);
                if self.unify(&a3, &Value::List(kept)) { self.pc += 1; true } else { false }
            }
            "subtract/3" => {
                let list = match self.get_reg_raw("A1").map(|v| self.deref_heap(&self.deref_var(&v))) {
                    Some(Value::List(items)) => items,
                    _ => return false,
                };
                let excluded = match self.get_reg_raw("A2").map(|v| self.deref_heap(&self.deref_var(&v))) {
                    Some(Value::List(items)) => items,
                    _ => return false,
                };
                let kept: Vec<Value> = list.into_iter()
                    .filter(|item| !excluded.contains(item))
                    .collect();
                let a3 = self.get_reg_raw("A3").unwrap_or(Value::Uninit);
                if self.unify(&a3, &Value::List(kept)) { self.pc += 1; true } else { false }
            }
            "intersection/3" => {
                let left = match self.get_reg_raw("A1")
                    .map(|v| self.deref_heap(&self.deref_var(&v))) {
                    Some(Value::List(items)) => items,
                    _ => return false,
                };
                let right = match self.get_reg_raw("A2")
                    .map(|v| self.deref_heap(&self.deref_var(&v))) {
                    Some(Value::List(items)) => items,
                    _ => return false,
                };
                let mark = self.trail.len();
                let mut common = Vec::with_capacity(left.len());
                for item in &left {
                    if self.builtin_unify_member(item, &right) {
                        common.push(item.clone());
                    }
                }
                let output = self.get_reg_raw("A3").unwrap_or(Value::Uninit);
                if self.unify(&output, &Value::List(common)) {
                    self.pc += 1; true
                } else {
                    self.unwind_trail_to(mark);
                    false
                }
            }
            "union/3" => {
                let left = match self.get_reg_raw("A1")
                    .map(|v| self.deref_heap(&self.deref_var(&v))) {
                    Some(Value::List(items)) => items,
                    _ => return false,
                };
                let right = match self.get_reg_raw("A2")
                    .map(|v| self.deref_heap(&self.deref_var(&v))) {
                    Some(Value::List(items)) => items,
                    _ => return false,
                };
                let mark = self.trail.len();
                let mut union = Vec::with_capacity(left.len() + right.len());
                union.extend(left.iter().cloned());
                for item in &right {
                    if !self.builtin_unify_member(item, &left) {
                        union.push(item.clone());
                    }
                }
                let output = self.get_reg_raw("A3").unwrap_or(Value::Uninit);
                if self.unify(&output, &Value::List(union)) {
                    self.pc += 1; true
                } else {
                    self.unwind_trail_to(mark);
                    false
                }
            }
            "list_to_set/2" => {
                let items = match self.get_reg_raw("A1")
                    .map(|v| self.deref_heap(&self.deref_var(&v))) {
                    Some(Value::List(items)) => items,
                    _ => return false,
                };
                let mark = self.trail.len();
                let mut unique = Vec::with_capacity(items.len());
                for item in &items {
                    if !self.builtin_unify_member(item, &unique) {
                        unique.push(item.clone());
                    }
                }
                let output = self.get_reg_raw("A2").unwrap_or(Value::Uninit);
                if self.unify(&output, &Value::List(unique)) {
                    self.pc += 1; true
                } else {
                    self.unwind_trail_to(mark);
                    false
                }
            }
            "select/3" => {
                let x_raw = self.get_reg_raw("A1").unwrap_or(Value::Uninit);
                let list = match self.get_reg_raw("A2").map(|v| self.deref_heap(&self.deref_var(&v))) {
                    Some(Value::List(items)) if !items.is_empty() => items,
                    _ => return false,
                };
                let rest_raw = self.get_reg_raw("A3").unwrap_or(Value::Uninit);
                self.builtin_select_attempt(&x_raw, &list, &rest_raw, 0)
            }
            "between/3" => {
                let low = match self.get_reg_raw("A1").map(|v| self.deref_var(&v)) {
                    Some(Value::Integer(n)) => n,
                    _ => return false,
                };
                let high = match self.get_reg_raw("A2").map(|v| self.deref_var(&v)) {
                    Some(Value::Integer(n)) => n,
                    _ => return false,
                };
                let x_raw = self.get_reg_raw("A3").unwrap_or(Value::Uninit);
                match self.deref_var(&x_raw) {
                    Value::Integer(x) => {
                        if low <= x && x <= high { self.pc += 1; true } else { false }
                    }
                    Value::Unbound(_) => self.builtin_between_attempt(&x_raw, low, high),
                    _ => false,
                }
            }
            "succ/2" => {
                // Bidirectional natural-number successor: succ(X, Y) iff
                // Y = X + 1, X >= 0, Y >= 1. Reached via the ISO
                // meta-builtin Call fallback (the shared compiler emits
                // call succ/2, not builtin_call).
                let v1 = self.get_reg_raw("A1").map(|v| self.deref_var(&v)).unwrap_or(Value::Uninit);
                let v2 = self.get_reg_raw("A2").map(|v| self.deref_var(&v)).unwrap_or(Value::Uninit);
                match (&v1, &v2) {
                    (Value::Integer(x), _) if *x >= 0 => {
                        let y = Value::Integer(x + 1);
                        if self.unify(&v2, &y) { self.pc += 1; true } else { false }
                    }
                    (Value::Unbound(_), Value::Integer(y)) if *y >= 1 => {
                        let x = Value::Integer(y - 1);
                        if self.unify(&v1, &x) { self.pc += 1; true } else { false }
                    }
                    _ => false,
                }
            }
            "plus/3" => {
                let vals: Vec<Value> = ["A1", "A2", "A3"].iter()
                    .map(|r| self.get_reg_raw(r).map(|v| self.deref_var(&v)).unwrap_or(Value::Uninit))
                    .collect();
                let ints: Vec<Option<i64>> = vals.iter().map(|v| match v {
                    Value::Integer(n) => Some(*n),
                    _ => None,
                }).collect();
                match (ints[0], ints[1], ints[2]) {
                    (Some(a), Some(b), Some(c)) => {
                        if a + b == c { self.pc += 1; true } else { false }
                    }
                    (Some(a), Some(b), None) => {
                        if self.unify(&vals[2], &Value::Integer(a + b)) { self.pc += 1; true } else { false }
                    }
                    (Some(a), None, Some(c)) => {
                        if self.unify(&vals[1], &Value::Integer(c - a)) { self.pc += 1; true } else { false }
                    }
                    (None, Some(b), Some(c)) => {
                        if self.unify(&vals[0], &Value::Integer(c - b)) { self.pc += 1; true } else { false }
                    }
                    _ => false,
                }
            }
            "sum_list/2" | "sumlist/2" | "max_list/2" | "min_list/2" => {
                let list = match self.get_reg_raw("A1").map(|v| self.deref_heap(&self.deref_var(&v))) {
                    Some(Value::List(items)) => items,
                    _ => return false,
                };
                let mut nums: Vec<f64> = Vec::with_capacity(list.len());
                let mut all_int = true;
                let mut ints: Vec<i64> = Vec::with_capacity(list.len());
                for item in &list {
                    match self.deref_var(item) {
                        Value::Integer(n) => { nums.push(n as f64); ints.push(n); }
                        Value::Float(f) => { nums.push(f); all_int = false; }
                        _ => return false,
                    }
                }
                let result = match op {
                    "sum_list/2" | "sumlist/2" => {
                        if all_int { Value::Integer(ints.iter().sum()) }
                        else { Value::Float(nums.iter().sum()) }
                    }
                    "max_list/2" => {
                        if nums.is_empty() { return false; }
                        if all_int { Value::Integer(*ints.iter().max().unwrap()) }
                        else { Value::Float(nums.iter().cloned().fold(f64::NEG_INFINITY, f64::max)) }
                    }
                    _ => {
                        if nums.is_empty() { return false; }
                        if all_int { Value::Integer(*ints.iter().min().unwrap()) }
                        else { Value::Float(nums.iter().cloned().fold(f64::INFINITY, f64::min)) }
                    }
                };
                let a2 = self.get_reg_raw("A2").unwrap_or(Value::Uninit);
                if self.unify(&a2, &result) { self.pc += 1; true } else { false }
            }
            "atom_length/2" | "string_length/2" => {
                let text = match self.get_reg_raw("A1").map(|v| self.deref_var(&v))
                    .as_ref().and_then(Self::value_atomic_text) {
                    Some(t) => t,
                    None => return false,
                };
                let len = Value::Integer(text.chars().count() as i64);
                let a2 = self.get_reg_raw("A2").unwrap_or(Value::Uninit);
                if self.unify(&a2, &len) { self.pc += 1; true } else { false }
            }
            "split_string/4" => {
                let text = match self.get_reg_raw("A1")
                    .map(|v| self.deref_heap(&self.deref_var(&v)))
                    .as_ref().and_then(Self::value_atomic_text) {
                    Some(t) => t,
                    None => return false,
                };
                let separators = match self.get_reg_raw("A2")
                    .map(|v| self.deref_heap(&self.deref_var(&v)))
                    .as_ref().and_then(Self::value_atomic_text) {
                    Some(t) => t.chars().collect::<Vec<_>>(),
                    None => return false,
                };
                let pads = match self.get_reg_raw("A3")
                    .map(|v| self.deref_heap(&self.deref_var(&v)))
                    .as_ref().and_then(Self::value_atomic_text) {
                    Some(t) => t.chars().collect::<Vec<_>>(),
                    None => return false,
                };
                let parts = text
                    .split(|ch| separators.contains(&ch))
                    .map(|part| {
                        Value::Atom(part.trim_matches(|ch| pads.contains(&ch)).to_string())
                    })
                    .collect::<Vec<_>>();
                let output = self.get_reg_raw("A4").unwrap_or(Value::Uninit);
                let mark = self.trail.len();
                if self.unify(&output, &Value::List(parts)) {
                    self.pc += 1;
                    true
                } else {
                    self.unwind_trail_to(mark);
                    false
                }
            }
            "atom_split/3" => {
                let text = match self.get_reg_raw("A1")
                    .map(|v| self.deref_heap(&self.deref_var(&v))) {
                    Some(Value::Atom(s)) => s,
                    _ => return false,
                };
                let separator = match self.get_reg_raw("A2")
                    .map(|v| self.deref_heap(&self.deref_var(&v))) {
                    Some(Value::Atom(s)) => s,
                    _ => return false,
                };
                let mut separator_chars = separator.chars();
                let separator_char = match separator_chars.next() {
                    Some(ch) if separator_chars.next().is_none() => ch,
                    _ => return false,
                };
                let parts = text
                    .split(separator_char)
                    .map(|part| Value::Atom(part.to_string()))
                    .collect::<Vec<_>>();
                let output = self.get_reg_raw("A3").unwrap_or(Value::Uninit);
                let mark = self.trail.len();
                if self.unify(&output, &Value::List(parts)) {
                    self.pc += 1;
                    true
                } else {
                    self.unwind_trail_to(mark);
                    false
                }
            }
            "atom_starts_with/2" | "atom_ends_with/2" | "atom_contains/2" => {
                let text = match self.get_reg_raw("A1")
                    .map(|v| self.deref_heap(&self.deref_var(&v))) {
                    Some(Value::Atom(s)) => s,
                    _ => return false,
                };
                let fragment = match self.get_reg_raw("A2")
                    .map(|v| self.deref_heap(&self.deref_var(&v))) {
                    Some(Value::Atom(s)) => s,
                    _ => return false,
                };
                let matched = match op {
                    "atom_starts_with/2" => text.starts_with(&fragment),
                    "atom_ends_with/2" => text.ends_with(&fragment),
                    "atom_contains/2" => text.contains(&fragment),
                    _ => false,
                };
                if matched { self.pc += 1; true } else { false }
            }
            "atom_concat/3" | "string_concat/3" => {
                let v1 = self.get_reg_raw("A1").map(|v| self.deref_var(&v)).unwrap_or(Value::Uninit);
                let v2 = self.get_reg_raw("A2").map(|v| self.deref_var(&v)).unwrap_or(Value::Uninit);
                let t1 = Self::value_atomic_text(&v1);
                let t2 = Self::value_atomic_text(&v2);
                if let (Some(t1), Some(t2)) = (&t1, &t2) {
                    let a3 = self.get_reg_raw("A3").unwrap_or(Value::Uninit);
                    let whole = Value::Atom(format!("{}{}", t1, t2));
                    return if self.unify(&a3, &whole) { self.pc += 1; true } else { false };
                }
                // Split mode: enumerate prefix/suffix pairs of a bound A3.
                let whole = match self.get_reg_raw("A3").map(|v| self.deref_var(&v))
                    .as_ref().and_then(Self::value_atomic_text) {
                    Some(t) => t,
                    None => return false,
                };
                let chars: Vec<char> = whole.chars().collect();
                let a1_raw = self.get_reg_raw("A1").unwrap_or(Value::Uninit);
                let a2_raw = self.get_reg_raw("A2").unwrap_or(Value::Uninit);
                self.builtin_atom_concat_attempt(&a1_raw, &a2_raw, &chars, 0)
            }
            "string_code/3" => {
                let offset = match self.get_reg_raw("A1")
                    .map(|v| self.deref_heap(&self.deref_var(&v))) {
                    Some(Value::Integer(n)) if n >= 1 => match usize::try_from(n - 1) {
                        Ok(i) => i,
                        Err(_) => return false,
                    },
                    _ => return false,
                };
                let text = match self.get_reg_raw("A2")
                    .map(|v| self.deref_heap(&self.deref_var(&v))) {
                    Some(Value::Atom(s)) => s,
                    _ => return false,
                };
                let code = match text.chars().nth(offset) {
                    Some(ch) => Value::Integer(ch as i64),
                    None => return false,
                };
                let a3 = self.get_reg_raw("A3").unwrap_or(Value::Uninit);
                if self.unify(&a3, &code) { self.pc += 1; true } else { false }
            }
            "char_code/2" => {
                let v1 = self.get_reg_raw("A1").map(|v| self.deref_var(&v)).unwrap_or(Value::Uninit);
                match &v1 {
                    Value::Atom(s) if s.chars().count() == 1 => {
                        let code = Value::Integer(s.chars().next().unwrap() as i64);
                        let a2 = self.get_reg_raw("A2").unwrap_or(Value::Uninit);
                        if self.unify(&a2, &code) { self.pc += 1; true } else { false }
                    }
                    _ => {
                        let code = match self.get_reg_raw("A2").map(|v| self.deref_var(&v)) {
                            Some(Value::Integer(n)) => n,
                            _ => return false,
                        };
                        let ch = match char::from_u32(code as u32) {
                            Some(c) => c,
                            None => return false,
                        };
                        if self.unify(&v1, &Value::Atom(ch.to_string())) { self.pc += 1; true } else { false }
                    }
                }
            }
            "atom_chars/2" | "string_chars/2" => {
                let v1 = self.get_reg_raw("A1").map(|v| self.deref_var(&v)).unwrap_or(Value::Uninit);
                if let Some(text) = Self::value_atomic_text(&v1) {
                    let chars: Vec<Value> = text.chars()
                        .map(|c| Value::Atom(c.to_string()))
                        .collect();
                    let a2 = self.get_reg_raw("A2").unwrap_or(Value::Uninit);
                    return if self.unify(&a2, &Value::List(chars)) { self.pc += 1; true } else { false };
                }
                let list = match self.get_reg_raw("A2").map(|v| self.deref_heap(&self.deref_var(&v))) {
                    Some(Value::List(items)) => items,
                    _ => return false,
                };
                let mut text = String::new();
                for item in &list {
                    match self.deref_var(item) {
                        Value::Atom(s) => text.push_str(&s),
                        _ => return false,
                    }
                }
                if self.unify(&v1, &Value::Atom(text)) { self.pc += 1; true } else { false }
            }
            "atom_string/2" | "string_to_atom/2" => {
                // Atoms double as strings in this runtime; both are
                // text-identity conversions in either direction.
                let v1 = self.get_reg_raw("A1").map(|v| self.deref_var(&v)).unwrap_or(Value::Uninit);
                let v2 = self.get_reg_raw("A2").map(|v| self.deref_var(&v)).unwrap_or(Value::Uninit);
                if let Some(t) = Self::value_atomic_text(&v1) {
                    let a2 = self.get_reg_raw("A2").unwrap_or(Value::Uninit);
                    return if self.unify(&a2, &Value::Atom(t)) { self.pc += 1; true } else { false };
                }
                if let Some(t) = Self::value_atomic_text(&v2) {
                    let a1 = self.get_reg_raw("A1").unwrap_or(Value::Uninit);
                    return if self.unify(&a1, &Value::Atom(t)) { self.pc += 1; true } else { false };
                }
                false
            }
            "upcase_atom/2" | "downcase_atom/2" => {
                let text = match self.get_reg_raw("A1").map(|v| self.deref_var(&v))
                    .as_ref().and_then(Self::value_atomic_text) {
                    Some(t) => t,
                    None => return false,
                };
                let cased = if op == "upcase_atom/2" { text.to_uppercase() } else { text.to_lowercase() };
                let a2 = self.get_reg_raw("A2").unwrap_or(Value::Uninit);
                if self.unify(&a2, &Value::Atom(cased)) { self.pc += 1; true } else { false }
            }
            "atom_number/2" => {
                let v1 = self.get_reg_raw("A1").map(|v| self.deref_var(&v)).unwrap_or(Value::Uninit);
                match &v1 {
                    Value::Atom(s) => {
                        let num = if let Ok(n) = s.parse::<i64>() {
                            Value::Integer(n)
                        } else if let Ok(f) = s.parse::<f64>() {
                            Value::Float(f)
                        } else {
                            return false;
                        };
                        let a2 = self.get_reg_raw("A2").unwrap_or(Value::Uninit);
                        if self.unify(&a2, &num) { self.pc += 1; true } else { false }
                    }
                    _ => {
                        let text = match self.get_reg_raw("A2").map(|v| self.deref_var(&v)) {
                            Some(Value::Integer(n)) => n.to_string(),
                            Some(Value::Float(f)) => format!("{}", f),
                            _ => return false,
                        };
                        if self.unify(&v1, &Value::Atom(text)) { self.pc += 1; true } else { false }
                    }
                }
            }
            "atomic_list_concat/2" => {
                let items = match self.get_reg_raw("A1").map(|v| self.deref_heap(&self.deref_var(&v))) {
                    Some(Value::List(items)) => items,
                    _ => return false,
                };
                let mut text = String::new();
                for item in &items {
                    match Self::value_atomic_text(&self.deref_var(item)) {
                        Some(t) => text.push_str(&t),
                        None => return false,
                    }
                }
                let a2 = self.get_reg_raw("A2").unwrap_or(Value::Uninit);
                if self.unify(&a2, &Value::Atom(text)) { self.pc += 1; true } else { false }
            }
            "atomic_list_concat/3" => {
                // Join mode (+List, +Sep, ?Atom) or split mode
                // (?List, +Sep nonempty, +Atom).
                let sep = match self.get_reg_raw("A2").map(|v| self.deref_var(&v))
                    .as_ref().and_then(Self::value_atomic_text) {
                    Some(s) => s,
                    None => return false,
                };
                let v1 = self.get_reg_raw("A1").map(|v| self.deref_heap(&self.deref_var(&v))).unwrap_or(Value::Uninit);
                match v1 {
                    Value::List(items) => {
                        let mut parts: Vec<String> = Vec::with_capacity(items.len());
                        for item in &items {
                            match Self::value_atomic_text(&self.deref_var(item)) {
                                Some(t) => parts.push(t),
                                None => return false,
                            }
                        }
                        let joined = Value::Atom(parts.join(&sep));
                        let a3 = self.get_reg_raw("A3").unwrap_or(Value::Uninit);
                        if self.unify(&a3, &joined) { self.pc += 1; true } else { false }
                    }
                    Value::Unbound(_) => {
                        if sep.is_empty() { return false; }
                        let whole = match self.get_reg_raw("A3").map(|v| self.deref_var(&v))
                            .as_ref().and_then(Self::value_atomic_text) {
                            Some(t) => t,
                            None => return false,
                        };
                        let parts: Vec<Value> = whole.split(&sep as &str)
                            .map(|p| Value::Atom(p.to_string()))
                            .collect();
                        let a1 = self.get_reg_raw("A1").unwrap_or(Value::Uninit);
                        if self.unify(&a1, &Value::List(parts)) { self.pc += 1; true } else { false }
                    }
                    _ => false,
                }
            }
            "char_type/2" => {
                // +Char mode only; the common type terms. Parameterized
                // forms unify their argument (digit weight, case pairs).
                let ch = match self.get_reg_raw("A1").map(|v| self.deref_var(&v)) {
                    Some(Value::Atom(s)) if s.chars().count() == 1 => s.chars().next().unwrap(),
                    _ => return false,
                };
                let ty = self.get_reg_raw("A2").map(|v| self.deref_heap(&self.deref_var(&v))).unwrap_or(Value::Uninit);
                match &ty {
                    Value::Atom(t) => {
                        let ok = match t.as_str() {
                            "alpha" => ch.is_alphabetic(),
                            "alnum" => ch.is_alphanumeric(),
                            "csym" => ch.is_alphanumeric() || ch == ''_'',
                            "csymf" => ch.is_alphabetic() || ch == ''_'',
                            "space" | "white" => ch.is_whitespace(),
                            "punct" => ch.is_ascii_punctuation(),
                            "graph" => !ch.is_whitespace() && !ch.is_control(),
                            "ascii" => ch.is_ascii(),
                            "upper" => ch.is_uppercase(),
                            "lower" => ch.is_lowercase(),
                            "end_of_line" => (ch as u32) == 10 || (ch as u32) == 13,
                            "newline" => (ch as u32) == 10,
                            _ => return false,
                        };
                        if ok { self.pc += 1; true } else { false }
                    }
                    Value::Str(f, args) if args.len() == 1 => {
                        let name = Self::display_functor_name(f, 1);
                        let arg = args[0].clone();
                        match name.as_str() {
                            "digit" => {
                                match ch.to_digit(10) {
                                    Some(w) => {
                                        if self.unify(&arg, &Value::Integer(w as i64)) {
                                            self.pc += 1; true
                                        } else { false }
                                    }
                                    None => false,
                                }
                            }
                            "to_lower" => {
                                let lo = ch.to_lowercase().next().unwrap_or(ch);
                                if self.unify(&arg, &Value::Atom(lo.to_string())) { self.pc += 1; true } else { false }
                            }
                            "to_upper" => {
                                let up = ch.to_uppercase().next().unwrap_or(ch);
                                if self.unify(&arg, &Value::Atom(up.to_string())) { self.pc += 1; true } else { false }
                            }
                            "upper" => {
                                if !ch.is_uppercase() { return false; }
                                let lo = ch.to_lowercase().next().unwrap_or(ch);
                                if self.unify(&arg, &Value::Atom(lo.to_string())) { self.pc += 1; true } else { false }
                            }
                            "lower" => {
                                if !ch.is_lowercase() { return false; }
                                let up = ch.to_uppercase().next().unwrap_or(ch);
                                if self.unify(&arg, &Value::Atom(up.to_string())) { self.pc += 1; true } else { false }
                            }
                            _ => false,
                        }
                    }
                    _ => false,
                }
            }
            "ground/1" => {
                let v = self.get_reg_raw("A1").unwrap_or(Value::Uninit);
                if self.value_is_ground(&v) { self.pc += 1; true } else { false }
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
            "throw/1" => {
                // ISO throw/1: deep-deref the ball and put it in flight.
                // The run loop sees the failing step with a pending ball
                // and aborts instead of backtracking; the nearest catch/3
                // whose catcher unifies consumes it.
                let ball = self.get_reg_raw("A1")
                    .map(|v| self.deref_heap(&self.deref_var(&v)))
                    .unwrap_or(Value::Atom("instantiation_error".to_string()));
                self.thrown_ball = Some(ball);
                false
            }
            "catch/3" => {
                // ISO catch/3 (first-solution semantics, like the F# port):
                // snapshot, meta-call the goal; on a thrown ball restore
                // the snapshot, unify the catcher, and run recovery —
                // rethrowing when the catcher does not unify. Mirrors the
                // F# CatcherFrame shape with mutable-state restoration in
                // place of immutable snapshots.
                let catch_pc = self.pc;
                let goal = self.get_reg_raw("A1")
                    .map(|v| self.deref_heap(&self.deref_var(&v)))
                    .unwrap_or(Value::Uninit);
                let catcher_raw = self.get_reg_raw("A2").unwrap_or(Value::Uninit);
                let recovery_raw = self.get_reg_raw("A3").unwrap_or(Value::Uninit);
                let saved_regs = self.save_regs();
                let trail_mark = self.trail.len();
                let heap_mark = self.heap.len();
                let stack_snapshot = self.stack.clone();
                let cp_depth = self.choice_points.len();
                let saved_cp = self.cp;
                let saved_cut = self.cut_barrier;
                if self.call_goal_value(&goal) {
                    // Goal succeeded: commit to its first solution
                    // (bindings carry forward) and advance past catch/3.
                    self.choice_points.truncate(cp_depth);
                    self.cp = saved_cp;
                    self.cut_barrier = saved_cut;
                    self.pc = catch_pc + 1;
                    return true;
                }
                match self.thrown_ball.take() {
                    None => {
                        // Plain goal failure: catch/3 fails. Restore the
                        // pre-goal view so partial bindings do not leak.
                        if self.choice_points.len() > cp_depth {
                            self.choice_points.truncate(cp_depth);
                        }
                        self.unwind_trail_to(trail_mark);
                        self.heap.truncate(heap_mark);
                        self.stack = stack_snapshot;
                        self.restore_regs(&saved_regs);
                        self.cp = saved_cp;
                        self.cut_barrier = saved_cut;
                        false
                    }
                    Some(ball) => {
                        if self.choice_points.len() > cp_depth {
                            self.choice_points.truncate(cp_depth);
                        }
                        self.unwind_trail_to(trail_mark);
                        self.heap.truncate(heap_mark);
                        self.stack = stack_snapshot;
                        self.restore_regs(&saved_regs);
                        self.cp = saved_cp;
                        self.cut_barrier = saved_cut;
                        let mark2 = self.trail.len();
                        if self.unify(&catcher_raw, &ball) {
                            let recovery = self.deref_heap(&self.deref_var(&recovery_raw));
                            if self.call_goal_value(&recovery) {
                                self.pc = catch_pc + 1;
                                true
                            } else { false }
                        } else {
                            // Catcher does not unify: rethrow for an
                            // outer catch frame.
                            self.unwind_trail_to(mark2);
                            self.thrown_ball = Some(ball);
                            false
                        }
                    }
                }
            }
            "maplist/2" | "maplist/3" | "maplist/4" | "maplist/5" => {
                // maplist(Goal, L1, ..., Lk): call Goal extended with the
                // i-th element of every list, for every i. Unbound list
                // arguments are unified with fresh-variable lists of the
                // length determined by the first bound list. Each element
                // call commits to its first solution.
                let nlists = _arity - 1;
                let goal = self.get_reg_raw("A1")
                    .map(|v| self.deref_heap(&self.deref_var(&v)))
                    .unwrap_or(Value::Uninit);
                let raw_lists: Vec<Value> = (0..nlists)
                    .map(|j| self.get_reg_raw(&format!("A{}", j + 2)).unwrap_or(Value::Uninit))
                    .collect();
                let mut elems: Vec<Option<Vec<Value>>> = Vec::with_capacity(nlists);
                let mut n: Option<usize> = None;
                for raw in &raw_lists {
                    match self.deref_heap(&self.deref_var(raw)) {
                        Value::List(items) => {
                            match n {
                                Some(len) if len != items.len() => return false,
                                _ => n = Some(items.len()),
                            }
                            elems.push(Some(items));
                        }
                        Value::Unbound(_) => elems.push(None),
                        _ => return false,
                    }
                }
                let n = match n {
                    Some(n) => n,
                    None => return false, // all lists unbound: unbounded
                };
                for j in 0..nlists {
                    if elems[j].is_none() {
                        let fresh: Vec<Value> = (0..n).map(|_| self.fresh_meta_var()).collect();
                        if !self.unify(&raw_lists[j], &Value::List(fresh.clone())) {
                            return false;
                        }
                        elems[j] = Some(fresh);
                    }
                }
                for i in 0..n {
                    let extra: Vec<Value> = (0..nlists)
                        .map(|j| self.deref_var(&elems[j].as_ref().unwrap()[i]))
                        .collect();
                    let g = match self.extend_goal(&goal, &extra) {
                        Some(g) => g,
                        None => return false,
                    };
                    if !self.call_goal_once(&g) { return false; }
                }
                self.pc += 1; true
            }
            "include/3" | "exclude/3" => {
                // Filter: keep elements for which the test call succeeds
                // (include) or fails (exclude). Test-call bindings are
                // trial-only and unwound after each element.
                let goal = self.get_reg_raw("A1")
                    .map(|v| self.deref_heap(&self.deref_var(&v)))
                    .unwrap_or(Value::Uninit);
                let items = match self.get_reg_raw("A2").map(|v| self.deref_heap(&self.deref_var(&v))) {
                    Some(Value::List(items)) => items,
                    _ => return false,
                };
                let keep_on = op == "include/3";
                let mut kept: Vec<Value> = Vec::new();
                for item in &items {
                    let g = match self.extend_goal(&goal, &[self.deref_var(item)]) {
                        Some(g) => g,
                        None => return false,
                    };
                    let mark = self.trail.len();
                    let ok = self.call_goal_once(&g);
                    self.unwind_trail_to(mark);
                    if !ok && self.thrown_ball.is_some() { return false; }
                    if ok == keep_on { kept.push(item.clone()); }
                }
                let a3 = self.get_reg_raw("A3").unwrap_or(Value::Uninit);
                if self.unify(&a3, &Value::List(kept)) { self.pc += 1; true } else { false }
            }
            "partition/4" => {
                let goal = self.get_reg_raw("A1")
                    .map(|v| self.deref_heap(&self.deref_var(&v)))
                    .unwrap_or(Value::Uninit);
                let items = match self.get_reg_raw("A2").map(|v| self.deref_heap(&self.deref_var(&v))) {
                    Some(Value::List(items)) => items,
                    _ => return false,
                };
                let mut incl: Vec<Value> = Vec::new();
                let mut excl: Vec<Value> = Vec::new();
                for item in &items {
                    let g = match self.extend_goal(&goal, &[self.deref_var(item)]) {
                        Some(g) => g,
                        None => return false,
                    };
                    let mark = self.trail.len();
                    let ok = self.call_goal_once(&g);
                    self.unwind_trail_to(mark);
                    if !ok && self.thrown_ball.is_some() { return false; }
                    if ok { incl.push(item.clone()); } else { excl.push(item.clone()); }
                }
                let a3 = self.get_reg_raw("A3").unwrap_or(Value::Uninit);
                let a4 = self.get_reg_raw("A4").unwrap_or(Value::Uninit);
                if self.unify(&a3, &Value::List(incl)) && self.unify(&a4, &Value::List(excl)) {
                    self.pc += 1; true
                } else { false }
            }
            "foldl/4" | "foldl/5" => {
                // foldl(Goal, L1[, L2], V0, V): thread an accumulator
                // through per-element calls Goal(X1[, X2], Acc0, Acc1).
                let two_lists = op == "foldl/5";
                let goal = self.get_reg_raw("A1")
                    .map(|v| self.deref_heap(&self.deref_var(&v)))
                    .unwrap_or(Value::Uninit);
                let l1 = match self.get_reg_raw("A2").map(|v| self.deref_heap(&self.deref_var(&v))) {
                    Some(Value::List(items)) => items,
                    _ => return false,
                };
                let l2: Option<Vec<Value>> = if two_lists {
                    match self.get_reg_raw("A3").map(|v| self.deref_heap(&self.deref_var(&v))) {
                        Some(Value::List(items)) if items.len() == l1.len() => Some(items),
                        _ => return false,
                    }
                } else { None };
                let acc0_reg = if two_lists { "A4" } else { "A3" };
                let out_reg = if two_lists { "A5" } else { "A4" };
                let mut acc = self.get_reg_raw(acc0_reg)
                    .map(|v| self.deref_var(&v))
                    .unwrap_or(Value::Uninit);
                for i in 0..l1.len() {
                    let next = self.fresh_meta_var();
                    let mut extra: Vec<Value> = vec![self.deref_var(&l1[i])];
                    if let Some(ref l2v) = l2 {
                        extra.push(self.deref_var(&l2v[i]));
                    }
                    extra.push(acc.clone());
                    extra.push(next.clone());
                    let g = match self.extend_goal(&goal, &extra) {
                        Some(g) => g,
                        None => return false,
                    };
                    if !self.call_goal_once(&g) { return false; }
                    acc = self.deref_var(&next);
                }
                let out = self.get_reg_raw(out_reg).unwrap_or(Value::Uninit);
                if self.unify(&out, &acc) { self.pc += 1; true } else { false }
            }
            _ => false,
        }
    }

    /// Append arguments to a callable (call/N semantics for the
    /// maplist/include/foldl family). Atoms become compounds; compound
    /// goals get the extra arguments appended after their own.
    fn extend_goal(&self, base: &Value, extra: &[Value]) -> Option<Value> {
        match base {
            Value::Atom(name) => Some(Value::Str(name.clone(), extra.to_vec())),
            Value::Str(f, args) => {
                let name = Self::display_functor_name(f, args.len());
                let mut all = args.clone();
                all.extend_from_slice(extra);
                Some(Value::Str(name, all))
            }
            _ => None,
        }
    }

    fn fresh_meta_var(&mut self) -> Value {
        self.var_counter += 1;
        Value::Unbound(format!("_M{}", self.var_counter))
    }

    /// First-solution meta-call used by the maplist family: any choice
    /// points the sub-call leaves behind are discarded (deterministic
    /// commit per element).
    fn call_goal_once(&mut self, goal: &Value) -> bool {
        let cp_depth = self.choice_points.len();
        let ok = self.call_goal_value(goal);
        if self.choice_points.len() > cp_depth {
            self.choice_points.truncate(cp_depth);
        }
        ok
    }

    /// Predicates the shared WAM compiler emits as Call/Execute (no
    /// is_builtin_pred entry) but that this runtime implements as
    /// builtins. Mirrors the F# isIsoMetaBuiltin routing.
    fn is_iso_meta_builtin(pred: &str) -> bool {
        matches!(pred, "catch/3" | "throw/1" | "succ/2")
    }

    /// Meta-call a goal VALUE (catch/3 goal and recovery, callable
    /// terms). Atoms true/fail are inlined; other goals dispatch to the
    /// builtin table first, then to a labelled predicate via a sub-run
    /// (the same architecture as the general negation path). Returns
    /// the first solution; on a thrown ball it returns false with the
    /// ball left in flight for the caller to inspect.
    fn call_goal_value(&mut self, goal: &Value) -> bool {
        match goal {
            Value::Atom(name) if name == "true" => true,
            Value::Atom(name) if name == "fail" || name == "false" => false,
            Value::Atom(name) => {
                let key = format!("{}/0", name);
                self.call_goal_key(&key, 0, &[])
            }
            Value::Str(functor, args) => {
                let arity = args.len();
                let name = Self::display_functor_name(functor, arity);
                let key = format!("{}/{}", name, arity);
                let dargs: Vec<Value> = args.iter().map(|a| self.deref_var(a)).collect();
                self.call_goal_key(&key, arity, &dargs)
            }
            _ => false,
        }
    }

    fn call_goal_key(&mut self, key: &str, arity: usize, args: &[Value]) -> bool {
        for (i, arg) in args.iter().enumerate() {
            self.set_reg(&format!("A{}", i + 1), arg.clone());
        }
        let saved_pc = self.pc;
        if key == "retract/1" {
            return self.dynamic_retract_call(saved_pc);
        }
        if key == "clause/2" {
            return self.dynamic_clause_call(saved_pc);
        }
        if key == "current_predicate/1" {
            return self.current_predicate_call(saved_pc);
        }
        if self.execute_builtin(key, arity) {
            self.pc = saved_pc;
            return true;
        }
        if self.thrown_ball.is_some() {
            return false;
        }
        if let Some(&target_pc) = self.labels.get(key) {
            let saved_cp = self.cp;
            self.pc = target_pc;
            self.cp = 0; // halt sentinel for the sub-run
            let ok = self.run();
            self.cp = saved_cp;
            self.pc = saved_pc;
            return ok;
        }
        if self.dynamic_call(key, saved_pc) {
            self.pc = saved_pc;
            return true;
        }
        false
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
            Value::Unbound(_) => {
                // A bound placeholder variable (e.g. the embedded arg of a
                // nested expression +(+(A,B),C)) derefs through the binding
                // table to the real subterm. Without this it fell to None and
                // any depth>=2 arithmetic failed.
                let d = self.deref_var(expr);
                if d == *expr { None } else { self.eval_arith(&d) }
            }
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
    % Foreign-lowered kernel wrappers keep the bare predicate name: they are
    % the project's public entry points (imported as `use crate::pred;` by
    % consumers, e.g. tests/test_wam_rust_runtime.pl's integration test).
    % Generic WAM wrappers keep the arity-suffixed name so same-name
    % predicates of different arities (read_term_from_atom/2 vs /3 in
    % runtime-parser mode) don't collide.
    (   option(foreign_lowering(ForeignSpecForName), Options),
        rust_foreign_spec(Pred/Arity, ForeignSpecForName, _, _)
    ->  rust_safe_function_name(Pred, FuncName)
    ;   rust_safe_function_name(Pred/Arity, FuncName)
    ),
    build_rust_wam_arg_list(Arity, ArgList),
    build_rust_wam_arg_setup(Arity, ArgSetup),
    foreign_wrapper_setup(Pred/Arity, WamCode, Options, InstrSetup, ForeignSetup, RunExpr),
    % T7: consult the compile-time parallel-aggregate gate (cost machinery).
    % Gated behind parallel_aggregates(true); otherwise AggAnno = '' and output
    % is byte-identical to before.
    rust_predicate_parallel_decision(Pred/Arity, Options, ParDecision),
    rust_parallel_annotation(ParDecision, AggAnno),
    format(string(RustCode),
'/// WAM-compiled predicate: ~w/~w
/// Compiled via WAM for predicates that resist native lowering.
~wpub fn ~w(~w) -> bool {
    use std::collections::HashMap;
~w
~w
~w
    ~w
}', [PredStr, Arity, AggAnno, FuncName, ArgList, InstrSetup, ForeignSetup, ArgSetup, RunExpr]).

% --- T7 compile-time parallel-aggregate gate --------------------------------
%
% First codegen consumer of the cost machinery: decide, per predicate, whether
% any forkable aggregate (findall/aggregate_all/bagof/setof) in its clause
% bodies is worth parallelising, and annotate the generated function so the
% phase-2 runtime can dispatch on it. Decision-only here — no semantics change.
% Feature-gated by parallel_aggregates(true) (default off → no output change).
% Fully defensive: any failure (module not loaded, no source, no aggregate)
% degrades to `sequential` and emits nothing.

rust_predicate_parallel_decision(Pred/Arity, Options, Decision) :-
    (   option(parallel_aggregates(true), Options),
        catch(rust_pred_par_decision(Pred/Arity, Options, D), _, fail)
    ->  Decision = D
    ;   Decision = sequential
    ).

rust_pred_par_decision(Pred/Arity, Options, Decision) :-
    option(module_name(Mod0), Options, user),
    ( atom(Mod0) -> Module = Mod0 ; Module = user ),
    functor(Head, Pred, Arity),
    findall(Body, clause(Module:Head, Body), Bodies),
    Bodies \== [],
    collect_forkable_generators(Bodies, Gens),
    Gens \== [],
    build_cost_model(Module, Model),
    (   member(Gen, Gens),
        goal_parallel_decision(Gen, Model, parallel)
    ->  goal_cost_tier(Gen, Model, Tier),
        Decision = parallel(Tier)
    ;   Decision = sequential
    ).

% Collect the generator goals of every forkable aggregate appearing anywhere in
% the given clause bodies.
collect_forkable_generators(Bodies, Gens) :-
    findall(Gen,
            ( member(B, Bodies),
              body_subgoal(B, G),
              nonvar(G),
              forkable_aggregate(G, _Tmpl, Gen) ),
            Gens0),
    sort(Gens0, Gens).

% Enumerate sub-goals of a body, descending through control constructs.
body_subgoal(G, _) :- var(G), !, fail.
body_subgoal((A, B), S) :- !, ( body_subgoal(A, S) ; body_subgoal(B, S) ).
body_subgoal((A ; B), S) :- !, ( body_subgoal(A, S) ; body_subgoal(B, S) ).
body_subgoal((A -> B), S) :- !, ( body_subgoal(A, S) ; body_subgoal(B, S) ).
body_subgoal((A *-> B), S) :- !, ( body_subgoal(A, S) ; body_subgoal(B, S) ).
body_subgoal(\+ A, S) :- !, body_subgoal(A, S).
body_subgoal(G, G).

rust_parallel_annotation(parallel(Tier), Anno) :- !,
    format(atom(Anno),
           '/// T7: parallel-eligible aggregate (cost gate tier: ~w)~n', [Tier]).
rust_parallel_annotation(_, '').

% --- T7 route-1 injection: native par_collect wrapper -----------------------
%
% When a predicate's single clause body IS a parallel-eligible forkable
% aggregate, emit it as a native Rust function that calls the runtime
% orchestrator par_collect over two synthesised helper predicates
% (__par_enum_*/1, __par_body_*/2) and reduces by aggregate type. The helpers
% are returned to the caller to add to the project's compile set (they become
% ordinary WAM predicate functions). Fails (predicate compiled normally) unless
% the clause is a single forkable aggregate that splits and the gate says
% parallel. Gated by parallel_aggregates(true).

rust_parallel_aggregate_wrapper(QPI, Options, Helpers, WrapperRust) :-
    option(parallel_aggregates(true), Options),
    ( QPI = Module:Pred/Arity -> true ; QPI = Pred/Arity, Module = user ),
    functor(Head, Pred, Arity),
    findall(t, clause(Module:Head, _), Marks), Marks = [t],  % exactly one clause
    clause(Module:Head, Body),                                % bind Head + Body
    forkable_aggregate(Body, _Tmpl, InnerGoal),
    Head =.. [Pred|Args],
    build_cost_model(Module, CM),
    aggregate_result(Body, Result0),
    % External inputs = head args the aggregate's inner goal reads (minus result):
    % they must be bound when the enum/body helpers run.
    rust_external_inputs(InnerGoal, Args, Result0, ExternalInputs),
    parallel_aggregate_transform(Body, ExternalInputs, CM, Pred, Helpers,
        par_aggregate(AggType, EnumName/EnumArity, BodyName/BodyArity, Result)),
    nth1(ResultIdx, Args, ResultArg), ResultArg == Result, !,
    rust_safe_function_name(Pred/Arity, FName),
    rust_safe_function_name(EnumName/EnumArity, EnumFn),
    rust_safe_function_name(BodyName/BodyArity, BodyFn),
    build_rust_wam_arg_list(Arity, ArgList),
    rust_agg_reduce(AggType, ReduceExpr),
    % Closure call-args: external-input clones (by head-arg position) then the
    % tuple var (enum) / tuple var + value var (body).
    rust_ext_clone_args(ExternalInputs, Args, CloneList),
    append(CloneList, ['__in'], EnumCA), atomic_list_concat(EnumCA, ', ', EnumCallArgs),
    append(CloneList, ['__in', '__v'], BodyCA), atomic_list_concat(BodyCA, ', ', BodyCallArgs),
    format(string(WrapperRust),
'/// T7 parallel aggregate (route 1): ~w/~w
pub fn ~w(~w) -> bool {
    let __base = vm.clone();
    let __vals = crate::par_aggregate::par_collect(&__base,
        |__m, __in| crate::~w(__m, ~w),
        |__m, __in, __v| crate::~w(__m, ~w));
    let __res = ~w;
    vm.unify(&a~w, &__res)
}', [Pred, Arity, FName, ArgList, EnumFn, EnumCallArgs, BodyFn, BodyCallArgs, ReduceExpr, ResultIdx]).

% External inputs: distinct head-arg variables the inner goal reads, excluding
% the result var. (Vars the aggregate enumerator/body need bound from the caller.)
rust_external_inputs(InnerGoal, Args, Result, ExternalInputs) :-
    term_variables(InnerGoal, IGVars),
    % include/3 (not findall) so the collected vars stay IDENTICAL to the head
    % args — rust_ext_clone_args/3 then finds each by ==.
    include(rust_is_ext_input(IGVars, Result), Args, Raw),
    rust_dedup_vars(Raw, ExternalInputs).

rust_is_ext_input(IGVars, Result, A) :-
    var(A), A \== Result, rust_var_member(IGVars, A).

rust_var_member([V|_], A) :- V == A, !.
rust_var_member([_|T], A) :- rust_var_member(T, A).

rust_dedup_vars([], []).
rust_dedup_vars([V|Vs], [V|R]) :- exclude(==(V), Vs, Vs1), rust_dedup_vars(Vs1, R).

% Map each external-input var to "a<HeadIndex>.clone()".
rust_ext_clone_args([], _, []).
rust_ext_clone_args([V|Vs], Args, [Clone|Rest]) :-
    nth1(Idx, Args, A), A == V, !,
    format(atom(Clone), 'a~w.clone()', [Idx]),
    rust_ext_clone_args(Vs, Args, Rest).

% Reduce the collected per-branch values by aggregate type (mirrors the
% aggregate_frame finalisation in the interpreter).
rust_agg_reduce(collect, "Value::List(__vals)") :- !.
rust_agg_reduce(count, "Value::Integer(__vals.len() as i64)") :- !.
rust_agg_reduce(sum, Expr) :- !,
    Expr = "{ let mut si: i64 = 0; let mut sf: f64 = 0.0; let mut isf = false; for v in &__vals { match v { Value::Integer(n) => { si += *n; sf += *n as f64; }, Value::Float(f) => { isf = true; sf += *f; }, _ => {} } } if isf { Value::Float(sf) } else { Value::Integer(si) } }".
rust_agg_reduce(max, Expr) :- !, rust_agg_minmax_reduce(">", Expr).
rust_agg_reduce(min, Expr) :- !, rust_agg_minmax_reduce("<", Expr).
% set: sorted, duplicate-free (standard order of terms via the runtime's
% term_compare; dedup adjacent equals after sort).
rust_agg_reduce(set, Expr) :- !,
    Expr = "{ let mut __s = __vals; __s.sort_by(|a, b| vm.term_compare(a, b)); __s.dedup_by(|a, b| vm.term_compare(a, b) == std::cmp::Ordering::Equal); Value::List(__s) }".

% max/min share a fold; Cmp is the Rust comparison operator (">" or "<").
% Mirrors the interpreter's aggregate_frame max/min (Integer/Float mixed).
rust_agg_minmax_reduce(Cmp, Expr) :-
    format(string(Expr),
"{ let mut __best: Option<Value> = None; for v in &__vals { let __take = match &__best { None => true, Some(p) => match (v, p) { (Value::Integer(a), Value::Integer(b)) => a ~w b, (Value::Float(a), Value::Float(b)) => a ~w b, (Value::Integer(a), Value::Float(b)) => (*a as f64) ~w *b, (Value::Float(a), Value::Integer(b)) => *a ~w (*b as f64), _ => false } }; if __take { __best = Some(v.clone()); } } __best.unwrap_or(Value::List(vec![])) }",
           [Cmp, Cmp, Cmp, Cmp]).

% Partition the project predicate list: predicates whose body is a
% parallel-eligible aggregate are replaced by (a) their synthesised enum/body
% helper predicates (compiled normally) and (b) a native par_collect wrapper
% (returned in Wrappers, appended to the project code). All others pass through.
rust_inject_parallel_aggregates(Preds0, Options, Preds, Wrappers) :-
    (   option(parallel_aggregates(true), Options)
    ->  rust_partition_par(Preds0, Options, Kept, HelperPIs, Wrappers),
        append(Kept, HelperPIs, Preds)
    ;   Preds = Preds0, Wrappers = []
    ).

rust_partition_par([], _, [], [], []).
rust_partition_par([PI|Rest], Options, Kept, HelperPIs, Wrappers) :-
    (   catch(rust_parallel_aggregate_wrapper(PI, Options, Helpers, Wrapper), _, fail)
    ->  rust_pi_module(PI, Module),
        maplist(rust_assert_helper(Module), Helpers, MyHelperPIs),
        rust_partition_par(Rest, Options, Kept, RestHelperPIs, RestWrappers),
        append(MyHelperPIs, RestHelperPIs, HelperPIs),
        Wrappers = [Wrapper|RestWrappers]
    ;   rust_partition_par(Rest, Options, Kept0, HelperPIs, Wrappers),
        Kept = [PI|Kept0]
    ).

rust_pi_module(Module:_, Module) :- !.
rust_pi_module(_, user).

% Assert one synthesised helper clause into its source module (replacing any
% prior version) and return its module-qualified PI for the compile set.
rust_assert_helper(Module, (Head :- Body), Module:Name/Arity) :-
    functor(Head, Name, Arity),
    functor(Clean, Name, Arity),
    retractall(Module:Clean),
    copy_term((Head :- Body), (H :- B)),
    assertz(Module:(H :- B)).

% --- T7 route 2: aggregates EMBEDDED in a larger clause body ----------------
%
% Unlike route 1 (whole-body aggregate -> native par_collect wrapper), an
% embedded aggregate must run mid-clause inside the WAM. This pass keeps the
% containing predicate as a WAM predicate but (a) synthesises the same
% enum/body helpers as route 1 (added to the compile set so they get entry
% labels) and (b) records a rewrite so the containing predicate's
% begin_aggregate..end_aggregate block is replaced by a single par_aggregate
% instruction referencing those helper labels. Gated by parallel_aggregates(true)
% and the same cost gate as route 1 (only expensive/recursive inner bodies).

rust_inject_embedded_par_aggregates(Preds0, Options, Preds, Rewrites) :-
    (   option(parallel_aggregates(true), Options)
    ->  rust_partition_embedded(Preds0, Options, HelperPIs, Rewrites),
        append(Preds0, HelperPIs, Preds)
    ;   Preds = Preds0, Rewrites = []
    ).

rust_partition_embedded([], _, [], []).
rust_partition_embedded([PI|Rest], Options, HelperPIs, Rewrites) :-
    (   catch(rust_embedded_par_aggregate(PI, Options, MyHelpers, Rewrite), _, fail)
    ->  rust_pi_module(PI, Module),
        maplist(rust_assert_helper(Module), MyHelpers, MyHelperPIs),
        rust_partition_embedded(Rest, Options, RestHelperPIs, RestRewrites),
        append(MyHelperPIs, RestHelperPIs, HelperPIs),
        Rewrites = [Rewrite|RestRewrites]
    ;   rust_partition_embedded(Rest, Options, HelperPIs, Rewrites)
    ).

% Succeeds for a single-clause predicate whose body contains (but is not
% itself) a forkable aggregate that the cost gate sends parallel and whose
% reduce type the par_aggregate handler supports.
rust_embedded_par_aggregate(QPI, _Options, Helpers,
        rewrite(Pred/Arity, EnumName, BodyName, EnumArity)) :-
    ( QPI = Module:Pred/Arity -> true ; QPI = Pred/Arity, Module = user ),
    functor(Head, Pred, Arity),
    findall(t, clause(Module:Head, _), [t]),     % exactly one clause
    clause(Module:Head, Body),
    \+ forkable_aggregate(Body, _, _),           % not whole-body (that is route 1)
    rust_body_embedded_aggregate(Body, AggGoal),
    % External inputs: variables the aggregate's inner goal reads that the
    % enclosing clause binds (head args / preceding goals). They become LEADING
    % params of the __par_enum/__par_body helpers (via the /6 transform) and, at
    % the WAM-text splice, their container registers are recorded on the
    % par_aggregate line so the ParAggregate handler captures their values and
    % threads them in — so e.g. eg_p(1,R) enumerates only link(1), not every link.
    % Order is inner-goal first-appearance, matching the block's first-read order
    % the splice uses for the register list. (If the splice can't line the
    % registers up 1:1 with these inputs it leaves the aggregate sequential.)
    rust_embedded_aggregate_inputs(Head, Body, AggGoal, ExternalInputs),
    build_cost_model(Module, CM),
    parallel_aggregate_transform(AggGoal, ExternalInputs, CM, Pred, Helpers,
        par_aggregate(AggType, EnumName/EnumArity, BodyName/_BodyArity, _Result)),
    rust_supported_par_agg_type(AggType),
    !.

% External inputs of an embedded aggregate: variables its inner goal reads that
% are bound by the enclosing clause (head or the other body goals), excluding the
% result var, in inner-goal first-appearance order. These are threaded into the
% parallel helpers (see rust_embedded_par_aggregate); [] for an input-less
% embedded aggregate.
rust_embedded_aggregate_inputs(Head, Body, AggGoal, Ext) :-
    forkable_aggregate(AggGoal, _Tmpl, InnerGoal),
    aggregate_result(AggGoal, Result),
    rust_conj_list(Body, Goals),
    exclude(==(AggGoal), Goals, OtherGoals),
    term_variables([Head|OtherGoals], OuterVars),
    term_variables(InnerGoal, InnerVars),
    % NB: a yall lambda would copy OuterVars/Result per call (breaking var
    % identity); rust_is_ext_input/3 (defined above for route 1) is a named
    % helper so == compares the real clause variables. It is symmetric in its
    % first two list/var roles, so reusing it here is correct.
    include(rust_is_ext_input(OuterVars, Result), InnerVars, Ext0),
    rust_dedup_vars(Ext0, Ext).

rust_conj_list((A, B), L) :- !, rust_conj_list(A, LA), rust_conj_list(B, LB), append(LA, LB, L).
rust_conj_list(G, [G]).

% Find the first forkable aggregate goal inside a (possibly nested) conjunction.
rust_body_embedded_aggregate((A, B), Agg) :-
    !,
    (   forkable_aggregate(A, _, _)
    ->  Agg = A
    ;   rust_body_embedded_aggregate(B, Agg)
    ).
rust_body_embedded_aggregate(Goal, Goal) :-
    forkable_aggregate(Goal, _, _).

% Reduce types the ParAggregate handler implements (must match the sequential
% aggregate_frame finalisation so par == seq).
rust_supported_par_agg_type(T) :- memberchk(T, [collect, count, sum, max, min]).

% Rewrite the WAM text lines of a predicate flagged for embedded parallelism:
% splice a single `par_aggregate` line over its begin_aggregate..end_aggregate
% block. No-op (Lines unchanged) when there is no matching rewrite directive.
rust_rewrite_embedded_aggregate(Lines0, PI, Rewrites, Lines) :-
    memberchk(rewrite(PI, EnumName, BodyName, EnumArity), Rewrites),
    !,
    rust_splice_par_aggregate(Lines0, EnumName, BodyName, EnumArity, Lines).
rust_rewrite_embedded_aggregate(Lines, _, _, Lines).

% Splice a single par_aggregate line over the begin_aggregate..end_aggregate
% block. The external-input registers are recovered from the block: Y-registers
% READ inside it but NOT written there (so bound by the enclosing clause), minus
% the value and result registers, in first-read order. That order matches the
% helpers' leading-param order (inner-goal first-appearance), so register i feeds
% param i. The transform's K (= EnumArity-1) must equal the register count; if
% not, we leave the block sequential (correct fallback) rather than mis-thread.
rust_splice_par_aggregate(Lines0, EnumName, BodyName, EnumArity, Lines) :-
    rust_find_begin_agg(Lines0, Before, Type, ValueReg, ResultReg, AfterBegin),
    rust_take_to_end_agg(AfterBegin, BlockLines, AfterEnd),
    rust_block_input_regs(BlockLines, ValueReg, ResultReg, InputRegs),
    length(InputRegs, K),
    ExpectedK is EnumArity - 1,
    (   K =:= ExpectedK
    ->  BodyArity is EnumArity + 1,
        (   InputRegs == []
        ->  format(string(ParLine), '    par_aggregate ~w, ~w/~w, ~w/~w, ~w',
                   [Type, EnumName, EnumArity, BodyName, BodyArity, ResultReg])
        ;   atomic_list_concat(InputRegs, ', ', RegsStr),
            format(string(ParLine), '    par_aggregate ~w, ~w/~w, ~w/~w, ~w, ~w',
                   [Type, EnumName, EnumArity, BodyName, BodyArity, ResultReg, RegsStr])
        ),
        append(Before, [ParLine|AfterEnd], Lines)
    ;   Lines = Lines0                            % count mismatch -> stay sequential
    ).

rust_find_begin_agg([Line|Rest], [], Type, ValueReg, ResultReg, Rest) :-
    wam_tokenize_line(Line, Parts),
    Parts = ["begin_aggregate", Type0, ValueReg0, ResultReg0|_],
    clean_comma(Type0, Type),
    clean_comma(ValueReg0, ValueReg),
    clean_comma(ResultReg0, ResultReg),
    !.
rust_find_begin_agg([Line|Rest], [Line|Before], Type, ValueReg, ResultReg, After) :-
    rust_find_begin_agg(Rest, Before, Type, ValueReg, ResultReg, After).

% Capture the block lines between begin_aggregate and end_aggregate (exclusive),
% returning the lines after end_aggregate as the tail.
rust_take_to_end_agg([Line|Rest], [], Rest) :-
    wam_tokenize_line(Line, Parts),
    Parts = ["end_aggregate"|_],
    !.
rust_take_to_end_agg([Line|Rest], [Line|Block], After) :-
    rust_take_to_end_agg(Rest, Block, After).

% External-input registers of an aggregate block: Y-registers read but not
% written inside it (bound by the enclosing clause), minus the value/result
% registers, deduplicated in first-read order.
rust_block_input_regs(BlockLines, ValueReg, ResultReg, InputRegs) :-
    rust_block_rw(BlockLines, Reads, Writes),
    include(rust_not_in(Writes), Reads, R1),
    include(rust_not_in([ValueReg, ResultReg]), R1, R2),
    rust_string_dedup(R2, InputRegs).

rust_not_in(List, X) :- \+ memberchk(X, List).

% Reads (first-appearance order) and Writes of Y-registers across a block.
rust_block_rw([], [], []).
rust_block_rw([Line|Rest], Reads, Writes) :-
    wam_tokenize_line(Line, Parts0),
    maplist(clean_comma, Parts0, Parts),
    rust_line_rw(Parts, LReads, LWrites),
    rust_block_rw(Rest, Reads0, Writes0),
    append(LReads, Reads0, Reads),
    append(LWrites, Writes0, Writes).

% Y-register reads (source operand) / writes (destination operand) per instr.
rust_line_rw(["put_value", Src, _],     R, []) :- ( rust_is_yreg(Src) -> R = [Src] ; R = [] ), !.
rust_line_rw(["get_value", Src, _],     R, []) :- ( rust_is_yreg(Src) -> R = [Src] ; R = [] ), !.
rust_line_rw(["set_value", Src],        R, []) :- ( rust_is_yreg(Src) -> R = [Src] ; R = [] ), !.
rust_line_rw(["unify_value", Src],      R, []) :- ( rust_is_yreg(Src) -> R = [Src] ; R = [] ), !.
rust_line_rw(["put_variable", Dst, _],  [], W) :- ( rust_is_yreg(Dst) -> W = [Dst] ; W = [] ), !.
rust_line_rw(["get_variable", Dst, _],  [], W) :- ( rust_is_yreg(Dst) -> W = [Dst] ; W = [] ), !.
rust_line_rw(["set_variable", Dst],     [], W) :- ( rust_is_yreg(Dst) -> W = [Dst] ; W = [] ), !.
rust_line_rw(["unify_variable", Dst],   [], W) :- ( rust_is_yreg(Dst) -> W = [Dst] ; W = [] ), !.
rust_line_rw(_, [], []).

% A Y-register token: 'Y' followed by >= 1 digits (e.g. "Y2"). A-registers and
% structure/atom operands are not external-input slots.
rust_is_yreg(S) :-
    atom_string(A, S),
    atom_chars(A, ['Y'|Ds]),
    Ds = [_|_],
    forall(member(D, Ds), char_type(D, digit)).

rust_string_dedup([], []).
rust_string_dedup([X|Xs], [X|R]) :- exclude(==(X), Xs, Xs1), rust_string_dedup(Xs1, R).

% --- T9 fact-table inline: detection + row extraction ----------------------
%
% Recognise a predicate whose every clause is a ground unit clause (a fact) and
% extract its rows (one ground arg-tuple per clause, source order). When the row
% count is in the inline window [t9_min_rows, t9_max_rows] this is a native fact
% table + indexed lookup (T9), an optimisation over T4. Pure analysis — no
% codegen. Fails (predicate handled by the existing T4/WAM path) when any clause
% is a rule or non-ground, or when the row count is outside the window. Below
% t9_min_rows the T4 inline cost is negligible; above t9_max_rows inlining (T4 or
% T9) bloats compile time / the binary, so the fact set should come from an
% EXTERNAL source (LMDB/TSV) instead — see rust_maybe_warn_oversized_facts/2.

rust_fact_table_classify(QPI, Options, fact_info(Arity, Rows)) :-
    ( QPI = Module:Pred/Arity -> true ; QPI = Pred/Arity, Module = user ),
    functor(Head, Pred, Arity),
    findall(Head-Body, clause(Module:Head, Body), Clauses),
    Clauses = [_|_],                                  % at least one clause
    forall(member(CH-CB, Clauses), rust_fact_clause(CH, CB)),
    findall(Args, ( member(FH-true, Clauses), FH =.. [_|Args] ), Rows),
    rust_t9_min_rows(Options, Min),
    rust_t9_max_rows(Options, Max),
    length(Rows, NR),
    NR >= Min,
    NR =< Max.

% a fact clause: body `true`, head args all ground.
rust_fact_clause(Head, Body) :-
    Body == true,
    Head =.. [_|Args],
    forall(member(A, Args), ground(A)).

rust_t9_min_rows(Options, N) :-
    ( member(t9_min_rows(N), Options) -> true ; N = 64 ).

% Upper bound on inlining a fact predicate. Above this many rows, inlining (T9
% data table or T4 instructions) is declined and the fact set should be provided
% by an external source. Configurable via t9_max_rows.
rust_t9_max_rows(Options, N) :-
    ( member(t9_max_rows(N), Options) -> true ; N = 256 ).

% True when QPI is an all-ground-facts predicate whose row count exceeds the
% inline cap (so it is too large to inline). NR is its row count, Max the cap.
rust_fact_table_oversized(QPI, Options, NR, Max) :-
    ( QPI = Module:Pred/Arity -> true ; QPI = Pred/Arity, Module = user ),
    functor(Head, Pred, Arity),
    findall(Head-Body, clause(Module:Head, Body), Clauses),
    Clauses = [_|_],
    forall(member(CH-CB, Clauses), rust_fact_clause(CH, CB)),
    length(Clauses, NR),
    rust_t9_max_rows(Options, Max),
    NR > Max.

% Emit a one-line warning when a large ground-fact predicate is being inlined
% (T4) because it exceeds the T9 inline cap — recommending an external fact
% source. Silent when fact-table inlining is explicitly disabled. Always succeeds.
rust_maybe_warn_oversized_facts(QPI, Options) :-
    (   \+ option(fact_table_inline(false), Options),
        rust_fact_table_oversized(QPI, Options, NR, Max)
    ->  ( QPI = M:P/A -> true ; QPI = P/A, M = user ),
        format(user_error,
            '  ~w:~w/~w: ~w ground facts exceed t9_max_rows (~w) — not inlined; use an external fact source (e.g. LMDB/TSV)~n',
            [M, P, A, NR, Max])
    ;   true
    ).

% --- T9 fact-table inline: emission -----------------------------------------
%
% emit_fact_table_rust(+Pred/Arity, +fact_info(Arity,Rows), +Options, -RustCode):
% emit a lazily-built (OnceLock) row table + first-argument hash index, and a
% public enumeration fn. The fn dereferences its args; when the first arg is a
% bound atomic value it selects that index bucket (O(1) lookup), otherwise it
% scans all rows; it then drives crate fact_table_attempt, which unifies every
% column and leaves a choice point per remaining candidate so backtrack() yields
% the next matching row — same solution sequence/order as the T4/WAM path.
emit_fact_table_rust(Pred/Arity, fact_info(Arity, Rows), _Options, RustCode) :-
    Arity >= 1,
    rust_safe_function_name(Pred/Arity, FName),
    % row literals: vec![ vec![<col literals>], ... ] in source order
    maplist(rust_fact_row_literal, Rows, RowLiterals),
    atomic_list_concat(RowLiterals, ',\n            ', RowsBody),
    % public fn signature: (vm, a1: Value, .., aN: Value)
    numlist(1, Arity, Idxs),
    findall(AD, ( member(I, Idxs), format(atom(AD), 'a~w: Value', [I]) ), ArgDecls),
    atomic_list_concat(['vm: &mut WamState'|ArgDecls], ', ', SigArgs),
    findall(AN, ( member(I, Idxs), format(atom(AN), 'a~w', [I]) ), ArgNames),
    atomic_list_concat(ArgNames, ', ', ArgsVec),
    length(Rows, NRows),
    format(string(RustCode),
'#[allow(clippy::type_complexity)]
fn ~w_table() -> &''static (Vec<Vec<Value>>, std::collections::HashMap<String, Vec<usize>>) {
    static T: std::sync::OnceLock<(Vec<Vec<Value>>, std::collections::HashMap<String, Vec<usize>>)> = std::sync::OnceLock::new();
    T.get_or_init(|| {
        let rows: Vec<Vec<Value>> = vec![
            ~w
        ];
        let mut idx: std::collections::HashMap<String, Vec<usize>> = std::collections::HashMap::new();
        for (i, r) in rows.iter().enumerate() {
            if let Some(k) = r[0].fact_index_key() {
                idx.entry(k).or_default().push(i);
            }
        }
        (rows, idx)
    })
}

/// T9 fact table: ~w/~w (~w rows). First-arg hash index + choice-point enumeration.
/// cont_pc is the call-site continuation (pc+1 for `call`, saved cp for `execute`).
fn ~w_run(vm: &mut WamState, args: Vec<Value>, cont_pc: usize) -> bool {
    let __args: Vec<Value> = args.iter().map(|v| vm.deref_var(&vm.deref_heap(v))).collect();
    let (__rows, __idx) = ~w_table();
    // Bound atomic first arg -> that index bucket; otherwise full scan.
    let __cands: Vec<Value> = match __args[0].fact_index_key() {
        Some(k) => __idx.get(&k).map(|is| is.iter().map(|&i| Value::List(__rows[i].clone())).collect()).unwrap_or_default(),
        None => __rows.iter().map(|r| Value::List(r.clone())).collect(),
    };
    vm.fact_table_attempt(__args, __cands, cont_pc)
}

/// Public entry: direct callers get the next-instruction continuation (pc+1).
pub fn ~w(~w) -> bool {
    let __cont = vm.pc + 1;
    ~w_run(vm, vec![~w], __cont)
}',
        [FName, RowsBody, Pred, Arity, NRows, FName, FName, FName, SigArgs, FName, ArgsVec]),
    !.

% One row -> `vec![<col1>, <col2>, ...]`.
rust_fact_row_literal(Row, Literal) :-
    maplist(rust_term_to_value_literal, Row, ColLits),
    atomic_list_concat(ColLits, ', ', Inner),
    format(string(Literal), 'vec![~w]', [Inner]).

% A ground Prolog term -> a Rust `Value::...` literal (matches the runtime Value
% enum: Atom/Integer/Float/Str/List). [] maps to an empty Value::List.
rust_term_to_value_literal(T, Lit) :- integer(T), !,
    format(string(Lit), 'Value::Integer(~w)', [T]).
rust_term_to_value_literal(T, Lit) :- float(T), !,
    format(string(Lit), 'Value::Float(~w)', [T]).
rust_term_to_value_literal(T, Lit) :- is_list(T), !,
    maplist(rust_term_to_value_literal, T, Es),
    atomic_list_concat(Es, ', ', Inner),
    format(string(Lit), 'Value::List(vec![~w])', [Inner]).
rust_term_to_value_literal(T, Lit) :- atom(T), !,
    escape_rust_string(T, E),
    format(string(Lit), 'Value::Atom("~w".to_string())', [E]).
rust_term_to_value_literal(T, Lit) :- string(T), !,
    escape_rust_string(T, E),
    format(string(Lit), 'Value::Atom("~w".to_string())', [E]).
rust_term_to_value_literal(T, Lit) :- compound(T), !,
    T =.. [F|Args],
    escape_rust_string(F, EF),
    maplist(rust_term_to_value_literal, Args, Es),
    atomic_list_concat(Es, ', ', Inner),
    format(string(Lit), 'Value::Str("~w".to_string(), vec![~w])', [EF, Inner]).

% --- T7 embedded-aggregate wiring, step 1: pure clause-lifting pass ----------
%
% Apply lift_embedded_aggregate across a predicate's clauses, lifting every
% embedded parallel-eligible aggregate into a whole-body helper predicate and
% replacing it in-place with a call. Returns the rewritten clauses + the
% synthesised helper clauses. Pure/read-only: it reads clauses via clause/2 and
% builds terms; it does NOT assert/retract, so the user's module is untouched.
% Succeeds only if at least one aggregate was lifted (else the predicate is
% compiled by the existing path). Each lift gets a unique Seed (Pred_Arity_N).

rust_lift_predicate_clauses(QPI, Model, Rewritten, Helpers) :-
    ( QPI = Module:Pred/Arity -> true ; QPI = Pred/Arity, Module = user ),
    functor(Tmpl, Pred, Arity),
    findall((Tmpl :- B), clause(Module:Tmpl, B), Clauses),
    Clauses = [_|_],
    rust_lift_clauses(Clauses, Model, Pred, Arity, 0, Rewritten, Helpers, Changed),
    Changed == true.

rust_lift_clauses([], _, _, _, _, [], [], false).
rust_lift_clauses([(H :- B)|Cs], Model, P, A, Seed0,
                  [(H :- NB)|RR], Helpers, Changed) :-
    rust_lift_clause_fully(H, B, Model, P, A, Seed0, NB, MyHelpers, Seed1),
    rust_lift_clauses(Cs, Model, P, A, Seed1, RR, RestHelpers, ChangedRest),
    append(MyHelpers, RestHelpers, Helpers),
    ( ( MyHelpers \== [] ; ChangedRest == true ) -> Changed = true ; Changed = false ).

% Lift every embedded aggregate in one clause body (iterate until none remain).
rust_lift_clause_fully(H, B, Model, P, A, Seed0, NB, Helpers, SeedN) :-
    format(atom(Seed), '~w_~w_~w', [P, A, Seed0]),
    ( lift_embedded_aggregate(H, B, Model, Seed, NB1, Helper)
    ->  Seed1 is Seed0 + 1,
        rust_lift_clause_fully(H, NB1, Model, P, A, Seed1, NB, RestHelpers, SeedN),
        Helpers = [Helper|RestHelpers]
    ;   NB = B, Helpers = [], SeedN = Seed0
    ).

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

% ============================================================================
% graph_analysis/4 — the pipeline DECLARATION (WAM_RUST_BRIDGE_CLUSTERING.md
% increment 3). Collapses the verbose per-predicate foreign_lowering blocks of
% increments 1+2 into one statement: the author declares the shared inputs
% (edge predicate, mu predicate, threshold, sketch k) and the list of query
% predicates ONCE, and this expands to exactly the
% foreign_lowering(foreign_predicate(Pred/Arity, SetupOps, [])) terms that were
% hand-written, threading the shared configs + the right native_kind /
% result-mode / layout / register_ffi_mu into each from the table below. The
% build-dependency (clusters -> bridges -> sketches) is already handled by the
% build-on-first-use ensure_* chain in state.rs, so this is declaration
% BUNDLING — not a new build mechanism.
% ============================================================================

:- dynamic graph_analysis_decl/4.

%% bridge_query_kind(+Pred/Arity, -NativeKind, -ResultMode, -ResultLayout)
%  The table mapping each known graph-analysis query predicate to its dispatch
%  native_kind and result shape (mirrors the merged dispatch arms / kernel
%  metadata). `wants_k` (below) says whether a predicate also takes the sketch k.
bridge_query_kind(category_bridge_score/2, category_bridge_score, deterministic, tuple(1)).
bridge_query_kind(bridge/3,                bridge,                stream,        tuple(3)).
bridge_query_kind(category_cluster/2,      category_cluster,      deterministic, tuple(1)).
bridge_query_kind(cluster_members/2,       cluster_members,       stream,        tuple(1)).

%% graph_analysis(+EdgePred, +Inputs, +Stages, +Queries) is det.
%  Directive form: record the declaration for later expansion by
%  graph_analysis_options/2. Inputs: [mu(MuPred), threshold(T)]; Stages:
%  [sketches(k=K), bridges, clusters]; Queries: a list of Pred/Arity.
graph_analysis(EdgePred, Inputs, Stages, Queries) :-
    assertz(graph_analysis_decl(EdgePred, Inputs, Stages, Queries)).

%% graph_analysis_options(-Options, -QueryPreds) is det.
%  Expand ALL asserted graph_analysis/4 declarations into the list of
%  foreign_lowering(...) options (Options) and the flat list of exposed query
%  predicate indicators (QueryPreds), ready to splice into write_wam_rust_project/3.
graph_analysis_options(Options, QueryPreds) :-
    findall(Os-Qs,
            ( graph_analysis_decl(E, I, S, Qs),
              graph_analysis_expand(E, I, S, Qs, Os) ),
            Pairs),
    pairs_keys_values(Pairs, OptsLists, QueryLists),
    append(OptsLists, Options),     % concat per-decl option lists
    append(QueryLists, QueryPreds). % concat per-decl query lists

%% graph_analysis_expand(+EdgePred, +Inputs, +Stages, +Queries, -Options) is det.
%  Pure expansion of ONE declaration into the per-query foreign_lowering terms.
%  Threads the shared edge_pred / mu_pred / threshold / k and harvests the mu
%  facts (MuPred/2) into each register_ffi_mu, exactly as the hand-written
%  increment-1/2 blocks did.
graph_analysis_expand(EdgePred, Inputs, Stages, Queries, Options) :-
    ( member(mu(MuPred), Inputs)        -> true ; MuPred = category_mu ),
    ( member(threshold(T), Inputs)      -> true ; T = 0.3 ),
    ( member(sketches(KSpec), Stages), ga_ksize(KSpec, K) -> true ; K = none ),
    ga_harvest_mu(MuPred, MuData),
    maplist(graph_analysis_query_lowering(EdgePred, MuPred, T, K, MuData),
            Queries, Options).

ga_ksize(k=K, K) :- !.
ga_ksize(K, K) :- integer(K).

%% ga_harvest_mu(+MuPred, -MuData)
%  Read the user's MuPred(Node, Score) facts into a Name-Score pair list (the
%  same data register_ffi_mu embeds). Empty if the predicate is undefined.
ga_harvest_mu(MuPred, MuData) :-
    ( catch(findall(N-Sc,
                ( G =.. [MuPred, N, Sc], catch(call(user:G), _, fail) ),
                MuData0), _, MuData0 = [])
    -> MuData = MuData0
    ;  MuData = [] ).

%% graph_analysis_query_lowering(+EdgePred,+MuPred,+T,+K,+MuData,+Pred/Arity,-Lowering)
%  Build the foreign_lowering(foreign_predicate(...)) term for one query
%  predicate from the shared inputs + the per-predicate table entry.
graph_analysis_query_lowering(EdgePred, MuPred, T, K, MuData, Pred/Arity,
        foreign_lowering(foreign_predicate(Pred/Arity, Ops, []))) :-
    bridge_query_kind(Pred/Arity, NativeKind, Mode, Layout),
    Base = [ register_foreign_native_kind(Pred/Arity, NativeKind),
             register_foreign_result_mode(Pred/Arity, Mode),
             register_foreign_result_layout(Pred/Arity, Layout),
             register_foreign_string_config(Pred/Arity, edge_pred, EdgePred),
             register_foreign_string_config(Pred/Arity, mu_pred, MuPred),
             register_foreign_f64_config(Pred/Arity, threshold, T) ],
    ( integer(K) -> KOps = [register_foreign_usize_config(Pred/Arity, k, K)] ; KOps = [] ),
    append(Base, KOps, Ops0),
    append(Ops0, [register_ffi_mu(Pred/Arity, MuPred, MuData)], Ops).

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
rust_foreign_setup_line(register_foreign_f64_config(Pred/Arity, Key, Value), Line) :-
    format(string(Line),
        '    vm.register_foreign_f64_config("~w/~w", "~w", ~w);', [Pred, Arity, Key, Value]).
rust_foreign_setup_line(register_ffi_mu(_Pred/_Arity, MuPred, Pairs), Line) :-
    % Load a node->score (μ membership) fact table for the bridge detector, the score-fact analogue
    % of register_indexed_atom_fact2. MuPred names the μ predicate (e.g. category_mu); Pairs is a list
    % of Name-Score terms harvested from the μ facts.
    rust_mu_pairs_literal(Pairs, PairsLiteral),
    format(string(Line),
        '    vm.register_ffi_mu("~w", &~w);', [MuPred, PairsLiteral]).
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

rust_mu_pairs_literal(Pairs, Literal) :-
    maplist(rust_mu_pair_literal, Pairs, PairLiterals),
    atomic_list_concat(PairLiterals, ', ', Joined),
    format(string(Literal), '[~w]', [Joined]).

rust_mu_pair_literal(Name-Score, Literal) :-
    escape_rust_string(Name, EName),
    format(string(Literal), '("~w", ~w)', [EName, Score]).

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
    wam_tokenize_line(Line, CleanParts),
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
    rust_const_value(CC, VExpr),
    format(string(Rust),
        'Instruction::GetConstant(~w, "~w".to_string())',
        [VExpr, CAi]).
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
    rust_const_value(C, VExpr),
    format(string(Rust), 'Instruction::UnifyConstant(~w)', [VExpr]).
wam_line_to_rust_instr(["put_constant", C, Ai], _, _, Rust) :-
    clean_comma(C, CC), clean_comma(Ai, CAi),
    rust_const_value(CC, VExpr),
    format(string(Rust),
        'Instruction::PutConstant(~w, "~w".to_string())',
        [VExpr, CAi]).

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
    rust_const_value(C, VExpr),
    format(string(Rust), 'Instruction::SetConstant(~w)', [VExpr]).
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
% T7 route 2: par_aggregate AggType, EnumLabel, BodyLabel, ResultReg [, InReg...]
% (EnumLabel/BodyLabel are predicate-indicator entry labels, e.g.
% __par_enum_foo/2). Any trailing tokens are the container registers holding the
% aggregate's external inputs, in helper-parameter order (empty for input-less).
% Emitted by rust_rewrite_embedded_aggregates.
wam_line_to_rust_instr(["par_aggregate", Type, EnumLabel, BodyLabel, ResultReg | InRegs0], _, _, Rust) :-
    clean_comma(Type, CType),
    clean_comma(EnumLabel, CEnumLabel),
    clean_comma(BodyLabel, CBodyLabel),
    clean_comma(ResultReg, CResultReg),
    maplist(clean_comma, InRegs0, InRegs),
    findall(Q, ( member(R, InRegs), format(atom(Q), '"~w".to_string()', [R]) ), QuotedRegs),
    atomic_list_concat(QuotedRegs, ', ', RegsInner),
    format(string(Rust),
        'Instruction::ParAggregate("~w".to_string(), "~w".to_string(), "~w".to_string(), "~w".to_string(), vec![~w])',
        [CType, CEnumLabel, CBodyLabel, CResultReg, RegsInner]).
wam_line_to_rust_instr(["try_me_else", Label], _, _, Rust) :-
    format(string(Rust),
        'Instruction::TryMeElse("~w".to_string())', [Label]).
wam_line_to_rust_instr(["try", Label], _, _, Rust) :-
    format(string(Rust),
        'Instruction::TryMeElse("~w".to_string())', [Label]).
wam_line_to_rust_instr(["trust_me"], _, _, "Instruction::TrustMe").
wam_line_to_rust_instr(["trust", _Label], _, _, "Instruction::TrustMe").
wam_line_to_rust_instr(["retry_me_else", Label], _, _, Rust) :-
    format(string(Rust),
        'Instruction::RetryMeElse("~w".to_string())', [Label]).
wam_line_to_rust_instr(["retry", Label], _, _, Rust) :-
    format(string(Rust),
        'Instruction::RetryMeElse("~w".to_string())', [Label]).
wam_line_to_rust_instr(["jump", Label], _, _, Rust) :-
    format(string(Rust),
        'Instruction::Jump("~w".to_string())', [Label]).
% M144: legacy cut_ite (only emitted when ite_use_y_level is off) pops
% the single ITE guard CP. Mis-cuts when the condition pushed inner CPs
% - the same limitation the LLVM target had pre-M17 - but strictly
% better than the previous NoOp translation, which never committed at
% all. The Rust compile path now defaults ite_use_y_level(true), so
% normal compiles emit get_level/cut instead.
wam_line_to_rust_instr(["cut_ite"], _, _, "Instruction::CutIte").
wam_line_to_rust_instr(["get_level", Yn], _, _, Rust) :-
    clean_comma(Yn, CYn),
    format(string(Rust), 'Instruction::GetLevel("~w".to_string())', [CYn]).
wam_line_to_rust_instr(["cut", Yn], _, _, Rust) :-
    clean_comma(Yn, CYn),
    format(string(Rust), 'Instruction::CutTo("~w".to_string())', [CYn]).
% Indexing instructions
wam_line_to_rust_instr(["switch_on_constant"|Entries], _, _, Rust) :-
    maplist(parse_index_entry_constant, Entries, RustEntries),
    atomic_list_concat(RustEntries, ', ', Joined),
    format(string(Rust), 'Instruction::SwitchOnConstant(vec![~w])', [Joined]).
wam_line_to_rust_instr(["switch_on_constant_fallthrough"|Entries], _, _, Rust) :-
    % Entries with a "default" target mean "fall through" for that key, so
    % they carry no jump and are dropped from the table. Emitting this as a
    % real instruction (rather than the unknown-fallback comment) keeps every
    % later label PC aligned — the missing instruction shifted them by one.
    exclude(is_default_index_entry, Entries, JumpEntries),
    maplist(parse_index_entry_constant, JumpEntries, RustEntries),
    atomic_list_concat(RustEntries, ', ', Joined),
    format(string(Rust),
        'Instruction::SwitchOnConstantFallthrough(vec![~w])', [Joined]).
wam_line_to_rust_instr(["switch_on_structure"|Entries], _, _, Rust) :-
    maplist(parse_index_entry_structure, Entries, RustEntries),
    atomic_list_concat(RustEntries, ', ', Joined),
    format(string(Rust), 'Instruction::SwitchOnStructure(vec![~w])', [Joined]).
wam_line_to_rust_instr(["switch_on_constant_a2"|Entries], _, _, Rust) :-
    maplist(parse_index_entry_constant, Entries, RustEntries),
    atomic_list_concat(RustEntries, ', ', Joined),
    format(string(Rust), 'Instruction::SwitchOnConstantA2(vec![~w])', [Joined]).
% Fallback for unknown instructions. Emit a real NoOp (not a bare comment):
% a comment is dropped from the vec! literal, which shifts every later label
% PC by one and corrupts backtracking (a missing trust_me/retry_me_else made
% the choice point loop). NoOp preserves alignment, and for the indexing
% hints that land here (switch_on_term, ...) falling through to the
% try_me_else chain is correct.
wam_line_to_rust_instr(Parts, _, _, Rust) :-
    atomic_list_concat(Parts, ' ', Joined),
    format(string(Rust), 'Instruction::NoOp /* unknown: ~w */', [Joined]).

%% rust_const_value(+Const, -RustValueExpr)
%  Render a WAM constant token as the right Value variant. Numeric
%  constants MUST become Value::Integer/Value::Float, not Value::Atom:
%  put_constant/get_constant previously emitted Value::Atom("28") for the
%  integer 28, so `R is <expr>` with R bound to a ground integer (the head
%  arg, e.g. cbi_arith(28) or the cfib(N,R) result check) failed — is/2's
%  result match only handles Unbound/Integer/Float, not Atom. That failure
%  triggered runaway backtracking in recursive programs (fib). set_/
%  unify_constant already did this; this routes the remaining two through
%  the same rule.
rust_const_value(C, Expr) :-
    wam_classify_constant_token(C, Class),
    (   Class = integer(N)
    ->  format(string(Expr), 'Value::Integer(~w)', [N])
    ;   Class = float(N)
    ->  format(string(Expr), 'Value::Float(~w)', [N])
    ;   Class = atom(A)
    ->  escape_rust_string(A, Escaped),
        format(string(Expr), 'Value::Atom("~w".to_string())', [Escaped])
    ).

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

%% is_default_index_entry(+Entry) — true for a "K:default" switch entry.
is_default_index_entry(Entry) :-
    ( string(Entry) -> S = Entry ; atom_string(Entry, S) ),
    sub_string(S, _, _, 0, ":default").

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
    (   PI = Module:Pred/Arity -> true ; PI = Pred/Arity, Module = user ),
    functor(Head, Pred, Arity),
    (   rust_runtime_parser_support_module(Module)
    ->  Clauses = []
    ;   catch(findall(Head-Body, Module:clause(Head, Body), Clauses),
              _, Clauses = [])
    ),
    (   Clauses \= [],
        detect_recursive_kernel(Pred, Arity, Clauses, Kernel)
    ->  format(atom(Key), '~w/~w', [Pred, Arity]),
        Kernels = [Key-Kernel|RestKernels]
    ;   Kernels = RestKernels
    ),
    detect_kernels(Rest, RestKernels).

%% rust_maybe_upgrade_bidirectional(+Options, +Kernel0, -Kernel)
%  kernel_mode(bidirectional) upgrades a detected category_ancestor
%  kernel to bidirectional_ancestor. Mirrors the Haskell/F# targets
%  (maybe_upgrade_bidirectional) — explicit opt-in, never default.
%  The upgraded kernel has a 5-ary interface:
%      Pred(Cat, Root, TotalHops, ParentHops, ChildHops)
%  streaming one solution per budget-feasible path between Cat and
%  Root (parent hops cost parent_step_cost, child hops cost
%  child_step_cost, pruned at cost_budget with an A* lower bound).
%  Cost overrides and the child edge predicate may be passed as
%  project options: parent_step_cost/1, child_step_cost/1,
%  cost_budget/1 (floats), child_pred/1 (atom; default
%  category_child — when absent at runtime the index is derived by
%  reversing the eager parent table).
rust_maybe_upgrade_bidirectional(Options,
        recursive_kernel(category_ancestor, Pred/4, Config0),
        recursive_kernel(bidirectional_ancestor, Pred/5, Config)) :-
    option(kernel_mode(bidirectional), Options),
    !,
    rust_bidirectional_config_extras(Options, Extras),
    append(Config0, Extras, Config).
rust_maybe_upgrade_bidirectional(_, Kernel, Kernel).

rust_bidirectional_config_extras(Options, Extras) :-
    findall(Term,
        ( member(Key, [parent_step_cost, child_step_cost, cost_budget,
                       child_pred]),
          Term =.. [Key, _],
          option(Term, Options) ),
        Extras).

%% rust_maybe_upgrade_boundary(+Options, +Kernel0, -Kernel)
%  boundary_optimization(true) upgrades a detected category_ancestor kernel to
%  the category_ancestor_boundary kernel (the boundary-distribution
%  optimization). Explicit opt-in, never default — see
%  WAM_RUST_BOUNDARY_DISTRIBUTION_{PHILOSOPHY,SPECIFICATION,CACHE_PLAN}.md.
%  The upgraded kernel has a 3-ary interface:
%      Pred(Cat, Root, Result)
%  emitting ONE deterministic result (no choice point) read from the
%  path-length measure of the spliced boundary histogram. The result_extractor
%  config selects which read: scalar (weighted_power), effective_distance
%  (d_eff = WeightSum^(-1/N)), distribution (the raw histogram), or
%  shortest_distance (the cycle-correct min-plus shortest hop-distance to root,
%  read from the distance cache rather than the histogram — increment 2).
%
%  Config extras (project options, with defaults):
%    boundary_weight_n(N)            — the functional exponent N (default 2.0)
%    boundary_result_extractor(E)    — scalar | effective_distance | distribution
%                                      | shortest_distance   (default scalar)
%
%  max_depth and edge_pred carry over from the detected category_ancestor
%  config. The boundary side-table (build_boundary_suffix) is a separate runtime
%  precompute; with an empty side-table the kernel degrades to full enumeration
%  (still correct — the splice is an exact mirror of the production walk), so the
%  lowering is correct by default and the speedup is unlocked once boundary
%  nodes are precomputed.
rust_maybe_upgrade_boundary(Options,
        recursive_kernel(category_ancestor, Pred/4, Config0),
        recursive_kernel(category_ancestor_boundary, Pred/3, Config)) :-
    option(boundary_optimization(true), Options),
    !,
    option(boundary_weight_n(N), Options, 2.0),
    option(boundary_result_extractor(Extractor), Options, scalar),
    append(Config0, [weight_n(N), result_extractor(Extractor)], Config).
rust_maybe_upgrade_boundary(_, Kernel, Kernel).

%% rust_maybe_upgrade_kernel(+Options, +Kernel0, -Kernel)
%  Apply the available opt-in upgrades to a detected category_ancestor kernel.
%  bidirectional and boundary are mutually exclusive (both rewrite
%  category_ancestor): if bidirectional fires first, the boundary upgrade no
%  longer matches; if neither option is set, both are no-ops.
rust_maybe_upgrade_kernel(Options, Kernel0, Kernel) :-
    rust_maybe_upgrade_bidirectional(Options, Kernel0, Kernel1),
    rust_maybe_upgrade_boundary(Options, Kernel1, Kernel).

%% rust_upgrade_kernel_pair(+Options, +Key0-Kernel0, -Key-Kernel)
%  Apply the bidirectional upgrade to a detected Key-Kernel pair,
%  rewriting the key to the upgraded predicate indicator (Pred/5) so
%  registration, dispatch, and the public wrapper stay consistent.
rust_upgrade_kernel_pair(Options, Key0-Kernel0, Key-Kernel) :-
    rust_maybe_upgrade_kernel(Options, Kernel0, Kernel),
    (   Kernel0 == Kernel
    ->  Key = Key0
    ;   Kernel = recursive_kernel(Kind, Pred/Arity, _),
        format(atom(Key), '~w/~w', [Pred, Arity]),
        format(user_error, '[WAM-Rust] kernel upgraded: ~w -> ~w (~w)~n',
               [Key0, Kind, Key])
    ).

%% rust_bidirectional_wrapper_code(+Pred, +Kernel, -Code)
%  Public 5-ary wrapper for an upgraded bidirectional_ancestor kernel:
%  Pred(Cat, Root, TotalHops, ParentHops, ChildHops). Self-registers
%  the kernel metadata + configs (like the other kernel wrappers) and
%  dispatches through execute_foreign_predicate. Parent edge facts must
%  be registered on the VM beforehand (register_ffi_fact_pairs or a
%  lazy lookup source); the child-direction index is derived on first
%  call when absent.
rust_bidirectional_wrapper_code(Pred, Kernel, Code) :-
    Kernel = recursive_kernel(bidirectional_ancestor, Pred/5, _),
    rust_safe_function_name(Pred, FuncName),
    format(atom(Key), '~w/5', [Pred]),
    with_output_to(string(RegLines), emit_kernel_registration(Key-Kernel)),
    format(string(Code),
'/// Bidirectional ancestor kernel wrapper (5-ary):
/// ~w(Cat, Root, TotalHops, ParentHops, ChildHops).
/// Upgraded from a detected category_ancestor kernel by
/// kernel_mode(bidirectional). Register parent edges on the VM first
/// (register_ffi_fact_pairs or a lazy lookup source); the
/// child-direction index is derived on first call when absent.
pub fn ~w(vm: &mut WamState, a1: Value, a2: Value, a3: Value, a4: Value, a5: Value) -> bool {
    vm.code = Vec::new();
    vm.labels = std::collections::HashMap::new();
    vm.pc = 1;
~w    vm.set_reg("A1", a1);
    vm.set_reg("A2", a2);
    vm.set_reg("A3", a3);
    vm.set_reg("A4", a4);
    vm.set_reg("A5", a5);
    vm.execute_foreign_predicate("~w", 5)
}', [Pred, FuncName, RegLines, Pred]).

%% rust_boundary_wrapper_code(+Pred, +Kernel, -Code)
%  Public 3-ary wrapper for an upgraded category_ancestor_boundary kernel:
%  Pred(Cat, Root, Result). Self-registers the kernel metadata + configs (like
%  the other kernel wrappers) and dispatches through execute_foreign_predicate,
%  binding A3 to the single deterministic result. Register parent edges on the
%  VM first (register_ffi_fact_pairs or a lazy lookup source). The boundary
%  side-table is a separate precompute (vm.build_boundary_suffix); with an empty
%  side-table the kernel still returns the correct aggregate by full enumeration.
rust_boundary_wrapper_code(Pred, Kernel, Code) :-
    Kernel = recursive_kernel(category_ancestor_boundary, Pred/3, _),
    rust_safe_function_name(Pred, FuncName),
    format(atom(Key), '~w/3', [Pred]),
    with_output_to(string(RegLines), emit_kernel_registration(Key-Kernel)),
    format(string(Code),
'/// Boundary-distribution ancestor kernel wrapper (3-ary):
/// ~w(Cat, Root, Result).
/// Upgraded from a detected category_ancestor kernel by
/// boundary_optimization(true). Register parent edges on the VM first
/// (register_ffi_fact_pairs or a lazy lookup source); precompute the
/// boundary side-table to unlock the splice speedup — call
/// vm.build_boundary_suffix_root_near(root, d_pre, max_depth, edge_pred)
/// after loading min_dist to pick + precompute the root-near band, or
/// vm.build_boundary_suffix(band, ..) for an explicit band. An empty
/// side-table degrades to full enumeration (still correct).
pub fn ~w(vm: &mut WamState, a1: Value, a2: Value, a3: Value) -> bool {
    vm.code = Vec::new();
    vm.labels = std::collections::HashMap::new();
    vm.pc = 1;
~w    vm.set_reg("A1", a1);
    vm.set_reg("A2", a2);
    vm.set_reg("A3", a3);
    vm.execute_foreign_predicate("~w", 3)
}', [Pred, FuncName, RegLines, Pred]).

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
        format('    vm.register_foreign_f64_config("~w", "~w", ~w);~n',
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
%    kernel_mode(bidirectional) — upgrade detected category_ancestor
%                          kernels to the 5-ary bidirectional_ancestor
%                          kernel (see rust_maybe_upgrade_bidirectional/3)
%    boundary_optimization(Bool) — upgrade detected category_ancestor
%                          kernels to the 3-ary category_ancestor_boundary
%                          kernel, the boundary-distribution optimization
%                          (default false; see rust_maybe_upgrade_boundary/3).
%                          Tunables: boundary_weight_n(N) (default 2.0),
%                          boundary_result_extractor(scalar | effective_distance
%                          | distribution | shortest_distance; default scalar).
%    csr_child_index(Bool) — emit src/csr_fact_source.rs, a LookupSource
%                          over the reverse-CSR artifact for the
%                          bidirectional kernel child direction
%                          (default: false)
write_wam_rust_project(Predicates, Options, ProjectDir) :-
    option(module_name(ModuleName), Options, 'wam_generated'),
    get_time(TimeStamp),
    format_time(string(Date), "%Y-%m-%d %H:%M:%S", TimeStamp),
    wam_target_runtime_parser(wam_rust, Options, RuntimeParserMode),
    rust_validate_runtime_parser_mode(Predicates, RuntimeParserMode),
    rust_project_predicates(Predicates, RuntimeParserMode, ProjectPredicates),

    % Detect recursive kernels in the predicate list. Detected kernels
    % are handled by the FFI (execute_foreign_predicate) at runtime and
    % are excluded from WAM compilation. The detected kernel list is used
    % to auto-generate setup_foreign_predicates() in lib.rs.
    (   option(no_kernels(true), Options)
    ->  DetectedKernels = [],
        format(user_error, '[WAM-Rust] kernel detection suppressed~n', [])
    ;   detect_kernels(ProjectPredicates, DetectedKernels0),
        maplist(rust_upgrade_kernel_pair(Options), DetectedKernels0, DetectedKernels),
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

    % LMDB options:
    %   lmdb_mode(none | cursor)            - whether LMDB is used at all
    %   lmdb_crate(lmdb_zero | heed | auto) - which Rust binding (when mode != none)
    %
    % auto currently resolves to lmdb_zero (the default).  See
    % docs/design/WAM_RUST_LMDB_CRATE_DECISION.md for why both are supported
    % and why lmdb-zero is the default.
    option(lmdb_mode(LmdbMode), Options, none),
    option(lmdb_crate(LmdbCrateOpt), Options, auto),
    resolve_lmdb_crate(LmdbCrateOpt, LmdbCrate),
    (   LmdbMode == cursor
    ->  (   LmdbCrate == lmdb_zero
        ->  UseLmdbZero = true,  UseHeed = false
        ;   LmdbCrate == heed
        ->  UseLmdbZero = false, UseHeed = true
        ;   format(user_error,
                '[WAM-Rust] ERROR: unknown lmdb_crate ~w, falling back to lmdb_zero~n',
                [LmdbCrate]),
            UseLmdbZero = true,  UseHeed = false
        )
    ;   UseLmdbZero = false, UseHeed = false
    ),

    % Generate Cargo.toml (conditionally adds exactly one of lmdb-zero or heed).
    % parallel(true) promotes rayon from an optional dep to a hard dep so the
    % generated bench can call par_iter without a --features flag.
    option(parallel(UseRayon), Options, false),
    render_named_template(rust_wam_cargo,
        [module_name=ModuleName,
         use_lmdb_zero=UseLmdbZero,
         use_heed=UseHeed,
         use_rayon=UseRayon],
        CargoContent),
    directory_file_path(ProjectDir, 'Cargo.toml', CargoPath),
    write_file(CargoPath, CargoContent),

    % Generate src/lmdb_fact_source.rs from the chosen crate's template
    (   UseLmdbZero == true
    ->  read_template_file('templates/targets/rust_wam/lmdb_fact_source_lmdb_zero.rs.mustache',
                           LmdbTemplate),
        render_template(LmdbTemplate, [date=Date], LmdbCode),
        directory_file_path(SrcDir, 'lmdb_fact_source.rs', LmdbPath),
        write_file(LmdbPath, LmdbCode)
    ;   UseHeed == true
    ->  read_template_file('templates/targets/rust_wam/lmdb_fact_source_heed.rs.mustache',
                           LmdbTemplate),
        render_template(LmdbTemplate, [date=Date], LmdbCode),
        directory_file_path(SrcDir, 'lmdb_fact_source.rs', LmdbPath),
        write_file(LmdbPath, LmdbCode)
    ;   true
    ),

    % Generate src/csr_fact_source.rs when the CSR child index is
    % requested: a LookupSource over the reverse-CSR artifact
    % (build_reverse_csr_artifact.py), typically registered under
    % "category_child/2" for the bidirectional kernel child direction.
    option(csr_child_index(UseCsr), Options, false),
    (   UseCsr == true
    ->  read_template_file('templates/targets/rust_wam/csr_fact_source.rs.mustache',
                           CsrTemplate),
        render_template(CsrTemplate, [date=Date], CsrCode),
        directory_file_path(SrcDir, 'csr_fact_source.rs', CsrPath),
        write_file(CsrPath, CsrCode)
    ;   true
    ),

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
    (   IncludeRuntime == true
    ->  compile_dynamic_db_helpers_to_rust(DynamicDbCode)
    ;   DynamicDbCode = ''
    ),
    render_template(StateTemplate, [date=Date, dynamic_db_methods=DynamicDbCode], StateBase),
    (   IncludeRuntime == true
    ->  compile_wam_runtime_to_rust(Options, RuntimeCode),
        format(string(StateCode), "~w\n\n~w", [StateBase, RuntimeCode])
    ;   StateCode = StateBase
    ),
    directory_file_path(SrcDir, 'state.rs', StatePath),
    write_file(StatePath, StateCode),

    % Write par_aggregate.rs (T7 parallel-aggregate runtime) from template file
    read_template_file('templates/targets/rust_wam/par_aggregate.rs.mustache', ParAggTemplate),
    render_template(ParAggTemplate, [date=Date], ParAggCode),
    directory_file_path(SrcDir, 'par_aggregate.rs', ParAggPath),
    write_file(ParAggPath, ParAggCode),

    % Write boundary_cache.rs (P1: exact path-length-histogram splice core) from
    % template file. See WAM_RUST_BOUNDARY_DISTRIBUTION_CACHE_PLAN.md.
    read_template_file('templates/targets/rust_wam/boundary_cache.rs.mustache', BoundaryTemplate),
    render_template(BoundaryTemplate, [date=Date], BoundaryCode),
    directory_file_path(SrcDir, 'boundary_cache.rs', BoundaryPath),
    write_file(BoundaryPath, BoundaryCode),

    % Generate setup_foreign_predicates function for detected kernels
    generate_setup_foreign_predicates_rust(DetectedKernels, SetupForeignCode),

    % Compile predicates and generate lib.rs
    pairs_keys(DetectedKernels, DetectedKeys),
    compile_predicates_for_project(ProjectPredicates, [foreign_pred_keys(DetectedKeys)|Options], PredicatesCode),
    format(string(FullPredicatesCode), "~w\n\n~w", [SetupForeignCode, PredicatesCode]),
    render_named_template(rust_wam_lib,
        [module_name=ModuleName, date=Date, predicates_code=FullPredicatesCode,
         use_lmdb_zero=UseLmdbZero,
         use_heed=UseHeed,
         use_csr=UseCsr],
        LibContent),
    directory_file_path(SrcDir, 'lib.rs', LibPath),
    write_file(LibPath, LibContent),

    % Generate src/main.rs benchmark harness from template file.
    % Rust crate names auto-convert dashes to underscores, so e.g.
    % package "wam-rust-bench" => crate "wam_rust_bench" for imports.
    atom_string(ModuleName, ModuleNameStr),
    string_chars(ModuleNameStr, ModuleChars),
    maplist([C, U]>>(C == '-' -> U = '_' ; U = C), ModuleChars, UnderscoreChars),
    string_chars(CrateName, UnderscoreChars),
    read_template_file('templates/targets/rust_wam/main.rs.mustache', MainTemplate),
    (   memberchk(_-recursive_kernel(bidirectional_ancestor, _, _), DetectedKernels)
    ->  HasBidirectional = true
    ;   HasBidirectional = false
    ),
    render_template(MainTemplate,
        [date=Date, crate_name=CrateName, bidirectional_kernel=HasBidirectional],
        MainContent),
    directory_file_path(SrcDir, 'main.rs', MainRsPath),
    write_file(MainRsPath, MainContent),

    format('WAM Rust project created at: ~w~n', [ProjectDir]),
    format('  Predicates compiled: ~w~n', [ProjectPredicates]).

%% rust_project_predicates(+UserPreds, +Mode, -ProjectPreds)
%  In `compiled` mode, append the portable parser plus the target-agnostic
%  wrapper predicates. Other modes leave the user predicate list unchanged.
rust_project_predicates(Predicates, compiled(prolog_term_parser), ProjectPredicates) :-
    !,
    rust_runtime_parser_predicates(ParserPreds),
    rust_runtime_parser_wrapper_predicates(WrapperPreds),
    append([Predicates, ParserPreds, WrapperPreds], Combined),
    sort(Combined, ProjectPredicates).
rust_project_predicates(Predicates, _RuntimeParserMode, Predicates).

rust_runtime_parser_support_module(prolog_term_parser).
rust_runtime_parser_support_module(cpp_runtime_parser_wrappers).

rust_runtime_parser_predicates(Predicates) :-
    findall(prolog_term_parser:Name/Arity,
        (   current_predicate(prolog_term_parser:Name/Arity),
            functor(Head, Name, Arity),
            \+ predicate_property(prolog_term_parser:Head, imported_from(_)),
            once(clause(prolog_term_parser:Head, _))
        ),
        Raw),
    sort(Raw, Predicates).

rust_runtime_parser_wrapper_predicates(Predicates) :-
    findall(cpp_runtime_parser_wrappers:Name/Arity,
        (   current_predicate(cpp_runtime_parser_wrappers:Name/Arity),
            functor(Head, Name, Arity),
            \+ predicate_property(cpp_runtime_parser_wrappers:Head,
                                  imported_from(_)),
            once(clause(cpp_runtime_parser_wrappers:Head, _))
        ),
        Raw),
    sort(Raw, Predicates).

rust_validate_runtime_parser_mode(Predicates, none) :-
    !,
    (   rust_predicates_parser_dependency(Predicates, Pred, Builtin)
    ->  throw(error(permission_error(use, runtime_parser, Builtin),
                    context(write_wam_rust_project/3,
                            parser_disabled_for_predicate(Pred))))
    ;   true
    ).
rust_validate_runtime_parser_mode(_Predicates, _Mode).

rust_predicates_parser_dependency(Predicates, Pred, Builtin) :-
    member(Pred, Predicates),
    rust_predicate_clause(Pred, _Head, Body),
    parser_dependent_body_goal(Body, Builtin),
    !.

rust_predicate_clause(Module:Name/Arity, Head, Body) :-
    !,
    functor(Head, Name, Arity),
    clause(Module:Head, Body).
rust_predicate_clause(Name/Arity, Head, Body) :-
    functor(Head, Name, Arity),
    clause(user:Head, Body).

%% compile_predicates_for_project(+Predicates, +Options, -Code)
%  Two-pass compilation: collects all WAM-fallback predicates into a shared
%  code+labels table so cross-predicate WAM calls (Execute/Call) work.
%  Native-lowered predicates are compiled independently as before.
compile_predicates_for_project(Predicates0, Options, Code) :-
    % Pass 0 (T7 route-1): pull out predicates whose body is a parallel-eligible
    % aggregate — synthesise their enum/body helpers (added to the compile set)
    % and emit a native par_collect wrapper for each. No-op unless
    % parallel_aggregates(true).
    rust_inject_parallel_aggregates(Predicates0, Options, Predicates1, ParWrappers),
    % Pass 0.5 (T7 route-2): aggregates EMBEDDED in larger clause bodies —
    % synthesise their enum/body helpers (added to the compile set) and record a
    % rewrite to splice a par_aggregate instruction over the begin/end block.
    rust_inject_embedded_par_aggregates(Predicates1, Options, Predicates, EmbeddedRewrites),
    % Pass 1: classify each predicate as native, wam, or failed
    classify_predicates(Predicates, Options, Classified),
    % Pass 2: collect WAM entries with cumulative PCs, build shared table
    collect_wam_entries(Classified, 1, EmbeddedRewrites, WamEntries, AllInstrParts, AllLabelParts),
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
}

pub fn shared_wam_program() -> (Vec<Instruction>, HashMap<String, usize>) {
    let (code, labels) = get_shared_wam();
    (code.clone(), labels.clone())
}', [AllLabels, AllInstrs])
    ;   % No shared-WAM predicates in this project. Still emit
        % shared_wam_program/0: templates/targets/rust_wam/main.rs.mustache
        % (and materialisation_setup.rs.mustache) import and call it
        % unconditionally, so omitting it breaks compilation of every
        % all-native/all-kernel project.
        SharedCode =
'pub fn shared_wam_program() -> (Vec<Instruction>, HashMap<String, usize>) {
    (Vec::new(), HashMap::new())
}'
    ),
    % Pass 3: generate code for each predicate
    generate_predicate_codes(Classified, WamEntries, Options, PredCodes),
    % Combine shared table + predicate code (+ T7 parallel wrappers)
    (   SharedCode == ""
    ->  atomic_list_concat(PredCodes, '\n\n', Body0)
    ;   atomic_list_concat(PredCodes, '\n\n', PredCodesStr),
        format(string(Body0), "~w\n\n~w", [SharedCode, PredCodesStr])
    ),
    (   ParWrappers == []
    ->  Body1 = Body0
    ;   atomic_list_concat(ParWrappers, '\n\n', WrapStr),
        format(string(Body1), "~w\n\n// --- T7 parallel-aggregate wrappers (route 1) ---\n~w",
               [Body0, WrapStr])
    ),
    % T9 call-site dispatch: a crate-level fact_table_call referenced by the
    % Call/Execute handlers (always emitted; empty match when no fact tables).
    rust_fact_table_dispatch_fn(Classified, DispatchCode),
    format(string(Code), "~w\n\n~w", [Body1, DispatchCode]).

%% rust_fact_table_dispatch_fn(+Classified, -Code)
%  Emit `fact_table_call(vm, pred, cont_pc) -> Option<bool>`: a match over the
%  fact-table predicates routing a WAM call/execute to the right `<fn>_run`,
%  reading args from the A registers. Some(ok) if pred is a fact table, else None.
rust_fact_table_dispatch_fn(Classified, Code) :-
    findall(P/A,
            member(classified(_, P, A, fact_table, _), Classified),
            FactPIs),
    maplist(rust_fact_dispatch_arm, FactPIs, Arms),
    atomic_list_concat(Arms, '\n', ArmsStr),
    format(string(Code),
'/// T9 call-site dispatch: route a WAM call/execute of a fact-table predicate to
/// its enumerator. Some(ok) if `pred` is a fact table, None otherwise.
#[allow(unused_variables)]
pub fn fact_table_call(vm: &mut WamState, pred: &str, cont_pc: usize) -> Option<bool> {
    match pred {
~w        _ => None,
    }
}', [ArmsStr]).

rust_fact_dispatch_arm(Pred/Arity, Arm) :-
    rust_safe_function_name(Pred/Arity, FName),
    numlist(1, Arity, Idxs),
    findall(Read,
            ( member(I, Idxs),
              format(atom(Read),
                '            let a~w = vm.get_reg_raw("A~w").unwrap_or(Value::Unbound("_A~w".to_string()));',
                [I, I, I]) ),
            Reads),
    atomic_list_concat(Reads, '\n', ReadsStr),
    findall(AN, ( member(I, Idxs), format(atom(AN), 'a~w', [I]) ), ANs),
    atomic_list_concat(ANs, ', ', ArgsVec),
    format(string(Arm),
'        "~w/~w" => {
~w
            Some(~w_run(vm, vec![~w], cont_pc))
        }', [Pred, Arity, ReadsStr, FName, ArgsVec]).

%% rust_pred_has_control_constructs(+Module:Pred/Arity)
%  True if any clause body uses a backtracking-driven control construct
%  (\+, ->, ;, once, forall, or a cut). The native Rust emitter
%  (rust_target.pl) targets deterministic shapes (tail-recursion kernels,
%  fact lookups) and does NOT model these: it mis-lowers them into a whole
%  stdin-driven program whose body, under include_main(false), dangles at
%  module level (syntax error). Mirrors go_pred_has_control_constructs/1;
%  used to decline native and route such predicates to the lowered/WAM path.
rust_pred_has_control_constructs(Module:Pred/Arity) :-
    functor(Head, Pred, Arity),
    clause(Module:Head, Body),
    rust_body_has_control(Body),
    !.

rust_body_has_control(G) :- var(G), !, fail.
rust_body_has_control((_ -> _)) :- !.
rust_body_has_control((_ ; _)) :- !.
rust_body_has_control(\+ _) :- !.
rust_body_has_control(not(_)) :- !.
rust_body_has_control(once(_)) :- !.
rust_body_has_control(forall(_, _)) :- !.
rust_body_has_control(!) :- !.
rust_body_has_control((A , B)) :- !, ( rust_body_has_control(A) -> true ; rust_body_has_control(B) ).
rust_body_has_control(_) :- fail.

%% classify_predicates(+Predicates, +Options, -Classified)
%  Returns list of classify(Module, Pred, Arity, Strategy, ExtraData) terms.
classify_predicates([], _, []).
classify_predicates([PredIndicator|Rest], Options, [Entry|RestEntries]) :-
    (   PredIndicator = Module:Pred/Arity -> true
    ;   PredIndicator = Pred/Arity, Module = user
    ),
    % Warn (once, here) if this is a large ground-fact predicate we will NOT
    % inline because it exceeds the cap — it falls through to T4 below, but the
    % right fix is an external fact source.
    rust_maybe_warn_oversized_facts(Module:Pred/Arity, Options),
    (   % T9 fact-table inline: an all-ground-facts predicate whose row count is
        % in the inline window [t9_min_rows, t9_max_rows] compiles to a static
        % row table + first-arg hash index + choice-point enumeration, instead of
        % T4 instruction sequences. Default in-range (faster compile + correct vs
        % T4); opt out with fact_table_inline(false). Checked first so it wins.
        \+ option(fact_table_inline(false), Options),
        rust_fact_table_classify(Module:Pred/Arity, Options, FInfo),
        catch(emit_fact_table_rust(Pred/Arity, FInfo, Options, FactCode), _, fail)
    ->  format(user_error, '  ~w/~w: fact table (T9)~n', [Pred, Arity]),
        Entry = classified(Module, Pred, Arity, fact_table, FactCode)
    ;   % Check for auto-detectable FFI kernel FIRST (unless suppressed)
        \+ option(no_kernels(true), Options),
        \+ rust_runtime_parser_support_module(Module),
        functor(KHead, Pred, Arity),
        findall(KHead-KBody, Module:clause(KHead, KBody), KClauses),
        KClauses \= [],
        detect_recursive_kernel(Pred, Arity, KClauses, Kernel0)
    ->  rust_maybe_upgrade_kernel(Options, Kernel0, Kernel),
        format(user_error, '  ~w/~w: ffi kernel (~w)~n', [Pred, Arity, Kernel]),
        % Besides the CallForeign dispatch route, emit a public Rust wrapper
        % function (bare predicate name) so library consumers can call the
        % kernel directly: `use crate::pred; pred(&mut vm, args...)`.
        % Without this, ffi_kernel predicates had no callable symbol at all.
        (   Kernel = recursive_kernel(bidirectional_ancestor, _, _)
        ->  % Upgraded kernel: the public interface widens to 5 args
            % (Cat, Root, TotalHops, ParentHops, ChildHops), so the
            % legacy 4-ary spec wrapper does not apply.
            rust_bidirectional_wrapper_code(Pred, Kernel, WrapperCode),
            Entry = classified(Module, Pred, Arity, ffi_kernel,
                               kernel_with_wrapper(Kernel, WrapperCode))
        ;   Kernel = recursive_kernel(category_ancestor_boundary, _, _)
        ->  % Upgraded boundary kernel: the public interface narrows to 3 args
            % (Cat, Root, Result) — one deterministic aggregate/distribution.
            rust_boundary_wrapper_code(Pred, Kernel, WrapperCode),
            Entry = classified(Module, Pred, Arity, ffi_kernel,
                               kernel_with_wrapper(Kernel, WrapperCode))
        ;   catch(rust_target:rust_foreign_lowering_spec(Pred, Arity, KClauses,
                                                          ForeignSpec),
                  _, fail),
            catch(compile_wam_predicate_to_rust(Pred/Arity, '',
                      [foreign_lowering(ForeignSpec)|Options], WrapperCode),
                  _, fail)
        ->  Entry = classified(Module, Pred, Arity, ffi_kernel,
                               kernel_with_wrapper(Kernel, WrapperCode))
        ;   Entry = classified(Module, Pred, Arity, ffi_kernel, Kernel)
        )
    ;   % Try native Rust lowering (disable WAM fallback inside rust_target
        % so we can distinguish truly-native from WAM-needing predicates).
        % Decline native for backtracking control constructs (\+, ->, ;,
        % once, forall, cut): the native emitter mis-lowers them (see
        % rust_pred_has_control_constructs/1), so route them to the
        % lowered/WAM path which models choicepoints correctly.
        \+ rust_pred_has_control_constructs(Module:Pred/Arity),
        catch(
            rust_target:compile_predicate_to_rust(Module:Pred/Arity,
                [include_main(false), wam_fallback(false)|Options], PredCode),
            _, fail)
    ->  format(user_error, '  ~w/~w: native lowering~n', [Pred, Arity]),
        Entry = classified(Module, Pred, Arity, native, PredCode)
    ;   % Fall back to WAM compilation
        option(wam_fallback(WamFB), Options, true),
        WamFB \== false,
        % M144: emit if-then-else with get_level Yn / cut Yn rather than
        % the legacy cut_ite. The naive single-CP cut_ite mis-commits
        % when the condition pushes inner choice points; get_level/cut
        % snapshots and restores the CP depth exactly (same change the
        % LLVM target made in M17). The Rust runtime now implements
        % both instructions.
        ( memberchk(ite_use_y_level(_), Options)
        -> WamOptions = Options
        ;  WamOptions = [ite_use_y_level(true)|Options]
        ),
        wam_target:compile_predicate_to_wam(Module:Pred/Arity, WamOptions, WamCode),
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
collect_wam_entries([], _, _, [], [], []).
collect_wam_entries([classified(_, Pred, Arity, wam, WamCode)|Rest], PC, Rewrites,
                    [wam_entry(Pred, Arity, PC)|RestEntries],
                    AllInstrs, AllLabels) :-
    atom_string(WamCode, WamStr),
    split_string(WamStr, "\n", "", Lines0),
    rust_rewrite_embedded_aggregate(Lines0, Pred/Arity, Rewrites, Lines),
    wam_lines_to_rust(Lines, PC, Pred/Arity, [], InstrParts, LabelParts),
    % Count instructions to compute next PC
    length(InstrParts, InstrCount),
    NextPC is PC + InstrCount,
    collect_wam_entries(Rest, NextPC, Rewrites, RestEntries, RestInstrs, RestLabels),
    append(InstrParts, RestInstrs, AllInstrs),
    append(LabelParts, RestLabels, AllLabels).
collect_wam_entries([classified(_, Pred, Arity, lowered, lowered_code(_, WamCode))|Rest], PC, Rewrites,
                    [wam_entry(Pred, Arity, PC)|RestEntries],
                    AllInstrs, AllLabels) :-
    atom_string(WamCode, WamStr),
    split_string(WamStr, "\n", "", Lines0),
    rust_rewrite_embedded_aggregate(Lines0, Pred/Arity, Rewrites, Lines),
    wam_lines_to_rust(Lines, PC, Pred/Arity, [], InstrParts, LabelParts),
    length(InstrParts, InstrCount),
    NextPC is PC + InstrCount,
    collect_wam_entries(Rest, NextPC, Rewrites, RestEntries, RestInstrs, RestLabels),
    append(InstrParts, RestInstrs, AllInstrs),
    append(LabelParts, RestLabels, AllLabels).
collect_wam_entries([_|Rest], PC, Rewrites, Entries, Instrs, Labels) :-
    collect_wam_entries(Rest, PC, Rewrites, Entries, Instrs, Labels).

%% generate_predicate_codes(+Classified, +WamEntries, +Options, -PredCodes)
%  Generates Rust code for each classified predicate. Options is threaded so the
%  shared-table WAM path can consult the T7 parallel-aggregate gate.
generate_predicate_codes([], _, _, []).
generate_predicate_codes([classified(_, Pred, Arity, ffi_kernel,
                                     kernel_with_wrapper(Kernel, WrapperCode))|Rest],
                         WamEntries, Options, [Code|RestCodes]) :-
    !,
    Kernel = recursive_kernel(Kind, _, _),
    format(string(Code),
        "// Strategy: ffi_kernel (~w)\n// ~w/~w dispatched via CallForeign → execute_foreign_predicate\n~w",
        [Kind, Pred, Arity, WrapperCode]),
    generate_predicate_codes(Rest, WamEntries, Options, RestCodes).
generate_predicate_codes([classified(_, Pred, Arity, ffi_kernel, Kernel)|Rest],
                         WamEntries, Options, [Code|RestCodes]) :-
    % FFI kernel — no WAM code needed; handled by setup_foreign_predicates
    % and execute_foreign_predicate at runtime via CallForeign dispatch.
    Kernel = recursive_kernel(Kind, _, _),
    format(string(Code),
        "// Strategy: ffi_kernel (~w)\n// ~w/~w dispatched via CallForeign → execute_foreign_predicate",
        [Kind, Pred, Arity]),
    generate_predicate_codes(Rest, WamEntries, Options, RestCodes).
generate_predicate_codes([classified(_, _Pred, _Arity, native, PredCode)|Rest],
                         WamEntries, Options, [Code|RestCodes]) :-
    format(string(Code), "// Strategy: native\n~w", [PredCode]),
    generate_predicate_codes(Rest, WamEntries, Options, RestCodes).
generate_predicate_codes([classified(_, _Pred, _Arity, fact_table, PredCode)|Rest],
                         WamEntries, Options, [Code|RestCodes]) :-
    format(string(Code), "// Strategy: fact_table (T9)\n~w", [PredCode]),
    generate_predicate_codes(Rest, WamEntries, Options, RestCodes).
generate_predicate_codes([classified(_, _Pred, _Arity, wam_foreign, PredCode)|Rest],
                         WamEntries, Options, [Code|RestCodes]) :-
    format(string(Code), "// Strategy: wam\n~w", [PredCode]),
    generate_predicate_codes(Rest, WamEntries, Options, RestCodes).
generate_predicate_codes([classified(_, _Pred, _Arity, lowered, lowered_code(PredCode, _))|Rest],
                         WamEntries, Options, [Code|RestCodes]) :-
    format(string(Code), "// Strategy: lowered\n~w", [PredCode]),
    generate_predicate_codes(Rest, WamEntries, Options, RestCodes).
generate_predicate_codes([classified(_, Pred, Arity, wam, _WamCode)|Rest],
                         WamEntries, Options, [Code|RestCodes]) :-
    % Look up this predicate's start PC in the shared table
    member(wam_entry(Pred, Arity, StartPC), WamEntries),
    compile_wam_predicate_to_rust_shared(Pred/Arity, StartPC, Options, Code),
    generate_predicate_codes(Rest, WamEntries, Options, RestCodes).
generate_predicate_codes([classified(_, _Pred, _Arity, failed, PredCode)|Rest],
                         WamEntries, Options, [Code|RestCodes]) :-
    format(string(Code), "// Strategy: failed\n~w", [PredCode]),
    generate_predicate_codes(Rest, WamEntries, Options, RestCodes).

%% compile_wam_predicate_to_rust_shared(+Pred/Arity, +StartPC, +Options, -RustCode)
%  Generates a thin WAM predicate wrapper that references the shared code table.
%  Consults the T7 parallel-aggregate gate (cost machinery) so project-mode
%  codegen carries the same parallel-eligibility annotation as the standalone
%  path; gated behind parallel_aggregates(true), so default output is unchanged.
compile_wam_predicate_to_rust_shared(Pred/Arity, StartPC, Options, RustCode) :-
    atom_string(Pred, PredStr),
    rust_safe_function_name(Pred/Arity, FuncName),
    build_rust_wam_arg_list(Arity, ArgList),
    build_rust_wam_arg_setup(Arity, ArgSetup),
    rust_predicate_parallel_decision(Pred/Arity, Options, ParDecision),
    rust_parallel_annotation(ParDecision, AggAnno),
    format(string(RustCode),
'// Strategy: wam
/// WAM-compiled predicate: ~w/~w (shared table, pc=~w)
/// Compiled via WAM for predicates that resist native lowering.
~wpub fn ~w(~w) -> bool {
    let (code, labels) = get_shared_wam();
    vm.code = code.clone();
    vm.labels = labels.clone();
    vm.pc = ~w;
~w
    vm.run()
}', [PredStr, Arity, StartPC, AggAnno, FuncName, ArgList, StartPC, ArgSetup]).

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
    ;   resolve_template_path(RelativePath, AbsPath), exists_file(AbsPath)
    ->  read_file_to_string(AbsPath, Content, [])
    ;   format(atom(Content),
            '// Template not found: ~w', [RelativePath])
    ).

%% resolve_template_path(+RelativePath, -AbsPath)
%  Resolve a repo-root-relative template path against this module's source
%  location, so generation works from any working directory (e.g. the
%  conformance harness runs from tests/, where the cwd-relative
%  'templates/...' path does not exist). Mirrors the python/go targets.
resolve_template_path(RelativePath, AbsPath) :-
    source_file(wam_rust_target:write_wam_rust_project(_,_,_), ThisFile),
    file_directory_name(ThisFile, TargetsDir),   % src/unifyweaver/targets
    atomic_list_concat([TargetsDir, '/../../../', RelativePath], Raw),
    absolute_file_name(Raw, AbsPath).

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

%% resolve_lmdb_crate(+Spec, -Crate)
%
%  Resolve the lmdb_crate codegen option to a concrete crate name.
%  Spec is one of: lmdb_zero | heed | auto.
%  When auto, defaults to lmdb_zero (current empirical preference; see
%  docs/design/WAM_RUST_LMDB_CRATE_DECISION.md).
%
%  Future heuristics (target core count, fixture write-permissions,
%  measured throughput) would slot in here, following the pattern from
%  resolve_auto_cache_strategy/2 and resolve_auto_lmdb_cache_mode/2 in
%  src/unifyweaver/core/cost_model.pl.

resolve_lmdb_crate(lmdb_zero, lmdb_zero) :- !.
resolve_lmdb_crate(heed, heed) :- !.
resolve_lmdb_crate(auto, lmdb_zero) :- !.
resolve_lmdb_crate(Other, lmdb_zero) :-
    format(user_error,
        '[WAM-Rust] WARNING: unknown lmdb_crate ~w, defaulting to lmdb_zero~n',
        [Other]).
