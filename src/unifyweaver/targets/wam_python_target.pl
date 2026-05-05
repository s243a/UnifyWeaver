:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% wam_python_target.pl - WAM-to-Python Transpilation Target
%
% Transpiles WAM runtime predicates (wam_runtime.pl) to Python code.
% Uses trail-based mutable state (classical WAM), not persistent data
% structures. No external Python dependencies.
%
% Two emit modes:
%   1. Interpreter mode: dispatch table + run_wam(instrs, state) loop
%   2. Lowered/function mode: each predicate as a Python function with
%      explicit continuation passing (model on Rust target)
%
% Register encoding (same as all other WAM targets):
%   A1 → 1, A2 → 2, ..., AN → N
%   X1 → 101, X2 → 102, ..., XN → 100+N
%   Y1 → 201, Y2 → 202, ..., YN → 200+N
%
% See: docs/design/WAM_RUST_TRANSPILATION_IMPLEMENTATION_PLAN.md

:- module(wam_python_target, [
	compile_step_wam_to_python/2,          % +Options, -PythonCode
	compile_wam_helpers_to_python/2,       % +Options, -PythonCode
	compile_wam_runtime_to_python/2,       % +Options, -PythonCode
	compile_wam_predicate_to_python/4,     % +Pred/Arity, +WamCode, +Options, -PythonCode
	wam_instruction_to_python_literal/2,   % +WamInstr, -PyLiteral
	wam_line_to_python_literal/2,          % +Parts, -PyLit
	write_wam_python_project/3,            % +Predicates, +Options, +ProjectDir
	emit_wam_python/3                      % +Predicates, +Options, +Mode
]).

:- use_module(library(lists)).
:- use_module(library(option)).
:- use_module('../core/template_system').
:- use_module('../bindings/python_wam_bindings').
:- use_module('../targets/wam_target', [compile_predicate_to_wam/3]).
:- use_module('../targets/wam_python_lowered_emitter', [
	emit_lowered_python/4,
	is_deterministic_pred_py/1,
	parse_wam_text_py/2,
	python_func_name/2
]).

% ============================================================================
% TOP-LEVEL ENTRY POINT
% ============================================================================

%% emit_wam_python(+Predicates, +Options, +Mode)
%  Top-level entry: dispatches to interpreter or lowered mode.
%  Mode is one of: interpreter, lowered
emit_wam_python(Predicates, Options, interpreter) :-
	!,
	option(project_dir(ProjectDir), Options, 'wam_python_out'),
	write_wam_python_project(Predicates, Options, ProjectDir).
emit_wam_python(Predicates, Options, lowered) :-
	!,
	option(project_dir(ProjectDir), Options, 'wam_python_out'),
	merge_options([emit_mode(lowered)], Options, Options1),
	write_wam_python_project(Predicates, Options1, ProjectDir).
emit_wam_python(Predicates, Options, _Mode) :-
	% Default to interpreter mode
	emit_wam_python(Predicates, Options, interpreter).

% ============================================================================
% PHASE 2: step_wam/3 → Python if/elif chain
% ============================================================================

%% compile_step_wam_to_python(+Options, -PythonCode)
%  Generates the body of the step() method as Python if/elif dispatch.
%  Each step_wam/3 clause becomes one elif branch.
compile_step_wam_to_python(_Options, PythonCode) :-
	findall(Branch, compile_step_branch(Branch), Branches),
	atomic_list_concat(Branches, '\n', BranchesCode),
	format(string(PythonCode),
'    def step(self, instr):
        """Execute a single WAM instruction. Returns True on success."""
~w
        else:
            return False', [BranchesCode]).

%% compile_step_branch(-BranchCode)
%  Each WAM instruction maps to an if/elif branch.
compile_step_branch(BranchCode) :-
	wam_instruction_branch(InstrCheck, BodyCode),
	format(string(BranchCode),
		'        ~w:\n~w', [InstrCheck, BodyCode]).

% --- Head Unification Instructions ---

wam_instruction_branch('if instr[0] == \"get_constant\"', Body) :-
	Body = '            c, ai = instr[1], instr[2]
            val = get_reg(self, ai)
            if val is None:
                return False
            d = deref(val, self)
            if isinstance(d, Var):
                bind(d, c, self)
                self.regs[ai] = c
                return True
            return d == c'.

wam_instruction_branch('elif instr[0] == \"get_variable\"', Body) :-
	Body = '            xn, ai = instr[1], instr[2]
            val = get_reg(self, ai)
            if val is None:
                return False
            set_reg(self, xn, deref(val, self))
            return True'.

wam_instruction_branch('elif instr[0] == \"get_value\"', Body) :-
	Body = '            xn, ai = instr[1], instr[2]
            val_a = get_reg(self, ai)
            val_x = get_reg(self, xn)
            if val_a is None:
                return False
            if val_x is None:
                return False
            return unify(val_a, val_x, self)'.

wam_instruction_branch('elif instr[0] == \"get_structure\"', Body) :-
	Body = '            fn, arity, ai = instr[1], instr[2], instr[3]
            val = get_reg(self, ai)
            if val is None:
                return False
            d = deref(val, self)
            if isinstance(d, Var):
                # Write mode
                addr = heap_put(self, Compound(fn, [None] * arity))
                bind(d, Ref(addr), self)
                self.s = addr + 1
                self.mode = \'write\'
                return True
            if isinstance(d, Compound) and d.functor == fn and len(d.args) == arity:
                # Read mode
                self.s = 0
                self.mode = \'read\'
                self._struct_args = list(d.args)
                return True
            if isinstance(d, Ref):
                cell = heap_get(self, d.addr)
                if isinstance(cell, Compound) and cell.functor == fn and len(cell.args) == arity:
                    self.s = 0
                    self.mode = \'read\'
                    self._struct_args = list(cell.args)
                    return True
            return False'.

wam_instruction_branch('elif instr[0] == \"get_list\"', Body) :-
	Body = '            ai = instr[1]
            val = get_reg(self, ai)
            if val is None:
                return False
            d = deref(val, self)
            if isinstance(d, Var):
                addr = heap_put(self, Compound(\".\", [None, None]))
                bind(d, Ref(addr), self)
                self.s = addr + 1
                self.mode = \'write\'
                return True
            if isinstance(d, Compound) and d.functor == \".\" and len(d.args) == 2:
                self.s = 0
                self.mode = \'read\'
                self._struct_args = list(d.args)
                return True
            return False'.

wam_instruction_branch('elif instr[0] == \"get_nil\"', Body) :-
	Body = '            ai = instr[1]
            val = get_reg(self, ai)
            if val is None:
                return False
            d = deref(val, self)
            if isinstance(d, Var):
                bind(d, Atom(\"[]\"), self)
                self.regs[ai] = Atom(\"[]\")
                return True
            return isinstance(d, Atom) and d.name == \"[]\"'.

wam_instruction_branch('elif instr[0] == \"get_integer\"', Body) :-
	Body = '            n, ai = instr[1], instr[2]
            val = get_reg(self, ai)
            if val is None:
                return False
            d = deref(val, self)
            if isinstance(d, Var):
                bind(d, Int(n), self)
                self.regs[ai] = Int(n)
                return True
            return isinstance(d, Int) and d.n == n'.

wam_instruction_branch('elif instr[0] == \"get_float\"', Body) :-
	Body = '            f, ai = instr[1], instr[2]
            val = get_reg(self, ai)
            if val is None:
                return False
            d = deref(val, self)
            if isinstance(d, Var):
                bind(d, Float(f), self)
                self.regs[ai] = Float(f)
                return True
            return isinstance(d, Float) and d.f == f'.

% --- Unify Instructions ---

wam_instruction_branch('elif instr[0] == \"unify_variable\"', Body) :-
	Body = '            xn = instr[1]
            if self.mode == \'read\':
                if hasattr(self, \'_struct_args\') and self.s < len(self._struct_args):
                    set_reg(self, xn, self._struct_args[self.s])
                    self.s += 1
                    return True
                return False
            else:
                v = Var()
                addr = heap_put(self, v)
                set_reg(self, xn, v)
                self.s += 1
                return True'.

wam_instruction_branch('elif instr[0] == \"unify_value\"', Body) :-
	Body = '            xn = instr[1]
            val = get_reg(self, xn)
            if val is None:
                return False
            if self.mode == \'read\':
                if hasattr(self, \'_struct_args\') and self.s < len(self._struct_args):
                    result = unify(val, self._struct_args[self.s], self)
                    self.s += 1
                    return result
                return False
            else:
                heap_put(self, val)
                self.s += 1
                return True'.

wam_instruction_branch('elif instr[0] == \"unify_constant\"', Body) :-
	Body = '            c = instr[1]
            if self.mode == \'read\':
                if hasattr(self, \'_struct_args\') and self.s < len(self._struct_args):
                    result = unify(c, self._struct_args[self.s], self)
                    self.s += 1
                    return result
                return False
            else:
                heap_put(self, c)
                self.s += 1
                return True'.

wam_instruction_branch('elif instr[0] == \"unify_nil\"', Body) :-
	Body = '            if self.mode == \'read\':
                if hasattr(self, \'_struct_args\') and self.s < len(self._struct_args):
                    result = unify(Atom(\"[]\"), self._struct_args[self.s], self)
                    self.s += 1
                    return result
                return False
            else:
                heap_put(self, Atom(\"[]\"))
                self.s += 1
                return True'.

wam_instruction_branch('elif instr[0] == \"unify_void\"', Body) :-
	Body = '            n = instr[1] if len(instr) > 1 else 1
            if self.mode == \'read\':
                self.s += n
                return True
            else:
                for _ in range(n):
                    heap_put(self, Var())
                    self.s += 1
                return True'.

% --- Body Construction Instructions ---

wam_instruction_branch('elif instr[0] == \"put_variable\"', Body) :-
	Body = '            xn, ai = instr[1], instr[2]
            v = Var()
            set_reg(self, xn, v)
            set_reg(self, ai, v)
            return True'.

wam_instruction_branch('elif instr[0] == \"put_value\"', Body) :-
	Body = '            xn, ai = instr[1], instr[2]
            val = get_reg(self, xn)
            if val is not None:
                set_reg(self, ai, val)
                return True
            return False'.

wam_instruction_branch('elif instr[0] == \"put_unsafe_value\"', Body) :-
	Body = '            yn, ai = instr[1], instr[2]
            val = get_reg(self, yn)
            if val is None:
                return False
            d = deref(val, self)
            if isinstance(d, Var):
                new_var = Var()
                addr = heap_put(self, new_var)
                bind(d, new_var, self)
                set_reg(self, ai, new_var)
            else:
                set_reg(self, ai, d)
            return True'.

wam_instruction_branch('elif instr[0] == \"put_constant\"', Body) :-
	Body = '            c, ai = instr[1], instr[2]
            set_reg(self, ai, c)
            return True'.

wam_instruction_branch('elif instr[0] == \"put_nil\"', Body) :-
	Body = '            ai = instr[1]
            set_reg(self, ai, Atom(\"[]\"))
            return True'.

wam_instruction_branch('elif instr[0] == \"put_integer\"', Body) :-
	Body = '            n, ai = instr[1], instr[2]
            set_reg(self, ai, Int(n))
            return True'.

wam_instruction_branch('elif instr[0] == \"put_float\"', Body) :-
	Body = '            f, ai = instr[1], instr[2]
            set_reg(self, ai, Float(f))
            return True'.

wam_instruction_branch('elif instr[0] == \"put_structure\"', Body) :-
	Body = '            fn, arity, ai = instr[1], instr[2], instr[3]
            addr = heap_put(self, Compound(fn, [None] * arity))
            set_reg(self, ai, Ref(addr))
            self.s = 0
            self.mode = \'write\'
            self._struct_write_addr = addr
            self._struct_write_idx = 0
            self._struct_write_arity = arity
            return True'.

wam_instruction_branch('elif instr[0] == \"put_list\"', Body) :-
	Body = '            ai = instr[1]
            addr = heap_put(self, Compound(\".\", [None, None]))
            set_reg(self, ai, Ref(addr))
            self.s = 0
            self.mode = \'write\'
            self._struct_write_addr = addr
            self._struct_write_idx = 0
            self._struct_write_arity = 2
            return True'.

% --- Control Instructions ---

wam_instruction_branch('elif instr[0] == \"call\"', Body) :-
	Body = '            pred, arity = instr[1], instr[2]
            target_pc = self.labels.get(pred)
            if target_pc is not None:
                self.cp = self._pc + 1
                self._pc = target_pc - 1  # will be incremented by run loop
                return True
            # Try foreign predicate
            return execute_foreign(pred, arity, None, self)'.

wam_instruction_branch('elif instr[0] == \"execute\"', Body) :-
	Body = '            pred = instr[1]
            target_pc = self.labels.get(pred)
            if target_pc is not None:
                self._pc = target_pc - 1  # will be incremented by run loop
                return True
            return False'.

wam_instruction_branch('elif instr[0] == \"proceed\"', Body) :-
	Body = '            if self.cp is None:
                self.halt = True
                return True
            self._pc = self.cp - 1  # will be incremented by run loop
            self.cp = None
            return True'.

wam_instruction_branch('elif instr[0] == \"fail\"', Body) :-
	Body = '            return False'.

wam_instruction_branch('elif instr[0] == \"halt\"', Body) :-
	Body = '            self.halt = True
            return True'.

wam_instruction_branch('elif instr[0] == \"allocate\"', Body) :-
	Body = '            frame = EnvFrame(saved_cp=self.cp)
            self.stack.append(frame)
            self.e = len(self.stack) - 1
            self.cut_b = self.b
            return True'.

wam_instruction_branch('elif instr[0] == \"deallocate\"', Body) :-
	Body = '            if self.e >= 0 and self.e < len(self.stack):
                frame = self.stack[self.e]
                if isinstance(frame, EnvFrame):
                    self.cp = frame.saved_cp
                    return True
            return False'.

% --- Choice Point Instructions ---

wam_instruction_branch('elif instr[0] == \"try_me_else\"', Body) :-
	Body = '            label = instr[1]
            n_args = instr[2] if len(instr) > 2 else 20
            next_pc = self.labels.get(label)
            if next_pc is None:
                return False
            push_choice_point(self, n_args, next_pc)
            return True'.

wam_instruction_branch('elif instr[0] == \"retry_me_else\"', Body) :-
	Body = '            label = instr[1]
            next_pc = self.labels.get(label)
            if next_pc is None:
                return False
            if not restore_choice_point(self):
                return False
            if self.b >= 0 and self.b < len(self.stack):
                cp = self.stack[self.b]
                if isinstance(cp, ChoicePoint):
                    cp.next_clause = next_pc
            return True'.

wam_instruction_branch('elif instr[0] == \"trust_me\"', Body) :-
	Body = '            if not restore_choice_point(self):
                return False
            # Pop the choice point
            if self.b >= 0 and self.b < len(self.stack):
                cp = self.stack[self.b]
                if isinstance(cp, ChoicePoint):
                    self.b = cp.saved_b
            return True'.

wam_instruction_branch('elif instr[0] == \"try\"', Body) :-
	Body = '            label = instr[1]
            n_args = instr[2] if len(instr) > 2 else 20
            target_pc = self.labels.get(label)
            if target_pc is None:
                return False
            push_choice_point(self, n_args, self._pc + 1)
            self._pc = target_pc - 1
            return True'.

wam_instruction_branch('elif instr[0] == \"retry\"', Body) :-
	Body = '            label = instr[1]
            target_pc = self.labels.get(label)
            if target_pc is None:
                return False
            if not restore_choice_point(self):
                return False
            if self.b >= 0 and self.b < len(self.stack):
                cp = self.stack[self.b]
                if isinstance(cp, ChoicePoint):
                    cp.next_clause = self._pc + 1
            self._pc = target_pc - 1
            return True'.

wam_instruction_branch('elif instr[0] == \"trust\"', Body) :-
	Body = '            label = instr[1]
            target_pc = self.labels.get(label)
            if target_pc is None:
                return False
            if not restore_choice_point(self):
                return False
            if self.b >= 0 and self.b < len(self.stack):
                cp = self.stack[self.b]
                if isinstance(cp, ChoicePoint):
                    self.b = cp.saved_b
            self._pc = target_pc - 1
            return True'.

% --- Cut Instructions ---

wam_instruction_branch('elif instr[0] == \"neck_cut\"', Body) :-
	Body = '            if self.b > self.cut_b:
                # Remove choice points above cut barrier
                while self.b > self.cut_b and self.b >= 0 and self.b < len(self.stack):
                    cp = self.stack[self.b]
                    if isinstance(cp, ChoicePoint):
                        self.b = cp.saved_b
                    else:
                        break
            return True'.

wam_instruction_branch('elif instr[0] == \"get_level\"', Body) :-
	Body = '            yn = instr[1]
            set_reg(self, yn, Int(self.b))
            return True'.

wam_instruction_branch('elif instr[0] == \"cut\"', Body) :-
	Body = '            yn = instr[1]
            val = get_reg(self, yn)
            if val is not None and isinstance(val, Int):
                saved_b = val.n
                while self.b > saved_b and self.b >= 0 and self.b < len(self.stack):
                    cp = self.stack[self.b]
                    if isinstance(cp, ChoicePoint):
                        self.b = cp.saved_b
                    else:
                        break
                return True
            return False'.

% --- Indexing Instructions ---

wam_instruction_branch('elif instr[0] == \"switch_on_term\"', Body) :-
	Body = '            var_label, const_label, list_label, struct_label = instr[1], instr[2], instr[3], instr[4]
            val = get_reg(self, 1)  # A1
            if val is None or isinstance(val, Var):
                target = self.labels.get(var_label)
            elif isinstance(val, Atom) or isinstance(val, Int) or isinstance(val, Float):
                target = self.labels.get(const_label)
            elif isinstance(val, Compound) and val.functor == \".\":
                target = self.labels.get(list_label)
            elif isinstance(val, Compound):
                target = self.labels.get(struct_label)
            else:
                target = self.labels.get(var_label)
            if target is not None:
                self._pc = target - 1
                return True
            return False'.

wam_instruction_branch('elif instr[0] == \"switch_on_constant\"', Body) :-
	Body = '            table = instr[1]  # dict: value → label
            val = get_reg(self, 1)  # A1
            if val is None:
                return False
            d = deref(val, self)
            if isinstance(d, Var):
                return True  # skip dispatch for unbound
            key = None
            if isinstance(d, Atom):
                key = d.name
            elif isinstance(d, Int):
                key = d.n
            if key is not None and key in table:
                target = self.labels.get(table[key])
                if target is not None:
                    self._pc = target - 1
                    return True
            return False'.

wam_instruction_branch('elif instr[0] == \"switch_on_structure\"', Body) :-
	Body = '            table = instr[1]  # dict: functor/arity → label
            val = get_reg(self, 1)  # A1
            if val is None:
                return False
            d = deref(val, self)
            if isinstance(d, Compound):
                key = f\"{d.functor}/{len(d.args)}\"
                if key in table:
                    target = self.labels.get(table[key])
                    if target is not None:
                        self._pc = target - 1
                        return True
            return False'.

% --- Arithmetic Instructions ---

wam_instruction_branch('elif instr[0] == \"is\"', Body) :-
	Body = '            target_reg, expr_reg = instr[1], instr[2]
            expr_val = get_reg(self, expr_reg)
            if expr_val is None:
                return False
            result = eval_arith(expr_val, self)
            if result is None:
                return False
            if result == int(result):
                result_val = Int(int(result))
            else:
                result_val = Float(result)
            target_val = get_reg(self, target_reg)
            if target_val is None or isinstance(target_val, Var):
                set_reg(self, target_reg, result_val)
                if isinstance(target_val, Var):
                    bind(target_val, result_val, self)
                return True
            return unify(target_val, result_val, self)'.

wam_instruction_branch('elif instr[0] == \"builtin_call\"', Body) :-
	Body = '            op, arity = instr[1], instr[2]
            return self._execute_builtin(op, arity)'.

wam_instruction_branch('elif instr[0] == \"call_foreign\"', Body) :-
	Body = '            pred, arity = instr[1], instr[2]
            args = [get_reg(self, i + 1) for i in range(arity)]
            return execute_foreign(pred, arity, args, self)'.

% ============================================================================
% PHASE 3: Helper predicates → Python functions
% ============================================================================

%% compile_wam_helpers_to_python(+Options, -PythonCode)
%  Generates Python methods for WAM runtime helpers.
compile_wam_helpers_to_python(_Options, PythonCode) :-
	compile_run_loop_to_python(RunLoopCode),
	compile_backtrack_method_to_python(BacktrackCode),
	compile_fetch_to_python(FetchCode),
	compile_execute_builtin_to_python(BuiltinCode),
	compile_collect_results_to_python(CollectCode),
	atomic_list_concat([
		RunLoopCode, '\n\n',
		BacktrackCode, '\n\n',
		FetchCode, '\n\n',
		BuiltinCode, '\n\n',
		CollectCode
	], PythonCode).

compile_run_loop_to_python(Code) :-
	Code = '    def run(self):
        """Main execution loop. Runs until halt, failure, or step limit."""
        self._pc = 0
        while True:
            if self.halt:
                return True
            if self.step_limit > 0 and self.step_count >= self.step_limit:
                return False
            instr = self._fetch()
            if instr is None:
                return False
            if not self.step(instr):
                if not self._backtrack():
                    return False
            else:
                self._pc += 1
            self.step_count += 1'.

compile_backtrack_method_to_python(Code) :-
	Code = '    def _backtrack(self):
        """Restore state from the top choice point."""
        while self.b >= 0 and self.b < len(self.stack):
            cp = self.stack[self.b]
            if not isinstance(cp, ChoicePoint):
                return False
            # Undo trail bindings
            unwind_trail(self, cp.trail_top)
            # Trim heap (dict-based)
            heap_trim(self, cp.heap_top)
            # Restore registers
            for i in range(len(cp.saved_regs)):
                self.regs[i] = cp.saved_regs[i]
            # Restore control
            self.e = cp.saved_e
            self.cp = cp.saved_cp
            self.cut_b = cp.cut_b
            # Jump to next clause
            self._pc = cp.next_clause - 1
            return True
        return False'.

compile_fetch_to_python(Code) :-
	Code = '    def _fetch(self):
        """Fetch the instruction at the current PC."""
        if 0 <= self._pc < len(self.code):
            return self.code[self._pc]
        return None'.

compile_execute_builtin_to_python(Code) :-
	Code = '    def _execute_builtin(self, op, arity):
        """Execute a built-in predicate by name."""
        # Arithmetic comparisons
        if op == \"=:=\" and arity == 2:
            a = eval_arith(get_reg(self, 1), self)
            b = eval_arith(get_reg(self, 2), self)
            return a is not None and b is not None and a == b
        if op == \"=\\\\=\" and arity == 2:
            a = eval_arith(get_reg(self, 1), self)
            b = eval_arith(get_reg(self, 2), self)
            return a is not None and b is not None and a != b
        if op == \"<\" and arity == 2:
            a = eval_arith(get_reg(self, 1), self)
            b = eval_arith(get_reg(self, 2), self)
            return a is not None and b is not None and a < b
        if op == \">\" and arity == 2:
            a = eval_arith(get_reg(self, 1), self)
            b = eval_arith(get_reg(self, 2), self)
            return a is not None and b is not None and a > b
        if op == \">=\" and arity == 2:
            a = eval_arith(get_reg(self, 1), self)
            b = eval_arith(get_reg(self, 2), self)
            return a is not None and b is not None and a >= b
        if op == \"=<\" and arity == 2:
            a = eval_arith(get_reg(self, 1), self)
            b = eval_arith(get_reg(self, 2), self)
            return a is not None and b is not None and a <= b
        # Unification
        if op == \"=\" and arity == 2:
            return unify(get_reg(self, 1), get_reg(self, 2), self)
        if op == \"\\\\=\" and arity == 2:
            return not unify(get_reg(self, 1), get_reg(self, 2), self)
        # Type checks
        if op == \"atom\" and arity == 1:
            d = deref(get_reg(self, 1), self)
            return isinstance(d, Atom)
        if op == \"integer\" and arity == 1:
            d = deref(get_reg(self, 1), self)
            return isinstance(d, Int)
        if op == \"float\" and arity == 1:
            d = deref(get_reg(self, 1), self)
            return isinstance(d, Float)
        if op == \"number\" and arity == 1:
            d = deref(get_reg(self, 1), self)
            return isinstance(d, (Int, Float))
        if op == \"var\" and arity == 1:
            d = deref(get_reg(self, 1), self)
            return isinstance(d, Var)
        if op == \"nonvar\" and arity == 1:
            d = deref(get_reg(self, 1), self)
            return not isinstance(d, Var)
        if op == \"compound\" and arity == 1:
            d = deref(get_reg(self, 1), self)
            return isinstance(d, Compound)
        # I/O
        if op == \"write\" and arity == 1:
            d = deref(get_reg(self, 1), self)
            print(_format_value(d), end=\"\")
            return True
        if op == \"writeln\" and arity == 1:
            d = deref(get_reg(self, 1), self)
            print(_format_value(d))
            return True
        if op == \"nl\" and arity == 0:
            print()
            return True
        # Term manipulation
        if op == \"copy_term\" and arity == 2:
            original = deref(get_reg(self, 1), self)
            copy = _deep_copy_term(original, {}, self)
            set_reg(self, 2, copy)
            return True
        if op == \"functor\" and arity == 3:
            d = deref(get_reg(self, 1), self)
            if isinstance(d, Compound):
                return (unify(get_reg(self, 2), Atom(d.functor), self) and
                        unify(get_reg(self, 3), Int(len(d.args)), self))
            if isinstance(d, Atom):
                return (unify(get_reg(self, 2), d, self) and
                        unify(get_reg(self, 3), Int(0), self))
            return False
        if op == \"=..\" and arity == 2:
            d = deref(get_reg(self, 1), self)
            if isinstance(d, Compound):
                lst = Atom(d.functor)
                for arg in reversed(d.args):
                    lst = Compound(\".\", [arg, lst])
                return unify(get_reg(self, 2), lst, self)
            if isinstance(d, Atom):
                return unify(get_reg(self, 2), Compound(\".\", [d, Atom(\"[]\")]), self)
            return False
        return False'.

compile_collect_results_to_python(Code) :-
	Code = '    def collect_results(self, arity):
        """Gather results from A1..An registers after a successful run."""
        results = []
        for i in range(1, arity + 1):
            val = get_reg(self, i)
            if val is not None:
                results.append(val)
            else:
                results.append(None)
        return results'.

% ============================================================================
% Utility: format Value for printing
% ============================================================================

compile_format_value_to_python(Code) :-
	Code = '
def _format_value(val):
    """Format a Value for Prolog-style output."""
    if isinstance(val, Atom):
        return val.name
    if isinstance(val, Int):
        return str(val.n)
    if isinstance(val, Float):
        return str(val.f)
    if isinstance(val, Compound):
        if val.functor == \".\" and len(val.args) == 2:
            return \"[\" + _format_list(val) + \"]\"
        args_str = \", \".join(_format_value(a) for a in val.args)
        return f\"{val.functor}({args_str})\"
    if isinstance(val, Var):
        return \"_\" if val.ref is None else _format_value(val.ref)
    if isinstance(val, Ref):
        return f\"ref({val.addr})\"
    return str(val)

def _format_list(val):
    """Format a ./2 cons-cell chain as a Prolog list."""
    items = []
    current = val
    while isinstance(current, Compound) and current.functor == \".\" and len(current.args) == 2:
        items.append(_format_value(current.args[0]))
        current = current.args[1]
    if isinstance(current, Atom) and current.name == \"[]\":
        return \", \".join(items)
    items.append(\"|\" + _format_value(current))
    return \", \".join(items)

def _deep_copy_term(val, var_map, state):
    """Deep copy a term, renaming variables."""
    if isinstance(val, Var):
        vid = id(val)
        if vid in var_map:
            return var_map[vid]
        new_var = Var()
        var_map[vid] = new_var
        if val.ref is not None:
            new_var.ref = _deep_copy_term(val.ref, var_map, state)
        return new_var
    if isinstance(val, Compound):
        new_args = [_deep_copy_term(a, var_map, state) for a in val.args]
        return Compound(val.functor, new_args)
    return val'.

% ============================================================================
% PHASE 4: WAM Instruction Serialization
% ============================================================================

%% python_value_literal(+PrologTerm, -PyLiteral)
%  Convert a Prolog term to a Python value literal.
python_value_literal(atom(A), PyVal) :- !,
	format(atom(PyVal), 'Atom("~w")', [A]).
python_value_literal(integer(I), PyVal) :- !,
	format(atom(PyVal), 'Int(~w)', [I]).
python_value_literal(N, PyVal) :- integer(N), !,
	format(atom(PyVal), 'Int(~w)', [N]).
python_value_literal(N, PyVal) :- float(N), !,
	format(atom(PyVal), 'Float(~w)', [N]).
python_value_literal(A, PyVal) :- atom(A), !,
	format(atom(PyVal), 'Atom("~w")', [A]).
python_value_literal(S, PyVal) :- string(S), !,
	format(atom(PyVal), 'Atom("~w")', [S]).

%% wam_instruction_to_python_literal(+WamInstr, -PyLiteral)
%  Convert WAM instruction term to Python tuple literal.

wam_instruction_to_python_literal(get_constant(C, Ai), Py) :-
	python_value_literal(C, PyVal),
	format(atom(Py), '("get_constant", ~w, ~w)', [PyVal, Ai]).
wam_instruction_to_python_literal(get_variable(Xn, Ai), Py) :-
	format(atom(Py), '("get_variable", ~w, ~w)', [Xn, Ai]).
wam_instruction_to_python_literal(get_value(Xn, Ai), Py) :-
	format(atom(Py), '("get_value", ~w, ~w)', [Xn, Ai]).
wam_instruction_to_python_literal(get_structure(Fn, Arity, Ai), Py) :-
	format(atom(Py), '("get_structure", "~w", ~w, ~w)', [Fn, Arity, Ai]).
wam_instruction_to_python_literal(get_list(Ai), Py) :-
	format(atom(Py), '("get_list", ~w)', [Ai]).
wam_instruction_to_python_literal(get_nil(Ai), Py) :-
	format(atom(Py), '("get_nil", ~w)', [Ai]).
wam_instruction_to_python_literal(get_integer(N, Ai), Py) :-
	format(atom(Py), '("get_integer", ~w, ~w)', [N, Ai]).
wam_instruction_to_python_literal(get_float(F, Ai), Py) :-
	format(atom(Py), '("get_float", ~w, ~w)', [F, Ai]).

wam_instruction_to_python_literal(unify_variable(Xn), Py) :-
	format(atom(Py), '("unify_variable", ~w)', [Xn]).
wam_instruction_to_python_literal(unify_value(Xn), Py) :-
	format(atom(Py), '("unify_value", ~w)', [Xn]).
wam_instruction_to_python_literal(unify_constant(C), Py) :-
	python_value_literal(C, PyVal),
	format(atom(Py), '("unify_constant", ~w)', [PyVal]).
wam_instruction_to_python_literal(unify_nil, Py) :-
	Py = '("unify_nil",)'.
wam_instruction_to_python_literal(unify_void(N), Py) :-
	format(atom(Py), '("unify_void", ~w)', [N]).

wam_instruction_to_python_literal(put_variable(Xn, Ai), Py) :-
	format(atom(Py), '("put_variable", ~w, ~w)', [Xn, Ai]).
wam_instruction_to_python_literal(put_value(Xn, Ai), Py) :-
	format(atom(Py), '("put_value", ~w, ~w)', [Xn, Ai]).
wam_instruction_to_python_literal(put_unsafe_value(Yn, Ai), Py) :-
	format(atom(Py), '("put_unsafe_value", ~w, ~w)', [Yn, Ai]).
wam_instruction_to_python_literal(put_constant(C, Ai), Py) :-
	python_value_literal(C, PyVal),
	format(atom(Py), '("put_constant", ~w, ~w)', [PyVal, Ai]).
wam_instruction_to_python_literal(put_nil(Ai), Py) :-
	format(atom(Py), '("put_nil", ~w)', [Ai]).
wam_instruction_to_python_literal(put_integer(N, Ai), Py) :-
	format(atom(Py), '("put_integer", ~w, ~w)', [N, Ai]).
wam_instruction_to_python_literal(put_float(F, Ai), Py) :-
	format(atom(Py), '("put_float", ~w, ~w)', [F, Ai]).
wam_instruction_to_python_literal(put_structure(Fn, Ai, Arity), Py) :-
	format(atom(Py), '("put_structure", "~w", ~w, ~w)', [Fn, Arity, Ai]).
wam_instruction_to_python_literal(put_list(Ai), Py) :-
	format(atom(Py), '("put_list", ~w)', [Ai]).

wam_instruction_to_python_literal(call(P, N), Py) :-
	format(atom(Py), '("call", "~w", ~w)', [P, N]).
wam_instruction_to_python_literal(execute(P), Py) :-
	format(atom(Py), '("execute", "~w")', [P]).
wam_instruction_to_python_literal(proceed, Py) :-
	Py = '("proceed",)'.
wam_instruction_to_python_literal(fail, Py) :-
	Py = '("fail",)'.
wam_instruction_to_python_literal(halt, Py) :-
	Py = '("halt",)'.
wam_instruction_to_python_literal(allocate, Py) :-
	Py = '("allocate",)'.
wam_instruction_to_python_literal(deallocate, Py) :-
	Py = '("deallocate",)'.

wam_instruction_to_python_literal(try_me_else(Label), Py) :-
	format(atom(Py), '("try_me_else", "~w")', [Label]).
wam_instruction_to_python_literal(retry_me_else(Label), Py) :-
	format(atom(Py), '("retry_me_else", "~w")', [Label]).
wam_instruction_to_python_literal(trust_me, Py) :-
	Py = '("trust_me",)'.
wam_instruction_to_python_literal(try(Label), Py) :-
	format(atom(Py), '("try", "~w")', [Label]).
wam_instruction_to_python_literal(retry(Label), Py) :-
	format(atom(Py), '("retry", "~w")', [Label]).
wam_instruction_to_python_literal(trust(Label), Py) :-
	format(atom(Py), '("trust", "~w")', [Label]).

wam_instruction_to_python_literal(neck_cut, Py) :-
	Py = '("neck_cut",)'.
wam_instruction_to_python_literal(get_level(Yn), Py) :-
	format(atom(Py), '("get_level", ~w)', [Yn]).
wam_instruction_to_python_literal(cut(Yn), Py) :-
	format(atom(Py), '("cut", ~w)', [Yn]).

wam_instruction_to_python_literal(switch_on_term(V, C, L, S), Py) :-
	format(atom(Py), '("switch_on_term", "~w", "~w", "~w", "~w")', [V, C, L, S]).
wam_instruction_to_python_literal(switch_on_constant(Table), Py) :-
	switch_table_to_python_dict(Table, PyDict),
	format(atom(Py), '("switch_on_constant", ~w)', [PyDict]).
wam_instruction_to_python_literal(switch_on_structure(Table), Py) :-
	switch_table_to_python_dict(Table, PyDict),
	format(atom(Py), '("switch_on_structure", ~w)', [PyDict]).

wam_instruction_to_python_literal(is(Target, Expr), Py) :-
	format(atom(Py), '("is", ~w, ~w)', [Target, Expr]).
wam_instruction_to_python_literal(builtin_call(Op, Arity), Py) :-
	format(atom(Py), '("builtin_call", "~w", ~w)', [Op, Arity]).
wam_instruction_to_python_literal(call_foreign(Pred, Arity), Py) :-
	format(atom(Py), '("call_foreign", "~w", ~w)', [Pred, Arity]).

%% switch_table_to_python_dict(+Table, -PyDict)
%  Convert a switch table (list of Key-Label pairs) to a Python dict literal.
switch_table_to_python_dict(Table, PyDict) :-
	(   is_list(Table)
	->  maplist(switch_entry_to_python, Table, Entries),
		atomic_list_concat(Entries, ', ', EntriesStr),
		format(atom(PyDict), '{~w}', [EntriesStr])
	;   PyDict = '{}'
	).

switch_entry_to_python(Key-Label, Entry) :-
	(   atom(Key) -> format(atom(Entry), '"~w": "~w"', [Key, Label])
	;   integer(Key) -> format(atom(Entry), '~w: "~w"', [Key, Label])
	;   format(atom(Entry), '"~w": "~w"', [Key, Label])
	).

% ============================================================================
% WAM line parsing → Python instruction literals
% ============================================================================

%% wam_line_to_python_literal(+Parts, -PyLit)
%  Parse a split WAM text line into a Python instruction literal.

%% clean_comma(+S, -Clean)
%  Remove trailing comma from a string/atom.
clean_comma(S, Clean) :-
	atom_string(S, Str),
	(   sub_string(Str, Before, 1, 0, ",")
	->  sub_string(Str, 0, Before, _, CleanStr),
		atom_string(Clean, CleanStr)
	;   Clean = S
	).

%% escape_python_string(+In, -Out)
%  Escape a string for Python string literal.
escape_python_string(In, Out) :-
	atom_string(In, S),
	split_string(S, "\\", "", Parts),
	atomic_list_concat(Parts, "\\\\", S1),
	split_string(S1, "\"", "", Parts2),
	atomic_list_concat(Parts2, "\\\"", Out).

wam_line_to_python_literal(["get_constant", C, Ai], Py) :-
	clean_comma(C, CC), clean_comma(Ai, CAi),
	escape_python_string(CC, ECC),
	format(atom(Py), '("get_constant", Atom("~w"), ~w)', [ECC, CAi]).
wam_line_to_python_literal(["get_variable", Xn, Ai], Py) :-
	clean_comma(Xn, CXn), clean_comma(Ai, CAi),
	format(atom(Py), '("get_variable", ~w, ~w)', [CXn, CAi]).
wam_line_to_python_literal(["get_value", Xn, Ai], Py) :-
	clean_comma(Xn, CXn), clean_comma(Ai, CAi),
	format(atom(Py), '("get_value", ~w, ~w)', [CXn, CAi]).
wam_line_to_python_literal(["get_structure", Fn, Ai], Py) :-
	clean_comma(Fn, CFn), clean_comma(Ai, CAi),
	(   split_string(CFn, "/", "", [_Name, ArStr])
	->  number_string(Arity, ArStr)
	;   Arity = 0
	),
	escape_python_string(CFn, EFn),
	format(atom(Py), '("get_structure", "~w", ~w, ~w)', [EFn, Arity, CAi]).
wam_line_to_python_literal(["get_list", Ai], Py) :-
	clean_comma(Ai, CAi),
	format(atom(Py), '("get_list", ~w)', [CAi]).
wam_line_to_python_literal(["get_nil", Ai], Py) :-
	clean_comma(Ai, CAi),
	format(atom(Py), '("get_nil", ~w)', [CAi]).
wam_line_to_python_literal(["get_integer", N, Ai], Py) :-
	clean_comma(N, CN), clean_comma(Ai, CAi),
	format(atom(Py), '("get_integer", ~w, ~w)', [CN, CAi]).
wam_line_to_python_literal(["get_float", F, Ai], Py) :-
	clean_comma(F, CF), clean_comma(Ai, CAi),
	format(atom(Py), '("get_float", ~w, ~w)', [CF, CAi]).
wam_line_to_python_literal(["unify_variable", Xn], Py) :-
	clean_comma(Xn, CXn),
	format(atom(Py), '("unify_variable", ~w)', [CXn]).
wam_line_to_python_literal(["unify_value", Xn], Py) :-
	clean_comma(Xn, CXn),
	format(atom(Py), '("unify_value", ~w)', [CXn]).
wam_line_to_python_literal(["unify_constant", C], Py) :-
	clean_comma(C, CC),
	escape_python_string(CC, ECC),
	format(atom(Py), '("unify_constant", Atom("~w"))', [ECC]).
wam_line_to_python_literal(["unify_nil"], Py) :-
	Py = '("unify_nil",)'.
wam_line_to_python_literal(["unify_void", N], Py) :-
	clean_comma(N, CN),
	format(atom(Py), '("unify_void", ~w)', [CN]).
% --- set_* instructions (always write mode, used after put_structure/put_list) ---
wam_line_to_python_literal(["set_variable", Xn], Py) :-
	clean_comma(Xn, CXn),
	format(atom(Py), '("set_variable", ~w)', [CXn]).
wam_line_to_python_literal(["set_value", Xn], Py) :-
	clean_comma(Xn, CXn),
	format(atom(Py), '("set_value", ~w)', [CXn]).
wam_line_to_python_literal(["set_local_value", Xn], Py) :-
	clean_comma(Xn, CXn),
	format(atom(Py), '("set_local_value", ~w)', [CXn]).
wam_line_to_python_literal(["set_constant", C], Py) :-
	clean_comma(C, CC),
	escape_python_string(CC, ECC),
	format(atom(Py), '("set_constant", Atom("~w"))', [ECC]).
wam_line_to_python_literal(["set_nil"], Py) :-
	Py = '("set_nil",)'.
wam_line_to_python_literal(["set_integer", N], Py) :-
	clean_comma(N, CN),
	format(atom(Py), '("set_integer", ~w)', [CN]).
wam_line_to_python_literal(["set_void", N], Py) :-
	clean_comma(N, CN),
	format(atom(Py), '("set_void", ~w)', [CN]).
wam_line_to_python_literal(["put_variable", Xn, Ai], Py) :-
	clean_comma(Xn, CXn), clean_comma(Ai, CAi),
	format(atom(Py), '("put_variable", ~w, ~w)', [CXn, CAi]).
wam_line_to_python_literal(["put_value", Xn, Ai], Py) :-
	clean_comma(Xn, CXn), clean_comma(Ai, CAi),
	format(atom(Py), '("put_value", ~w, ~w)', [CXn, CAi]).
wam_line_to_python_literal(["put_unsafe_value", Yn, Ai], Py) :-
	clean_comma(Yn, CYn), clean_comma(Ai, CAi),
	format(atom(Py), '("put_unsafe_value", ~w, ~w)', [CYn, CAi]).
wam_line_to_python_literal(["put_constant", C, Ai], Py) :-
	clean_comma(C, CC), clean_comma(Ai, CAi),
	escape_python_string(CC, ECC),
	format(atom(Py), '("put_constant", Atom("~w"), ~w)', [ECC, CAi]).
wam_line_to_python_literal(["put_nil", Ai], Py) :-
	clean_comma(Ai, CAi),
	format(atom(Py), '("put_nil", ~w)', [CAi]).
wam_line_to_python_literal(["put_integer", N, Ai], Py) :-
	clean_comma(N, CN), clean_comma(Ai, CAi),
	format(atom(Py), '("put_integer", ~w, ~w)', [CN, CAi]).
wam_line_to_python_literal(["put_float", F, Ai], Py) :-
	clean_comma(F, CF), clean_comma(Ai, CAi),
	format(atom(Py), '("put_float", ~w, ~w)', [CF, CAi]).
wam_line_to_python_literal(["put_structure", Fn, Ai], Py) :-
	clean_comma(Fn, CFn), clean_comma(Ai, CAi),
	(   split_string(CFn, "/", "", [_Name, ArStr])
	->  number_string(Arity, ArStr)
	;   Arity = 0
	),
	escape_python_string(CFn, EFn),
	format(atom(Py), '("put_structure", "~w", ~w, ~w)', [EFn, Arity, CAi]).
wam_line_to_python_literal(["put_list", Ai], Py) :-
	clean_comma(Ai, CAi),
	format(atom(Py), '("put_list", ~w)', [CAi]).
wam_line_to_python_literal(["call", P, N], Py) :-
	clean_comma(P, CP), clean_comma(N, CN),
	escape_python_string(CP, EP),
	format(atom(Py), '("call", "~w", ~w)', [EP, CN]).
wam_line_to_python_literal(["execute", P], Py) :-
	clean_comma(P, CP),
	escape_python_string(CP, EP),
	format(atom(Py), '("execute", "~w")', [EP]).
wam_line_to_python_literal(["proceed"], Py) :-
	Py = '("proceed",)'.
wam_line_to_python_literal(["fail"], Py) :-
	Py = '("fail",)'.
wam_line_to_python_literal(["halt"], Py) :-
	Py = '("halt",)'.
wam_line_to_python_literal(["allocate"], Py) :-
	Py = '("allocate",)'.
wam_line_to_python_literal(["deallocate"], Py) :-
	Py = '("deallocate",)'.
wam_line_to_python_literal(["try_me_else", Label], Py) :-
	clean_comma(Label, CL),
	format(atom(Py), '("try_me_else", "~w")', [CL]).
wam_line_to_python_literal(["retry_me_else", Label], Py) :-
	clean_comma(Label, CL),
	format(atom(Py), '("retry_me_else", "~w")', [CL]).
wam_line_to_python_literal(["trust_me"], Py) :-
	Py = '("trust_me",)'.
wam_line_to_python_literal(["try", Label], Py) :-
	clean_comma(Label, CL),
	format(atom(Py), '("try", "~w")', [CL]).
wam_line_to_python_literal(["retry", Label], Py) :-
	clean_comma(Label, CL),
	format(atom(Py), '("retry", "~w")', [CL]).
wam_line_to_python_literal(["trust", Label], Py) :-
	clean_comma(Label, CL),
	format(atom(Py), '("trust", "~w")', [CL]).
wam_line_to_python_literal(["neck_cut"], Py) :-
	Py = '("neck_cut",)'.
wam_line_to_python_literal(["get_level", Yn], Py) :-
	clean_comma(Yn, CYn),
	format(atom(Py), '("get_level", ~w)', [CYn]).
wam_line_to_python_literal(["cut", Yn], Py) :-
	clean_comma(Yn, CYn),
	format(atom(Py), '("cut", ~w)', [CYn]).
wam_line_to_python_literal(["switch_on_term"|Args], Py) :-
	maplist(clean_comma, Args, CArgs),
	CArgs = [V, C, L, S],
	format(atom(Py), '("switch_on_term", "~w", "~w", "~w", "~w")', [V, C, L, S]).
wam_line_to_python_literal(["is", Target, Expr], Py) :-
	clean_comma(Target, CT), clean_comma(Expr, CE),
	format(atom(Py), '("is", ~w, ~w)', [CT, CE]).
wam_line_to_python_literal(["builtin_call", Op, Arity], Py) :-
	clean_comma(Op, COp), clean_comma(Arity, CArity),
	escape_python_string(COp, EOp),
	format(atom(Py), '("builtin_call", "~w", ~w)', [EOp, CArity]).
wam_line_to_python_literal(["call_foreign", Pred, Arity], Py) :-
	clean_comma(Pred, CP), clean_comma(Arity, CA),
	escape_python_string(CP, EP),
	format(atom(Py), '("call_foreign", "~w", ~w)', [EP, CA]).
% --- aggregate instructions ---
wam_line_to_python_literal(["begin_aggregate", AggType, ValueReg, ResultReg], Py) :-
	clean_comma(AggType, CAT),
	clean_comma(ValueReg, CVR),
	clean_comma(ResultReg, CRR),
	format(atom(Py), '("begin_aggregate", "~w", ~w, ~w)', [CAT, CVR, CRR]).
wam_line_to_python_literal(["end_aggregate", ValueReg], Py) :-
	clean_comma(ValueReg, CVR),
	format(atom(Py), '("end_aggregate", ~w)', [CVR]).
% --- cut_ite and jump ---
wam_line_to_python_literal(["cut_ite"], Py) :-
	Py = '("cut_ite",)'.
wam_line_to_python_literal(["jump", Label], Py) :-
	clean_comma(Label, CL),
	escape_python_string(CL, EL),
	format(atom(Py), '("jump", "~w")', [EL]).
% --- Indexed-fact + category-ancestor kernel instructions (Rust parity) ---
wam_line_to_python_literal(["call_indexed_atom_fact2", Pred], Py) :-
	clean_comma(Pred, CP),
	escape_python_string(CP, EP),
	format(atom(Py), '("call_indexed_atom_fact2", "~w")', [EP]).
wam_line_to_python_literal(["base_category_ancestor", CatReg, TargetReg, VisitedReg], Py) :-
	clean_comma(CatReg, CC),
	clean_comma(TargetReg, CT),
	clean_comma(VisitedReg, CV),
	format(atom(Py), '("base_category_ancestor", ~w, ~w, ~w)', [CC, CT, CV]).
wam_line_to_python_literal(["base_category_ancestor_bind", CatReg, TargetReg, HopsReg, VisitedReg], Py) :-
	clean_comma(CatReg, CC),
	clean_comma(TargetReg, CT),
	clean_comma(HopsReg, CH),
	clean_comma(VisitedReg, CV),
	format(atom(Py), '("base_category_ancestor_bind", ~w, ~w, ~w, ~w)', [CC, CT, CH, CV]).
wam_line_to_python_literal(["recurse_category_ancestor", MidReg, RootReg, ChildHopsReg, VisitedReg, Pred, Skip], Py) :-
	clean_comma(MidReg, CM),
	clean_comma(RootReg, CR),
	clean_comma(ChildHopsReg, CCH),
	clean_comma(VisitedReg, CV),
	clean_comma(Pred, CP),
	clean_comma(Skip, CSk),
	escape_python_string(CP, EP),
	format(atom(Py), '("recurse_category_ancestor", ~w, ~w, ~w, ~w, "~w", ~w)',
		[CM, CR, CCH, CV, EP, CSk]).
wam_line_to_python_literal(["return_add1", OutReg, InReg], Py) :-
	clean_comma(OutReg, CO),
	clean_comma(InReg, CI),
	format(atom(Py), '("return_add1", ~w, ~w)', [CO, CI]).

% ============================================================================
% PHASE 5: Predicate Compilation
% ============================================================================

%% python_safe_id(+Atom, -SafeStr)
%  Convert a Prolog atom to a valid Python identifier by replacing
%  illegal characters: '$' -> '__dollar__', '-' -> '_', '\\+' -> 'not_'
python_safe_id(Atom, SafeStr) :-
	atom_string(Atom, Str),
	sub_string_replace(Str, "$", "__", Str1),
	sub_string_replace(Str1, "-", "_", Str2),
	sub_string_replace(Str2, "\\+", "not_", SafeStr).

%% sub_string_replace(+Str, +From, +To, -Result)
sub_string_replace(Str, From, To, Result) :-
	(   sub_string(Str, Before, _, After, From)
	->  sub_string(Str, 0, Before, _, Prefix),
	    sub_string(Str, _, After, 0, Suffix),
	    sub_string_replace(Suffix, From, To, RestReplaced),
	    string_concat(Prefix, To, PrefixTo),
	    string_concat(PrefixTo, RestReplaced, Result)
	;   Result = Str
	).

%% compile_wam_predicate_to_python(+Pred/Arity, +WamCode, +Options, -PythonCode)
%  Converts WAM instruction output for a predicate to Python code.
%  Emits a register_predicate(raw_program) call so the flat program can be
%  built by load_program in main.py.
compile_wam_predicate_to_python(Pred/Arity, WamCode, Options, PythonCode) :-
	atom_string(Pred, PredStr),
	python_func_name(Pred/Arity, FuncName),
	option(registrar_prefix(RegistrarPrefix), Options, ''),
	atom_concat(RegistrarPrefix, FuncName, RegistrarName),
	% Build the label key: "Pred/Arity"
	format(atom(LabelKey), '~w/~w', [PredStr, Arity]),
	wam_code_to_python_instructions(WamCode, Pred/Arity, InstrLiterals, _LabelLiterals),
	format(string(PythonCode),
'def ~w(raw_program):
    """Register WAM code for ~w/~w into raw_program dict."""
    raw_program["~w"] = (
~w
    )
', [RegistrarName, PredStr, Arity, LabelKey, InstrLiterals]).

%% compile_lowered_wam_predicate_to_python(+Pred/Arity, +WamCode, +Options, -PythonCode)
%  Emits a direct lowered Python predicate plus a separate registrar that
%  inserts a call_lowered stub into raw_program. The separate registrar avoids
%  a name collision with the pred_* lowered function itself.
compile_lowered_wam_predicate_to_python(Pred/Arity, WamCode, Options, PythonCode) :-
	parse_wam_text_py(WamCode, Instrs),
	is_deterministic_pred_py(Instrs),
	emit_lowered_python(Pred/Arity, Instrs, Options, LoweredFn),
	atom_string(Pred, PredStr),
	python_func_name(Pred/Arity, FuncName),
	atom_concat(register_, FuncName, RegistrarName),
	format(atom(LabelKey), '~w/~w', [PredStr, Arity]),
	format(string(Registrar),
'def ~w(raw_program):
    """Register lowered WAM code for ~w/~w into raw_program dict."""
    raw_program["~w"] = (
        ("call_lowered", ~w, ~w),
        ("proceed",),
    )
', [RegistrarName, PredStr, Arity, LabelKey, FuncName, Arity]),
	atomic_list_concat([LoweredFn, '\n\n', Registrar], PythonCode).

%% build_python_wam_arg_list(+Arity, -ArgList)
%  Build the Python function argument list.
build_python_wam_arg_list(0, "args_state") :- !.
build_python_wam_arg_list(Arity, ArgList) :-
	numlist(1, Arity, Indices),
	maplist([I, S]>>format(atom(S), "a~w", [I]), Indices, Parts),
	atomic_list_concat(['args_state'|Parts], ', ', ArgList).

%% build_python_wam_arg_setup(+Arity, -Setup)
%  Build register setup code for predicate arguments.
build_python_wam_arg_setup(0, "") :- !.
build_python_wam_arg_setup(Arity, Setup) :-
	numlist(1, Arity, Indices),
	maplist([I, S]>>format(string(S),
		'    set_reg(state, ~w, a~w)', [I, I]), Indices, Lines),
	atomic_list_concat(Lines, '\n', Setup).

%% wam_code_to_python_instructions(+WamCode, +PredIndicator, -InstrLiterals, -LabelLiterals)
%  Parse WAM text code into Python instruction literals and label assignments.
wam_code_to_python_instructions(WamCode, _PredIndicator, InstrLiterals, LabelLiterals) :-
	atom_string(WamCode, WamStr),
	split_string(WamStr, "\n", "", Lines),
	wam_lines_to_python(Lines, 0, InstrParts, LabelParts),
	atomic_list_concat(InstrParts, '\n', InstrLiterals),
	atomic_list_concat(LabelParts, '\n', LabelLiterals).

wam_lines_to_python([], _, [], []).
wam_lines_to_python([Line|Rest], PC, Instrs, Labels) :-
	split_string(Line, " \t,", " \t,", Parts),
	delete(Parts, "", CleanParts),
	(   CleanParts == []
	->  wam_lines_to_python(Rest, PC, Instrs, Labels)
	;   CleanParts = [First|_],
		(   % Label line: "pred/2:" or "L_label:"
			sub_string(First, _, 1, 0, ":")
		->  sub_string(First, 0, _, 1, LabelName),
			% Emit a __label__ pseudo-instruction so load_program can record the PC
			format(string(LabelEntry), '        ("__label__", "~w"),', [LabelName]),
			Labels = [LabelEntry|RestLabels],
			Instrs = [LabelEntry|RestInstrs],
			wam_lines_to_python(Rest, PC, RestInstrs, RestLabels)
		;   % Instruction line
			(   wam_line_to_python_literal(CleanParts, PyInstr)
			->  format(string(InstrEntry), '        ~w,', [PyInstr]),
				NPC is PC + 1,
				Instrs = [InstrEntry|RestInstrs],
				wam_lines_to_python(Rest, NPC, RestInstrs, Labels)
			;   % Unknown instruction — skip with comment
				atomic_list_concat(CleanParts, " ", LineStr),
				format(string(InstrEntry), '        # SKIP: ~w', [LineStr]),
				NPC is PC + 1,
				Instrs = [InstrEntry|RestInstrs],
				wam_lines_to_python(Rest, NPC, RestInstrs, Labels)
			)
		)
	).

% ============================================================================
% PHASE 6: compile_wam_runtime_to_python
% ============================================================================

%% compile_wam_runtime_to_python(+Options, -PythonCode)
%  Generates the complete runtime Python code (types + step + helpers).
compile_wam_runtime_to_python(Options, PythonCode) :-
	python_wam_type_header(TypeHeader),
	compile_step_wam_to_python(Options, StepCode),
	compile_wam_helpers_to_python(Options, HelpersCode),
	compile_format_value_to_python(FormatCode),
	atomic_list_concat([
		TypeHeader, '\n\n',
		FormatCode, '\n\n',
		'# === WamState methods (mixed into class) ===\n',
		StepCode, '\n\n',
		HelpersCode
	], PythonCode).

% ============================================================================
% PHASE 7: Project Generation
% ============================================================================

%% write_wam_python_project(+Predicates, +Options, +ProjectDir)
%  Generates a runnable Python WAM project in ProjectDir.
%
%  Generated files:
%    - wam_runtime.py   — WAM runtime (copied from wam_python_runtime/ or
%                         generated from bindings if the static file is absent)
%    - predicates.py    — compiled predicates (via compile_wam_predicate_to_python/4)
%    - main.py          — entry point: parses argv, runs a named predicate,
%                         prints register results
%    - __init__.py      — empty; marks the directory as a Python package
%
%  Options recognised:
%    module_name(Name)  — Python module name (default: 'wam_generated')
%    project_dir(Dir)   — overridden by ProjectDir argument
%    emit_mode(Mode)    — interpreter (default) or lowered
%    parallel(true)     — emit a __main__ block that calls run_parallel
%
%  Follows the same layout as write_wam_fsharp_project/3 in wam_fsharp_target.pl.
write_wam_python_project(Predicates, Options, ProjectDir) :-
	option(module_name(ModuleName), Options, 'wam_generated'),

	% Create directory structure
	make_directory_path(ProjectDir),

	% Copy static runtime
	copy_static_runtime(ProjectDir),

	% Generate __init__.py (makes directory a Python package)
	directory_file_path(ProjectDir, '__init__.py', InitPath),
	write_file(InitPath, ""),

	% Generate predicates.py
	compile_all_predicates(Predicates, Options, PredicatesCode),
	directory_file_path(ProjectDir, 'predicates.py', PredPath),
	write_file(PredPath, PredicatesCode),

	% Generate main.py — benchmark driver > parallel > plain
	(   option(benchmark(true), Options)
	->  generate_benchmark_main_py(Predicates, ModuleName, MainCode)
	;   option(parallel(true), Options)
	->  generate_parallel_main_py(Predicates, ModuleName, MainCode)
	;   generate_main_py(Predicates, ModuleName, MainCode)
	),
	directory_file_path(ProjectDir, 'main.py', MainPath),
	write_file(MainPath, MainCode),

	format('WAM Python project created at: ~w~n', [ProjectDir]),
	format('  Predicates compiled: ~w~n', [Predicates]).

%% copy_static_runtime(+ProjectDir)
%  Copy the static WamRuntime.py into the project directory.
%  Uses module source location for reliable path resolution.
copy_static_runtime(ProjectDir) :-
	directory_file_path(ProjectDir, 'wam_runtime.py', DestPath),
	% Resolve relative to this module's source file
	(   source_file(wam_python_target:compile_step_wam_to_python(_,_), ThisFile),
		file_directory_name(ThisFile, ThisDir),
		directory_file_path(ThisDir, 'wam_python_runtime/WamRuntime.py', SrcPath),
		exists_file(SrcPath)
	->  copy_file(SrcPath, DestPath)
	;   % Generate from bindings if static file not found
		compile_wam_runtime_to_python([], RuntimeCode),
		write_file(DestPath, RuntimeCode)
	).

%% is_ffi_predicate(+Functor, +Arity, +Options)
%  True if this predicate is handled by a registered foreign predicate,
%  meaning we can skip WAM compilation and emit a direct execute_foreign call.
is_ffi_predicate(Functor, Arity, Options) :-
	option(foreign_predicates(FPs), Options, []),
	member(Functor/Arity, FPs).

%% compile_all_predicates(+Predicates, +Options, -Code)
%  Compile all predicates into Python code.
%  FFI-owned fact predicates are skipped (emit a direct foreign call stub).
compile_all_predicates([], _Options, "# No predicates compiled\n").
compile_all_predicates(Predicates, Options, Code) :-
	Predicates \= [],
	maplist(compile_one_predicate(Options), Predicates, PredCodes),
	% Collect function names for build_program() registry
	maplist(pred_func_name(Options), Predicates, FuncNames),
	build_program_func(FuncNames, BuildProgramCode),
	atomic_list_concat([
		'"""WAM-compiled predicates — generated by UnifyWeaver"""\n',
		'from wam_runtime import *\n',
		'# WAM register index constants (A1..A8 = 1..8, X1..X20 = 1..20, Y1..Y20 = 201..220)\n',
		'A1,A2,A3,A4,A5,A6,A7,A8 = 1,2,3,4,5,6,7,8\n',
		'X1,X2,X3,X4,X5,X6,X7,X8,X9,X10 = 1,2,3,4,5,6,7,8,9,10\n',
		'X11,X12,X13,X14,X15,X16,X17,X18,X19,X20 = 11,12,13,14,15,16,17,18,19,20\n',
		'Y1,Y2,Y3,Y4,Y5,Y6,Y7,Y8,Y9,Y10 = 201,202,203,204,205,206,207,208,209,210\n',
		'Y11,Y12,Y13,Y14,Y15,Y16,Y17,Y18,Y19,Y20 = 211,212,213,214,215,216,217,218,219,220\n\n'
		| PredCodes
	], '\n\n', PredSection),
	atomic_list_concat([PredSection, '\n\n', BuildProgramCode], Code).

%% pred_func_name(+Options, +PredSpec, -FuncName)
%  Get the Python function name for a predicate (for build_program registry).
pred_func_name(Options, _Module:Pred/Arity, FuncName) :- !,
	pred_registrar_func_name(Options, Pred/Arity, FuncName).
pred_func_name(Options, _Module:Pred/Arity-_, FuncName) :- !,
	pred_registrar_func_name(Options, Pred/Arity, FuncName).
pred_func_name(Options, Pred/Arity-_, FuncName) :- !,
	pred_registrar_func_name(Options, Pred/Arity, FuncName).
pred_func_name(Options, Pred/Arity, FuncName) :-
	pred_registrar_func_name(Options, Pred/Arity, FuncName).

pred_registrar_func_name(Options, Pred/Arity, RegistrarName) :-
	python_func_name(Pred/Arity, FuncName),
	(   option(emit_mode(lowered), Options)
	->  atom_concat(register_, FuncName, RegistrarName)
	;   RegistrarName = FuncName
	).

%% build_program_func(+FuncNames, -Code)
%  Emit a build_program(raw_program) function that calls all pred_* registrars.
build_program_func(FuncNames, Code) :-
	maplist([N, Line]>>format(atom(Line), '    ~w(raw_program)', [N]),
		FuncNames, Lines),
	atomic_list_concat(Lines, '\n', Body),
	format(string(Code),
'def build_program(raw_program=None):\n    if raw_program is None:\n        raw_program = {}\n~w\n    return raw_program\n',
	[Body]).

compile_one_predicate(Options, _Module:PredSpec, PredCode) :-
	!,
	compile_one_predicate(Options, PredSpec, PredCode).
compile_one_predicate(Options, Pred/Arity-WamCode, PredCode) :-
	(   is_ffi_predicate(Pred, Arity, Options)
	->  compile_ffi_stub_predicate(Pred, Arity, Options, PredCode)
	;   option(emit_mode(lowered), Options),
	    compile_lowered_wam_predicate_to_python(Pred/Arity, WamCode, Options, PredCode)
	->  true
	;   option(emit_mode(lowered), Options)
	->  compile_wam_predicate_to_python(Pred/Arity, WamCode, [registrar_prefix(register_)|Options], PredCode)
	;   compile_wam_predicate_to_python(Pred/Arity, WamCode, Options, PredCode)
	).
compile_one_predicate(Options, Pred/Arity, PredCode) :-
	(   is_ffi_predicate(Pred, Arity, Options)
	->  compile_ffi_stub_predicate(Pred, Arity, PredCode)
	;   (   compile_predicate_to_wam(Pred/Arity, [], WamCode)
		->  compile_one_predicate(Options, Pred/Arity-WamCode, PredCode)
		;   atom_string(Pred, PredStr),
			format(string(PredCode),
				'# Could not compile ~w/~w to WAM\n', [PredStr, Arity])
		)
	).

%% compile_ffi_stub_predicate(+Pred, +Arity, -Code)
%  Emit a registration function that inserts a call_foreign stub into raw_program.
compile_ffi_stub_predicate(Pred, Arity, PredCode) :-
	compile_ffi_stub_predicate(Pred, Arity, [], PredCode).

compile_ffi_stub_predicate(Pred, Arity, Options, PredCode) :-
	atom_string(Pred, PredStr),
	python_func_name(Pred/Arity, FuncName),
	(   option(emit_mode(lowered), Options)
	->  atom_concat(register_, FuncName, RegistrarName)
	;   RegistrarName = FuncName
	),
	format(atom(LabelKey), '~w/~w', [PredStr, Arity]),
	format(string(PredCode),
'def ~w(raw_program):
    """Register FFI stub for ~w/~w — dispatches via call_foreign."""
    raw_program["~w"] = (
        ("call_foreign", "~w", ~w),
        ("proceed",),
    )
', [RegistrarName, PredStr, Arity, LabelKey, PredStr, Arity]).

%% generate_main_py(+Predicates, +ModuleName, -Code)
%  Generate a main.py entry point.
generate_main_py(_Predicates, ModuleName, Code) :-
	format(string(Code),
'"""~w — WAM-to-Python entry point (generated by UnifyWeaver)"""
import sys
from wam_runtime import *
from predicates import *

def _init_query_args(query, state):
    try:
        arity = int(query.rsplit("/", 1)[1])
    except (IndexError, ValueError):
        return
    for i in range(1, arity + 1):
        if get_reg(state, i) is None:
            set_reg(state, i, state.fresh_var())

def main():
    state = WamState()
    # Build a raw program dict from predicates, then flatten + pre-resolve
    raw_program = build_program()
    code, labels = load_program(raw_program)
    # Run a user-specified query
    if len(sys.argv) > 1:
        query = sys.argv[1]
        if query in labels:
            _init_query_args(query, state)
            if run_wam(code, labels, query, state):
                results = []
                for i in range(1, 11):
                    val = get_reg(state, i)
                    if val is not None:
                        results.append((i, val))
                for i, r in results:
                    print(f"A{i} = {repr(r)}")
            else:
                print("false.")
        else:
            print(f"Unknown predicate: {query}")
            sys.exit(1)
    else:
        print("Usage: python main.py <predicate_name>")
        sys.exit(1)

if __name__ == "__main__":
    main()
', [ModuleName]).

%% generate_parallel_main_py(+Predicates, +ModuleName, -Code)
%  Generate a main.py entry point that uses run_parallel for concurrent seeds.
generate_parallel_main_py(_Predicates, ModuleName, Code) :-
	format(string(Code),
'"""~w — WAM-to-Python parallel entry point (generated by UnifyWeaver)"""
import sys
from wam_runtime import *
from predicates import *

def main():
    # Build a raw program dict from predicates, then flatten + pre-resolve
    raw_program = build_program()
    code, labels = load_program(raw_program)

    # Define seeds — each seed is a list of initial register values
    seeds = [
        [Atom("query")],
    ]

    # Run seeds in parallel (ProcessPoolExecutor, falls back to threads)
    max_workers = 0  # 0 = use cpu_count
    results = run_parallel(code, labels, sys.argv[1] if len(sys.argv) > 1 else "query", seeds, max_workers)
    for i, res in enumerate(results):
        if res is not None:
            print(f"Seed {i}: {res}")
        else:
            print(f"Seed {i}: false.")

if __name__ == "__main__":
    main()
', [ModuleName]).

%% generate_benchmark_main_py(+Predicates, +ModuleName, -Code)
%  Generate a main.py benchmark driver that:
%    1. Loads TSV facts from a data directory (argv[1])
%    2. Registers category_parent/2 as a foreign predicate backed by the TSV dict
%    3. Runs the effective_distance query with Stopwatch timing
%    4. Outputs query_ms / total_ms compatible with the cross-target benchmark table
generate_benchmark_main_py(_Predicates, ModuleName, Code) :-
	format(string(Code),
'"""~w — WAM benchmark driver (generated by UnifyWeaver)"""
import sys
import os
import csv
import time
from wam_runtime import *
from predicates import *


def load_tsv_pairs(path):
    """Load a two-column TSV (with header row) into dict: key -> [val, ...]"""
    result = {}
    if not os.path.exists(path):
        return result
    with open(path, newline=\'\', encoding=\'utf-8\') as f:
        reader = csv.reader(f, delimiter=\'\t\')
        next(reader, None)  # skip header row
        for row in reader:
            if len(row) >= 2:
                k, v = row[0], row[1]
                result.setdefault(k, []).append(v)
    return result


def load_single_column(path):
    """Load a one-column TSV (with header row) into a list."""
    result = []
    if not os.path.exists(path):
        return result
    with open(path, newline=\'\', encoding=\'utf-8\') as f:
        reader = csv.reader(f, delimiter=\'\t\')
        next(reader, None)  # skip header row
        for row in reader:
            if row:
                result.append(row[0])
    return result


def register_category_parent(category_parents):
    """Register category_parent/2 as a foreign predicate backed by TSV dict."""
    def category_parent_foreign(args, state, resume_ip=-1):
        child = deref(args[0], state)
        if not isinstance(child, Atom):
            return False
        parents = category_parents.get(child.name, [])
        if not parents:
            return False
        # Push a choice point for each extra parent using closure-based continuations.
        # Each closure unifies A2 with its parent and returns resume_ip.
        for p in reversed(parents[1:]):
            parent_atom = make_atom(p)
            def make_cont(pa, rip):
                def cont(s):
                    v = deref(get_reg(s, 2), s)
                    if unify(v, pa, s):
                        return rip
                    return -1
                return cont
            push_choice_point(state, 2, make_cont(parent_atom, resume_ip))
        # Unify A2 with the first parent
        return unify(get_reg(state, 2), make_atom(parents[0]), state)
    register_foreign(\'category_parent\', 2, category_parent_foreign)


def run_benchmark(data_dir, n_reps):
    t0 = time.perf_counter()

    # Load TSV facts
    category_parents = load_tsv_pairs(os.path.join(data_dir, \'category_parent.tsv\'))
    article_categories = load_tsv_pairs(os.path.join(data_dir, \'article_category.tsv\'))
    roots = load_single_column(os.path.join(data_dir, \'root_categories.tsv\'))
    root = roots[0] if roots else \'Physics\'

    load_ms = int((time.perf_counter() - t0) * 1000)
    print(f\'load_ms={load_ms}\', file=sys.stderr)

    # Register foreign predicates
    register_category_parent(category_parents)

    # Pre-flatten category_parent pairs once (used for indexed-table init below).
    category_parent_pairs = [
        (child, p) for child, parents in category_parents.items() for p in parents
    ]

    # Build WAM program via predicates.build_program()
    raw_program = build_program()
    code, labels = load_program(raw_program)

    # Collect seeds: all article -> category mappings
    seeds = []
    for article, cats in article_categories.items():
        for cat in cats:
            seeds.append((article, cat, root))
    if not seeds:
        seeds = [(\'test\', root, root)]

    setup_ms = int((time.perf_counter() - t0) * 1000)
    query_pred = \'category_ancestor$effective_distance_sum_selected/3\'
    print(f\'setup_ms={setup_ms} queryPred={query_pred}\', file=sys.stderr)

    best_query_ms = None
    n_solutions = 0
    for rep in range(1, n_reps + 1):
        t_q = time.perf_counter()
        rep_solutions = 0
        for (article, cat, rt) in seeds:
            state = WamState()
            # Rust-parity: populate indexed fact table for O(1) category_parent/2
            # lookups consumed by call_indexed_atom_fact2 / base_category_ancestor*.
            register_indexed_atom_fact2_pairs(state, \'category_parent/2\', category_parent_pairs)
            set_reg(state, 1, make_atom(cat))
            set_reg(state, 2, make_atom(rt))
            set_reg(state, 3, state.fresh_var())
            try:
                ok = run_wam(code, labels, query_pred, state)
                if ok:
                    rep_solutions += 1
            except Exception:
                pass
        q_ms = int((time.perf_counter() - t_q) * 1000)
        print(f\'rep={rep} query_ms={q_ms} seeds={len(seeds)} solutions={rep_solutions}\', file=sys.stderr)
        if best_query_ms is None or q_ms < best_query_ms:
            best_query_ms = q_ms
        n_solutions = rep_solutions

    total_ms = int((time.perf_counter() - t0) * 1000)
    print(f\'query_ms={best_query_ms} seeds={len(seeds)} solutions={n_solutions} reps={n_reps}\')
    print(f\'total_ms={total_ms}\')


def main():
    if len(sys.argv) < 2:
        print(\'Usage: python main.py <data_dir> [n_reps]\', file=sys.stderr)
        sys.exit(1)
    data_dir = sys.argv[1]
    n_reps = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    run_benchmark(data_dir, n_reps)


if __name__ == \'__main__\':
    main()
', [ModuleName]).

%% write_file(+Path, +Content)
%  Write content to a file.
write_file(Path, Content) :-
	open(Path, write, Stream),
	write(Stream, Content),
	close(Stream).
