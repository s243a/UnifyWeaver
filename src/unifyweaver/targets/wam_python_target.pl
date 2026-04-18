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

% ============================================================================
% PHASE 5: Predicate Compilation
% ============================================================================

%% compile_wam_predicate_to_python(+Pred/Arity, +WamCode, +Options, -PythonCode)
%  Converts WAM instruction output for a predicate to Python code.
compile_wam_predicate_to_python(Pred/Arity, WamCode, _Options, PythonCode) :-
	atom_string(Pred, PredStr),
	build_python_wam_arg_list(Arity, ArgList),
	build_python_wam_arg_setup(Arity, ArgSetup),
	wam_code_to_python_instructions(WamCode, Pred/Arity, InstrLiterals, LabelLiterals),
	format(string(PythonCode),
'def wam_~w(~w):
    """WAM-compiled predicate: ~w/~w"""
    state = args_state
~w
    # Labels
~w
    # Instructions
    state.code = (
~w
    )
    return state.run()
', [PredStr, ArgList, PredStr, Arity, ArgSetup, LabelLiterals, InstrLiterals]).

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
			format(string(LabelInsert),
				'    state.labels["~w"] = ~w', [LabelName, PC]),
			Labels = [LabelInsert|RestLabels],
			wam_lines_to_python(Rest, PC, Instrs, RestLabels)
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

	% Generate main.py — parallel variant when parallel(true)
	(   option(parallel(true), Options)
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
	atomic_list_concat([
		'"""WAM-compiled predicates — generated by UnifyWeaver"""\n',
		'from wam_runtime import *\n\n'
		| PredCodes
	], '\n\n', Code).

compile_one_predicate(Options, _Module:PredSpec, PredCode) :-
	!,
	compile_one_predicate(Options, PredSpec, PredCode).
compile_one_predicate(Options, Pred/Arity-WamCode, PredCode) :-
	(   is_ffi_predicate(Pred, Arity, Options)
	->  % Skip WAM compilation — emit direct foreign call stub
		atom_string(Pred, PredStr),
		python_func_name(Pred/Arity, FuncName),
		format(string(PredCode),
'def ~w(state):
    """FFI-owned predicate: ~w/~w — dispatched to foreign."""
    _args = [deref(state.regs[i+1], state) for i in range(~w)]
    return execute_foreign("~w", ~w, _args, state)
', [FuncName, PredStr, Arity, Arity, PredStr, Arity])
	;   compile_wam_predicate_to_python(Pred/Arity, WamCode, Options, PredCode)
	).
compile_one_predicate(Options, Pred/Arity, PredCode) :-
	(   is_ffi_predicate(Pred, Arity, Options)
	->  atom_string(Pred, PredStr),
		python_func_name(Pred/Arity, FuncName),
		format(string(PredCode),
'def ~w(state):
    """FFI-owned predicate: ~w/~w — dispatched to foreign."""
    _args = [deref(state.regs[i+1], state) for i in range(~w)]
    return execute_foreign("~w", ~w, _args, state)
', [FuncName, PredStr, Arity, Arity, PredStr, Arity])
	;   (   compile_predicate_to_wam(Pred/Arity, [], WamCode)
		->  compile_wam_predicate_to_python(Pred/Arity, WamCode, Options, PredCode)
		;   atom_string(Pred, PredStr),
			format(string(PredCode),
				'# Could not compile ~w/~w to WAM\n', [PredStr, Arity])
		)
	).

%% generate_main_py(+Predicates, +ModuleName, -Code)
%  Generate a main.py entry point.
generate_main_py(_Predicates, ModuleName, Code) :-
	format(string(Code),
'"""~w — WAM-to-Python entry point (generated by UnifyWeaver)"""
import sys
from wam_runtime import *
from predicates import *

def main():
    state = WamState()
    # Build a raw program dict from predicates, then flatten + pre-resolve
    raw_program = {}  # populated by predicate modules
    code, labels = load_program(raw_program)
    # Run a user-specified query
    if len(sys.argv) > 1:
        query = sys.argv[1]
        if query in labels:
            if run_wam(code, labels, query, state):
                results = []
                for i in range(1, 11):
                    val = get_reg(state, i)
                    if val is not None:
                        results.append((i, val))
                for i, r in results:
                    print(f"A{i} = {_format_value(r)}")
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
    raw_program = {}  # populated by predicate modules
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

%% write_file(+Path, +Content)
%  Write content to a file.
write_file(Path, Content) :-
	open(Path, write, Stream),
	write(Stream, Content),
	close(Stream).
