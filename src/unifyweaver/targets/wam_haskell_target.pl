:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% wam_haskell_target.pl - WAM-to-Haskell Transpilation Target
%
% Compiles WAM instructions to Haskell code using persistent data structures.
% Unlike the Rust WAM target (which interprets instructions at runtime),
% this target natively lowers each WAM instruction to Haskell expressions.
%
% Key design: Data.Map for registers and bindings gives O(1) snapshots
% for choice points (structural sharing), eliminating the clone overhead
% that made the Rust WAM 343x slower than SWI-Prolog.
%
% Architecture:
%   - WamState record with Data.Map fields
%   - Each compiled predicate becomes a Haskell function
%   - TryMeElse/RetryMeElse/TrustMe become choice point list operations
%   - Backtracking = swap to saved Data.Map reference (O(1))
%   - Facts loaded as Data.Map lookup tables (first-argument indexing)
%
% Pipeline:
%   Prolog source → haskell_target.pl (native lowering, preferred)
%                 → wam_target.pl (WAM compilation, fallback)
%                 → wam_haskell_target.pl (THIS FILE: WAM → Haskell)
%
% See: docs/design/WAM_RUST_STATE_MANAGEMENT_RETROSPECTIVE.md

:- module(wam_haskell_target, [
    compile_wam_predicate_to_haskell/4,  % +Pred/Arity, +WamCode, +Options, -HaskellCode
    compile_wam_runtime_to_haskell/3,    % +Options, +DetectedKernels, -HaskellCode
    write_wam_haskell_project/3,         % +Predicates, +Options, +ProjectDir
    wam_haskell_resolve_emit_mode/2,     % +Options, -Mode
    wam_haskell_partition_predicates/5   % +Mode, +Predicates, +DetectedKernels, -InterpretedList, -LoweredList
]).

:- use_module(library(lists)).
:- use_module(library(pairs)).
:- use_module(library(option)).
:- use_module(library(filesex), [make_directory_path/1, directory_file_path/3]).
:- use_module('../targets/wam_target', [compile_predicate_to_wam/3]).
:- use_module('../core/recursive_kernel_detection',
             [detect_recursive_kernel/4, kernel_metadata/4, kernel_config/2,
              kernel_register_layout/2, kernel_native_call/2, kernel_template_file/2]).
:- use_module('../core/template_system', [render_template/3]).
:- use_module('../core/purity_certificate',
             [analyze_predicate_purity/2]).

% Phase 3: the real lowerability check and emission helpers live in the
% wam_haskell_lowered_emitter module. We reexport so existing callers can
% still see wam_haskell_lowerable/3 through this module.
:- reexport('wam_haskell_lowered_emitter',
            [wam_haskell_lowerable/3, lower_predicate_to_haskell/4]).

%% ============================================================================
%% emit_mode selector (Phase 1 of WAM-lowered Haskell path)
%% ============================================================================
%%
%% See docs/design/WAM_HASKELL_LOWERED_{PHILOSOPHY,SPECIFICATION,IMPLEMENTATION_PLAN}.md
%%
%% Three modes are recognised:
%%   - interpreter          : every predicate compiles via the existing
%%                            instruction-array interpreter path (default).
%%   - functions            : every predicate attempts lowering to a
%%                            standalone Haskell function. Any predicate
%%                            that fails wam_haskell_lowerable/3 falls back
%%                            to the interpreter automatically.
%%   - mixed(HotPreds)      : the predicates named in HotPreds attempt
%%                            lowering; the rest go to the interpreter.
%%
%% Selector hierarchy, checked in order:
%%   1. emit_mode(Mode) option on the write_wam_haskell_project/3 call
%%   2. user:wam_haskell_emit_mode(Mode) dynamic fact (if defined)
%%   3. default: interpreter
%%
%% Phase 1 ships the plumbing only. wam_haskell_lowerable/3 is a stub that
%% always fails, so all three modes route every predicate to the interpreter.
%% Output is byte-identical to the pre-Phase-1 state. Real lowering starts
%% in Phase 3 of the implementation plan.

:- multifile user:wam_haskell_emit_mode/1.

%% wam_haskell_resolve_emit_mode(+Options, -Mode)
%  Resolve the emit_mode selector using the hierarchy above. Throws a
%  domain_error on an unknown value.
wam_haskell_resolve_emit_mode(Options, Mode) :-
    (   option(emit_mode(M0), Options)
    ->  wam_haskell_validate_emit_mode(M0, Mode)
    ;   catch(user:wam_haskell_emit_mode(M1), _, fail)
    ->  wam_haskell_validate_emit_mode(M1, Mode)
    ;   Mode = interpreter
    ).

wam_haskell_validate_emit_mode(interpreter, interpreter) :- !.
wam_haskell_validate_emit_mode(functions, functions)     :- !.
wam_haskell_validate_emit_mode(mixed(L), mixed(L)) :-
    is_list(L), !.
wam_haskell_validate_emit_mode(Other, _) :-
    throw(error(domain_error(wam_haskell_emit_mode, Other),
                wam_haskell_resolve_emit_mode/2)).

%% wam_haskell_lowerable/3 lives in wam_haskell_lowered_emitter.pl and is
%% re-exported above. Phase 1 shipped a stub that always failed; Phase 3
%% replaces it with a real whitelist check against the WAM instruction
%% sequence.

%% wam_haskell_partition_predicates(+Mode, +Predicates, +DetectedKernels, -InterpretedList, -LoweredList)
%  Partition Predicates into the two sublists based on Mode and the
%  lowerability check. Detected kernels are always excluded from lowering
%  (they use FFI via CallForeign — lowering them would be dead code).
wam_haskell_partition_predicates(interpreter, Predicates, _, Predicates, []) :- !.
wam_haskell_partition_predicates(functions, Predicates, DK, Interpreted, Lowered) :- !,
    pairs_keys(DK, KernelKeys),
    wam_haskell_partition_try_lower(Predicates, KernelKeys, Interpreted, Lowered).
wam_haskell_partition_predicates(mixed(HotPreds), Predicates, DK, Interpreted, Lowered) :- !,
    pairs_keys(DK, KernelKeys),
    wam_haskell_partition_mixed(Predicates, HotPreds, KernelKeys, Interpreted, Lowered).

% functions mode: every non-kernel predicate attempts lowering.
wam_haskell_partition_try_lower([], _, [], []).
wam_haskell_partition_try_lower([P|Rest], KK, Interpreted, Lowered) :-
    pred_key(P, Key),
    (   member(Key, KK)
    ->  % Detected kernel — skip lowering, use FFI
        Interpreted = [P|IR],
        wam_haskell_partition_try_lower(Rest, KK, IR, Lowered)
    ;   wam_haskell_predicate_wamcode(P, WamCode),
        (   wam_haskell_lowerable(P, WamCode, _Reason)
        ->  Lowered = [P|LR],
            wam_haskell_partition_try_lower(Rest, KK, Interpreted, LR)
        ;   Interpreted = [P|IR],
            wam_haskell_partition_try_lower(Rest, KK, IR, Lowered)
        )
    ).

% mixed(HotPreds) mode: predicates in HotPreds attempt lowering; rest are interpreted.
% Detected kernels are always excluded from lowering.
wam_haskell_partition_mixed([], _, _, [], []).
wam_haskell_partition_mixed([P|Rest], HotPreds, KK, Interpreted, Lowered) :-
    pred_key(P, Key),
    (   member(Key, KK)
    ->  % Detected kernel — skip lowering
        Interpreted = [P|IR],
        wam_haskell_partition_mixed(Rest, HotPreds, KK, IR, Lowered)
    ;   wam_haskell_indicator_in_list(P, HotPreds)
    ->  wam_haskell_predicate_wamcode(P, WamCode),
        (   wam_haskell_lowerable(P, WamCode, _Reason)
        ->  Lowered = [P|LR],
            wam_haskell_partition_mixed(Rest, HotPreds, KK, Interpreted, LR)
        ;   Interpreted = [P|IR],
            wam_haskell_partition_mixed(Rest, HotPreds, KK, IR, Lowered)
        )
    ;   Interpreted = [P|IR],
        wam_haskell_partition_mixed(Rest, HotPreds, KK, IR, Lowered)
    ).

pred_key(P, Key) :-
    (P = _Mod:Pred/Arity -> true ; P = Pred/Arity),
    format(atom(Key), '~w/~w', [Pred, Arity]).

% Handle both Module:Pred/Arity and Pred/Arity comparisons against HotPreds.
wam_haskell_indicator_in_list(P, HotPreds) :-
    member(P, HotPreds), !.
wam_haskell_indicator_in_list(_Mod:Pred/Arity, HotPreds) :-
    member(Pred/Arity, HotPreds), !.

% Compile WAM code on demand for the lowerability check and emission.
wam_haskell_predicate_wamcode(PredIndicator, WamCode) :-
    (   PredIndicator = _Module:Pred/Arity -> true
    ;   PredIndicator = Pred/Arity
    ),
    wam_target:compile_predicate_to_wam(Pred/Arity, [], WamCode).

%% lower_all(+LoweredList, +BasePCMap, +DetectedKernels, -LoweredEntries)
%  Run the Phase 3+ emitter over each predicate in LoweredList, using
%  BasePCMap (a list of PredIndicator-StartPC pairs computed from the
%  merged instruction array) to offset local PCs to global PCs.
%  DetectedKernels (Key-Kernel pairs) is passed through so the emitter
%  can emit callForeign for known foreign predicates.
lower_all([], _, _, []).
lower_all([P|Rest], BasePCMap, DetectedKernels, [Entry|RestEntries]) :-
    wam_haskell_predicate_wamcode(P, WamCode),
    predicate_base_pc(P, BasePCMap, BasePC),
    pairs_keys(DetectedKernels, ForeignKeys),
    lower_predicate_to_haskell(P, WamCode,
        [base_pc(BasePC), foreign_preds(ForeignKeys)], Entry),
    lower_all(Rest, BasePCMap, DetectedKernels, RestEntries).

predicate_base_pc(P, Map, PC) :-
    (   P = _Mod:Pred/Arity -> true ; P = Pred/Arity ),
    format(atom(Key), '~w/~w', [Pred, Arity]),
    (   member(Key-PC, Map) -> true ; PC = 1 ).

%% generate_kernel_haskell(+DetectedKernels, -KernelFunctionsCode, -ExecuteForeignCode)
%  Render Mustache templates for detected kernels into Haskell source.
%  DetectedKernels is a list of Key-Kernel pairs from detect_kernels/2.
%  KernelFunctionsCode: native kernel function bodies (one per kernel).
%  ExecuteForeignCode: the executeForeign dispatch function with entries
%  for each detected kernel.
generate_kernel_haskell([], KF, EF) :- !,
    KF = "-- No kernels detected; no native functions generated.",
    EF = "executeForeign :: WamContext -> String -> WamState -> Maybe WamState\nexecuteForeign _ _ _ = Nothing".
generate_kernel_haskell(DetectedKernels, KernelFunctionsCode, ExecuteForeignCode) :-
    % For each detected kernel, find its template and render
    maplist(render_kernel_function, DetectedKernels, KernelParts),
    atomic_list_concat(KernelParts, '\n\n', KernelFunctionsCode),
    % Generate executeForeign with entries for each kernel
    generate_execute_foreign(DetectedKernels, ExecuteForeignCode).

render_kernel_function(Key-Kernel, Code) :-
    Kernel = recursive_kernel(Kind, _, ConfigOps),
    (   kernel_template_file(Kind, TemplateFile)
    ->  read_kernel_template(TemplateFile, Template),
        % Build template vars from config ops
        config_ops_to_template_vars(ConfigOps, TemplateVars),
        render_template(Template, TemplateVars, Code0),
        atom_string(Code0, Code)
    ;   format(atom(Code), '-- Kernel ~w: no template available', [Key])
    ).

%% config_ops_to_template_vars(+ConfigOps, -TemplateVars)
%  Convert kernel config ops to Mustache template key=value pairs.
config_ops_to_template_vars([], []).
config_ops_to_template_vars([Op|Rest], [Key=Value|RestVars]) :-
    Op =.. [Key, RawValue],
    (   RawValue = Pred/_ -> Value = Pred  % edge_pred(foo/2) -> foo
    ;   Value = RawValue
    ),
    config_ops_to_template_vars(Rest, RestVars).

read_kernel_template(FileName, Template) :-
    atom_concat('templates/targets/haskell_wam/', FileName, RelPath),
    (   source_file(wam_haskell_target, SrcFile)
    ->  file_directory_name(SrcFile, SrcDir),
        file_directory_name(SrcDir, TargetsDir),
        file_directory_name(TargetsDir, UnifyWeaverDir),
        file_directory_name(UnifyWeaverDir, ProjectDir),
        atom_concat(ProjectDir, '/', P1),
        atom_concat(P1, RelPath, AbsPath)
    ;   AbsPath = RelPath
    ),
    (   exists_file(AbsPath)
    ->  read_file_to_string(AbsPath, Template, [])
    ;   format(atom(Template), '-- Template not found: ~w', [AbsPath])
    ).

generate_execute_foreign(DetectedKernels, Code) :-
    with_output_to(string(Code), (
        format("executeForeign :: WamContext -> String -> WamState -> Maybe WamState~n"),
        forall(member(KV, DetectedKernels), emit_execute_foreign_entry(KV)),
        format("executeForeign _ _ _ = Nothing~n")
    )).

%% emit_execute_foreign_entry(+Key-Kernel)
%  Generate a single executeForeign clause from kernel metadata.
%  Reads kernel_register_layout and kernel_native_call to produce the
%  register reading, native call, result binding, and choice point code.
emit_execute_foreign_entry(Key-Kernel) :-
    Kernel = recursive_kernel(Kind, _, ConfigOps),
    (   kernel_register_layout(Kind, RegSpecs),
        kernel_native_call(Kind, CallSpec)
    ->  % Resolve config_facts_from references using the kernel's config ops
        resolve_call_spec(CallSpec, ConfigOps, ResolvedCallSpec),
        emit_ef_clause(Key, RegSpecs, ResolvedCallSpec)
    ;   format('-- executeForeign: no metadata for ~w~n', [Key])
    ).

%% resolve_call_spec(+CallSpec, +ConfigOps, -ResolvedCallSpec)
%  Replace config_facts_from(ConfigKey) with config_facts(ActualName)
%  by looking up the ConfigKey in the kernel's config ops.
resolve_call_spec(call(Func, Args), ConfigOps, call(Func, ResolvedArgs)) :-
    maplist(resolve_arg_spec(ConfigOps), Args, ResolvedArgs).

resolve_arg_spec(ConfigOps, config_facts_from(ConfigKey), config_facts(FactName)) :- !,
    % Look up ConfigKey in config ops: e.g., edge_pred(foo/2) → foo
    Op =.. [ConfigKey, RawValue],
    member(Op, ConfigOps),
    (   RawValue = Pred/_ -> FactName = Pred ; FactName = RawValue ).
resolve_arg_spec(ConfigOps, config_weighted_facts_from(ConfigKey), config_weighted_facts(FactName)) :- !,
    % Weighted variant: resolves to wcFfiWeightedFacts lookup at codegen
    Op =.. [ConfigKey, RawValue],
    member(Op, ConfigOps),
    (   RawValue = Pred/_ -> FactName = Pred ; FactName = RawValue ).
resolve_arg_spec(_, Arg, Arg).

emit_ef_clause(Key, RegSpecs, call(FuncName, ArgSpecs)) :-
    % Function header
    format('executeForeign !ctx "~w" s =~n', [Key]),
    % Emit let-bindings for input registers (deref from WAM regs)
    include(is_input_reg, RegSpecs, InputRegs),
    format('  let '),
    emit_input_let_bindings(InputRegs, first),
    % Emit config bindings from ArgSpecs
    emit_config_let_bindings(ArgSpecs),
    % Find output register(s). Single-output kernels (category_ancestor,
    % transitive_closure2) keep the existing fast path; multi-output
    % kernels (transitive_distance3, weighted_shortest_path3) use a
    % tuple-based bindResult that binds multiple registers per solution.
    include(is_output_reg, RegSpecs, OutputRegs),
    % Emit case expression over input regs, then native call + binding inside
    emit_case_and_call(InputRegs, OutputRegs, FuncName, ArgSpecs),
    format('~n').

is_input_reg(input(_, _)).
is_output_reg(output(_, _)).

%% emit_input_let_bindings(+InputRegs, +Position)
%  Emit let-bindings that read and deref each input register.
emit_input_let_bindings([], _).
emit_input_let_bindings([input(RegN, Type)|Rest], Pos) :-
    reg_var_name(RegN, VarName),
    reg_default_value(Type, Default),
    (   Pos = first -> true ; format('      ') ),
    format('~w = derefVar (wsBindings s) $ fromMaybe (~w) (IM.lookup ~w (wsRegs s))~n',
           [VarName, Default, RegN]),
    emit_input_let_bindings(Rest, rest).

reg_var_name(N, Name) :-
    format(atom(Name), 'r~w', [N]).

reg_default_value(atom, 'Atom ""').
reg_default_value(integer, 'Integer 0').
reg_default_value(vlist_atoms, 'VList []').

%% emit_config_let_bindings(+ArgSpecs)
%  Emit let-bindings for config_facts and config_int arguments.
emit_config_let_bindings([]).
emit_config_let_bindings([config_facts(FactKey)|Rest]) :-
    % FFI facts are stored interned in wcFfiFacts (IntMap [Int])
    format('      ~w_facts = fromMaybe IM.empty $ Map.lookup "~w" (wcFfiFacts ctx)~n',
           [FactKey, FactKey]),
    emit_config_let_bindings(Rest).
emit_config_let_bindings([config_weighted_facts(FactKey)|Rest]) :-
    % Weighted FFI facts: IntMap [(Int, Double)] (target, weight pairs)
    format('      ~w_facts = fromMaybe IM.empty $ Map.lookup "~w" (wcFfiWeightedFacts ctx)~n',
           [FactKey, FactKey]),
    emit_config_let_bindings(Rest).
emit_config_let_bindings([config_int(ConfigKey, Default)|Rest]) :-
    format('      ~w_cfg = fromMaybe ~w $ Map.lookup "~w" (wcForeignConfig ctx)~n',
           [ConfigKey, Default, ConfigKey]),
    emit_config_let_bindings(Rest).
emit_config_let_bindings([_|Rest]) :-
    emit_config_let_bindings(Rest).

%% emit_call_args(+ArgSpecs, +InputRegs)
%  Emit the argument list for the native kernel function call.
emit_call_args([], _).
emit_call_args([Spec|Rest], InputRegs) :-
    format(' '),
    emit_one_call_arg(Spec, InputRegs),
    emit_call_args(Rest, InputRegs).

emit_one_call_arg(config_facts(FactKey), _) :-
    format('~w_facts', [FactKey]).
emit_one_call_arg(config_weighted_facts(FactKey), _) :-
    format('~w_facts', [FactKey]).
emit_one_call_arg(config_int(ConfigKey, _), _) :-
    format('~w_cfg', [ConfigKey]).
emit_one_call_arg(reg(RegN), InputRegs) :-
    member(input(RegN, Type), InputRegs),
    reg_var_name(RegN, VarName),
    emit_reg_extraction(VarName, Type).
emit_one_call_arg(derived(length, RegN), InputRegs) :-
    member(input(RegN, _Type), InputRegs),
    reg_var_name(RegN, VarName),
    format('(length [v | Atom v <- ~wL])', [VarName]).

%% emit_reg_extraction(+VarName, +Type)
%  Emit the extraction expression for a kernel call argument. Atoms and
%  atom lists are interned via wcAtomIntern so the kernel operates on
%  Int IDs instead of Strings (eliminates hashing in the hot loop).
emit_reg_extraction(VarName, atom) :-
    format('(fromMaybe (-1) (Map.lookup ~wS (wcAtomIntern ctx)))', [VarName]).
emit_reg_extraction(VarName, vlist_atoms) :-
    format('[fromMaybe (-1) (Map.lookup v (wcAtomIntern ctx)) | Atom v <- ~wL]', [VarName]).
emit_reg_extraction(VarName, integer) :-
    format('~wI', [VarName]).

%% emit_case_and_call(+InputRegs, +OutputRegs, +FuncName, +ArgSpecs)
%  Emit the case expression that pattern-matches inputs, then the native
%  call and stream binding INSIDE the case branch (so extracted string
%  variables are in scope for the call).
emit_case_and_call(InputRegs, OutputRegs, FuncName, ArgSpecs) :-
    length(InputRegs, NInputs),
    (   NInputs =:= 1
    ->  InputRegs = [input(RegN1, _)],
        reg_var_name(RegN1, ScrutName),
        format('  in case ~w of~n', [ScrutName]),
        emit_single_case_branch(InputRegs, OutputRegs, FuncName, ArgSpecs)
    ;   format('  in case ('),
        emit_scrutinee_tuple(InputRegs, first),
        format(') of~n'),
        format('    ('),
        emit_pattern_tuple(InputRegs, first),
        format(') ->~n'),
        emit_native_call_and_binding(OutputRegs, FuncName, ArgSpecs, InputRegs, "      ")
    ),
    format('    _ -> Nothing~n').

emit_single_case_branch([input(RegN, Type)|_], OutputRegs, FuncName, ArgSpecs) :-
    reg_var_name(RegN, VarName),
    type_pattern(Type, VarName, Pattern),
    format('    ~w ->~n', [Pattern]),
    emit_native_call_and_binding(OutputRegs, FuncName, ArgSpecs, [input(RegN, Type)], "      ").

emit_scrutinee_tuple([], _).
emit_scrutinee_tuple([input(RegN, _)|Rest], Pos) :-
    reg_var_name(RegN, VarName),
    (   Pos = first -> true ; format(', ') ),
    format('~w', [VarName]),
    emit_scrutinee_tuple(Rest, rest).

emit_pattern_tuple([], _).
emit_pattern_tuple([input(RegN, Type)|Rest], Pos) :-
    reg_var_name(RegN, VarName),
    type_pattern(Type, VarName, Pattern),
    (   Pos = first -> true ; format(', ') ),
    format('~w', [Pattern]),
    emit_pattern_tuple(Rest, rest).

type_pattern(atom, VarName, Pattern) :-
    format(atom(Pattern), 'Atom ~wS', [VarName]).
type_pattern(integer, VarName, Pattern) :-
    format(atom(Pattern), 'Integer ~wI', [VarName]).
type_pattern(vlist_atoms, VarName, Pattern) :-
    format(atom(Pattern), 'VList ~wL', [VarName]).

%% emit_native_call_and_binding(+OutputRegs, +FuncName, +ArgSpecs, +InputRegs, +Indent)
%  Emit the native function call followed by stream binding, all inside
%  a case branch at the given indentation level.
%
%  All kernels — single- AND multi-output — go through the FFIStreamRetry
%  path. The single-output fast path with HopsRetry was retained for
%  back-compat with category_ancestor/transitive_closure2, but had a
%  latent bug: its resume logic hardcodes `Integer (fromIntegral h)`,
%  which is wrong for atom/float outputs. The multi-output path is
%  type-correct by construction — it carries pre-wrapped Values in the
%  choice point — so it's strictly better for any non-integer output.
%
%  For single-output kernels (length OutputRegs = 1), the emitted code
%  treats the result as a 1-tuple: `bindResult rv_1` (not a tuple
%  pattern; `emit_tuple_pattern(1)` produces just `rv_1`).
emit_native_call_and_binding(OutputRegs, FuncName, ArgSpecs, InputRegs, Indent) :-
    format('~wlet results = ~w', [Indent, FuncName]),
    emit_call_args(ArgSpecs, InputRegs),
    format('~n'),
    emit_stream_binding_multi(OutputRegs, Indent).

%% emit_stream_binding_single(+OutRegN, +OutType, +Indent)
%  Single-output stream binding. Unchanged from the original for
%  back-compat with category_ancestor and transitive_closure2.
emit_stream_binding_single(OutRegN, OutType, Indent) :-
    result_wrap_expr(OutType, WrapExpr),
    format('~w    retPC = wsCP s~n', [Indent]),
    format('~w    outReg = derefVar (wsBindings s) $ fromMaybe (Unbound (-1)) (IM.lookup ~w (wsRegs s))~n', [Indent, OutRegN]),
    format('~w    bindResult rv =~n', [Indent]),
    format('~w      case outReg of~n', [Indent]),
    format('~w        Unbound vid ->~n', [Indent]),
    format('~w          s { wsPC = retPC~n', [Indent]),
    format('~w            , wsRegs = IM.insert ~w (~w) (wsRegs s)~n', [Indent, OutRegN, WrapExpr]),
    format('~w            , wsBindings = IM.insert vid (~w) (wsBindings s)~n', [Indent, WrapExpr]),
    format('~w            , wsTrail = TrailEntry vid (IM.lookup vid (wsBindings s)) : wsTrail s~n', [Indent]),
    format('~w            , wsTrailLen = wsTrailLen s + 1 }~n', [Indent]),
    format('~w        _ -> s { wsPC = retPC, wsRegs = IM.insert ~w (~w) (wsRegs s) }~n', [Indent, OutRegN, WrapExpr]),
    format('~win case results of~n', [Indent]),
    format('~w  [] -> Nothing~n', [Indent]),
    format('~w  [h] -> Just (bindResult h)~n', [Indent]),
    format('~w  (h:restResults) ->~n', [Indent]),
    format('~w    let s1 = bindResult h~n', [Indent]),
    format('~w        outVar = case outReg of { Unbound v -> v; _ -> -1 }~n', [Indent]),
    format('~w        cp = ChoicePoint~n', [Indent]),
    format('~w          { cpNextPC = retPC, cpRegs = wsRegs s, cpStack = wsStack s~n', [Indent]),
    format('~w          , cpCP = wsCP s, cpTrailLen = wsTrailLen s~n', [Indent]),
    format('~w          , cpHeapLen = wsHeapLen s, cpBindings = wsBindings s~n', [Indent]),
    format('~w          , cpCutBar = wsCutBar s, cpAggFrame = Nothing~n', [Indent]),
    format('~w          , cpBuiltin = Just (HopsRetry outVar (map fromIntegral restResults) retPC)~n', [Indent]),
    format('~w          }~n', [Indent]),
    format('~w    in Just (s1 { wsCPs = cp : wsCPs s, wsCPsLen = wsCPsLen s + 1 })~n', [Indent]).

%% emit_stream_binding_multi(+OutputRegs, +Indent)
%  Multi-output stream binding. The kernel returns `[(T1, T2, ...)]` and
%  bindResult takes a tuple, binding each output to its register.
%  Uses FFIStreamRetry to store remaining tuples (as pre-wrapped Values)
%  in the choice point.
emit_stream_binding_multi(OutputRegs, Indent) :-
    length(OutputRegs, NOuts),
    format('~w    retPC = wsCP s~n', [Indent]),
    % Deref each output register
    emit_multi_out_derefs(OutputRegs, Indent),
    % Emit bindResult: pattern matches a tuple, binds each output
    format('~w    bindResult ', [Indent]),
    emit_tuple_pattern(NOuts),
    format(' =~n', []),
    format('~w      let ', [Indent]),
    emit_multi_wrap_bindings(OutputRegs, 1),
    format('~w      in s { wsPC = retPC~n', [Indent]),
    emit_multi_reg_updates(OutputRegs, Indent),
    emit_multi_binding_updates(OutputRegs, Indent),
    emit_multi_trail_updates(OutputRegs, Indent),
    format('~w         , wsTrailLen = wsTrailLen s + ~w~n', [Indent, NOuts]),
    format('~w         }~n', [Indent]),
    % Dispatch over result stream
    format('~win case results of~n', [Indent]),
    format('~w  [] -> Nothing~n', [Indent]),
    format('~w  [h] -> Just (bindResult h)~n', [Indent]),
    format('~w  (h:restResults) ->~n', [Indent]),
    format('~w    let s1 = bindResult h~n', [Indent]),
    emit_multi_outvars(OutputRegs, Indent),
    format('~w        restWrapped = map (\\', [Indent]),
    emit_tuple_pattern(NOuts),
    format(' -> [', []),
    emit_multi_wrap_list(OutputRegs, 1),
    format(']) restResults~n', []),
    format('~w        cp = ChoicePoint~n', [Indent]),
    format('~w          { cpNextPC = retPC, cpRegs = wsRegs s, cpStack = wsStack s~n', [Indent]),
    format('~w          , cpCP = wsCP s, cpTrailLen = wsTrailLen s~n', [Indent]),
    format('~w          , cpHeapLen = wsHeapLen s, cpBindings = wsBindings s~n', [Indent]),
    format('~w          , cpCutBar = wsCutBar s, cpAggFrame = Nothing~n', [Indent]),
    format('~w          , cpBuiltin = Just (FFIStreamRetry ', [Indent]),
    emit_outregs_list(OutputRegs),
    format(' outVars restWrapped retPC)~n', []),
    format('~w          }~n', [Indent]),
    format('~w    in Just (s1 { wsCPs = cp : wsCPs s, wsCPsLen = wsCPsLen s + 1 })~n', [Indent]).

%% Helpers for multi-output emission:

% Emit `outReg_1 = ...`, `outReg_2 = ...`
emit_multi_out_derefs([], _).
emit_multi_out_derefs([output(RegN, _)|Rest], Indent) :-
    format('~w    outReg_~w = derefVar (wsBindings s) $ fromMaybe (Unbound (-1)) (IM.lookup ~w (wsRegs s))~n',
           [Indent, RegN, RegN]),
    emit_multi_out_derefs(Rest, Indent).

% Emit tuple pattern like (rv_1, rv_2)
emit_tuple_pattern(1) :- format('rv_1', []).
emit_tuple_pattern(N) :-
    N > 1,
    format('(', []),
    emit_tuple_pattern_args(1, N),
    format(')', []).

emit_tuple_pattern_args(I, N) :-
    I < N, !,
    format('rv_~w, ', [I]),
    I1 is I + 1,
    emit_tuple_pattern_args(I1, N).
emit_tuple_pattern_args(N, N) :-
    format('rv_~w', [N]).

% Emit `w_1 = WrapExpr_1; w_2 = WrapExpr_2` for wrapped values
emit_multi_wrap_bindings([], _) :- format('~n', []).
emit_multi_wrap_bindings([output(_, Type)|Rest], I) :-
    result_wrap_expr_for_rv(Type, I, WrapExpr),
    (   I =:= 1
    ->  format('w_~w = ~w', [I, WrapExpr])
    ;   format('; w_~w = ~w', [I, WrapExpr])
    ),
    I1 is I + 1,
    emit_multi_wrap_bindings(Rest, I1).

% Emit `, wsRegs = IM.insert R1 w_1 $ IM.insert R2 w_2 $ wsRegs s`
emit_multi_reg_updates(OutputRegs, Indent) :-
    format('~w         , wsRegs = ', [Indent]),
    emit_reg_insert_chain(OutputRegs, 1),
    format('wsRegs s~n', []).

emit_reg_insert_chain([], _).
emit_reg_insert_chain([output(RegN, _)|Rest], I) :-
    format('IM.insert ~w w_~w $ ', [RegN, I]),
    I1 is I + 1,
    emit_reg_insert_chain(Rest, I1).

% Emit `, wsBindings = IM.insert vid_1 w_1 $ IM.insert vid_2 w_2 $ wsBindings s`
% Only inserts for Unbound outputs; bound outputs are no-ops.
emit_multi_binding_updates(OutputRegs, Indent) :-
    format('~w         , wsBindings = ', [Indent]),
    emit_binding_insert_chain(OutputRegs, 1),
    format('wsBindings s~n', []).

emit_binding_insert_chain([], _).
emit_binding_insert_chain([output(RegN, _)|Rest], I) :-
    format('(case outReg_~w of { Unbound v -> IM.insert v w_~w; _ -> id }) $ ',
           [RegN, I]),
    I1 is I + 1,
    emit_binding_insert_chain(Rest, I1).

% Emit `, wsTrail = TrailEntry vid_1 _ : TrailEntry vid_2 _ : wsTrail s`
emit_multi_trail_updates(OutputRegs, Indent) :-
    format('~w         , wsTrail = ', [Indent]),
    emit_trail_entry_chain(OutputRegs, 1),
    format('wsTrail s~n', []).

emit_trail_entry_chain([], _).
emit_trail_entry_chain([output(RegN, _)|Rest], I) :-
    format('(case outReg_~w of { Unbound v -> (TrailEntry v (IM.lookup v (wsBindings s)) :); _ -> id }) $ ',
           [RegN]),
    I1 is I + 1,
    emit_trail_entry_chain(Rest, I1).

% outVars list: [case outReg_R1 of { Unbound v -> v; _ -> -1 }, ...]
emit_multi_outvars(OutputRegs, Indent) :-
    format('~w        outVars = [', [Indent]),
    emit_outvars_list(OutputRegs, 1),
    format(']~n', []).

emit_outvars_list([], _).
emit_outvars_list([output(RegN, _)|Rest], I) :-
    (   I =:= 1 -> true ; format(', ', []) ),
    format('case outReg_~w of { Unbound v -> v; _ -> -1 }', [RegN]),
    I1 is I + 1,
    emit_outvars_list(Rest, I1).

% Output register numbers list for FFIStreamRetry
emit_outregs_list(OutputRegs) :-
    format('[', []),
    emit_outregs_list_items(OutputRegs, 1),
    format(']', []).

emit_outregs_list_items([], _).
emit_outregs_list_items([output(RegN, _)|Rest], I) :-
    (   I =:= 1 -> true ; format(', ', []) ),
    format('~w', [RegN]),
    I1 is I + 1,
    emit_outregs_list_items(Rest, I1).

% restWrapped list wrapping each tuple value
emit_multi_wrap_list([], _).
emit_multi_wrap_list([output(_, Type)|Rest], I) :-
    result_wrap_expr_for_rv(Type, I, WrapExpr),
    (   I =:= 1 -> true ; format(', ', []) ),
    format('~w', [WrapExpr]),
    I1 is I + 1,
    emit_multi_wrap_list(Rest, I1).

%% result_wrap_expr_for_rv(+Type, +Index, -HaskellExpr)
%  Like result_wrap_expr but using rv_<I> instead of rv.
result_wrap_expr_for_rv(integer, I, Expr) :-
    format(atom(Expr), 'Integer (fromIntegral rv_~w)', [I]).
result_wrap_expr_for_rv(atom, I, Expr) :-
    format(atom(Expr), 'Atom (fromMaybe "" (IM.lookup rv_~w (wcAtomDeintern ctx)))', [I]).
result_wrap_expr_for_rv(float, I, Expr) :-
    format(atom(Expr), 'Float rv_~w', [I]).

%% result_wrap_expr(+Type, -HaskellExpr)
%  Haskell expression wrapping `rv` (kernel result) into a Value
%  constructor. For `atom`, the kernel returns Int atom IDs that must
%  be de-interned back to Strings before wrapping in the Atom constructor.
%  For `integer` and `float`, no interning is involved.
result_wrap_expr(integer, 'Integer (fromIntegral rv)').
result_wrap_expr(atom, 'Atom (fromMaybe "" (IM.lookup rv (wcAtomDeintern ctx)))').
result_wrap_expr(float, 'Float rv').

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

%% compute_base_pcs(+Predicates, -BasePCMap)
%  Compute global start PCs for each predicate, matching the PC
%  assignment in compile_predicates_merged/4. Returns a list of
%  "pred/arity"-StartPC pairs.
compute_base_pcs(Predicates, Map) :-
    compute_base_pcs_(Predicates, 1, Map).

compute_base_pcs_([], _, []).
compute_base_pcs_([PI|Rest], StartPC, [Key-StartPC|RestMap]) :-
    (   PI = _Mod:Pred/Arity -> true ; PI = Pred/Arity ),
    format(atom(Key), '~w/~w', [Pred, Arity]),
    wam_haskell_predicate_wamcode(PI, WamCode),
    count_wam_instructions(WamCode, Count),
    NextPC is StartPC + Count,
    compute_base_pcs_(Rest, NextPC, RestMap).

count_wam_instructions(WamCode, Count) :-
    atom_string(WamCode, S),
    split_string(S, "\n", "", Lines),
    include(is_wam_instruction_line, Lines, InstrLines),
    length(InstrLines, Count).

is_wam_instruction_line(Line) :-
    split_string(Line, "", " \t", [Trimmed]),
    Trimmed \== "",
    \+ sub_string(Trimmed, _, 1, 0, ":").

% ============================================================================
% Haskell WAM Runtime Data Types
% ============================================================================
%
% Generated Haskell code uses these types:
%
%   data Value = Atom String | Integer Int | Float Double
%              | VList [Value] | Str String [Value]
%              | Unbound String | Ref Int
%              deriving (Eq, Ord, Show)
%
%   data WamState = WamState
%     { wsPC        :: !Int
%     , wsRegs      :: !(Map String Value)    -- A/X registers
%     , wsStack     :: ![EnvFrame]            -- environment frames
%     , wsHeap      :: ![Value]               -- term construction
%     , wsTrail     :: ![TrailEntry]           -- binding history
%     , wsCP        :: !Int                    -- continuation pointer
%     , wsCPs       :: ![ChoicePoint]          -- choice point stack
%     , wsBindings  :: !(Map String Value)     -- variable bindings
%     , wsCutBar    :: !Int                    -- cut barrier
%     }
%
%   data EnvFrame = EnvFrame !Int !(Map String Value)  -- saved CP + Y-regs
%
%   data TrailEntry = TrailEntry !String !(Maybe Value)
%
%   data ChoicePoint = ChoicePoint
%     { cpNextPC   :: !Int
%     , cpRegs     :: !(Map String Value)
%     , cpStack    :: ![EnvFrame]
%     , cpCP       :: !Int
%     , cpTrailLen :: !Int
%     , cpHeapLen  :: !Int
%     , cpBindings :: !(Map String Value)     -- O(1) snapshot!
%     , cpCutBar   :: !Int
%     }
%
% The critical insight: Map String Value uses structural sharing.
% When a ChoicePoint saves cpBindings = wsBindings state, no data is copied.
% Both point to the same tree. Mutations create new nodes only along the
% modified path (O(log n) per insert). Backtracking = swap the reference.

% ============================================================================
% PHASE 1: WAM Instruction → Haskell Expression
% ============================================================================

%% wam_to_haskell(+Instruction, -HaskellExpr)
%  Translates a single WAM instruction to a Haskell state transformation.
%  Each instruction is a function: WamState -> Maybe WamState
%  Nothing = failure, Just s = success with new state.

wam_to_haskell(get_constant(C, Ai), Code) :-
    format(string(Code),
'  let val = Map.lookup "~w" (wsRegs s)
  in case val of
    Just v | v == ~w -> Just (s { wsPC = wsPC s + 1 })
    Just (Unbound var) -> Just (s { wsPC = wsPC s + 1
                                  , wsRegs = Map.insert "~w" ~w (wsRegs s)
                                  , wsBindings = Map.insert var ~w (wsBindings s)
                                  , wsTrail = TrailEntry ("__binding__" ++ var) (Map.lookup var (wsBindings s)) : wsTrail s
                                  })
    _ -> Nothing', [Ai, C, Ai, C, C]).

wam_to_haskell(get_variable(Xn, Ai), Code) :-
    format(string(Code),
'  case Map.lookup "~w" (wsRegs s) of
    Just val -> let derefed = derefVar (wsBindings s) val
                    s1 = putReg "~w" derefed s
                in Just (s1 { wsPC = wsPC s + 1 })
    Nothing -> Nothing', [Ai, Xn]).

wam_to_haskell(put_value(Xn, Ai), Code) :-
    format(string(Code),
'  case getReg "~w" s of
    Just val -> Just (s { wsPC = wsPC s + 1
                        , wsRegs = Map.insert "~w" val (wsRegs s)
                        })
    Nothing -> Nothing', [Xn, Ai]).

wam_to_haskell(put_variable(Xn, Ai), Code) :-
    format(string(Code),
'  let var = Unbound ("_V" ++ show (wsPC s))
      s1 = putReg "~w" var s
  in Just (s1 { wsPC = wsPC s + 1
              , wsRegs = Map.insert "~w" var (wsRegs s1)
              })', [Xn, Ai]).

wam_to_haskell(put_constant(C, Ai), Code) :-
    format(string(Code),
'  Just (s { wsPC = wsPC s + 1
           , wsRegs = Map.insert "~w" ~w (wsRegs s)
           })', [Ai, C]).

wam_to_haskell(call(Pred, _Arity), Code) :-
    format(string(Code),
'  Just (s { wsPC = lookupLabel "~w" s
           , wsCP = wsPC s + 1
           })', [Pred]).

wam_to_haskell(proceed, Code) :-
    Code = '  let ret = wsCP s
  in if ret == 0 then Just (s { wsPC = 0 })  -- halt
     else Just (s { wsPC = ret, wsCP = 0 })'.

wam_to_haskell(allocate, Code) :-
    Code = '  let frame = EnvFrame (wsCP s) Map.empty
  in Just (s { wsPC = wsPC s + 1
             , wsStack = frame : wsStack s
             , wsCutBar = length (wsCPs s)
             })'.

wam_to_haskell(deallocate, Code) :-
    Code = '  case wsStack s of
    (EnvFrame oldCP _ : rest) -> Just (s { wsPC = wsPC s + 1
                                         , wsStack = rest
                                         , wsCP = oldCP
                                         })
    _ -> Nothing'.

wam_to_haskell(try_me_else(Label), Code) :-
    format(string(Code),
'  let cp = ChoicePoint
        { cpNextPC   = lookupLabel "~w" s
        , cpRegs     = wsRegs s       -- O(1): shared reference
        , cpStack    = wsStack s      -- O(1): shared reference
        , cpCP       = wsCP s
        , cpTrailLen = length (wsTrail s)
        , cpHeapLen  = length (wsHeap s)
        , cpBindings = wsBindings s   -- O(1): shared reference
        , cpCutBar   = wsCutBar s
        }
  in Just (s { wsPC = wsPC s + 1
             , wsCPs = cp : wsCPs s
             })', [Label]).

wam_to_haskell(trust_me, Code) :-
    Code = '  case wsCPs s of
    (_ : rest) -> Just (s { wsPC = wsPC s + 1, wsCPs = rest })
    [] -> Nothing'.

wam_to_haskell(retry_me_else(Label), Code) :-
    format(string(Code),
'  case wsCPs s of
    (cp : rest) -> let cp'' = cp { cpNextPC = lookupLabel "~w" s }
                   in Just (s { wsPC = wsPC s + 1
                              , wsCPs = cp'' : rest
                              })
    [] -> Nothing', [Label]).

wam_to_haskell(builtin_call('!/0', 0), Code) :-
    Code = '  Just (s { wsPC = wsPC s + 1
             , wsCPs = take (wsCutBar s) (wsCPs s)
             })'.

wam_to_haskell(builtin_call('is/2', 2), Code) :-
    Code = '  let expr = derefVar (wsBindings s) $ fromMaybe (Integer 0) (Map.lookup "A2" (wsRegs s))
      result = evalArith (wsBindings s) expr
      lhs = derefVar (wsBindings s) <$> Map.lookup "A1" (wsRegs s)
  in case (lhs, result) of
    (Just (Unbound var), Just r) ->
      let val = if fromIntegral (round r) == r then Integer (round r) else Float r
      in Just (s { wsPC = wsPC s + 1
                 , wsRegs = Map.insert "A1" val (wsRegs s)
                 , wsBindings = Map.insert var val (wsBindings s)
                 , wsTrail = TrailEntry ("__binding__" ++ var) (Map.lookup var (wsBindings s)) : wsTrail s
                 })
    (Just (Integer n), Just r) | fromIntegral n == r -> Just (s { wsPC = wsPC s + 1 })
    (Just (Float f), Just r) | f == r -> Just (s { wsPC = wsPC s + 1 })
    _ -> Nothing'.

wam_to_haskell(builtin_call('length/2', 2), Code) :-
    Code = '  let listVal = derefVar (wsBindings s) $ fromMaybe (VList []) (Map.lookup "A1" (wsRegs s))
      len = case listVal of VList items -> length items ; _ -> -1
      lhs = derefVar (wsBindings s) <$> Map.lookup "A2" (wsRegs s)
  in if len < 0 then Nothing
     else case lhs of
       Just (Unbound var) ->
         let val = Integer len
         in Just (s { wsPC = wsPC s + 1
                    , wsRegs = Map.insert "A2" val (wsRegs s)
                    , wsBindings = Map.insert var val (wsBindings s)
                    , wsTrail = TrailEntry ("__binding__" ++ var) (Map.lookup var (wsBindings s)) : wsTrail s
                    })
       Just (Integer n) | n == len -> Just (s { wsPC = wsPC s + 1 })
       _ -> Nothing'.

wam_to_haskell(builtin_call('</2', 2), Code) :-
    Code = '  let v1 = evalArith (wsBindings s) =<< (derefVar (wsBindings s) <$> Map.lookup "A1" (wsRegs s))
      v2 = evalArith (wsBindings s) =<< (derefVar (wsBindings s) <$> Map.lookup "A2" (wsRegs s))
  in case (v1, v2) of
    (Just a, Just b) | a < b -> Just (s { wsPC = wsPC s + 1 })
    _ -> Nothing'.

% Negation-as-failure: fast path for member/2
wam_to_haskell(builtin_call('\\+/1', 1), Code) :-
    Code = '  case derefHeap (wsHeap s) =<< Map.lookup "A1" (wsRegs s) of
    Just (Str "member/2" [needle, haystack]) ->
      let needle'' = derefVar (wsBindings s) needle
          haystack'' = derefVar (wsBindings s) haystack
          found = case haystack'' of
            VList items -> any (\\item -> derefVar (wsBindings s) item == needle'') items
            _ -> False
      in if found then Nothing  -- member succeeded, \\+ fails
         else Just (s { wsPC = wsPC s + 1 })
    _ -> Nothing  -- unsupported goal'.

% ============================================================================
% PHASE 2: Backtrack Function
% ============================================================================

backtrack_haskell(Code) :-
    Code = '-- | Restore state from the top choice point.
-- Dispatches: aggregate frame -> finalize, builtin -> resumeBuiltin, normal -> restore.
backtrack :: WamState -> Maybe WamState
backtrack s = case wsCPs s of
  [] -> Nothing
  (cp : rest) ->
    -- 1. Aggregate frame: finalize
    case cpAggFrame cp of { Just af -> finalizeAggregate (afReturnPC af) s; Nothing ->
    -- 2. Builtin state: resume (fact_retry etc.)
    case cpBuiltin cp of { Just bs -> resumeBuiltin bs cp rest s; Nothing ->
    -- 3. Normal: restore from CP
    let trailLen = cpTrailLen cp
        diff = wsTrailLen s - trailLen
        newEntries = reverse $ take diff (wsTrail s)
        restoredBindings = foldl'' undoBinding (cpBindings cp) newEntries
    in Just s { wsPC       = cpNextPC cp
              , wsRegs     = cpRegs cp
              , wsStack    = cpStack cp
              , wsCP       = cpCP cp
              , wsTrail    = drop diff (wsTrail s)
              , wsTrailLen = trailLen
              , wsHeap     = take (cpHeapLen cp) (wsHeap s)
              , wsHeapLen  = cpHeapLen cp
              , wsBindings = restoredBindings
              , wsCutBar   = cpCutBar cp
              } } }
  where
    undoBinding bindings (TrailEntry vid mOld) =
      case mOld of
        Just old -> IM.insert vid old bindings
        Nothing  -> IM.delete vid bindings

-- | Resume a builtin choice point. Tries next match, updates or pops CP.
resumeBuiltin :: BuiltinState -> ChoicePoint -> [ChoicePoint] -> WamState -> Maybe WamState
resumeBuiltin (FactRetry _ [] _) _ rest s =
  backtrack (s { wsCPs = rest, wsCPsLen = wsCPsLen s - 1 })
resumeBuiltin (FactRetry vid (v:vs) retPC) cp rest s =
  let newBindings = IM.insert vid (Atom v) (cpBindings cp)
      newRegs = IM.insert 2 (Atom v) (cpRegs cp)
      newCPs = case vs of
        [] -> rest
        _  -> cp { cpBuiltin = Just (FactRetry vid vs retPC) } : rest
      diff = wsTrailLen s - cpTrailLen cp
  in Just s { wsPC = retPC, wsRegs = newRegs, wsStack = cpStack cp
            , wsCP = cpCP cp
            , wsTrail = drop diff (wsTrail s)
            , wsTrailLen = cpTrailLen cp
            , wsHeap = take (cpHeapLen cp) (wsHeap s)
            , wsHeapLen = cpHeapLen cp
            , wsBindings = newBindings, wsCutBar = cpCutBar cp, wsCPs = newCPs }
resumeBuiltin (HopsRetry _ [] _) _ rest s =
  backtrack (s { wsCPs = rest, wsCPsLen = wsCPsLen s - 1 })
resumeBuiltin (HopsRetry vid (h:hs) retPC) cp rest s =
  let newBindings = IM.insert vid (Integer (fromIntegral h)) (cpBindings cp)
      newRegs = IM.insert 3 (Integer (fromIntegral h)) (cpRegs cp)
      newCPs = case hs of
        [] -> rest
        _  -> cp { cpBuiltin = Just (HopsRetry vid hs retPC) } : rest
      diff = wsTrailLen s - cpTrailLen cp
  in Just s { wsPC = retPC, wsRegs = newRegs, wsStack = cpStack cp
            , wsCP = cpCP cp
            , wsTrail = drop diff (wsTrail s)
            , wsTrailLen = cpTrailLen cp
            , wsHeap = take (cpHeapLen cp) (wsHeap s)
            , wsHeapLen = cpHeapLen cp
            , wsBindings = newBindings, wsCutBar = cpCutBar cp, wsCPs = newCPs }

-- Multi-output FFI retry: each remaining tuple is already a list of
-- wrapped Values matching outRegs/outVars in order.
resumeBuiltin (FFIStreamRetry _ _ [] _) _ rest s =
  backtrack (s { wsCPs = rest, wsCPsLen = wsCPsLen s - 1 })
resumeBuiltin (FFIStreamRetry outRegs outVars (tuple:rest_tuples) retPC) cp rest s =
  let -- Insert each (reg, value) from the tuple into the registers.
      newRegs = foldr (\\(rN, v) m -> IM.insert rN v m) (cpRegs cp)
                      (zip outRegs tuple)
      -- Insert each (varId, value) into bindings, skipping varId = -1
      -- (meaning the output was originally bound, so no binding update).
      newBindings = foldr (\\(vid, v) m ->
                             if vid == -1 then m else IM.insert vid v m)
                          (cpBindings cp)
                          (zip outVars tuple)
      newCPs = case rest_tuples of
        [] -> rest
        _  -> cp { cpBuiltin = Just (FFIStreamRetry outRegs outVars rest_tuples retPC) } : rest
      diff = wsTrailLen s - cpTrailLen cp
  in Just s { wsPC = retPC, wsRegs = newRegs, wsStack = cpStack cp
            , wsCP = cpCP cp
            , wsTrail = drop diff (wsTrail s)
            , wsTrailLen = cpTrailLen cp
            , wsHeap = take (cpHeapLen cp) (wsHeap s)
            , wsHeapLen = cpHeapLen cp
            , wsBindings = newBindings, wsCutBar = cpCutBar cp, wsCPs = newCPs }

-- | Backtrack skipping past the aggregate_frame CP. If the top CP is
-- an aggregate frame, return Nothing (inner solutions exhausted).
-- Otherwise, normal backtrack.
backtrackInner :: Int -> WamState -> Maybe WamState
backtrackInner returnPC s = case wsCPs s of
  (cp : _)
    | Just _ <- cpAggFrame cp -> Nothing  -- reached aggregate frame = done
    | otherwise -> backtrack s
  [] -> Nothing

-- | Finalize an aggregate: pop CPs to the aggregate frame, apply the
-- aggregation function, bind the result register.
-- | Update only the nearest aggregate frame CP with returnPC. O(k) where
-- k is the number of inner CPs above the aggregate frame, not O(n) over all CPs.
updateNearestAggFrame :: Int -> [ChoicePoint] -> [ChoicePoint]
updateNearestAggFrame _ [] = []
updateNearestAggFrame rpc (cp:rest) = case cpAggFrame cp of
  Just af -> cp { cpAggFrame = Just af { afReturnPC = rpc } } : rest
  Nothing -> cp : updateNearestAggFrame rpc rest

finalizeAggregate :: Int -> WamState -> Maybe WamState
finalizeAggregate returnPC s = go (wsCPs s)
  where
    go [] = Nothing
    go (cp : rest) = case cpAggFrame cp of
      Just (AggFrame typ _valReg resReg _ _) ->
        let accum = reverse (wsAggAccum s)
            result = applyAggregation typ accum
            -- Restore the CP snapshot state so we can read Y-registers
            -- from the stack (cpRegs only has A/X registers).
            cpState = s { wsRegs = cpRegs cp, wsStack = cpStack cp
                        , wsBindings = cpBindings cp }
            resVal = derefVar (cpBindings cp) <$> getReg resReg cpState
            -- Restore trail to the CP snapshot (drop entries added since)
            diff = wsTrailLen s - cpTrailLen cp
            restoredTrail = drop diff (wsTrail s)
            (finalRegs, finalBindings, finalStack, finalTrail, finalTrailLen) = case resVal of
              Just (Unbound vid) ->
                ( IM.insert resReg result (cpRegs cp)
                , IM.insert vid result (cpBindings cp)
                , putRegStack resReg result (cpStack cp)
                , TrailEntry vid (IM.lookup vid (cpBindings cp)) : restoredTrail
                , cpTrailLen cp + 1
                )
              _ -> (cpRegs cp, cpBindings cp, cpStack cp, restoredTrail, cpTrailLen cp)
        in Just s { wsPC = returnPC
                  , wsRegs = finalRegs
                  , wsStack = finalStack
                  , wsBindings = finalBindings
                  , wsTrail = finalTrail
                  , wsTrailLen = finalTrailLen
                  , wsHeap = take (cpHeapLen cp) (wsHeap s)
                  , wsHeapLen = cpHeapLen cp
                  , wsCP = cpCP cp
                  , wsCPs = rest
                  , wsCPsLen = wsCPsLen s - 1
                  , wsAggAccum = []
                  }
      Nothing -> go rest  -- skip non-aggregate CPs
    putRegStack rid val [] = []
    putRegStack rid val (EnvFrame ecp yregs : rest) =
      EnvFrame ecp (IM.insert rid val yregs) : rest
    putRegStack rid val (x : rest) = x : putRegStack rid val rest

-- ============================================================================
-- Phase 4.2: Intra-query parallel fork at ParTryMeElse
-- ============================================================================

-- | Entry point for ParTryMeElse execution. Picks fork vs sequential.
-- Accepts either a Left label (pre-resolution) or Right targetPC
-- (post-resolution) for the else branch. The "this branch" always
-- starts at wsPC + 1 regardless of which variant fired.
-- | Phase 4.5: minimum branch count below which the fork is not worth
-- the spark overhead. With fewer than this many branches, the fork
-- falls back to sequential TryMeElse. Default 3: a 2-clause predicate
-- (like category_ancestor base+recursive) stays sequential; a
-- multi-clause predicate with 3+ alternatives forks.
--
-- Rationale: parMap rdeepseq has fixed overhead per spark (~5-10μs on
-- GHC 9.x). With 2 branches where one is trivial, the overhead exceeds
-- the benefit. With 3+ balanced branches, the amortized overhead per
-- branch drops below the per-branch work.
forkMinBranches :: Int
forkMinBranches = 3

forkOrSequential :: WamContext -> WamState -> Either String Int -> Maybe WamState
forkOrSequential !ctx s elseTarget =
  case currentAggMergeStrategy s of
    Just ms | isForkableStrategy ms ->
      let elsePC = case elseTarget of
            Right pc -> pc
            Left lbl -> fromMaybe (-1) (Map.lookup lbl (wcLabels ctx))
      in if elsePC > 0
         then let branches = enumerateParBranches ctx (wsPC s) elsePC
              in if length branches >= forkMinBranches
                 then Just (forkParBranches ctx s ms elsePC)
                 else fallback  -- too few branches; overhead > benefit
         else fallback
    _ -> fallback
  where
    fallback = case elseTarget of
      Left lbl -> step ctx s (TryMeElse lbl)
      Right pc -> step ctx s (TryMeElsePc pc)

-- | Locate the nearest surrounding aggregate frame and return its
-- merge strategy. Returns Nothing when no aggregate frame is active.
currentAggMergeStrategy :: WamState -> Maybe MergeStrategy
currentAggMergeStrategy s = go (wsCPs s)
  where
    go [] = Nothing
    go (cp : rest) = case cpAggFrame cp of
      Just af -> Just (afMergeStrategy af)
      Nothing -> go rest

-- | Phase 4.2 scope: only sum and count merge strategies fork.
-- Findall/bag/set land in Phase 4.3; race/negation in 4.4.
isForkableStrategy :: MergeStrategy -> Bool
isForkableStrategy MergeSumInt    = True
isForkableStrategy MergeSumDouble = True
isForkableStrategy MergeCount     = True
isForkableStrategy _              = False

-- | Enumerate the entry PCs of every branch in a Par* choice-point
-- chain. The chain is laid out as:
--
--     ParTryMeElse  L1   <-- starting at parPC (wsPC s)
--     <branch 0 body>
--   L1:
--     ParRetryMeElse L2
--     <branch 1 body>
--   L2:
--     ParTrustMe
--     <branch N body>
--
-- Each branch''s body begins at `chainOpPC + 1`. We walk the chain by
-- following the else-label of each non-terminal op. Returns the entry
-- PC of every branch in chain order.
enumerateParBranches :: WamContext -> Int -> Int -> [Int]
enumerateParBranches ctx parPC elsePC =
    (parPC + 1) : collectRest elsePC
  where
    (lo, hi) = bounds (wcCode ctx)
    collectRest pc
      | pc < lo || pc > hi = []  -- safety: malformed chain
      | otherwise = case wcCode ctx ! pc of
          ParRetryMeElse nextLabel ->
            (pc + 1) : collectRest (fromMaybe (-1) (Map.lookup nextLabel (wcLabels ctx)))
          ParRetryMeElsePc nextPC ->
            (pc + 1) : collectRest nextPC
          ParTrustMe -> [pc + 1]
          -- Pre-Par variants can appear if someone mixed sequential
          -- and parallel chain entries. Treat them as chain
          -- terminators for safety — the fork still covers everything
          -- up to that point.
          RetryMeElse _   -> []
          RetryMeElsePc _ -> []
          TrustMe         -> []
          _ -> []

-- | Run one branch of a forked Par* chain. Starts from the given
-- branch entry PC with a fresh wsAggAccum, reuses the parent''s
-- bindings / CPs / trail. Runs until the branch''s own sub-solutions
-- exhaust. Returns the values the branch contributed to the
-- aggregate — the parent merges these across branches.
--
-- Key invariant: when the branch''s EndAggregate would fire
-- finalizeAggregate (i.e. the outer aggregate CP is next to pop), we
-- instead *stop* and return wsAggAccum. The fork driver then merges
-- all branches'' contributions and calls finalizeAggregate once at
-- the outer level.
runBranchForFork :: WamContext -> WamState -> Int -> [Value]
runBranchForFork !ctx !parent !branchPC =
    let branchInit = parent
          { wsPC = branchPC
          , wsAggAccum = []
          -- Protect parent CPs from being removed by !/0 inside the
          -- branch. The branch''s wsCutBar is set to the parent''s
          -- current CP depth so only CPs the branch itself creates
          -- can be cut. Without this, a clause like
          --   p(…) :- max_depth(M), length(V,D), D<M, !, …
          -- would pop the parent''s aggregate-frame CP.
          , wsCutBar = wsCPsLen parent
          }
    in runBranchLoop branchInit
  where
    runBranchLoop !s
      | wsPC s < fst (bounds (wcCode ctx)) = wsAggAccum s
      | wsPC s > snd (bounds (wcCode ctx)) = wsAggAccum s
      | otherwise =
          let instr = wcCode ctx ! wsPC s
          in case instr of
               EndAggregate valReg ->
                 let val = derefVar (wsBindings s) $
                           fromMaybe (Integer 0) (getReg valReg s)
                     s1 = s { wsAggAccum = val : wsAggAccum s
                            , wsPC = wsPC s + 1 }
                 -- Prefer backtrackInner: if the branch has more
                 -- internal solutions, keep exploring. Otherwise the
                 -- branch is exhausted — return its contribution
                 -- without finalizing the outer aggregate.
                 in case backtrackInner (wsPC s + 1) s1 of
                      Just s2 -> runBranchLoop s2
                      Nothing -> wsAggAccum s1
               -- Suppress nested forks: redirect Par* to sequential
               -- equivalents inside a branch. Only the OUTERMOST
               -- ParTryMeElse (the one that triggered forkParBranches)
               -- actually forks; inner recursive calls to the same
               -- predicate use sequential choice points. Without this,
               -- recursion depth D with branching factor B creates
               -- B^D nested parMap sparks — exponential explosion.
               _ -> let seqInstr = case instr of
                         ParTryMeElse lbl   -> TryMeElse lbl
                         ParTryMeElsePc p   -> TryMeElsePc p
                         ParRetryMeElse lbl -> RetryMeElse lbl
                         ParRetryMeElsePc p -> RetryMeElsePc p
                         ParTrustMe         -> TrustMe
                         other              -> other
                    in case step ctx s seqInstr of
                         Just s2 -> runBranchLoop s2
                         Nothing ->
                           -- Custom backtrack: if the top CP has an
                           -- aggregate frame, DON''T call finalizeAggregate
                           -- (which would clear wsAggAccum). Instead,
                           -- return our accumulated values — the branch
                           -- is done. Without this, the standard
                           -- backtrack function wipes wsAggAccum by
                           -- calling finalizeAggregate when it hits the
                           -- aggregate-frame CP.
                           case wsCPs s of
                             (cp : _) | Just _ <- cpAggFrame cp ->
                               wsAggAccum s
                             _ -> case backtrack s of
                               Just s3 -> runBranchLoop s3
                               Nothing -> wsAggAccum s

-- | Fork every branch of a Par* chain and merge their aggregate
-- contributions via the outer aggregate''s strategy. Returns the
-- post-finalize state (ready to resume after the outer EndAggregate).
forkParBranches :: WamContext -> WamState -> MergeStrategy -> Int -> WamState
forkParBranches !ctx !s _strategy !elsePC =
  let branchPCs = enumerateParBranches ctx (wsPC s) elsePC
      branchResults = parMap rdeepseq (runBranchForFork ctx s) branchPCs
      allValues = concat branchResults
      -- Combine into the current state''s accumulator before
      -- finalizing. finalizeAggregate''s applyAggregation folds
      -- these per the aggregate''s afType (which matches the
      -- strategy — sum folds as sum, count counts, etc.).
      combined = s { wsAggAccum = allValues ++ wsAggAccum s }
      -- finalizeAggregate wants the returnPC. For the outer aggregate
      -- this is the PC just after the matching EndAggregate. We
      -- locate it by scanning forward from wsPC s (the ParTryMeElse)
      -- for the first EndAggregate. All Par* branches share this
      -- returnPC because they share the enclosing BeginAggregate.
      retPC = findOuterEndAggregate ctx (wsPC s)
  in case finalizeAggregate retPC combined of
       Just sf -> sf
       Nothing -> combined  -- shouldn''t happen; defensive

-- | Scan forward from the given PC looking for the first EndAggregate.
-- Returns the PC immediately after it (the aggregate''s return
-- target). Returns 0 on overrun; finalizeAggregate handles that
-- gracefully via its CP walk.
findOuterEndAggregate :: WamContext -> Int -> Int
findOuterEndAggregate !ctx !startPC =
    let (_, hi) = bounds (wcCode ctx)
    in go (startPC + 1) hi
  where
    go !pc !hi
      | pc > hi   = 0
      | otherwise = case wcCode ctx ! pc of
          EndAggregate _ -> pc + 1
          _              -> go (pc + 1) hi

-- | Apply aggregation function to collected values.
applyAggregation :: String -> [Value] -> Value
applyAggregation "sum" vals =
  let toNum (Integer n) = fromIntegral n
      toNum (Float f) = f
      toNum _ = 0
      s = sum (map toNum vals)
  in if fromIntegral (round s :: Int) == s then Integer (round s) else Float s
applyAggregation "count" vals = Integer (length vals)
applyAggregation "collect" vals = VList vals
applyAggregation _ vals = VList vals

-- ============================================================================
-- Foreign Function Interface: native Haskell implementations of expensive
-- recursive predicates. Auto-generated from kernel detection.
-- ============================================================================

{{kernel_functions}}

-- | Execute a foreign predicate call. Computes all results natively,
-- returns first result with CPs for the rest.
-- | Indexed fact dispatch for 2-arg facts via BuiltinState CP.
-- O(1) Map lookup, first match returned, FactRetry CP for the rest.
callIndexedFact2 :: WamContext -> String -> WamState -> Maybe WamState
callIndexedFact2 !ctx pred s =
  let basePred = takeWhile (/= ''/'') pred
      retPC = wsCP s
  in case Map.lookup basePred (wcForeignFacts ctx) of
    Nothing -> Nothing
    Just factIndex ->
      let a1 = derefVar (wsBindings s) $ fromMaybe (Atom "") (IM.lookup 1 (wsRegs s))
          a2 = derefVar (wsBindings s) $ fromMaybe (Unbound (-1)) (IM.lookup 2 (wsRegs s))
      in case a1 of
        Atom key -> case Map.lookup key factIndex of
          Just (v:rest) -> case a2 of
            Unbound vid ->
              let newRegs = IM.insert 2 (Atom v) (wsRegs s)
                  newBindings = IM.insert vid (Atom v) (wsBindings s)
                  newTrail = TrailEntry vid (IM.lookup vid (wsBindings s)) : wsTrail s
                  newCPs = case rest of
                    [] -> wsCPs s  -- single match, no CP
                    _  -> ChoicePoint
                            { cpNextPC = retPC, cpRegs = wsRegs s, cpStack = wsStack s
                            , cpCP = wsCP s, cpTrailLen = wsTrailLen s
                            , cpHeapLen = wsHeapLen s, cpBindings = wsBindings s
                            , cpCutBar = wsCutBar s, cpAggFrame = Nothing
                            , cpBuiltin = Just (FactRetry vid rest retPC)
                            } : wsCPs s
                  newCPsLen = case rest of { [] -> wsCPsLen s; _ -> wsCPsLen s + 1 }
              in Just (s { wsPC = retPC, wsRegs = newRegs, wsBindings = newBindings
                         , wsTrail = newTrail, wsTrailLen = wsTrailLen s + 1
                         , wsCPs = newCPs, wsCPsLen = newCPsLen })
            Atom existing ->
              if existing == v then Just (s { wsPC = retPC })
              else case filter (== existing) rest of
                (_:_) -> Just (s { wsPC = retPC })
                [] -> Nothing
            _ -> Nothing
          _ -> Nothing
        _ -> Nothing

{{execute_foreign}}

-- | Bind an output register to a value WITHOUT advancing PC.
-- Used by term-inspection builtins that need to bind multiple
-- output positions in sequence before a single PC advance at the
-- end of the case. If the register is already bound to an equal
-- value, succeeds without side-effects; if it''s bound to an
-- unequal value, fails. Otherwise binds and trails.
bindOutput :: Int -> Value -> WamState -> Maybe WamState
bindOutput reg val s = case derefVar (wsBindings s) <$> IM.lookup reg (wsRegs s) of
  Just (Unbound vid) -> Just (s
    { wsRegs = IM.insert reg val (wsRegs s)
    , wsBindings = IM.insert vid val (wsBindings s)
    , wsTrail = TrailEntry vid (IM.lookup vid (wsBindings s)) : wsTrail s
    , wsTrailLen = wsTrailLen s + 1
    })
  Just existing | existing == val -> Just s
  _ -> Nothing

-- | copy_term/2 walker: recursively copies a Value, mapping each
-- distinct source variable id to exactly one fresh destination
-- variable id to preserve sharing within the copy. Threaded state
-- is (counter, varMap). Atomic values clone as-is.
copyTermWalk :: Int -> IM.IntMap Int -> Value -> (Value, Int, IM.IntMap Int)
copyTermWalk !c !m (Unbound vid) = case IM.lookup vid m of
  Just nv -> (Unbound nv, c, m)
  Nothing -> (Unbound c, c + 1, IM.insert vid c m)
copyTermWalk !c !m (Str fn args) =
  let (newArgs, c1, m1) = copyTermArgs c m args
  in (Str fn newArgs, c1, m1)
copyTermWalk !c !m (VList items) =
  let (newItems, c1, m1) = copyTermArgs c m items
  in (VList newItems, c1, m1)
copyTermWalk !c !m v = (v, c, m)

copyTermArgs :: Int -> IM.IntMap Int -> [Value] -> ([Value], Int, IM.IntMap Int)
copyTermArgs !c !m [] = ([], c, m)
copyTermArgs !c !m (x : xs) =
  let (x1, c1, m1) = copyTermWalk c m x
      (xs1, c2, m2) = copyTermArgs c1 m1 xs
  in (x1 : xs1, c2, m2)

-- | Unify two values, binding unbound variables.
unifyVal :: Value -> Value -> WamState -> Maybe WamState
unifyVal (Unbound vid) val s =
  Just (s { wsPC = wsPC s + 1
          , wsBindings = IM.insert vid val (wsBindings s)
          , wsTrail = TrailEntry vid (IM.lookup vid (wsBindings s)) : wsTrail s
          , wsTrailLen = wsTrailLen s + 1
          })
unifyVal val (Unbound vid) s =
  Just (s { wsPC = wsPC s + 1
          , wsBindings = IM.insert vid val (wsBindings s)
          , wsTrail = TrailEntry vid (IM.lookup vid (wsBindings s)) : wsTrail s
          , wsTrailLen = wsTrailLen s + 1
          })
unifyVal a b s | a == b = Just (s { wsPC = wsPC s + 1 })
               | otherwise = Nothing'.

% ============================================================================
% PHASE 3: Step Function + Run Loop
% ============================================================================

step_function_haskell(Code) :-
    Code = '-- | Execute a single WAM instruction.
-- The WamContext argument is read-only and threaded through (does NOT
-- become part of any per-step record allocation).
step :: WamContext -> WamState -> Instruction -> Maybe WamState
step !ctx s (GetConstant c ai) =
  let val = derefVar (wsBindings s) <$> IM.lookup ai (wsRegs s)
  in case val of
    Just v | v == c -> Just (s { wsPC = wsPC s + 1 })
    Just (Unbound vid) ->
      Just (s { wsPC = wsPC s + 1
              , wsRegs = IM.insert ai c (wsRegs s)
              , wsBindings = IM.insert vid c (wsBindings s)
              , wsTrail = TrailEntry vid (IM.lookup vid (wsBindings s)) : wsTrail s
              , wsTrailLen = wsTrailLen s + 1
              })
    _ -> Nothing

step !ctx s (GetVariable xn ai) =
  case IM.lookup ai (wsRegs s) of
    Just val -> let dv = derefVar (wsBindings s) val
                in Just ((putReg xn dv s) { wsPC = wsPC s + 1 })
    Nothing -> Nothing

step !ctx s (GetValue xn ai) =
  let va = derefVar (wsBindings s) <$> IM.lookup ai (wsRegs s)
      vx = getReg xn s
  in case (va, vx) of
    (Just a, Just x) | a == x -> Just (s { wsPC = wsPC s + 1 })
    (Just (Unbound vid), Just x) ->
      Just (s { wsPC = wsPC s + 1
              , wsRegs = IM.insert ai x (wsRegs s)
              , wsBindings = IM.insert vid x (wsBindings s)
              , wsTrail = TrailEntry vid (IM.lookup vid (wsBindings s)) : wsTrail s
              , wsTrailLen = wsTrailLen s + 1
              })
    _ -> Nothing

step !ctx s (PutConstant c ai) =
  Just (s { wsPC = wsPC s + 1, wsRegs = IM.insert ai c (wsRegs s) })

step !ctx s (PutVariable xn ai) =
  let vid = wsVarCounter s
      var = Unbound vid
      s1 = putReg xn var s
  in Just (s1 { wsPC = wsPC s + 1
              , wsRegs = IM.insert ai var (wsRegs s1)
              , wsVarCounter = vid + 1
              })

step !ctx s (PutValue xn ai) =
  case getReg xn s of
    Just val -> Just (s { wsPC = wsPC s + 1, wsRegs = IM.insert ai val (wsRegs s) })
    Nothing -> Nothing

step !ctx s (PutStructure fn ai arity) =
  Just (s { wsPC = wsPC s + 1
          , wsBuilder = BuildStruct fn ai arity []
          })

step !ctx s (PutList ai) =
  Just (s { wsPC = wsPC s + 1
           , wsBuilder = BuildList ai []
           })

step !ctx s (SetValue xn) =
  case getReg xn s of
    Just val -> addToBuilder val s
    Nothing -> Nothing

step !ctx s (SetConstant c) =
  addToBuilder c s

-- Fast path: call has been pre-resolved to a target PC at load time.
-- No string lookup, no foreign/indexed dispatch — just a jump.
step !ctx s (CallResolved pc _arity) =
  Just (s { wsPC = pc, wsCP = wsPC s + 1 })

-- Foreign call: compile-time resolved. executeForeign is the sole dispatch
-- path — Nothing means no solutions (backtrack), never fallthrough.
step !ctx s (CallForeign pred _arity) =
  executeForeign ctx pred (s { wsCP = wsPC s + 1 })

-- Call dispatch for non-foreign, non-resolved predicates. Foreign predicates
-- are handled by CallForeign (resolved at compile time), so executeForeign
-- is NOT checked here — no ambiguity between "unhandled" and "no solutions".
step !ctx s (Call pred _arity) =
  let sc = s { wsCP = wsPC s + 1 }
  in case Map.lookup pred (wcLoweredPredicates ctx) of
    Just fn -> fn ctx sc
    Nothing -> case callIndexedFact2 ctx pred sc of
      Just sr -> Just sr
      Nothing -> case Map.lookup pred (wcLabels ctx) of
        Just pc -> Just (s { wsPC = pc, wsCP = wsPC s + 1 })
        Nothing -> Nothing

-- Jump: unconditional jump to a label (used in if-then-else compilation)
step !ctx s (Jump label) =
  case Map.lookup label (wcLabels ctx) of
    Just pc -> Just (s { wsPC = pc })
    Nothing -> Nothing

-- JumpPc: pre-resolved jump (no label lookup)
step !ctx s (JumpPc pc) = Just (s { wsPC = pc })

-- ExecutePc: pre-resolved tail call (direct PC jump, no wsCP change)
step !ctx s (ExecutePc pc) = Just (s { wsPC = pc })

-- Execute: tail call, like Call but without setting wsCP
step !ctx s (Execute pred) =
  case Map.lookup pred (wcLoweredPredicates ctx) of
    Just fn -> fn ctx s
    Nothing -> case callIndexedFact2 ctx pred s of
      Just sr -> Just sr
      Nothing -> case Map.lookup pred (wcLabels ctx) of
        Just pc -> Just (s { wsPC = pc })
        Nothing -> Nothing

step !ctx s Proceed =
  let ret = wsCP s
  in if ret == 0 then Just (s { wsPC = 0 })
     else Just (s { wsPC = ret, wsCP = 0 })

step !ctx s Allocate =
  let frame = EnvFrame (wsCP s) IM.empty
  in Just (s { wsPC = wsPC s + 1
             , wsStack = frame : wsStack s
             , wsCutBar = wsCPsLen s
             })

step !ctx s Deallocate =
  case wsStack s of
    (EnvFrame oldCP _ : rest) -> Just (s { wsPC = wsPC s + 1, wsStack = rest, wsCP = oldCP })
    _ -> Nothing

step !ctx s (TryMeElse label) =
  let nextPC = fromMaybe 0 $ Map.lookup label (wcLabels ctx)
      cp = ChoicePoint
        { cpNextPC   = nextPC
        , cpRegs     = wsRegs s
        , cpStack    = wsStack s
        , cpCP       = wsCP s
        , cpTrailLen = wsTrailLen s
        , cpHeapLen  = wsHeapLen s
        , cpBindings = wsBindings s
        , cpCutBar   = wsCutBar s
        , cpAggFrame = Nothing, cpBuiltin = Nothing
        }
  in Just (s { wsPC = wsPC s + 1, wsCPs = cp : wsCPs s, wsCPsLen = wsCPsLen s + 1 })

step !ctx s TrustMe =
  case wsCPs s of
    (_ : rest) -> Just (s { wsPC = wsPC s + 1, wsCPs = rest, wsCPsLen = wsCPsLen s - 1 })
    [] -> Nothing

step !ctx s (RetryMeElse label) =
  case wsCPs s of
    (cp : rest) ->
      let nextPC = fromMaybe 0 $ Map.lookup label (wcLabels ctx)
      in Just (s { wsPC = wsPC s + 1, wsCPs = cp { cpNextPC = nextPC } : rest })
    [] -> Nothing

-- Pre-resolved variants: direct PC, no label lookup
step !ctx s (TryMeElsePc nextPC) =
  let cp = ChoicePoint
        { cpNextPC   = nextPC
        , cpRegs     = wsRegs s
        , cpStack    = wsStack s
        , cpCP       = wsCP s
        , cpTrailLen = wsTrailLen s
        , cpHeapLen  = wsHeapLen s
        , cpBindings = wsBindings s
        , cpCutBar   = wsCutBar s
        , cpAggFrame = Nothing, cpBuiltin = Nothing
        }
  in Just (s { wsPC = wsPC s + 1, wsCPs = cp : wsCPs s, wsCPsLen = wsCPsLen s + 1 })

step !ctx s (RetryMeElsePc nextPC) =
  case wsCPs s of
    (cp : rest) -> Just (s { wsPC = wsPC s + 1, wsCPs = cp { cpNextPC = nextPC } : rest })
    [] -> Nothing

-- Phase 4.1 parallel-forkable variants. For now they alias their
-- sequential counterparts — the instructions mark the predicate as
-- fork-safe but the runtime doesn''t fork yet. Phase 4.2 will split
-- these handlers off to do actual parMap-based forking at the
-- surrounding aggregate boundary.
--
-- Phase 4.2: when a ParTryMeElse fires inside a fork-compatible
-- aggregate (sum / count), we collect all N alternative branch
-- entry PCs, run each branch in parallel via `parMap rdeepseq`, then
-- merge the accumulated values via the aggregate strategy. Each
-- branch''s EndAggregate is intercepted so it appends to that
-- branch''s local wsAggAccum without finalizing the outer aggregate.
-- Falls back to the sequential TryMeElse handler when the enclosing
-- aggregate is not fork-compatible (or when there is no aggregate at
-- all).
--
-- ParRetryMeElse / ParTrustMe still delegate to their sequential
-- counterparts — once ParTryMeElse has chosen to fork, the runtime
-- never walks through those; they''re only reached if the fork
-- path bailed out to sequential.
step !ctx s (ParTryMeElse label)    = forkOrSequential ctx s (Left label)
step !ctx s (ParRetryMeElse label)  = step ctx s (RetryMeElse label)
step !ctx s ParTrustMe              = step ctx s TrustMe
step !ctx s (ParTryMeElsePc pc)     = forkOrSequential ctx s (Right pc)
step !ctx s (ParRetryMeElsePc pc)   = step ctx s (RetryMeElsePc pc)

step !ctx s (SwitchOnConstantPc table) =
  let val = derefVar (wsBindings s) <$> IM.lookup 1 (wsRegs s)
  in case val of
    Just (Unbound _) -> Just (s { wsPC = wsPC s + 1 })
    Just (Atom key) -> case Map.lookup key table of
      Just pc -> Just (s { wsPC = pc })
      Nothing -> Nothing
    Just (Integer n) -> case Map.lookup (show n) table of
      Just pc -> Just (s { wsPC = pc })
      Nothing -> Nothing
    _ -> Nothing

step !ctx s (BuiltinCall "!/0" _) =
  -- Cut: truncate wsCPs to the barrier depth saved at clause Allocate.
  Just (s { wsPC = wsPC s + 1, wsCPs = take (wsCutBar s) (wsCPs s), wsCPsLen = wsCutBar s })

-- CutIte: soft cut for if-then-else — pops exactly the top choice point
-- (the one pushed by try_me_else for the Else branch). Unlike !/0 which
-- truncates to wsCutBar (clause-level), this only removes the immediately
-- enclosing if-then-else CP, preserving aggregate frames and outer CPs.
step !ctx s CutIte =
  case wsCPs s of
    (_cp : rest) -> Just (s { wsPC = wsPC s + 1, wsCPs = rest, wsCPsLen = wsCPsLen s - 1 })
    [] -> Just (s { wsPC = wsPC s + 1 })  -- no CP to pop (shouldn''t happen)

-- Type-checking builtins
step !ctx s (BuiltinCall "nonvar/1" _) =
  case derefVar (wsBindings s) <$> IM.lookup 1 (wsRegs s) of
    Just (Unbound _) -> Nothing
    Just _           -> Just (s { wsPC = wsPC s + 1 })
    Nothing          -> Nothing

step !ctx s (BuiltinCall "var/1" _) =
  case derefVar (wsBindings s) <$> IM.lookup 1 (wsRegs s) of
    Just (Unbound _) -> Just (s { wsPC = wsPC s + 1 })
    _                -> Nothing

step !ctx s (BuiltinCall "atom/1" _) =
  case derefVar (wsBindings s) <$> IM.lookup 1 (wsRegs s) of
    Just (Atom _) -> Just (s { wsPC = wsPC s + 1 })
    _             -> Nothing

step !ctx s (BuiltinCall "integer/1" _) =
  case derefVar (wsBindings s) <$> IM.lookup 1 (wsRegs s) of
    Just (Integer _) -> Just (s { wsPC = wsPC s + 1 })
    _                -> Nothing

step !ctx s (BuiltinCall "number/1" _) =
  case derefVar (wsBindings s) <$> IM.lookup 1 (wsRegs s) of
    Just (Integer _) -> Just (s { wsPC = wsPC s + 1 })
    Just (Float _)   -> Just (s { wsPC = wsPC s + 1 })
    _                -> Nothing

step !ctx s (BuiltinCall "is/2" _) =
  let expr = derefVar (wsBindings s) $ fromMaybe (Integer 0) (IM.lookup 2 (wsRegs s))
      result = evalArith (wsBindings s) expr
      lhs = derefVar (wsBindings s) <$> IM.lookup 1 (wsRegs s)
  in case (lhs, result) of
    (Just (Unbound vid), Just r) ->
      let val = if fromIntegral (round r :: Int) == r then Integer (round r) else Float r
      in Just (s { wsPC = wsPC s + 1
                 , wsRegs = IM.insert 1 val (wsRegs s)
                 , wsBindings = IM.insert vid val (wsBindings s)
                 , wsTrail = TrailEntry vid (IM.lookup vid (wsBindings s)) : wsTrail s
                 , wsTrailLen = wsTrailLen s + 1
                 })
    (Just (Integer n), Just r) | fromIntegral n == r -> Just (s { wsPC = wsPC s + 1 })
    _ -> Nothing

step !ctx s (BuiltinCall "length/2" _) =
  let listVal = derefVar (wsBindings s) $ fromMaybe (VList []) (IM.lookup 1 (wsRegs s))
  in case listVal of
    VList items ->
      let len = length items
          lhs = derefVar (wsBindings s) <$> IM.lookup 2 (wsRegs s)
      in case lhs of
        Just (Unbound vid) ->
          let val = Integer len
          in Just (s { wsPC = wsPC s + 1
                     , wsRegs = IM.insert 2 val (wsRegs s)
                     , wsBindings = IM.insert vid val (wsBindings s)
                     , wsTrail = TrailEntry vid (IM.lookup vid (wsBindings s)) : wsTrail s
                     , wsTrailLen = wsTrailLen s + 1
                     })
        Just (Integer n) | n == len -> Just (s { wsPC = wsPC s + 1 })
        _ -> Nothing
    _ -> Nothing

step !ctx s (BuiltinCall "</2" _) =
  let v1 = evalArith (wsBindings s) =<< (derefVar (wsBindings s) <$> IM.lookup 1 (wsRegs s))
      v2 = evalArith (wsBindings s) =<< (derefVar (wsBindings s) <$> IM.lookup 2 (wsRegs s))
  in case (v1, v2) of
    (Just a, Just b) | a < b -> Just (s { wsPC = wsPC s + 1 })
    _ -> Nothing

step !ctx s (BuiltinCall "\\\\+/1" _) =
  let goal = IM.lookup 1 (wsRegs s) >>= derefHeap (wsHeap s)
  in case goal of
    Just (Str fn [needle, haystack]) | "member" `isPrefixOf` fn ->
      let n = derefVar (wsBindings s) needle
          h = derefVar (wsBindings s) haystack
          found = case h of
            VList items -> any (\\item -> derefVar (wsBindings s) item == n) items
            _ -> False
      in if found then Nothing else Just (s { wsPC = wsPC s + 1 })
    _ -> Nothing

-- SwitchOnConstant: dispatch on A1 value via O(log n) Map lookup
step !ctx s (SwitchOnConstant table) =
  let val = derefVar (wsBindings s) <$> IM.lookup 1 (wsRegs s)
  in case val of
    Just (Unbound _) -> Just (s { wsPC = wsPC s + 1 })  -- unbound: skip
    Just v -> case Map.lookup v table of
      Just label -> case Map.lookup label (wcLabels ctx) of
        Just pc -> Just (s { wsPC = pc })
        Nothing -> Nothing
      Nothing -> Nothing  -- no match: fail
    Nothing -> Nothing

step !ctx s (BuiltinCall ">/2" _) =
  let v1 = evalArith (wsBindings s) =<< (derefVar (wsBindings s) <$> IM.lookup 1 (wsRegs s))
      v2 = evalArith (wsBindings s) =<< (derefVar (wsBindings s) <$> IM.lookup 2 (wsRegs s))
  in case (v1, v2) of
    (Just a, Just b) | a > b -> Just (s { wsPC = wsPC s + 1 })
    _ -> Nothing

-- member/2 builtin: A1=Elem, A2=List. Creates choice points for backtracking.
step !ctx s (BuiltinCall "member/2" _) =
  let elem_ = derefVar (wsBindings s) $ fromMaybe (Unbound (-1)) (IM.lookup 1 (wsRegs s))
      list_ = derefVar (wsBindings s) $ fromMaybe (VList []) (IM.lookup 2 (wsRegs s))
  in case list_ of
    VList (x:_) -> unifyVal elem_ x s
    _ -> Nothing

-- begin_aggregate: push aggregate frame CP, initialize accumulator, continue to goal body
step !ctx s (BeginAggregate typ valReg resReg) =
  let cp = ChoicePoint
        { cpNextPC   = wsPC s
        , cpRegs     = wsRegs s
        , cpStack    = wsStack s
        , cpCP       = wsCP s
        , cpTrailLen = wsTrailLen s
        , cpHeapLen  = wsHeapLen s
        , cpBindings = wsBindings s
        , cpCutBar   = wsCutBar s
        , cpAggFrame = Just (AggFrame typ valReg resReg 0
                                      (inferMergeStrategy typ))
        , cpBuiltin = Nothing
        }
  in Just (s { wsPC = wsPC s + 1
             , wsCPs = cp : wsCPs s
             , wsCPsLen = wsCPsLen s + 1
             , wsAggAccum = []
             })

-- end_aggregate: collect value, store returnPC in nearest aggregate frame, force backtrack
step !ctx s (EndAggregate valReg) =
  let val = derefVar (wsBindings s) $ fromMaybe (Integer 0) (getReg valReg s)
      returnPC = wsPC s + 1
      -- Update only the nearest (first) aggregate frame CP, not all CPs
      updatedCPs = updateNearestAggFrame returnPC (wsCPs s)
      s1 = s { wsAggAccum = val : wsAggAccum s, wsCPs = updatedCPs }
  in case backtrackInner returnPC s1 of
    Just s2 -> Just s2
    Nothing -> finalizeAggregate returnPC s1

-- functor/3: A1 = T, A2 = N, A3 = A. Read and construct modes
-- are dispatched on A1''s tag after dereferencing.
step !_ctx s (BuiltinCall "functor/3" _) =
  let t = derefVar (wsBindings s) <$> IM.lookup 1 (wsRegs s)
  in case t of
    Just (Unbound vid) ->
      -- Construct mode: need A2 (atom name) and A3 (integer arity).
      let nArg = derefVar (wsBindings s) <$> IM.lookup 2 (wsRegs s)
          aArg = derefVar (wsBindings s) <$> IM.lookup 3 (wsRegs s)
      in case (nArg, aArg) of
        (Just nameVal, Just (Integer arity)) | arity >= 0 ->
          let mBuilt = if arity == 0
                then Just (nameVal, wsVarCounter s)
                else case nameVal of
                  Atom fname ->
                    let c0 = wsVarCounter s
                        newArgs = [Unbound (c0 + i) | i <- [0 .. arity - 1]]
                    in Just (Str fname newArgs, c0 + arity)
                  _ -> Nothing
          in case mBuilt of
            Nothing -> Nothing
            Just (built, newCounter) -> Just (s
              { wsPC = wsPC s + 1
              , wsRegs = IM.insert 1 built (wsRegs s)
              , wsBindings = IM.insert vid built (wsBindings s)
              , wsTrail = TrailEntry vid (IM.lookup vid (wsBindings s)) : wsTrail s
              , wsTrailLen = wsTrailLen s + 1
              , wsVarCounter = newCounter
              })
        _ -> Nothing
    Just tVal ->
      -- Read mode: extract functor name and arity.
      let mInfo = case tVal of
            Str fn args -> Just (Atom fn, length args)
            VList [] -> Just (Atom "[]", 0)
            VList _ -> Just (Atom ".", 2)
            Atom _ -> Just (tVal, 0)
            Integer _ -> Just (tVal, 0)
            Float _ -> Just (tVal, 0)
            _ -> Nothing
      in case mInfo of
        Nothing -> Nothing
        Just (name, arity) ->
          case bindOutput 2 name s of
            Nothing -> Nothing
            Just s1 -> case bindOutput 3 (Integer arity) s1 of
              Nothing -> Nothing
              Just s2 -> Just (s2 { wsPC = wsPC s2 + 1 })
    Nothing -> Nothing

-- arg/3: A1 = N (integer, 1-based), A2 = T (compound/list),
-- A3 = output unified with the selected argument.
step !_ctx s (BuiltinCall "arg/3" _) =
  let n = derefVar (wsBindings s) <$> IM.lookup 1 (wsRegs s)
      t = derefVar (wsBindings s) <$> IM.lookup 2 (wsRegs s)
  in case (n, t) of
    (Just (Integer idx), Just tVal) | idx >= 1 ->
      let mArg = case tVal of
            Str _ args | idx <= length args -> Just (args !! (idx - 1))
            VList (x : _) | idx == 1 -> Just x
            VList (_ : xs) | idx == 2 -> Just (VList xs)
            _ -> Nothing
      in case mArg of
        Nothing -> Nothing
        Just a -> case bindOutput 3 a s of
          Nothing -> Nothing
          Just s1 -> Just (s1 { wsPC = wsPC s1 + 1 })
    _ -> Nothing

-- =../2 (univ): A1 = T, A2 = L. Decompose (instantiated A1) or
-- compose (unbound A1, list in A2).
step !_ctx s (BuiltinCall "=../2" _) =
  let t = derefVar (wsBindings s) <$> IM.lookup 1 (wsRegs s)
  in case t of
    Just (Unbound vid) ->
      -- Compose mode: read proper list from A2.
      let l = derefVar (wsBindings s) <$> IM.lookup 2 (wsRegs s)
      in case l of
        Just (VList items) ->
          let mBuilt = case items of
                [] -> Nothing
                [x] -> Just x
                (Atom fname : rest) -> Just (Str fname rest)
                _ -> Nothing
          in case mBuilt of
            Nothing -> Nothing
            Just built -> Just (s
              { wsPC = wsPC s + 1
              , wsRegs = IM.insert 1 built (wsRegs s)
              , wsBindings = IM.insert vid built (wsBindings s)
              , wsTrail = TrailEntry vid (IM.lookup vid (wsBindings s)) : wsTrail s
              , wsTrailLen = wsTrailLen s + 1
              })
        _ -> Nothing
    Just tVal ->
      -- Decompose mode: build list from T.
      let mList = case tVal of
            Str fn args -> Just (VList (Atom fn : args))
            Atom _ -> Just (VList [tVal])
            Integer _ -> Just (VList [tVal])
            Float _ -> Just (VList [tVal])
            VList [] -> Just (VList [Atom "[]"])
            VList (x : xs) -> Just (VList [Atom ".", x, VList xs])
            _ -> Nothing
      in case mList of
        Nothing -> Nothing
        Just lv -> case bindOutput 2 lv s of
          Nothing -> Nothing
          Just s1 -> Just (s1 { wsPC = wsPC s1 + 1 })
    Nothing -> Nothing

-- copy_term/2: A1 = T, A2 = Copy. Walks T with a var map to
-- preserve sharing, bumps wsVarCounter, binds A2 to the fresh copy.
step !_ctx s (BuiltinCall "copy_term/2" _) =
  let t = derefVar (wsBindings s) <$> IM.lookup 1 (wsRegs s)
  in case t of
    Just tVal ->
      let (copy, newCounter, _) = copyTermWalk (wsVarCounter s) IM.empty tVal
          s0 = s { wsVarCounter = newCounter }
      in case bindOutput 2 copy s0 of
        Nothing -> Nothing
        Just s1 -> Just (s1 { wsPC = wsPC s1 + 1 })
    Nothing -> Nothing

-- Fallback for unhandled instructions
step _ _ _ = Nothing'.

run_loop_haskell(Code) :-
    Code = '-- | Main execution loop. Runs until halt (pc=0) or failure.
-- Uses unsafeFetchInstr to avoid Maybe wrapping in the hot path.
-- Bounds are guaranteed by the WAM compiler: PC=0 is halt, otherwise PC
-- always points to a valid instruction within the code array.
-- The WamContext is read-only and threaded through (no per-step alloc).
run :: WamContext -> WamState -> Maybe WamState
run !ctx !s
  | wsPC s == 0 = Just s  -- halt
  | otherwise =
      let !instr = unsafeFetchInstr (wsPC s) (wcCode ctx)
      in case step ctx s instr of
           Just !s'' -> run ctx s''
           Nothing -> case backtrack s of
             Just !s'' -> run ctx s''
             Nothing -> Nothing

-- | Dispatch a Call to another predicate, trying all resolution paths.
-- Used by lowered predicate functions for inter-predicate calls.
-- Non-foreign call dispatch for lowered functions. Foreign predicates
-- are dispatched via callForeign (compile-time resolved), so
-- executeForeign is NOT checked here.
{-# NOINLINE dispatchCall #-}
dispatchCall :: WamContext -> String -> WamState -> Maybe WamState
dispatchCall !ctx pred !sc =
  case Map.lookup pred (wcLoweredPredicates ctx) of
    Just fn -> fn ctx sc
    Nothing -> case callIndexedFact2 ctx pred sc of
      Just sr -> Just sr
      Nothing -> case Map.lookup pred (wcLabels ctx) of
        Just pc -> run ctx (sc { wsPC = pc })
        Nothing -> Nothing

-- Foreign call for lowered functions. Calls executeForeign directly;
-- Nothing means no solutions (backtrack). No fallthrough.
{-# INLINE callForeign #-}
callForeign :: WamContext -> String -> WamState -> Maybe WamState
callForeign !ctx pred !sc = executeForeign ctx pred sc'.

% ============================================================================
% PHASE 4: Project Generation
% ============================================================================

%% write_wam_haskell_project(+Predicates, +Options, +ProjectDir)
%  Generates a complete Haskell project (cabal or stack) with:
%  - WamTypes.hs: data types and utility functions
%  - WamRuntime.hs: run loop and backtracking
%  - Predicates.hs: compiled predicates
%  - Main.hs: benchmark driver
write_wam_haskell_project(Predicates, Options, ProjectDir) :-
    make_directory_path(ProjectDir),
    directory_file_path(ProjectDir, 'src', SrcDir),
    make_directory_path(SrcDir),

    % Determine map backend: HashMap (faster) or Map (default fallback)
    option(use_hashmap(UseHM), Options, true),

    % Detect recursive kernels in the predicate list. Detected kernels
    % are handled by the FFI (executeForeign) at runtime and are excluded
    % from generic lowering. The detected kernel list is used to auto-
    % populate foreignPreds in Main.hs.
    % no_kernels(true) suppresses kernel detection — all predicates go
    % through the WAM interpreter (no FFI). Useful for benchmarking.
    (   option(no_kernels(true), Options)
    ->  DetectedKernels = [],
        format(user_error, '[WAM-Haskell] kernel detection suppressed~n', [])
    ;   detect_kernels(Predicates, DetectedKernels),
        (   DetectedKernels \= []
        ->  pairs_keys(DetectedKernels, DetectedKeys),
            format(user_error, '[WAM-Haskell] detected kernels: ~w~n', [DetectedKeys])
        ;   true
        )
    ),

    % Resolve emit_mode and partition predicates. Detected kernels are
    % explicitly excluded from lowering (they use FFI via CallForeign).
    wam_haskell_resolve_emit_mode(Options, EmitMode),
    wam_haskell_partition_predicates(EmitMode, Predicates, DetectedKernels, InterpretedList, LoweredList),
    length(InterpretedList, NInterp),
    length(LoweredList, NLower),
    format(user_error,
           '[WAM-Haskell] emit_mode=~w  interpreted=~w  lowered=~w~n',
           [EmitMode, NInterp, NLower]),

    % Generate WamTypes.hs
    generate_wam_types_hs(TypesCode0),
    apply_hashmap_rewrite(UseHM, types, TypesCode0, TypesCode),
    directory_file_path(SrcDir, 'WamTypes.hs', TypesPath),
    write_hs_file(TypesPath, TypesCode),

    % Generate WamRuntime.hs
    compile_wam_runtime_to_haskell(Options, DetectedKernels, RuntimeCode0),
    apply_hashmap_rewrite(UseHM, runtime, RuntimeCode0, RuntimeCode),
    directory_file_path(SrcDir, 'WamRuntime.hs', RuntimePath),
    write_hs_file(RuntimePath, RuntimeCode),

    % ALL predicates go into the interpreter's instruction array — even
    % lowered ones — so backtrack can land on alternate clauses that the
    % lowered function doesn't handle. Phase 4+ lowered functions only
    % inline clause 1; clause 2+ runs through the interpreter on backtrack.
    compile_predicates_to_haskell(Predicates, Options, PredsCode0),
    apply_hashmap_rewrite(UseHM, generic, PredsCode0, PredsCode),
    directory_file_path(SrcDir, 'Predicates.hs', PredsPath),
    write_hs_file(PredsPath, PredsCode),

    % Lower each predicate in LoweredList via the Phase 3+ emitter.
    % We first compute global base PCs matching the merged instruction
    % array so the lowered functions' PC references (wsCP for Call return
    % addresses, wsPC for step calls) are correct.
    compute_base_pcs(Predicates, BasePCMap),
    lower_all(LoweredList, BasePCMap, DetectedKernels, LoweredEntries),
    generate_lowered_hs(LoweredEntries, LoweredCode0),
    apply_hashmap_rewrite(UseHM, generic, LoweredCode0, LoweredCode),
    directory_file_path(SrcDir, 'Lowered.hs', LoweredPath),
    write_hs_file(LoweredPath, LoweredCode),

    % Generate cabal file
    option(module_name(ModName), Options, 'wam-haskell-bench'),
    generate_cabal_file(ModName, UseHM, Options, CabalCode),
    format(atom(CabalFile), '~w.cabal', [ModName]),
    directory_file_path(ProjectDir, CabalFile, CabalPath),
    write_hs_file(CabalPath, CabalCode),

    % Generate Main.hs
    generate_main_hs(Predicates, DetectedKernels, Options, MainCode0),
    apply_hashmap_rewrite(UseHM, main, MainCode0, MainCode),
    directory_file_path(SrcDir, 'Main.hs', MainPath),
    write_hs_file(MainPath, MainCode),

    format(user_error, '[WAM-Haskell] Generated project at: ~w (hashmap=~w)~n', [ProjectDir, UseHM]).

%% apply_hashmap_rewrite(+UseHM, +Module, +InCode, -OutCode)
%  When UseHM=true, rewrite Data.Map.Strict references to Data.HashMap.Strict.
apply_hashmap_rewrite(false, _, Code, Code) :- !.
apply_hashmap_rewrite(true, Module, Code0, Code) :-
    % Replace import line
    replace_substr(Code0, "import qualified Data.Map.Strict as Map",
                   "import qualified Data.HashMap.Strict as Map", Code1),
    % Replace Map.Map type constructor with Map.HashMap
    replace_substr(Code1, "Map.Map ", "Map.HashMap ", Code2),
    % HashMap has no toAscList — use toList instead (loses ordering, but
    % the only use site builds a SwitchOnConstant which doesn''t need it)
    replace_substr(Code2, "Map.toAscList", "Map.toList", Code3),
    % For WamTypes, add Hashable instance for Value (needed for HashMap keys)
    (   Module == types
    ->  replace_substr(Code3,
            "module WamTypes where\n\nimport qualified Data.HashMap.Strict as Map",
            "{-# LANGUAGE DeriveGeneric #-}\nmodule WamTypes where\n\nimport qualified Data.HashMap.Strict as Map\nimport Data.Hashable (Hashable)\nimport GHC.Generics (Generic)",
            Code4),
        replace_substr(Code4,
            "deriving (Eq, Ord, Show)",
            "deriving (Eq, Ord, Show, Generic)\ninstance Hashable Value",
            Code)
    ;   Code = Code3
    ).

%% replace_substr(+Str, +From, +To, -Result)
%  Replace all occurrences of From with To in Str.
replace_substr(Str, From, To, Result) :-
    atom_string(Str, S),
    atom_string(From, FS),
    atom_string(To, TS),
    re_split_replace(S, FS, TS, R),
    atom_string(Result, R).

re_split_replace(S, From, To, Result) :-
    (   sub_string(S, B, _Len, A, From)
    ->  sub_string(S, 0, B, _, Before),
        sub_string(S, _, A, 0, After),
        re_split_replace(After, From, To, AfterR),
        atom_string(BeforeA, Before),
        atom_string(ToA, To),
        atom_string(AfterRA, AfterR),
        atomic_list_concat([BeforeA, ToA, AfterRA], R),
        atom_string(R, Result)
    ;   Result = S
    ).

%% generate_lowered_hs(+LoweredEntries, -Code)
%  Emit the Lowered.hs module containing one function per predicate in
%  LoweredEntries and a loweredPredicates dispatch Map. Each entry is a
%  term lowered(PredName, FuncName, HaskellCode) produced by the
%  wam_haskell_lowered_emitter:lower_predicate_to_haskell/4 helper.
%
%  When the list is empty, emits a skeleton module with
%  loweredPredicates = Map.empty (Phase 2 shape). The Lowered.hs module
%  is emitted unconditionally so Main.hs can unconditionally
%  `import qualified Lowered`.
%
%  See docs/design/WAM_HASKELL_LOWERED_SPECIFICATION.md §2.1.
generate_lowered_hs([], Code) :- !,
    with_output_to(string(Code), (
        format("{-# LANGUAGE BangPatterns #-}~n"),
        format("-- WAM-lowered Haskell predicates (empty — no preds lowered).~n"),
        format("module Lowered where~n~n"),
        format("import qualified Data.Map.Strict as Map~n"),
        format("import qualified Data.IntMap.Strict as IM~n"),
        format("import WamTypes~n"),
        format("import Data.Maybe (fromMaybe)~n"),
        format("import WamRuntime~n~n"),
        format("loweredPredicates :: Map.Map String (WamContext -> WamState -> Maybe WamState)~n"),
        format("loweredPredicates = Map.empty~n")
    )).
generate_lowered_hs(LoweredEntries, Code) :-
    LoweredEntries = [_|_],  % non-empty
    with_output_to(string(Code), (
        format("{-# LANGUAGE BangPatterns #-}~n"),
        format("-- WAM-lowered Haskell predicates.~n"),
        format("--~n"),
        format("-- One function per predicate in the lowered partition, plus a~n"),
        format("-- dispatch map wired into WamContext.wcLoweredPredicates by Main.hs.~n"),
        format("module Lowered where~n~n"),
        format("import qualified Data.Map.Strict as Map~n"),
        format("import qualified Data.IntMap.Strict as IM~n"),
        format("import WamTypes~n"),
        format("import Data.Maybe (fromMaybe)~n"),
        format("import WamRuntime~n~n"),
        % Function definitions
        forall(member(lowered(_, _, HsCode), LoweredEntries),
               format("~w~n", [HsCode])),
        % Dispatch map
        format("loweredPredicates :: Map.Map String (WamContext -> WamState -> Maybe WamState)~n"),
        format("loweredPredicates = Map.fromList~n"),
        format("    [ "),
        emit_lowered_entries(LoweredEntries),
        format("    ]~n")
    )).

% Emit the Map.fromList entries, one per line with a leading comma on
% all but the first.
emit_lowered_entries([lowered(PredName, FuncName, _)|Rest]) :-
    format("(\"~w\", ~w)~n", [PredName, FuncName]),
    emit_lowered_entries_rest(Rest).
emit_lowered_entries_rest([]).
emit_lowered_entries_rest([lowered(PredName, FuncName, _)|Rest]) :-
    format("    , (\"~w\", ~w)~n", [PredName, FuncName]),
    emit_lowered_entries_rest(Rest).

%% generate_main_hs(+Predicates, +DetectedKernels, +Options, -Code)
%  Generates Main.hs — a benchmark driver for effective-distance.
%  Reads main.hs.mustache and populates template variables.
%  Options:
%    query_pred(Pred/Arity) — use a WAM-compiled aggregation predicate
%      instead of the default collectSolutions loop. The predicate should
%      accept (Cat, Root, WeightSum) and return the accumulated weight sum.
generate_main_hs(_Predicates, DetectedKernels, Options, Code) :-
    read_kernel_template('main.hs.mustache', Template),
    detected_kernel_keys(DetectedKernels, Keys),
    format_foreign_preds(Keys, ForeignPredsStr),
    generate_query_body(Options, QueryBody),
    generate_merged_code_build(DetectedKernels, Options, MergedCodeBuild),
    render_template(Template,
        [ foreign_preds=ForeignPredsStr
        , query_body=QueryBody
        , merged_code_build=MergedCodeBuild
        ], Code).

%% generate_merged_code_build(+DetectedKernels, +Options, -Code)
%  Emit the merged-code construction block for Main.hs.
%  When an FFI kernel handles fact lookups, the WAM-compiled fact code
%  is never executed — so we skip buildFact2Code to save ~42% of runtime.
%  Options:
%    skip_fact_wam(true|false) — override auto-detection
%  Auto-detection: if any kernel is detected, skip WAM-compilation of
%  all fact predicates (conservative for benchmark template).
generate_merged_code_build(DetectedKernels, Options, Code) :-
    (   option(skip_fact_wam(Skip), Options)
    ->  true
    ;   DetectedKernels \= []
    ->  Skip = true
    ;   Skip = false
    ),
    (   Skip == true
    ->  Code =
'    let mergedCodeRaw = allCode
        mergedLabels = allLabels'
    ;   Code =
'    let baseLen = length allCode
        (cpCode, cpLabels) = buildFact2Code "category_parent" categoryParents (baseLen + 1)
        cpEnd = baseLen + length cpCode
        (acCode, acLabels) = buildFact2Code "article_category" articleCategories (cpEnd + 1)
        acEnd = cpEnd + length acCode
        (rcCode, rcLabels) = buildFact1Code "root_category" roots (acEnd + 1)

        mergedCodeRaw = allCode ++ cpCode ++ acCode ++ rcCode
        mergedLabels = Map.union allLabels
                     $ Map.fromList (cpLabels ++ acLabels ++ rcLabels)'
    ).

generate_query_body(Options, QueryBody) :-
    % Emit a PURE expression so the seed loop can use `parMap rdeepseq`.
    % We use explicit braces and semicolons on the `let` to avoid Haskell's
    % column-alignment layout rules (the template renderer doesn't re-indent
    % continuation lines to match the {{query_body}} insertion column).
    (   member(query_pred(QueryPred), Options)
    ->  % Optimized: call WAM-compiled aggregation predicate per seed.
        format(atom(QueryPred1), '~w', [QueryPred]),
        format(atom(QueryBody),
'let { wsVarId = 1000000 ; s0 = emptyState { wsPC = fromMaybe 1 $ Map.lookup "~w" mergedLabels, wsRegs = IM.fromList [ (1, Atom cat), (2, Atom root), (3, Unbound wsVarId) ], wsCP = 0 } ; !result = case run ctx s0 of { Just s1 -> case IM.lookup wsVarId (wsBindings s1) of { Just v -> case extractDouble (derefVar (wsBindings s1) v) of { Just ws -> ws ; Nothing -> 0.0 } ; Nothing -> 0.0 } ; Nothing -> 0.0 } } in (cat, result)',
            [QueryPred1])
    ;   % Default: collectSolutions loop for category_ancestor/4
        QueryBody =
'let { hopsVarId = 1000000 ; s0 = emptyState { wsPC = fromMaybe 1 $ Map.lookup "category_ancestor/4" mergedLabels, wsRegs = IM.fromList [ (1, Atom cat), (2, Atom root), (3, Unbound hopsVarId), (4, VList [Atom cat]) ], wsCP = 0 } ; !solutions = collectSolutions ctx s0 hopsVarId ; !weightSum = sum [((hops + 1) ** negN) | hops <- solutions] } in (cat, weightSum)'
    ).

%% detected_kernel_keys(+DetectedKernels, -Keys)
%  Extract the string keys from Key-Kernel pairs.
detected_kernel_keys([], []).
detected_kernel_keys([Key-_|Rest], [Key|RestKeys]) :-
    detected_kernel_keys(Rest, RestKeys).

%% format_foreign_preds(+Keys, -HaskellListStr)
%  Format a list of predicate indicators as a Haskell list literal body.
%  E.g., ['category_ancestor/4', 'closure/2'] -> '"category_ancestor/4", "closure/2"'
format_foreign_preds([], '').
format_foreign_preds(Keys, Str) :-
    Keys \= [],
    maplist([Key, Quoted]>>(
        format(atom(Quoted), '"~w"', [Key])
    ), Keys, QuotedKeys),
    atomic_list_concat(QuotedKeys, ', ', Str).

build_predicate_loads([], '    let allCode = []\n    let allLabels = Map.empty').
build_predicate_loads(Predicates, Code) :-
    Predicates \= [],
    % Build code concatenation and label union
    maplist([PredInd, FN]>>(
        (PredInd = _M:P/A -> true ; PredInd = P/A),
        format(atom(FN), '~w_~w', [P, A])
    ), Predicates, FuncNames),
    % Generate Haskell code to merge all predicate code/labels
    maplist([FN, Expr]>>(
        format(string(Expr), '~w_code', [FN])
    ), FuncNames, CodeExprs),
    atomic_list_concat(CodeExprs, ' ++ ', CodeConcat),
    maplist([FN, Expr]>>(
        format(string(Expr), '~w_labels', [FN])
    ), FuncNames, LabelExprs),
    % Union with offset adjustment
    % For simplicity, use Map.unions (labels need PC offset per predicate)
    atomic_list_concat(LabelExprs, ' `Map.union` ', LabelUnion),
    format(string(Code),
'    let allCode = ~w
    let allLabels = ~w', [CodeConcat, LabelUnion]).

%% compile_wam_runtime_to_haskell(+Options, +DetectedKernels, -Code)
compile_wam_runtime_to_haskell(_Options, DetectedKernels, Code) :-
    step_function_haskell(StepCode),
    backtrack_haskell(BacktrackCodeTemplate),
    run_loop_haskell(RunCode),
    % Render kernel-specific Haskell from Mustache templates
    generate_kernel_haskell(DetectedKernels, KernelFunctionsCode, ExecuteForeignCode),
    % Inject into the backtrack_haskell template via {{placeholder}} substitution
    render_template(BacktrackCodeTemplate,
                    [kernel_functions=KernelFunctionsCode,
                     execute_foreign=ExecuteForeignCode],
                    BacktrackCode),
    format(string(Code),
'{-# LANGUAGE BangPatterns #-}
module WamRuntime where

import qualified Data.Map.Strict as Map
import qualified Data.IntMap.Strict as IM
import qualified Data.IntSet as IS
import Data.Array (Array, listArray, (!), bounds)
import qualified Data.Set as Set
import Data.List (isPrefixOf, foldl'')
import Data.Maybe (fromMaybe)
-- Phase 4.2: intra-query parallelism. parMap/rdeepseq spark the
-- alternative clauses of a forkable ParTryMeElse choice point; the
-- WamState NFData instance lives in WamTypes.
import Control.Parallel.Strategies (parMap, rdeepseq)
import Control.DeepSeq (NFData(..), deepseq)
import WamTypes

~w

~w

~w

-- | Dereference an Unbound variable through the binding table.
{-# INLINE derefVar #-}
derefVar :: IM.IntMap Value -> Value -> Value
derefVar bindings (Unbound vid) =
  case IM.lookup vid bindings of
    Just val -> derefVar bindings val
    Nothing  -> Unbound vid
derefVar _ v = v

-- | Evaluate arithmetic expression.
evalArith :: IM.IntMap Value -> Value -> Maybe Double
evalArith _ (Integer n) = Just (fromIntegral n)
evalArith _ (Float f) = Just f
evalArith bindings (Atom s) = case reads s of
  [(n, "")] -> Just n
  _ -> Nothing
evalArith bindings (Str op [a]) = do
  va <- evalArith bindings (derefVar bindings a)
  let bareOp = takeWhile (/= ''/'') op
  case bareOp of
    "-" -> Just (negate va)
    "abs" -> Just (abs va)
    _ -> Nothing
evalArith bindings (Str op [a, b]) = do
  va <- evalArith bindings (derefVar bindings a)
  vb <- evalArith bindings (derefVar bindings b)
  let bareOp = takeWhile (/= ''/'') op
  case bareOp of
    "+" -> Just (va + vb)
    "-" -> Just (va - vb)
    "*" -> Just (va * vb)
    "**" -> Just (va ** vb)
    "^" -> Just (va ** vb)
    "/" -> if vb /= 0 then Just (va / vb) else Nothing
    "//" -> if vb /= 0 then Just (fromIntegral (truncate va `div` truncate vb :: Int)) else Nothing
    "mod" -> if vb /= 0 then Just (fromIntegral (truncate va `mod` truncate vb :: Int)) else Nothing
    _ -> Nothing
evalArith _ _ = Nothing

-- | Get register value. Y-registers (id >= 200) come from the env frame.
{-# INLINE getReg #-}
getReg :: Int -> WamState -> Maybe Value
getReg rid s
  | rid >= 200 = findYReg rid (wsStack s)
  | otherwise = derefVar (wsBindings s) <$> IM.lookup rid (wsRegs s)
  where
    findYReg _ [] = Nothing
    findYReg r (EnvFrame _ yregs : _) = derefVar (wsBindings s) <$> IM.lookup r yregs
    findYReg r (_ : rest) = findYReg r rest

-- | Set register value. Y-registers go to the topmost env frame.
{-# INLINE putReg #-}
putReg :: Int -> Value -> WamState -> WamState
putReg rid val s
  | rid >= 200 = s { wsStack = updateTopEnv rid val (wsStack s) }
  | otherwise = s { wsRegs = IM.insert rid val (wsRegs s) }
  where
    updateTopEnv _ _ [] = []
    updateTopEnv r v (EnvFrame cp yregs : rest) =
      EnvFrame cp (IM.insert r v yregs) : rest
    updateTopEnv r v (x : rest) = x : updateTopEnv r v rest

-- | Dereference a heap reference.
derefHeap :: [Value] -> Value -> Maybe Value
derefHeap heap (Ref addr)
  | addr >= 0 && addr < length heap = Just (heap !! addr)
  | otherwise = Nothing
derefHeap _ (Str fn args) = Just (Str fn args)
derefHeap _ (Unbound n) = Just (Unbound n)
derefHeap _ v = Just v

-- | Add a value to the current structure/list builder.
addToBuilder :: Value -> WamState -> Maybe WamState
addToBuilder val s = case wsBuilder s of
  BuildStruct fn ai arity args ->
    -- Cons to front (O(1)) and reverse only on finalize. Track count via list length
    -- but only when finalizing — args grows from 0 to arity, max arity is small.
    let args'' = val : args
    in if length args'' == arity
       then Just (s { wsPC = wsPC s + 1
                    , wsRegs = IM.insert ai (Str fn (reverse args'')) (wsRegs s)
                    , wsBuilder = NoBuilder
                    })
       else Just (s { wsPC = wsPC s + 1
                    , wsBuilder = BuildStruct fn ai arity args''
                    })
  BuildList ai args ->
    -- BuildList always has exactly 2 args [head, tail]
    let args'' = val : args
    in if length args'' == 2
       then let [tl, hd] = args''   -- reversed because we cons-built
                list = case tl of
                  VList items -> VList (hd : items)
                  Atom "[]"  -> VList [hd]
                  _           -> VList [hd, tl]
            in Just (s { wsPC = wsPC s + 1
                       , wsRegs = IM.insert ai list (wsRegs s)
                       , wsBuilder = NoBuilder
                       })
       else Just (s { wsPC = wsPC s + 1
                    , wsBuilder = BuildList ai args''
                    })
  NoBuilder ->
    -- No builder active, just push to heap (fallback)
    Just (s { wsPC = wsPC s + 1, wsHeap = wsHeap s ++ [val], wsHeapLen = wsHeapLen s + 1 })

-- | Lookup a label in the label map (now in WamContext).
lookupLabel :: String -> WamContext -> Int
lookupLabel label ctx = fromMaybe 0 $ Map.lookup label (wcLabels ctx)

-- | Fetch instruction at PC (1-indexed). Bounds-checked, returns Maybe.
fetchInstr :: Int -> Array Int Instruction -> Maybe Instruction
fetchInstr pc code
  | let (lo, hi) = bounds code in pc < lo || pc > hi = Nothing
  | otherwise = Just (code ! pc)

-- | Unsafe fetch — no bounds check, no Maybe wrapping. Use only when the
-- caller can prove PC is in bounds (the run loop handles PC=0 as halt
-- separately, and a well-formed WAM program never jumps out of bounds).
{-# INLINE unsafeFetchInstr #-}
unsafeFetchInstr :: Int -> Array Int Instruction -> Instruction
unsafeFetchInstr pc code = code ! pc

-- | Resolve Call instructions at project load time:
--   - Foreign predicates (detected kernels) → CallForeign (direct FFI, Nothing = fail)
--   - Known labels → CallResolved (direct PC jump, no dispatch)
--   - Everything else → left as Call (runtime dispatch chain)
resolveCallInstrs :: Map.Map String Int -> [String] -> [Instruction] -> [Instruction]
resolveCallInstrs labels foreignPreds = map resolve
  where
    resolve (Call pred arity)
      | pred `elem` foreignPreds = CallForeign pred arity
      | otherwise = case Map.lookup pred labels of
          Just pc -> CallResolved pc arity
          Nothing -> Call pred arity
    resolve (Execute pred) = case Map.lookup pred labels of
      Just pc -> ExecutePc pc
      Nothing -> Execute pred
    resolve (Jump label) = case Map.lookup label labels of
      Just pc -> JumpPc pc
      Nothing -> Jump label
    resolve (TryMeElse label) = case Map.lookup label labels of
      Just pc -> TryMeElsePc pc
      Nothing -> TryMeElse label
    resolve (RetryMeElse label) = case Map.lookup label labels of
      Just pc -> RetryMeElsePc pc
      Nothing -> RetryMeElse label
    -- Phase 4.1 Par* variants resolve the same way as their sequential
    -- counterparts. The instruction carries forkability intent; label
    -- resolution is orthogonal.
    resolve (ParTryMeElse label) = case Map.lookup label labels of
      Just pc -> ParTryMeElsePc pc
      Nothing -> ParTryMeElse label
    resolve (ParRetryMeElse label) = case Map.lookup label labels of
      Just pc -> ParRetryMeElsePc pc
      Nothing -> ParRetryMeElse label
    resolve (SwitchOnConstant table) =
      let extractKey (Atom s) = s
          extractKey (Integer n) = show n
          extractKey v = show v
      in SwitchOnConstantPc (Map.fromList [(extractKey v, pc) | (v, label) <- Map.toList table,
                                                                 Just pc <- [Map.lookup label labels]])
    resolve i = i
', [StepCode, BacktrackCode, RunCode]).

%% generate_wam_types_hs(-Code)
generate_wam_types_hs(Code) :-
    Code = 'module WamTypes where

import qualified Data.Map.Strict as Map
import qualified Data.IntMap.Strict as IM
import Data.Array (Array, listArray, (!), bounds)
-- Phase 4.2: NFData is needed for parMap rdeepseq to fully evaluate
-- each forked branch''s contribution before the merge step.
import Control.DeepSeq (NFData(..))

data Value = Atom String
           | Integer !Int
           | Float !Double
           | VList [Value]
           | Str String [Value]
           | Unbound !Int   -- variable ID (interned via wsVarCounter)
           | Ref Int
           deriving (Eq, Ord, Show)

data EnvFrame = EnvFrame {-# UNPACK #-} !Int !(IM.IntMap Value)
              deriving (Show)

data TrailEntry = TrailEntry {-# UNPACK #-} !Int !(Maybe Value)
                deriving (Show)

data ChoicePoint = ChoicePoint
  { cpNextPC   :: {-# UNPACK #-} !Int
  , cpRegs     :: !(IM.IntMap Value)
  , cpStack    :: ![EnvFrame]
  , cpCP       :: {-# UNPACK #-} !Int
  , cpTrailLen :: {-# UNPACK #-} !Int
  , cpHeapLen  :: {-# UNPACK #-} !Int
  , cpBindings :: !(IM.IntMap Value)
  , cpCutBar   :: {-# UNPACK #-} !Int
  , cpAggFrame :: !(Maybe AggFrame)
  , cpBuiltin  :: !(Maybe BuiltinState)
  } deriving (Show)

-- | Builtin state for choice points that need custom retry logic.
data BuiltinState
  = FactRetry !Int ![String] !Int  -- variable ID, remaining values, returnPC
  | HopsRetry !Int ![Int] !Int     -- variable ID, remaining Hops values, returnPC
    -- Multi-output FFI kernel retry. Each remaining tuple is already
    -- wrapped as a list of Values (pre-interned / wrapped at call site).
    -- outRegs and outVars are parallel lists (same length as each tuple).
    -- outVars contains -1 for originally-bound outputs (no binding update).
  | FFIStreamRetry ![Int] ![Int] ![[Value]] !Int  -- outRegs, outVars, remaining tuples, returnPC
  deriving (Show)

-- | Aggregate frame for begin_aggregate/end_aggregate.
data AggFrame = AggFrame
  { afType      :: !String         -- "sum", "count", "collect", etc.
  , afValueReg  :: !Int            -- register ID holding value per solution
  , afResultReg :: !Int            -- register ID for final result
  , afReturnPC  :: !Int            -- PC after end_aggregate
  , afMergeStrategy :: !MergeStrategy  -- Phase 4.2: derived from afType;
                                       -- carried on the frame so inner
                                       -- ParTryMeElse choice points can
                                       -- decide whether to fork without
                                       -- re-parsing the type string.
  } deriving (Show)

-- | Phase 4.2: how to combine per-branch aggregate values when forking.
-- Commutative-and-associative strategies (sum/count) go in Phase 4.2;
-- findall/bag/set arrive in Phase 4.3; race/negation in 4.4. Unknown
-- aggregate types yield MergeSequential, which disables forking.
data MergeStrategy
  = MergeSumInt
  | MergeSumDouble
  | MergeCount
  | MergeFindall       -- Phase 4.3
  | MergeBag           -- Phase 4.3
  | MergeSet           -- Phase 4.3
  | MergeRace          -- Phase 4.4
  | MergeNegation      -- Phase 4.4
  | MergeSequential    -- Default fallback; fork disabled
  deriving (Show, Eq)

-- | Phase 4.2: fork context threaded from BeginAggregate through to the
-- inner ParTryMeElse choice point that does the actual fork.
data ForkContext = ForkContext
  { fcMergeStrategy :: !MergeStrategy
  , fcWorkEstimate  :: !(Maybe Double)  -- microseconds, Phase 4.5
  } deriving (Show)

-- | Parse a MergeStrategy from the aggregate type string (as stored in
-- AggFrame.afType). Returns MergeSequential for anything we don''t
-- recognize, which causes the ParTryMeElse fork to fall back to the
-- sequential TryMeElse handler.
inferMergeStrategy :: String -> MergeStrategy
inferMergeStrategy "sum"   = MergeSumDouble
inferMergeStrategy "count" = MergeCount
inferMergeStrategy "bag"   = MergeBag
inferMergeStrategy "set"   = MergeSet
inferMergeStrategy "findall" = MergeFindall
inferMergeStrategy _       = MergeSequential

-- | Phase 4.2: NFData instances so parMap rdeepseq can spark forked
-- branches. We need this only for the types that end up in a parMap
-- result — for us that''s `[Value]` (each branch''s contributed
-- aggregate values). Everything is first-order / strict already;
-- these definitions just walk the structure to force evaluation.
instance NFData Value where
  rnf (Atom s)         = rnf s
  rnf (Integer n)      = rnf n
  rnf (Float f)        = rnf f
  rnf (VList xs)       = rnf xs
  rnf (Str name args)  = rnf name `seq` rnf args
  rnf (Unbound n)      = rnf n
  rnf (Ref n)          = rnf n

-- | Builder for PutStructure/PutList + SetValue/SetConstant sequences.
data Builder = BuildStruct !String !Int !Int ![Value]  -- functor, target reg ID, arity, collected args
             | BuildList !Int ![Value]                  -- target reg ID, collected [head, tail]
             | NoBuilder
             deriving (Show)

-- | Read-only context. Threaded through the run loop / step function as
-- a separate argument so it doesn''t pay the per-step record-update cost
-- on the mutable WamState. Built once at startup, never modified.
data WamContext = WamContext
  { wcCode          :: !(Array Int Instruction)
  , wcLabels        :: !(Map.Map String Int)
  , wcForeignFacts  :: !(Map.Map String (Map.Map String [String]))
  , wcForeignConfig :: !(Map.Map String Int)
  , wcLoweredPredicates :: !(Map.Map String (WamContext -> WamState -> Maybe WamState))
  -- | Atom interning table for the FFI boundary. Populated at startup
  -- with String -> Int IDs. Used by executeForeign to convert WAM-side
  -- String atoms to Int keys before calling native kernels.
  , wcAtomIntern    :: !(Map.Map String Int)
  -- | Reverse intern table (Int -> String) for de-interning kernel
  -- results that return Ints but need to be wrapped as Atom String.
  , wcAtomDeintern  :: !(IM.IntMap String)
  -- | Fact indexes keyed by interned Int atoms. Used exclusively by the
  -- FFI kernel path. Populated per-kernel from edge_pred config.
  -- Separate from wcForeignFacts so callIndexedFact2 (WAM path) is
  -- unaffected.
  , wcFfiFacts      :: !(Map.Map String (IM.IntMap [Int]))
  -- | Weighted fact indexes for kernels that need (target, weight)
  -- pairs per edge (e.g., weighted_shortest_path3 / Dijkstra). Used
  -- exclusively by the FFI kernel path. Populated from 3-column fact
  -- sources — not wired into the default Main.hs template yet, so
  -- standalone benchmarks build this directly.
  , wcFfiWeightedFacts :: !(Map.Map String (IM.IntMap [(Int, Double)]))
  }
-- Note: no `deriving (Show)` because wcLoweredPredicates is function-valued
-- and functions have no Show instance. Add a manual instance if needed.

-- | Mutable state. Updated on every WAM step. Held separate from WamContext
-- so each step transition only allocates a record with the fields that
-- actually change.
data WamState = WamState
  { wsPC       :: {-# UNPACK #-} !Int
  , wsRegs     :: !(IM.IntMap Value)
  , wsStack    :: ![EnvFrame]
  , wsHeap     :: ![Value]
  , wsHeapLen  :: {-# UNPACK #-} !Int
  , wsTrail    :: ![TrailEntry]
  , wsTrailLen :: {-# UNPACK #-} !Int
  , wsCP       :: {-# UNPACK #-} !Int
  , wsCPs      :: ![ChoicePoint]
  , wsCPsLen   :: {-# UNPACK #-} !Int
  , wsBindings :: !(IM.IntMap Value)
  , wsCutBar   :: {-# UNPACK #-} !Int
  , wsBuilder  :: !Builder
  , wsVarCounter :: {-# UNPACK #-} !Int
  , wsAggAccum :: ![Value]
  } deriving (Show)

-- | Instruction type for the WAM.
-- | Register IDs are pre-interned at compile time as Ints to avoid string
-- hashing on register access. Encoding:
--   A1-A99: 1-99
--   X1-X99: 101-199
--   Y1-Y99: 201-299
type RegId = Int

data Instruction
  = GetConstant Value !RegId
  | GetVariable !RegId !RegId
  | GetValue !RegId !RegId
  | PutConstant Value !RegId
  | PutVariable !RegId !RegId
  | PutValue !RegId !RegId
  | PutStructure String !RegId !Int   -- functor, target reg, arity (pre-parsed)
  | PutList !RegId
  | SetValue !RegId
  | SetConstant Value
  | Allocate
  | Deallocate
  | Call String !Int                  -- pre-resolution form (string-keyed)
  | CallResolved !Int !Int            -- post-resolution: target PC + arity
  | CallForeign String !Int           -- compile-time resolved foreign pred (Nothing = fail)
  | Execute String
  | ExecutePc !Int                      -- post-resolution: direct PC jump (tail call)
  | Jump String                         -- unconditional jump to label
  | JumpPc !Int                         -- post-resolution: direct PC jump
  | CutIte                              -- soft cut: pop one CP (if-then-else)
  | Proceed
  | TryMeElsePc !Int                   -- post-resolution: direct PC for else branch
  | RetryMeElsePc !Int                 -- post-resolution: direct PC for next branch
  | SwitchOnConstantPc !(Map.Map String Int) -- post-resolution: atom string -> PC
  | BuiltinCall String !Int
  | TryMeElse String
  | RetryMeElse String
  | TrustMe
  -- Phase 4.1: parallel-forkable variants. Emitted by the compiler
  -- when the predicate has a pure purity certificate. At Phase 4.1
  -- they dispatch to the same sequential handlers as the non-Par
  -- variants — the instructions carry intent, not behavior. Phase 4.2
  -- will add the runtime fork. See
  -- docs/design/WAM_HASKELL_INTRA_QUERY_SPEC.md §2.
  | ParTryMeElse String
  | ParRetryMeElse String
  | ParTrustMe
  | ParTryMeElsePc !Int
  | ParRetryMeElsePc !Int
  | SwitchOnConstant (Map.Map Value String)   -- pre-built Map for O(log n) dispatch
  | BeginAggregate String !RegId !RegId   -- type, valueReg, resultReg
  | EndAggregate !RegId                   -- valueReg
  deriving (Show, Eq)

-- | Build the read-only context from compiled code and labels. Called
-- once at project startup. The context is then threaded into runLoop
-- and step as a separate argument.
mkContext :: [Instruction] -> Map.Map String Int -> WamContext
mkContext codeList labels =
  let n = length codeList
      code = listArray (1, n) codeList
  in WamContext
    { wcCode          = code
    , wcLabels        = labels
    , wcForeignFacts  = Map.empty
    , wcForeignConfig = Map.empty
    , wcLoweredPredicates = Map.empty
    , wcAtomIntern    = Map.empty
    , wcAtomDeintern  = IM.empty
    , wcFfiFacts      = Map.empty
    , wcFfiWeightedFacts = Map.empty
    }

-- | Create initial empty mutable state. The cold fields (code, labels,
-- foreign facts/config) live in WamContext now.
emptyState :: WamState
emptyState = WamState
  { wsPC       = 1
  , wsRegs     = IM.empty
  , wsStack    = []
  , wsHeap     = []
  , wsHeapLen  = 0
  , wsTrail    = []
  , wsTrailLen = 0
  , wsCP       = 0
  , wsCPs      = []
  , wsCPsLen   = 0
  , wsBindings = IM.empty
  , wsCutBar   = 0
  , wsBuilder  = NoBuilder
  , wsVarCounter = 0
  , wsAggAccum = []
  }
'.

%% generate_cabal_file(+Name, +UseHM, +Options, -Code)
%  Options: profiling(true) adds -prof -fprof-auto -rtsopts for GHC profiling.
generate_cabal_file(Name, UseHM, Options, Code) :-
    % deepseq + parallel for seed-level parMap rdeepseq
    (   UseHM == true
    ->  Deps = "base >= 4.12, containers >= 0.6, array, time >= 1.8, unordered-containers >= 0.2, hashable >= 1.2, deepseq >= 1.4, parallel >= 3.2"
    ;   Deps = "base >= 4.12, containers >= 0.6, array, time >= 1.8, deepseq >= 1.4, parallel >= 3.2"
    ),
    % -threaded enables multi-core runtime (+RTS -N to use cores).
    % -rtsopts is needed so +RTS flags are accepted at runtime.
    (   option(profiling(true), Options)
    ->  GhcOpts = "-O2 -threaded -rtsopts -prof -fprof-auto"
    ;   GhcOpts = "-O2 -threaded -rtsopts"
    ),
    format(string(Code),
'cabal-version: 2.4
name:          ~w
version:       0.1.0.0
build-type:    Simple

executable ~w
  main-is:          Main.hs
  hs-source-dirs:   src
  other-modules:    WamTypes, WamRuntime, Predicates, Lowered
  build-depends:    ~w
  default-language: Haskell2010
  ghc-options:      ~w
', [Name, Name, Deps, GhcOpts]).

%% compile_predicates_to_haskell(+Predicates, +Options, -Code)
%  Compiles all predicates into a single merged code array and label map,
%  with proper PC offsets for each predicate.
compile_predicates_to_haskell(Predicates, Options, Code) :-
    compile_predicates_merged(Predicates, 1, Options, AllInstrs, AllLabels),
    atomic_list_concat(AllInstrs, '\n    , ', InstrCode),
    atomic_list_concat(AllLabels, '\n    , ', LabelCode),
    format(string(Code),
'module Predicates where

import qualified Data.Map.Strict as Map
import WamTypes

-- | Merged WAM code for all predicates.
allCode :: [Instruction]
allCode =
    [ ~w
    ]

-- | Merged label map for all predicates.
allLabels :: Map.Map String Int
allLabels = Map.fromList
    [ ~w
    ]
', [InstrCode, LabelCode]).

compile_predicates_merged([], _, _, [], []).
compile_predicates_merged([PredIndicator|Rest], StartPC, Options, AllInstrs, AllLabels) :-
    (   PredIndicator = _Module:Pred/Arity -> true
    ;   PredIndicator = Pred/Arity
    ),
    wam_target:compile_predicate_to_wam(PredIndicator, [], WamCode),
    format(user_error, '  ~w/~w: compiled to WAM (PC=~w)~n', [Pred, Arity, StartPC]),
    atom_string(WamCode, WamStr),
    split_string(WamStr, "\n", "", Lines),
    wam_lines_to_haskell(Lines, StartPC, InstrExprs0, LabelExprs, NextPC),
    % Phase 4.1: if the purity certificate says this predicate is
    % safe to parallelize, rewrite its choice-point instructions
    % (TryMeElse / RetryMeElse / TrustMe) to the Par* variants.
    maybe_parallelize_instrs(PredIndicator, Options, InstrExprs0, InstrExprs),
    compile_predicates_merged(Rest, NextPC, Options, RestInstrs, RestLabels),
    append(InstrExprs, RestInstrs, AllInstrs),
    append(LabelExprs, RestLabels, AllLabels).

%% maybe_parallelize_instrs(+PredIndicator, +Options, +InstrExprs0, -InstrExprs)
%  Rewrite TryMeElse → ParTryMeElse etc. when the predicate certifies
%  pure with confidence >= 0.85 and intra_query_parallel/1 hasn't
%  been disabled. Otherwise leaves instructions alone.
%
%  Confidence threshold: 0.85 — catches user-declared (1.0),
%  kernel-registry-certified (1.0), and blacklist-clean (0.9) as
%  forkable. Inferred / low-confidence verdicts fall back to sequential.
maybe_parallelize_instrs(PredIndicator, Options, InstrExprs0, InstrExprs) :-
    (   option(intra_query_parallel(false), Options)
    ->  InstrExprs = InstrExprs0
    ;   purity_certificate:analyze_predicate_purity(
            PredIndicator,
            purity_cert(pure, _, Conf, _)),
        Conf >= 0.85
    ->  maplist(parallelize_choice_instr, InstrExprs0, InstrExprs),
        ( InstrExprs \== InstrExprs0
        -> ( PredIndicator = _:Pred/Arity -> true ; PredIndicator = Pred/Arity ),
           format(user_error,
                  '    [Par] ~w/~w: emitting Par* for forkable choice points~n',
                  [Pred, Arity])
        ; true
        )
    ;   InstrExprs = InstrExprs0
    ).

%% parallelize_choice_instr(+InstrStr0, -InstrStr)
%  String-level rewrite of choice-point instructions:
%    TryMeElse   "…"  →  ParTryMeElse   "…"
%    RetryMeElse "…"  →  ParRetryMeElse "…"
%    TrustMe          →  ParTrustMe
%
%  Accepts either atom or string input; returns the same type so
%  callers and tests can mix representations without surprises.
parallelize_choice_instr(S0, S) :-
    atom_string(A0, S0),
    ( sub_atom(A0, 0, _, _, 'TryMeElse ')
    -> atom_concat('Par', A0, A)
    ; sub_atom(A0, 0, _, _, 'RetryMeElse ')
    -> atom_concat('Par', A0, A)
    ; A0 == 'TrustMe'
    -> A = 'ParTrustMe'
    ; A = A0
    ),
    ( atom(S0) -> S = A ; atom_string(A, S) ).

compile_single_predicate_to_haskell(PredIndicator, _Options, Code) :-
    (   PredIndicator = _Module:Pred/Arity -> true
    ;   PredIndicator = Pred/Arity
    ),
    wam_target:compile_predicate_to_wam(PredIndicator, [], WamCode),
    format(user_error, '  ~w/~w: compiled to WAM~n', [Pred, Arity]),
    % Parse WAM text into Haskell instruction list and label map
    atom_string(WamCode, WamStr),
    split_string(WamStr, "\n", "", Lines),
    wam_lines_to_haskell(Lines, 1, InstrExprs, LabelExprs),
    atomic_list_concat(InstrExprs, '\n    , ', InstrCode),
    atomic_list_concat(LabelExprs, '\n    , ', LabelCode),
    format(atom(FuncName), '~w_~w', [Pred, Arity]),
    format(string(Code),
'-- WAM-compiled predicate: ~w/~w
~w_code :: [Instruction]
~w_code =
    [ ~w
    ]

~w_labels :: Map.Map String Int
~w_labels = Map.fromList
    [ ~w
    ]', [Pred, Arity, FuncName, FuncName, InstrCode, FuncName, FuncName, LabelCode]).

compile_wam_predicate_to_haskell(PredIndicator, WamCode, _Options, Code) :-
    (   PredIndicator = _Module:Pred/Arity -> true
    ;   PredIndicator = Pred/Arity
    ),
    (   string(WamCode)
    ->  WamStr = WamCode
    ;   atom_string(WamCode, WamStr)
    ),
    split_string(WamStr, "\n", "", Lines),
    wam_lines_to_haskell(Lines, 1, InstrExprs, LabelExprs),
    atomic_list_concat(InstrExprs, '\n    , ', InstrCode),
    atomic_list_concat(LabelExprs, '\n    , ', LabelCode),
    format(atom(FuncName), '~w_~w', [Pred, Arity]),
    format(string(Code),
'-- WAM-compiled predicate: ~w/~w
~w_code :: [Instruction]
~w_code =
    [ ~w
    ]

~w_labels :: Map.Map String Int
~w_labels = Map.fromList
    [ ~w
    ]', [Pred, Arity, FuncName, FuncName, InstrCode, FuncName, FuncName, LabelCode]).

%% wam_lines_to_haskell(+Lines, +PC, -InstrExprs, -LabelExprs, -NextPC)
%  Parses WAM assembly lines into Haskell Instruction constructor expressions
%  and label (String, Int) pairs. Returns NextPC for merging multiple predicates.
wam_lines_to_haskell([], PC, [], [], PC).
wam_lines_to_haskell([Line|Rest], PC, Instrs, Labels, FinalPC) :-
    split_string(Line, " \t,", " \t,", Parts),
    delete(Parts, "", CleanParts),
    (   CleanParts == []
    ->  wam_lines_to_haskell(Rest, PC, Instrs, Labels, FinalPC)
    ;   CleanParts = [First|_],
        (   sub_string(First, _, 1, 0, ":")
        ->  sub_string(First, 0, _, 1, LabelName),
            format(string(LabelExpr), '("~w", ~w)', [LabelName, PC]),
            Labels = [LabelExpr|RestLabels],
            wam_lines_to_haskell(Rest, PC, Instrs, RestLabels, FinalPC)
        ;   wam_instr_to_haskell(CleanParts, HsExpr),
            NPC is PC + 1,
            Instrs = [HsExpr|RestInstrs],
            wam_lines_to_haskell(Rest, NPC, RestInstrs, Labels, FinalPC)
        )
    ).

%% reg_name_to_int(+RegName, -Int)
%  Encode register name string to integer ID for IntMap-based register storage.
%  A1-A99 -> 1-99, X1-X99 -> 101-199, Y1-Y99 -> 201-299.
reg_name_to_int(Reg, Int) :-
    atom_string(RegA, Reg),
    sub_atom(RegA, 0, 1, _, Bank),
    sub_atom(RegA, 1, _, 0, NumA),
    atom_number(NumA, Num),
    (   Bank == 'A' -> Int = Num
    ;   Bank == 'X' -> Int is Num + 100
    ;   Bank == 'Y' -> Int is Num + 200
    ;   Int = 0
    ).

%% wam_instr_to_haskell(+Parts, -HaskellExpr)
%  Converts parsed WAM instruction parts to a Haskell Instruction constructor.
wam_instr_to_haskell(["get_constant", C, Ai], Hs) :-
    clean_comma(C, CC), clean_comma(Ai, CAi),
    wam_value_to_haskell(CC, HsVal),
    reg_name_to_int(CAi, AiI),
    format(string(Hs), 'GetConstant (~w) ~w', [HsVal, AiI]).
wam_instr_to_haskell(["get_variable", Xn, Ai], Hs) :-
    clean_comma(Xn, CXn), clean_comma(Ai, CAi),
    reg_name_to_int(CXn, XnI), reg_name_to_int(CAi, AiI),
    format(string(Hs), 'GetVariable ~w ~w', [XnI, AiI]).
wam_instr_to_haskell(["get_value", Xn, Ai], Hs) :-
    clean_comma(Xn, CXn), clean_comma(Ai, CAi),
    reg_name_to_int(CXn, XnI), reg_name_to_int(CAi, AiI),
    format(string(Hs), 'GetValue ~w ~w', [XnI, AiI]).
wam_instr_to_haskell(["put_constant", C, Ai], Hs) :-
    clean_comma(C, CC), clean_comma(Ai, CAi),
    wam_value_to_haskell(CC, HsVal),
    reg_name_to_int(CAi, AiI),
    format(string(Hs), 'PutConstant (~w) ~w', [HsVal, AiI]).
wam_instr_to_haskell(["put_variable", Xn, Ai], Hs) :-
    clean_comma(Xn, CXn), clean_comma(Ai, CAi),
    reg_name_to_int(CXn, XnI), reg_name_to_int(CAi, AiI),
    format(string(Hs), 'PutVariable ~w ~w', [XnI, AiI]).
wam_instr_to_haskell(["put_value", Xn, Ai], Hs) :-
    clean_comma(Xn, CXn), clean_comma(Ai, CAi),
    reg_name_to_int(CXn, XnI), reg_name_to_int(CAi, AiI),
    format(string(Hs), 'PutValue ~w ~w', [XnI, AiI]).
wam_instr_to_haskell(["put_structure", FN, Ai], Hs) :-
    clean_comma(FN, CFN), clean_comma(Ai, CAi),
    parse_functor_arity(CFN, Arity),
    reg_name_to_int(CAi, AiI),
    format(string(Hs), 'PutStructure "~w" ~w ~w', [CFN, AiI, Arity]).
wam_instr_to_haskell(["put_list", Ai], Hs) :-
    clean_comma(Ai, CAi), reg_name_to_int(CAi, AiI),
    format(string(Hs), 'PutList ~w', [AiI]).
wam_instr_to_haskell(["set_value", Xn], Hs) :-
    clean_comma(Xn, CXn), reg_name_to_int(CXn, XnI),
    format(string(Hs), 'SetValue ~w', [XnI]).
wam_instr_to_haskell(["set_constant", C], Hs) :-
    wam_value_to_haskell(C, HsVal),
    format(string(Hs), 'SetConstant (~w)', [HsVal]).
wam_instr_to_haskell(["allocate"], "Allocate").
wam_instr_to_haskell(["deallocate"], "Deallocate").
wam_instr_to_haskell(["call", P, N], Hs) :-
    clean_comma(P, CP), clean_comma(N, CN),
    (   number_string(Num, CN) -> true ; Num = 0 ),
    format(string(Hs), 'Call "~w" ~w', [CP, Num]).
wam_instr_to_haskell(["execute", P], Hs) :-
    format(string(Hs), 'Execute "~w"', [P]).
wam_instr_to_haskell(["proceed"], "Proceed").
wam_instr_to_haskell(["jump", Label], Hs) :-
    format(string(Hs), 'Jump "~w"', [Label]).
wam_instr_to_haskell(["cut_ite"], "CutIte").
wam_instr_to_haskell(["builtin_call", Op, N], Hs) :-
    clean_comma(Op, COp), clean_comma(N, CN),
    (   number_string(Num, CN) -> true ; Num = 0 ),
    escape_haskell_string(COp, ECOp),
    format(string(Hs), 'BuiltinCall "~w" ~w', [ECOp, Num]).
wam_instr_to_haskell(["try_me_else", Label], Hs) :-
    format(string(Hs), 'TryMeElse "~w"', [Label]).
wam_instr_to_haskell(["trust_me"], "TrustMe").
wam_instr_to_haskell(["retry_me_else", Label], Hs) :-
    format(string(Hs), 'RetryMeElse "~w"', [Label]).
wam_instr_to_haskell(["set_variable", Xn], Hs) :-
    format(string(Hs), 'SetVariable "~w"', [Xn]).
%% switch_on_constant key1:label1, key2:label2, ...
wam_instr_to_haskell(["switch_on_constant"|Entries], Hs) :-
    parse_switch_entries(Entries, HsPairs),
    atomic_list_concat(HsPairs, ', ', PairsStr),
    format(string(Hs), 'SwitchOnConstant (Map.fromList [~w])', [PairsStr]).
wam_instr_to_haskell(["switch_on_constant_a2"|Entries], Hs) :-
    parse_switch_entries(Entries, HsPairs),
    atomic_list_concat(HsPairs, ', ', PairsStr),
    format(string(Hs), 'SwitchOnConstant (Map.fromList [~w])', [PairsStr]).
wam_instr_to_haskell(["begin_aggregate", Type, ValReg, ResReg], Hs) :-
    clean_comma(Type, CT), clean_comma(ValReg, CV), clean_comma(ResReg, CR),
    reg_name_to_int(CV, VI), reg_name_to_int(CR, RI),
    format(string(Hs), 'BeginAggregate "~w" ~w ~w', [CT, VI, RI]).
wam_instr_to_haskell(["end_aggregate", ValReg], Hs) :-
    clean_comma(ValReg, CV),
    reg_name_to_int(CV, VI),
    format(string(Hs), 'EndAggregate ~w', [VI]).
% Fallback for unknown instructions
wam_instr_to_haskell(Parts, Hs) :-
    atomic_list_concat(Parts, ' ', Joined),
    format(string(Hs), '-- UNKNOWN: ~w\n    Proceed', [Joined]).

%% parse_functor_arity(+FunctorString, -Arity)
%  Extract the arity from "name/N" format. Defaults to 0 if no slash.
parse_functor_arity(FN, Arity) :-
    atom_string(FNA, FN),
    (   sub_atom(FNA, Before, 1, _, '/'),
        After is Before + 1,
        sub_atom(FNA, After, _, 0, ArityStr),
        atom_number(ArityStr, Arity)
    ->  true
    ;   Arity = 0
    ).

%% wam_value_to_haskell(+WamVal, -HaskellExpr)
%  Converts a WAM constant to a Haskell Value constructor.
wam_value_to_haskell(Val, Hs) :-
    (   number_string(N, Val), integer(N)
    ->  % Wrap negative integers in parens so Haskell parses correctly:
        % Integer (-5) not Integer -5
        (   N < 0
        ->  format(string(Hs), 'Integer (~w)', [N])
        ;   format(string(Hs), 'Integer ~w', [N])
        )
    ;   number_string(F, Val), float(F)
    ->  (   F < 0
        ->  format(string(Hs), 'Float (~w)', [F])
        ;   format(string(Hs), 'Float ~w', [F])
        )
    ;   format(string(Hs), 'Atom "~w"', [Val])
    ).

%% clean_comma(+Str, -Clean) — strip trailing comma
clean_comma(Str, Clean) :-
    (   sub_string(Str, _, 1, 0, ",")
    ->  sub_string(Str, 0, _, 1, Clean)
    ;   Clean = Str
    ).

%% parse_switch_entries(+Entries, -HaskellPairs)
%  Parse "key:label" pairs from switch_on_constant instruction.
parse_switch_entries([], []).
parse_switch_entries([Entry|Rest], [HsPair|HsRest]) :-
    clean_comma(Entry, CEntry),
    (   sub_atom(CEntry, Before, 1, _, ':')
    ->  sub_atom(CEntry, 0, Before, _, Key),
        After is Before + 1,
        sub_atom(CEntry, After, _, 0, Label),
        wam_value_to_haskell(Key, HsKey),
        format(string(HsPair), '(~w, "~w")', [HsKey, Label])
    ;   format(string(HsPair), '(Atom "~w", "default")', [CEntry])
    ),
    parse_switch_entries(Rest, HsRest).

%% escape_haskell_string(+In, -Out) — escape backslashes for Haskell string literals
escape_haskell_string(In, Out) :-
    atom_string(In, S),
    split_string(S, "\\", "", Parts),
    atomic_list_concat(Parts, "\\\\", Out).

%% write_hs_file(+Path, +Content)
write_hs_file(Path, Content) :-
    setup_call_cleanup(
        open(Path, write, Stream),
        format(Stream, "~w", [Content]),
        close(Stream)
    ).
