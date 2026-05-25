:- initialization(main, main).

:- use_module('../../src/unifyweaver/targets/wam_rust_target').
:- use_module('../../src/unifyweaver/targets/prolog_target').

%% generate_wam_rust_matrix_benchmark.pl
%%
%% Generates a result-producing effective-distance Rust benchmark through the
%% generic WAM-Rust target. This mirrors the Haskell matrix modes:
%%
%%   interpreter + kernels_off => pure WAM interpreter
%%   interpreter + kernels_on  => WAM interpreter with FFI kernels
%%   functions   + kernels_off => lowered Rust helpers with WAM fallback
%%   functions   + kernels_on  => lowered Rust helpers with FFI kernels

benchmark_workload_path(Path) :-
    source_file(benchmark_workload_path(_), ThisFile),
    file_directory_name(ThisFile, Here),
    directory_file_path(Here, 'effective_distance.pl', Path).

main :-
    current_prolog_flag(argv, Argv),
    (   Argv = [_FactsPath, OutputDir, VariantAtom, EmitModeAtom, KernelModeAtom, LmdbModeAtom, LmdbCrateAtom, LmdbMaterialisationAtom]
    ->  true
    ;   Argv = [_FactsPath, OutputDir, VariantAtom, EmitModeAtom, KernelModeAtom, LmdbModeAtom, LmdbCrateAtom]
    ->  LmdbMaterialisationAtom = eager
    ;   Argv = [_FactsPath, OutputDir, VariantAtom, EmitModeAtom, KernelModeAtom, LmdbModeAtom]
    ->  LmdbCrateAtom = auto, LmdbMaterialisationAtom = eager
    ;   Argv = [_FactsPath, OutputDir, VariantAtom, EmitModeAtom, KernelModeAtom]
    ->  LmdbModeAtom = none, LmdbCrateAtom = auto, LmdbMaterialisationAtom = eager
    ;   format(user_error,
            'Usage: ... -- <facts.pl> <output-dir> <seeded|accumulated> <interpreter|functions> <kernels_on|kernels_off> [<none|cursor>] [<lmdb_zero|heed|auto>] [<eager|lazy|cached|auto>]~n',
            []),
        halt(1)
    ),
    generate(VariantAtom, EmitModeAtom, KernelModeAtom, LmdbModeAtom, LmdbCrateAtom, LmdbMaterialisationAtom, OutputDir),
    halt(0).

main :-
    format(user_error, 'Error: generation failed~n', []),
    halt(1).

generate(VariantAtom, EmitModeAtom, KernelModeAtom, LmdbModeAtom, LmdbCrateAtom, LmdbMaterialisationAtom, OutputDir) :-
    benchmark_workload_path(WorkloadPath),
    load_files(WorkloadPath, [silent(true)]),
    retractall(user:mode(category_ancestor(_, _, _, _))),
    assertz(user:mode(category_ancestor(-, +, -, +))),
    parse_variant(VariantAtom, OptimizationOptions),
    parse_emit_mode(EmitModeAtom, EmitMode),
    parse_kernel_mode(KernelModeAtom, KernelOptions),
    parse_lmdb_mode(LmdbModeAtom, LmdbOptions),
    parse_lmdb_crate(LmdbCrateAtom, LmdbCrateOptions),
    parse_lmdb_materialisation(LmdbMaterialisationAtom, LmdbMaterialisation, LmdbMaterialisationOptions),
    validate_lmdb_materialisation_combo(LmdbMaterialisation, LmdbModeAtom),
    BasePreds = [dimension_n/1, max_depth/1, category_ancestor/4],
    prolog_target:generate_prolog_script(BasePreds, OptimizationOptions, ScriptCode),
    tmp_file_stream(text, TmpPath, TmpStream),
    write(TmpStream, ScriptCode),
    close(TmpStream),
    load_files(TmpPath, [silent(true)]),
    delete_file(TmpPath),
    collect_wam_predicates(VariantAtom, Predicates),
    append([[module_name(wam_rust_matrix_bench), wam_fallback(true), emit_mode(EmitMode), parallel(true)],
            KernelOptions, LmdbOptions, LmdbCrateOptions, LmdbMaterialisationOptions], Options),
    write_wam_rust_project(Predicates, Options, OutputDir),
    write_matrix_main(OutputDir, EmitModeAtom, KernelModeAtom, LmdbModeAtom, LmdbMaterialisation),
    format(user_error,
           '[WAM-Rust-Matrix] variant=~w emit_mode=~w kernels=~w lmdb=~w lmdb_crate=~w materialisation=~w output=~w~n',
           [VariantAtom, EmitMode, KernelModeAtom, LmdbModeAtom, LmdbCrateAtom, LmdbMaterialisation, OutputDir]).

parse_lmdb_mode(none, []).
parse_lmdb_mode(cursor, [lmdb_mode(cursor)]).

parse_lmdb_crate(auto, [lmdb_crate(auto)]).
parse_lmdb_crate(lmdb_zero, [lmdb_crate(lmdb_zero)]).
parse_lmdb_crate(heed, [lmdb_crate(heed)]).

% Phase R7: lmdb_materialisation option (eager | lazy | cached | auto).
% - eager: current behaviour — build runtime_category_parents Vec at startup.
% - lazy:  parent edges read on-demand via LookupSource trait + LMDB cursor.
% - cached: not yet implemented (R8); errors here.
% - auto: R7 returns eager; R8 will wire the cost-model resolver.
parse_lmdb_materialisation(eager, eager, [lmdb_materialisation(eager)]).
parse_lmdb_materialisation(auto,  eager, [lmdb_materialisation(eager)]).
parse_lmdb_materialisation(lazy,  lazy,  [lmdb_materialisation(lazy)]).
parse_lmdb_materialisation(cached, _, _) :-
    format(user_error,
        'Error: lmdb_materialisation(cached) not yet implemented (see Phase R8 in docs/design/WAM_LMDB_LAZY_IMPLEMENTATION_PLAN.md)~n', []),
    halt(1).

% Lazy mode only makes sense when the LMDB is opened at runtime
% (i.e., lmdb_mode == cursor). Without it there is no LmdbFactSource
% for the LookupSource backend to wrap.
validate_lmdb_materialisation_combo(eager, _).
validate_lmdb_materialisation_combo(lazy, cursor) :- !.
validate_lmdb_materialisation_combo(lazy, LmdbModeAtom) :-
    format(user_error,
        'Error: lmdb_materialisation(lazy) requires lmdb_mode(cursor); got lmdb_mode(~w)~n',
        [LmdbModeAtom]),
    halt(1).

parse_variant(seeded, [
    dialect(swi),
    branch_pruning(false),
    min_closure(false)
]).
parse_variant(accumulated, [
    dialect(swi),
    branch_pruning(false),
    min_closure(false),
    seeded_accumulation(auto)
]).

parse_emit_mode(interpreter, interpreter).
parse_emit_mode(functions, functions).

parse_kernel_mode(kernels_on, []).
parse_kernel_mode(kernels_off, [no_kernels(true)]).

collect_wam_predicates(seeded, [
    user:dimension_n/1,
    user:max_depth/1,
    user:category_ancestor/4,
    user:power_sum_bound/4
]).
collect_wam_predicates(accumulated, [
    user:dimension_n/1,
    user:max_depth/1,
    user:category_ancestor/4,
    user:'category_ancestor$power_sum_bound'/3,
    user:'category_ancestor$power_sum_selected'/3,
    user:'category_ancestor$power_sum_grouped'/3,
    user:'category_ancestor$power_sum_grouped$grouped'/2,
    user:'category_ancestor$power_sum_grouped$grouped$sum_pairs'/2,
    user:'category_ancestor$effective_distance_sum_selected'/3,
    user:'category_ancestor$effective_distance_sum_bound'/3
]).

write_matrix_main(OutputDir, EmitModeAtom, KernelModeAtom, LmdbModeAtom, LmdbMaterialisation) :-
    directory_file_path(OutputDir, 'src', SrcDir),
    directory_file_path(SrcDir, 'main.rs', MainPath),
    format(string(ModeMetric), 'wam_rust_matrix_~w_~w', [EmitModeAtom, KernelModeAtom]),
    (   LmdbModeAtom == cursor
    ->  rust_main_template_lmdb(Template)
    ;   rust_main_template(Template)
    ),
    format(string(BaseCode), Template, [ModeMetric]),
    apply_lmdb_materialisation_transform(LmdbMaterialisation, BaseCode, Code),
    setup_call_cleanup(
        open(MainPath, write, Stream, [encoding(utf8)]),
        format(Stream, '~w', [Code]),
        close(Stream)
    ).

% R7: for lmdb_materialisation(lazy), rewrite the eager-Vec build block
% in the LMDB main template to use the LookupSource trait. The eager
% template stays the source of truth — we string-substitute the
% materialisation block + WAM kernel dispatch. R8 should replace this
% post-process hack with a proper template-section / compiler-option.
apply_lmdb_materialisation_transform(eager, Code, Code).
apply_lmdb_materialisation_transform(lazy, BaseCode, Code) :-
    rewrite_eager_to_lazy(BaseCode, Code).

% --- R7 lazy-mode post-process rewrite ---
%
% Performs two textual edits to the rendered LMDB main.rs:
%   1. Inject a `LazyCategoryParents` struct + LookupSource impl after
%      the LmdbFactSource use-statement.
%   2. Replace the eager runtime_category_parents Vec build (plus
%      register_indexed_atom_fact2 + ffi_facts population +
%      setup_foreign_predicates) with the lazy setup block that
%      registers category_parent/2 as a foreign predicate dispatched
%      via the "lazy_lmdb_lookup" handler.
%
% The generated lazy setup also patches the in-memory WAM Vec<Instruction>
% in place: any CallIndexedAtomFact2("category_parent") is rewritten to
% CallForeign("category_parent", 2). TODO(R8): replace this in-place
% Vec rewrite with proper compiler-option support in wam_target.pl.
rewrite_eager_to_lazy(BaseCode, Code) :-
    inject_lazy_struct(BaseCode, Code1),
    replace_eager_setup_block(Code1, Code).

inject_lazy_struct(Source, Result) :-
    StructAnchor = "use wam_rust_matrix_bench::lmdb_fact_source::LmdbFactSource;",
    lazy_struct_definition(StructCode),
    find_and_split_once(Source, StructAnchor, Prefix, Suffix),
    atomics_to_string([Prefix, StructAnchor, StructCode, Suffix], Result).

replace_eager_setup_block(Source, Result) :-
    BlockStart = "    // Build runtime_category_parents",
    BlockEnd = "    setup_foreign_predicates(&mut vm);",
    lazy_setup_block(Replacement),
    find_and_split_once(Source, BlockStart, Prefix, MidPlusEnd),
    find_and_split_once(MidPlusEnd, BlockEnd, _DiscardedMid, Suffix),
    atomics_to_string([Prefix, Replacement, Suffix], Result).

% First-match deterministic split: Source = Prefix ++ Needle ++ Suffix.
find_and_split_once(Source, Needle, Prefix, Suffix) :-
    once((
        string_concat(Prefix, Rest, Source),
        string_concat(Needle, Suffix, Rest)
    )).

% Rust source for the LazyCategoryParents struct + LookupSource impl.
% Inserted right after `use ... LmdbFactSource;`. Uses the fully-qualified
% trait path so we don't need to also edit the use-statements block.
lazy_struct_definition(Code) :-
    Code = "\n\n// R7 lazy-mode wrapper: bundles LmdbFactSource with the\n// s2i/i2s atom intern maps so the kernel can call into a single\n// LookupSource for category_parent/2.\nstruct LazyCategoryParents {\n    source: LmdbFactSource,\n    s2i: HashMap<String, i32>,\n    i2s: HashMap<i32, String>,\n}\n\nimpl wam_rust_matrix_bench::state::LookupSource for LazyCategoryParents {\n    fn lookup_key_for_atom(&self, atom: &str) -> Option<i32> {\n        self.s2i.get(atom).copied()\n    }\n    fn lookup_parents(&self, key: i32) -> Vec<i32> {\n        self.source.lookup_parents(key).unwrap_or_default()\n    }\n    fn atom_for_key(&self, key: i32) -> Option<String> {\n        self.i2s.get(&key).cloned()\n    }\n}\n".

% Replacement for the eager-Vec build + register/ffi_facts/setup block.
lazy_setup_block(Code) :-
    Code = "    // R7 lazy: skip the eager runtime_category_parents Vec build.\n    // The kernel reads parent edges on-demand via the LookupSource\n    // trait. Materialisation cost drops from O(demand_set_edges) to\n    // O(1) at setup time; per-call cost shifts to the kernel.\n    // The eager path only loads i2s (it translates ints to strings\n    // when building the Vec). Lazy needs s2i too to translate the\n    // kernel's atom A1 to the LMDB int key.\n    let s2i = lmdb.load_s2i().expect(\"load_s2i\");\n    let load_ms = started.elapsed().as_millis();\n\n    let (mut code, mut labels) = shared_wam_program();\n    // R7 lazy: rewrite category_parent dispatch from CallIndexedAtomFact2\n    // to CallForeign so it lands in the \"lazy_lmdb_lookup\" handler.\n    // TODO(R8): move this rewrite into the WAM compiler via a proper\n    // option, instead of patching the emitted Vec<Instruction> here.\n    for instr in code.iter_mut() {\n        if let Instruction::CallIndexedAtomFact2(pred) = instr {\n            if pred == \"category_parent\" {\n                *instr = Instruction::CallForeign(\"category_parent\".to_string(), 2);\n            }\n        }\n    }\n    resolve_targets(&mut code, &labels);\n\n    let mut vm = WamState::new(code, labels);\n    // Register category_parent/2 as a foreign predicate backed by the\n    // LookupSource trait wrapping LmdbFactSource.\n    let lazy_source = std::sync::Arc::new(LazyCategoryParents {\n        source: lmdb.clone(),\n        s2i: s2i.clone(),\n        i2s: i2s.clone(),\n    });\n    vm.register_lazy_lookup(\"category_parent/2\", lazy_source);\n    vm.register_foreign_predicate(\"category_parent/2\");\n    vm.register_foreign_native_kind(\"category_parent/2\", \"lazy_lmdb_lookup\");\n    vm.register_foreign_result_layout(\"category_parent/2\", \"tuple:1\");\n    vm.register_foreign_result_mode(\"category_parent/2\", \"stream\");\n    // Pre-intern every atom the kernel might produce. Mirrors what the\n    // eager path does implicitly via ffi_facts population.\n    for cat in s2i.keys() {\n        vm.intern_atom(cat);\n    }\n    setup_foreign_predicates(&mut vm);".

rust_main_template('use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::time::Instant;

use wam_rust_matrix_bench::foreign_pred_keys;
use wam_rust_matrix_bench::setup_foreign_predicates;
use wam_rust_matrix_bench::shared_wam_program;
use wam_rust_matrix_bench::instructions::Instruction;
use wam_rust_matrix_bench::state::WamState;
use wam_rust_matrix_bench::value::Value;

fn load_tsv_pairs(path: &Path) -> Vec<(String, String)> {
    let file = File::open(path).unwrap_or_else(|e| panic!("cannot open {}: {}", path.display(), e));
    let reader = BufReader::new(file);
    let mut pairs = Vec::new();
    for line in reader.lines().skip(1) {
        let line = line.unwrap();
        let line = line.trim();
        if line.is_empty() || line.starts_with(''#'') {
            continue;
        }
        let parts: Vec<&str> = line.split(''\\t'').collect();
        if parts.len() >= 2 {
            pairs.push((parts[0].to_string(), parts[1].to_string()));
        }
    }
    pairs
}

fn load_single_column(path: &Path) -> Vec<String> {
    let file = File::open(path).unwrap_or_else(|e| panic!("cannot open {}: {}", path.display(), e));
    let reader = BufReader::new(file);
    reader
        .lines()
        .skip(1)
        .filter_map(Result::ok)
        .map(|line| line.trim().to_string())
        .filter(|line| !line.is_empty() && !line.starts_with(''#''))
        .collect()
}

fn build_indexed_fact2(pairs: &[(String, String)]) -> HashMap<String, Vec<String>> {
    let mut grouped: HashMap<String, Vec<String>> = HashMap::new();
    for (a, b) in pairs {
        grouped.entry(a.clone()).or_default().push(b.clone());
    }
    grouped
}

fn build_reverse_fact2(pairs: &[(String, String)]) -> HashMap<String, Vec<String>> {
    let mut grouped: HashMap<String, Vec<String>> = HashMap::new();
    for (child, parent) in pairs {
        grouped.entry(parent.clone()).or_default().push(child.clone());
    }
    grouped
}

fn compute_reachable_to_root(root: &str, reverse_index: &HashMap<String, Vec<String>>, max_depth: usize) -> HashSet<String> {
    use std::collections::VecDeque;
    let mut reachable = HashSet::new();
    let mut best_depth: HashMap<String, usize> = HashMap::new();
    let mut queue: VecDeque<(String, usize)> = VecDeque::from([(root.to_string(), 0)]);
    while let Some((current, depth)) = queue.pop_front() {
        if let Some(prev) = best_depth.get(&current) {
            if depth >= *prev {
                continue;
            }
        }
        best_depth.insert(current.clone(), depth);
        reachable.insert(current.clone());
        if depth >= max_depth {
            continue;
        }
        if let Some(children) = reverse_index.get(&current) {
            for child in children {
                queue.push_back((child.clone(), depth + 1));
            }
        }
    }
    reachable
}

fn append_fact2(code: &mut Vec<Instruction>, labels: &mut HashMap<String, usize>, pred_name: &str, pairs: &[(String, String)]) {
    use std::collections::BTreeMap;
    if pairs.is_empty() {
        return;
    }
    let mut groups: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for (a, b) in pairs {
        groups.entry(a.clone()).or_default().push(b.clone());
    }
    labels.insert(format!("{}/2", pred_name), code.len() + 1);
    let switch_idx = code.len();
    code.push(Instruction::Proceed);
    let mut dispatch = Vec::new();
    for (key, values) in groups {
        let group_label = format!("{}_g_{}", pred_name, key);
        dispatch.push((Value::Atom(key.clone()), group_label.clone()));
        labels.insert(group_label, code.len() + 1);
        for (i, b) in values.iter().enumerate() {
            labels.insert(format!("{}_g_{}_{}", pred_name, key, i), code.len() + 1);
            if values.len() > 1 {
                if i == 0 {
                    code.push(Instruction::TryMeElse(format!("{}_g_{}_{}", pred_name, key, i + 1)));
                } else if i == values.len() - 1 {
                    code.push(Instruction::TrustMe);
                } else {
                    code.push(Instruction::RetryMeElse(format!("{}_g_{}_{}", pred_name, key, i + 1)));
                }
            }
            code.push(Instruction::GetConstant(Value::Atom(key.clone()), "A1".to_string()));
            code.push(Instruction::GetConstant(Value::Atom(b.clone()), "A2".to_string()));
            code.push(Instruction::Proceed);
        }
    }
    code[switch_idx] = Instruction::SwitchOnConstant(dispatch);
}

fn optimize_benchmark_code(code: &mut Vec<Instruction>) {
    let mut i = 0usize;
    while i < code.len() {
        if i + 5 < code.len() {
            let replacement = match (&code[i], &code[i + 1], &code[i + 2], &code[i + 3], &code[i + 4], &code[i + 5]) {
                (
                    Instruction::PutValue(list_reg, a1),
                    Instruction::PutVariable(tmp_reg, a2),
                    Instruction::BuiltinCall(op_len, 2),
                    Instruction::PutValue(tmp_reg2, a1b),
                    Instruction::PutValue(limit_reg, a2b),
                    Instruction::BuiltinCall(op_lt, 2),
                ) if a1 == "A1"
                    && a2 == "A2"
                    && op_len == "length/2"
                    && tmp_reg == tmp_reg2
                    && a1b == "A1"
                    && a2b == "A2"
                    && op_lt == "</2" =>
                    Some(Instruction::ListLengthLt(list_reg.clone(), limit_reg.clone(), 6)),
                _ => None,
            };
            if let Some(instr) = replacement {
                code[i] = instr;
                i += 6;
                continue;
            }
        }
        if i + 1 < code.len() {
            let replacement = match (&code[i], &code[i + 1]) {
                (Instruction::PutVariable(limit_reg, a1), Instruction::Call(pred, 1))
                    if a1 == "A1" && pred == "max_depth/1" =>
                    Some(Instruction::LoadRegisterConstant(Value::Integer(10), limit_reg.clone(), 2)),
                _ => None,
            };
            if let Some(instr) = replacement {
                code[i] = instr;
                i += 2;
                continue;
            }
        }
        if i + 10 < code.len() {
            let replacement = match (&code[i], &code[i + 1], &code[i + 2], &code[i + 3], &code[i + 4], &code[i + 5], &code[i + 6], &code[i + 7], &code[i + 8], &code[i + 9], &code[i + 10]) {
                (
                    Instruction::GetConstant(Value::Atom(raw), hops_reg),
                    Instruction::GetVariable(visited_reg, visited_arg),
                    Instruction::PutValue(cat_reg, a1),
                    Instruction::PutValue(target_reg, a2),
                    Instruction::Call(pred, 2),
                    Instruction::Deallocate,
                    Instruction::PutStructure(functor, a1b),
                    Instruction::SetValue(target_reg2),
                    Instruction::SetValue(visited_reg2),
                    Instruction::BuiltinCall(op, 1),
                    Instruction::Proceed,
                ) if raw == "1"
                    && visited_arg == "A4"
                    && a1 == "A1"
                    && a2 == "A2"
                    && pred == "category_parent/2"
                    && functor == "member/2"
                    && a1b == "A1"
                    && target_reg == target_reg2
                    && visited_reg == visited_reg2
                    && op == r"\\+/1" =>
                    Some(Instruction::BaseCategoryAncestorBind(
                        cat_reg.clone(),
                        target_reg.clone(),
                        hops_reg.clone(),
                        visited_arg.clone(),
                    )),
                _ => None,
            };
            if let Some(instr) = replacement {
                code[i] = instr;
                i += 11;
                continue;
            }
        }
        if i + 8 < code.len() {
            let replacement = match (&code[i], &code[i + 1], &code[i + 2], &code[i + 3], &code[i + 4], &code[i + 5], &code[i + 6], &code[i + 7], &code[i + 8]) {
                (
                    Instruction::PutValue(cat_reg, a1),
                    Instruction::PutValue(target_reg, a2),
                    Instruction::Call(pred, 2),
                    Instruction::PutStructure(functor, a1b),
                    Instruction::SetValue(target_reg2),
                    Instruction::SetValue(visited_reg),
                    Instruction::Deallocate,
                    Instruction::BuiltinCall(op, 1),
                    Instruction::Proceed,
                ) if a1 == "A1"
                    && a2 == "A2"
                    && pred == "category_parent/2"
                    && functor == "member/2"
                    && a1b == "A1"
                    && target_reg == target_reg2
                    && op == r"\\+/1" =>
                    Some(Instruction::BaseCategoryAncestor(cat_reg.clone(), target_reg.clone(), visited_reg.clone())),
                _ => None,
            };
            if let Some(instr) = replacement {
                code[i] = instr;
                i += 9;
                continue;
            }
        }
        if i + 8 < code.len() {
            let replacement = match (&code[i], &code[i + 1], &code[i + 2], &code[i + 3], &code[i + 4], &code[i + 5], &code[i + 6], &code[i + 7], &code[i + 8]) {
                (
                    Instruction::PutValue(cat_reg, a1),
                    Instruction::PutValue(target_reg, a2),
                    Instruction::Call(pred, 2),
                    Instruction::Deallocate,
                    Instruction::PutStructure(functor, a1b),
                    Instruction::SetValue(target_reg2),
                    Instruction::SetValue(visited_reg),
                    Instruction::BuiltinCall(op, 1),
                    Instruction::Proceed,
                ) if a1 == "A1"
                    && a2 == "A2"
                    && pred == "category_parent/2"
                    && functor == "member/2"
                    && a1b == "A1"
                    && target_reg == target_reg2
                    && op == r"\\+/1" =>
                    Some(Instruction::BaseCategoryAncestor(
                        cat_reg.clone(),
                        target_reg.clone(),
                        visited_reg.clone(),
                    )),
                _ => None,
            };
            if let Some(instr) = replacement {
                code[i] = instr;
                i += 9;
                continue;
            }
        }
        if i + 3 < code.len() {
            let replacement = match (&code[i], &code[i + 1], &code[i + 2], &code[i + 3]) {
                (
                    Instruction::PutStructure(functor, a1),
                    Instruction::SetValue(elem_reg),
                    Instruction::SetValue(list_reg),
                    Instruction::BuiltinCall(op, 1),
                ) if functor == "member/2" && a1 == "A1" && op == r"\\+/1" =>
                    Some(Instruction::NotMember(elem_reg.clone(), list_reg.clone(), 4)),
                _ => None,
            };
            if let Some(instr) = replacement {
                code[i] = instr;
                i += 4;
                continue;
            }
        }
        if i + 6 < code.len() {
            let replacement = match (&code[i], &code[i + 1], &code[i + 2], &code[i + 3], &code[i + 4], &code[i + 5], &code[i + 6]) {
                (
                    Instruction::PutValue(mid_reg, a1),
                    Instruction::PutValue(root_reg, a2),
                    Instruction::PutVariable(child_hops_reg, a3),
                    Instruction::PutList(a4),
                    Instruction::SetValue(mid_reg2),
                    Instruction::SetValue(visited_reg),
                    Instruction::Call(pred, 4),
                ) if a1 == "A1"
                    && a2 == "A2"
                    && a3 == "A3"
                    && a4 == "A4"
                    && mid_reg == mid_reg2
                    && pred == "category_ancestor/4" =>
                    Some(Instruction::RecurseCategoryAncestor(
                        mid_reg.clone(),
                        root_reg.clone(),
                        child_hops_reg.clone(),
                        visited_reg.clone(),
                        pred.clone(),
                        7,
                    )),
                _ => None,
            };
            if let Some(instr) = replacement {
                code[i] = instr;
                i += 7;
                continue;
            }
        }
        if i + 6 < code.len() {
            let replacement = match (&code[i], &code[i + 1], &code[i + 2], &code[i + 3], &code[i + 4], &code[i + 5], &code[i + 6]) {
                (
                    Instruction::PutValue(out_reg, a1),
                    Instruction::PutStructure(functor, a2),
                    Instruction::SetValue(in_reg),
                    Instruction::SetConstant(Value::Integer(1)),
                    Instruction::Deallocate,
                    Instruction::BuiltinCall(op, 2),
                    Instruction::Proceed,
                ) if a1 == "A1" && a2 == "A2" && functor == "+/2" && op == "is/2" =>
                    Some(Instruction::ReturnAdd1(out_reg.clone(), in_reg.clone())),
                _ => None,
            };
            if let Some(instr) = replacement {
                code[i] = instr;
                i += 7;
                continue;
            }
        }
        i += 1;
    }
}

fn resolve_targets(code: &mut Vec<Instruction>, labels: &HashMap<String, usize>) {
    let foreign_preds = foreign_pred_keys();
    optimize_benchmark_code(code);
    for instr in code.iter_mut() {
        let replacement = match instr {
            Instruction::Call(pred, arity) if foreign_preds.contains(pred) => {
                Some(Instruction::CallForeign(pred.clone(), *arity))
            }
            Instruction::Call(pred, arity) if pred == "category_parent/2" && *arity == 2 => {
                Some(Instruction::CallIndexedAtomFact2(pred.clone()))
            }
            Instruction::Call(pred, arity) => {
                labels.get(pred).copied().map(|pc| Instruction::CallPc(pc, *arity))
            }
            Instruction::Execute(pred) => {
                labels.get(pred).copied().map(Instruction::ExecutePc)
            }
            Instruction::RecurseCategoryAncestor(mid_reg, root_reg, child_hops_reg, visited_reg, pred, skip) => {
                labels.get(pred).copied().map(|pc| {
                    Instruction::RecurseCategoryAncestorPc(
                        mid_reg.clone(),
                        root_reg.clone(),
                        child_hops_reg.clone(),
                        visited_reg.clone(),
                        pc,
                        *skip,
                    )
                })
            }
            Instruction::TryMeElse(label) => {
                labels.get(label).copied().map(Instruction::TryMeElsePc)
            }
            Instruction::RetryMeElse(label) => {
                labels.get(label).copied().map(Instruction::RetryMeElsePc)
            }
            Instruction::SwitchOnConstant(table) => {
                let mut resolved = Vec::with_capacity(table.len());
                let mut ok = true;
                for (value, label) in table.iter() {
                    if let Some(&pc) = labels.get(label) {
                        resolved.push((value.clone(), pc));
                    } else {
                        ok = false;
                        break;
                    }
                }
                ok.then_some(Instruction::SwitchOnConstantPc(resolved))
            }
            _ => None,
        };
        if let Some(new_instr) = replacement {
            *instr = new_instr;
        }
    }
}

fn value_to_f64(value: &Value) -> Option<f64> {
    match value {
        Value::Integer(v) => Some(*v as f64),
        Value::Float(v) => Some(*v),
        Value::Atom(raw) => raw.parse::<f64>().ok(),
        _ => None,
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: wam_rust_matrix_bench <category_parent.tsv> <article_category.tsv>");
        std::process::exit(1);
    }
    let edge_path = Path::new(&args[1]);
    let article_path = Path::new(&args[2]);
    let facts_dir = edge_path.parent().unwrap_or_else(|| Path::new("."));
    let root_path = facts_dir.join("root_categories.tsv");

    let started = Instant::now();
    let category_parents = load_tsv_pairs(edge_path);
    let article_categories = load_tsv_pairs(article_path);
    let roots = load_single_column(&root_path);
    // WAM_ROOT_ID env var lets the harness pick a specific root by name,
    // matching the equivalent override on the Haskell side and on the
    // LMDB-mode bench.  Empty/missing env var falls back to roots[0].
    let root = std::env::var("WAM_ROOT_ID").ok()
        .filter(|s| !s.is_empty())
        .or_else(|| roots.first().cloned())
        .unwrap_or_default();
    let max_depth_limit = 10usize;
    let reverse_category_parents = build_reverse_fact2(&category_parents);
    let reachable_to_root = compute_reachable_to_root(&root, &reverse_category_parents, max_depth_limit);
    let runtime_category_parents: Vec<(String, String)> = category_parents
        .iter()
        .filter(|(_, parent)| reachable_to_root.contains(parent))
        .cloned()
        .collect();
    let load_ms = started.elapsed().as_millis();

    let (mut code, mut labels) = shared_wam_program();
    append_fact2(&mut code, &mut labels, "category_parent", &runtime_category_parents);
    resolve_targets(&mut code, &labels);

    let mut vm = WamState::new(code, labels);
    vm.register_indexed_atom_fact2("category_parent/2", build_indexed_fact2(&runtime_category_parents));
    for (_pred, table) in &vm.indexed_atom_fact2.clone() {
        for (key, vals) in table {
            vm.intern_atom(key);
            for val in vals {
                vm.intern_atom(val);
            }
        }
    }
    for (pred, table) in &vm.indexed_atom_fact2.clone() {
        let pred_name = pred.split(''/'').next().unwrap_or(pred);
        let mut interned_table: HashMap<u32, Vec<u32>> = HashMap::new();
        for (key, vals) in table {
            let kid = vm.intern_atom(key);
            let vids: Vec<u32> = vals.iter().map(|v| vm.intern_atom(v)).collect();
            interned_table.insert(kid, vids);
        }
        vm.ffi_facts.insert(pred_name.to_string(), interned_table);
    }
    setup_foreign_predicates(&mut vm);

    let mut seed_cats: Vec<String> = article_categories.iter().map(|(_, cat)| cat.clone()).collect();
    seed_cats.sort();
    seed_cats.dedup();
    if let Ok(filter_raw) = std::env::var("WAM_SEED_FILTER") {
        let wanted: HashSet<String> = filter_raw
            .split("|")
            .map(str::trim)
            .filter(|part| !part.is_empty())
            .map(str::to_string)
            .collect();
        if !wanted.is_empty() {
            seed_cats.retain(|cat| wanted.contains(cat));
        }
    }
    if let Ok(limit_raw) = std::env::var("WAM_SEED_LIMIT") {
        match limit_raw.parse::<usize>() {
            Ok(limit) if limit > 0 && seed_cats.len() > limit => seed_cats.truncate(limit),
            Ok(_) => {}
            Err(_) => eprintln!("[WAM-Rust-Matrix] WARNING: WAM_SEED_LIMIT={} is not a valid usize, ignoring", limit_raw),
        }
    }

    let query_start = Instant::now();
    let n = 5.0f64;
    let inv_n = -1.0 / n;
    let step_limit = std::env::var("WAM_STEP_LIMIT")
        .ok()
        .and_then(|raw| raw.parse::<u64>().ok())
        .unwrap_or(1_000_000);
    let mut total_steps = 0u64;
    let mut total_backtracks = 0u64;
    let mut seed_weight_sums: HashMap<String, f64> = HashMap::new();

    for cat in &seed_cats {
        vm.reset_query();
        vm.step_limit = step_limit;
        let mut weight_sum = 0.0f64;
        if vm.foreign_predicates.contains("category_ancestor/4") {
            let cat_id = vm.intern_atom(cat);
            let root_id = vm.intern_atom(&root);
            let visited_ids = vec![cat_id];
            let mut hops = Vec::new();
            vm.collect_native_category_ancestor_hops(
                cat_id,
                root_id,
                &visited_ids,
                max_depth_limit,
                "category_parent",
                &mut hops,
            );
            for hop in hops.iter().take(10001) {
                let distance = (*hop as f64) + 1.0;
                weight_sum += distance.powf(-n);
            }
        } else {
            vm.set_reg("A1", Value::Atom(cat.clone()));
            vm.set_reg("A2", Value::Atom(root.clone()));
            vm.set_reg("A3", Value::Unbound("Hops".to_string()));
            vm.set_reg("A4", Value::List(vec![Value::Atom(cat.clone())]));
            let mut first = true;
            let mut solutions = 0usize;
            loop {
                let succeeded = if first {
                    first = false;
                    if let Some(&pc) = vm.labels.get("category_ancestor/4") {
                        vm.pc = pc;
                        vm.cp = 0;
                        vm.run()
                    } else {
                        false
                    }
                } else {
                    vm.backtrack() && vm.run()
                };
                if !succeeded {
                    break;
                }
                if let Some(hops) = vm.bindings
                    .get("Hops")
                    .cloned()
                    .or_else(|| vm.regs.get(3).cloned().map(|v| vm.deref_var(&v)))
                    .and_then(|value| value_to_f64(&value)) {
                    let distance = hops + 1.0;
                    weight_sum += distance.powf(-n);
                    solutions += 1;
                }
                if solutions > 10000 {
                    break;
                }
            }
        }
        total_steps += vm.step_count;
        total_backtracks += vm.backtrack_count;
        if weight_sum > 0.0 {
            seed_weight_sums.insert(cat.clone(), weight_sum);
        }
    }
    let query_ms = query_start.elapsed().as_millis();

    let agg_start = Instant::now();
    let mut article_sums: HashMap<String, f64> = HashMap::new();
    for (article, cat) in &article_categories {
        let entry = article_sums.entry(article.clone()).or_insert(0.0);
        if *cat == root {
            *entry += 1.0;
        }
        if let Some(weight_sum) = seed_weight_sums.get(cat) {
            *entry += *weight_sum;
        }
    }
    let mut rows: Vec<(f64, String)> = article_sums
        .into_iter()
        .filter_map(|(article, weight_sum)| {
            (weight_sum > 0.0).then(|| (weight_sum.powf(inv_n), article))
        })
        .collect();
    rows.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap().then(a.1.cmp(&b.1)));
    let aggregation_ms = agg_start.elapsed().as_millis();

    println!("article\\troot_category\\teffective_distance");
    for (distance, article) in rows {
        println!("{}\\t{}\\t{:.6}", article, root, distance);
    }

    eprintln!("mode=~w");
    eprintln!("load_ms={}", load_ms);
    eprintln!("query_ms={}", query_ms);
    eprintln!("aggregation_ms={}", aggregation_ms);
    eprintln!("total_ms={}", started.elapsed().as_millis());
    eprintln!("seed_count={}", seed_cats.len());
    if let Ok(seed_limit) = std::env::var("WAM_SEED_LIMIT") {
        eprintln!("seed_limit={}", seed_limit);
    }
    if let Ok(seed_filter) = std::env::var("WAM_SEED_FILTER") {
        eprintln!("seed_filter={}", seed_filter);
    }
    eprintln!("tuple_count={}", seed_weight_sums.len());
    eprintln!("total_steps={}", total_steps);
    eprintln!("total_backtracks={}", total_backtracks);
}
').

rust_main_template_lmdb('use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::time::Instant;

use wam_rust_matrix_bench::foreign_pred_keys;
use wam_rust_matrix_bench::setup_foreign_predicates;
use wam_rust_matrix_bench::shared_wam_program;
use wam_rust_matrix_bench::instructions::Instruction;
use wam_rust_matrix_bench::state::WamState;
use wam_rust_matrix_bench::value::Value;
use wam_rust_matrix_bench::lmdb_fact_source::LmdbFactSource;

fn load_tsv_pairs(path: &Path) -> Vec<(String, String)> {
    let file = File::open(path).unwrap_or_else(|e| panic!("cannot open {}: {}", path.display(), e));
    let reader = BufReader::new(file);
    let mut pairs = Vec::new();
    for line in reader.lines().skip(1) {
        let line = line.unwrap();
        let line = line.trim();
        if line.is_empty() || line.starts_with(''#'') {
            continue;
        }
        let parts: Vec<&str> = line.split(''\\t'').collect();
        if parts.len() >= 2 {
            pairs.push((parts[0].to_string(), parts[1].to_string()));
        }
    }
    pairs
}

fn load_single_column(path: &Path) -> Vec<String> {
    let file = File::open(path).unwrap_or_else(|e| panic!("cannot open {}: {}", path.display(), e));
    let reader = BufReader::new(file);
    reader
        .lines()
        .skip(1)
        .filter_map(Result::ok)
        .map(|line| line.trim().to_string())
        .filter(|line| !line.is_empty() && !line.starts_with(''#''))
        .collect()
}

fn build_indexed_fact2(pairs: &[(String, String)]) -> HashMap<String, Vec<String>> {
    let mut grouped: HashMap<String, Vec<String>> = HashMap::new();
    for (a, b) in pairs {
        grouped.entry(a.clone()).or_default().push(b.clone());
    }
    grouped
}

fn build_reverse_fact2(pairs: &[(String, String)]) -> HashMap<String, Vec<String>> {
    let mut grouped: HashMap<String, Vec<String>> = HashMap::new();
    for (child, parent) in pairs {
        grouped.entry(parent.clone()).or_default().push(child.clone());
    }
    grouped
}

fn compute_reachable_to_root(root: &str, reverse_index: &HashMap<String, Vec<String>>, max_depth: usize) -> HashSet<String> {
    use std::collections::VecDeque;
    let mut reachable = HashSet::new();
    let mut best_depth: HashMap<String, usize> = HashMap::new();
    let mut queue: VecDeque<(String, usize)> = VecDeque::from([(root.to_string(), 0)]);
    while let Some((current, depth)) = queue.pop_front() {
        if let Some(prev) = best_depth.get(&current) {
            if depth >= *prev {
                continue;
            }
        }
        best_depth.insert(current.clone(), depth);
        reachable.insert(current.clone());
        if depth >= max_depth {
            continue;
        }
        if let Some(children) = reverse_index.get(&current) {
            for child in children {
                queue.push_back((child.clone(), depth + 1));
            }
        }
    }
    reachable
}

fn append_fact2(code: &mut Vec<Instruction>, labels: &mut HashMap<String, usize>, pred_name: &str, pairs: &[(String, String)]) {
    use std::collections::BTreeMap;
    if pairs.is_empty() {
        return;
    }
    let mut groups: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for (a, b) in pairs {
        groups.entry(a.clone()).or_default().push(b.clone());
    }
    labels.insert(format!("{}/2", pred_name), code.len() + 1);
    let switch_idx = code.len();
    code.push(Instruction::Proceed);
    let mut dispatch = Vec::new();
    for (key, values) in groups {
        let group_label = format!("{}_g_{}", pred_name, key);
        dispatch.push((Value::Atom(key.clone()), group_label.clone()));
        labels.insert(group_label, code.len() + 1);
        for (i, b) in values.iter().enumerate() {
            labels.insert(format!("{}_g_{}_{}", pred_name, key, i), code.len() + 1);
            if values.len() > 1 {
                if i == 0 {
                    code.push(Instruction::TryMeElse(format!("{}_g_{}_{}", pred_name, key, i + 1)));
                } else if i == values.len() - 1 {
                    code.push(Instruction::TrustMe);
                } else {
                    code.push(Instruction::RetryMeElse(format!("{}_g_{}_{}", pred_name, key, i + 1)));
                }
            }
            code.push(Instruction::GetConstant(Value::Atom(key.clone()), "A1".to_string()));
            code.push(Instruction::GetConstant(Value::Atom(b.clone()), "A2".to_string()));
            code.push(Instruction::Proceed);
        }
    }
    code[switch_idx] = Instruction::SwitchOnConstant(dispatch);
}

fn optimize_benchmark_code(code: &mut Vec<Instruction>) {
    let mut i = 0usize;
    while i < code.len() {
        if i + 5 < code.len() {
            let replacement = match (&code[i], &code[i + 1], &code[i + 2], &code[i + 3], &code[i + 4], &code[i + 5]) {
                (
                    Instruction::PutValue(list_reg, a1),
                    Instruction::PutVariable(tmp_reg, a2),
                    Instruction::BuiltinCall(op_len, 2),
                    Instruction::PutValue(tmp_reg2, a1b),
                    Instruction::PutValue(limit_reg, a2b),
                    Instruction::BuiltinCall(op_lt, 2),
                ) if a1 == "A1"
                    && a2 == "A2"
                    && op_len == "length/2"
                    && tmp_reg == tmp_reg2
                    && a1b == "A1"
                    && a2b == "A2"
                    && op_lt == "</2" =>
                    Some(Instruction::ListLengthLt(list_reg.clone(), limit_reg.clone(), 6)),
                _ => None,
            };
            if let Some(instr) = replacement {
                code[i] = instr;
                i += 6;
                continue;
            }
        }
        if i + 1 < code.len() {
            let replacement = match (&code[i], &code[i + 1]) {
                (Instruction::PutVariable(limit_reg, a1), Instruction::Call(pred, 1))
                    if a1 == "A1" && pred == "max_depth/1" =>
                    Some(Instruction::LoadRegisterConstant(Value::Integer(10), limit_reg.clone(), 2)),
                _ => None,
            };
            if let Some(instr) = replacement {
                code[i] = instr;
                i += 2;
                continue;
            }
        }
        if i + 10 < code.len() {
            let replacement = match (&code[i], &code[i + 1], &code[i + 2], &code[i + 3], &code[i + 4], &code[i + 5], &code[i + 6], &code[i + 7], &code[i + 8], &code[i + 9], &code[i + 10]) {
                (
                    Instruction::GetConstant(Value::Atom(raw), hops_reg),
                    Instruction::GetVariable(visited_reg, visited_arg),
                    Instruction::PutValue(cat_reg, a1),
                    Instruction::PutValue(target_reg, a2),
                    Instruction::Call(pred, 2),
                    Instruction::Deallocate,
                    Instruction::PutStructure(functor, a1b),
                    Instruction::SetValue(target_reg2),
                    Instruction::SetValue(visited_reg2),
                    Instruction::BuiltinCall(op, 1),
                    Instruction::Proceed,
                ) if raw == "1"
                    && visited_arg == "A4"
                    && a1 == "A1"
                    && a2 == "A2"
                    && pred == "category_parent/2"
                    && functor == "member/2"
                    && a1b == "A1"
                    && target_reg == target_reg2
                    && visited_reg == visited_reg2
                    && op == r"\\+/1" =>
                    Some(Instruction::BaseCategoryAncestorBind(
                        cat_reg.clone(),
                        target_reg.clone(),
                        hops_reg.clone(),
                        visited_arg.clone(),
                    )),
                _ => None,
            };
            if let Some(instr) = replacement {
                code[i] = instr;
                i += 11;
                continue;
            }
        }
        if i + 8 < code.len() {
            let replacement = match (&code[i], &code[i + 1], &code[i + 2], &code[i + 3], &code[i + 4], &code[i + 5], &code[i + 6], &code[i + 7], &code[i + 8]) {
                (
                    Instruction::PutValue(cat_reg, a1),
                    Instruction::PutValue(target_reg, a2),
                    Instruction::Call(pred, 2),
                    Instruction::PutStructure(functor, a1b),
                    Instruction::SetValue(target_reg2),
                    Instruction::SetValue(visited_reg),
                    Instruction::Deallocate,
                    Instruction::BuiltinCall(op, 1),
                    Instruction::Proceed,
                ) if a1 == "A1"
                    && a2 == "A2"
                    && pred == "category_parent/2"
                    && functor == "member/2"
                    && a1b == "A1"
                    && target_reg == target_reg2
                    && op == r"\\+/1" =>
                    Some(Instruction::BaseCategoryAncestor(cat_reg.clone(), target_reg.clone(), visited_reg.clone())),
                _ => None,
            };
            if let Some(instr) = replacement {
                code[i] = instr;
                i += 9;
                continue;
            }
        }
        if i + 8 < code.len() {
            let replacement = match (&code[i], &code[i + 1], &code[i + 2], &code[i + 3], &code[i + 4], &code[i + 5], &code[i + 6], &code[i + 7], &code[i + 8]) {
                (
                    Instruction::PutValue(cat_reg, a1),
                    Instruction::PutValue(target_reg, a2),
                    Instruction::Call(pred, 2),
                    Instruction::Deallocate,
                    Instruction::PutStructure(functor, a1b),
                    Instruction::SetValue(target_reg2),
                    Instruction::SetValue(visited_reg),
                    Instruction::BuiltinCall(op, 1),
                    Instruction::Proceed,
                ) if a1 == "A1"
                    && a2 == "A2"
                    && pred == "category_parent/2"
                    && functor == "member/2"
                    && a1b == "A1"
                    && target_reg == target_reg2
                    && op == r"\\+/1" =>
                    Some(Instruction::BaseCategoryAncestor(
                        cat_reg.clone(),
                        target_reg.clone(),
                        visited_reg.clone(),
                    )),
                _ => None,
            };
            if let Some(instr) = replacement {
                code[i] = instr;
                i += 9;
                continue;
            }
        }
        if i + 3 < code.len() {
            let replacement = match (&code[i], &code[i + 1], &code[i + 2], &code[i + 3]) {
                (
                    Instruction::PutStructure(functor, a1),
                    Instruction::SetValue(elem_reg),
                    Instruction::SetValue(list_reg),
                    Instruction::BuiltinCall(op, 1),
                ) if functor == "member/2" && a1 == "A1" && op == r"\\+/1" =>
                    Some(Instruction::NotMember(elem_reg.clone(), list_reg.clone(), 4)),
                _ => None,
            };
            if let Some(instr) = replacement {
                code[i] = instr;
                i += 4;
                continue;
            }
        }
        if i + 6 < code.len() {
            let replacement = match (&code[i], &code[i + 1], &code[i + 2], &code[i + 3], &code[i + 4], &code[i + 5], &code[i + 6]) {
                (
                    Instruction::PutValue(mid_reg, a1),
                    Instruction::PutValue(root_reg, a2),
                    Instruction::PutVariable(child_hops_reg, a3),
                    Instruction::PutList(a4),
                    Instruction::SetValue(mid_reg2),
                    Instruction::SetValue(visited_reg),
                    Instruction::Call(pred, 4),
                ) if a1 == "A1"
                    && a2 == "A2"
                    && a3 == "A3"
                    && a4 == "A4"
                    && mid_reg == mid_reg2
                    && pred == "category_ancestor/4" =>
                    Some(Instruction::RecurseCategoryAncestor(
                        mid_reg.clone(),
                        root_reg.clone(),
                        child_hops_reg.clone(),
                        visited_reg.clone(),
                        pred.clone(),
                        7,
                    )),
                _ => None,
            };
            if let Some(instr) = replacement {
                code[i] = instr;
                i += 7;
                continue;
            }
        }
        if i + 6 < code.len() {
            let replacement = match (&code[i], &code[i + 1], &code[i + 2], &code[i + 3], &code[i + 4], &code[i + 5], &code[i + 6]) {
                (
                    Instruction::PutValue(out_reg, a1),
                    Instruction::PutStructure(functor, a2),
                    Instruction::SetValue(in_reg),
                    Instruction::SetConstant(Value::Integer(1)),
                    Instruction::Deallocate,
                    Instruction::BuiltinCall(op, 2),
                    Instruction::Proceed,
                ) if a1 == "A1" && a2 == "A2" && functor == "+/2" && op == "is/2" =>
                    Some(Instruction::ReturnAdd1(out_reg.clone(), in_reg.clone())),
                _ => None,
            };
            if let Some(instr) = replacement {
                code[i] = instr;
                i += 7;
                continue;
            }
        }
        i += 1;
    }
}

fn resolve_targets(code: &mut Vec<Instruction>, labels: &HashMap<String, usize>) {
    let foreign_preds = foreign_pred_keys();
    optimize_benchmark_code(code);
    for instr in code.iter_mut() {
        let replacement = match instr {
            Instruction::Call(pred, arity) if foreign_preds.contains(pred) => {
                Some(Instruction::CallForeign(pred.clone(), *arity))
            }
            Instruction::Call(pred, arity) if pred == "category_parent/2" && *arity == 2 => {
                Some(Instruction::CallIndexedAtomFact2(pred.clone()))
            }
            Instruction::Call(pred, arity) => {
                labels.get(pred).copied().map(|pc| Instruction::CallPc(pc, *arity))
            }
            Instruction::Execute(pred) => {
                labels.get(pred).copied().map(Instruction::ExecutePc)
            }
            Instruction::RecurseCategoryAncestor(mid_reg, root_reg, child_hops_reg, visited_reg, pred, skip) => {
                labels.get(pred).copied().map(|pc| {
                    Instruction::RecurseCategoryAncestorPc(
                        mid_reg.clone(),
                        root_reg.clone(),
                        child_hops_reg.clone(),
                        visited_reg.clone(),
                        pc,
                        *skip,
                    )
                })
            }
            Instruction::TryMeElse(label) => {
                labels.get(label).copied().map(Instruction::TryMeElsePc)
            }
            Instruction::RetryMeElse(label) => {
                labels.get(label).copied().map(Instruction::RetryMeElsePc)
            }
            Instruction::SwitchOnConstant(table) => {
                let mut resolved = Vec::with_capacity(table.len());
                let mut ok = true;
                for (value, label) in table.iter() {
                    if let Some(&pc) = labels.get(label) {
                        resolved.push((value.clone(), pc));
                    } else {
                        ok = false;
                        break;
                    }
                }
                ok.then_some(Instruction::SwitchOnConstantPc(resolved))
            }
            _ => None,
        };
        if let Some(new_instr) = replacement {
            *instr = new_instr;
        }
    }
}

fn value_to_f64(value: &Value) -> Option<f64> {
    match value {
        Value::Integer(v) => Some(*v as f64),
        Value::Float(v) => Some(*v),
        Value::Atom(raw) => raw.parse::<f64>().ok(),
        _ => None,
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: wam_rust_matrix_bench <fixture-dir>");
        eprintln!("  fixture-dir must contain lmdb_resident/ + article_category.tsv + root_ids.txt");
        std::process::exit(1);
    }
    let fixture_dir = std::path::Path::new(&args[1]);
    let lmdb_subdir = std::env::var("WAM_LMDB_SUBDIR").unwrap_or_else(|_| "lmdb_resident".to_string());
    let lmdb_path: std::path::PathBuf = fixture_dir.join(&lmdb_subdir);
    let article_path = fixture_dir.join("article_category.tsv");
    let root_ids_path = fixture_dir.join("root_ids.txt");

    let started = Instant::now();
    let lmdb = LmdbFactSource::open(lmdb_path.to_str().expect("lmdb path is utf-8"))
        .expect("LmdbFactSource::open");
    let i2s = lmdb.load_i2s().expect("load_i2s");
    let article_categories = load_tsv_pairs(&article_path);

    // Root resolution: WAM_ROOT_ID env wins; else first id from root_ids.txt.
    let root_id: i32 = std::env::var("WAM_ROOT_ID")
        .ok()
        .and_then(|s| s.parse().ok())
        .or_else(|| {
            let file = std::fs::File::open(&root_ids_path).ok()?;
            std::io::BufReader::new(file)
                .lines()
                .filter_map(Result::ok)
                .map(|line| line.trim().to_string())
                .find(|line| !line.is_empty() && !line.starts_with(''#''))
                .and_then(|line| line.parse::<i32>().ok())
        })
        .expect("no root id (set WAM_ROOT_ID or populate root_ids.txt)");
    let root: String = i2s.get(&root_id).cloned().unwrap_or_else(|| {
        eprintln!("[WAM-Rust-Matrix] WARNING: root_id {} not in i2s; using empty string", root_id);
        String::new()
    });

    let max_depth_limit = 10usize;
    let reachable_ids: HashSet<i32> = lmdb
        .reachable_to_root(root_id, max_depth_limit)
        .expect("reachable_to_root");

    // Build runtime_category_parents by iterating each reachable child and
    // looking up its parents.  Only edges where BOTH endpoints are in the
    // demand set are kept, matching TSV-mode filtering semantics.
    let mut runtime_category_parents: Vec<(String, String)> = Vec::new();
    for child_id in &reachable_ids {
        let parents = lmdb.lookup_parents(*child_id).unwrap_or_default();
        for parent_id in parents {
            if reachable_ids.contains(&parent_id) {
                if let (Some(c), Some(p)) = (i2s.get(child_id), i2s.get(&parent_id)) {
                    runtime_category_parents.push((c.clone(), p.clone()));
                }
            }
        }
    }
    let load_ms = started.elapsed().as_millis();

    let (mut code, mut labels) = shared_wam_program();
    append_fact2(&mut code, &mut labels, "category_parent", &runtime_category_parents);
    resolve_targets(&mut code, &labels);

    let mut vm = WamState::new(code, labels);
    vm.register_indexed_atom_fact2("category_parent/2", build_indexed_fact2(&runtime_category_parents));
    for (_pred, table) in &vm.indexed_atom_fact2.clone() {
        for (key, vals) in table {
            vm.intern_atom(key);
            for val in vals {
                vm.intern_atom(val);
            }
        }
    }
    for (pred, table) in &vm.indexed_atom_fact2.clone() {
        let pred_name = pred.split(''/'').next().unwrap_or(pred);
        let mut interned_table: HashMap<u32, Vec<u32>> = HashMap::new();
        for (key, vals) in table {
            let kid = vm.intern_atom(key);
            let vids: Vec<u32> = vals.iter().map(|v| vm.intern_atom(v)).collect();
            interned_table.insert(kid, vids);
        }
        vm.ffi_facts.insert(pred_name.to_string(), interned_table);
    }
    setup_foreign_predicates(&mut vm);

    let mut seed_cats: Vec<String> = article_categories.iter().map(|(_, cat)| cat.clone()).collect();
    seed_cats.sort();
    seed_cats.dedup();
    if let Ok(filter_raw) = std::env::var("WAM_SEED_FILTER") {
        let wanted: HashSet<String> = filter_raw
            .split("|")
            .map(str::trim)
            .filter(|part| !part.is_empty())
            .map(str::to_string)
            .collect();
        if !wanted.is_empty() {
            seed_cats.retain(|cat| wanted.contains(cat));
        }
    }
    if let Ok(limit_raw) = std::env::var("WAM_SEED_LIMIT") {
        match limit_raw.parse::<usize>() {
            Ok(limit) if limit > 0 && seed_cats.len() > limit => seed_cats.truncate(limit),
            Ok(_) => {}
            Err(_) => eprintln!("[WAM-Rust-Matrix] WARNING: WAM_SEED_LIMIT={} is not a valid usize, ignoring", limit_raw),
        }
    }

    let query_start = Instant::now();
    let n = 5.0f64;
    let inv_n = -1.0 / n;
    let step_limit = std::env::var("WAM_STEP_LIMIT")
        .ok()
        .and_then(|raw| raw.parse::<u64>().ok())
        .unwrap_or(1_000_000);
    let mut total_steps = 0u64;
    let mut total_backtracks = 0u64;
    let mut seed_weight_sums: HashMap<String, f64> = HashMap::new();

    for cat in &seed_cats {
        vm.reset_query();
        vm.step_limit = step_limit;
        let mut weight_sum = 0.0f64;
        if vm.foreign_predicates.contains("category_ancestor/4") {
            let cat_id = vm.intern_atom(cat);
            let root_id = vm.intern_atom(&root);
            let visited_ids = vec![cat_id];
            let mut hops = Vec::new();
            vm.collect_native_category_ancestor_hops(
                cat_id,
                root_id,
                &visited_ids,
                max_depth_limit,
                "category_parent",
                &mut hops,
            );
            for hop in hops.iter().take(10001) {
                let distance = (*hop as f64) + 1.0;
                weight_sum += distance.powf(-n);
            }
        } else {
            vm.set_reg("A1", Value::Atom(cat.clone()));
            vm.set_reg("A2", Value::Atom(root.clone()));
            vm.set_reg("A3", Value::Unbound("Hops".to_string()));
            vm.set_reg("A4", Value::List(vec![Value::Atom(cat.clone())]));
            let mut first = true;
            let mut solutions = 0usize;
            loop {
                let succeeded = if first {
                    first = false;
                    if let Some(&pc) = vm.labels.get("category_ancestor/4") {
                        vm.pc = pc;
                        vm.cp = 0;
                        vm.run()
                    } else {
                        false
                    }
                } else {
                    vm.backtrack() && vm.run()
                };
                if !succeeded {
                    break;
                }
                if let Some(hops) = vm.bindings
                    .get("Hops")
                    .cloned()
                    .or_else(|| vm.regs.get(3).cloned().map(|v| vm.deref_var(&v)))
                    .and_then(|value| value_to_f64(&value)) {
                    let distance = hops + 1.0;
                    weight_sum += distance.powf(-n);
                    solutions += 1;
                }
                if solutions > 10000 {
                    break;
                }
            }
        }
        total_steps += vm.step_count;
        total_backtracks += vm.backtrack_count;
        if weight_sum > 0.0 {
            seed_weight_sums.insert(cat.clone(), weight_sum);
        }
    }
    let query_ms = query_start.elapsed().as_millis();

    let agg_start = Instant::now();
    let mut article_sums: HashMap<String, f64> = HashMap::new();
    for (article, cat) in &article_categories {
        let entry = article_sums.entry(article.clone()).or_insert(0.0);
        if *cat == root {
            *entry += 1.0;
        }
        if let Some(weight_sum) = seed_weight_sums.get(cat) {
            *entry += *weight_sum;
        }
    }
    let mut rows: Vec<(f64, String)> = article_sums
        .into_iter()
        .filter_map(|(article, weight_sum)| {
            (weight_sum > 0.0).then(|| (weight_sum.powf(inv_n), article))
        })
        .collect();
    rows.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap().then(a.1.cmp(&b.1)));
    let aggregation_ms = agg_start.elapsed().as_millis();

    println!("article\\troot_category\\teffective_distance");
    for (distance, article) in rows {
        println!("{}\\t{}\\t{:.6}", article, root, distance);
    }

    eprintln!("mode=~w");
    eprintln!("load_ms={}", load_ms);
    eprintln!("query_ms={}", query_ms);
    eprintln!("aggregation_ms={}", aggregation_ms);
    eprintln!("total_ms={}", started.elapsed().as_millis());
    eprintln!("seed_count={}", seed_cats.len());
    if let Ok(seed_limit) = std::env::var("WAM_SEED_LIMIT") {
        eprintln!("seed_limit={}", seed_limit);
    }
    if let Ok(seed_filter) = std::env::var("WAM_SEED_FILTER") {
        eprintln!("seed_filter={}", seed_filter);
    }
    eprintln!("demand_set_size={}", reachable_ids.len());
    eprintln!("root_id={}", root_id);
    eprintln!("tuple_count={}", seed_weight_sums.len());
    eprintln!("total_steps={}", total_steps);
    eprintln!("total_backtracks={}", total_backtracks);
}
').
