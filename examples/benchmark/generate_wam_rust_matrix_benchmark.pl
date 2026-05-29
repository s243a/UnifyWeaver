:- initialization(main, main).

:- use_module('../../src/unifyweaver/targets/wam_rust_target').
:- use_module('../../src/unifyweaver/targets/prolog_target').
:- use_module('../../src/unifyweaver/core/template_system', [render_template/3]).
:- use_module('../../src/unifyweaver/core/cost_model',
              [resolve_auto_lmdb_materialisation/2, resolve_lmdb_cache_capacity/2]).

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
    (   Argv = [_FactsPath, OutputDir, VariantAtom, EmitModeAtom, KernelModeAtom, LmdbModeAtom, LmdbCrateAtom, LmdbMaterialisationAtom, FactCountAtom]
    ->  true
    ;   Argv = [_FactsPath, OutputDir, VariantAtom, EmitModeAtom, KernelModeAtom, LmdbModeAtom, LmdbCrateAtom, LmdbMaterialisationAtom]
    ->  FactCountAtom = '0'
    ;   Argv = [_FactsPath, OutputDir, VariantAtom, EmitModeAtom, KernelModeAtom, LmdbModeAtom, LmdbCrateAtom]
    ->  LmdbMaterialisationAtom = eager, FactCountAtom = '0'
    ;   Argv = [_FactsPath, OutputDir, VariantAtom, EmitModeAtom, KernelModeAtom, LmdbModeAtom]
    ->  LmdbCrateAtom = auto, LmdbMaterialisationAtom = eager, FactCountAtom = '0'
    ;   Argv = [_FactsPath, OutputDir, VariantAtom, EmitModeAtom, KernelModeAtom]
    ->  LmdbModeAtom = none, LmdbCrateAtom = auto, LmdbMaterialisationAtom = eager, FactCountAtom = '0'
    ;   format(user_error,
            'Usage: ... -- <facts.pl> <output-dir> <seeded|accumulated> <interpreter|functions> <kernels_on|kernels_off> [<none|cursor>] [<lmdb_zero|heed|auto>] [<eager|lazy|cached|auto>] [<fact-count>]~n',
            []),
        halt(1)
    ),
    generate(VariantAtom, EmitModeAtom, KernelModeAtom, LmdbModeAtom, LmdbCrateAtom, LmdbMaterialisationAtom, FactCountAtom, OutputDir),
    halt(0).

main :-
    format(user_error, 'Error: generation failed~n', []),
    halt(1).

generate(VariantAtom, EmitModeAtom, KernelModeAtom, LmdbModeAtom, LmdbCrateAtom, LmdbMaterialisationAtom, FactCountAtom, OutputDir) :-
    benchmark_workload_path(WorkloadPath),
    load_files(WorkloadPath, [silent(true)]),
    retractall(user:mode(category_ancestor(_, _, _, _))),
    assertz(user:mode(category_ancestor(-, +, -, +))),
    parse_variant(VariantAtom, OptimizationOptions),
    parse_emit_mode(EmitModeAtom, EmitMode),
    parse_kernel_mode(KernelModeAtom, KernelOptions),
    parse_lmdb_mode(LmdbModeAtom, LmdbOptions),
    parse_lmdb_crate(LmdbCrateAtom, LmdbCrateOptions),
    materialisation_metadata(FactCountAtom, Metadata),
    resolve_lmdb_materialisation(LmdbMaterialisationAtom, Metadata,
                                 LmdbMaterialisation, LmdbMaterialisationOptions),
    resolve_lmdb_cache_capacity(Metadata, CacheCapacity),
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
    write_matrix_main(OutputDir, EmitModeAtom, KernelModeAtom, LmdbModeAtom, LmdbMaterialisation, CacheCapacity),
    format(user_error,
           '[WAM-Rust-Matrix] variant=~w emit_mode=~w kernels=~w lmdb=~w lmdb_crate=~w materialisation=~w(from ~w) cache_capacity=~w output=~w~n',
           [VariantAtom, EmitMode, KernelModeAtom, LmdbModeAtom, LmdbCrateAtom, LmdbMaterialisation, LmdbMaterialisationAtom, CacheCapacity, OutputDir]).

parse_lmdb_mode(none, []).
parse_lmdb_mode(cursor, [lmdb_mode(cursor)]).

parse_lmdb_crate(auto, [lmdb_crate(auto)]).
parse_lmdb_crate(lmdb_zero, [lmdb_crate(lmdb_zero)]).
parse_lmdb_crate(heed, [lmdb_crate(heed)]).

% lmdb_materialisation option (eager | lazy | cached | auto).
% - eager: build runtime_category_parents Vec at startup.
% - lazy:  parent edges read on-demand via LookupSource trait + LMDB cursor.
% - cached: lazy + a sharded bounded cache (CachedLookup decorator).
% - auto (R8b): the cost-model resolver picks eager/lazy/cached from
%   fixture metadata (fact_count, workload_segregated, ...). Mode
%   selection is codegen-time and static — MemAvailable is read at
%   *runtime* for the cache-capacity clamp, never baked in here.
%
% The matrix bench regenerates one project per fixture scale, so
% fact_count is a legitimate generation-time parameter. When omitted
% (FactCountAtom = '0') the resolver's safe default is eager.
materialisation_metadata(FactCountAtom, Metadata) :-
    ( atom_number(FactCountAtom, FactCount0) -> true ; FactCount0 = 0 ),
    FactCount is max(0, FactCount0),
    Metadata = [fact_count(FactCount)].

% Resolve the requested materialisation atom against fixture metadata.
% Explicit eager/lazy/cached pass through unchanged; auto runs the
% cost-model decision tree. Always re-derives the option list from the
% resolved mode so downstream codegen sees the concrete mode.
resolve_lmdb_materialisation(RequestedAtom, Metadata, Mode, [lmdb_materialisation(Mode)]) :-
    resolve_auto_lmdb_materialisation([lmdb_materialisation(RequestedAtom) | Metadata], Mode).

% Lazy / cached modes only make sense when the LMDB is opened at
% runtime (i.e., lmdb_mode == cursor). Without it there is no
% LmdbFactSource for the LookupSource backend to wrap.
validate_lmdb_materialisation_combo(eager, _).
validate_lmdb_materialisation_combo(lazy, cursor) :- !.
validate_lmdb_materialisation_combo(lazy, LmdbModeAtom) :-
    format(user_error,
        'Error: lmdb_materialisation(lazy) requires lmdb_mode(cursor); got lmdb_mode(~w)~n',
        [LmdbModeAtom]),
    halt(1).
validate_lmdb_materialisation_combo(cached, cursor) :- !.
validate_lmdb_materialisation_combo(cached, LmdbModeAtom) :-
    format(user_error,
        'Error: lmdb_materialisation(cached) requires lmdb_mode(cursor); got lmdb_mode(~w)~n',
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

write_matrix_main(OutputDir, EmitModeAtom, KernelModeAtom, LmdbModeAtom, LmdbMaterialisation, CacheCapacity) :-
    directory_file_path(OutputDir, 'src', SrcDir),
    directory_file_path(SrcDir, 'main.rs', MainPath),
    format(string(ModeMetric), 'wam_rust_matrix_~w_~w', [EmitModeAtom, KernelModeAtom]),
    (   LmdbModeAtom == cursor
    ->  rust_main_template_lmdb(Template)
    ;   rust_main_template(Template)
    ),
    format(string(BaseCode), Template, [ModeMetric]),
    apply_lmdb_materialisation_transform(LmdbMaterialisation, CacheCapacity, BaseCode, Code),
    setup_call_cleanup(
        open(MainPath, write, Stream, [encoding(utf8)]),
        format(Stream, '~w', [Code]),
        close(Stream)
    ).

% R8a: for lmdb_materialisation(lazy|cached), render the
% materialisation-mode-dispatched setup block from the mustache
% template and splice it in place of the eager block in
% `rust_main_template_lmdb`. The LazyCategoryParents struct (shared
% by lazy and cached) is injected after the LmdbFactSource use
% statement. Eager mode is a pass-through (the base template
% already has the eager setup inline).
apply_lmdb_materialisation_transform(eager, _CacheCapacity, Code, Code).
apply_lmdb_materialisation_transform(lazy, CacheCapacity, BaseCode, Code) :-
    rewrite_for_lookup_source_mode(lazy, CacheCapacity, BaseCode, Code).
apply_lmdb_materialisation_transform(cached, CacheCapacity, BaseCode, Code) :-
    rewrite_for_lookup_source_mode(cached, CacheCapacity, BaseCode, Code).

% --- R8a lookup-source-mode rewrite (lazy + cached) ---
%
% Used by apply_lmdb_materialisation_transform for both lazy and
% cached modes. Renders the materialisation-setup mustache template
% with the appropriate {{case}} branch active, then splices it into
% the rendered base main.rs at the eager-block anchors. Also injects
% the LazyCategoryParents struct (shared between lazy and cached) at
% module scope.
rewrite_for_lookup_source_mode(Mode, CacheCapacity, BaseCode, Code) :-
    materialisation_setup_dict(Mode, CacheCapacity, Dict),
    materialisation_setup_template(SetupTemplate),
    render_template(SetupTemplate, Dict, RenderedSetup),
    lazy_struct_template(StructTemplate),
    render_template(StructTemplate, [], RenderedStruct),
    inject_lazy_struct(BaseCode, RenderedStruct, Code1),
    replace_eager_setup_block(Code1, RenderedSetup, Code).

% Build the Dict passed to render_template for the materialisation
% setup template. cache_capacity_default is the §3.1
% `unrestricted_working_set` clamp computed by R8b's resolver
% (resolve_lmdb_cache_capacity/2); the generated Rust applies the
% MemAvailable-dependent clamp on top of it at runtime. cache_cap_pct
% and cache_floor_bytes parameterise that runtime clamp (§3.1).
materialisation_setup_dict(Mode, CacheCapacity, [
    materialisation = Mode,
    cache_capacity_default = CacheCapacity,
    cache_shards_default = 4,
    cache_cap_pct = '0.5',
    cache_floor_bytes = 536870912
]).

% Locate + read the materialisation-setup mustache template.
materialisation_setup_template(Code) :-
    rust_wam_template_path('materialisation_setup.rs.mustache', Path),
    read_file_to_string(Path, Code, []).

% Locate + read the LazyCategoryParents struct mustache.
lazy_struct_template(Code) :-
    rust_wam_template_path('lazy_category_parents.rs.mustache', Path),
    read_file_to_string(Path, Code, []).

% Compute the absolute path of a template under templates/targets/rust_wam/.
% Mirrors the F# pattern (`fsharp_*_template_source/1`): walk up from
% this source file to project root, append the templates/ path.
rust_wam_template_path(Filename, Path) :-
    source_file(rust_wam_template_path(_, _), SrcFile),
    file_directory_name(SrcFile, BenchDir),         % examples/benchmark/
    file_directory_name(BenchDir, ExamplesDir),     % examples/
    file_directory_name(ExamplesDir, ProjectRoot),  % project root
    atomic_list_concat([ProjectRoot,
                        '/templates/targets/rust_wam/', Filename], Path).

inject_lazy_struct(Source, RenderedStruct, Result) :-
    StructAnchor = "use wam_lib::lmdb_fact_source::LmdbFactSource;",
    find_and_split_once(Source, StructAnchor, Prefix, Suffix),
    atomics_to_string([Prefix, StructAnchor, "\n", RenderedStruct, Suffix], Result).

replace_eager_setup_block(Source, RenderedSetup, Result) :-
    BlockStart = "    // Build runtime_category_parents",
    BlockEnd = "    setup_foreign_predicates(&mut vm);",
    find_and_split_once(Source, BlockStart, Prefix, MidPlusEnd),
    find_and_split_once(MidPlusEnd, BlockEnd, _DiscardedMid, Suffix),
    atomics_to_string([Prefix, RenderedSetup, Suffix], Result).

% First-match deterministic split: Source = Prefix ++ Needle ++ Suffix.
find_and_split_once(Source, Needle, Prefix, Suffix) :-
    once((
        string_concat(Prefix, Rest, Source),
        string_concat(Needle, Suffix, Rest)
    )).

% (R7's inline lazy_struct_definition/1 and lazy_setup_block/1 were
% retired in R8a — their content now lives in
% templates/targets/rust_wam/{lazy_category_parents,materialisation_setup}.rs.mustache
% and is rendered via the template_system.)

rust_main_template('use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::time::Instant;

use wam_lib::foreign_pred_keys;
use wam_lib::setup_foreign_predicates;
use wam_lib::shared_wam_program;
use wam_lib::instructions::Instruction;
use wam_lib::state::WamState;
use wam_lib::value::Value;

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

use wam_lib::foreign_pred_keys;
use wam_lib::setup_foreign_predicates;
use wam_lib::shared_wam_program;
use wam_lib::instructions::Instruction;
use wam_lib::state::WamState;
use wam_lib::value::Value;
use wam_lib::lmdb_fact_source::LmdbFactSource;

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
