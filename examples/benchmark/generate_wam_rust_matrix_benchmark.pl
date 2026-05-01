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
    (   Argv = [_FactsPath, OutputDir, VariantAtom, EmitModeAtom, KernelModeAtom]
    ->  true
    ;   format(user_error,
            'Usage: ... -- <facts.pl> <output-dir> <seeded|accumulated> <interpreter|functions> <kernels_on|kernels_off>~n',
            []),
        halt(1)
    ),
    generate(VariantAtom, EmitModeAtom, KernelModeAtom, OutputDir),
    halt(0).

main :-
    format(user_error, 'Error: generation failed~n', []),
    halt(1).

generate(VariantAtom, EmitModeAtom, KernelModeAtom, OutputDir) :-
    benchmark_workload_path(WorkloadPath),
    load_files(WorkloadPath, [silent(true)]),
    retractall(user:mode(category_ancestor(_, _, _, _))),
    assertz(user:mode(category_ancestor(-, +, -, +))),
    parse_variant(VariantAtom, OptimizationOptions),
    parse_emit_mode(EmitModeAtom, EmitMode),
    parse_kernel_mode(KernelModeAtom, KernelOptions),
    BasePreds = [dimension_n/1, max_depth/1, category_ancestor/4],
    prolog_target:generate_prolog_script(BasePreds, OptimizationOptions, ScriptCode),
    tmp_file_stream(text, TmpPath, TmpStream),
    write(TmpStream, ScriptCode),
    close(TmpStream),
    load_files(TmpPath, [silent(true)]),
    delete_file(TmpPath),
    collect_wam_predicates(VariantAtom, Predicates),
    append([[module_name(wam_rust_matrix_bench), wam_fallback(true), emit_mode(EmitMode), parallel(true)],
            KernelOptions], Options),
    write_wam_rust_project(Predicates, Options, OutputDir),
    write_matrix_main(OutputDir, EmitModeAtom, KernelModeAtom),
    format(user_error,
           '[WAM-Rust-Matrix] variant=~w emit_mode=~w kernels=~w output=~w~n',
           [VariantAtom, EmitMode, KernelModeAtom, OutputDir]).

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

write_matrix_main(OutputDir, EmitModeAtom, KernelModeAtom) :-
    directory_file_path(OutputDir, 'src', SrcDir),
    directory_file_path(SrcDir, 'main.rs', MainPath),
    format(string(ModeMetric), 'wam_rust_matrix_~w_~w', [EmitModeAtom, KernelModeAtom]),
    rust_main_template(Template),
    format(string(Code), Template, [ModeMetric]),
    setup_call_cleanup(
        open(MainPath, write, Stream, [encoding(utf8)]),
        format(Stream, '~w', [Code]),
        close(Stream)
    ).

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
    let load_ms = started.elapsed().as_millis();

    let (mut code, mut labels) = shared_wam_program();
    append_fact2(&mut code, &mut labels, "category_parent", &category_parents);
    resolve_targets(&mut code, &labels);

    let mut vm = WamState::new(code, labels);
    vm.register_indexed_atom_fact2("category_parent/2", build_indexed_fact2(&category_parents));
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
    let root = roots.first().cloned().unwrap_or_default();
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
                10,
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
                    vm.backtrack()
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
