:- initialization(main, main).

:- use_module('../../src/unifyweaver/targets/prolog_target').
:- use_module('../../src/unifyweaver/targets/wam_target').
:- use_module('../../src/unifyweaver/targets/wam_rust_target').
:- use_module(library(option)).
:- use_module(library(lists)).

%% generate_wam_effective_distance_benchmark.pl
%%
%% Generates a hybrid WAM-Rust benchmark for the effective-distance workload.
%%
%% Pipeline:
%%   1. Load the effective-distance Prolog workload
%%   2. Generate optimized predicates via prolog_target (seeded accumulation)
%%   3. Compile the optimized predicates to WAM instructions
%%   4. Emit a Rust Cargo project with a MERGED code vector and benchmark driver
%%
%% The merged code vector is critical: all predicates (category_ancestor/4,
%% dimension_n/1, max_depth/1) share one instruction array so that WAM Call
%% instructions can dispatch to any predicate via the label map.
%%
%% Fact predicates (category_parent/2, article_category/2, root_category/1)
%% are loaded at runtime from TSV files and injected into the code vector.
%%
%% Usage:
%%   swipl -q -s generate_wam_effective_distance_benchmark.pl -- <facts.pl> <output-dir>

benchmark_workload_path(Path) :-
    source_file(benchmark_workload_path(_), ThisFile),
    file_directory_name(ThisFile, Here),
    directory_file_path(Here, 'effective_distance.pl', Path).

main :-
    current_prolog_flag(argv, Argv),
    (   Argv = [FactsPath, OutputDir]
    ->  true
    ;   format(user_error,
            'Usage: swipl -q -s generate_wam_effective_distance_benchmark.pl -- <facts.pl> <output-dir>~n',
            []),
        halt(1)
    ),
    generate_wam_benchmark(FactsPath, OutputDir),
    halt(0).

main :-
    format(user_error, 'Error: generation failed~n', []),
    halt(1).

generate_wam_benchmark(_FactsPath, OutputDir) :-
    % Load the benchmark workload to get predicate definitions
    benchmark_workload_path(WorkloadPath),
    load_files(WorkloadPath, [silent(true)]),
    retractall(user:mode(category_ancestor(_, _, _, _))),
    assertz(user:mode(category_ancestor(-, +, -, +))),

    % Compile core predicates to WAM
    WamPredicates = [
        user:dimension_n/1,
        user:max_depth/1,
        user:category_ancestor/4
    ],
    compile_predicates_to_merged_rust(WamPredicates, MergedInstrCode, MergedLabelCode),

    % Generate project
    make_directory_path(OutputDir),
    directory_file_path(OutputDir, 'src', SrcDir),
    make_directory_path(SrcDir),

    % Write Cargo.toml
    directory_file_path(OutputDir, 'Cargo.toml', CargoPath),
    write_cargo_toml(CargoPath),

    % Copy runtime support files from templates
    write_runtime_files(SrcDir),

    % Write main.rs with merged WAM instructions and benchmark driver
    directory_file_path(SrcDir, 'main.rs', MainPath),
    write_main_rs(MainPath, MergedInstrCode, MergedLabelCode),

    format(user_error, '[WAM-Rust] Generated benchmark project at: ~w~n', [OutputDir]).

%% compile_predicates_to_merged_rust(+PredIndicators, -InstrCode, -LabelCode)
%  Compiles multiple predicates to WAM, then merges their instruction arrays
%  and label maps into a single combined Rust code fragment.
compile_predicates_to_merged_rust(PredIndicators, InstrCode, LabelCode) :-
    % StartPC=2 because the code vec starts with a Proceed halt sentinel at index 0,
    % and fetch(pc) returns code[pc-1], so PC=2 maps to code[1] which is the first
    % compiled instruction.
    compile_predicates_to_merged_rust_(PredIndicators, 2, InstrParts, LabelParts),
    atomic_list_concat(InstrParts, '\n', InstrCode),
    atomic_list_concat(LabelParts, '\n', LabelCode).

compile_predicates_to_merged_rust_([], _, [], []).
compile_predicates_to_merged_rust_([PredIndicator|Rest], StartPC, AllInstrs, AllLabels) :-
    wam_target:compile_predicate_to_wam(PredIndicator, [], WamCode),
    (   PredIndicator = _Module:Pred/Arity -> true
    ;   PredIndicator = Pred/Arity
    ),
    format(user_error, '  Compiled ~w/~w to WAM~n', [Pred, Arity]),
    % Parse WAM code into Rust instruction/label fragments with offset
    wam_code_to_merged_rust(WamCode, StartPC, InstrParts, LabelParts, NextPC),
    compile_predicates_to_merged_rust_(Rest, NextPC, RestInstrs, RestLabels),
    append(InstrParts, RestInstrs, AllInstrs),
    append(LabelParts, RestLabels, AllLabels).

%% wam_code_to_merged_rust(+WamCode, +StartPC, -InstrParts, -LabelParts, -NextPC)
%  Like wam_code_to_rust_instructions but with a PC offset for merging.
wam_code_to_merged_rust(WamCode, StartPC, InstrParts, LabelParts, NextPC) :-
    atom_string(WamCode, WamStr),
    split_string(WamStr, "\n", "", Lines),
    wam_lines_to_merged_rust(Lines, StartPC, InstrParts, LabelParts, NextPC).

wam_lines_to_merged_rust([], PC, [], [], PC).
wam_lines_to_merged_rust([Line|Rest], PC, Instrs, Labels, FinalPC) :-
    split_string(Line, " \t,", " \t,", Parts),
    delete(Parts, "", CleanParts),
    (   CleanParts == []
    ->  wam_lines_to_merged_rust(Rest, PC, Instrs, Labels, FinalPC)
    ;   CleanParts = [First|_],
        (   sub_string(First, _, 1, 0, ":")
        ->  sub_string(First, 0, _, 1, LabelName),
            format(string(LabelInsert),
                '    all_labels.insert("~w".to_string(), ~w);', [LabelName, PC]),
            Labels = [LabelInsert|RestLabels],
            wam_lines_to_merged_rust(Rest, PC, Instrs, RestLabels, FinalPC)
        ;   wam_rust_target:wam_line_to_rust_instr(CleanParts, RustInstr),
            format(string(InstrEntry), '        ~w,', [RustInstr]),
            NPC is PC + 1,
            Instrs = [InstrEntry|RestInstrs],
            wam_lines_to_merged_rust(Rest, NPC, RestInstrs, Labels, FinalPC)
        )
    ).

%% write_cargo_toml(+Path)
write_cargo_toml(Path) :-
    setup_call_cleanup(
        open(Path, write, S),
        (   format(S, '[package]~n', []),
            format(S, 'name = "hybrid_ed_bench"~n', []),
            format(S, 'version = "0.1.0"~n', []),
            format(S, 'edition = "2021"~n', [])
        ),
        close(S)
    ).

%% write_runtime_files(+SrcDir)
%  Writes the WAM runtime support files (value.rs, instructions.rs, state.rs).
write_runtime_files(SrcDir) :-
    get_time(TimeStamp),
    format_time(string(Date), "%Y-%m-%d %H:%M:%S", TimeStamp),
    Options = [],

    wam_rust_target:read_template_file('templates/targets/rust_wam/value.rs.mustache', VT),
    wam_rust_target:render_template(VT, [date=Date], ValueCode),
    directory_file_path(SrcDir, 'value.rs', VP),
    write_file(VP, ValueCode),

    wam_rust_target:read_template_file('templates/targets/rust_wam/instructions.rs.mustache', IT),
    wam_rust_target:render_template(IT, [date=Date], InstrCode),
    directory_file_path(SrcDir, 'instructions.rs', IP),
    write_file(IP, InstrCode),

    wam_rust_target:read_template_file('templates/targets/rust_wam/state.rs.mustache', ST),
    wam_rust_target:render_template(ST, [date=Date], StateBase),
    wam_rust_target:compile_wam_runtime_to_rust(Options, RuntimeCode),
    format(string(StateCode), "~w\n\n~w", [StateBase, RuntimeCode]),
    directory_file_path(SrcDir, 'state.rs', SP),
    write_file(SP, StateCode).

write_file(Path, Content) :-
    setup_call_cleanup(
        open(Path, write, Stream),
        format(Stream, "~w", [Content]),
        close(Stream)
    ).

%% write_main_rs(+Path, +MergedInstrCode, +MergedLabelCode)
write_main_rs(Path, MergedInstrCode, MergedLabelCode) :-
    format(string(Code),
'// Generated WAM-Rust effective-distance benchmark driver
// Compiled predicates: category_ancestor/4, dimension_n/1, max_depth/1
// Fact predicates (category_parent/2, article_category/2, root_category/1)
// are loaded at runtime and injected into the WAM code vector.

use std::collections::HashMap;
use std::io::{BufRead, BufReader};
use std::fs::File;
use std::time::Instant;

mod value;
mod instructions;
mod state;

use value::Value;
use instructions::Instruction;
use state::WamState;

#[derive(Debug)]
struct SeedProfile {
    category: String,
    elapsed_ms: u128,
    steps: u64,
    backtracks: u64,
    solutions: u32,
    weight_sum: f64,
}

/// Load a two-column TSV file into pairs (skips the header line).
fn load_tsv_pairs(path: &str) -> Vec<(String, String)> {
    let file = File::open(path).unwrap_or_else(|e| panic!("Cannot open {}: {}", path, e));
    let reader = BufReader::new(file);
    let mut pairs = Vec::new();
    for line in reader.lines().skip(1) {
        let line = line.unwrap();
        let line = line.trim();
        if line.is_empty() || line.starts_with(''#'') { continue; }
        let parts: Vec<&str> = line.split(''\\t'').collect();
        if parts.len() >= 2 {
            pairs.push((parts[0].to_string(), parts[1].to_string()));
        }
    }
    pairs
}

/// Load single-column values (skips the header line).
fn load_single_column(path: &str) -> Vec<String> {
    let file = File::open(path).unwrap_or_else(|e| panic!("Cannot open {}: {}", path, e));
    let reader = BufReader::new(file);
    let mut vals = Vec::new();
    for line in reader.lines().skip(1) {
        let line = line.unwrap();
        let line = line.trim().to_string();
        if line.is_empty() || line.starts_with(''#'') { continue; }
        vals.push(line);
    }
    vals
}

/// Build WAM fact instructions for a 2-column relation with first-argument indexing.
/// Groups facts by first argument and emits a SwitchOnConstant dispatch table
/// followed by per-group try/retry/trust chains.
fn append_fact2(
    code: &mut Vec<Instruction>,
    labels: &mut HashMap<String, usize>,
    pred_name: &str,
    pairs: &[(String, String)],
) {
    use std::collections::BTreeMap;
    if pairs.is_empty() { return; }

    // Group facts by first argument
    let mut groups: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for (a, b) in pairs {
        groups.entry(a.clone()).or_default().push(b.clone());
    }

    // Entry point: SwitchOnConstant dispatch on A1
    let entry_pc = code.len() + 1;
    labels.insert(format!("{}/2", pred_name), entry_pc);

    // Build dispatch table entries (will be filled with labels after emitting groups)
    let mut dispatch: Vec<(Value, String)> = Vec::new();
    let switch_idx = code.len();
    code.push(Instruction::Proceed); // placeholder — replaced below

    // Emit per-group fact chains
    for (key, values) in &groups {
        let group_label = format!("{}_g_{}", pred_name, key);
        dispatch.push((Value::Atom(key.clone()), group_label.clone()));
        labels.insert(group_label, code.len() + 1);

        for (i, b) in values.iter().enumerate() {
            let fact_label = format!("{}_g_{}_{}", pred_name, key, i);
            labels.insert(fact_label, code.len() + 1);

            if values.len() > 1 {
                if i == 0 {
                    code.push(Instruction::TryMeElse(
                        format!("{}_g_{}_{}", pred_name, key, i + 1)));
                } else if i == values.len() - 1 {
                    code.push(Instruction::TrustMe);
                } else {
                    code.push(Instruction::RetryMeElse(
                        format!("{}_g_{}_{}", pred_name, key, i + 1)));
                }
            }
            code.push(Instruction::GetConstant(Value::Atom(key.clone()), "A1".to_string()));
            code.push(Instruction::GetConstant(Value::Atom(b.clone()), "A2".to_string()));
            code.push(Instruction::Proceed);
        }
    }

    // Replace the placeholder with the actual SwitchOnConstant
    code[switch_idx] = Instruction::SwitchOnConstant(dispatch);
}

/// Build WAM fact instructions for a 1-column relation with first-argument indexing.
fn append_fact1(
    code: &mut Vec<Instruction>,
    labels: &mut HashMap<String, usize>,
    pred_name: &str,
    values: &[String],
) {
    if values.is_empty() { return; }

    let entry_pc = code.len() + 1;
    labels.insert(format!("{}/1", pred_name), entry_pc);

    // For 1-column facts, SwitchOnConstant dispatch is a direct jump
    let mut dispatch: Vec<(Value, String)> = Vec::new();
    let switch_idx = code.len();
    code.push(Instruction::Proceed); // placeholder

    for (i, val) in values.iter().enumerate() {
        let fact_label = format!("{}_{}", pred_name, i);
        dispatch.push((Value::Atom(val.clone()), fact_label.clone()));
        labels.insert(fact_label, code.len() + 1);
        code.push(Instruction::GetConstant(Value::Atom(val.clone()), "A1".to_string()));
        code.push(Instruction::Proceed);
    }

    code[switch_idx] = Instruction::SwitchOnConstant(dispatch);
}

fn resolve_call_targets(code: &mut Vec<Instruction>, labels: &HashMap<String, usize>) {
    for instr in code.iter_mut() {
        let replacement = match instr {
            Instruction::Call(pred, arity) => {
                labels.get(pred).copied().map(|pc| Instruction::CallPc(pc, *arity))
            }
            Instruction::Execute(pred) => {
                labels.get(pred).copied().map(Instruction::ExecutePc)
            }
            _ => None,
        };
        if let Some(new_instr) = replacement {
            *instr = new_instr;
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: hybrid_ed_bench <facts-dir>");
        std::process::exit(1);
    }
    let facts_dir = &args[1];

    let start = Instant::now();

    // Load facts
    let category_parents = load_tsv_pairs(&format!("{}/category_parent.tsv", facts_dir));
    let article_categories = load_tsv_pairs(&format!("{}/article_category.tsv", facts_dir));
    let roots = load_single_column(&format!("{}/root_categories.tsv", facts_dir));

    let load_ms = start.elapsed().as_millis();

    // Build merged WAM code vector:
    // 1. Compiled predicates (category_ancestor/4, dimension_n/1, max_depth/1)
    // 2. Runtime fact predicates (category_parent/2, article_category/2, root_category/1)
    let mut all_code: Vec<Instruction> = vec![
        Instruction::Proceed, // index 0 = halt sentinel
        // --- Compiled predicates (generated at build time) ---
~w
    ];
    let mut all_labels: HashMap<String, usize> = HashMap::new();
~w

    // --- Runtime fact predicates ---
    append_fact2(&mut all_code, &mut all_labels, "category_parent", &category_parents);
    append_fact2(&mut all_code, &mut all_labels, "article_category", &article_categories);
    append_fact1(&mut all_code, &mut all_labels, "root_category", &roots);
    resolve_call_targets(&mut all_code, &all_labels);

    // Create VM with merged code
    let mut vm = WamState::new(all_code, all_labels);

    let query_start = Instant::now();

    // Collect unique seed categories (categories that articles belong to)
    let mut seed_cats: Vec<String> = article_categories.iter().map(|(_, c)| c.clone()).collect();
    seed_cats.sort();
    seed_cats.dedup();
    let seed_count = seed_cats.len();

    let root = roots[0].clone();
    let n: f64 = 5.0;
    let neg_n: f64 = -n;
    let profile_enabled = std::env::var("WAM_PROFILE").ok().as_deref() == Some("1");

    // For each seed category, run category_ancestor(Cat, Root, Hops, [Cat])
    // through the WAM VM and compute weight_sum = Σ (Hops+1)^(-n)
    let mut seed_weight_sums: HashMap<String, f64> = HashMap::new();
    let mut seed_profiles: Vec<SeedProfile> = Vec::new();
    let mut total_steps: u64 = 0;
    let mut total_backtracks: u64 = 0;

    for cat in &seed_cats {
        let mut weight_sum: f64 = 0.0;
        let seed_start = Instant::now();

        // Reset VM mutable state (code/labels are shared, not cloned)
        vm.reset_query();
        vm.step_limit = 500_000; // cap exploration per seed category

        vm.set_reg("A1", Value::Atom(cat.clone()));
        vm.set_reg("A2", Value::Atom(root.clone()));
        vm.set_reg("A3", Value::Unbound("Hops".to_string()));
        vm.set_reg("A4", Value::List(vec![Value::Atom(cat.clone())]));

        if let Some(&pc) = vm.labels.get("category_ancestor/4") {
            vm.pc = pc;
            vm.cp = 0;

            let mut solutions = 0u32;
            loop {
                let succeeded = vm.run();
                if succeeded {
                    // Read Hops from the binding table, not A3 directly.
                    // A3 gets overwritten by recursive calls, but the original
                    // Unbound("Hops") variable is bound via bind_var.
                    if let Some(hops_val) = vm.bindings.get("Hops").cloned()
                        .or_else(|| vm.regs.get("A3").cloned().map(|v| vm.deref_var(&v))) {
                        let hops = match &hops_val {
                            Value::Integer(h) => *h as f64,
                            Value::Float(h) => *h,
                            Value::Atom(s) => match s.parse::<f64>() {
                                Ok(v) => v,
                                Err(_) => { if !vm.backtrack() { break; } continue; }
                            },
                            _ => { if !vm.backtrack() { break; } continue; }
                        };
                        let d = hops + 1.0;
                        weight_sum += d.powf(neg_n);
                        solutions += 1;
                    }
                    // Safety limit: deep paths contribute negligibly to d_eff
                    // (with n=5, 10000 paths each at depth 10 add < 0.06 to weight_sum)
                    if solutions > 10000 { break; }
                    if !vm.backtrack() { break; }
                } else {
                    break;
                }
            }

            total_steps += vm.step_count;
            total_backtracks += vm.backtrack_count;
            if profile_enabled {
                let profile = SeedProfile {
                    category: cat.clone(),
                    elapsed_ms: seed_start.elapsed().as_millis(),
                    steps: vm.step_count,
                    backtracks: vm.backtrack_count,
                    solutions,
                    weight_sum,
                };
                eprintln!(
                    "seed_progress category={} elapsed_ms={} steps={} backtracks={} solutions={} weight_sum={:.6}",
                    profile.category,
                    profile.elapsed_ms,
                    profile.steps,
                    profile.backtracks,
                    profile.solutions,
                    profile.weight_sum,
                );
                seed_profiles.push(profile);
            }
        }

        if weight_sum > 0.0 {
            seed_weight_sums.insert(cat.clone(), weight_sum);
        }
    }

    let query_ms = query_start.elapsed().as_millis();
    let agg_start = Instant::now();

    // Aggregate per-article weight sums
    let mut article_sums: HashMap<String, f64> = HashMap::new();
    for (article, cat) in &article_categories {
        let entry = article_sums.entry(article.clone()).or_insert(0.0);
        if *cat == root {
            *entry += 1.0; // direct: distance=1, weight=1^(-n)=1
        }
        if let Some(&ws) = seed_weight_sums.get(cat) {
            *entry += ws;
        }
    }

    // Compute effective distance
    let inv_n = -1.0 / n;
    let mut results: Vec<(f64, String)> = Vec::new();
    for (article, ws) in &article_sums {
        if *ws > 0.0 {
            results.push((ws.powf(inv_n), article.clone()));
        }
    }
    results.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap().then(a.1.cmp(&b.1)));

    let agg_ms = agg_start.elapsed().as_millis();

    // Output TSV
    println!("article\\troot_category\\teffective_distance");
    for (deff, article) in &results {
        println!("{}\\t{}\\t{:.6}", article, root, deff);
    }

    let total_ms = start.elapsed().as_millis();
    eprintln!("mode=wam_rust_accumulated");
    eprintln!("load_ms={}", load_ms);
    eprintln!("query_ms={}", query_ms);
    eprintln!("aggregation_ms={}", agg_ms);
    eprintln!("total_ms={}", total_ms);
    eprintln!("seed_count={}", seed_count);
    eprintln!("tuple_count={}", seed_weight_sums.len());
    eprintln!("article_count={}", results.len());
    eprintln!("total_steps={}", total_steps);
    eprintln!("total_backtracks={}", total_backtracks);

    if profile_enabled {
        seed_profiles.sort_by(|a, b| {
            b.elapsed_ms.cmp(&a.elapsed_ms)
                .then(b.steps.cmp(&a.steps))
                .then(a.category.cmp(&b.category))
        });
        eprintln!("profile_top_seeds={}", seed_profiles.len().min(10));
        for seed in seed_profiles.iter().take(10) {
            eprintln!(
                "seed_profile category={} elapsed_ms={} steps={} backtracks={} solutions={} weight_sum={:.6}",
                seed.category,
                seed.elapsed_ms,
                seed.steps,
                seed.backtracks,
                seed.solutions,
                seed.weight_sum,
            );
        }
    }
}
', [MergedInstrCode, MergedLabelCode]),
    setup_call_cleanup(
        open(Path, write, Stream),
        format(Stream, '~w', [Code]),
        close(Stream)
    ).
