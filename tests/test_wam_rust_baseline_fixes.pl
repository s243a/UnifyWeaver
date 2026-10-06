:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (s243a)
%
% test_wam_rust_baseline_fixes.pl
%
% Regression harness for phase R-1 of
% docs/proposals/wam_rust_heap_cell_rewrite_design.md (section 12.1): small
% programs where the Rust WAM target differed from SWI-Prolog. Every program
% is compiled through write_wam_rust_project/3, the query is run by the
% compiled binary, and the output is compared with SWI-Prolog's answer for
% the SAME clauses, computed in this process.
%
% One source of truth: bf_clause/2 holds each program's clauses; they are
% asserted into user: (so SWI runs them) and handed to the Rust emitter.
%
% Three checks per program:
%
%   * all solutions, interpreted entry. A failure-driven driver
%       bf_<name> :- Setup, ( Query, write(Out), nl, fail ; true ).
%     is compiled with the program and run by its WAM label. Its stdout and
%     its success flag (the driver cannot fail under SWI) must equal SWI's.
%     Run in emit_mode(interpreter) and emit_mode(functions).
%   * first solution, lowered entry. When emit_mode(functions) lowered the
%     query predicate, its Rust function is called directly with the query
%     arguments; the oracle is once/1 (the lowered tier is first-solution by
%     contract).
%   * the functions-mode crate also builds with
%     `--no-default-features --features decorate_sort` (the non-interned
%     `Sym = String` path), and the lowered checks run on that build too.
%
% catch/3 in this runtime is first-solution by design (out of scope for R-1),
% so a program whose answer depends on re-entering a catch goal is oracled
% against the same clauses with each catch(G,C,R) read as catch(once(G),C,R);
% bf_first_solution_catch/1 lists them and the test also pins SWI's own
% (re-entrant) answer so the documented difference stays visible.
%
%   swipl -q -g run_tests -t halt tests/test_wam_rust_baseline_fixes.pl

:- module(test_wam_rust_baseline_fixes,
          [test_wam_rust_baseline_fixes/0]).

:- use_module(library(plunit)).
:- use_module(library(lists)).
:- use_module(library(apply)).
:- use_module(library(filesex), [make_directory_path/1,
                                 delete_directory_and_contents/1]).
:- use_module(library(process)).
:- use_module(library(readutil)).
:- use_module('../src/unifyweaver/targets/wam_rust_target',
              [write_wam_rust_project/3]).

% ---------------------------------------------------------------------
% Programs. bf_program(Name, Item, Query, Out, Setup)
%   Query/Out : the query whose solutions are printed (write(Out)).
%   Setup     : goal run once by the driver before the query loop.
% ---------------------------------------------------------------------

% R-1f: atom constants and fresh variables in lowered code.
bf_program(lf_atoms,  'R-1f', lf_atoms(b, R),   R,   true).
bf_program(lf_fresh,  'R-1f', lf_fresh(R),      R,   true).
bf_program(lf_nil,    'R-1f', lf_nil([], R),    R,   true).

bf_clause(lf_atoms, (lf_atoms(X, R) :- ( X == a -> R = first ; R = second ))).
bf_clause(lf_fresh, (lf_fresh(R) :- lf_pair(Y, Z), Y = left, R = p(Y, Z))).
bf_clause(lf_fresh, lf_pair(_, right)).
bf_clause(lf_nil,   (lf_nil([], R) :- R = empty)).
bf_clause(lf_nil,   (lf_nil([_|_], R) :- R = cons)).

%% bf_first_solution_catch(?Name)
%  Programs oracled against first-solution catch/3 (see the header).
:- dynamic bf_first_solution_catch/1.

%% bf_dynamic(?PI)
%  Predicates the drivers create at run time with assertz/1. Declared dynamic
%  in user: for SWI; not compiled for Rust (the runtime's dynamic database
%  serves them).
:- dynamic bf_dynamic/1.

% ---------------------------------------------------------------------
% Installation
% ---------------------------------------------------------------------

bf_driver_name(Name, D) :- atom_concat(bf_, Name, D).

bf_clause_head(Clause, F/A) :-
    ( Clause = (H :- _) -> true ; H = Clause ),
    functor(H, F, A).

bf_program_pis(PIs) :-
    findall(PI, (bf_clause(_, C), bf_clause_head(C, PI)), PIs0),
    findall(D/0, (bf_program(N, _, _, _, _), bf_driver_name(N, D)), PIs1),
    append(PIs0, PIs1, PIs2),
    sort(PIs2, PIs).

install_bf_programs :-
    bf_program_pis(PIs),
    forall(member(F/A, PIs),
           ( functor(H, F, A), catch(retractall(user:H), _, true) )),
    forall(bf_dynamic(F/A),
           ( functor(H, F, A),
             catch(retractall(user:H), _, true),
             dynamic(user:F/A) )),
    forall(bf_clause(_, C), assertz(user:C)),
    forall(bf_program(N, _, Q, Out, Setup),
           ( bf_driver_name(N, D),
             assertz(user:(D :- Setup, ( Q, write(Out), nl, fail ; true ))) )).

bf_rust_preds(Preds) :-
    bf_program_pis(PIs),
    findall(user:PI, member(PI, PIs), Preds).

% ---------------------------------------------------------------------
% SWI oracle
% ---------------------------------------------------------------------

bf_reset_dynamic :-
    forall(bf_dynamic(F/A),
           ( functor(H, F, A), catch(retractall(user:H), _, true) )).

%% swi_all(+Name, -Out, -Ok)
swi_all(Name, Out, Ok) :-
    bf_reset_dynamic,
    bf_driver_name(Name, D),
    (   bf_first_solution_catch(Name)
    ->  bf_with_first_solution_catch(Name, user:D, Raw, Ok)
    ;   with_output_to(string(Raw),
                       ( catch(user:D, _, fail) -> Ok = true ; Ok = false ))
    ),
    normalize_output(Raw, Out).

%% swi_reentrant_all(+Name, -Out)
%  SWI's own answer, with its re-entrant catch/3.
swi_reentrant_all(Name, Out) :-
    bf_reset_dynamic,
    bf_driver_name(Name, D),
    with_output_to(string(Raw), ( catch(user:D, _, fail) -> true ; true )),
    normalize_output(Raw, Out).

%% bf_with_first_solution_catch(+Name, +Goal, -Raw, -Ok)
%  Run Goal with every clause of program Name whose body calls catch/3
%  replaced by the catch(once(G),C,R) reading, then restore the clauses.
bf_with_first_solution_catch(Name, Goal, Raw, Ok) :-
    findall(C, bf_clause(Name, C), Cs),
    maplist(bf_once_catch_clause, Cs, Cs1),
    setup_call_cleanup(
        ( forall(member(C, Cs), retract(user:C)),
          forall(member(C, Cs1), assertz(user:C)) ),
        with_output_to(string(Raw),
                       ( catch(Goal, _, fail) -> Ok = true ; Ok = false )),
        ( forall(member(C, Cs1), retract(user:C)),
          forall(member(C, Cs), assertz(user:C)) )).

bf_once_catch_clause((H :- B), (H :- B1)) :- !, bf_once_catch_body(B, B1).
bf_once_catch_clause(C, C).

bf_once_catch_body(V, V) :- var(V), !.
bf_once_catch_body(catch(G, C, R), catch(once(G1), C, R1)) :- !,
    bf_once_catch_body(G, G1), bf_once_catch_body(R, R1).
bf_once_catch_body(T, T1) :-
    compound(T), memberchk(T, [(_,_), (_;_), (_->_), \+(_)]), !,
    T =.. [F|As], maplist(bf_once_catch_body, As, As1), T1 =.. [F|As1].
bf_once_catch_body(T, T).

%% swi_once(+Name, -Out)
swi_once(Name, Out) :-
    bf_reset_dynamic,
    bf_program(Name, _, Q, O, Setup),
    with_output_to(string(Raw),
                   catch(( call(user:Setup), call(user:Q) -> write(O), nl ; true ),
                         _, true)),
    normalize_output(Raw, Out).

normalize_output(Raw, Out) :-
    split_string(Raw, "\n", "\r", Lines0),
    exclude(==(""), Lines0, Lines),
    atomic_list_concat(Lines, "\n", Joined),
    split_string(Joined, " \t", "", Parts),
    atomic_list_concat(Parts, '', Out).

% ---------------------------------------------------------------------
% Rust builds
% ---------------------------------------------------------------------

cargo_available :-
    catch(
        ( process_create(path(cargo), ['--version'],
                         [stdout(null), stderr(null), process(Pid)]),
          process_wait(Pid, exit(0)) ),
        _, fail).

%% bf_mode(?Mode, ?EmitOpts, ?CargoFeatureArgs, ?Dir)
%  functions_nodef shares the functions crate (same generated source) and
%  builds it into a second target dir without default features.
bf_mode(interpreter,     [emit_mode(interpreter)], '',
        'output/rust_wam_baseline_fixes_interpreter').
bf_mode(functions,       [emit_mode(functions)],   '',
        'output/rust_wam_baseline_fixes_functions').
bf_mode(functions_nodef, [emit_mode(functions)],
        '--no-default-features --features decorate_sort',
        'output/rust_wam_baseline_fixes_functions').

bf_target_dir(functions_nodef, 'target_nodef') :- !.
bf_target_dir(_, 'target').

:- dynamic bf_built/1.
:- dynamic bf_generated/1.

bf_generate(Dir, EmitOpts) :-
    (   bf_generated(Dir)
    ->  true
    ;   ( exists_directory(Dir) -> delete_directory_and_contents(Dir) ; true ),
        make_directory_path(Dir),
        install_bf_programs,
        bf_rust_preds(Preds),
        append([module_name('bfprobe'), wam_fallback(true)], EmitOpts, Opts),
        write_wam_rust_project(Preds, Opts, Dir),
        write_bf_driver(Dir),
        assertz(bf_generated(Dir))
    ).

%% bf_ensure_built(+Mode)
bf_ensure_built(Mode) :-
    (   bf_built(Mode)
    ->  true
    ;   bf_mode(Mode, EmitOpts, Feats, Dir),
        bf_generate(Dir, EmitOpts),
        bf_target_dir(Mode, TD),
        format(atom(Cmd),
               'cd ~w && CARGO_TARGET_DIR=~w cargo build ~w --bin bf_probe 2>&1',
               [Dir, TD, Feats]),
        process_create(path(sh), ['-c', Cmd],
                       [stdout(pipe(Out)), stderr(std), process(Pid)]),
        read_string(Out, _, OutStr), close(Out),
        process_wait(Pid, Status),
        (   Status == exit(0)
        ->  assertz(bf_built(Mode))
        ;   format(user_error, "~n[cargo build output, ~w]~n~w~n", [Mode, OutStr]),
            throw(bf_build_failed(Mode, Status))
        )
    ).

bf_bin(Mode, Bin) :-
    bf_mode(Mode, _, _, Dir),
    bf_target_dir(Mode, TD),
    format(atom(Bin), '~w/~w/debug/bf_probe', [Dir, TD]).

%% lowered_entry(+Dir, +Name, -Key)
%  Key is pred_arity when the functions-mode emitter lowered the query
%  predicate of program Name.
lowered_entry(Dir, Name, Key) :-
    bf_program(Name, _, Q, _, _),
    functor(Q, F, A),
    format(atom(Key), '~w_~w', [F, A]),
    atomic_list_concat([Dir, '/src/lib.rs'], LibRs),
    exists_file(LibRs),
    read_file_to_string(LibRs, S, []),
    format(atom(Sym), 'pub fn lowered_~w(', [Key]),
    sub_string(S, _, _, _, Sym).

write_bf_driver(Dir) :-
    atomic_list_concat([Dir, '/src/bin/bf_probe'], BinDir),
    make_directory_path(BinDir),
    atomic_list_concat([BinDir, '/main.rs'], Path),
    bf_driver_source(Head, Tail),
    findall(Arm,
            ( bf_program(N, _, _, _, _),
              lowered_entry(Dir, N, Key),
              format(atom(Arm),
                     '            "~w" => run_lowered(&mut vm, bfprobe::lowered_~w, &rest),~n',
                     [Key, Key]) ),
            Arms0),
    sort(Arms0, Arms),
    atomic_list_concat(Arms, ArmsText),
    setup_call_cleanup(open(Path, write, S),
                       format(S, '~w~w~w', [Head, ArmsText, Tail]),
                       close(S)).

bf_driver_source(
"// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 John William Creighton (s243a)
//
// bf_probe: run a 0-arity driver predicate by its WAM label (program output
// comes from the compiled write/1), or call a LOWERED function directly with
// the query arguments (`--lowered key arg..`, `_` = the output variable).
#![allow(unused_imports, dead_code, unused_mut, unused_variables)]
use bfprobe::state::WamState;
use bfprobe::value::Value;
use bfprobe::{setup_foreign_predicates, shared_wam_program};

const OUT: &str = \"_bf_out\";

fn parse_arg(s: &str) -> Value {
    if s == \"_\" {
        Value::Unbound(OUT.into())
    } else if let Ok(n) = s.parse::<i64>() {
        Value::Integer(n)
    } else if s == \"[]\" {
        Value::Atom(\"[]\".into())
    } else {
        Value::Atom(s.into())
    }
}

fn run_lowered(vm: &mut WamState, f: fn(&mut WamState) -> bool, args: &[String]) {
    vm.reset_query();
    vm.cp = 0;
    for (i, a) in args.iter().enumerate() {
        vm.set_reg(&format!(\"A{}\", i + 1), parse_arg(a));
    }
    let out_reg = args.iter().position(|a| a == \"_\").map(|i| format!(\"A{}\", i + 1));
    if f(vm) {
        match out_reg {
            None => println!(\"yes\"),
            Some(r) => {
                let bound = vm.deref_heap(&Value::Unbound(OUT.into()));
                let out = if bound.is_unbound() {
                    vm.get_reg(&r).map(|v| vm.deref_heap(&v)).unwrap_or(bound)
                } else {
                    bound
                };
                println!(\"{}\", out);
            }
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (code, labels) = shared_wam_program();
    let mut vm = WamState::new(code, labels);
    setup_foreign_predicates(&mut vm);

    if args.get(1).map(|s| s.as_str()) == Some(\"--lowered\") {
        let key = args.get(2).cloned().unwrap_or_default();
        let rest: Vec<String> = args.iter().skip(3).cloned().collect();
        match key.as_str() {
",
"            other => { eprintln!(\"NO_LOWERED {}\", other); std::process::exit(3); }
        }
        return;
    }

    let key = match args.get(1) {
        Some(k) => k.clone(),
        None => { eprintln!(\"usage: bf_probe <pred/arity>\"); std::process::exit(2); }
    };
    let target = match vm.labels.get(&key) {
        Some(pc) => *pc,
        None => { eprintln!(\"NO_LABEL {}\", key); std::process::exit(3); }
    };
    vm.reset_query();
    vm.cp = 0;
    vm.pc = target;
    let ok = vm.run();
    eprintln!(\"run={}\", ok);
}
").

%% bf_run(+Mode, +Args, -Out, -Err)
bf_run(Mode, Args, Out, Err) :-
    bf_bin(Mode, Bin),
    process_create(Bin, Args,
                   [stdout(pipe(O)), stderr(pipe(E)), process(Pid)]),
    read_string(O, _, S1), read_string(E, _, S2),
    close(O), close(E),
    process_wait(Pid, _),
    normalize_output(S1, Out),
    Err = S2.

%% rust_all(+Mode, +Name, -Out, -Ok)
rust_all(Mode, Name, Out, Ok) :-
    bf_driver_name(Name, D),
    format(atom(Key), '~w/0', [D]),
    bf_run(Mode, [Key], Out, Err),
    (   sub_string(Err, _, _, _, "run=true") -> Ok = true
    ;   sub_string(Err, _, _, _, "run=false") -> Ok = false
    ;   Ok = crashed(Err)
    ).

%% rust_once(+Mode, +Name, -Out) is semidet.
%  Fails when the query predicate was not lowered.
rust_once(Mode, Name, Out) :-
    bf_mode(Mode, _, _, Dir),
    lowered_entry(Dir, Name, Key),
    bf_program(Name, _, Q, O, _),
    Q =.. [_|QArgs],
    maplist(bf_arg_text(O), QArgs, ArgTexts),
    bf_run(Mode, ['--lowered', Key|ArgTexts], Out, _).

bf_arg_text(O, A, '_') :- A == O, !.
bf_arg_text(_, A, T) :- format(atom(T), '~w', [A]).

% ---------------------------------------------------------------------
% Checks
% ---------------------------------------------------------------------

bf_all_mode(interpreter).
bf_all_mode(functions).

check_all(Mode, Name) :-
    bf_ensure_built(Mode),
    swi_all(Name, SOut, SOk),
    rust_all(Mode, Name, ROut, ROk),
    (   SOut == ROut, SOk == ROk
    ->  true
    ;   format(user_error,
               'BASELINE DIVERGENCE [~w] ~w~n  swi : ~q run=~w~n  rust: ~q run=~w~n',
               [Mode, Name, SOut, SOk, ROut, ROk]),
        fail
    ).

check_once(Mode, Name) :-
    bf_ensure_built(Mode),
    (   rust_once(Mode, Name, ROut)
    ->  swi_once(Name, SOut),
        (   SOut == ROut
        ->  true
        ;   format(user_error,
                   'LOWERED DIVERGENCE [~w] ~w~n  swi once: ~q~n  rust    : ~q~n',
                   [Mode, Name, SOut, ROut]),
            fail
        )
    ;   format(user_error, '  [~w] ~w: query predicate not lowered (skipped)~n',
               [Mode, Name])
    ).

%% bf_lowered_required(?Name)
%  Programs whose query predicate MUST be lowered in functions mode (so the
%  lowered check cannot silently become a skip).
bf_lowered_required(lf_atoms).
bf_lowered_required(lf_fresh).
bf_lowered_required(lf_nil).

test_wam_rust_baseline_fixes :-
    run_tests(wam_rust_baseline_fixes).

:- begin_tests(wam_rust_baseline_fixes, [condition(cargo_available),
                                        setup(install_bf_programs)]).

test(all_solutions_match_swi,
     [forall(( bf_all_mode(Mode), bf_program(Name, _, _, _, _) ))]) :-
    once(check_all(Mode, Name)).

test(lowered_first_solution_matches_swi,
     [forall(( member(Mode, [functions, functions_nodef]),
               bf_program(Name, _, _, _, _) ))]) :-
    once(check_once(Mode, Name)).

test(lowered_entries_present, [forall(bf_lowered_required(Name))]) :-
    bf_ensure_built(functions),
    bf_mode(functions, _, _, Dir),
    once(lowered_entry(Dir, Name, _)).

test(first_solution_catch_documented,
     [forall(bf_first_solution_catch(Name))]) :-
    install_bf_programs,
    swi_reentrant_all(Name, Reentrant),
    once(swi_all(Name, FirstSol, _)),
    Reentrant \== FirstSol.

:- end_tests(wam_rust_baseline_fixes).
