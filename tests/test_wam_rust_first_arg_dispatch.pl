:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (s243a)
%
% test_wam_rust_first_arg_dispatch.pl
%
% Regression harness for D138: first-argument (and second-argument)
% indexing in the Rust WAM target when several clauses share an index key.
%
% The shared compiler (wam_target.pl) indexes a multi-clause predicate with
% a switch_on_constant / switch_on_structure / switch_on_term header. A key
% matched by exactly one clause jumps to that clause's body label; a key
% matched by SEVERAL clauses (none of them clause 1) jumps to a dedicated
% dispatch chain
%
%     L_p_2_const_k_dispatch:
%         try   L_p_2_2_body
%         retry L_p_2_4_body
%         trust L_p_2_5_body
%
% (standard WAM try/retry/trust: push a choice point whose alternative is
% the NEXT chain instruction, then jump to the body). The mixed
% constant/variable form (switch_on_constant_fallthrough) and every
% second-argument form instead jump to a clause's linear-chain entry
% L_p_2_<I> (a retry_me_else / trust_me line) meaning "clauses I..N".
%
% Programs are compiled through write_wam_rust_project/3 and run by the
% compiled binary; output is compared with SWI-Prolog's answer for the
% SAME clauses, computed in this process (same harness shape as
% tests/test_wam_rust_baseline_fixes.pl):
%
%   * all solutions, interpreted entry, emit_mode(interpreter) and
%     emit_mode(functions): a failure-driven driver
%         fd_<name> :- ( Query, write(Out), nl, fail ; true ).
%     is run by its WAM label; stdout and the success flag must equal SWI's.
%   * first solution, lowered entry: when emit_mode(functions) lowered the
%     query predicate and the query's arguments are atomic (plus the output
%     variable), its Rust function is called directly; oracle once/1.
%
%   swipl -q -g run_tests -t halt tests/test_wam_rust_first_arg_dispatch.pl

:- module(test_wam_rust_first_arg_dispatch,
          [test_wam_rust_first_arg_dispatch/0]).

:- use_module(library(plunit)).
:- use_module(library(lists)).
:- use_module(library(apply)).
:- use_module(library(aggregate)).
:- use_module(library(filesex), [make_directory_path/1,
                                 delete_directory_and_contents/1]).
:- use_module(library(process)).
:- use_module(library(readutil)).
:- use_module('../src/unifyweaver/targets/wam_rust_target',
              [write_wam_rust_project/3]).

% ---------------------------------------------------------------------
% Programs. fd_program(Name, Query, Out)
% ---------------------------------------------------------------------

:- discontiguous fd_program/3, fd_clause/2.

% --- atom keys: duplicate key in the middle / at the end ---------------
fd_program(dup_mid,        dp(2, R),  R).
fd_program(dup_mid_unique, dp(1, R),  R).
fd_program(dup_mid_miss,   dp(9, R),  R).
fd_program(dup_mid_open,   dp(K, R),  o(K, R)).
fd_clause(dp, dp(1, p)).
fd_clause(dp, dp(2, q)).
fd_clause(dp, dp(2, r)).

% Distinct shared keys must not alias after dispatch-label sanitization.
fd_program(label_collision_dash, dlabel('a-b', R), R).
fd_program(label_collision_under, dlabel(a_b, R), R).
fd_clause(dlabel, dlabel(a, 0)).
fd_clause(dlabel, dlabel('a-b', 1)).
fd_clause(dlabel, dlabel(a_b, 2)).
fd_clause(dlabel, dlabel('a-b', 3)).
fd_clause(dlabel, dlabel(a_b, 4)).

fd_program(struct_label_dash, dslabel(R), R).
fd_program(struct_label_under, dslabelu(R), R).
fd_clause(dslabel, dsl(a(x), 0)).
fd_clause(dslabel, dsl('a-b'(x), 1)).
fd_clause(dslabel, dsl(a_b(x), 2)).
fd_clause(dslabel, dsl('a-b'(y), 3)).
fd_clause(dslabel, dsl(a_b(y), 4)).
fd_clause(dslabel, (dslabel(R) :- member(T, ['a-b'(x),'a-b'(y)]), dsl(T,R))).
fd_clause(dslabel, (dslabelu(R) :- member(T, [a_b(x),a_b(y)]), dsl(T,R))).

fd_program(dup_last,       da(b, R),  R).
fd_program(dup_last_c,     da(c, R),  R).
fd_clause(da, da(a, 1)).
fd_clause(da, da(b, 2)).
fd_clause(da, da(c, 3)).
fd_clause(da, da(b, 4)).

% three clauses share a key: try / retry / trust
fd_program(dup_three,      d3(k, R),  R).
fd_program(dup_three_m,    d3(m, R),  R).
fd_clause(d3, d3(a, 0)).
fd_clause(d3, d3(k, 1)).
fd_clause(d3, d3(m, x)).
fd_clause(d3, d3(k, 2)).
fd_clause(d3, d3(k, 3)).

% --- the duplicate group contains clause 1 ("default": linear chain) ---
fd_program(dup_first,      df(a, R),  R).
fd_program(dup_first_b,    df(b, R),  R).
fd_clause(df, df(a, 1)).
fd_clause(df, df(a, 2)).
fd_clause(df, df(b, 3)).

% --- every clause shares one key ----------------------------------------
fd_program(all_same,       dall(k, R), R).
fd_program(all_same_miss,  dall(z, R), R).
fd_clause(dall, dall(k, 1)).
fd_clause(dall, dall(k, 2)).
fd_clause(dall, dall(k, 3)).

% --- integer / float keys -------------------------------------------------
fd_program(int_keys,       dint(3, R), R).
fd_program(int_keys_2,     dint(2, R), R).
fd_clause(dint, dint(1, a)).
fd_clause(dint, dint(2, b)).
fd_clause(dint, dint(3, c)).
fd_clause(dint, dint(2, d)).
fd_clause(dint, dint(3, e)).

fd_program(float_keys,     dfl(2.5, R), R).
fd_clause(dfl, dfl(1.5, a)).
fd_clause(dfl, dfl(2.5, b)).
fd_clause(dfl, dfl(2.5, c)).

% 1 and 1.0 are different keys
fd_program(num_keys_int,   dnum(1, R),   R).
fd_program(num_keys_float, dnum(1.0, R), R).
fd_clause(dnum, dnum(2, x)).
fd_clause(dnum, dnum(1, a)).
fd_clause(dnum, dnum(1.0, b)).
fd_clause(dnum, dnum(1, c)).

% negative integers, the atoms true/false, quoted atoms, [] as keys
fd_program(neg_keys,       dneg(-1, R), R).
fd_clause(dneg, dneg(0, z)).
fd_clause(dneg, dneg(-1, a)).
fd_clause(dneg, dneg(1, b)).
fd_clause(dneg, dneg(-1, c)).

fd_program(bool_keys,      dbool(true, R),  R).
fd_program(bool_keys_f,    dbool(false, R), R).
fd_clause(dbool, dbool(x, 0)).
fd_clause(dbool, dbool(true, 1)).
fd_clause(dbool, dbool(false, 2)).
fd_clause(dbool, dbool(true, 3)).

fd_program(quoted_keys,    dqa('Hello World', R), R).
fd_program(quoted_colon,   dqa('a:b', R),         R).
fd_program(quoted_upper,   dqa('Up', R),          R).
fd_program(nil_key,        dqa([], R),            R).
fd_clause(dqa, dqa(x, 0)).
fd_clause(dqa, dqa('Hello World', 1)).
fd_clause(dqa, dqa('a:b', 2)).
fd_clause(dqa, dqa('Hello World', 3)).
fd_clause(dqa, dqa('Up', 4)).
fd_clause(dqa, dqa([], 5)).
fd_clause(dqa, dqa('a:b', 6)).

% [] reached as the end of a list (an empty list VALUE, not the atom)
fd_program(nil_tail,       dnil(R), R).
fd_clause(dnil, (dnil(R) :- X = [a], X = [_|T], dqa(T, R))).

% a quoted functor name in a structure key
fd_program(struct_quoted,  dsq('Foo'(X), R), o(X, R)).
fd_clause(dsq, dsq(g(0), z)).
fd_clause(dsq, dsq('Foo'(1), a)).
fd_clause(dsq, dsq(g(2), b)).
fd_clause(dsq, dsq('Foo'(3), c)).

% --- structure keys (switch_on_structure) --------------------------------
fd_program(struct_keys,    dst(f(X), R), o(X, R)).
fd_program(struct_keys_g,  dst(g(X), R), o(X, R)).
fd_clause(dst, dst(g(0), z)).
fd_clause(dst, dst(f(1), a)).
fd_clause(dst, dst(g(2), b)).
fd_clause(dst, dst(f(3), c)).

% the structure argument arrives as a term value (from findall's copy, a
% fact's list element), not only as a heap cell built by put_structure
fd_program(struct_from_list, dsl(R), R).
fd_clause(dst, (dsl(o(X, R)) :- member(T, [f(1), g(2)]), dst(T, R), T =.. [_, X])).
fd_program(struct_from_findall, dsf_all(R), R).
fd_clause(dst, (dsf_all(o(X, R)) :- findall(T, member(T, [f(_), g(_)]), Ts),
                                     member(T, Ts), dst(T, R), T =.. [_, X])).
fd_program(struct_from_univ, dsu(R), R).
fd_clause(dst, (dsu(o(X, R)) :- T =.. [f, X], dst(T, R))).

% same name, different arity: f/1 and f/2 are different keys
fd_program(struct_arity,   dsa(f(X), R),    o(X, R)).
fd_program(struct_arity_2, dsa(f(X, Y), R), o(X, Y, R)).
fd_clause(dsa, dsa(h(0), z)).
fd_clause(dsa, dsa(f(1), a)).
fd_clause(dsa, dsa(f(1, 2), b)).
fd_clause(dsa, dsa(f(3), c)).

% --- list / mixed first args (switch_on_term) ----------------------------
fd_program(list_keys,      dl([_|_], R), R).
fd_program(list_keys_nil,  dl([], R),    R).
fd_clause(dl, dl([], e)).
fd_clause(dl, dl([a], one)).
fd_clause(dl, dl([b, c], two)).

fd_program(list_chain,     dl2([Y], R), o(Y, R)).
fd_clause(dl2, dl2(x, 0)).
fd_clause(dl2, dl2([a], one)).
fd_clause(dl2, dl2([b], two)).

fd_program(mixed_atom,     dm(a, R),    R).
fd_program(mixed_struct,   dm(f(Z), R), o(Z, R)).
fd_clause(dm, dm(a, 1)).
fd_clause(dm, dm(f(x), 2)).
fd_clause(dm, dm(a, 3)).
fd_clause(dm, dm(f(y), 4)).
fd_clause(dm, dm(b, 5)).

% --- indexed and variable-first-arg clauses interleaved -------------------
fd_program(var_mix_b,      dv(b, R), R).
fd_program(var_mix_c,      dv(c, R), R).
fd_program(var_mix_a,      dv(a, R), R).
fd_clause(dv, dv(a, 1)).
fd_clause(dv, dv(b, 2)).
fd_clause(dv, dv(X, v(X))).
fd_clause(dv, dv(b, 3)).
fd_clause(dv, dv(c, 4)).

% a duplicate key inside the indexed prefix
fd_program(var_mix_dup,    dv2(b, R), R).
fd_clause(dv2, dv2(a, 1)).
fd_clause(dv2, dv2(b, 2)).
fd_clause(dv2, dv2(b, 3)).
fd_clause(dv2, dv2(_, v)).
fd_clause(dv2, dv2(b, 4)).

% the key of clause 1 repeats inside the indexed prefix
fd_program(var_mix_first,  dv3(a, R), R).
fd_clause(dv3, dv3(a, 1)).
fd_clause(dv3, dv3(b, 2)).
fd_clause(dv3, dv3(a, 3)).
fd_clause(dv3, dv3(_, v)).

% first clause has a variable first arg (no A1 index at all)
fd_program(var_first,      dvf(a, R), R).
fd_clause(dvf, dvf(X, v(X))).
fd_clause(dvf, dvf(a, 1)).
fd_clause(dvf, dvf(b, 2)).
fd_clause(dvf, dvf(a, 3)).

% --- second-argument indexing (all-variable first args) ------------------
fd_program(a2_dup,         ds2(x, b, R), R).
fd_program(a2_dup_a,       ds2(x, a, R), R).
fd_clause(ds2, ds2(_, a, 1)).
fd_clause(ds2, ds2(_, b, 2)).
fd_clause(ds2, ds2(_, a, 3)).
fd_clause(ds2, ds2(_, c, 4)).
fd_clause(ds2, ds2(_, b, 5)).

% under a caller choice point: the linear-chain targets must not rewrite
% (retry_me_else) or pop (trust_me) the caller's choice point
fd_program(a2_outer,       dw2(R), R).
fd_clause(ds2, (dw2(R) :- member(K, [b, a, c]), ds2(x, K, R))).
fd_program(a2_last_outer,  dw3(R), R).
fd_clause(dw3, ds3(_, a, 1)).
fd_clause(dw3, ds3(_, b, 2)).
fd_clause(dw3, ds3(_, z, 3)).
fd_clause(dw3, (dw3(R) :- member(K, [z, b]), ds3(x, K, R))).
fd_program(var_mix_outer,  dwv(R), R).
fd_clause(dv, (dwv(R) :- member(K, [b, c]), dv(K, R))).

fd_program(a2_fallthrough, dsf(x, b, R), R).
fd_clause(dsf, dsf(_, a, 1)).
fd_clause(dsf, dsf(_, b, 2)).
fd_clause(dsf, dsf(_, _, v)).
fd_clause(dsf, dsf(_, b, 3)).

% --- cut inside a dispatched clause --------------------------------------
fd_program(cut_neck,       dc(2, R), R).
fd_program(cut_neck_outer, dcw(R),   R).
fd_clause(dc, dc(1, a)).
fd_clause(dc, (dc(2, b) :- !)).
fd_clause(dc, dc(2, c)).
fd_clause(dc, dc(3, d)).
fd_clause(dc, (dcw(R) :- member(K, [2, 3]), dc(K, R))).

fd_program(cut_after_call, dcc(2, R), R).
fd_clause(dcc, dcc(1, a)).
fd_clause(dcc, (dcc(2, R) :- dcm(R), !)).
fd_clause(dcc, dcc(2, z)).
fd_clause(dcc, dcm(m1)).
fd_clause(dcc, dcm(m2)).

% the cut is in the RETRY position of the chain
fd_program(cut_retry,       dcr(2, R), R).
fd_program(cut_retry_outer, dcrw(R),   R).
fd_clause(dcr, dcr(1, a)).
fd_clause(dcr, dcr(2, b)).
fd_clause(dcr, (dcr(2, c) :- !)).
fd_clause(dcr, dcr(2, d)).
fd_clause(dcr, dcr(3, e)).
fd_clause(dcr, (dcrw(R) :- member(K, [2, 3]), dcr(K, R))).

% --- dispatch inside findall / aggregate_all ------------------------------
fd_program(in_findall,     dfa(L), L).
fd_clause(dfa, (dfa(L) :- findall(R, da(b, R), L))).
fd_program(in_count,       dagg(N), N).
fd_clause(dagg, (dagg(N) :- aggregate_all(count, d3(k, _), N))).
fd_program(in_bag,         dbag(L), L).
fd_clause(dbag, (dbag(L) :- aggregate_all(bag(R), dint(3, R), L))).
fd_program(in_nested,      dnf(L), L).
fd_clause(dnf, (dnf(L) :- findall(o(K, Rs), ( member(K, [a, b, c]),
                                          findall(R, da(K, R), Rs) ), L))).

% --- deep backtracking into a dispatch chain from a later goal -----------
fd_program(bt_filter,      dbt(R), R).
fd_clause(dbt, (dbt(R) :- da(b, R), R > 3)).
fd_program(bt_pairs,       dbt2(P), P).
fd_clause(dbt2, (dbt2(o(X, Y)) :- dp(2, X), dp(2, Y))).
fd_program(bt_last,        dbt3(R), R).
fd_clause(dbt3, (dbt3(R) :- d3(k, R), dbt_ok(R))).
fd_clause(dbt3, dbt_ok(3)).
fd_program(bt_deep,        dbt4(P), P).
fd_clause(dbt4, (dbt4(o(R, S)) :- d3(k, R), dbt_mul(R, S), S >= 20)).
fd_clause(dbt4, (dbt_mul(R, S) :- S is R * 10)).

% dispatched clauses with rule bodies that themselves dispatch
fd_program(rule_bodies,    drb(2, R), R).
fd_clause(drb, (drb(1, X) :- X = one)).
fd_clause(drb, (drb(2, X) :- dp(2, X))).
fd_clause(drb, (drb(2, X) :- da(b, X))).

% recursion through a duplicate-key edge relation (the extra depth
% argument keeps it off the transitive-closure kernel, so the WAM
% dispatch is what runs)
fd_program(reach,          dreach(a, Y, 0), Y).
fd_clause(dreach, dedge(a, b)).
fd_clause(dreach, dedge(b, c)).
fd_clause(dreach, dedge(b, d)).
fd_clause(dreach, dedge(c, e)).
fd_clause(dreach, dedge(d, e)).
fd_clause(dreach, (dreach(X, Y, _) :- dedge(X, Y))).
fd_clause(dreach, (dreach(X, Y, D) :- D < 5, dedge(X, Z), D1 is D + 1,
                                      dreach(Z, Y, D1))).

% --- more than 8 clauses ---------------------------------------------------
fd_program(big_dup,        big(b, R),  R).
fd_program(big_unique,     big(k9, R), R).
fd_clause(big, big(k0, 0)).
fd_clause(big, big(k1, 1)).
fd_clause(big, big(b, 2)).
fd_clause(big, big(k3, 3)).
fd_clause(big, big(b, 4)).
fd_clause(big, big(k5, 5)).
fd_clause(big, big(k6, 6)).
fd_clause(big, big(b, 7)).
fd_clause(big, big(k8, 8)).
fd_clause(big, big(k9, 9)).
fd_clause(big, big(b, 10)).
fd_clause(big, big(k11, 11)).

fd_program(big_int,        bigi(7, R), R).
fd_clause(bigi, bigi(0, a)).
fd_clause(bigi, bigi(7, b)).
fd_clause(bigi, bigi(1, c)).
fd_clause(bigi, bigi(2, d)).
fd_clause(bigi, bigi(3, e)).
fd_clause(bigi, bigi(7, f)).
fd_clause(bigi, bigi(4, g)).
fd_clause(bigi, bigi(5, h)).
fd_clause(bigi, bigi(6, i)).
fd_clause(bigi, bigi(7, j)).

% ---------------------------------------------------------------------
% Installation
% ---------------------------------------------------------------------

fd_driver_name(Name, D) :- atom_concat(fd_, Name, D).

fd_clause_head(Clause, F/A) :-
    ( Clause = (H :- _) -> true ; H = Clause ),
    functor(H, F, A).

fd_program_pis(PIs) :-
    findall(PI, (fd_clause(_, C), fd_clause_head(C, PI)), PIs0),
    findall(D/0, (fd_program(N, _, _), fd_driver_name(N, D)), PIs1),
    append(PIs0, PIs1, PIs2),
    sort(PIs2, PIs).

install_fd_programs :-
    fd_program_pis(PIs),
    forall(member(F/A, PIs),
           ( functor(H, F, A), catch(retractall(user:H), _, true) )),
    forall(fd_clause(_, C), assertz(user:C)),
    forall(fd_program(N, Q, Out),
           ( fd_driver_name(N, D),
             assertz(user:(D :- ( Q, write(Out), nl, fail ; true ))) )).

fd_rust_preds(Preds) :-
    fd_program_pis(PIs),
    findall(user:PI, member(PI, PIs), Preds).

% ---------------------------------------------------------------------
% SWI oracle
% ---------------------------------------------------------------------

swi_all(Name, Out, Ok) :-
    fd_driver_name(Name, D),
    with_output_to(string(Raw),
                   ( catch(user:D, _, fail) -> Ok = true ; Ok = false )),
    normalize_output(Raw, Out).

swi_once(Name, Out) :-
    fd_program(Name, Q, O),
    with_output_to(string(Raw),
                   catch(( call(user:Q) -> write(O), nl ; true ), _, true)),
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

fd_mode(interpreter, [emit_mode(interpreter)],
        'output/rust_wam_first_arg_dispatch_interpreter').
fd_mode(functions,   [emit_mode(functions)],
        'output/rust_wam_first_arg_dispatch_functions').

:- dynamic fd_built/1.

fd_generate(Dir, EmitOpts) :-
    ( exists_directory(Dir) -> delete_directory_and_contents(Dir) ; true ),
    make_directory_path(Dir),
    install_fd_programs,
    fd_rust_preds(Preds),
    append([module_name('fdprobe'), wam_fallback(true)], EmitOpts, Opts),
    write_wam_rust_project(Preds, Opts, Dir),
    write_fd_driver(Dir).

%  Deterministic: a failing check must not backtrack into the project
%  writer (which would regenerate and rebuild the crate).
fd_ensure_built(Mode) :-
    (   fd_built(Mode)
    ->  true
    ;   fd_mode(Mode, EmitOpts, Dir),
        once(fd_generate(Dir, EmitOpts)),
        format(atom(Cmd),
               'cd ~w && cargo build --bin fd_probe 2>&1', [Dir]),
        process_create(path(sh), ['-c', Cmd],
                       [stdout(pipe(Out)), stderr(std), process(Pid)]),
        read_string(Out, _, OutStr), close(Out),
        process_wait(Pid, Status),
        (   Status == exit(0)
        ->  assertz(fd_built(Mode))
        ;   format(user_error, "~n[cargo build output, ~w]~n~w~n", [Mode, OutStr]),
            throw(fd_build_failed(Mode, Status))
        )
    ),
    !.

fd_bin(Mode, Bin) :-
    fd_mode(Mode, _, Dir),
    format(atom(Bin), '~w/target/debug/fd_probe', [Dir]).

%% lowered_entry(+Dir, +Name, -Key)
lowered_entry(Dir, Name, Key) :-
    fd_program(Name, Q, _),
    functor(Q, F, A),
    format(atom(Key), '~w_~w', [F, A]),
    atomic_list_concat([Dir, '/src/lib.rs'], LibRs),
    exists_file(LibRs),
    read_file_to_string(LibRs, S, []),
    format(atom(Sym), 'pub fn lowered_~w(', [Key]),
    sub_string(S, _, _, _, Sym).

write_fd_driver(Dir) :-
    atomic_list_concat([Dir, '/src/bin/fd_probe'], BinDir),
    make_directory_path(BinDir),
    atomic_list_concat([BinDir, '/main.rs'], Path),
    fd_driver_source(Head, Tail),
    findall(Arm,
            ( fd_program(N, _, _),
              lowered_entry(Dir, N, Key),
              format(atom(Arm),
                     '            "~w" => run_lowered(&mut vm, fdprobe::lowered_~w, &rest),~n',
                     [Key, Key]) ),
            Arms0),
    sort(Arms0, Arms),
    atomic_list_concat(Arms, ArmsText),
    setup_call_cleanup(open(Path, write, S),
                       format(S, '~w~w~w', [Head, ArmsText, Tail]),
                       close(S)).

fd_driver_source(
"// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 John William Creighton (s243a)
//
// fd_probe: run a 0-arity driver predicate by its WAM label (program output
// comes from the compiled write/1), or call a LOWERED function directly with
// the query arguments (`--lowered key arg..`, `_` = the output variable).
#![allow(unused_imports, dead_code, unused_mut, unused_variables)]
use fdprobe::state::WamState;
use fdprobe::value::Value;
use fdprobe::{setup_foreign_predicates, shared_wam_program};

const OUT: &str = \"_fd_out\";

fn parse_arg(s: &str) -> Value {
    if s == \"_\" {
        Value::Unbound(OUT.into())
    } else if let Ok(n) = s.parse::<i64>() {
        Value::Integer(n)
    } else if let Ok(f) = s.parse::<f64>() {
        Value::Float(f)
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
        None => { eprintln!(\"usage: fd_probe <pred/arity>\"); std::process::exit(2); }
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

%  Each run is capped at 60 s: before the fix some programs looped forever
%  (a dispatch target rewrote the caller's choice point). A run that is cut
%  off prints no `run=` line and is reported as crashed.
fd_run(Mode, Args, Out, Err) :-
    fd_bin(Mode, Bin),
    process_create(path(timeout), ['60', Bin|Args],
                   [stdout(pipe(O)), stderr(pipe(E)), process(Pid)]),
    read_string(O, _, S1), read_string(E, _, S2),
    close(O), close(E),
    process_wait(Pid, _),
    normalize_output(S1, Out),
    Err = S2.

rust_all(Mode, Name, Out, Ok) :-
    fd_driver_name(Name, D),
    format(atom(Key), '~w/0', [D]),
    fd_run(Mode, [Key], Out, Err),
    (   sub_string(Err, _, _, _, "run=true") -> Ok = true
    ;   sub_string(Err, _, _, _, "run=false") -> Ok = false
    ;   Ok = crashed(Err)
    ).

%% lowered_callable(+Name, -ArgTexts) is semidet.
%  The direct lowered call passes atomic arguments plus ONE output
%  variable, so only queries of that shape are checked this way.
lowered_callable(Name, ArgTexts) :-
    fd_program(Name, Q, O),
    var(O),
    Q =.. [_|QArgs],
    maplist(fd_arg_text(O), QArgs, ArgTexts).

fd_arg_text(O, A, '_') :- A == O, !.
fd_arg_text(_, A, T) :- atomic(A), format(atom(T), '~w', [A]).

rust_once(Mode, Name, Out) :-
    fd_mode(Mode, _, Dir),
    lowered_entry(Dir, Name, Key),
    lowered_callable(Name, ArgTexts),
    fd_run(Mode, ['--lowered', Key|ArgTexts], Out, _).

% ---------------------------------------------------------------------
% Checks
% ---------------------------------------------------------------------

check_all(Mode, Name) :-
    fd_ensure_built(Mode),
    swi_all(Name, SOut, SOk),
    rust_all(Mode, Name, ROut, ROk),
    (   SOut == ROut, SOk == ROk
    ->  true
    ;   format(user_error,
               'DISPATCH DIVERGENCE [~w] ~w~n  swi : ~q run=~w~n  rust: ~q run=~w~n',
               [Mode, Name, SOut, SOk, ROut, ROk]),
        fail
    ).

check_once(Mode, Name) :-
    fd_ensure_built(Mode),
    (   fd_lowered_out_of_scope(Name, Why)
    ->  format(user_error, '  [~w] ~w: lowered entry not checked (~w)~n',
               [Mode, Name, Why])
    ;   rust_once(Mode, Name, ROut)
    ->  swi_once(Name, SOut),
        (   SOut == ROut
        ->  true
        ;   format(user_error,
                   'LOWERED DIVERGENCE [~w] ~w~n  swi once: ~q~n  rust    : ~q~n',
                   [Mode, Name, SOut, ROut]),
            fail
        )
    ;   true    % query predicate not lowered, or not callable directly
    ).

%% fd_lowered_out_of_scope(?Name, ?Why)
%  Programs whose DIRECT lowered entry differs from once/1 for a reason that
%  has nothing to do with indexing (the lowered parser drops switch_on_* and
%  the dispatch chains altogether). Their interpreted entries are checked in
%  both modes, which is what an interpreted caller reaches.
%   * The lowered emitter drops begin_aggregate/end_aggregate (see
%     test_wam_rust_baseline_fixes.pl, lowered_aggregates_unsupported).
%   * A lowered body runs each user call to its first solution inside a
%     raised backtrack floor and cannot backtrack into it; the interpreted
%     entry's lowered_dispatch guard declines such a call. Called directly,
%     `p(R) :- q(R), R > 3` sees only q's first solution.
%   * The lowered emitter splits WAM lines on blanks and commas, so
%     `get_constant 'Hello World', A1` becomes four tokens and is dropped;
%     the clause then matches any first argument.
fd_lowered_out_of_scope(in_findall,    lowered_aggregates_unsupported).
fd_lowered_out_of_scope(in_count,      lowered_aggregates_unsupported).
fd_lowered_out_of_scope(in_bag,        lowered_aggregates_unsupported).
fd_lowered_out_of_scope(in_nested,     lowered_aggregates_unsupported).
fd_lowered_out_of_scope(struct_from_findall, lowered_aggregates_unsupported).
fd_lowered_out_of_scope(bt_filter,     lowered_call_is_first_solution).
fd_lowered_out_of_scope(bt_last,       lowered_call_is_first_solution).
fd_lowered_out_of_scope(bt_deep,       lowered_call_is_first_solution).
fd_lowered_out_of_scope(quoted_keys,   lowered_parser_splits_quoted_atoms).
fd_lowered_out_of_scope(quoted_colon,  lowered_parser_splits_quoted_atoms).
fd_lowered_out_of_scope(quoted_upper,  lowered_parser_splits_quoted_atoms).
fd_lowered_out_of_scope(nil_key,       lowered_parser_splits_quoted_atoms).

fd_all_mode(interpreter).
fd_all_mode(functions).

test_wam_rust_first_arg_dispatch :-
    run_tests(wam_rust_first_arg_dispatch).

:- begin_tests(wam_rust_first_arg_dispatch,
               [condition(cargo_available), setup(install_fd_programs)]).

test(all_solutions_match_swi,
     [forall(( fd_all_mode(Mode), fd_program(Name, _, _) ))]) :-
    once(check_all(Mode, Name)).

test(lowered_first_solution_matches_swi,
     [forall(fd_program(Name, _, _))]) :-
    once(check_once(functions, Name)).

% The dispatch chains this suite targets really are emitted (so the
% programs above keep exercising try/retry/trust and the linear-chain
% entries, not just the fall-through path).
test(dispatch_shapes_emitted) :-
    fd_mode(interpreter, _, Dir),
    fd_ensure_built(interpreter),
    atomic_list_concat([Dir, '/src/lib.rs'], LibRs),
    read_file_to_string(LibRs, S, []),
    % R1: chain and switch targets are emitted as PCs (TryPc(<pc of the
    % body label>), SwitchOnStructurePc, ...).
    fd_label_pc(S, "L_d3_2_2_body", P2),
    fd_label_pc(S, "L_d3_2_4_body", P4),
    fd_label_pc(S, "L_d3_2_5_body", P5),
    format(string(T2), "Instruction::TryPc(~w)", [P2]),
    format(string(T4), "Instruction::RetryPc(~w)", [P4]),
    format(string(T5), "Instruction::TrustPc(~w)", [P5]),
    forall(member(Needle, [T2, T4, T5,
                           "Instruction::SwitchOnStructurePc(",
                           "Instruction::SwitchOnConstantFallthroughPc(",
                           "Instruction::SwitchOnConstantA2Pc("]),
           (   sub_string(S, _, _, _, Needle)
           ->  true
           ;   format(user_error, 'missing in generated code: ~w~n', [Needle]),
               fail
           )).

:- end_tests(wam_rust_first_arg_dispatch).

%% fd_label_pc(+LibRs, +Label, -PC): the PC `labels.insert` gives Label.
fd_label_pc(S, Label, PC) :-
    format(string(Pre), "labels.insert(\"~w\".to_string(), ", [Label]),
    sub_string(S, B, L, _, Pre),
    E is B + L,
    sub_string(S, E, 12, _, Tail),
    split_string(Tail, ")", "", [PCS|_]),
    number_string(PC, PCS),
    !.
