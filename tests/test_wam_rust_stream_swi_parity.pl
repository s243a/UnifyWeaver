:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (s243a)
%
% test_wam_rust_stream_swi_parity.pl
%
% Stream and lazy-input parity between the Rust WAM target and SWI-Prolog.
% Companion to docs/proposals/wam_rust_streams_lazy_lists_review.md.
%
% Every program below is compiled through write_wam_rust_project/3 (with the
% portable term parser, runtime_parser(compiled)), the crate is built once,
% and each 0-arity driver is run by its WAM label with a fixed stdin. The
% oracle is a SEPARATE swipl process running the same clauses with the same
% stdin, so stdin semantics (consumption across backtracking, end_of_file,
% peek) are SWI's real ones, not an in-process emulation.
%
% Programs whose behaviour the Rust runtime does not implement yet are
% declared blocked(Reason) and are not run; the reason names the gap row in
% the review document. Drivers print operator-free, list-free text because
% write/1 in the Rust runtime does not honour operators (gap C1); that
% cosmetic difference is pinned by its own blocked test (write_ops).
%
%   swipl -q -g run_tests -t halt tests/test_wam_rust_stream_swi_parity.pl

:- module(test_wam_rust_stream_swi_parity,
          [test_wam_rust_stream_swi_parity/0]).

:- set_prolog_flag(encoding, utf8).

:- use_module(library(plunit)).
:- use_module(library(lists)).
:- use_module(library(apply)).
:- use_module(library(filesex), [make_directory_path/1,
                                 delete_directory_and_contents/1]).
:- use_module(library(process)).
:- use_module(library(readutil)).
:- use_module('../src/unifyweaver/targets/wam_rust_target',
              [write_wam_rust_project/3]).

:- discontiguous sp_program/3, sp_clause/2.

sp_dir('output/rust_wam_stream_swi_parity').
sp_file(lines, 'output/rust_wam_stream_swi_parity/in_lines.txt').
sp_file(aaab,  'output/rust_wam_stream_swi_parity/in_aaab.txt').

% ---------------------------------------------------------------------
% sp_program(Name, Stdin, Status)   Status = supported | blocked(Why)
% The driver is sp_<Name>/0; it prints with write/1 + nl/0.
% ---------------------------------------------------------------------

% S1: a term consumed by read/1 stays consumed when the goal fails and a
% later alternative reads again (standard stream semantics).
sp_program(read_retry, "a. b. c.", supported).
sp_clause(read_retry, (sp_read_retry :- ( read(_), fail ; read(Y) ), write(Y), nl)).

% S2: read/1 behind a clause choice point; the second clause must not
% re-see the first term.
sp_program(read_after_fail_cp, "p(1). p(2). p(3).", supported).
sp_clause(read_after_fail_cp, (sp_read_after_fail_cp :- sp_rr(X), write(X), nl)).
sp_clause(read_after_fail_cp, (sp_rr(X) :- read(p(X)), X > 5)).
sp_clause(read_after_fail_cp, (sp_rr(X) :- read(p(X)))).

% S3: read/1 interleaved with a failure-driven loop; end_of_file after
% the last term.
sp_program(read_mixed, "t(1). t(2).", supported).
sp_clause(read_mixed, (sp_read_mixed :- ( sp_rm(X), read(T), write(X), write(' '), write(T), nl, fail ; true ))).
sp_clause(read_mixed, sp_rm(a)).
sp_clause(read_mixed, sp_rm(b)).
sp_clause(read_mixed, sp_rm(c)).

% S4: repeated read/1 at end of input keeps answering end_of_file.
sp_program(read_eof, "a.", supported).
sp_clause(read_eof, (sp_read_eof :- read(X), read(Y), read(Z), write(X), nl, write(Y), nl, write(Z), nl)).

% S5: read_term/2 with variable_names/1; names are bound to themselves so
% the printed term is deterministic.
sp_program(read_term_vn, "f(X,Y,X).", supported).
sp_clause(read_term_vn, (sp_read_term_vn :- read_term(T, [variable_names(Vs)]), sp_bind_names(Vs), T =.. [F|Args], write(F), nl, sp_write_list(Args), sp_write_names(Vs))).
sp_clause(read_term_vn, sp_write_list([])).
sp_clause(read_term_vn, (sp_write_list([X|R]) :- write(X), nl, sp_write_list(R))).
sp_clause(read_term_vn, sp_bind_names([])).
sp_clause(read_term_vn, (sp_bind_names([N=V|R]) :- V = N, sp_bind_names(R))).
sp_clause(read_term_vn, sp_write_names([])).
sp_clause(read_term_vn, (sp_write_names([N=_|R]) :- write(N), nl, sp_write_names(R))).

% S6: output already written is not undone by failure (write/1).
sp_program(write_fail, "", supported).
sp_clause(write_fail, (sp_write_fail :- ( write(hello), fail ; true ), write(world), nl)).

% S7: same for format/2 and put_char/1.
sp_program(format_fail, "", supported).
sp_clause(format_fail, (sp_format_fail :- ( format("~w-~w", [a,b]), put_char(c), fail ; true ), nl)).

% S8: the Rust-native line reader (stream_open/2, read_line/2,
% stream_close/1) consumes a line even when the reading goal fails. SWI
% runs the same program through the shims in swi_shim/1.
sp_program(line_reader, "", supported).
sp_clause(line_reader, (sp_line_reader :- stream_open(F, H), ( read_line(H, _), fail ; read_line(H, L2) ), stream_close(H), write(L2), nl)) :- sp_file(lines, F).

% S9: a DCG that backtracks over an in-memory code list (SWI's own
% translation of  g(N) --> as(N), "b", "\n".  as(0) --> [].
% as(N) --> "a", as(M), {N is M+1}.). This is the grammar the lazy-list
% programs below run over a file.
sp_program(dcg_mem, "", supported).
sp_clause(dcg_mem, (sp_g(N, A, B) :- sp_as(N, A, C), C = [98|D], D = [10|B])).
sp_clause(dcg_mem, (sp_as(0, A, B) :- A = B)).
sp_clause(dcg_mem, (sp_as(A, [97|B], C) :- sp_as(D, B, E), A is D+1, C = E)).
sp_clause(dcg_mem, (sp_dcg_mem :- Cs = [97,97,97,98,10], sp_g(N, Cs, []), write(N), nl)).

% ---- not implemented in the Rust runtime (see the review's gap list) ----

sp_program(peek_stdin, "xy", blocked('G1: get_char/1, peek_char/1 have no Rust arm (BuiltinCall fails)')).
sp_clause(peek_stdin, (sp_peek_stdin :- peek_char(A), get_char(B), get_char(C), get_char(D), peek_char(E), write(A), write(B), write(C), write(D), write(E), nl)).

sp_program(file_lines, "", blocked('G2: open/3, close/1, read_line_to_string/2, at_end_of_stream/1 have no Rust arm')).
sp_clause(file_lines, (sp_file_lines :- open(F, read, S), read_line_to_string(S, L1),
                        ( at_end_of_stream(S) -> E = yes ; E = no ),
                        read_line_to_string(S, L2),
                        ( at_end_of_stream(S) -> E2 = yes ; E2 = no ),
                        close(S), write(L1), write(' '), write(E), write(' '), write(L2), write(' '), write(E2), nl)) :- sp_file(lines, F).

sp_program(stream_pos, "", blocked('G3: stream_property/2, set_stream_position/2, stream_position_data/3 unsupported')).
sp_clause(stream_pos, (sp_stream_pos :- open(F, read, S), stream_property(S, position(P0)),
                        get_char(S, C1), get_char(S, C2), set_stream_position(S, P0),
                        get_char(S, C3), stream_property(S, position(P1)),
                        stream_position_data(char_count, P1, N), close(S),
                        write(C1), write(C2), write(C3), write(N), nl)) :- sp_file(aaab, F).

sp_program(bad_streams, "", blocked('G4: stream builtins fail instead of throwing ISO error terms')).
sp_clause(bad_streams, (sp_bad_streams :- sp_try(get_char(nostream, _)), sp_try(close(foo)),
                        sp_try(open('/nonexistent/x', read, _)), sp_try(peek_char(42, _)),
                        sp_try(get_char(_, _)), sp_try(at_end_of_stream(foo)),
                        sp_try(set_stream_position(user_input, bad)))).
sp_clause(bad_streams, (sp_try(G) :- functor(G, F, A),
                        ( catch(G, error(E, _), (write(caught(F, A, E)), nl)) -> true
                        ; write(failed(F, A)), nl ))).

sp_program(read_syntax, "foo(. ok.", blocked('G5: read/1 fails silently on a syntax error instead of throwing and resyncing')).
sp_clause(read_syntax, (sp_read_syntax :- catch(read(T), error(syntax_error(_), _), T = syntax), write(T), nl, read(T2), write(T2), nl)).

sp_program(eof0, "", blocked('G6: at_end_of_stream/0 is an unresolved goal (fails), so the else branch runs: wrong answer')).
sp_clause(eof0, (sp_eof0 :- ( at_end_of_stream -> write(eof) ; write(more) ), nl)).

sp_program(cur_out, "", blocked('G7: current_output/1, write/2, nl/1 unsupported')).
sp_clause(cur_out, (sp_cur_out :- current_output(S), write(S, hi), nl(S))).

sp_program(wot, "", blocked('G8: with_output_to/2 is registered as a builtin but has no Rust arm')).
sp_clause(wot, (sp_wot :- with_output_to(string(S), ( write(a), fail ; write(b) )), write(S), nl)).

sp_program(pff, "", blocked('L1: phrase_from_file/2 (library(pure_input)) unsupported; needs lazy lists')).
sp_clause(pff, (sp_pff :- phrase_from_file(sp_g(N), F), write(N), nl)) :- sp_file(aaab, F).

sp_program(lazy_list3, "", blocked('L2: lazy_list/3 (library(lazy_lists)) unsupported; needs attributed variables')).
sp_clause(lazy_list3, (sp_lazy_list3 :- lazy_list(sp_nxt, 1, L), L = [A,B,C|_], write(A), write(B), write(C), nl,
                        ( L = [1,2,4|_] -> write(wrong) ; L = [1,2,3,4|_] -> write(replay_ok) ; write(nomatch) ), nl,
                        sp_count(L, 0, Len), write(Len), nl)).
sp_clause(lazy_list3, (sp_nxt(S0, S1, S0) :- S0 < 6, S1 is S0 + 1)).
sp_clause(lazy_list3, sp_count([], N, N)).
sp_clause(lazy_list3, (sp_count([_|T], N0, N) :- N1 is N0+1, sp_count(T, N1, N))).

sp_program(stll, "", blocked('L3: stream_to_lazy_list/2 unsupported; needs lazy lists over a positioned stream')).
sp_clause(stll, (sp_stll :- open(F, read, S), stream_to_lazy_list(S, L),
                        ( L = [0'x|_] -> R1 = x ; L = [0'a,0'a|_] -> R1 = aa ; R1 = none ),
                        ( sp_g(N, L, []) -> true ; N = nog ), close(S), write(R1), write(' '), write(N), nl)) :- sp_file(aaab, F).

sp_program(write_ops, "", blocked('C1: write/1 ignores operators and prints lists with ", " (cosmetic)')).
sp_clause(write_ops, (sp_write_ops :- write(a-b), nl, write([1,2]), nl, write(1+2*3), nl)).

% SWI-only definitions of the Rust-native line-reader builtins.
swi_shim((stream_open(P, S) :- open(P, read, S))).
swi_shim((read_line(S, L) :- read_line_to_string(S, L0), ( L0 == end_of_file -> L = end_of_file ; atom_string(L, L0) ))).
swi_shim((stream_close(S) :- close(S))).

% ---------------------------------------------------------------------
% Plumbing
% ---------------------------------------------------------------------

sp_driver(Name, D) :- atom_concat(sp_, Name, D).

all_clauses(Cs) :- findall(C, sp_clause(_, C), Cs).

clause_head(C, H) :- ( C = (H0 :- _) -> H = H0 ; H = C ).

install_programs :-
    all_clauses(Cs),
    forall(( member(C, Cs), clause_head(C, H), functor(H, F, A) ),
           dynamic(user:F/A)),
    forall(member(C, Cs), assertz(user:C)).

rust_preds(Ps) :-
    all_clauses(Cs),
    findall(user:F/A, ( member(C, Cs), clause_head(C, H), functor(H, F, A) ), Ps0),
    sort(Ps0, Ps).

write_inputs :-
    sp_file(lines, L),
    setup_call_cleanup(open(L, write, S1), format(S1, "one~ntwo~n", []), close(S1)),
    sp_file(aaab, A),
    setup_call_cleanup(open(A, write, S2), format(S2, "aaab~n", []), close(S2)).

swi_file(F) :- sp_dir(D), atom_concat(D, '/sp_swi.pl', F).

write_swi_file :-
    swi_file(Path),
    all_clauses(Cs),
    setup_call_cleanup(open(Path, write, S),
        ( format(S, ":- use_module(library(pure_input)).~n", []),
          format(S, ":- use_module(library(lazy_lists)).~n", []),
          format(S, ":- use_module(library(readutil)).~n", []),
          format(S, ":- style_check(-singleton).~n:- style_check(-discontiguous).~n", []),
          forall(member(C, Cs), portray_clause(S, C)),
          forall(swi_shim(C), portray_clause(S, C)) ),
        close(S)).

probe_main(
"#![allow(unused_imports, dead_code, unused_mut, unused_variables)]
// sp_probe: run a 0-arity driver predicate by its WAM label; program output
// comes from the compiled write/1 family, the success flag goes to stderr.
use spprobe::state::WamState;
use spprobe::{setup_foreign_predicates, shared_wam_program};
fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (code, labels) = shared_wam_program();
    let mut vm = WamState::new(code, labels);
    setup_foreign_predicates(&mut vm);
    let key = args.get(1).cloned().unwrap_or_default();
    let target = match vm.labels.get(&key) {
        Some(pc) => *pc,
        None => { eprintln!(\"NO_LABEL {}\", key); std::process::exit(3); }
    };
    vm.reset_query();
    vm.cp = 0;
    vm.pc = target;
    let ok = vm.run();
    use std::io::Write;
    let _ = std::io::stdout().flush();
    eprintln!(\"run={}\", ok);
}
").

:- dynamic sp_built/0.

ensure_built :-
    (   sp_built
    ->  true
    ;   sp_dir(Dir),
        ( exists_directory(Dir) -> delete_directory_and_contents(Dir) ; true ),
        make_directory_path(Dir),
        write_inputs,
        install_programs,
        write_swi_file,
        rust_preds(Ps),
        write_wam_rust_project(Ps, [module_name(spprobe), wam_fallback(true),
                                    runtime_parser(compiled),
                                    emit_mode(interpreter)], Dir),
        atomic_list_concat([Dir, '/src/bin'], BinDir),
        make_directory_path(BinDir),
        atomic_list_concat([BinDir, '/sp_probe.rs'], Main),
        probe_main(Src),
        setup_call_cleanup(open(Main, write, S), format(S, "~s", [Src]), close(S)),
        format(atom(Cmd), 'cd ~w && cargo build --bin sp_probe 2>&1', [Dir]),
        process_create(path(sh), ['-c', Cmd],
                       [stdout(pipe(Out)), stderr(std), process(Pid)]),
        read_string(Out, _, OutStr), close(Out),
        process_wait(Pid, Status),
        (   Status == exit(0)
        ->  assertz(sp_built)
        ;   format(user_error, "~n[cargo build output]~n~w~n", [OutStr]),
            throw(sp_build_failed(Status))
        )
    ).

run_proc(Exe, Args, Stdin, Out, Err) :-
    process_create(Exe, Args,
                   [stdin(pipe(I)), stdout(pipe(O)), stderr(pipe(E)), process(Pid)]),
    format(I, "~s", [Stdin]), close(I),
    read_string(O, _, Out), read_string(E, _, Err), close(O), close(E),
    process_wait(Pid, _).

%% swi_run(+Name, -Out) : the oracle, a separate swipl on the same clauses.
swi_run(Name, Out) :-
    sp_program(Name, Stdin, _),
    swi_file(F),
    sp_driver(Name, Drv),
    format(atom(G), "(~w -> true ; write(user_error, 'DRIVER_FAILED'))", [Drv]),
    run_proc(path(swipl), ['-q', '-g', G, '-t', 'halt', F], Stdin, Out0, Err),
    (   sub_string(Err, _, _, _, "DRIVER_FAILED")
    ->  throw(swi_driver_failed(Name, Err))
    ;   true
    ),
    normalize(Out0, Out).

%% rust_run(+Name, -Out, -Ok)
rust_run(Name, Out, Ok) :-
    sp_program(Name, Stdin, _),
    sp_dir(Dir), atom_concat(Dir, '/target/debug/sp_probe', Bin),
    sp_driver(Name, Drv), format(atom(Key), '~w/0', [Drv]),
    run_proc(Bin, [Key], Stdin, Out0, Err),
    normalize(Out0, Out),
    (   sub_string(Err, _, _, _, "run=true") -> Ok = true
    ;   sub_string(Err, _, _, _, "run=false") -> Ok = false
    ;   Ok = crashed(Err)
    ).

normalize(S0, S) :- split_string(S0, "", " \n\t", [S]).

%% check(+Name): Rust output and success flag equal SWI's.
check(Name) :-
    swi_run(Name, Want),
    rust_run(Name, Got, Ok),
    (   Ok == true, Got == Want
    ->  true
    ;   format(user_error, "~n[~w] SWI: ~q~n[~w] Rust: ~q (run=~w)~n",
               [Name, Want, Name, Got, Ok]),
        fail
    ).

cargo_ok :-
    catch(( process_create(path(cargo), ['--version'],
                           [stdout(null), stderr(null), process(P)]),
            process_wait(P, exit(0)) ), _, fail).

:- begin_tests(wam_rust_stream_swi_parity).

test(read_retry,          [condition(cargo_ok), setup(ensure_built)]) :- check(read_retry).
test(read_after_fail_cp,  [condition(cargo_ok), setup(ensure_built)]) :- check(read_after_fail_cp).
test(read_mixed,          [condition(cargo_ok), setup(ensure_built)]) :- check(read_mixed).
test(read_eof,            [condition(cargo_ok), setup(ensure_built)]) :- check(read_eof).
test(read_term_vn,        [condition(cargo_ok), setup(ensure_built)]) :- check(read_term_vn).
test(write_fail,          [condition(cargo_ok), setup(ensure_built)]) :- check(write_fail).
test(format_fail,         [condition(cargo_ok), setup(ensure_built)]) :- check(format_fail).
test(line_reader,         [condition(cargo_ok), setup(ensure_built)]) :- check(line_reader).
test(dcg_mem,             [condition(cargo_ok), setup(ensure_built)]) :- check(dcg_mem).

% Blocked until the gap they name is closed; each still compiles into the
% crate (the Rust side fails at run time, it does not fail to build).
test(peek_stdin,  [blocked('G1 get_char/peek_char')])            :- check(peek_stdin).
test(file_lines,  [blocked('G2 open/close/read_line_to_string')]) :- check(file_lines).
test(stream_pos,  [blocked('G3 stream_property/position')])      :- check(stream_pos).
test(bad_streams, [blocked('G4 ISO error terms')])               :- check(bad_streams).
test(read_syntax, [blocked('G5 read/1 syntax error')])           :- check(read_syntax).
test(eof0,        [blocked('G6 at_end_of_stream/0')])            :- check(eof0).
test(cur_out,     [blocked('G7 current_output/write/2')])        :- check(cur_out).
test(wot,         [blocked('G8 with_output_to/2')])              :- check(wot).
test(pff,         [blocked('L1 phrase_from_file/2')])            :- check(pff).
test(lazy_list3,  [blocked('L2 lazy_list/3')])                   :- check(lazy_list3).
test(stll,        [blocked('L3 stream_to_lazy_list/2')])         :- check(stll).
test(write_ops,   [blocked('C1 write/1 operators')])             :- check(write_ops).

:- end_tests(wam_rust_stream_swi_parity).

test_wam_rust_stream_swi_parity :-
    run_tests(wam_rust_stream_swi_parity).
