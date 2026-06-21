% test_wam_rust_embedded_input_threading.pl
%
% T7 route-2 input threading — codegen-level checks (no cargo needed).
% Verifies that an embedded aggregate reading enclosing-clause inputs lowers to a
% ParAggregate instruction whose helpers take the inputs as LEADING params and
% whose instruction carries the container input registers, and that an input-less
% embedded aggregate stays at the bare 4-field form.

:- use_module('../src/unifyweaver/targets/wam_rust_target', [write_wam_rust_project/3]).

:- dynamic eg_link/2, il_fact/1.
eg_guard(_).
eg_link(1, 3). eg_link(2, 4).
eg_dn(0, []).
eg_dn(N, [N|T]) :- N > 0, M is N - 1, eg_dn(M, T).
eg_p(N, R) :- eg_guard(N), findall(D, (eg_link(N, X), eg_dn(X, D)), R).

% input-less embedded aggregate (no enclosing-clause var in the inner goal)
il_guard(_).
il_fact(1). il_fact(2). il_fact(3).
il_dn(0, []).
il_dn(N, [N|T]) :- N > 0, M is N - 1, il_dn(M, T).
il_p(R) :- il_guard(g), findall(D, (il_fact(X), il_dn(X, D)), R).

safe_rmdir(Dir) :- ( exists_directory(Dir) -> delete_directory_and_contents(Dir) ; true ).

gen_lib(Preds, Module, Dir, Src) :-
    safe_rmdir(Dir),
    once(write_wam_rust_project(Preds, [module_name(Module), parallel_aggregates(true)], Dir)),
    atom_concat(Dir, '/src/lib.rs', LibRs),
    read_file_to_string(LibRs, Src, []).

:- begin_tests(wam_rust_embedded_input_threading).

% input-taking embedded aggregate: helpers gain a leading param (arity /2, /3) and
% the ParAggregate instruction carries the container input register Y2.
test(input_taking_threads_register, [cleanup(safe_rmdir('output/test_emb_thread_unit'))]) :-
    gen_lib([user:eg_p/2, user:eg_guard/1, user:eg_link/2, user:eg_dn/2],
            eg, 'output/test_emb_thread_unit', Src),
    assertion(sub_string(Src, _, _, _, "ParAggregate")),
    assertion(sub_string(Src, _, _, _, "__par_enum_eg_p/2")),
    assertion(sub_string(Src, _, _, _, "__par_body_eg_p/3")),
    % the input register Y2 (head arg N) is threaded into the instruction
    assertion(sub_string(Src, _, _, _, "vec![\"Y2\".to_string()]")).

% input-less embedded aggregate: bare helpers (/1, /2) and an empty input-reg vec.
test(input_less_no_registers, [cleanup(safe_rmdir('output/test_emb_inputless_unit'))]) :-
    gen_lib([user:il_p/1, user:il_guard/1, user:il_fact/1, user:il_dn/2],
            il, 'output/test_emb_inputless_unit', Src),
    assertion(sub_string(Src, _, _, _, "ParAggregate")),
    assertion(sub_string(Src, _, _, _, "__par_enum_il_p/1")),
    assertion(sub_string(Src, _, _, _, "__par_body_il_p/2")),
    % no external inputs -> empty register vec
    assertion(sub_string(Src, _, _, _, "\"Y1\".to_string(), vec![])")).

:- end_tests(wam_rust_embedded_input_threading).
