% test_wam_rust_machine_forkable.pl
%
% Slice 2b, sub-step 1: the generated WAM machine (WamState) must be Clone + Send
% so the T7 parallel substrate (rust_runtime/par_aggregate.rs) can fork an
% independent machine per worker thread. state.rs.mustache now derives Clone and
% carries a compile-time `_wam_state_is_forkable` assertion pinning Clone + Send.
%
% This test proves it two ways: a fast structural check that the derive +
% assertion are emitted, and (when cargo is present) that a generated project
% actually compiles — a successful build means the static Send+Clone assertion
% held, i.e. the machine is genuinely forkable.

:- use_module('../src/unifyweaver/targets/wam_rust_target',
              [write_wam_rust_project/3, cargo_check_project/2]).
:- use_module(library(readutil), [read_file_to_string/3]).

:- dynamic mf_fact/1.
mf_fact(1). mf_fact(2). mf_fact(3).
mf_q(X) :- mf_fact(X).

safe_rmdir(Dir) :- ( exists_directory(Dir) -> delete_directory_and_contents(Dir) ; true ).

gen(Dir) :-
    safe_rmdir(Dir),
    once(write_wam_rust_project([user:mf_q/1, user:mf_fact/1], [module_name(user)], Dir)).

cargo_ok :-
    catch(( process_create(path(cargo), ['--version'],
                           [stdout(null), stderr(null), process(P)]),
            process_wait(P, exit(0)) ), _, fail).

:- begin_tests(wam_rust_machine_forkable).

% Structural: the machine derives Clone and the Send+Clone assertion is emitted.
test(state_derives_clone_and_asserts_forkable,
     [cleanup(safe_rmdir('output/test_mf_struct'))]) :-
    Dir = 'output/test_mf_struct',
    gen(Dir),
    atom_concat(Dir, '/src/state.rs', StateRs),
    read_file_to_string(StateRs, S, []),
    assertion(sub_string(S, _, _, _, "#[derive(Clone)]\npub struct WamState")),
    assertion(sub_string(S, _, _, _, "_wam_state_is_forkable")),
    assertion(sub_string(S, _, _, _, "_assert_send::<WamState>")),
    assertion(sub_string(S, _, _, _, "_assert_clone::<WamState>")).

% Compile: a generated project builds, so the static Clone+Send assertion held.
test(generated_project_compiles_with_forkable_machine,
     [condition(cargo_ok), cleanup(safe_rmdir('output/test_mf_cargo'))]) :-
    Dir = 'output/test_mf_cargo',
    gen(Dir),
    cargo_check_project(Dir, Result),
    assertion(memberchk(Result, [ok, not_available])).

:- end_tests(wam_rust_machine_forkable).
