% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% Public entry names that would collide with a symbol the module already
% declares or defines (a libc function such as dup/open/read, a runtime
% function such as step/backtrack, the driver's main) are renamed
% uw_pred_<Pred>; every other predicate keeps its plain name.

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module('../src/unifyweaver/targets/wam_llvm_target').

:- dynamic user:dup/2, user:unlink/1, user:es_plain/1.
user:dup(a, 1).
user:dup(a, 2).
user:unlink(x).
user:es_plain(y).

llvm_as_available :-
    catch(( process_create(path('llvm-as'), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

module_text(Preds, Options, Text) :-
    tmp_file_stream(text, Path, S), close(S),
    write_wam_llvm_project(Preds, [module_name(es)|Options], Path),
    read_file_to_string(Path, Text, []),
    (   llvm_as_available
    ->  atom_concat(Path, '.bc', BC),
        process_create(path('llvm-as'), [Path, '-o', BC],
                       [stderr(pipe(E)), process(Pid)]),
        read_string(E, _, Err), close(E),
        process_wait(Pid, Status),
        catch(delete_file(BC), _, true),
        assertion(Status-Err == exit(0)-"")
    ;   true
    ),
    catch(delete_file(Path), _, true).

:- begin_tests(wam_llvm_entry_symbol).

test(colliding_names_are_renamed) :-
    forall(member(P, [dup, open, read, write, close, time, stat, system,
                      main, step, backtrack, run_loop, execute_builtin]),
           ( wam_llvm_entry_symbol(P, S),
             atom_concat(uw_pred_, P, Want),
             assertion(S == Want) )).

test(other_names_are_kept) :-
    forall(member(P, [es_plain, grade, lw_add, path, ancestor]),
           ( wam_llvm_entry_symbol(P, S), assertion(S == P) )).

% The interpreter path (WAM entry function) and the lowered path (hybrid
% dispatcher / native wrapper) both use the renamed entry, and the module
% assembles -- before, a predicate named dup/2 was "invalid redefinition of
% function 'dup'".
test(module_with_colliding_predicates_assembles) :-
    forall(member(Opts, [[], [emit_mode(functions)]]),
           ( module_text([user:dup/2, user:unlink/1, user:es_plain/1], Opts, Text),
             assertion(sub_string(Text, _, _, _, "define i1 @uw_pred_dup(")),
             assertion(sub_string(Text, _, _, _, "define i1 @uw_pred_unlink(")),
             assertion(sub_string(Text, _, _, _, "define i1 @es_plain(")),
             assertion(\+ sub_string(Text, _, _, _, "define i1 @dup(")) )).

% Computing the reserved set must not disturb a build in progress: atom ids
% are per module, and the scan runs emitters that intern atoms.
test(reserved_set_leaves_atom_table_alone) :-
    retractall(wam_llvm_target:wam_llvm_reserved_symbols_cache(_)),
    aggregate_all(count, wam_llvm_target:atom_table_entry(_, _), Before),
    wam_llvm_entry_symbol(dup, _),
    aggregate_all(count, wam_llvm_target:atom_table_entry(_, _), After),
    assertion(Before == After).

:- end_tests(wam_llvm_entry_symbol).
