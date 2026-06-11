% test_wam_wat_unify_propagation.pl
%
% Regression test for the WAT unify scalar-propagation fix (in $unify_addrs,
% wam_wat_target.pl).
%
% THE BUG: when unify bound an unbound variable to a bound cell, it always
% stored Ref(other-cell) — even when the other cell was a SCALAR sitting in a
% transient argument register (e.g. `put_constant 7, A2` for `X = 7`). The
% variable then aliased that register; when a LATER goal reused the register
% (e.g. `put_constant 5, A2` for `X > 5`), the variable silently changed to the
% new value. So `X = 7, X > 5` evaluated as `5 > 5` (or `7 > 7`, depending on
% layout) → wrong result, and some forms (`X = 7, 5 < X`) span a Ref cycle and
% HUNG. `write(X)` looked fine only because it ran before the register was
% reused. This affected EVERY arithmetic comparison in the classic guard
% pattern `X = V, X <cmp> K`, on BOTH the interpreter and lowered paths.
%
% THE FIX: a scalar (atom/integer/float, tag < 3) is copied BY VALUE into the
% variable's heap cell; only compounds/cons/unbound cells (which live on the
% heap) are bound as a Ref. This is the standard WAM "never bind a variable to a
% transient cell" rule.
%
% These cases run the bytecode interpreter (the bug was in the shared runtime,
% so it manifests identically in interpreter and lowered modes). All would
% return wrong answers (or hang) before the fix.
%
% Skipped automatically when wat2wasm or node is unavailable.

:- use_module(library(lists)).
:- use_module(library(filesex)).
:- use_module('../src/unifyweaver/targets/wam_target').
:- use_module('../src/unifyweaver/targets/wam_wat_target').

:- dynamic user:gt_true/0, user:gt_false/0, user:eq_true/0, user:eq_false/0,
           user:lt_a2/0, user:ge_eq/0, user:both_vars/0, user:alias_chain/0,
           user:atom_keep/0.

% arithmetic comparison reads the live value of X, not the stale register
user:gt_true   :- X = 7, X > 3.        % 7 > 3 -> 1
user:gt_false  :- X = 7, X > 9.        % 7 > 9 -> 0
user:eq_true   :- X = 7, X =:= 7.      % -> 1
user:eq_false  :- X = 7, X =:= 3.      % 7 =:= 3 -> 0 (was wrongly 1: stale 7=:=7)
% X in the second argument position (this form HUNG before the fix)
user:lt_a2     :- X = 7, 5 < X.        % 5 < 7 -> 1
user:ge_eq     :- X = 7, X >= 7.       % -> 1
% both operands variables
user:both_vars :- X = 7, Y = 5, X > Y. % 7 > 5 -> 1
% variable-to-variable alias then compare (HUNG before the fix)
user:alias_chain :- X = 7, Y = X, Y =:= 7.   % -> 1
% atom binding must still propagate (scalar copy covers atoms too)
user:atom_keep :- X = foo, Y = 5, X == foo.  % -> 1

cases([ gt_true-1, gt_false-0, eq_true-1, eq_false-0,
        lt_a2-1, ge_eq-1, both_vars-1, alias_chain-1, atom_keep-1 ]).

tool_available(Tool) :-
    catch(process_create(path(Tool), ['--version'], [stdout(null), stderr(null)]),
          _, fail).
wat_tools_available :- tool_available(wat2wasm), tool_available(node).

:- begin_tests(wam_wat_unify_propagation, [condition(wat_tools_available)]).

test(scalar_propagates_through_goals) :-
    cases(Cs),
    findall(N, member(N-_, Cs), Names),
    build_module(Names, Wasm, Harness),
    findall(Name-Got-Want,
            ( member(Name-Want, Cs),
              format(atom(Export), '~w_0', [Name]),
              run_export(Harness, Wasm, Export, Got),
              Got =\= Want ),
            Failures),
    ( Failures == []
    ->  true
    ;   format(user_error, "~n[wat unify propagation mismatches]~n~q~n", [Failures]),
        throw(wam_wat_unify_failed(Failures))
    ).

:- end_tests(wam_wat_unify_propagation).

build_module(Names, WasmFile, Harness) :-
    Dir = 'output/test_wam_wat_unify',
    ( exists_directory(Dir) -> delete_directory_and_contents(Dir) ; true ),
    make_directory_path(Dir),
    findall(user:N/0, member(N, Names), Preds),
    atom_concat(Dir, '/u.wat', WatFile),
    write_wam_wat_project(Preds, [module_name(wat_unify), emit_mode(interpreter)], WatFile),
    file_name_extension(Base, _, WatFile),
    file_name_extension(Base, wasm, WasmFile),
    format(string(Cmd), "wat2wasm ~w -o ~w 2>&1", [WatFile, WasmFile]),
    process_create(path(sh), ['-c', Cmd], [stdout(pipe(Out)), process(Pid)]),
    read_string(Out, _, CompileOut), close(Out),
    process_wait(Pid, Exit),
    ( Exit == exit(0) -> true ; throw(wam_wat_unify_compile_failed(CompileOut)) ),
    ensure_harness(Harness).

run_export(Harness, WasmFile, Export, Result) :-
    process_create(path(node), [Harness, WasmFile, Export],
                   [stdout(pipe(Out)), stderr(null), process(Pid)]),
    read_string(Out, _, RunOut), close(Out),
    process_wait(Pid, _),
    ( sub_string(RunOut, _, _, _, "RESULT "),
      split_string(RunOut, " \n", " \n", Parts),
      append(_, ["RESULT", RS|_], Parts),
      number_string(Result, RS)
    -> true
    ; throw(wam_wat_unify_run_failed(Export, RunOut)) ).

ensure_harness(Path) :-
    Path = 'output/test_wam_wat_unify_harness.js',
    Src = "const fs=require('fs');\n\c
const [,,wasmPath,exportName]=process.argv;\n\c
const bytes=fs.readFileSync(wasmPath);\n\c
const imports={env:{print_i64:_=>0, print_char:_=>0, print_newline:_=>0}};\n\c
(async()=>{\n\c
  try{\n\c
    const{instance}=await WebAssembly.instantiate(bytes,imports);\n\c
    const fn=instance.exports[exportName];\n\c
    if(typeof fn!=='function'){console.log('ERROR export '+exportName+' not found');process.exit(1);}\n\c
    console.log('RESULT '+fn());\n\c
  }catch(e){console.log('ERROR '+e.message);process.exit(1);}\n\c
})();\n",
    setup_call_cleanup(open(Path, write, S, [encoding(utf8)]), write(S, Src), close(S)).
