% test_wam_wat_lowered_t2.pl
%
% End-to-end execution test for the WAT T2 lowering — if-then-else /
% negation / once ( ( C -> T ; E ), \+ G, once/1 ) — lowering type T2 in
% docs/proposals/WAM_LOWERING_TAXONOMY_AND_MATRIX.md. This closes the last
% ✗ in the T2 column (every other hybrid target already lowers ITE).
%
% The shared wam_ite_structurer folds the soft-cut block
% (try_me_else … <cond> cut_ite <then> jump … ; trust_me <else>) into
% ite(Cond,Then,Else); the WAT emitter then emits native WAT: the condition
% runs in a `(block $ite_condK (result i32) … (i32.const 1))` that branches out
% with 0 on the first failing instruction, a trail mark is saved before it, and
% the else branch is preceded by `$unwind_trail` (mirroring the bytecode
% try_me_else/cut_ite choice-point trail unwind).
%
% Two layers of assertions:
%   (1) ABSOLUTE correctness, on the cases the WAT runtime executes correctly
%       (condition truth -> branch selection, then/else body success/failure,
%       negation, once, and sequential ITE blocks): the lowered export returns
%       the Prolog-correct 1/0.
%   (2) PARITY: the lowered (emit_mode(functions)) result equals the bytecode
%       interpreter (emit_mode(interpreter)) result for EVERY case — the core
%       property of a faithful lowering.
%
% The cond_bind* cases thread a variable binding from the condition into the
% then-branch ( ( X = 7 -> X > 5 ; … ) ). These previously exposed a WAT-runtime
% bug — unify bound a variable to a Ref into a transient argument-register cell,
% so the variable silently changed when a later goal reused that register — and
% were asserted parity-only. That bug is now fixed in unify_addrs (scalars are
% copied by value, never bound as a Ref to a register), so they are absolute
% correctness cases and guard the fix against regression.
%
% Skipped automatically when wat2wasm or node is unavailable.

:- use_module(library(lists)).
:- use_module('../src/unifyweaver/targets/wam_target').
:- use_module('../src/unifyweaver/targets/wam_wat_target').
:- use_module('../src/unifyweaver/targets/wam_wat_lowered_emitter').

:- dynamic user:ite_then_ok/0, user:ite_then_body_fail/0,
           user:ite_else_ok/0, user:ite_else_body_fail/0,
           user:neg_true/0, user:neg_false/0,
           user:once_true/0, user:once_false/0,
           user:multi_ite/0, user:nest_ite/0,
           user:cond_bind/0, user:cond_bind_false/0.

% Branch selection by condition truth; branch body success/failure proves the
% RIGHT branch ran (not just that some branch succeeded).
user:ite_then_ok        :- ( 5 > 0 -> true ; fail ).   % then taken, then=true  -> 1
user:ite_then_body_fail :- ( 5 > 0 -> fail ; true ).   % then taken, then=fail  -> 0
user:ite_else_ok        :- ( 0 > 5 -> fail ; true ).   % else taken, else=true  -> 1
user:ite_else_body_fail :- ( 0 > 5 -> true ; fail ).   % else taken, else=fail  -> 0
% negation / once
user:neg_true   :- \+ 0 > 5.        % 0>5 false -> \+ succeeds -> 1
user:neg_false  :- \+ 5 > 0.        % 5>0 true  -> \+ fails    -> 0
user:once_true  :- once( 5 > 0 ).   % -> 1
user:once_false :- once( 0 > 5 ).   % -> 0
% sequential + nested ITE blocks (also guards against the duplicate-function
% emission that a nondeterministic structuring would cause).
user:multi_ite  :- ( 5 > 0 -> true ; fail ), ( 0 > 5 -> fail ; true ).         % -> 1
user:nest_ite   :- ( 5 > 0 -> ( 5 > 10 -> fail ; true ) ; fail ).             % -> 1
% condition binds a variable used (compared) in the then-branch — regression
% guard for the unify scalar-propagation fix. The then-branch must see X = 7.
user:cond_bind        :- ( X = 7 -> X > 5 ; fail ).   % then: 7 > 5 -> 1
user:cond_bind_false  :- ( X = 7 -> X > 9 ; fail ).   % then: 7 > 9 -> 0 (not stale)

% Name-Expected for the absolute-correctness cases.
correctness_cases([ ite_then_ok-1, ite_then_body_fail-0,
                    ite_else_ok-1, ite_else_body_fail-0,
                    neg_true-1, neg_false-0,
                    once_true-1, once_false-0,
                    multi_ite-1, nest_ite-1,
                    cond_bind-1, cond_bind_false-0 ]).

% All predicates for the gate and parity checks.
all_preds([ ite_then_ok, ite_then_body_fail, ite_else_ok, ite_else_body_fail,
            neg_true, neg_false, once_true, once_false,
            multi_ite, nest_ite, cond_bind, cond_bind_false ]).

tool_available(Tool) :-
    catch(process_create(path(Tool), ['--version'], [stdout(null), stderr(null)]),
          _, fail).

wat_tools_available :- tool_available(wat2wasm), tool_available(node).

:- begin_tests(wam_wat_lowered_t2, [condition(wat_tools_available)]).

% Every pin must lower as ite_lowered (the T2 path), not via the interpreter.
test(gate_picks_ite_lowered) :-
    all_preds(Ps),
    forall(member(Name, Ps),
           ( PI = Name/0,
             wam_target:compile_predicate_to_wam(PI, [], W),
             wam_wat_lowerable(PI, W, Reason),
             assertion(Reason == ite_lowered) )).

% The emitted WAT must contain the native if/else condition block + trail mark
% + unwind for the ITE.
test(emits_native_ite) :-
    wam_target:compile_predicate_to_wam(ite_else_ok/0, [], W),
    lower_predicate_to_wat(user:ite_else_ok/0, W, [], lowered(_, _, Code)),
    assertion(sub_atom(Code, _, _, _, 'block $ite_cond0')),
    assertion(sub_atom(Code, _, _, _, 'call $get_trail_top')),
    assertion(sub_atom(Code, _, _, _, 'call $unwind_trail')).

% Lowering must be deterministic: exactly one lowered function per predicate
% (a nondeterministic structuring previously emitted duplicates for sequential
% / nested ITE blocks, which wat2wasm rejects as a redefinition).
test(lowering_deterministic) :-
    forall(member(Name, [multi_ite, nest_ite, cond_bind]),
           ( PI = Name/0,
             wam_target:compile_predicate_to_wam(PI, [], W),
             findall(C, lower_predicate_to_wat(PI, W, [], C), Cs),
             length(Cs, N),
             assertion(N == 1) )).

% Absolute correctness: the lowered export returns the Prolog-correct 1/0.
test(t2_exec_correctness) :-
    correctness_cases(Cs),
    findall(Name, member(Name-_, Cs), Names),
    build_module(functions, Names, WasmFile, Harness),
    findall(Name-Got-Want,
            ( member(Name-Want, Cs),
              format(atom(Export), '~w_0', [Name]),
              run_export(Harness, WasmFile, Export, Got),
              Got =\= Want ),
            Failures),
    ( Failures == []
    ->  true
    ;   format(user_error, "~n[wat t2 correctness mismatches]~n~q~n", [Failures]),
        throw(wam_wat_t2_failed(Failures))
    ).

% Parity: the lowered fast path agrees with the bytecode interpreter for EVERY
% predicate (the defining property of a faithful lowering).
test(t2_lowered_matches_interpreter) :-
    all_preds(Names),
    build_module(functions,   Names, LoweredWasm, Harness),
    build_module(interpreter, Names, InterpWasm, _),
    findall(Name-Lo-In,
            ( member(Name, Names),
              format(atom(Export), '~w_0', [Name]),
              run_export(Harness, LoweredWasm, Export, Lo),
              run_export(Harness, InterpWasm, Export, In),
              Lo =\= In ),
            Disagreements),
    ( Disagreements == []
    ->  true
    ;   format(user_error, "~n[wat t2 lowered/interp disagreements]~n~q~n", [Disagreements]),
        throw(wam_wat_t2_parity_failed(Disagreements))
    ).

:- end_tests(wam_wat_lowered_t2).

% --- build + run helpers (wat2wasm + node, mirrors test_wam_wat_target.pl) ---

build_module(Mode, Names, WasmFile, Harness) :-
    format(atom(Dir), 'output/test_wam_wat_t2_~w', [Mode]),
    ( exists_directory(Dir) -> delete_directory_and_contents(Dir) ; true ),
    make_directory_path(Dir),
    findall(Name/0, member(Name, Names), Preds),
    format(atom(WatFile), '~w/t2.wat', [Dir]),
    format(atom(Module), 'wat_t2_~w', [Mode]),
    write_wam_wat_project(Preds, [module_name(Module), emit_mode(Mode)], WatFile),
    compile_wat(WatFile, WasmFile),
    ensure_harness(Harness).

compile_wat(WatFile, WasmFile) :-
    file_name_extension(Base, _, WatFile),
    file_name_extension(Base, wasm, WasmFile),
    format(string(Cmd), "wat2wasm ~w -o ~w 2>&1", [WatFile, WasmFile]),
    process_create(path(sh), ['-c', Cmd], [stdout(pipe(Out)), process(Pid)]),
    read_string(Out, _, CompileOut), close(Out),
    process_wait(Pid, Exit),
    ( Exit == exit(0) -> true
    ; throw(wam_wat_t2_compile_failed(CompileOut)) ).

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
    ; throw(wam_wat_t2_run_failed(Export, RunOut)) ).

ensure_harness(Path) :-
    Path = 'output/test_wam_wat_t2_harness.js',
    ( exists_file(Path) -> true
    ; harness_source(Src),
      setup_call_cleanup(open(Path, write, S), write(S, Src), close(S)) ).

harness_source(Src) :-
    Src = "const fs=require('fs');\n\c
const [,,wasmPath,exportName]=process.argv;\n\c
const bytes=fs.readFileSync(wasmPath);\n\c
const imports={env:{\n\c
  print_i64:_=>0, print_char:_=>0, print_newline:_=>0\n\c
}};\n\c
(async()=>{\n\c
  try{\n\c
    const{instance}=await WebAssembly.instantiate(bytes,imports);\n\c
    const fn=instance.exports[exportName];\n\c
    if(typeof fn!=='function'){console.log('ERROR export '+exportName+' not found');process.exit(1);}\n\c
    console.log('RESULT '+fn());\n\c
  }catch(e){console.log('ERROR '+e.message);process.exit(1);}\n\c
})();\n".
