% test_wam_wat_lowered_t5.pl
%
% End-to-end test for the WAT T5 lowering — multi-clause first-argument dispatch
% (lowering type T5 / clause_chain in
% docs/proposals/WAM_LOWERING_TAXONOMY_AND_MATRIX.md). WAT was the last target
% with no multi-clause lowering.
%
% A predicate that discriminates on a distinct first-argument constant
% (color(red). color(green). … or sz(small,1). sz(medium,2). …) lowers to ONE
% WAT function that tests A1 against each clause's discriminator and runs that
% clause's body inline — every clause is native, no interpreter hop for clauses
% 2+ when A1 is bound. The switch_on_* indexing prefix (which marks exactly
% these predicates) is stripped, then the shared wam_clause_chain front-end
% produces the guard list.
%
% WAT's lowered model is first-solution (the exported entry tries the lowered
% function, replaying the bytecode interpreter on failure; lowered functions are
% never entered from inside another predicate), so the first-solution cascade is
% faithful: distinct discriminators + bound A1 => at most one clause matches,
% deterministically. do_get_constant is a pure test when A1 is bound, so the
% cascade only binds nothing; an unbound A1 is deferred to the interpreter.
%
% Exec strategy: WAT exports take no parameters and wam_init clears the register
% file, so we cannot pass a bound A1 through the normal entry. Instead we
% generate the functions-mode project, then INJECT test-only exports that
% wam_init, bind the argument registers via do_put_constant, and call the
% lowered function directly. This exercises real per-discriminator dispatch
% (including non-first clauses) through the actual generated WAT.
%
% Skipped automatically when wat2wasm or node is unavailable.

:- use_module(library(lists)).
:- use_module(library(filesex)).
:- use_module('../src/unifyweaver/targets/wam_target').
:- use_module('../src/unifyweaver/targets/wam_wat_target').
:- use_module('../src/unifyweaver/targets/wam_wat_lowered_emitter').

:- dynamic user:color/1, user:sz/2, user:op/2, user:samearg/1.

% facts, distinct first-arg atoms (first-argument indexed)
user:color(red).
user:color(green).
user:color(blue).
% multi-arg facts, distinct first-arg atoms; remainder is a second head match
user:sz(small, 1).
user:sz(medium, 2).
user:sz(large, 3).
% distinct first-arg atoms with a builtin body (gate/codegen only — its
% remainder threads a variable through get_value/put_value, which the WAT
% runtime's known propagation limitation would bite, so it is not exec-tested)
user:op(add, R) :- R is 1 + 1.
user:op(mul, R) :- R is 2 * 3.
% NON-distinct first arg (variable head): must NOT be T5 (stays multi_clause_1)
user:samearg(X) :- X = a.
user:samearg(X) :- X = b.

tool_available(Tool) :-
    catch(process_create(path(Tool), ['--version'], [stdout(null), stderr(null)]),
          _, fail).

wat_tools_available :- tool_available(wat2wasm), tool_available(node).

:- begin_tests(wam_wat_lowered_t5, [condition(wat_tools_available)]).

% Distinct-first-arg predicates lower as clause_chain (T5); a variable-first-arg
% predicate declines T5 and lowers as multi_clause_n (T4, all clauses inline).
test(gate_picks_clause_chain) :-
    forall(member(PI, [color/1, sz/2, op/2]),
           ( wam_target:compile_predicate_to_wam(PI, [], W),
             wam_wat_lowerable(PI, W, Reason),
             assertion(Reason == clause_chain) )),
    wam_target:compile_predicate_to_wam(samearg/1, [], Ws),
    wam_wat_lowerable(samearg/1, Ws, RS),
    assertion(RS == multi_clause_n).

% The emitted WAT: an unbound-A1 guard, one do_get_constant guard per
% discriminator, and the clause body inline under each.
test(emits_guard_cascade) :-
    wam_target:compile_predicate_to_wam(color/1, [], W),
    lower_predicate_to_wat(user:color/1, W, [], lowered(_, _, Code)),
    % unbound check (tag 6) up front
    assertion(sub_atom(Code, _, _, _, 'call $val_tag')),
    assertion(sub_atom(Code, _, _, _, 'call $deref_reg_addr')),
    % one guard per clause — three do_get_constant tests
    findall(_, sub_atom(Code, _, _, _, 'if (call $do_get_constant'), Gs),
    assertion(length(Gs, 3)).

% Real per-discriminator dispatch through the generated WAT: each clause
% (including non-first clauses) is reachable natively; a non-matching
% discriminator returns 0; a remainder head-mismatch returns 0.
test(t5_exec_discrimination) :-
    Cases = [ tc_color_red-(color/1)-["red"]-1,
              tc_color_green-(color/1)-["green"]-1,   % non-first clause, native
              tc_color_blue-(color/1)-["blue"]-1,     % non-first clause, native
              tc_color_yellow-(color/1)-["yellow"]-0, % no matching clause
              tc_sz_small_1-(sz/2)-["small","1"]-1,
              tc_sz_medium_2-(sz/2)-["medium","2"]-1, % non-first clause, native
              tc_sz_large_3-(sz/2)-["large","3"]-1,   % non-first clause, native
              tc_sz_small_2-(sz/2)-["small","2"]-0,   % remainder head mismatch
              tc_sz_big_1-(sz/2)-["big","1"]-0 ],     % no matching clause
    build_injected_module([color/1, sz/2, op/2], Cases, Wasm, Harness),
    findall(Name-Got-Want,
            ( member(Name-_-_-Want, Cases),
              run_export(Harness, Wasm, Name, Got),
              Got =\= Want ),
            Failures),
    ( Failures == []
    ->  true
    ;   format(user_error, "~n[wat t5 discrimination mismatches]~n~q~n", [Failures]),
        throw(wam_wat_t5_failed(Failures))
    ).

:- end_tests(wam_wat_lowered_t5).

% --- build + inject + run helpers (wat2wasm + node) ---

%% build_injected_module(+Preds, +Cases, -WasmFile, -Harness)
%  Write the functions-mode project, append a test export per case (bind the
%  argument registers via do_put_constant, call the lowered function), compile.
build_injected_module(Preds, Cases, WasmFile, Harness) :-
    Dir = 'output/test_wam_wat_t5',
    ( exists_directory(Dir) -> delete_directory_and_contents(Dir) ; true ),
    make_directory_path(Dir),
    findall(M:P, (member(P, Preds), M = user), QPreds),
    atom_concat(Dir, '/t5.wat', WatFile),
    write_wam_wat_project(QPreds, [module_name(wat_t5), emit_mode(functions)], WatFile),
    read_file_to_string(WatFile, Src, []),
    wat_heap_base(Src, Heap),
    findall(Snip, ( member(C, Cases), case_export_snippet(Heap, C, Snip) ), Snips),
    atomic_list_concat(Snips, '\n', Injection),
    inject_before_last_paren(Src, Injection, Src2),
    atom_concat(Dir, '/t5_inj.wat', InjWat),
    setup_call_cleanup(open(InjWat, write, S, [encoding(utf8)]),
                       write(S, Src2), close(S)),
    atom_concat(Dir, '/t5_inj.wasm', WasmFile),
    compile_wat(InjWat, WasmFile),
    ensure_harness(Harness).

%% wat_heap_base(+Src, -Heap) — read the heap base an entry passes to wam_init.
wat_heap_base(Src, Heap) :-
    sub_atom(Src, B, _, _, '$wam_init (i32.const '), !,
    Start is B + 21,
    sub_atom(Src, Start, 40, _, Tail),
    atom_codes(Tail, Codes),
    leading_digits(Codes, DigitCodes),
    number_codes(Heap, DigitCodes).

leading_digits([C|Cs], [C|Ds]) :- code_type(C, digit), !, leading_digits(Cs, Ds).
leading_digits(_, []).

%% case_export_snippet(+Heap, +Name-(P/A)-ArgVals-_Want, -Snippet)
case_export_snippet(Heap, Name-(Pred/Arity)-ArgVals-_Want, Snippet) :-
    wat_lowered_func_name(Pred/Arity, LoweredName),
    findall(Bind,
            ( nth1(K, ArgVals, V),
              arg_reg_atom(K, RegAtom),
              wam_wat_target:wam_instruction_to_wat_operands(put_constant(V, RegAtom), [], _, Op1, Op2),
              format(atom(Bind),
                     '  (drop (call $do_put_constant (i64.const ~w) (i64.const ~w)))',
                     [Op1, Op2]) ),
            Binds),
    atomic_list_concat(Binds, '\n', BindLines),
    format(atom(Snippet),
'(func $~w (export "~w") (result i32)
  (call $wam_init (i32.const ~w))
~w
  (call $~w))',
        [Name, Name, Heap, BindLines, LoweredName]).

arg_reg_atom(1, 'A1').
arg_reg_atom(2, 'A2').
arg_reg_atom(3, 'A3').

%% inject_before_last_paren(+Src, +Injection, -Out)
inject_before_last_paren(Src, Injection, Out) :-
    atom_string(SrcA, Src),
    sub_atom(SrcA, Before, 1, After, ')'),
    sub_atom(SrcA, _, After, 0, Tail),
    \+ sub_atom(Tail, _, _, _, ')'), !,   % the LAST ')'
    sub_atom(SrcA, 0, Before, _, Head),
    atomic_list_concat([Head, '\n', Injection, '\n)', Tail], Out).

compile_wat(WatFile, WasmFile) :-
    format(string(Cmd), "wat2wasm ~w -o ~w 2>&1", [WatFile, WasmFile]),
    process_create(path(sh), ['-c', Cmd], [stdout(pipe(Out)), process(Pid)]),
    read_string(Out, _, CompileOut), close(Out),
    process_wait(Pid, Exit),
    ( Exit == exit(0) -> true
    ; throw(wam_wat_t5_compile_failed(CompileOut)) ).

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
    ; throw(wam_wat_t5_run_failed(Export, RunOut)) ).

ensure_harness(Path) :-
    Path = 'output/test_wam_wat_t5_harness.js',
    harness_source(Src),
    setup_call_cleanup(open(Path, write, S, [encoding(utf8)]), write(S, Src), close(S)).

harness_source(Src) :-
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
})();\n".
