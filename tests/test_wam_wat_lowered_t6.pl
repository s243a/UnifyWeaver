% test_wam_wat_lowered_t6.pl
%
% End-to-end test for the WAT T6 lowering — first-argument indexing via a binary
% search on the interned atom hash, lowering type T6 in
% docs/proposals/WAM_LOWERING_TAXONOMY_AND_MATRIX.md.
%
% WAT atoms are keyed by a sparse i64 hash (atom_hash_i64), so a dense br_table
% does not apply; the lowered T5 path is a linear do_get_constant cascade (O(n)).
% T6 replaces the cascade with a binary search (O(log n)) on the sorted clause
% hashes, comparing A1's val_payload against each node. Each matched leaf still
% runs do_get_constant before its body, so hash collisions / non-atom A1 are
% verified and fall through to the 0 return — semantics identical to T5.
% Benchmarked in V8 (wat2wasm + node): 1.16x at 8, 1.79x at 64, 6.72x at 256
% (V8 does not flatten the linear i64.eq chain).
%
% Gated like the other targets: fires only when there are >= t6_min_clauses
% clauses (default 8). Below the threshold the linear T5 cascade is kept.
%
% Skipped automatically when wat2wasm or node is unavailable.

:- use_module(library(lists)).
:- use_module('../src/unifyweaver/targets/wam_target').
:- use_module('../src/unifyweaver/targets/wam_wat_target').
:- use_module('../src/unifyweaver/targets/wam_wat_lowered_emitter').

:- dynamic user:shade/1, user:tone/2, user:few/1.

user:shade(s01). user:shade(s02). user:shade(s03). user:shade(s04).
user:shade(s05). user:shade(s06). user:shade(s07). user:shade(s08).
user:shade(s09). user:shade(s10).

user:tone(c01, bright). user:tone(c02, bright). user:tone(c03, bright).
user:tone(c04, bright). user:tone(c05, dark).   user:tone(c06, dark).
user:tone(c07, dark).   user:tone(c08, dark).   user:tone(c09, dark).
user:tone(c10, dark).

user:few(a). user:few(b). user:few(c).

tool_available(Exe) :-
    catch(( process_create(path(Exe), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, _) ), _, fail).

wat_tools_available :- tool_available(wat2wasm), tool_available(node).

:- begin_tests(wam_wat_lowered_t6, [condition(wat_tools_available)]).

% Codegen gate: shade/1 + tone/2 (>=8 clauses) emit the binary search (a local
% $h hash + i64.lt_u navigation); few/1 (3) keeps the linear T5 cascade; the
% threshold is overridable.
test(gate_picks_t6_for_many_t5_for_few) :-
    wam_target:compile_predicate_to_wam(shade/1, [], Ws),
    lower_predicate_to_wat(user:shade/1, Ws, [], lowered(_, _, ShadeCode)),
    assertion(sub_atom(ShadeCode, _, _, _, 'T6 first-argument indexing')),
    assertion(sub_atom(ShadeCode, _, _, _, 'i64.lt_u')),
    assertion(sub_atom(ShadeCode, _, _, _, 'local.set $h')),
    wam_target:compile_predicate_to_wam(tone/2, [], Wt),
    lower_predicate_to_wat(user:tone/2, Wt, [], lowered(_, _, ToneCode)),
    assertion(sub_atom(ToneCode, _, _, _, 'T6 first-argument indexing')),
    wam_target:compile_predicate_to_wam(few/1, [], Wf),
    lower_predicate_to_wat(user:few/1, Wf, [], lowered(_, _, FewCode)),
    assertion(\+ sub_atom(FewCode, _, _, _, 'T6 first-argument indexing')),
    assertion(sub_atom(FewCode, _, _, _, 'T5 first-argument dispatch')),
    lower_predicate_to_wat(user:few/1, Wf, [t6_min_clauses(3)], lowered(_, _, FewT6)),
    assertion(sub_atom(FewT6, _, _, _, 'T6 first-argument indexing')).

% Real dispatch through the generated WAT: each clause (first, middle, last) is
% reachable via the binary search; a non-matching key returns 0; a remainder
% head-mismatch returns 0; the below-threshold control (few/1) still works.
test(t6_exec_discrimination) :-
    Cases = [ tc_shade_s01-(shade/1)-["s01"]-1,
              tc_shade_s05-(shade/1)-["s05"]-1,   % middle clause via search
              tc_shade_s10-(shade/1)-["s10"]-1,   % last clause via search
              tc_shade_zz-(shade/1)-["zz"]-0,     % no matching clause
              tc_tone_c01_bright-(tone/2)-["c01","bright"]-1,
              tc_tone_c05_dark-(tone/2)-["c05","dark"]-1,   % middle clause via search
              tc_tone_c05_bright-(tone/2)-["c05","bright"]-0, % remainder mismatch
              tc_tone_zz_dark-(tone/2)-["zz","dark"]-0,     % no matching clause
              tc_few_a-(few/1)-["a"]-1,           % T5 control (below the gate)
              tc_few_z-(few/1)-["z"]-0 ],
    build_injected_module([shade/1, tone/2, few/1], Cases, Wasm, Harness),
    findall(Name-Got-Want,
            ( member(Name-_-_-Want, Cases),
              run_export(Harness, Wasm, Name, Got),
              Got =\= Want ),
            Failures),
    ( Failures == []
    ->  true
    ;   format(user_error, "~n[wat t6 discrimination mismatches]~n~q~n", [Failures]),
        throw(wam_wat_t6_failed(Failures))
    ).

:- end_tests(wam_wat_lowered_t6).

% --- build + inject + run helpers (wat2wasm + node), mirroring the T5 test ---

build_injected_module(Preds, Cases, WasmFile, Harness) :-
    Dir = 'output/test_wam_wat_t6',
    ( exists_directory(Dir) -> delete_directory_and_contents(Dir) ; true ),
    make_directory_path(Dir),
    findall(M:P, (member(P, Preds), M = user), QPreds),
    atom_concat(Dir, '/t6.wat', WatFile),
    write_wam_wat_project(QPreds, [module_name(wat_t6), emit_mode(functions)], WatFile),
    read_file_to_string(WatFile, Src, []),
    wat_heap_base(Src, Heap),
    findall(Snip, ( member(C, Cases), case_export_snippet(Heap, C, Snip) ), Snips),
    atomic_list_concat(Snips, '\n', Injection),
    inject_before_last_paren(Src, Injection, Src2),
    atom_concat(Dir, '/t6_inj.wat', InjWat),
    setup_call_cleanup(open(InjWat, write, S, [encoding(utf8)]),
                       write(S, Src2), close(S)),
    atom_concat(Dir, '/t6_inj.wasm', WasmFile),
    compile_wat(InjWat, WasmFile),
    ensure_harness(Harness).

wat_heap_base(Src, Heap) :-
    sub_atom(Src, B, _, _, '$wam_init (i32.const '), !,
    Start is B + 21,
    sub_atom(Src, Start, 40, _, Tail),
    atom_codes(Tail, Codes),
    leading_digits(Codes, DigitCodes),
    number_codes(Heap, DigitCodes).

leading_digits([C|Cs], [C|Ds]) :- code_type(C, digit), !, leading_digits(Cs, Ds).
leading_digits(_, []).

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

inject_before_last_paren(Src, Injection, Out) :-
    atom_string(SrcA, Src),
    sub_atom(SrcA, Before, 1, After, ')'),
    sub_atom(SrcA, _, After, 0, Tail),
    \+ sub_atom(Tail, _, _, _, ')'), !,
    sub_atom(SrcA, 0, Before, _, Head),
    atomic_list_concat([Head, '\n', Injection, '\n)', Tail], Out).

compile_wat(WatFile, WasmFile) :-
    format(string(Cmd), "wat2wasm ~w -o ~w 2>&1", [WatFile, WasmFile]),
    process_create(path(sh), ['-c', Cmd], [stdout(pipe(Out)), process(Pid)]),
    read_string(Out, _, CompileOut), close(Out),
    process_wait(Pid, Exit),
    ( Exit == exit(0) -> true
    ; throw(wam_wat_t6_compile_failed(CompileOut)) ).

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
    ; throw(wam_wat_t6_run_failed(Export, RunOut)) ).

ensure_harness(Path) :-
    Path = 'output/test_wam_wat_t6_harness.js',
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
