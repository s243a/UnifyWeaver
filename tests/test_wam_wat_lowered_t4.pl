% test_wam_wat_lowered_t4.pl
%
% End-to-end test for the WAT T4 lowering — multi-clause, ALL clauses inline
% (lowering type T4 / multi_clause_n in
% docs/proposals/WAM_LOWERING_TAXONOMY_AND_MATRIX.md). This closes the last
% multi-clause cell in the matrix.
%
% A multi-clause predicate that is NOT first-argument-discriminable (so the T5
% guard cascade declines) but whose every clause is deterministic + supported
% lowers to one WAT function that tries each clause in order and commits to the
% first that succeeds (WAT's lowered model is first-solution; the public entry
% replays the bytecode interpreter on a 0 return).
%
% Between clause attempts the argument registers A1..A_arity and the trail are
% restored to their predicate-entry state, exactly as the bytecode interpreter's
% choice point does — a clause body can overwrite an argument register (e.g.
% put_constant 100, A2) before failing, and trail unwinding alone does not undo
% a direct register overwrite. The qq/2 case below is the decisive register-
% restore test: clause 1 clobbers A2 then fails, and clause 2 only succeeds if
% A2 has been restored to its entry value.
%
% Exec strategy mirrors test_wam_wat_lowered_t5: WAT exports take no parameters
% and wam_init clears the register file, so we generate the functions-mode
% project, then INJECT test-only exports that wam_init, bind the argument
% registers via do_put_constant, and call the lowered function directly.
%
% Skipped automatically when wat2wasm or node is unavailable.

:- use_module(library(lists)).
:- use_module(library(filesex)).
:- use_module('../src/unifyweaver/targets/wam_target').
:- use_module('../src/unifyweaver/targets/wam_wat_target').
:- use_module('../src/unifyweaver/targets/wam_wat_lowered_emitter').

:- dynamic user:samearg/1, user:grade/2, user:qq/2, user:color/1.

% variable first argument -> NOT first-arg-discriminable -> T4, not T5
user:samearg(X) :- X = a.
user:samearg(X) :- X = b.
% guard bodies (a builtin per clause); variable first arg
user:grade(N, pass) :- N >= 50.
user:grade(N, fail) :- N < 50.
% register-restore case: clause 1 overwrites A2 (put_constant 100) before
% failing; clause 2 needs A2 restored to its entry value (50).
user:qq(X, 50) :- X > 100.
user:qq(_Y, 50) :- true.
% distinct first-arg atoms -> T5 (clause_chain), NOT T4 (over-claim guard)
user:color(red).
user:color(green).
user:color(blue).

tool_available(Tool) :-
    catch(process_create(path(Tool), ['--version'], [stdout(null), stderr(null)]),
          _, fail).
wat_tools_available :- tool_available(wat2wasm), tool_available(node).

:- begin_tests(wam_wat_lowered_t4, [condition(wat_tools_available)]).

% Non-first-arg-discriminable multi-clause predicates lower as multi_clause_n
% (T4); a distinct-first-arg predicate must stay clause_chain (T5).
test(gate_picks_multi_clause_n) :-
    forall(member(PI, [samearg/1, grade/2, qq/2]),
           ( wam_target:compile_predicate_to_wam(PI, [], W),
             wam_wat_lowerable(PI, W, Reason),
             assertion(Reason == multi_clause_n) )),
    wam_target:compile_predicate_to_wam(color/1, [], Wc),
    wam_wat_lowerable(color/1, Wc, Rc),
    assertion(Rc == clause_chain).

% The emitted WAT: argument-register snapshot, a per-clause block with a `br`
% out on failure, and the trail/register restore between clauses.
test(emits_snapshot_and_restore) :-
    wam_target:compile_predicate_to_wam(grade/2, [], W),
    lower_predicate_to_wat(user:grade/2, W, [], lowered(_, _, Code)),
    assertion(sub_atom(Code, _, _, _, 'call $get_trail_top')),
    assertion(sub_atom(Code, _, _, _, 'block $c1')),
    assertion(sub_atom(Code, _, _, _, 'block $c2')),
    assertion(sub_atom(Code, _, _, _, 'br $c1')),
    assertion(sub_atom(Code, _, _, _, 'call $unwind_trail')),
    assertion(sub_atom(Code, _, _, _, 'call $val_store (call $reg_offset')).

% Real execution through the generated WAT: discrimination across clauses, a
% non-first clause reached natively, and the decisive register-restore case.
test(t4_exec) :-
    Cases = [ tc_grade_70_pass-(grade/2)-["70","pass"]-1,    % clause 1
              tc_grade_30_pass-(grade/2)-["30","pass"]-0,    % neither matches
              tc_grade_30_fail-(grade/2)-["30","fail"]-1,    % clause 2, native
              tc_grade_70_fail-(grade/2)-["70","fail"]-0,
              tc_samearg_a-(samearg/1)-["a"]-1,              % clause 1
              tc_samearg_b-(samearg/1)-["b"]-1,              % clause 2, native
              tc_samearg_c-(samearg/1)-["c"]-0,
              tc_qq_30_50-(qq/2)-["30","50"]-1,              % REGISTER RESTORE
              tc_qq_200_50-(qq/2)-["200","50"]-1,            % clause 1
              tc_qq_30_99-(qq/2)-["30","99"]-0 ],            % arg2 mismatch
    build_injected_module([samearg/1, grade/2, qq/2], Cases, Wasm, Harness),
    findall(Name-Got-Want,
            ( member(Name-_-_-Want, Cases),
              run_export(Harness, Wasm, Name, Got),
              Got =\= Want ),
            Failures),
    ( Failures == []
    ->  true
    ;   format(user_error, "~n[wat t4 mismatches]~n~q~n", [Failures]),
        throw(wam_wat_t4_failed(Failures))
    ).

:- end_tests(wam_wat_lowered_t4).

% --- build + inject + run helpers (wat2wasm + node) ---

build_injected_module(Preds, Cases, WasmFile, Harness) :-
    Dir = 'output/test_wam_wat_t4',
    ( exists_directory(Dir) -> delete_directory_and_contents(Dir) ; true ),
    make_directory_path(Dir),
    findall(user:P, member(P, Preds), QPreds),
    atom_concat(Dir, '/t4.wat', WatFile),
    write_wam_wat_project(QPreds, [module_name(wat_t4), emit_mode(functions)], WatFile),
    read_file_to_string(WatFile, Src, []),
    wat_heap_base(Src, Heap),
    findall(Snip, ( member(C, Cases), case_export_snippet(Heap, C, Snip) ), Snips),
    atomic_list_concat(Snips, '\n', Injection),
    inject_before_last_paren(Src, Injection, Src2),
    atom_concat(Dir, '/t4_inj.wat', InjWat),
    setup_call_cleanup(open(InjWat, write, S, [encoding(utf8)]),
                       write(S, Src2), close(S)),
    atom_concat(Dir, '/t4_inj.wasm', WasmFile),
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
    ; throw(wam_wat_t4_compile_failed(CompileOut)) ).

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
    ; throw(wam_wat_t4_run_failed(Export, RunOut)) ).

ensure_harness(Path) :-
    Path = 'output/test_wam_wat_t4_harness.js',
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
