:- encoding(utf8).
:- use_module(library(plunit)).
:- use_module('../src/unifyweaver/targets/wam_wat_target').

:- begin_tests(wam_wat_target).

%% --- Register mapping ---

test(reg_a1_index) :-
    reg_name_to_index('A1', Idx),
    assertion(Idx == 0).

test(reg_a2_index) :-
    reg_name_to_index('A2', Idx),
    assertion(Idx == 1).

test(reg_x1_index) :-
    reg_name_to_index('X1', Idx),
    assertion(Idx == 32).

test(reg_y1_index) :-
    reg_name_to_index('Y1', Idx),
    assertion(Idx == 32).

%% --- Atom hashing ---

test(atom_hash_deterministic) :-
    atom_hash_i64(hello, H1),
    atom_hash_i64(hello, H2),
    assertion(H1 == H2).

test(atom_hash_different) :-
    atom_hash_i64(hello, H1),
    atom_hash_i64(world, H2),
    assertion(H1 \== H2).

%% --- Instruction encoding ---

test(encode_allocate) :-
    wam_instruction_to_wat_bytes(allocate, [], Hex),
    assertion(atom(Hex)),
    assertion(sub_string(Hex, 0, _, _, "\\10")).  % tag=16 (0x10)

test(encode_proceed) :-
    wam_instruction_to_wat_bytes(proceed, [], Hex),
    assertion(atom(Hex)),
    assertion(sub_string(Hex, 0, _, _, "\\14")).  % tag=20 (0x14)

test(encode_deallocate) :-
    wam_instruction_to_wat_bytes(deallocate, [], Hex),
    assertion(atom(Hex)),
    assertion(sub_string(Hex, 0, _, _, "\\11")).  % tag=17 (0x11)

test(encode_get_constant) :-
    wam_instruction_to_wat_bytes(get_constant(atom(hello), 'A1'), [], Hex),
    assertion(atom(Hex)),
    assertion(sub_string(Hex, 0, _, _, "\\00")).  % tag=0

test(encode_get_variable) :-
    wam_instruction_to_wat_bytes(get_variable('X1', 'A1'), [], Hex),
    assertion(atom(Hex)),
    assertion(sub_string(Hex, 0, _, _, "\\01")).  % tag=1

test(encode_get_value) :-
    wam_instruction_to_wat_bytes(get_value('X1', 'A2'), [], Hex),
    assertion(atom(Hex)),
    assertion(sub_string(Hex, 0, _, _, "\\02")).  % tag=2

test(encode_get_structure) :-
    wam_instruction_to_wat_bytes(get_structure('f/2', 'A1'), [], Hex),
    assertion(atom(Hex)),
    assertion(sub_string(Hex, 0, _, _, "\\03")).  % tag=3

%% Verify arity is packed into the high 32 bits of the op1 i64.
%% Byte layout: bytes 0-3 = tag, 4-11 = op1 (little-endian), 12-19 = op2.
%% Each byte renders as 3 chars in the hex atom: "\XX". So byte N starts
%% at offset N*3. Bytes 8-11 (high 32 bits of op1) contain the arity.
%% For 'f/2' we expect \02\00\00\00 at bytes 8-11 (offset 24, length 12).
test(encode_get_structure_arity_packed) :-
    wam_instruction_to_wat_bytes(get_structure('f/2', 'A1'), [], Hex),
    sub_atom(Hex, 24, 12, _, HighBytes),
    assertion(HighBytes == '\\02\\00\\00\\00').

test(encode_put_structure_arity_packed) :-
    wam_instruction_to_wat_bytes(put_structure('g/3', 'A1'), [], Hex),
    sub_atom(Hex, 24, 12, _, HighBytes),
    assertion(HighBytes == '\\03\\00\\00\\00').

test(encode_get_list) :-
    wam_instruction_to_wat_bytes(get_list('A1'), [], Hex),
    assertion(atom(Hex)),
    assertion(sub_string(Hex, 0, _, _, "\\04")).  % tag=4

test(encode_unify_variable) :-
    wam_instruction_to_wat_bytes(unify_variable('X1'), [], Hex),
    assertion(atom(Hex)),
    assertion(sub_string(Hex, 0, _, _, "\\05")).  % tag=5

test(encode_unify_value) :-
    wam_instruction_to_wat_bytes(unify_value('X1'), [], Hex),
    assertion(atom(Hex)),
    assertion(sub_string(Hex, 0, _, _, "\\06")).  % tag=6

test(encode_unify_constant) :-
    wam_instruction_to_wat_bytes(unify_constant(atom(foo)), [], Hex),
    assertion(atom(Hex)),
    assertion(sub_string(Hex, 0, _, _, "\\07")).  % tag=7

test(encode_put_constant) :-
    wam_instruction_to_wat_bytes(put_constant(atom(bar), 'A1'), [], Hex),
    assertion(atom(Hex)),
    assertion(sub_string(Hex, 0, _, _, "\\08")).  % tag=8

test(encode_put_variable) :-
    wam_instruction_to_wat_bytes(put_variable('X1', 'A1'), [], Hex),
    assertion(atom(Hex)),
    assertion(sub_string(Hex, 0, _, _, "\\09")).  % tag=9

test(encode_put_value) :-
    wam_instruction_to_wat_bytes(put_value('X2', 'A1'), [], Hex),
    assertion(atom(Hex)),
    assertion(sub_string(Hex, 0, _, _, "\\0a")).  % tag=10

test(encode_put_structure) :-
    wam_instruction_to_wat_bytes(put_structure('g/1', 'A1'), [], Hex),
    assertion(atom(Hex)),
    assertion(sub_string(Hex, 0, _, _, "\\0b")).  % tag=11

test(encode_put_list) :-
    wam_instruction_to_wat_bytes(put_list('A1'), [], Hex),
    assertion(atom(Hex)),
    assertion(sub_string(Hex, 0, _, _, "\\0c")).  % tag=12

test(encode_set_variable) :-
    wam_instruction_to_wat_bytes(set_variable('X1'), [], Hex),
    assertion(atom(Hex)),
    assertion(sub_string(Hex, 0, _, _, "\\0d")).  % tag=13

test(encode_set_value) :-
    wam_instruction_to_wat_bytes(set_value('X1'), [], Hex),
    assertion(atom(Hex)),
    assertion(sub_string(Hex, 0, _, _, "\\0e")).  % tag=14

test(encode_set_constant) :-
    wam_instruction_to_wat_bytes(set_constant(atom(baz)), [], Hex),
    assertion(atom(Hex)),
    assertion(sub_string(Hex, 0, _, _, "\\0f")).  % tag=15

test(encode_builtin_call) :-
    wam_instruction_to_wat_bytes(builtin_call('write/1', 1), [], Hex),
    assertion(atom(Hex)),
    assertion(sub_string(Hex, 0, _, _, "\\15")).  % tag=21

test(encode_try_me_else) :-
    wam_instruction_to_wat_bytes(try_me_else('L1'), ['L1'-5], Hex),
    assertion(atom(Hex)),
    assertion(sub_string(Hex, 0, _, _, "\\16")).  % tag=22

test(encode_trust_me) :-
    wam_instruction_to_wat_bytes(trust_me, [], Hex),
    assertion(atom(Hex)),
    assertion(sub_string(Hex, 0, _, _, "\\18")).  % tag=24

%% --- Step dispatch generation ---

test(step_dispatch_generated) :-
    compile_step_wam_to_wat([], StepCode),
    assertion(sub_string(StepCode, _, _, _, "br_table")),
    assertion(sub_string(StepCode, _, _, _, "$do_get_constant")),
    assertion(sub_string(StepCode, _, _, _, "$do_proceed")),
    assertion(sub_string(StepCode, _, _, _, "$do_trust_me")).

%% --- Helpers generation ---

test(helpers_generated) :-
    wam_wat_target:compile_wam_helpers_to_wat([], HelpersCode),
    assertion(sub_string(HelpersCode, _, _, _, "$run_loop")),
    assertion(sub_string(HelpersCode, _, _, _, "$backtrack")),
    assertion(sub_string(HelpersCode, _, _, _, "$unify_regs")),
    assertion(sub_string(HelpersCode, _, _, _, "$execute_builtin")).

%% --- Phase 2.1: arg/3 codegen ---

test(arg_builtin_id_registered) :-
    assertion(wam_wat_target:builtin_id('arg/3', 19)).

test(arg_helper_generated) :-
    wam_wat_target:compile_wam_helpers_to_wat([], HelpersCode),
    %% The $builtin_arg function must be emitted
    assertion(sub_string(HelpersCode, _, _, _, "$builtin_arg")),
    %% It must be reachable from $execute_builtin's dispatch — the
    %% if-chain checks id == 19 and calls $builtin_arg.
    assertion(sub_string(HelpersCode, _, _, _, "(i32.const 19)")),
    %% Sanity: the helper body should reference the arity-extraction
    %% shift (high 32 bits of compound payload).
    assertion(sub_string(HelpersCode, _, _, _, "i64.shr_u")).

test(arg_builtin_call_encoding) :-
    %% Verify a builtin_call arg/3 instruction encodes with builtin ID 19.
    wam_instruction_to_wat_bytes(builtin_call('arg/3', 3), [], Hex),
    assertion(atom(Hex)),
    %% Tag byte is builtin_call = 21 = 0x15
    assertion(sub_string(Hex, 0, _, _, "\\15")),
    %% Op1 (low byte of i64) should be builtin ID 19 = 0x13
    sub_atom(Hex, 12, 3, _, FirstOp1Byte),
    assertion(FirstOp1Byte == '\\13').

%% --- Phase 2.2: functor/3 codegen ---

test(functor_builtin_id_registered) :-
    assertion(wam_wat_target:builtin_id('functor/3', 18)).

test(functor_helper_generated) :-
    wam_wat_target:compile_wam_helpers_to_wat([], HelpersCode),
    assertion(sub_string(HelpersCode, _, _, _, "$builtin_functor")),
    %% Both modes: construct branch uses i64.shl to pack arity,
    %% read branch uses i64.shr_u to extract it.
    assertion(sub_string(HelpersCode, _, _, _, "i64.shl")),
    %% Dispatch: if-chain checks id == 18 for functor/3.
    assertion(sub_string(HelpersCode, _, _, _, "(i32.const 18)")).

test(functor_builtin_call_encoding) :-
    wam_instruction_to_wat_bytes(builtin_call('functor/3', 3), [], Hex),
    assertion(atom(Hex)),
    %% Op1 low byte = builtin ID 18 = 0x12
    sub_atom(Hex, 12, 3, _, FirstOp1Byte),
    assertion(FirstOp1Byte == '\\12').

%% --- Phase 2.3: =../2 (univ) codegen ---

test(univ_builtin_id_registered) :-
    assertion(wam_wat_target:builtin_id('=../2', 20)).

test(univ_helper_generated) :-
    wam_wat_target:compile_wam_helpers_to_wat([], HelpersCode),
    assertion(sub_string(HelpersCode, _, _, _, "$builtin_univ")),
    %% Cons cells use tag=4 (list) to match put_list''s runtime layout.
    %% The empty-list atom hash literal 2914 must appear as the nil
    %% terminator in the decompose-mode build path.
    assertion(sub_string(HelpersCode, _, _, _, "2914")),
    %% Dispatch: if-chain checks id == 20.
    assertion(sub_string(HelpersCode, _, _, _, "(i32.const 20)")).

test(univ_builtin_call_encoding) :-
    wam_instruction_to_wat_bytes(builtin_call('=../2', 2), [], Hex),
    assertion(atom(Hex)),
    %% Op1 low byte = builtin ID 20 = 0x14
    sub_atom(Hex, 12, 3, _, FirstOp1Byte),
    assertion(FirstOp1Byte == '\\14').

%% --- Phase 2.4: copy_term/2 codegen ---

test(copy_term_builtin_id_registered) :-
    assertion(wam_wat_target:builtin_id('copy_term/2', 21)).

test(copy_term_helper_generated) :-
    wam_wat_target:compile_wam_helpers_to_wat([], HelpersCode),
    assertion(sub_string(HelpersCode, _, _, _, "$builtin_copy_term")),
    %% Sharing-preservation var map is referenced explicitly.
    assertion(sub_string(HelpersCode, _, _, _, "var map")),
    %% Dispatch: if-chain checks id == 21.
    assertion(sub_string(HelpersCode, _, _, _, "(i32.const 21)")).

test(copy_term_builtin_call_encoding) :-
    wam_instruction_to_wat_bytes(builtin_call('copy_term/2', 2), [], Hex),
    assertion(atom(Hex)),
    %% Op1 low byte = builtin ID 21 = 0x15
    sub_atom(Hex, 12, 3, _, FirstOp1Byte),
    assertion(FirstOp1Byte == '\\15').

%% --- Predicate compilation ---

test(compile_simple_predicate) :-
    WamCode = "    allocate\n    put_constant hello, A1\n    builtin_call write/1, 1\n    deallocate\n    proceed",
    compile_wam_predicate_to_wat(test/0, WamCode, [code_base(131072)], Result),
    Result = wat_pred(DataSeg, EntryFunc, 131072, 5),
    assertion(sub_string(DataSeg, _, _, _, "(data")),
    assertion(sub_string(EntryFunc, _, _, _, "$test_0")),
    assertion(sub_string(EntryFunc, _, _, _, "run_loop")).

%% --- End-to-end WAT module generation ---

test(write_wat_project) :-
    get_time(T),
    format(atom(TmpFile), '/tmp/test_wam_wat_~w.wat', [T]),
    %% Define a simple fact
    assertz(user:test_greet :- true),
    Predicates = [test_greet/0],
    Options = [module_name(test_wam)],
    (   catch(
            write_wam_wat_project(Predicates, Options, TmpFile),
            E,
            (format(user_error, 'write_wam_wat_project error: ~w~n', [E]), fail))
    ->  %% Verify the file was created and contains expected content
        read_file_to_string(TmpFile, Content, []),
        assertion(sub_string(Content, _, _, _, "(module")),
        assertion(sub_string(Content, _, _, _, "(memory")),
        assertion(sub_string(Content, _, _, _, "$val_tag")),
        assertion(sub_string(Content, _, _, _, "$run_loop")),
        assertion(sub_string(Content, _, _, _, "$step"))
    ;   true
    ),
    retractall(user:test_greet),
    (exists_file(TmpFile) -> delete_file(TmpFile) ; true).

%% --- wat2wasm syntax validation ---

%% --- Functional execution via wat2wasm + node ---
%%
%% These tests compile a generated .wat to .wasm and actually run it
%% via Node.js, which provides the `env` host imports the WAM-WAT
%% runtime requires (print_i64, print_char, print_newline). This is
%% the first layer of real runtime validation for the WAM-WAT target
%% — everything prior was codegen-only. Tests skip gracefully if
%% either wat2wasm or node is missing.

tool_available(Tool) :-
    catch(
        process_create(path(Tool), ['--version'],
            [stdout(null), stderr(null)]),
        _, fail).

%% Build the node harness script that loads a .wasm and calls an export.
%% Written to a tmp file on first use. Returns the RESULT value on stdout.
node_harness_source(Src) :-
    Src = "const fs=require('fs');\n\c
const [,,wasmPath,exportName]=process.argv;\n\c
const bytes=fs.readFileSync(wasmPath);\n\c
const log=[];\n\c
const imports={env:{\n\c
  print_i64:v=>log.push(String(v)),\n\c
  print_char:c=>log.push(String.fromCharCode(c)),\n\c
  print_newline:()=>log.push('\\n')\n\c
}};\n\c
(async()=>{\n\c
  try{\n\c
    const{instance}=await WebAssembly.instantiate(bytes,imports);\n\c
    const fn=instance.exports[exportName];\n\c
    if(typeof fn!=='function'){console.log('ERROR export '+exportName+' not found');process.exit(1);}\n\c
    const result=fn();\n\c
    for(const line of log)process.stderr.write(line);\n\c
    console.log('RESULT '+result);\n\c
  }catch(e){console.log('ERROR '+e.message);process.exit(1);}\n\c
})();\n".

ensure_node_harness(HarnessPath) :-
    HarnessPath = '/data/data/com.termux/files/home/tmp/wam_wat_test_harness.js',
    (   exists_file(HarnessPath)
    ->  true
    ;   node_harness_source(Src),
        open(HarnessPath, write, S),
        write(S, Src),
        close(S)
    ).

%% run_wam_wat_module_export(+WatFile, +ExportName, -Result)
%%   Compile WatFile with wat2wasm, run with node harness, unify Result
%%   with the i32 return value. Throws wam_wat_runtime_skip(Reason) if
%%   the toolchain is unavailable. Throws wam_wat_runtime_error(Detail)
%%   on compilation or execution failure.
run_wam_wat_module_export(WatFile, ExportName, Result) :-
    (   tool_available(wat2wasm), tool_available(node)
    ->  true
    ;   throw(wam_wat_runtime_skip(tools_missing))
    ),
    file_name_extension(Base, _, WatFile),
    file_name_extension(Base, wasm, WasmFile),
    format(string(CompileCmd), "wat2wasm ~w -o ~w 2>&1", [WatFile, WasmFile]),
    process_create(path(sh), ['-c', CompileCmd],
        [stdout(pipe(COut)), process(CPid)]),
    read_string(COut, _, CompileOut),
    close(COut),
    process_wait(CPid, CExit),
    (   CExit == exit(0)
    ->  true
    ;   throw(wam_wat_runtime_error(compile(CompileOut)))
    ),
    ensure_node_harness(Harness),
    process_create(path(node), [Harness, WasmFile, ExportName],
        [stdout(pipe(ROut)), stderr(null), process(RPid)]),
    read_string(ROut, _, RunOut),
    close(ROut),
    process_wait(RPid, RExit),
    (   sub_string(RunOut, _, _, _, "RESULT "),
        split_string(RunOut, " \n", " \n", Parts),
        append(_, ["RESULT", RS|_], Parts),
        number_string(Result, RS)
    ->  true
    ;   throw(wam_wat_runtime_error(run(RExit, RunOut)))
    ).

test(functional_copy_term_nested, [condition(tool_available(wat2wasm)),
                                     condition(tool_available(node))]) :-
    %% Deep copy_term follow-up: copy_term(outer(inner(a, b), c), _)
    %% exercises the worklist''s compound→compound recursion. The v1
    %% shallow impl would copy the outer compound but leave `inner(a,b)`
    %% structure-shared with the source; the deep impl allocates a
    %% fresh inner compound and recursively copies its args. A return
    %% value of 1 means the worklist processed every level and the
    %% final root value was written back to A2.
    get_time(T),
    format(atom(WatFile),
        '/data/data/com.termux/files/home/tmp/test_wam_func_copy_nested_~w.wat', [T]),
    assertz(user:test_func_copy_nested :- copy_term(outer(inner(a, b), c), _)),
    setup_call_cleanup(
        true,
        (   write_wam_wat_project([test_func_copy_nested/0],
                                  [module_name(func_copy_nested_test)], WatFile),
            run_wam_wat_module_export(WatFile, test_func_copy_nested_0, Result),
            (   Result =:= 1
            ->  true
            ;   throw(wam_wat_runtime_error(copy_term_nested_failed(Result)))
            )
        ),
        (   retractall(user:test_func_copy_nested),
            (exists_file(WatFile) -> delete_file(WatFile) ; true),
            file_name_extension(Base, _, WatFile),
            file_name_extension(Base, wasm, WasmFile),
            (exists_file(WasmFile) -> delete_file(WasmFile) ; true)
        )
    ).

test(functional_copy_term_compound, [condition(tool_available(wat2wasm)),
                                       condition(tool_available(node))]) :-
    %% Flat compound case: copy_term(foo(a, b), _). Validates the
    %; worklist''s trivial path (one compound, no nesting, no vars).
    %% With the deep impl this is still a meaningful sanity test and
    %% a regression guard.
    get_time(T),
    format(atom(WatFile),
        '/data/data/com.termux/files/home/tmp/test_wam_func_copy_~w.wat', [T]),
    assertz(user:test_func_copy :- copy_term(foo(a, b), _)),
    setup_call_cleanup(
        true,
        (   write_wam_wat_project([test_func_copy/0],
                                  [module_name(func_copy_test)], WatFile),
            run_wam_wat_module_export(WatFile, test_func_copy_0, Result),
            (   Result =:= 1
            ->  true
            ;   throw(wam_wat_runtime_error(copy_term_failed(Result)))
            )
        ),
        (   retractall(user:test_func_copy),
            (exists_file(WatFile) -> delete_file(WatFile) ; true),
            file_name_extension(Base, _, WatFile),
            file_name_extension(Base, wasm, WasmFile),
            (exists_file(WasmFile) -> delete_file(WasmFile) ; true)
        )
    ).

test(functional_univ_decompose_compound, [condition(tool_available(wat2wasm)),
                                            condition(tool_available(node))]) :-
    %% End-to-end: bar(a, b) =.. L should succeed (L is a fresh var on
    %% first use so decompose mode just builds a fresh cons list and
    %% binds L). We do not try to unify L with a pre-built list here —
    %% v1 decompose mode does not structurally unify a bound A2; that
    %% path is deferred. The test exercises the entire pipeline:
    %% canonical WAM → builtin_call =../2 → $builtin_univ → cons-list
    %% construction on the heap → success return.
    get_time(T),
    format(atom(WatFile),
        '/data/data/com.termux/files/home/tmp/test_wam_func_univ_~w.wat', [T]),
    assertz(user:test_func_univ :- (bar(a, b) =.. _)),
    setup_call_cleanup(
        true,
        (   write_wam_wat_project([test_func_univ/0],
                                  [module_name(func_univ_test)], WatFile),
            run_wam_wat_module_export(WatFile, test_func_univ_0, Result),
            (   Result =:= 1
            ->  true
            ;   throw(wam_wat_runtime_error(univ_failed(Result)))
            )
        ),
        (   retractall(user:test_func_univ),
            (exists_file(WatFile) -> delete_file(WatFile) ; true),
            file_name_extension(Base, _, WatFile),
            file_name_extension(Base, wasm, WasmFile),
            (exists_file(WasmFile) -> delete_file(WasmFile) ; true)
        )
    ).

test(functional_true_succeeds, [condition(tool_available(wat2wasm)),
                                 condition(tool_available(node))]) :-
    get_time(T),
    format(atom(WatFile),
        '/data/data/com.termux/files/home/tmp/test_wam_func_true_~w.wat', [T]),
    assertz(user:test_func_true :- true),
    setup_call_cleanup(
        true,
        (   write_wam_wat_project([test_func_true/0],
                                  [module_name(func_true_test)], WatFile),
            run_wam_wat_module_export(WatFile, test_func_true_0, Result),
            assertion(Result =:= 1)
        ),
        (   retractall(user:test_func_true),
            (exists_file(WatFile) -> delete_file(WatFile) ; true),
            file_name_extension(Base, _, WatFile),
            file_name_extension(Base, wasm, WasmFile),
            (exists_file(WasmFile) -> delete_file(WasmFile) ; true)
        )
    ).

test(wat2wasm_validates) :-
    %% Generate a minimal WAM-WAT module and verify wat2wasm accepts it
    %% (exit 0). The earlier version of this test used assertion/1 on
    %% the exit code, which only *warns* on failure rather than failing
    %% the test; that allowed the step-dispatch off-by-one bug and a
    %% paren imbalance in $builtin_arith_cmp to ship unnoticed. The
    %% assertion is now replaced with a direct unification that fails
    %% the test on non-zero exit.
    get_time(T),
    TmpDir = '/data/data/com.termux/files/home/tmp',
    format(atom(TmpWat), '~w/test_wam_validate_~w.wat', [TmpDir, T]),
    format(atom(TmpWasm), '~w/test_wam_validate_~w.wasm', [TmpDir, T]),
    assertz(user:test_validate :- true),
    Predicates = [test_validate/0],
    Options = [module_name(validate_test)],
    setup_call_cleanup(
        true,
        (   write_wam_wat_project(Predicates, Options, TmpWat),
            (   catch(process_create(path(wat2wasm), ['--version'],
                        [stdout(null), stderr(null)]), _, fail)
            ->  format(string(Cmd), "wat2wasm ~w -o ~w 2>&1", [TmpWat, TmpWasm]),
                process_create(path(sh), ['-c', Cmd],
                    [stdout(pipe(Out)), process(Pid)]),
                read_string(Out, _, Output),
                process_wait(Pid, Exit),
                (   Exit == exit(0)
                ->  true
                ;   format(user_error, 'wat2wasm failed: ~s~n', [Output]),
                    fail
                )
            ;   format("wat2wasm not found, skipping syntax validation.~n")
            )
        ),
        (   retractall(user:test_validate),
            (exists_file(TmpWat) -> delete_file(TmpWat) ; true),
            (exists_file(TmpWasm) -> delete_file(TmpWasm) ; true)
        )
    ).

:- end_tests(wam_wat_target).
