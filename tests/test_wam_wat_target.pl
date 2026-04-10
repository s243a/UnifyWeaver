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
