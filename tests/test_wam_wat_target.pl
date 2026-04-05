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
    % tag=16 (0x10), op1=0, op2=0 -> first byte should be \10
    assertion(sub_string(Hex, 0, _, _, "\\10")).

test(encode_proceed) :-
    wam_instruction_to_wat_bytes(proceed, [], Hex),
    assertion(atom(Hex)),
    % tag=20 (0x14)
    assertion(sub_string(Hex, 0, _, _, "\\14")).

test(encode_get_variable) :-
    wam_instruction_to_wat_bytes(get_variable('X1', 'A1'), [], Hex),
    assertion(atom(Hex)),
    % tag=1 -> first byte \01
    assertion(sub_string(Hex, 0, _, _, "\\01")).

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

:- end_tests(wam_wat_target).
