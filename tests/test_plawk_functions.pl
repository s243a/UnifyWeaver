:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)

:- use_module(library(plunit)).
:- use_module(library(filesex), [make_directory_path/1]).
:- use_module(library(process)).
:- use_module('helpers/smoke_paths', [tmp_root/1, clean_dir/1]).
:- use_module('../src/unifyweaver/targets/wam_llvm_target').
:- use_module('../examples/plawk/parser/plawk_parser').
:- use_module('../examples/plawk/codegen/plawk_native_codegen').

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

:- begin_tests(plawk_functions, [condition(clang_available)]).

test(functions_desugar_to_prolog_clauses) :-
    plawk_parse_source("BEGIN { BINFMT = \"i64\" }\nfunction f(a, b) { return a + b * 2 }\n{ n++ }\nEND { print n }\n",
        _, [Clause]),
    % awk precedence: * binds tighter than +
    assertion(Clause = (f(_A, _B, R) :- R is _ + _ * 2)),
    Clause = (f(3, 4, Result) :- Goal),
    call(Goal),
    assertion(Result =:= 11).

test(percent_maps_to_mod_and_parens_group) :-
    plawk_parse_source("BEGIN { BINFMT = \"i64\" }\nfunction g(a, b) { return (a + b) % 7 }\n{ n++ }\nEND { print n }\n",
        _, [(g(A, B, R) :- R is Goal)]),
    assertion(Goal == mod(A + B, 7)),
    ( A = 5, B = 9, V is Goal, assertion(V =:= 0) ).

test(function_rejections) :-
    % identifiers must be parameters
    assertion(\+ plawk_parse_source("BEGIN { BINFMT = \"i64\" }\nfunction bad(a) { return a + q }\n{ n++ }\nEND { print n }\n", _, _)),
    % the body is a single return expression
    assertion(\+ plawk_parse_source("BEGIN { BINFMT = \"i64\" }\nfunction bad(a) { x = a }\n{ n++ }\nEND { print n }\n", _, _)).

test(surface_functions_end_to_end) :-
    % integer function in i64 context, float-constant function through
    % float(...): s = (3*3+1) + (5*5+1) = 36, w = 1.5 + 2.5 = 4.
    Src = "BEGIN { BINFMT = \"i64 f64\" }\nfunction plawk_fnt_scale(a, b) { return a * b + 1 }\nfunction plawk_fnt_half(x) { return x * 0.5 }\n{ s += plawk_fnt_scale($1, $1)\n  w += float(plawk_fnt_half($1)) }\nEND { print s, w }\n",
    run_fn_smoke(Src, [rec(3, 0.25), rec(5, 0.25)], "36 4\n").

test(surface_functions_mix_with_prolog_blocks) :-
    % sugar and explicit clauses share one clause list and one bridge
    Src = "@prolog\nplawk_fnt_hot(X) :- X > 10.\n@end\nBEGIN { BINFMT = \"i64 f64\" }\nfunction plawk_fnt_twice(a) { return a * 2 }\nplawk_fnt_hot($1) { s += plawk_fnt_twice($1) }\nEND { print s }\n",
    run_fn_smoke(Src, [rec(3, 0.25), rec(20, 0.25), rec(11, 0.25)], "62\n").

:- end_tests(plawk_functions).

% --- helpers ---------------------------------------------------------------

write_i64_le(Out, V) :-
    V64 is V /\ 0xFFFFFFFFFFFFFFFF,
    forall(between(0, 7, I),
        ( Byte is (V64 >> (8 * I)) /\ 0xFF, put_byte(Out, Byte) )).

double_bits(0.25, 0x3FD0000000000000).

write_fn_records(Path, Recs) :-
    setup_call_cleanup(
        open(Path, write, Out, [type(binary)]),
        forall(member(rec(I, F), Recs),
            ( write_i64_le(Out, I),
              double_bits(F, Bits),
              write_i64_le(Out, Bits) )),
        close(Out)).

run_fn_smoke(Src, Recs, ExpectedOutput) :-
    tmp_root(Root),
    directory_file_path(Root, 'uw_plawk_functions', Dir),
    clean_dir(Dir),
    make_directory_path(Dir),
    directory_file_path(Dir, 'input.bin', InputPath),
    plawk_parse_source(Src, Program, Clauses),
    plawk_prolog_block_preds(Clauses, Preds),
    directory_file_path(Dir, 'probe.ll', LLPath),
    write_wam_llvm_project(Preds, [module_name('plawk_functions')], LLPath),
    wam_llvm_last_compile_counts(InstrCount, LabelCount),
    plawk_program_native_driver_ir(Program, InputPath,
        [wam_vm(InstrCount, LabelCount)], DriverIR),
    setup_call_cleanup(
        open(LLPath, append, Out, [encoding(utf8)]),
        ( nl(Out), write(Out, DriverIR) ),
        close(Out)),
    directory_file_path(Dir, 'probe_bin', BinPath),
    format(atom(Cmd), 'clang -w ~w -o ~w -lm 2>&1', [LLPath, BinPath]),
    process_create(path(sh), ['-c', Cmd],
                   [stdout(pipe(BuildS)), stderr(std), process(BuildPid)]),
    read_string(BuildS, _, BuildOut),
    close(BuildS),
    process_wait(BuildPid, BuildStatus),
    ( BuildStatus == exit(0)
    -> true
    ;  format(user_error, "~n[plawk functions build output]~n~w~n", [BuildOut]),
       throw(plawk_functions_build_failed(BuildStatus))
    ),
    write_fn_records(InputPath, Recs),
    process_create(BinPath, [],
                   [stdout(pipe(Stdout)), stderr(std), process(Pid)]),
    read_string(Stdout, _, OutStr),
    close(Stdout),
    process_wait(Pid, Status),
    assertion(Status == exit(0)),
    assertion(OutStr == ExpectedOutput),
    !.
