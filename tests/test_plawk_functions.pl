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
    % auto-coerce prefixes the body with per-param numeric coercion goals; the
    % final conjunct is the arithmetic. awk precedence: * binds tighter than +.
    Clause = (f(_A, _B, R) :- Body),
    fn_body_last(Body, (R is Arith)),
    assertion(Arith = _ + _ * 2),
    Clause = (f(3, 4, Result) :- Goal),
    call(Goal),
    assertion(Result =:= 11).

test(percent_maps_to_mod_and_parens_group) :-
    plawk_parse_source("BEGIN { BINFMT = \"i64\" }\nfunction g(a, b) { return (a + b) % 7 }\n{ n++ }\nEND { print n }\n",
        _, [Clause]),
    Clause = (g(_A, _B, R) :- Body),
    fn_body_last(Body, (R is Goal)),
    % % maps to mod and the parens group the sum: mod(_+_, 7)
    assertion(Goal = mod(_ + _, 7)),
    Clause = (g(5, 9, V) :- CallGoal),
    call(CallGoal),
    assertion(V =:= 0).

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

% --- optional arg typing (typed-fast path over auto-coerce) ------------------
% `function f(num x)` declares x numeric, so the synthesized clause skips the
% auto-coercion goal -- the head var IS the value var (Principle 2: static type
% knowledge elides the runtime coercion). Layered on auto-coerce, which stays
% the default for unannotated params.

test(typed_param_skips_coercion) :-
    plawk_parse_source("BEGIN { BINFMT = \"i64\" }\nfunction f(num x) { return x * 2 }\n{ n++ }\nEND { print n }\n",
        _, [Clause]),
    % no coercion prefix: the body is exactly the arithmetic
    assertion(Clause = (f(X, R) :- R is X * 2)),
    Clause = (f(5, Result) :- Goal),
    call(Goal),
    assertion(Result =:= 10).

test(untyped_param_still_coerces) :-
    plawk_parse_source("BEGIN { BINFMT = \"i64\" }\nfunction g(x) { return x * 2 }\n{ n++ }\nEND { print n }\n",
        _, [Clause]),
    Clause = (g(_X, R) :- Body),
    fn_body_last(Body, (R is _Arith)),
    % the body has a coercion conjunct before the arithmetic (not a bare `is`)
    assertion(Body = (_Coerce, _Rest)).

test(mixed_typed_untyped_params) :-
    plawk_parse_source("BEGIN { BINFMT = \"i64\" }\nfunction h(num a, b) { return a + b }\n{ n++ }\nEND { print n }\n",
        _, [Clause]),
    % a is typed (no coercion), b is untyped (one coercion goal) -> exactly one
    % coercion conjunct precedes the `is`
    Clause = (h(_A, _B, R) :- (Coerce, R is _Arith)),
    assertion(Coerce = (number(_) -> _ ; _)),
    Clause = (h(4, 6, V) :- FullGoal),
    call(FullGoal),
    assertion(V =:= 10).

% A typed function used end-to-end in i64 (BINFMT) mode: the field is already a
% number, so the typed-fast path runs with no coercion.
test(typed_function_binfmt_end_to_end) :-
    Src = "BEGIN { BINFMT = \"i64 f64\" }\nfunction plawk_fnt_scale(num a, num b) { return a * b + 1 }\n{ s += plawk_fnt_scale($1, $1) }\nEND { print s }\n",
    run_fn_smoke(Src, [rec(3, 0.25), rec(5, 0.25)], "36\n").

% --- auto-coerce (awk semantics): a function called on TEXT fields -----------
% In text mode (no BINFMT) `$1` is an atom. awk auto-coerces a value used
% numerically, so `print dbl($1)` must coerce the field to a number before the
% arithmetic. The synthesized clause wraps each param in
% `(number(H) -> V = H ; atom_number(H, V))`, so text fields Just Work.

test(auto_coerce_unary_text_field) :-
    Src = "function dbl(x) { return x * 2 }\n{ print dbl($1) }\n",
    build_run_text('acdbl', Src, "5\n10\n", Out, St),
    assertion(St == 0),
    assertion(Out == "10\n20\n").

test(auto_coerce_binary_text_fields) :-
    Src = "function add(a, b) { return a + b }\n{ print add($1, $2) }\n",
    build_run_text('acadd', Src, "5 7\n10 23\n", Out, St),
    assertion(St == 0),
    assertion(Out == "12\n33\n").

:- end_tests(plawk_functions).

% --- helpers ---------------------------------------------------------------

% Peel a body conjunction down to its final conjunct.
fn_body_last((_, Rest), Last) :- !, fn_body_last(Rest, Last).
fn_body_last(Goal, Goal).

% Build a text-mode plawk program via the CLI and run it on Input (stdin),
% capturing stdout. Used for the auto-coerce end-to-end checks.
build_run_text(Name, Src, Input, Out, RunStatus) :-
    tmp_root(Root),
    directory_file_path(Root, 'uw_plawk_functions', Dir),
    ( exists_directory(Dir) -> true ; make_directory_path(Dir) ),
    directory_file_path(Dir, Name, Prog0),
    atom_concat(Prog0, '.plawk', Prog),
    setup_call_cleanup(open(Prog, write, S, [encoding(utf8)]),
        write(S, Src), close(S)),
    atom_concat(Prog0, '_bin', Bin),
    process_create(path(swipl),
        ['examples/plawk/bin/plawk', build, Prog, '-o', Bin],
        [stdout(null), stderr(null), process(BPid)]),
    process_wait(BPid, exit(0)),
    process_create(Bin, [],
        [stdin(pipe(In)), stdout(pipe(RS)), stderr(std), process(RPid)]),
    format(In, "~w", [Input]),
    close(In),
    read_string(RS, _, Out),
    close(RS),
    process_wait(RPid, exit(RunStatus)).

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
