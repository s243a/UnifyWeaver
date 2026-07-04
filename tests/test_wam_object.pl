:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% Phase 5 (JIT) slice 1: runtime-loadable WAM objects (.wamo).
% Covers the writer (subset validation + byte-stream shape) and, when
% clang is available, the full round trip: write a .wamo grammar, build a
% host binary carrying the loader, load the object at runtime, and run it.

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(filesex), [make_directory_path/1]).
:- use_module('../src/unifyweaver/targets/wam_llvm_target').
:- use_module('../src/unifyweaver/targets/wam_target').

% --- grammar predicates the tests compile into objects ---------------------
% Accumulator sum: exercises try_me_else / trust_me / get_list /
% unify_variable / builtin_call is/2 / execute (the tier2 loadable subset).
sum3([], A, A).
sum3([H|T], A, R) :- A1 is A + H, sum3(T, A1, R).
answer(R) :- sum3([100, 10, 9], 0, R).          % -> 119
answer_swapped(R) :- sum3([7, 8, 5, 1000], 0, R). % -> 1020

uses_float(X) :- X is 1.5 + 2.5.                % float constant -> rejected

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

:- begin_tests(wam_object).

% The writer emits a well-formed .wamo byte stream: "WAMO" magic, version
% 1, and the section counts we can recover by re-parsing the tokens.
test(encode_produces_well_formed_stream) :-
    wam_object_encode([user:answer/1, user:sum3/3],
        [wamo_entry(answer/1)], Codes),
    string_codes(Text, Codes),
    sub_string(Text, 0, 4, _, "WAMO"),
    split_string(Text, "\n \t", "\n \t", Parts0),
    exclude(==(""), Parts0, Parts),
    % tokens: WAMO 1 <entry> <natoms> ... ; version must be "1", entry "0"
    Parts = ["WAMO", "1", "0" | _],
    !.

% call/N meta-calls, float constants and switch-on-constant tables are
% outside slice 1 -- the writer rejects them loudly.
test(float_constant_is_rejected,
        [throws(error(wamo_unsupported(float_constant(_)), _))]) :-
    wam_object_encode([user:uses_float/1], [wamo_entry(uses_float/1)], _).

% A requested entry predicate with no label is a hard error.
test(missing_entry_is_rejected,
        [throws(error(wamo_entry_not_found(_), _))]) :-
    wam_object_encode([user:answer/1, user:sum3/3],
        [wamo_entry(nonesuch/2)], _).

% Full round trip: write a grammar, build a host binary that carries the
% loader, load the object at runtime and run it. The host never saw the
% grammar at compile time.
test(host_loads_and_runs_object_at_runtime,
        [condition(clang_available)]) :-
    obj_dir(Dir),
    directory_file_path(Dir, 'grammar.wamo', Wamo),
    write_wam_object([user:answer/1, user:sum3/3],
        [wamo_entry(answer/1)], Wamo),
    build_host(Dir, Host),
    run_host(Host, Wamo, Out, 0),
    assertion(Out == "119\n"),
    !.

% Same host binary, a different grammar object -> different answer, with no
% host rebuild. This is the point of runtime-loadable objects.
test(same_host_runs_a_swapped_grammar,
        [condition(clang_available)]) :-
    obj_dir(Dir),
    directory_file_path(Dir, 'host_bin', Host),
    ( exists_file(Host) -> true ; build_host(Dir, Host) ),
    directory_file_path(Dir, 'grammar2.wamo', Wamo2),
    write_wam_object([user:answer_swapped/1, user:sum3/3],
        [wamo_entry(answer_swapped/1)], Wamo2),
    run_host(Host, Wamo2, Out, 0),
    assertion(Out == "1020\n"),
    !.

:- end_tests(wam_object).

% --- helpers ---------------------------------------------------------------

obj_dir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_wam_object', Dir),
    ( exists_directory(Dir) -> true ; make_directory_path(Dir) ).

% Build a host binary: a trivial module carrying the .wamo loader, plus a
% main() that loads argv[1], runs the entry, and prints the integer.
build_host(Dir, Host) :-
    directory_file_path(Dir, 'host.ll', LL),
    with_output_to(string(_),
        write_wam_llvm_project([user:answer/1],
            [module_name(wam_object_host), emit_wamo_loader(true)], LL)),
    host_main_ir(MainIR),
    setup_call_cleanup(
        open(LL, append, S, [encoding(utf8)]),
        write(S, MainIR),
        close(S)),
    directory_file_path(Dir, 'host_bin', Host),
    format(atom(Cmd), 'clang -w -O2 ~w -o ~w -lm 2>&1', [LL, Host]),
    process_create(path(sh), ['-c', Cmd],
        [stdout(pipe(CS)), stderr(std), process(Pid)]),
    read_string(CS, _, ClangOut),
    close(CS),
    process_wait(Pid, Status),
    ( Status == exit(0) -> true
    ; throw(error(clang_failed(ClangOut), _)) ).

host_main_ir(
'\n@.wam_object_fmt = private constant [6 x i8] c"%lld\\0A\\00"\n\n\c
define i32 @main(i32 %argc, i8** %argv) {\n\c
entry:\n\c
  %p1 = getelementptr i8*, i8** %argv, i64 1\n\c
  %path = load i8*, i8** %p1\n\c
  %obj = call { %WamState*, i32 } @wam_object_load(i8* %path)\n\c
  %vm = extractvalue { %WamState*, i32 } %obj, 0\n\c
  %pc = extractvalue { %WamState*, i32 } %obj, 1\n\c
  %vm_null = icmp eq %WamState* %vm, null\n\c
  br i1 %vm_null, label %load_fail, label %run\n\c
run:\n\c
  %r = call { i64, i1 } @wam_object_call_i64(%WamState* %vm, i32 %pc, i32 0, %Value* null, i32 0)\n\c
  %val = extractvalue { i64, i1 } %r, 0\n\c
  %ok = extractvalue { i64, i1 } %r, 1\n\c
  br i1 %ok, label %print, label %run_fail\n\c
print:\n\c
  %fmt = getelementptr [6 x i8], [6 x i8]* @.wam_object_fmt, i32 0, i32 0\n\c
  %pr = call i32 (i8*, ...) @printf(i8* %fmt, i64 %val)\n\c
  ret i32 0\n\c
load_fail:\n\c
  ret i32 20\n\c
run_fail:\n\c
  ret i32 21\n\c
}\n').

run_host(Host, Wamo, Out, ExpectedStatus) :-
    process_create(Host, [Wamo],
        [stdout(pipe(S)), stderr(std), process(Pid)]),
    read_string(S, _, Out),
    close(S),
    process_wait(Pid, exit(Status)),
    assertion(Status == ExpectedStatus).
