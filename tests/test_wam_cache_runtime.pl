:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% Runtime primitive for the plawk multi-pass persistent cache
% (PLAWK_MULTIPASS_CACHE.md, phase 1): @wam_cache_open / _commit / _close in
% the LLVM/WAM target. The cache handle IS a %WamAssocI64Table*; open builds
% a table and loads (key,value) i64 pairs from a flat little-endian file,
% commit writes the table back, close frees it. The i64 fast path reuses the
% assoc helpers on the handle.
%
% There is no plawk surface for the cache yet (that is phase 1b), so this
% test exercises the primitive directly: it builds a real plawk binary with
% `--keep-ll` to obtain a module carrying the full runtime (types, libc
% declares, assoc helpers, and the new cache functions), then swaps the
% generated `main` for a synthesized one that drives the cache and prints
% what it reads back. Running the binary twice against the same file proves
% the file round-trip and cross-run persistence. This is a file-backed
% backend; the LMDB backend (larger-than-RAM) rides the same ABI later.

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(filesex), [make_directory_path/1]).

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

% The synthesized entry point: open a store at argv[1], accumulate key 1 by
% 5 and key 2 by 3, commit, close; reopen and read the committed values back,
% printing "v1 v2". Run 1 over an empty file -> "5 3"; run 2 loads that and
% accumulates again -> "10 6".
cache_test_main("
@.cachetest_fmt = private constant [9 x i8] c\"%ld %ld\\0A\\00\"

define i32 @main(i32 %argc, i8** %argv) {
entry:
  %argv1_ptr = getelementptr i8*, i8** %argv, i64 1
  %path = load i8*, i8** %argv1_ptr
  %t = call %WamAssocI64Table* @wam_cache_open(i8* %path)
  %a = call i64 @wam_assoc_i64_inc(%WamAssocI64Table* %t, i64 1, i64 5)
  %b = call i64 @wam_assoc_i64_inc(%WamAssocI64Table* %t, i64 2, i64 3)
  call void @wam_cache_commit(%WamAssocI64Table* %t, i8* %path)
  call void @wam_cache_close(%WamAssocI64Table* %t)
  %t2 = call %WamAssocI64Table* @wam_cache_open(i8* %path)
  %v1 = call i64 @wam_assoc_i64_get(%WamAssocI64Table* %t2, i64 1)
  %v2 = call i64 @wam_assoc_i64_get(%WamAssocI64Table* %t2, i64 2)
  %fmt = getelementptr [9 x i8], [9 x i8]* @.cachetest_fmt, i32 0, i32 0
  %r = call i32 (i8*, ...) @printf(i8* %fmt, i64 %v1, i64 %v2)
  call void @wam_cache_close(%WamAssocI64Table* %t2)
  ret i32 0
}
").

:- begin_tests(wam_cache_runtime).

% Round-trip within one run, then cross-run persistence across two runs.
test(cache_roundtrip_and_persistence, [condition(clang_available)]) :-
    cr_dir(Dir),
    % A trivial plawk program just to obtain a module with the full runtime.
    directory_file_path(Dir, 'seed.plawk', Seed),
    setup_call_cleanup(open(Seed, write, S, [encoding(utf8)]),
        write(S, "{ c[$1]++ }\nEND { for (k in c) print k, c[k] }\n"), close(S)),
    directory_file_path(Dir, 'seed_bin', SeedBin),
    cli([build, Seed, '-o', SeedBin, '--keep-ll'], 0),
    atom_concat(SeedBin, '.ll', SeedLL),
    read_file_to_string(SeedLL, LL0, []),
    % Swap the generated entry point for our cache driver.
    ( sub_string(LL0, _, _, _, "define i32 @main(")
    -> true ; throw(no_generated_main) ),
    string_replace(LL0, "define i32 @main(",
        "define i32 @plawk_unused_main(", LL1),
    cache_test_main(TestMain),
    string_concat(LL1, TestMain, LLFinal),
    directory_file_path(Dir, 'cachetest.ll', TestLL),
    setup_call_cleanup(open(TestLL, write, S2, [encoding(utf8)]),
        write(S2, LLFinal), close(S2)),
    directory_file_path(Dir, 'cachetest_bin', TestBin),
    format(atom(ClangCmd), 'clang -w -O2 ~w -o ~w -lm 2>&1', [TestLL, TestBin]),
    process_create(path(sh), ['-c', ClangCmd],
        [stdout(pipe(CS)), process(CPid)]),
    read_string(CS, _, ClangOut), close(CS),
    process_wait(CPid, CStatus),
    ( CStatus == exit(0) -> true
    ; throw(clang_failed(CStatus, ClangOut)) ),
    directory_file_path(Dir, 'store.cache', Store),
    ( exists_file(Store) -> delete_file(Store) ; true ),
    % Run 1 over an empty store: 5 3.
    run_bin(TestBin, [Store], Out1),
    assertion(Out1 == "5 3\n"),
    % Run 2 loads the committed store and accumulates again: 10 6.
    run_bin(TestBin, [Store], Out2),
    assertion(Out2 == "10 6\n"),
    !.

:- end_tests(wam_cache_runtime).

% --- helpers ---------------------------------------------------------------

cr_dir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_wam_cache_runtime', Dir),
    ( exists_directory(Dir) -> true ; make_directory_path(Dir) ).

string_replace(In, From, To, Out) :-
    sub_string(In, Before, _, After, From),
    sub_string(In, 0, Before, _, Pre),
    string_length(In, Len), string_length(From, FLen),
    Start is Before + FLen, Rest is Len - Start,
    sub_string(In, Start, Rest, 0, Post),
    ( After >= 0 -> true ; true ),
    string_concat(Pre, To, Tmp0),
    string_concat(Tmp0, Post, Out).

cli(Args, ExpectedStatus) :-
    process_create(path(swipl), ['examples/plawk/bin/plawk' | Args],
        [stdout(pipe(S)), stderr(std), process(Pid)]),
    read_string(S, _, _), close(S),
    process_wait(Pid, exit(Status)),
    assertion(Status == ExpectedStatus).

run_bin(Bin, Args, Out) :-
    process_create(Bin, Args,
        [stdout(pipe(S)), stderr(std), process(Pid)]),
    read_string(S, _, Out), close(S),
    process_wait(Pid, exit(Status)),
    assertion(Status == 0).
