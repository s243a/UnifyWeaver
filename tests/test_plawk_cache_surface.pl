:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% plawk multi-pass persistent cache, phase 1b: the SURFACE over the phase-1a
% runtime engine. `BEGIN cache("path") { declare NAME }` declares a table
% backed by a store file; the codegen creates the assoc table as usual, then
% loads the store into it at setup (@wam_cache_load) and commits it at END
% (@wam_cache_commit). So a histogram accumulates across separate runs of the
% same binary against the same store -- and a pre-populated store (from a
% prior run or seeded externally in the same flat format) is read in on open.
% File-backed backend; the LMDB backend rides the same surface later.

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(filesex), [make_directory_path/1]).
:- use_module('../examples/plawk/parser/plawk_parser').
:- use_module('../examples/plawk/codegen/plawk_native_codegen').

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

:- begin_tests(plawk_cache_surface).

% A BEGIN cache block declares a table as a cache_table begin action.
test(cache_block_parses) :-
    plawk_parse_string(
        "BEGIN cache(\"h.db\") { declare c }\n\c
         { c[$1]++ }\n\c
         END { for (k in c) print k, c[k] }\n",
        program([begin([cache_table(c, "h.db", file)])],
            [rule(always, [inc_assoc(var(c), field(1))])],
            [end([for_in(var(k), var(c),
                [print([var(k), assoc(var(c), var(k))])])])])),
    !.

% Two declares in one block share the store path.
test(multi_declare_parses) :-
    plawk_parse_string(
        "BEGIN cache(\"s.db\") { declare a ; declare b }\n{ a[$1]++ }\n",
        program([begin([cache_table(a, "s.db", file), cache_table(b, "s.db", file)])],
            _, _)),
    !.

% Cross-run persistence: the histogram counts accumulate across separate
% invocations of the binary against the same store file. Run 1 over an empty
% store yields a=2,b=1; run 2 loads that (pre-population from the prior run)
% and adds again -> a=4,b=2; run 3 -> a=6,b=3.
test(histogram_persists_across_runs, [condition(clang_available)]) :-
    cs_dir(Dir),
    directory_file_path(Dir, 'hist.db', Store),
    ( exists_file(Store) -> delete_file(Store) ; true ),
    format(string(Src),
        "BEGIN cache(\"~w\") { declare c }\n\c
         { c[$1]++ }\n\c
         END { for (k in c) print k, c[k] }\n", [Store]),
    build_bin(Dir, 'hist', Src, Bin),
    directory_file_path(Dir, 'in.txt', Input),
    setup_call_cleanup(open(Input, write, SI, [encoding(utf8)]),
        write(SI, "a\na\nb\n"), close(SI)),
    run_sorted(Bin, [Input], S1), assertion(S1 == ["a 2", "b 1"]),
    run_sorted(Bin, [Input], S2), assertion(S2 == ["a 4", "b 2"]),
    run_sorted(Bin, [Input], S3), assertion(S3 == ["a 6", "b 3"]),
    !.

% A fresh binary (recompiled) reading a store a DIFFERENT binary populated:
% pre-population survives across program rebuilds, not just reruns.
test(store_survives_rebuild, [condition(clang_available)]) :-
    cs_dir(Dir),
    directory_file_path(Dir, 'shared.db', Store),
    ( exists_file(Store) -> delete_file(Store) ; true ),
    format(string(Src),
        "BEGIN cache(\"~w\") { declare c }\n\c
         { c[$1]++ }\n\c
         END { for (k in c) print k, c[k] }\n", [Store]),
    directory_file_path(Dir, 'seed.txt', SeedIn),
    setup_call_cleanup(open(SeedIn, write, S0, [encoding(utf8)]),
        write(S0, "x\nx\nx\n"), close(S0)),
    build_bin(Dir, 'writer', Src, WriterBin),
    run_sorted(WriterBin, [SeedIn], _),               % store now has x=3
    build_bin(Dir, 'reader', Src, ReaderBin),          % separate compile
    directory_file_path(Dir, 'more.txt', MoreIn),
    setup_call_cleanup(open(MoreIn, write, S1, [encoding(utf8)]),
        write(S1, "x\n"), close(S1)),
    run_sorted(ReaderBin, [MoreIn], S), assertion(S == ["x 4"]),
    !.

:- end_tests(plawk_cache_surface).

% --- helpers ---------------------------------------------------------------

cs_dir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_cache_surface', Dir),
    ( exists_directory(Dir) -> true ; make_directory_path(Dir) ).

build_bin(Dir, Name, Src, Bin) :-
    directory_file_path(Dir, Name, Prog0),
    atom_concat(Prog0, '.plawk', Prog),
    setup_call_cleanup(open(Prog, write, S, [encoding(utf8)]),
        write(S, Src), close(S)),
    atom_concat(Prog0, '_bin', Bin),
    cli([build, Prog, '-o', Bin], 0).

run_sorted(Bin, Args, Sorted) :-
    process_create(Bin, Args,
        [stdout(pipe(S)), stderr(std), process(Pid)]),
    read_string(S, _, Out), close(S),
    process_wait(Pid, exit(Status)),
    assertion(Status == 0),
    split_string(Out, "\n", "", L0), exclude(==(""), L0, L), msort(L, Sorted).

cli(Args, ExpectedStatus) :-
    process_create(path(swipl), ['examples/plawk/bin/plawk' | Args],
        [stdout(pipe(S)), stderr(std), process(Pid)]),
    read_string(S, _, _), close(S),
    process_wait(Pid, exit(Status)),
    assertion(Status == ExpectedStatus).
