:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% Cross-pass scalars: a scalar accumulator (`acc += 1` count, `acc += $N`
% sum) folded in one pass and read in a later pass. Because passes are
% separate functions, the accumulator is a module global (@plawk_scalar_<acc>,
% zero-initialised), so its value persists across passes -- pass 1 folds the
% grand total, pass 2 annotates each record with it. See
% PLAWK_MULTIPASS_CACHE.md. v1: the program also carries a shared table (the
% multi-pass driver is table-based); a pure-scalar (no-table) program and
% arithmetic in prints (`$2 / total`) are follow-ons.

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(filesex), [make_directory_path/1]).

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

:- begin_tests(plawk_crosspass_scalar).

% Count records in pass 1 (total += 1), annotate each record in pass 2 with
% its per-key count and the grand total. Over a a b: total = 3.
test(crosspass_count, [condition(clang_available)]) :-
    cd(Dir),
    Src = "pass { c[$1]++ ; total += 1 }\npass { print $1, c[$1], total }\n",
    run_sorted(Dir, 'cnt', Src, "a\na\nb\n", S),
    assertion(S == ["a 2 3", "a 2 3", "b 1 3"]),
    !.

% Sum field 2 in pass 1 (total += $2), print each record with the grand
% total in pass 2. Over a=10, a=20, b=5: total = 35.
test(crosspass_field_sum, [condition(clang_available)]) :-
    cd(Dir),
    Src = "pass { c[$1]++ ; total += $2 }\npass { print $1, total }\n",
    run_sorted(Dir, 'sum', Src, "a 10\na 20\nb 5\n", S),
    assertion(S == ["a 35", "a 35", "b 35"]),
    !.

% A scalar accumulator in a single-pass assoc program (END for-in) compiles
% and runs -- the accumulator is dead here (END dumps the table) but the
% scalar action must not break the pure-assoc chain.
test(single_pass_scalar_ok, [condition(clang_available)]) :-
    cd(Dir),
    Src = "{ c[$1]++ ; total += 1 }\nEND { for (k in c) print k, c[k] }\n",
    run_sorted(Dir, 'sp', Src, "a\na\nb\n", S),
    assertion(S == ["a 2", "b 1"]),
    !.

:- end_tests(plawk_crosspass_scalar).

% --- helpers ---------------------------------------------------------------

cd(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_crosspass_scalar', Dir),
    ( exists_directory(Dir) -> true ; make_directory_path(Dir) ).

run_sorted(Dir, Name, Src, Input, Sorted) :-
    directory_file_path(Dir, Name, Prog0),
    atom_concat(Prog0, '.plawk', Prog),
    setup_call_cleanup(open(Prog, write, S, [encoding(utf8)]),
        write(S, Src), close(S)),
    atom_concat(Prog0, '_bin', Bin),
    cli([build, Prog, '-o', Bin], 0),
    atom_concat(Prog0, '_in.txt', In),
    setup_call_cleanup(open(In, write, SI, [encoding(utf8)]),
        write(SI, Input), close(SI)),
    process_create(Bin, [In], [stdout(pipe(PS)), stderr(std), process(Pid)]),
    read_string(PS, _, Out), close(PS), process_wait(Pid, exit(0)),
    split_string(Out, "\n", "", L0), exclude(==(""), L0, L), msort(L, Sorted).

cli(Args, ExpectedStatus) :-
    process_create(path(swipl), ['examples/plawk/bin/plawk' | Args],
        [stdout(pipe(S)), stderr(std), process(Pid)]),
    read_string(S, _, _), close(S),
    process_wait(Pid, exit(Status)),
    assertion(Status == ExpectedStatus).
