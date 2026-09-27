:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% Rules that mix array (assoc) actions with scalar work -- the mixed driver.
%
% ---------------------------------------------------------------------------
% PR 1 of the mixed-rules plan (design reviewed before implementation). The rule
% walker already had table rows (field-key `c[$1]++`, var-key updates, `delete`,
% `arr[k]` reads), shared by the scalar and mixed chains. What declined was
% ADMISSION:
%
%   - plawk_mixed_assoc_count_plan/3 required the END to print an array element or
%     some rule to test `in`, so a program whose tables are only WRITTEN declined:
%         { c[$1]++; n++ } END { print n }        (exit 3; gawk: the record count)
%     The mixed plan is program-wide, so splitting the work across two rules did
%     not help either. The guard is gone; a table-establishing action and a scalar
%     slot (or a conditional) are still required, so pure-assoc and pure-scalar
%     programs keep their own drivers -- IR of every program that compiled before
%     is byte-identical.
%   - there was no mixed driver for a program with NO END; one now exists (the
%     single-print clause minus the END print: end_print frees the tables).
%
% Later PRs: field-key reads in a body print (`print n, c[$1]`), split with scalar
% work, `n = split(...)`, loops over split arrays, for-in in a mixed body.
%
% gawk is the oracle (verified against gawk 5.1.0, LC_ALL=C).

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(lists)).
:- use_module(library(filesex), [make_directory_path/1]).

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

input("a 1\nb 2\na 3\n").

:- begin_tests(plawk_mixed_rules).

test(written_tables_with_scalar_end, [condition(clang_available)]) :-
    run("{ c[$1]++; n++ } END { print n }\n", "3\n"),
    !,
    run("{ c[$1]++; s += $2 } END { print s }\n", "6\n"),
    !.

% The mixed plan is program-wide: the same work split across two rules.
test(across_two_rules, [condition(clang_available)]) :-
    run("{ c[$1]++ } { n++ } END { print n }\n", "3\n"),
    !.

% No END at all.
test(no_end, [condition(clang_available)]) :-
    run("{ c[$1]++; n++ }\n", ""),
    !.

% The shapes that compiled before still do (and their IR is unchanged).
test(end_reading_the_table_still_works, [condition(clang_available)]) :-
    run("{ c[$1]++; n++ } END { print n, c[\"a\"] }\n", "3 2\n"),
    !,
    run("{ c[$1]++; n++ } END { print n; print c[\"a\"] }\n", "3\n2\n"),
    !.

% Forms the later PRs teach decline or fail to parse -- never exit 4.
test(later_forms_do_not_miscompile) :-
    forall(member(Src,
            [ "{ c[$1]++; n++; print n, c[$1] }\n",
              "{ c[$1]++; if (c[$1] > 1) print \"dup\", $1 }\n",
              "{ split($1, a, \",\"); n++; print a[2], n }\n",
              "{ n = split($1, a, \",\"); print n }\n"
            ]),
        build_status_not_4(Src)),
    !.

:- end_tests(plawk_mixed_rules).

% --- helpers ---------------------------------------------------------------

odir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_mixed_rules', Dir),
    ( exists_directory(Dir) -> true ; make_directory_path(Dir) ).

run(Src, Expected) :-
    input(Input),
    odir(Dir),
    directory_file_path(Dir, 'mr_bin', Bin),
    % A build that unexpectedly declines must not silently run a stale binary.
    ( exists_file(Bin) -> delete_file(Bin) ; true ),
    directory_file_path(Dir, 'mr', Prog0),
    atom_concat(Prog0, '.plawk', Prog),
    setup_call_cleanup(open(Prog, write, S, [encoding(utf8)]),
        write(S, Src), close(S)),
    atom_concat(Prog0, '_in.txt', In),
    setup_call_cleanup(open(In, write, SI, [encoding(utf8)]),
        write(SI, Input), close(SI)),
    build_status_of(Prog, Bin, BuildStatus),
    assertion(BuildStatus == 0),
    process_create(Bin, [In], [stdout(pipe(PS)), stderr(std), process(Pid)]),
    read_string(PS, _, Out),
    close(PS),
    process_wait(Pid, exit(0)),
    ( Out == Expected
    -> true
    ;  format(user_error, "~n~w~n  got      ~q~n  expected ~q~n",
           [Src, Out, Expected]), fail
    ).

build_status_not_4(Src) :-
    odir(Dir),
    directory_file_path(Dir, 'mr_status', Prog0),
    atom_concat(Prog0, '.plawk', Prog),
    atom_concat(Prog0, '_bin', Bin),
    setup_call_cleanup(open(Prog, write, S, [encoding(utf8)]),
        write(S, Src), close(S)),
    build_status_of(Prog, Bin, Status),
    ( memberchk(Status, [0, 2, 3])
    -> true
    ;  format(user_error, "~n~w~n  build exit ~w (4 = clang miscompile)~n",
           [Src, Status]), fail
    ).

build_status_of(Prog, Bin, Status) :-
    process_create(path(swipl), ['examples/plawk/bin/plawk', build, Prog, '-o', Bin],
        [stdout(pipe(O)), stderr(null), process(Pid)]),
    read_string(O, _, _), close(O),
    process_wait(Pid, exit(Status)).
