:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% Nested string builtins and computed substr bounds.
%
% ---------------------------------------------------------------------------
% Two of the commonest awk one-liners failed to parse (exit 2):
%
%     { print toupper(substr($1,1,1)) substr($1,2) }     capitalise a word
%     { print substr($0, index($0, " ") + 1) }           everything after a space
%
% - substr bounds may be COMPUTED: the parser takes i64 arithmetic after the
%   literal-bound productions (so a literal call keeps its AST); codegen admits
%   only always-integral leaves -- length, index, NR, NF, literals -- and declines
%   the rest (a field or a variable as a bound).
% - A computed start below 1 is clamped to 1 with the length KEPT, which is what
%   gawk 5.1 does (substr("hello", 0, 2) is "he", substr("hello", -3, 2) is "he");
%   the runtime subslicer alone returns empty there, and index() returns 0 when the
%   needle is absent, so `substr($0, index($0, "z"))` depends on it.
% - toupper/tolower accept a substr (the case printer maps the substr slice) and
%   length accepts a case builtin or a substr -- a narrow guard, not a widening of
%   the shared builtin-argument vocabulary.
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

input("alice smith 30\nbob jones 7\nCarol Lee x\nsolo\n").

:- begin_tests(plawk_nested_builtins).

test(capitalise, [condition(clang_available)]) :-
    run("{ print toupper(substr($1,1,1)) substr($1,2) }\n", "Alice\nBob\nCarol\nSolo\n"),
    !,
    run("{ print toupper(substr($1, 1, 3)) }\n", "ALI\nBOB\nCAR\nSOL\n"),
    !.

test(after_and_before_the_first_space, [condition(clang_available)]) :-
    run("{ print substr($0, index($0, \" \") + 1) }\n",
        "smith 30\njones 7\nLee x\nsolo\n"),
    !,
    run("{ print substr($0, 1, index($0, \" \") - 1) }\n", "alice\nbob\nCarol\n\n"),
    !,
    run("{ print tolower(substr($0, index($0, \" \") + 1)) }\n",
        "smith 30\njones 7\nlee x\nsolo\n"),
    !.

test(length_based_bounds, [condition(clang_available)]) :-
    run("{ print substr($1, 2, length($1) - 2) }\n", "lic\no\naro\nol\n"),
    !,
    run("{ print substr($2, length($2)) }\n", "h\ns\ne\n\n"),
    !,
    run("{ print substr($1, NF) }\n", "ice\nb\nrol\nsolo\n"),
    !.

% A start below 1 (index() is 0 when the needle is absent) clamps to 1, as gawk.
test(start_below_one_clamps, [condition(clang_available)]) :-
    run("{ print substr($0, index($0, \"z\")) }\n",
        "alice smith 30\nbob jones 7\nCarol Lee x\nsolo\n"),
    !,
    % past the end: empty
    run("{ print substr($1, length($1) + 5) \"|\" }\n", "|\n|\n|\n|\n"),
    !.

test(length_of_nested, [condition(clang_available)]) :-
    run("{ print length(toupper($1)) }\n", "5\n3\n5\n4\n"),
    !,
    run("{ print length(substr($0, 3)) }\n", "12\n9\n9\n2\n"),
    !.

test(printf_of_a_computed_substr, [condition(clang_available)]) :-
    run("{ printf \"%s|\\n\", substr($0, index($0, \" \") + 1) }\n",
        "smith 30|\njones 7|\nLee x|\nsolo|\n"),
    !.

% Forms not reached here decline or fail to parse -- never exit 4.
test(unreached_forms_do_not_miscompile) :-
    forall(member(Src,
            [ "END { print substr($0, index($0, \" \") + 1) }\n",
              "{ printf \"%s\\n\", toupper(substr($1,1,1)) }\n",
              "{ print substr($0, $3) }\n",
              "{ n++; print substr($0, n) }\n",
              "END { print toupper(substr($1,1,1)) }\n",
              "{ print index(toupper($1), \"O\") }\n"
            ]),
        build_status_not_4(Src)),
    !.

:- end_tests(plawk_nested_builtins).

% --- helpers ---------------------------------------------------------------

odir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_nested_builtins', Dir),
    ( exists_directory(Dir) -> true ; make_directory_path(Dir) ).

run(Src, Expected) :-
    input(Input),
    odir(Dir),
    directory_file_path(Dir, 'nb_bin', Bin),
    % A build that unexpectedly declines must not silently run a stale binary.
    ( exists_file(Bin) -> delete_file(Bin) ; true ),
    directory_file_path(Dir, 'nb', Prog0),
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
    directory_file_path(Dir, 'nb_status', Prog0),
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
