:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% Array key and value kinds -- docs/design/PLAWK_ARRAY_VALUE_MODEL.md.
%
% PR 1 (key hygiene): a special variable as a key (`a[NR]`, `a[FILENAME]`,
% `a[FS]`, ...) parsed as a user scalar, resolved to a phantom slot, and died in
% clang on an undefined @plawk_scalar_<Name> -- exit 4, the worst outcome. It now
% parses as a special(_) key (the parser's one special-variable list), which has no
% lowering yet, so the program declines (exit 3). A never-assigned user variable key
% (and a gawk-only name such as ARGIND) is caught on the generated IR: every driver
% scalar global it references must be defined, else the program declines.
% Integer-literal keys on a counted (interned-text) table mean their decimal text,
% so c[5] and c["5"] are one element, as in awk.
%
% gawk 5.1.0 is the oracle (LC_ALL=C).

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(filesex), [make_directory_path/1]).
:- use_module('../examples/plawk/parser/plawk_parser').

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

input("a 5\nb 7\na 3\n").

:- begin_tests(plawk_array_kinds).

test(special_variable_keys_parse_as_specials) :-
    plawk_parse_string("{ a[NR]++ }\n",
        program([], [rule(always, [inc_assoc(var(a), special('NR'))])], [])),
    plawk_parse_string("{ a[NF]++ }\n",
        program([], [rule(always, [inc_assoc(var(a), special('NF'))])], [])),
    !.

test(other_special_variable_keys_parse_as_specials) :-
    forall(member(Name-Src,
            [ 'FILENAME'-"{ a[FILENAME]++ }\n", 'FS'-"{ a[FS]++ }\n",
              'SUBSEP'-"{ a[SUBSEP]++ }\n", 'FNR'-"{ a[FNR]++ }\n" ]),
        plawk_parse_string(Src,
            program([], [rule(always, [inc_assoc(var(a), special(Name))])], []))),
    !.

% no longer exit 4 (clang on an undefined @plawk_scalar_<Name>): a clean decline.
% The first review of this PR found FILENAME, FS, ... and a never-assigned variable
% still reaching clang; every one is pinned here.
test(special_variable_keys_decline_cleanly) :-
    forall(member(Key, ['NR', 'NF', 'FNR', 'FILENAME', 'FS', 'OFS', 'ORS', 'RS',
                        'SUBSEP', 'RT', 'CONVFMT', 'OFMT', 'IGNORECASE',
                        % gawk-only names, parsed as user variables: caught on the IR
                        'ARGIND', 'ERRNO',
                        % a user variable that nothing assigns
                        unassigned]),
        ( format(string(Src), "{ a[~w]++ } END { for (k in a) print k, a[k] }~n", [Key]),
          build_status_is(Src, 3) )),
    !.

test(special_variable_keys_decline_in_other_positions) :-
    forall(member(Src,
            [
              "{ a[NR] = $0 } END { print a[1] }\n",
              "{ delete a[NR] }\n",
              "{ if (NR in a) print \"y\" }\n"
            ]),
        build_status_is(Src, 3)),
    !.

% a user variable key is unchanged
test(user_variable_key_still_parses) :-
    plawk_parse_string("{ k = $1; a[k]++ }\n",
        program([], [rule(always, [set(var(k), field(1)), inc_assoc(var(a), var(k))])], [])),
    !.

% on a counted table an integer key is its decimal text: c[5] == c["5"]
test(integer_literal_key_is_its_decimal_text, [condition(clang_available)]) :-
    run("{ c[5]++ } END { print c[5], c[\"5\"] }\n", "3 3\n"),
    !,
    run("{ c[\"5\"]++ } END { print c[5], c[\"5\"] }\n", "3 3\n"),
    !,
    run("{ c[5]++; c[$1]++ } END { print c[5], c[\"a\"] }\n", "3 2\n"),
    !.

:- end_tests(plawk_array_kinds).

% --- helpers ---------------------------------------------------------------

odir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_array_kinds', Dir),
    ( exists_directory(Dir) -> true ; make_directory_path(Dir) ).

run(Src, Expected) :-
    input(Input),
    odir(Dir),
    directory_file_path(Dir, 'ak_bin', Bin),
    ( exists_file(Bin) -> delete_file(Bin) ; true ),
    directory_file_path(Dir, 'ak', Prog0),
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

build_status_is(Src, Expected) :-
    odir(Dir),
    directory_file_path(Dir, 'ak_status', Prog0),
    atom_concat(Prog0, '.plawk', Prog),
    atom_concat(Prog0, '_bin', Bin),
    setup_call_cleanup(open(Prog, write, S, [encoding(utf8)]),
        write(S, Src), close(S)),
    build_status_of(Prog, Bin, Status),
    ( Status == Expected
    -> true
    ;  format(user_error, "~n~w~n  build exit ~w, expected ~w~n",
           [Src, Status, Expected]), fail
    ).

build_status_of(Prog, Bin, Status) :-
    process_create(path(swipl), ['examples/plawk/bin/plawk', build, Prog, '-o', Bin],
        [stdout(pipe(O)), stderr(null), process(Pid)]),
    read_string(O, _, _), close(O),
    process_wait(Pid, exit(Status)).
