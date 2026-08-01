:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% plawk: a BARE `print` -- no argument list.
%
% In awk `print` means `print $0`, and it is one of the most common things anyone
% writes (`/pattern/ { print }`). plawk made it a PARSE ERROR everywhere:
%
%   { print }                 PARSE ERROR
%   /b/ { print }             PARSE ERROR
%   $1 == "b" { print }       PARSE ERROR
%   { if (NR == 2) print }    PARSE ERROR
%
% while `print $0` worked. It is a desugaring, not a feature: the bare form parses
% to the SAME `print([field(0)])` term the explicit `$0` form produces, so no
% codegen path learns anything new and the emitted IR is identical. A test asserts
% the two parse trees are equal.
%
% The whole risk is in the keyword boundary -- `print` must not swallow the prefix
% of `printf`, nor of an identifier like `printer`. identifier_boundary//0 handles
% both, and both are pinned below.
%
% BEGIN and END decline, exactly as the explicit `print $0` does there: neither has
% a current record. gawk differs (it keeps `$0` in END and treats it as empty in
% BEGIN), so those are recorded as the pre-existing no-record-outside-the-loop
% boundary rather than as this change's doing.
%
% gawk 5.2 is the oracle for every expectation here.

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(filesex), [make_directory_path/1]).
:- use_module('../examples/plawk/parser/plawk_parser').
:- use_module('../examples/plawk/codegen/plawk_native_codegen').

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

% Three records: "a 1" / "b 2" / "c 3".
input("a 1\nb 2\nc 3\n").

:- begin_tests(plawk_bare_print).

% --- the bare form in every rule context ----------------------------------

test(bare_print_every_record, [condition(clang_available)]) :-
    run("{ print }\n", "a 1\nb 2\nc 3\n"),
    !.

test(bare_print_under_regex_pattern, [condition(clang_available)]) :-
    run("/b/ { print }\n", "b 2\n"),
    !.

test(bare_print_under_string_pattern, [condition(clang_available)]) :-
    run("$1 == \"b\" { print }\n", "b 2\n"),
    !.

test(bare_print_under_nr_pattern, [condition(clang_available)]) :-
    run("NR == 2 { print }\n", "b 2\n"),
    !.

test(bare_print_under_numeric_pattern, [condition(clang_available)]) :-
    run("$2 > 2 { print }\n", "c 3\n"),
    !.

% Inside an `if`, braceless -- the shape `if (c) print` is as common as the rule
% form and exercises the same production from a different caller.
test(bare_print_in_braceless_if, [condition(clang_available)]) :-
    run("{ if (NR == 2) print }\n", "b 2\n"),
    !.

test(bare_print_in_if_else, [condition(clang_available)]) :-
    run("{ if (NR == 2) print; else print \"no\" }\n", "no\nb 2\nno\n"),
    !.

test(bare_print_in_braced_if, [condition(clang_available)]) :-
    run("{ if (NR == 2) { print } }\n", "b 2\n"),
    !.

% Beside another statement, and twice.
test(bare_print_beside_another_statement, [condition(clang_available)]) :-
    run("{ n++; print }\n", "a 1\nb 2\nc 3\n"),
    !.

test(two_bare_prints, [condition(clang_available)]) :-
    run("/b/ { print; print }\n", "b 2\nb 2\n"),
    !.

% ORS: the bare form must behave EXACTLY as `print $0` does. It does -- but note
% that both emit a newline rather than the ORS, which diverges from gawk (`b 2|`).
% That is a PRE-EXISTING bug in the whole-record fast path (it formats with a
% hardcoded "%s\n"), present at the merge base and unrelated to this desugaring;
% `print $1` honours ORS correctly. Asserted as an EQUIVALENCE rather than against
% gawk, so this test pins what this change is responsible for without blessing the
% wrong output -- when the fast path is fixed, both sides move together and this
% still holds.
test(bare_print_matches_explicit_under_ors, [condition(clang_available)]) :-
    run_out("BEGIN { ORS = \"|\" } /b/ { print }\n", BareOut),
    run_out("BEGIN { ORS = \"|\" } /b/ { print $0 }\n", ExplicitOut),
    assertion(BareOut == ExplicitOut),
    !.

% --- the keyword boundary -------------------------------------------------

% `printf` must still be printf, not a bare `print` followed by a stray `f`.
test(printf_still_parses_as_printf, [condition(clang_available)]) :-
    run("{ printf \"%s\\n\", $1 }\n", "a\nb\nc\n"),
    !.

test(printf_with_no_args_still_printf, [condition(clang_available)]) :-
    run("/b/ { printf \"hi\\n\" }\n", "hi\n"),
    !.

% An identifier that merely starts with `print` is still an identifier.
test(identifier_starting_with_print, [condition(clang_available)]) :-
    run("{ printer = NR; print printer }\n", "1\n2\n3\n"),
    !.

% --- regressions: the argument forms --------------------------------------

test(explicit_whole_record_unchanged, [condition(clang_available)]) :-
    run("{ print $0 }\n", "a 1\nb 2\nc 3\n"),
    !.

test(single_field_unchanged, [condition(clang_available)]) :-
    run("{ print $1 }\n", "a\nb\nc\n"),
    !.

test(field_list_unchanged, [condition(clang_available)]) :-
    run("{ print $1, $2 }\n", "a 1\nb 2\nc 3\n"),
    !.

test(string_literal_unchanged, [condition(clang_available)]) :-
    run("{ print \"x\" }\n", "x\nx\nx\n"),
    !.

% --- clean declines: no record outside the rule loop ---------------------

% BEGIN and END have no current record. gawk keeps `$0` in END (printing the last
% record) and treats it as empty in BEGIN; plawk declines in both -- and declines
% the EXPLICIT `print $0` there identically, so the bare form inherits that
% pre-existing boundary rather than introducing one. Pinned in pairs so a future
% fix to either context updates both.
test(bare_print_in_end_declines) :-
    build_status("{ n++ } END { print }\n", 3),
    !.

test(explicit_whole_record_in_end_declines) :-
    build_status("{ n++ } END { print $0 }\n", 3),
    !.

test(bare_print_in_begin_declines) :-
    build_status("BEGIN { print }\n{ print $1 }\n", 3),
    !.

test(explicit_whole_record_in_begin_declines) :-
    build_status("BEGIN { print $0 }\n{ print $1 }\n", 3),
    !.

% --- structure: a desugaring, not a feature ------------------------------

% The bare form parses to the very term the explicit `$0` form does, so nothing
% downstream can distinguish them -- which is why no codegen path changed.
test(bare_print_parses_as_explicit_whole_record) :-
    plawk_parse_string("/b/ { print }\n", Bare),
    plawk_parse_string("/b/ { print $0 }\n", Explicit),
    assertion(Bare == Explicit),
    assertion(Bare = program([], [rule(_, [print([field(0)])])], [])),
    !.

% Same in an `if` branch, where a different caller reaches the production.
test(bare_print_in_if_parses_as_explicit) :-
    plawk_parse_string("{ if (NR == 2) print }\n", Bare),
    plawk_parse_string("{ if (NR == 2) print $0 }\n", Explicit),
    assertion(Bare == Explicit),
    !.

% The argument form still wins when arguments are present: `print $1` must not
% parse as a bare print that leaves `$1` behind (which would then fail at `}`).
test(argument_form_takes_precedence) :-
    plawk_parse_string("{ print $1 }\n",
        program([], [rule(always, [print([field(1)])])], [])),
    !.

:- end_tests(plawk_bare_print).

% --- helpers ---------------------------------------------------------------

odir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_bare_print', Dir),
    ( exists_directory(Dir) -> true ; make_directory_path(Dir) ).

run(Src, Expected) :-
    odir(Dir),
    input(Input),
    directory_file_path(Dir, 'bp_bin', Bin),
    % A build that unexpectedly declines must not silently run a stale binary.
    ( exists_file(Bin) -> delete_file(Bin) ; true ),
    directory_file_path(Dir, 'bp', Prog0),
    atom_concat(Prog0, '.plawk', Prog),
    setup_call_cleanup(open(Prog, write, S, [encoding(utf8)]),
        write(S, Src), close(S)),
    atom_concat(Prog0, '_in.txt', In),
    setup_call_cleanup(open(In, write, SI, [encoding(utf8)]),
        write(SI, Input), close(SI)),
    cli([build, Prog, '-o', Bin], 0),
    process_create(Bin, [In], [stdout(pipe(PS)), stderr(std), process(Pid)]),
    read_string(PS, _, Out),
    close(PS),
    process_wait(Pid, exit(0)),
    ( Out == Expected
    -> true
    ;  format(user_error, "~n~w~n  got      ~q~n  expected ~q~n",
           [Src, Out, Expected]), fail
    ).

% As run/2 but returning the output instead of comparing it.
run_out(Src, Out) :-
    odir(Dir),
    input(Input),
    directory_file_path(Dir, 'bp_bin', Bin),
    ( exists_file(Bin) -> delete_file(Bin) ; true ),
    directory_file_path(Dir, 'bp', Prog0),
    atom_concat(Prog0, '.plawk', Prog),
    setup_call_cleanup(open(Prog, write, S, [encoding(utf8)]),
        write(S, Src), close(S)),
    atom_concat(Prog0, '_in.txt', In),
    setup_call_cleanup(open(In, write, SI, [encoding(utf8)]),
        write(SI, Input), close(SI)),
    cli([build, Prog, '-o', Bin], 0),
    process_create(Bin, [In], [stdout(pipe(PS)), stderr(std), process(Pid)]),
    read_string(PS, _, Out),
    close(PS),
    process_wait(Pid, exit(0)).

% Build only, asserting the CLI status (2 = parse error, 3 = parses but outside the
% compilable surface).
build_status(Src, ExpectedStatus) :-
    odir(Dir),
    directory_file_path(Dir, 'bp_reject', Prog0),
    atom_concat(Prog0, '.plawk', Prog),
    setup_call_cleanup(open(Prog, write, S, [encoding(utf8)]),
        write(S, Src), close(S)),
    atom_concat(Prog0, '_bin', Bin),
    cli([build, Prog, '-o', Bin], ExpectedStatus).

cli(Args, ExpectedStatus) :-
    process_create(path(swipl), ['examples/plawk/bin/plawk' | Args],
        [stdout(pipe(S)), stderr(std), process(Pid)]),
    read_string(S, _, _), close(S),
    process_wait(Pid, exit(Status)),
    assertion(Status == ExpectedStatus).
