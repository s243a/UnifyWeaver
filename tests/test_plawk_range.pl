:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% plawk range patterns `/start/,/end/ { ... }`: the rule fires for every record
% from a /start/ match through a /end/ match (inclusive), tracked by a per-rule
% i1 latch global. The end is tested only once the range is active, so a line
% matching both start and end starts a range that continues (the common gawk
% semantics). v1 endpoints are regexes; general-pattern endpoints (NR==1,NR==3)
% are a follow-on.

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(filesex), [make_directory_path/1]).
:- use_module('../examples/plawk/parser/plawk_parser').

:- begin_tests(plawk_range).

% --- parsing ----------------------------------------------------------------

test(range_parses) :-
    plawk_parse_string("/start/,/end/ { print $0 }\n",
        program([], [rule(range(contains("start"), contains("end")),
            [print([field(0)])])], [])),
    !.

test(range_prefix_endpoints_parse) :-
    plawk_parse_string("/^BEGIN/,/^END/ { print $1 }\n",
        program([], [rule(range(prefix("BEGIN"), prefix("END")),
            [print([field(1)])])], [])),
    !.

% a plain single regex rule is unaffected (no comma -> not a range).
test(plain_regex_unaffected) :-
    plawk_parse_string("/a/ { print $1 }\n",
        program([], [rule(contains("a"), [print([field(1)])])], [])),
    !.

% --- runtime ----------------------------------------------------------------

% the section between markers is printed inclusive; outside is skipped.
test(range_inclusive, [condition(clang_available)]) :-
    ldir(Dir),
    build_run(Dir, 'inc', "/start/,/end/ { print $0 }\n",
        "a\nstart\nb\nc\nend\nd\n", Out, St),
    assertion(St == 0), assertion(Out == "start\nb\nc\nend\n"), !.

% two separate ranges in one stream both fire (the latch re-arms).
test(range_reentry, [condition(clang_available)]) :-
    ldir(Dir),
    build_run(Dir, 're', "/S/,/E/ { print $0 }\n",
        "x\nS\n1\nE\ny\nS\n2\nE\nz\n", Out, St),
    assertion(St == 0), assertion(Out == "S\n1\nE\nS\n2\nE\n"), !.

% a start with no matching end prints to EOF.
test(range_unterminated, [condition(clang_available)]) :-
    ldir(Dir),
    build_run(Dir, 'un', "/S/,/E/ { print $0 }\n",
        "a\nS\nb\nc\n", Out, St),
    assertion(St == 0), assertion(Out == "S\nb\nc\n"), !.

% fields are accessible inside a range body.
test(range_field_access, [condition(clang_available)]) :-
    ldir(Dir),
    build_run(Dir, 'fl', "/S/,/E/ { print $2 }\n",
        "S k1\nrow v1\nrow v2\nE k2\n", Out, St),
    assertion(St == 0), assertion(Out == "k1\nv1\nv2\nk2\n"), !.

% a range rule coexists with an ordinary rule + END.
test(range_coexists, [condition(clang_available)]) :-
    ldir(Dir),
    build_run(Dir, 'co',
        "/S/,/E/ { print \"in:\", $0 }\n{ n++ }\nEND { print \"total\", n }\n",
        "a\nS\nb\nE\nc\n", Out, St),
    assertion(St == 0), assertion(Out == "in: S\nin: b\nin: E\ntotal 5\n"), !.

% a record matching neither marker never fires the range.
test(range_no_match, [condition(clang_available)]) :-
    ldir(Dir),
    build_run(Dir, 'nm', "/S/,/E/ { print $0 }\n",
        "a\nb\nc\n", Out, St),
    assertion(St == 0), assertion(Out == ""), !.

:- end_tests(plawk_range).

% --- helpers ---------------------------------------------------------------

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

ldir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_range', Dir),
    ( exists_directory(Dir) -> true ; make_directory_path(Dir) ).

build_run(Dir, Name, Src, Input, Out, RunStatus) :-
    directory_file_path(Dir, Name, Prog0),
    atom_concat(Prog0, '.plawk', Prog),
    setup_call_cleanup(open(Prog, write, S, [encoding(utf8)]),
        write(S, Src), close(S)),
    atom_concat(Prog0, '_bin', Bin),
    process_create(path(swipl), ['examples/plawk/bin/plawk', build, Prog, '-o', Bin],
        [stdout(null), stderr(null), process(BPid)]),
    process_wait(BPid, exit(0)),
    process_create(Bin, [],
        [stdin(pipe(In)), stdout(pipe(RS)), stderr(std), process(RPid)]),
    format(In, "~w", [Input]),
    close(In),
    read_string(RS, _, Out),
    close(RS),
    process_wait(RPid, exit(RunStatus)).
