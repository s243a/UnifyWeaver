:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% Paragraph mode: `BEGIN { RS = "" }`. An empty record separator tells awk to
% read records as blank-line-separated paragraphs rather than by a fixed
% byte-string. The startup helper stores `i64 0` into @wam_rs_len; both record
% readers (@wam_stream_read_line_transient_value and its persistent sibling)
% treat rs_len == 0 as paragraph mode:
%
%   * leading blank lines are skipped (a newline seen while the record is empty
%     does not start a record);
%   * a record boundary is two adjacent newlines ("\n\n"), trimmed off;
%   * runs of blank lines collapse into a single boundary (via the next
%     record's leading-blank skip);
%   * a lone trailing newline at EOF is stripped.
%
% In paragraph mode a record may span multiple lines. The default FS treats an
% embedded newline as field whitespace, so `$1 $2 ...` field-split across the
% lines of a paragraph.

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(filesex), [make_directory_path/1]).

:- begin_tests(plawk_paragraph).

% Basic paragraph records: blank line separates single-line paragraphs.
test(para_basic, [condition(clang_available)]) :-
    sdir(Dir),
    build_run(Dir, 'pb', "BEGIN { RS = \"\" } { print NR, $0 }\n",
        "a\n\nb\n\nc", Out),
    assertion(Out == "1 a\n2 b\n3 c\n"), !.

% A paragraph spanning multiple lines is one record; $0 keeps the embedded
% newline.
test(para_multiline_record, [condition(clang_available)]) :-
    sdir(Dir),
    build_run(Dir, 'pm', "BEGIN { RS = \"\" } { print NR, \":\", $0 }\n",
        "a\nb\n\nc", Out),
    assertion(Out == "1 : a\nb\n2 : c\n"), !.

% Leading blank lines are skipped: the first record is the first non-blank
% paragraph.
test(para_leading_blanks, [condition(clang_available)]) :-
    sdir(Dir),
    build_run(Dir, 'pl', "BEGIN { RS = \"\" } { print NR, $0 }\n",
        "\n\n\na\n\nb", Out),
    assertion(Out == "1 a\n2 b\n"), !.

% Runs of blank lines collapse into a single record boundary.
test(para_blank_run_collapse, [condition(clang_available)]) :-
    sdir(Dir),
    build_run(Dir, 'pc', "BEGIN { RS = \"\" } { print NR, $0 }\n",
        "a\n\n\n\nb", Out),
    assertion(Out == "1 a\n2 b\n"), !.

% A lone trailing newline at EOF is stripped from the final record.
test(para_trailing_newline, [condition(clang_available)]) :-
    sdir(Dir),
    build_run(Dir, 'pt', "BEGIN { RS = \"\" } { print NR, \":\", $0, \":e\" }\n",
        "a\n\nb\n", Out),
    assertion(Out == "1 : a :e\n2 : b :e\n"), !.

% Default FS splits fields across the embedded newlines of a paragraph: $3
% comes from the paragraph's second line.
test(para_field_split, [condition(clang_available)]) :-
    sdir(Dir),
    build_run(Dir, 'pf', "BEGIN { RS = \"\" } { print NF, $1, $3 }\n",
        "1 3\nx z\n\n2 4\nq r", Out),
    assertion(Out == "4 1 x\n4 2 q\n"), !.

% NR counts paragraphs, not lines.
test(para_nr_counts_records, [condition(clang_available)]) :-
    sdir(Dir),
    build_run(Dir, 'pn', "BEGIN { RS = \"\" } { seen++ } END { print NR }\n",
        "a\nb\nc\n\nd\ne\n\nf", Out),
    assertion(Out == "3\n"), !.

:- end_tests(plawk_paragraph).

% --- helpers ---------------------------------------------------------------

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

sdir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_paragraph', Dir),
    ( exists_directory(Dir) -> true ; make_directory_path(Dir) ).

build_run(Dir, Name, Src, Input, Out) :-
    directory_file_path(Dir, Name, Prog0),
    atom_concat(Prog0, '.plawk', Prog),
    setup_call_cleanup(open(Prog, write, S, [encoding(utf8)]),
        write(S, Src), close(S)),
    atom_concat(Prog0, '_bin', Bin),
    process_create(path(swipl), ['examples/plawk/bin/plawk', build, Prog, '-o', Bin],
        [stdout(null), stderr(null), process(BPid)]),
    process_wait(BPid, exit(0)),
    process_create(Bin, ['-'],
        [stdin(pipe(In)), stdout(pipe(RS)), stderr(std), process(RPid)]),
    format(In, "~w", [Input]),
    close(In),
    read_string(RS, _, Out),
    close(RS),
    process_wait(RPid, exit(0)).
