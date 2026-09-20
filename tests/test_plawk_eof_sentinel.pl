:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% A data line whose text is literally `end_of_file` must NOT be mistaken for
% end-of-input.
%
% ---------------------------------------------------------------------------
% EOF BY IDENTITY, NOT BY TEXT
%
% The record loop reads each line with wam_stream_read_line_transient_value, which
% returns a successful data record as the RESERVED TRANSIENT ID (2^62, its text held
% in a shared transient buffer) and real end-of-input as the interned `end_of_file`
% ATOM (a distinct, small id). The loop used to decide EOF by stringifying the id and
% strcmp-ing the text against "end_of_file" -- so a data line reading `end_of_file`
% (a transient, id 2^62, whose text is "end_of_file") matched and the loop STOPPED,
% silently truncating the input at that line. This was a wrong-output/data-loss bug,
% the worst class, surfaced by two independent code reviews of the END-only work.
%
% The fix compares the id against the transient sentinel (2^62): the only
% non-transient value the reader hands the loop is the end_of_file atom, so
% `id != 2^62` is an exact EOF test that does not look at the text at all. It lives in
% the shared driver-loop template (src/unifyweaver/targets/wam_llvm_target.pl, two
% sites) plus the plawk assoc driver's own inline loop -- every record loop.
%
% gawk is the oracle: `end_of_file` is ordinary data to it.

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(filesex), [make_directory_path/1]).
:- use_module('../examples/plawk/parser/plawk_parser').
:- use_module('../examples/plawk/codegen/llvm/plawk_native_codegen').

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

:- begin_tests(plawk_eof_sentinel).

% --- the sentinel as data, in every position, across driver types --------

% Scalar whole-record: the record after the sentinel must still be processed.
test(scalar_whole_record_not_truncated, [condition(clang_available)]) :-
    run_with("a\nend_of_file\nb\n", "{ print NR, $0 }\n",
        "1 a\n2 end_of_file\n3 b\n"),
    !.

% A bare `print` ( = print $0 ) over the sentinel line.
test(bare_print_over_sentinel, [condition(clang_available)]) :-
    run_with("a\nend_of_file\nb\n", "{ print }\n", "a\nend_of_file\nb\n"),
    !.

% The sentinel as the ONLY record.
test(sentinel_as_only_record, [condition(clang_available)]) :-
    run_with("end_of_file\n", "{ print NR, $0 }\n", "1 end_of_file\n"),
    !.

% The sentinel as the LAST record (before real EOF).
test(sentinel_as_last_record, [condition(clang_available)]) :-
    run_with("x\ny\nend_of_file\n", "{ print NR, $0 }\n",
        "1 x\n2 y\n3 end_of_file\n"),
    !.

% NR counts the sentinel line -- END sees the true record count.
test(nr_counts_the_sentinel, [condition(clang_available)]) :-
    run_with("a\nend_of_file\nb\n", "END { print NR }\n", "3\n"),
    !,
    run_with("end_of_file\n", "END { print NR }\n", "1\n"),
    !.

% A pattern that matches the literal string -- proof the record reaches the rules.
test(pattern_matches_the_literal_sentinel, [condition(clang_available)]) :-
    run_with("a\nend_of_file\nb\n",
        "$1 == \"end_of_file\" { print \"matched\", NR }\n", "matched 2\n"),
    !.

% --- across the other drivers -------------------------------------------

% Assoc: the sentinel becomes an ordinary key.
test(assoc_key_from_sentinel, [condition(clang_available)]) :-
    run_with("a\nend_of_file\nb\n",
        "{ c[$1]++ } END { print c[\"end_of_file\"], NR }\n", "1 3\n"),
    !.

% Assoc driver printing $0 over the sentinel (the assoc $0 path is separate from the
% scalar one, so it is exercised explicitly).
test(assoc_whole_record_over_sentinel, [condition(clang_available)]) :-
    run_with("a\nend_of_file\nb\n",
        "{ print $0; c[$1]++ } END { print c[\"a\"] }\n",
        "a\nend_of_file\nb\n1\n"),
    !.

% Mixed scalar+assoc.
test(mixed_over_sentinel, [condition(clang_available)]) :-
    run_with("a\nend_of_file\nb\n",
        "{ n++; c[$1]++ } END { print NR, c[\"end_of_file\"] }\n", "3 1\n"),
    !.

% --- regressions: normal and empty input, the core loop unchanged --------

test(normal_input_unchanged, [condition(clang_available)]) :-
    run_with("5 boot\n7 disk\n", "{ print NR, $1 }\n", "1 5\n2 7\n"),
    !,
    run_with("5 boot\n7 disk\n", "{ n++ } END { print n }\n", "2\n"),
    !,
    run_with("5 boot\n7 disk\n", "{ c[$1]++ } END { print c[\"5\"] }\n", "1\n"),
    !.

test(empty_input_unchanged, [condition(clang_available)]) :-
    run_with("", "END { print NR }\n", "0\n"),
    !,
    run_with("", "{ print }\n", ""),
    !.

% A near-miss text must NOT be treated as EOF either (it never was, but pin it).
test(near_miss_text_is_ordinary_data, [condition(clang_available)]) :-
    run_with("end_of_files\nend_of_file\n", "{ print NR, $0 }\n",
        "1 end_of_files\n2 end_of_file\n"),
    !.

:- end_tests(plawk_eof_sentinel).

% --- helpers ---------------------------------------------------------------

odir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_eof_sentinel', Dir),
    ( exists_directory(Dir) -> true ; make_directory_path(Dir) ).

run_with(Input, Src, Expected) :-
    odir(Dir),
    directory_file_path(Dir, 'eof_bin', Bin),
    ( exists_file(Bin) -> delete_file(Bin) ; true ),
    directory_file_path(Dir, 'eof', Prog0),
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

cli(Args, ExpectedStatus) :-
    process_create(path(swipl), ['examples/plawk/bin/plawk' | Args],
        [stdout(pipe(S)), stderr(std), process(Pid)]),
    read_string(S, _, _), close(S),
    process_wait(Pid, exit(Status)),
    assertion(Status == ExpectedStatus).
