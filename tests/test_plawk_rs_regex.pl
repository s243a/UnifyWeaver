:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% GNU-style regular-expression RS. A nonempty RS longer than one character is
% a POSIX ERE; the text matched by it terminates (and is excluded from) the
% current record. A one-character RS remains literal. GNU awk also exposes the
% exact matched separator as RT and ignores zero-length matches while looking
% for record boundaries.
%
% Both native readers are covered: the main input loop uses the transient
% reader, while getline uses the persistent reader.

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(filesex), [make_directory_path/1]).
:- use_module(library(readutil), [read_file_to_string/3]).

:- begin_tests(plawk_regex_rs).

% A variable-width ERE consumes the complete leftmost-longest match and trims
% every matched separator from $0.
test(regex_rs_variable_width_and_trimming, [condition(clang_available)]) :-
    rdir(Dir),
    build_run(Dir, 'vw',
        "BEGIN { RS = \"[[:digit:]]+\" } { print NR, $0 }\n",
        "alpha1beta22gamma333delta", Out, St),
    assertion(St == exit(0)),
    assertion(Out == "1 alpha\n2 beta\n3 gamma\n4 delta\n"), !.

% Alternation, a POSIX character class, grouping, and a multi-byte alternative.
test(regex_rs_alternatives_and_classes, [condition(clang_available)]) :-
    rdir(Dir),
    build_run(Dir, 'ac',
        "BEGIN { RS = \"([[:upper:]]+)|(--+)\" } { print $0 }\n",
        "aaABCbb---ccZZdd", Out, St),
    assertion(St == exit(0)),
    assertion(Out == "aa\nbb\ncc\ndd\n"), !.

% With the complete buffered prefix available, POSIX leftmost-longest chooses
% the optional tail and the longer shared-prefix alternative.
test(regex_rs_optional_tail_is_longest, [condition(clang_available)]) :-
    rdir(Dir),
    build_run(Dir, 'op',
        "BEGIN { RS = \"a(bc)?\" } { print $0, RT }\n",
        "xabcY", Out, St),
    assertion(St == exit(0)),
    assertion(Out == "x abc\nY \n"), !.

test(regex_rs_shared_prefix_alternative_is_longest,
        [condition(clang_available)]) :-
    rdir(Dir),
    build_run(Dir, 'sp',
        "BEGIN { RS = \"a|aabc\" } { print $0, RT }\n",
        "xaabcY", Out, St),
    assertion(St == exit(0)),
    assertion(Out == "x aabc\nY \n"), !.

% A separator at EOF terminates the preceding record but does not manufacture
% an extra empty final record.
test(regex_rs_trailing_separator, [condition(clang_available)]) :-
    rdir(Dir),
    build_run(Dir, 'tr',
        "BEGIN { RS = \"[,:]+\" } { print NR, $0 }\n",
        "a,b:::c,,", Out, St),
    assertion(St == exit(0)),
    assertion(Out == "1 a\n2 b\n3 c\n"), !.

% Leading and consecutive separators yield empty records; a final separator
% still does not create a record after EOF.
test(regex_rs_leading_and_consecutive_separators, [condition(clang_available)]) :-
    rdir(Dir),
    build_run(Dir, 'lc',
        "BEGIN { RS = \"[,]\" } { print length($0) }\n",
        ",a,,b,", Out, St),
    assertion(St == exit(0)),
    assertion(Out == "0\n1\n0\n1\n"), !.

% Record splitting happens before field splitting. This composes a regex RS
% with a regex FS and checks NF as well as projected fields.
test(regex_rs_regex_fs_composition, [condition(clang_available)]) :-
    rdir(Dir),
    build_run(Dir, 'fs',
        "BEGIN { RS = \"[|]+\"; FS = \"[[:space:]]*:[[:space:]]*\" } { print $1, $2, NF }\n",
        "a: 1||b:2|c :   3", Out, St),
    assertion(St == exit(0)),
    assertion(Out == "a 1 2\nb 2 2\nc 3 2\n"), !.

% Backward compatibility: a one-byte metacharacter is a literal RS, not an ERE.
test(regex_rs_single_char_stays_literal, [condition(clang_available)]) :-
    rdir(Dir),
    build_run(Dir, 'sl',
        "BEGIN { RS = \".\" } { print NR, $0 }\n",
        "a.b.c", Out, St),
    assertion(St == exit(0)),
    assertion(Out == "1 a\n2 b\n3 c\n"), !.

% RT is the exact text matched by RS, including the longest variable-width
% match. It is reset to the empty string for the final unterminated record.
test(regex_rs_rt_matched_text, [condition(clang_available)]) :-
    rdir(Dir),
    build_run(Dir, 'rt',
        "BEGIN { RS = \"[0-9]+|--+\" } { print RT }\n",
        "a123b---c", Out, St),
    assertion(St == exit(0)),
    assertion(Out == "123\n---\n\n"), !.

% The clean EOF probe after a separator-terminated last record must not erase
% the last successful RT before END runs.
test(regex_rs_rt_survives_clean_eof, [condition(clang_available)]) :-
    rdir(Dir),
    build_run(Dir, 're',
        "BEGIN { RS = \"[,:]+\" } { print $0 } END { print RT }\n",
        "a,b:::c,,", Out, St),
    assertion(St == exit(0)),
    assertion(Out == "a\nb\nc\n,,\n"), !.

% ^ is anchored to the physical beginning of the stream, not to each record
% buffer produced after an earlier separator.
test(regex_rs_caret_only_matches_file_start, [condition(clang_available)]) :-
    rdir(Dir),
    build_run(Dir, 'an',
        "BEGIN { RS = \"^X|--\" } { print length($0) }\n",
        "Xa--Xb", Out, St),
    assertion(St == exit(0)),
    assertion(Out == "0\n1\n2\n"), !.

% getline is backed by the persistent reader, so it must honor the same regex
% RS semantics as the transient main-loop reader.
test(regex_rs_getline_persistent_reader, [condition(clang_available)]) :-
    rdir(Dir),
    data_file(Dir, 'getline_data.txt', "first12second345third", DataPath),
    format(atom(Src),
        "BEGIN { RS = \"[0-9]+\" } { while ((getline v < \"~w\") > 0) print v }\n",
        [DataPath]),
    build_run(Dir, 'gl', Src, "trigger", Out, St),
    assertion(St == exit(0)),
    assertion(Out == "first\nsecond\nthird\n"), !.

% The persistent reader also preserves its final matched RT across the failed
% getline call that terminates the loop.
test(regex_rs_getline_rt_survives_clean_eof, [condition(clang_available)]) :-
    rdir(Dir),
    data_file(Dir, 'getline_rt_data.txt', "first12second345", DataPath),
    format(atom(Src),
        "BEGIN { RS = \"[0-9]+\" } { while ((getline v < \"~w\") > 0) print v } END { print RT }\n",
        [DataPath]),
    build_run(Dir, 'gr', Src, "trigger", Out, St),
    assertion(St == exit(0)),
    assertion(Out == "first\nsecond\n345\n"), !.

% The separator starts in the final two bytes of the 4096-byte reader block,
% continues after refill, and also forces the record output buffer to grow.
% This catches boundary bugs in the transient reader and confirms RT survives
% the later record terminator write.
test(regex_rs_crosses_refill_and_output_growth, [condition(clang_available)]) :-
    rdir(Dir),
    length(PrefixCodes, 4094),
    maplist(=(0'a), PrefixCodes),
    string_codes(Prefix, PrefixCodes),
    string_concat(Prefix, "12345b", Input),
    build_run(Dir, 'bd',
        "BEGIN { RS = \"[[:digit:]]+\" } { print length($0), RT }\n",
        Input, Out, St),
    assertion(St == exit(0)),
    assertion(Out == "4094 12345\n1 \n"), !.

% A $-anchored match can be valid exactly at one block boundary, then disappear
% when the next block arrives and reveal a shorter alternative. The post-match
% tail spans both blocks and must be copied back for the following record.
test(regex_rs_replays_cross_block_tail_after_match_changes,
        [condition(clang_available)]) :-
    rdir(Dir),
    length(PrefixCodes, 4094),
    maplist(=(0'x), PrefixCodes),
    string_codes(Prefix, PrefixCodes),
    string_concat(Prefix, "abX", Input),
    build_run(Dir, 'rb',
        "BEGIN { RS = \"ab$|a\" } { print length($0), RT }\n",
        Input, Out, St),
    assertion(St == exit(0)),
    assertion(Out == "4094 a\n2 \n"), !.

% A long record with no separator is scanned once per reader block rather than
% once per byte. The bounded runner makes a repeated-prefix regression fail
% promptly instead of turning into a regex denial of service.
test(regex_rs_long_no_match_is_bounded, [condition(clang_available)]) :-
    rdir(Dir),
    length(InputCodes, 5000),
    maplist(=(0'x), InputCodes),
    string_codes(Input, InputCodes),
    build_run(Dir, 'pf',
        "BEGIN { RS = \"[x]+END\" } { print length($0) }\n",
        Input, Out, St),
    assertion(St == exit(0)),
    assertion(Out == "5000\n"), !.

% A malformed multi-character ERE must be rejected with a diagnostic and a
% normal nonzero exit, either while building or when the generated binary
% initializes its regex. A signal or timeout is not a clean rejection.
test(regex_rs_invalid_ere_rejected_cleanly, [condition(clang_available)]) :-
    rdir(Dir),
    build_run_outcome(Dir, 'bad',
        "BEGIN { RS = \"[0-9\" } { print $0 }\n", "abc",
        Out, BuildStatus, RunStatus, Diagnostic),
    assertion(BuildStatus == exit(0)),
    assertion(RunStatus == exit(2)),
    assertion(Out == ""),
    assertion(sub_string(Diagnostic, _, _, _,
        "plawk: invalid RS regular expression")), !.

% Regex engines can report an empty match for (). Record splitting must ignore
% it (GNU awk semantics), make forward progress, and return the input once.
% The file-backed capture plus timeout keeps a broken reader from hanging CI.
test(regex_rs_zero_length_match_is_ignored, [condition(clang_available)]) :-
    rdir(Dir),
    build_run(Dir, 'zl',
        "BEGIN { RS = \"()\" } { print NR, $0 }\n",
        "abc", Out, St),
    assertion(St == exit(0)),
    assertion(Out == "1 abc\n"), !.

:- end_tests(plawk_regex_rs).

% --- helpers ---------------------------------------------------------------

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

rdir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_regex_rs', Dir),
    ( exists_directory(Dir) -> true ; make_directory_path(Dir) ).

data_file(Dir, Name, Contents, Path) :-
    directory_file_path(Dir, Name, Path),
    setup_call_cleanup(open(Path, write, S, [encoding(utf8)]),
        write(S, Contents), close(S)).

build_run(Dir, Name, Src, Input, Out, RunStatus) :-
    build_run_outcome(Dir, Name, Src, Input,
        Out, exit(0), RunStatus, _Diagnostic).

build_run_outcome(Dir, Name, Src, Input,
        Out, BuildStatus, RunStatus, Diagnostic) :-
    directory_file_path(Dir, Name, Prog0),
    atom_concat(Prog0, '.plawk', Prog),
    setup_call_cleanup(open(Prog, write, S, [encoding(utf8)]),
        write(S, Src), close(S)),
    atom_concat(Prog0, '_bin', Bin),
    atom_concat(Prog0, '_build.out', BuildOutPath),
    atom_concat(Prog0, '_build.err', BuildErrPath),
    run_build(Prog, Bin, BuildOutPath, BuildErrPath, BuildStatus),
    read_file_to_string(BuildOutPath, BuildOut, []),
    read_file_to_string(BuildErrPath, BuildErr, []),
    (   BuildStatus == exit(0)
    ->  run_binary_bounded(Prog0, Bin, Input, Out, RunStatus, RunErr),
        string_concat(BuildOut, BuildErr, BuildDiagnostic),
        string_concat(BuildDiagnostic, RunErr, Diagnostic)
    ;   Out = "",
        RunStatus = not_run,
        string_concat(BuildOut, BuildErr, Diagnostic)
    ).

run_build(Prog, Bin, OutPath, ErrPath, Status) :-
    setup_call_cleanup(
        ( open(OutPath, write, OutS, [encoding(utf8)]),
          open(ErrPath, write, ErrS, [encoding(utf8)])
        ),
        ( process_create(path(swipl),
              ['examples/plawk/bin/plawk', build, Prog, '-o', Bin],
              [ stdout(stream(OutS)), stderr(stream(ErrS)), process(Pid) ]),
          process_wait(Pid, Status)
        ),
        ( close(OutS), close(ErrS) )
    ).

run_binary_bounded(Prog0, Bin, Input, Out, Status, Err) :-
    atom_concat(Prog0, '_input.txt', InputPath),
    atom_concat(Prog0, '_run.out', OutPath),
    atom_concat(Prog0, '_run.err', ErrPath),
    setup_call_cleanup(open(InputPath, write, InputWrite, [encoding(utf8)]),
        write(InputWrite, Input), close(InputWrite)),
    setup_call_cleanup(
        ( open(InputPath, read, InputRead, [type(binary)]),
          open(OutPath, write, OutWrite, [type(binary)]),
          open(ErrPath, write, ErrWrite, [type(binary)])
        ),
        ( process_create(Bin, ['-'],
              [ stdin(stream(InputRead)), stdout(stream(OutWrite)),
                stderr(stream(ErrWrite)), process(Pid) ]),
          process_wait(Pid, WaitStatus, [timeout(10)]),
          finish_bounded_process(Pid, WaitStatus, Status)
        ),
        ( close(InputRead), close(OutWrite), close(ErrWrite) )
    ),
    read_file_to_string(OutPath, Out, []),
    read_file_to_string(ErrPath, Err, []).

finish_bounded_process(Pid, timeout, timeout) :-
    !,
    process_kill(Pid),
    process_wait(Pid, _).
finish_bounded_process(_Pid, Status, Status).
