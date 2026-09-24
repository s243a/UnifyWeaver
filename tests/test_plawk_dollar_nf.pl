:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% `$NF` and `$(NF+K)` / `$(NF-K)`: the last field, and fields counted from it.
%
% ---------------------------------------------------------------------------
% One of the most common awk idioms (`{ print $NF }`), and it failed to parse
% (exit 2): the grammar only knew `$` followed by an integer literal.
%
% It parses to a DISTINCT functor, `field_nf(Offset)`, not `field(nf(Offset))`:
% dozens of codegen rows take `field(Index)` and compare Index arithmetically, and a
% non-integer there would throw instead of declining. Every context that has not
% been taught `field_nf` declines; the ones taught here are print (rule body, `if`
% branches, concatenation, printf) and END (the retained record).
%
% Lowering: count the fields, add the literal offset, take that field through the
% SUBSLICER -- because `$(NF-1)` on a one-field record is `$0`, which only the
% subslicer serves. A NEGATIVE index is fatal in awk: gawk prints "attempt to access
% field -1" and exits 2, and so does plawk (a runtime guard, called only for
% negative offsets).
%
% gawk is the oracle (verified against gawk 5.1.0, LC_ALL=C), exit status included.

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(lists)).
:- use_module(library(filesex), [make_directory_path/1]).
:- use_module('../examples/plawk/parser/plawk_parser').
:- use_module('../examples/plawk/codegen/llvm/plawk_native_codegen').

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

% Records of 3, 1, 0 and 4 fields (the last with runs of blanks).
input("alice 30 NY\nbob\n\ncarol 35  NY  x\n").

:- begin_tests(plawk_dollar_nf).

% --- parsing ---------------------------------------------------------------

test(forms_parse_to_field_nf) :-
    forall(member(Src-Offset,
            [ "{ print $NF }\n" - 0,
              "{ print $(NF) }\n" - 0,
              "{ print $(NF-1) }\n" - -1,
              "{ print $( NF - 2 ) }\n" - -2,
              "{ print $(NF+1) }\n" - 1
            ]),
        plawk_parse_string(Src,
            program([], [rule(always, [print([field_nf(Offset)])])], []))),
    !.

% `$NFX` is not `$NF` followed by X.
test(nf_needs_an_identifier_boundary) :-
    \+ plawk_parse_string("{ print $NFX }\n", _),
    !.

% --- rule body -------------------------------------------------------------

test(last_field, [condition(clang_available)]) :-
    run("{ print $NF }\n", "NY\nbob\n\nx\n"),
    !.

test(with_other_fields_and_in_concat, [condition(clang_available)]) :-
    run("{ print $1, $NF }\n", "alice NY\nbob bob\n \ncarol x\n"),
    !,
    run("{ print $NF $1 }\n", "NYalice\nbobbob\n\nxcarol\n"),
    !.

% Past the last field is empty.
test(positive_offset_is_past_the_end, [condition(clang_available)]) :-
    run("{ print $(NF+1) \"|\" }\n", "|\n|\n|\n|\n"),
    !.

% `$(NF-1)` on a one-field record is `$0`, the whole record.
test(offset_reaching_field_zero_is_the_record, [condition(clang_available)]) :-
    run_with("a b c\nsolo\n", "{ print $(NF-1) }\n", "b\nsolo\n"),
    !.

test(printf_argument, [condition(clang_available)]) :-
    run("{ printf \"[%s]\\n\", $NF }\n", "[NY]\n[bob]\n[]\n[x]\n"),
    !.

test(inside_a_pattern_rule_and_an_if, [condition(clang_available)]) :-
    run("NR > 1 { print $NF }\n", "bob\n\nx\n"),
    !,
    run("{ if (NF > 1) print $(NF-1) }\n", "30\nNY\n"),
    !.

test(custom_field_separator, [condition(clang_available)]) :-
    run_with("a:b:c\nx:y\n", "BEGIN { FS = \":\" } { print $NF, $(NF-1) }\n",
        "c b\ny x\n"),
    !,
    % default FS: the whole colon-joined line is one field
    run_with("a:b:c\nx:y\n", "{ print $NF }\n", "a:b:c\nx:y\n"),
    !.

% --- END: the retained last record ------------------------------------------

test(end_reads_the_last_record, [condition(clang_available)]) :-
    run("END { print $NF }\n", "x\n"),
    !,
    run("END { print $(NF-1), $NF }\n", "NY x\n"),
    !,
    run("{ n++ } END { print $NF, n }\n", "x 4\n"),
    !.

test(end_printf, [condition(clang_available)]) :-
    run("END { printf \"%s|\\n\", $NF }\n", "x|\n"),
    !.

% `$(NF-4)` on the four-field last record is `$0`.
test(end_offset_reaching_the_record, [condition(clang_available)]) :-
    run("END { print $(NF-4) }\n", "carol 35  NY  x\n"),
    !.

% --- a negative index is fatal, exit 2, after the output so far -------------

test(negative_index_is_fatal_like_gawk, [condition(clang_available)]) :-
    % record 3 is empty: NF-1 = -1
    run_status("{ print $(NF-1) }\n", "30\nbob\n", 2),
    !,
    % record 2 has one field: NF-2 = -1
    run_status("{ print $(NF-2) \"|\" }\n", "alice|\n", 2),
    !.

% Only a NEGATIVE offset pays for the runtime guard.
test(guard_only_for_negative_offsets) :-
    ir_of("{ print $(NF-1) }\n", IRNeg),
    sub_atom(IRNeg, _, _, _, 'call i64 @wam_awk_field_index_checked('),
    ir_of("{ print $NF, $(NF+1) }\n", IRPos),
    \+ sub_atom(IRPos, _, _, _, 'call i64 @wam_awk_field_index_checked('),
    !.

% --- contexts not taught here decline, never miscompile ---------------------

test(untaught_contexts_do_not_miscompile) :-
    forall(member(Src,
            [ "$NF == \"x\" { print }\n",
              "{ c[$NF]++ } END { for (k in c) print k }\n",
              "{ x = $NF; print x }\n",
              "{ print length($NF) }\n",
              "{ print toupper($NF) }\n",
              "{ c[$1]++ } END { print $NF }\n",
              "{ print $NF; c[$1]++ }\n",
              "END { while (i < 2) { print $NF; i++ } }\n"
            ]),
        build_status_not_4(Src)),
    !.

:- end_tests(plawk_dollar_nf).

% --- helpers ---------------------------------------------------------------

odir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_dollar_nf', Dir),
    ( exists_directory(Dir) -> true ; make_directory_path(Dir) ).

run(Src, Expected) :-
    input(Input),
    run_input_status(Input, Src, Expected, 0).

run_with(Input, Src, Expected) :-
    run_input_status(Input, Src, Expected, 0).

run_status(Src, Expected, ExpectedRC) :-
    input(Input),
    run_input_status(Input, Src, Expected, ExpectedRC).

run_input_status(Input, Src, Expected, ExpectedRC) :-
    odir(Dir),
    directory_file_path(Dir, 'dnf_bin', Bin),
    % A build that unexpectedly declines must not silently run a stale binary.
    ( exists_file(Bin) -> delete_file(Bin) ; true ),
    directory_file_path(Dir, 'dnf', Prog0),
    atom_concat(Prog0, '.plawk', Prog),
    setup_call_cleanup(open(Prog, write, S, [encoding(utf8)]),
        write(S, Src), close(S)),
    atom_concat(Prog0, '_in.txt', In),
    setup_call_cleanup(open(In, write, SI, [encoding(utf8)]),
        write(SI, Input), close(SI)),
    cli([build, Prog, '-o', Bin], 0),
    process_create(Bin, [In], [stdout(pipe(PS)), stderr(null), process(Pid)]),
    read_string(PS, _, Out),
    close(PS),
    process_wait(Pid, exit(RC)),
    ( Out == Expected, RC == ExpectedRC
    -> true
    ;  format(user_error, "~n~w~n  got      ~q (exit ~w)~n  expected ~q (exit ~w)~n",
           [Src, Out, RC, Expected, ExpectedRC]), fail
    ).

build_status_not_4(Src) :-
    odir(Dir),
    directory_file_path(Dir, 'dnf_status', Prog0),
    atom_concat(Prog0, '.plawk', Prog),
    atom_concat(Prog0, '_bin', Bin),
    setup_call_cleanup(open(Prog, write, S, [encoding(utf8)]),
        write(S, Src), close(S)),
    process_create(path(swipl), ['examples/plawk/bin/plawk', build, Prog, '-o', Bin],
        [stdout(pipe(O)), stderr(null), process(Pid)]),
    read_string(O, _, _), close(O),
    process_wait(Pid, exit(Status)),
    ( memberchk(Status, [0, 2, 3])
    -> true
    ;  format(user_error, "~n~w~n  build exit ~w (4 = clang miscompile)~n",
           [Src, Status]), fail
    ).

ir_of(Src, IR) :-
    plawk_parse_string(Src, Program),
    plawk_program_native_driver_ir(Program, 'input.txt', IR).

cli(Args, ExpectedStatus) :-
    process_create(path(swipl), ['examples/plawk/bin/plawk' | Args],
        [stdout(pipe(S)), stderr(std), process(Pid)]),
    read_string(S, _, _), close(S),
    process_wait(Pid, exit(Status)),
    assertion(Status == ExpectedStatus).
