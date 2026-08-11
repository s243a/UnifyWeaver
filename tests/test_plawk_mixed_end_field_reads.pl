:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% Record reads in an END print of the MIXED route -- `END { print $N, c["x"] }` and
% `END { print "n=" n, c["x"] }`.
%
% ---------------------------------------------------------------------------
% THE CHEAP HALF OF A CAPABILITY
%
% The END print-field vocabulary is six kinds across three walkers. The mixed row was:
%
%            assoc  concat  field  special  string  var
%   mixed     yes     NO     NO      yes     yes    yes
%
% Both remaining cells landed here for two clauses each, and the reason is the whole point of
% doing the NF change first. A field read in END needs the RETAINED last record -- the real
% one is gone by then -- which is the `EndRecord` capability, and NF needed exactly the same
% thing. Once that was threaded into this route (obtain the token, splice the retain IR into
% the record loop, emit the retain globals pay-per-use), these two cells cost a clause each
% that reuses the scalar route's emitters: plawk_end_lastrec_field_lines//3 and
% plawk_end_concat_parts//5.
%
% So the two kinds of missing cell have different economics, and the ORDER matters:
%
%   missing capability -> expensive, additive, can fail loudly (NF: an exit-4 clang failure
%                         when the token was threaded but its globals were not).
%   the cells after it -> a clause each, reusing emitters that already exist.
%
% Before the capability was threaded these cells were unreachable at any price. That is worth
% remembering when sizing a row: pay for the capability once, then the row fills in cheaply --
% and picking the capability-shaped cell first is what makes the rest cheap.
%
% NOT copies. Both clauses call the SAME emitters the scalar route calls. Copying them is the
% mistake the bare-scalar print made (tests/test_plawk_mixed_scalar_print.pl), where a second
% copy silently stopped learning and un-did shipped behaviour for a subset of programs.
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

% Three records; the LAST is "7 disk" -- $1 is 7, $2 is disk, $0 is "7 disk", NF is 2 and
% $3 is absent. c["5"] is 2. The last record differs from the first, so a test cannot pass by
% reading the wrong record.
input("5 boot\n5 trace\n7 disk\n").

:- begin_tests(plawk_mixed_end_field_reads).

% --- a field read beside an assoc read -----------------------------------

test(first_field_of_the_last_record, [condition(clang_available)]) :-
    run("{ n++; c[$1]++ } END { print $1, c[\"5\"] }\n", "7 2\n"),
    !.

test(second_field_of_the_last_record, [condition(clang_available)]) :-
    run("{ n++; c[$1]++ } END { print $2, c[\"5\"] }\n", "disk 2\n"),
    !.

test(whole_record, [condition(clang_available)]) :-
    run("{ n++; c[$1]++ } END { print $0, c[\"5\"] }\n", "7 disk 2\n"),
    !.

% A field past NF is the empty string, not garbage from the buffer's tail.
test(a_field_past_nf_is_empty, [condition(clang_available)]) :-
    run("{ n++; c[$1]++ } END { print $3, c[\"5\"] }\n", " 2\n"),
    !.

% The field read is of the LAST record. With a different first record, reading the wrong one
% is visible rather than coincidentally right.
test(it_is_the_last_record_not_the_first, [condition(clang_available)]) :-
    run_with("aaa 1\nzzz 9\n", "{ n++; c[$1]++ } END { print $1, c[\"aaa\"] }\n",
        "zzz 1\n"),
    !.

% --- concatenation ------------------------------------------------------

test(concat_of_a_literal_and_a_scalar, [condition(clang_available)]) :-
    run("{ n++; c[$1]++ } END { print \"n=\" n, c[\"5\"] }\n", "n=3 2\n"),
    !.

test(concat_of_a_literal_and_a_field, [condition(clang_available)]) :-
    run("{ n++; c[$1]++ } END { print \"f=\" $1, c[\"5\"] }\n", "f=7 2\n"),
    !.

% --- regressions: the cells the row already had -------------------------

test(nf_unchanged, [condition(clang_available)]) :-
    run("{ n++; c[$1]++ } END { print NF, c[\"5\"] }\n", "2 2\n"),
    !.

test(nr_unchanged, [condition(clang_available)]) :-
    run("{ n++; c[$1]++ } END { print NR, c[\"5\"] }\n", "3 2\n"),
    !.

test(bare_scalar_unchanged, [condition(clang_available)]) :-
    run("{ n++; c[$1]++ } END { print n, c[\"5\"] }\n", "3 2\n"),
    !.

test(scalar_only_field_read_unchanged, [condition(clang_available)]) :-
    run("{ n++ } END { print $1 }\n", "7\n"),
    !.

test(scalar_only_concat_unchanged, [condition(clang_available)]) :-
    run("{ n++ } END { print \"n=\" n }\n", "n=3\n"),
    !.

% --- the field read must NOT be captured by the generic expression clause

% `field(N)` would otherwise fall into the mixed walker's catch-all scalar-expression clause
% and lower as an IN-LOOP field read -- against a record that no longer exists by END. The
% clause order is what prevents that, and clause order carries no marker of its own, so this
% pins the outcome: the emitted instructions must be the retained-record projection's.
test(the_field_read_uses_the_retained_record_projection) :-
    build_ll("{ n++; c[$1]++ } END { print $1, c[\"5\"] }\n", LL),
    assertion(sub_string(LL, _, _, _, "@plawk_lastrec_transient")),
    assertion(sub_string(LL, _, _, _, "call void @plawk_lastrec_store")),
    !.

% ...and both routes emit the same field instructions, which is the property that reusing the
% emitter buys.
test(both_routes_emit_the_same_field_read) :-
    field_lines("{ n++ } END { print $1 }\n", ScalarOnly),
    field_lines("{ n++; c[$1]++ } END { print $1, c[\"5\"] }\n", Mixed),
    assertion(ScalarOnly \== []),
    assertion(ScalarOnly == Mixed),
    !.

% Pay-per-use still holds: a mixed program reading no record field carries no retain buffer.
test(no_retain_buffer_when_no_field_is_read) :-
    build_ll("{ n++; c[$1]++ } END { print n, c[\"5\"] }\n", LL),
    assertion(\+ sub_string(LL, _, _, _, "@plawk_lastrec_buf")),
    !.

% --- the boundary, unchanged and still pinned ---------------------------

% These two STILL decline, but the reason CHANGED and this comment must change with
% it or it protects a stale attribution. The old reason -- plawk_end_list_bodies/8
% carried no token -- is retired: the dispatcher threads EndRecord now, and the same
% programs WITH an assoc read in the END list compile (pinned below). What remains
% is the driver-SELECTION boundary this suite already pins for the single-print
% form: an END that reads no table never reaches the mixed driver at all.
test(the_statement_list_form_without_an_assoc_read_still_declines) :-
    build_status("{ n++; c[$1]++ } END { print $1; print n }\n", 3),
    build_status("{ n++; c[$1]++ } END { print NF; print n }\n", 3),
    !.

% ...and the token half is retired: the same statements compile once any statement
% in the list reads a table.
test(the_statement_list_form_with_an_assoc_read_works,
        [condition(clang_available)]) :-
    run("{ n++; c[$1]++ } END { print $1; print n, c[\"5\"] }\n", "7\n3 2\n"),
    !,
    run("{ n++; c[$1]++ } END { print NF; print c[\"5\"] }\n", "2\n2\n"),
    !.

:- end_tests(plawk_mixed_end_field_reads).

% --- helpers ---------------------------------------------------------------

odir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_mixed_end_field_reads', Dir),
    ( exists_directory(Dir) -> true ; make_directory_path(Dir) ).

run(Src, Expected) :-
    input(Input),
    run_with(Input, Src, Expected).

run_with(Input, Src, Expected) :-
    odir(Dir),
    directory_file_path(Dir, 'mf_bin', Bin),
    % A build that unexpectedly declines must not silently run a stale binary.
    ( exists_file(Bin) -> delete_file(Bin) ; true ),
    directory_file_path(Dir, 'mf', Prog0),
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

build_status(Src, ExpectedStatus) :-
    odir(Dir),
    directory_file_path(Dir, 'mf_reject', Prog0),
    atom_concat(Prog0, '.plawk', Prog),
    setup_call_cleanup(open(Prog, write, S, [encoding(utf8)]),
        write(S, Src), close(S)),
    atom_concat(Prog0, '_bin', Bin),
    cli([build, Prog, '-o', Bin], ExpectedStatus).

build_ll(Src, LL) :-
    plawk_parse_string(Src, Program),
    plawk_program_native_driver_ir(Program, 'input.txt', IR),
    atom_string(IR, LL).

% The instruction lines the END field projection contributes, by their own generated names.
field_lines(Src, Lines) :-
    build_ll(Src, LL),
    split_string(LL, "\n", " ", All),
    include(field_line, All, Lines).

% The field projection's OWN generated names -- `%end_field_N_*` for `$N` and `%end_rec_N_*`
% for `$0` (plawk_end_lastrec_field_lines//3 names them, and "end_lastrec" appears only in the
% runtime's global/function names, not in the per-print instructions, which is why an earlier
% version of this filter matched nothing at all and the `\== []` guard caught it).
field_line(Line) :-
    (   sub_string(Line, _, _, _, "end_field_")
    ;   sub_string(Line, _, _, _, "end_rec_")
    ),
    !.

cli(Args, ExpectedStatus) :-
    process_create(path(swipl), ['examples/plawk/bin/plawk' | Args],
        [stdout(pipe(S)), stderr(std), process(Pid)]),
    read_string(S, _, _), close(S),
    process_wait(Pid, exit(Status)),
    assertion(Status == ExpectedStatus).
