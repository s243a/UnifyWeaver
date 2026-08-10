:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% Record reads and specials in an END print of the ASSOC-ONLY route -- a program with no
% scalar slots at all: `{ c[$1]++ } END { print $1, NR, NF, c["x"] }`.
%
% ---------------------------------------------------------------------------
% THE LAST ROW OF THE MATRIX, AND THREE DIFFERENT REASONS FOR ONE HOLE
%
% The END print-field vocabulary is six kinds across three walkers. This route's row was the
% emptiest -- 2 of 6 -- and closing it separated three costs that all LOOK like "a missing
% clause" from outside:
%
%   field, NF, concat   needed the `EndRecord` CAPABILITY: the retained last record, since the
%                       real one is gone by END. Paid once, in this route's TWO driver clauses
%                       (obtain the token, splice the retain IR into the record loop before
%                       the rule chain, emit the retain globals pay-per-use).
%   NR                  needed nothing but something to ask. plawk_end_nr_value/2 already falls
%                       back to `%plawk_nr`, the counter this driver emits when a print
%                       mentions NR. `state_plan([], [])` is not a stub: it is the honest
%                       statement that this route HAS no scalar slots.
%   RT                  needed nothing at all -- it reads @wam_rt_ptr directly. It was absent
%                       for no reason beyond nobody adding the row.
%
% So of the four holes, ONE was a real capability and the other three were rows. That ratio is
% the argument for auditing a row before sizing it: "four missing cells" and "one capability
% plus three clauses" are the same list and very different work.
%
% Every clause reuses the scalar route's emitter. None is a copy -- copying is what made the
% bare-scalar print drift (tests/test_plawk_mixed_scalar_print.pl), where the second copy
% silently un-did shipped behaviour for a subset of programs.
%
% ---------------------------------------------------------------------------
% WHAT IS STILL REFUSED, AND WHY EACH IS A DIFFERENT KIND OF NO
%
%   a concat containing an ASSOC READ (`print "k=" c["x"]`) -- not a missing cell. The concat
%     cell IS filled; what is missing is a PART KIND inside the shared concat emitter, which
%     handles literals, fields and specials but not a table read. A concat of parts it does
%     handle works here (pinned).
%   the STATEMENT-LIST form (`END { print $1; print c["x"] }`) -- reached through
%     plawk_end_list_bodies/8, which serves both chains and carries no token. One boundary,
%     both routes, pinned in both suites.
%   BINARY mode -- plawk_end_record_source/4 requires an integer FS, so a binfmt program gets
%     `no_end_record` automatically and a field read declines instead of slicing a fixed-layout
%     record as text. That refusal is load-bearing, not an omission.
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

% Three records; the LAST is "7 disk" -- $1 is 7, $2 is disk, $0 is "7 disk", NF is 2, NR is
% 3. c["5"] is 2 and c["7"] is 1. The last record differs from the first, so reading the wrong
% record is visible.
input("5 boot\n5 trace\n7 disk\n").

:- begin_tests(plawk_assoc_end_record).

% --- a field read, with no scalar slot anywhere in the program -----------

test(first_field_of_the_last_record, [condition(clang_available)]) :-
    run("{ c[$1]++ } END { print $1, c[\"5\"] }\n", "7 2\n"),
    !.

test(second_field, [condition(clang_available)]) :-
    run("{ c[$1]++ } END { print $2, c[\"5\"] }\n", "disk 2\n"),
    !.

test(whole_record, [condition(clang_available)]) :-
    run("{ c[$1]++ } END { print $0, c[\"5\"] }\n", "7 disk 2\n"),
    !.

test(a_field_past_nf_is_empty, [condition(clang_available)]) :-
    run("{ c[$1]++ } END { print $3, c[\"5\"] }\n", " 2\n"),
    !.

% It is the LAST record. With a distinct first record, reading the wrong one is visible rather
% than coincidentally right.
test(it_is_the_last_record_not_the_first, [condition(clang_available)]) :-
    run_with("aaa 1\nzzz 9\n", "{ c[$1]++ } END { print $1, c[\"aaa\"] }\n", "zzz 1\n"),
    !.

% --- the specials --------------------------------------------------------

test(nr_in_the_assoc_route, [condition(clang_available)]) :-
    run("{ c[$1]++ } END { print NR, c[\"5\"] }\n", "3 2\n"),
    !.

test(nf_in_the_assoc_route, [condition(clang_available)]) :-
    run("{ c[$1]++ } END { print NF, c[\"5\"] }\n", "2 2\n"),
    !.

% RT is the record terminator -- a newline under the default RS, so this prints an empty first
% field followed by the count.
test(rt_in_the_assoc_route, [condition(clang_available)]) :-
    run("{ c[$1]++ } END { print RT, c[\"5\"] }\n", "\n 2\n"),
    !.

% All of them at once, which is also the shape that would break if the separator index drifted
% while threading a sixth argument through every clause.
test(field_nr_nf_and_an_assoc_read_together, [condition(clang_available)]) :-
    run("{ c[$1]++ } END { print $1, NR, NF, c[\"5\"] }\n", "7 3 2 2\n"),
    !.

% --- concat, for the part kinds the shared emitter handles ---------------

test(concat_of_a_literal_and_a_field, [condition(clang_available)]) :-
    run("{ c[$1]++ } END { print \"f=\" $1, c[\"5\"] }\n", "f=7 2\n"),
    !.

% --- regressions: what this route already did ---------------------------

test(a_literal_key_read_unchanged, [condition(clang_available)]) :-
    run("{ c[$1]++ } END { print c[\"5\"] }\n", "2\n"),
    !.

test(two_literal_key_reads_keep_their_separator, [condition(clang_available)]) :-
    run("{ c[$1]++ } END { print c[\"5\"], c[\"7\"] }\n", "2 1\n"),
    !.

test(a_string_literal_field_unchanged, [condition(clang_available)]) :-
    run("{ c[$1]++ } END { print \"total\", c[\"5\"] }\n", "total 2\n"),
    !.

test(the_forin_end_route_unchanged, [condition(clang_available)]) :-
    run_sorted("{ c[$1]++ } END { for (k in c) print k, c[k] }\n", "5 2\n7 1\n"),
    !.

% A positional (split) table still reads by raw position -- the key-space rule is untouched by
% the capability threading.
test(a_positional_read_unchanged, [condition(clang_available)]) :-
    run("{ split($0, a, \" \") } END { print a[1] }\n", "7\n"),
    !.

% --- pay-per-use --------------------------------------------------------

test(no_retain_buffer_when_no_field_is_read) :-
    build_ll("{ c[$1]++ } END { print c[\"5\"] }\n", LL),
    assertion(\+ sub_string(LL, _, _, _, "@plawk_lastrec_buf")),
    !.

test(the_retain_buffer_and_its_store_appear_together) :-
    build_ll("{ c[$1]++ } END { print $1, c[\"5\"] }\n", LL),
    assertion(sub_string(LL, _, _, _, "@plawk_lastrec_buf")),
    assertion(sub_string(LL, _, _, _, "define void @plawk_lastrec_store")),
    assertion(sub_string(LL, _, _, _, "call void @plawk_lastrec_store")),
    !.

% --- one emitter across all three routes --------------------------------
%
% The field projection must lower identically in the scalar-only, mixed and assoc-only routes.
% That three-way equality is what reusing the emitter buys, and it is the property whose
% absence produced this campaign's bare-scalar-print divergence.
test(all_three_routes_emit_the_same_field_read) :-
    field_lines("{ n++ } END { print $1 }\n", ScalarOnly),
    field_lines("{ n++; c[$1]++ } END { print $1, c[\"5\"] }\n", Mixed),
    field_lines("{ c[$1]++ } END { print $1, c[\"5\"] }\n", AssocOnly),
    assertion(ScalarOnly \== []),
    assertion(ScalarOnly == Mixed),
    assertion(ScalarOnly == AssocOnly),
    !.

% --- the three remaining refusals, each with its own reason -------------

% A concat containing an ASSOC READ. Not a missing cell -- the concat cell is filled (above);
% the shared concat emitter has no PART KIND for a table read. Pinned as a pair with the
% working concat so the difference is visible.
test(a_concat_containing_an_assoc_read_declines) :-
    build_status("{ c[$1]++ } END { print \"k=\" c[\"5\"] }\n", 3),
    !.

% WAS a pair of declines: plawk_end_list_bodies/8 carried no token. It does now --
% the statement-list dispatcher threads EndRecord from the driver, which also
% splices the retain IR and emits its globals. Pinned as the same pair, inverted,
% and the per-statement rename disambiguates the two projections' SSA names, which
% the two-record-reads case below exercises directly.
test(the_statement_list_form_now_works, [condition(clang_available)]) :-
    run("{ c[$1]++ } END { print $1; print c[\"5\"] }\n", "7\n2\n"),
    !,
    run("{ c[$1]++ } END { print NF; print c[\"5\"] }\n", "2\n2\n"),
    !,
    % Two record-reading statements in one list: both projections must resolve, each
    % under its own statement-renamed names.
    run("{ c[$1]++ } END { print $1; print $2 }\n", "7\ndisk\n"),
    !.

:- end_tests(plawk_assoc_end_record).

% --- helpers ---------------------------------------------------------------

odir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_assoc_end_record', Dir),
    ( exists_directory(Dir) -> true ; make_directory_path(Dir) ).

run(Src, Expected) :-
    input(Input),
    run_(Input, Src, Expected, plain).

run_with(Input, Src, Expected) :-
    run_(Input, Src, Expected, plain).

run_sorted(Src, Expected) :-
    input(Input),
    run_(Input, Src, Expected, sorted).

run_(Input, Src, Expected, Mode) :-
    odir(Dir),
    directory_file_path(Dir, 'ae_bin', Bin),
    % A build that unexpectedly declines must not silently run a stale binary.
    ( exists_file(Bin) -> delete_file(Bin) ; true ),
    directory_file_path(Dir, 'ae', Prog0),
    atom_concat(Prog0, '.plawk', Prog),
    setup_call_cleanup(open(Prog, write, S, [encoding(utf8)]),
        write(S, Src), close(S)),
    atom_concat(Prog0, '_in.txt', In),
    setup_call_cleanup(open(In, write, SI, [encoding(utf8)]),
        write(SI, Input), close(SI)),
    cli([build, Prog, '-o', Bin], 0),
    process_create(Bin, [In], [stdout(pipe(PS)), stderr(std), process(Pid)]),
    read_string(PS, _, Out0),
    close(PS),
    process_wait(Pid, exit(0)),
    ( Mode == sorted -> sort_lines(Out0, Out) ; Out = Out0 ),
    ( Out == Expected
    -> true
    ;  format(user_error, "~n~w~n  got      ~q~n  expected ~q~n",
           [Src, Out, Expected]), fail
    ).

sort_lines(In, Out) :-
    split_string(In, "\n", "", Parts0),
    ( append(Parts, [""], Parts0) -> true ; Parts = Parts0 ),
    msort(Parts, Sorted),
    atomic_list_concat(Sorted, '\n', Joined),
    ( Sorted == [] -> Out = "" ; format(string(Out), "~w\n", [Joined]) ).

build_status(Src, ExpectedStatus) :-
    odir(Dir),
    directory_file_path(Dir, 'ae_reject', Prog0),
    atom_concat(Prog0, '.plawk', Prog),
    setup_call_cleanup(open(Prog, write, S, [encoding(utf8)]),
        write(S, Src), close(S)),
    atom_concat(Prog0, '_bin', Bin),
    cli([build, Prog, '-o', Bin], ExpectedStatus).

build_ll(Src, LL) :-
    plawk_parse_string(Src, Program),
    plawk_program_native_driver_ir(Program, 'input.txt', IR),
    atom_string(IR, LL).

% The field projection's OWN generated names -- `%end_field_N_*` for `$N`, `%end_rec_N_*` for
% `$0`. Never a driver-wide mnemonic, and never "end_lastrec", which names only the runtime's
% globals and functions rather than any per-print instruction.
field_lines(Src, Lines) :-
    build_ll(Src, LL),
    split_string(LL, "\n", " ", All),
    include(field_line, All, Lines).

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
