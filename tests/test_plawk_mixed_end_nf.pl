:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% `END { print NF }` in the MIXED route -- a program whose rules also touch an assoc table.
%
% ---------------------------------------------------------------------------
% A THREADED CAPABILITY, NOT A DELETED DUPLICATE
%
% The END print-field vocabulary is six kinds across three walkers, and every END-shaped gap
% this campaign has closed was a missing cell. The `special` row reads:
%
%                NR     RT     NF
%   scalar       yes    yes    yes
%   mixed        yes    yes    yes  <- was no
%   assoc        no     no     no
%
% The previous collapse (the bare-scalar print, tests/test_plawk_mixed_scalar_print.pl) was a
% STALE DUPLICATE: two copies of one emitter, one of which had stopped learning, and deleting
% it closed four divergences for free. This cell is the OTHER kind, and the distinction is
% worth keeping because it predicts the cost:
%
%   stale duplicate   -> the behaviour already exists ten lines away; deleting the copy
%                        closes the gap and cannot regress anything, since one emitter cannot
%                        drift from itself.
%   missing capability -> the behaviour cannot be expressed in that route at all until
%                        something is THREADED to it. Additive work, and it can fail loudly.
%
% NF in END needs the last record, which is gone by the time END runs, so it needs
% `EndRecord` -- the token plawk_end_record_source/4 issues TOGETHER WITH the retain IR that
% makes it true. The mixed driver never asked for it. Closing this cell therefore meant:
% obtain the token, splice the retain IR into the record loop before the rule chain (so a
% `next` cannot skip the copy), emit the retain globals pay-per-use, and thread the token to
% the walker -- then reuse plawk_end_lastrec_nf_lines//2 rather than copy it.
%
% HOW IT FAILED ON THE WAY, which is the useful part: threading the token WITHOUT emitting
% the globals produced a program that called `@plawk_lastrec_transient` with nobody defining
% it -- a clang failure, exit 4. That is this capability's one loud failure mode, and it is
% why the token and the IR that backs it are issued as a PAIR by one predicate: a projection
% that compiles without its store would print empty instead.
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

% Three records; the LAST is "7 disk", so NF is 2 and NR is 3. c["5"] is 2.
input("5 boot\n5 trace\n7 disk\n").

:- begin_tests(plawk_mixed_end_nf).

% --- NF beside an assoc read --------------------------------------------

test(nf_then_assoc_read, [condition(clang_available)]) :-
    run("{ n++; c[$1]++ } END { print NF, c[\"5\"] }\n", "2 2\n"),
    !.

% NF second, so the value is not merely the first field falling out of some default.
test(assoc_read_then_nf, [condition(clang_available)]) :-
    run("{ n++; c[$1]++ } END { print c[\"5\"], NF }\n", "2 2\n"),
    !.

% All four field kinds the mixed walker now covers, in one print.
test(nf_nr_scalar_and_assoc_together, [condition(clang_available)]) :-
    run("{ n++; c[$1]++ } END { print NF, NR, n, c[\"5\"] }\n", "2 3 3 2\n"),
    !.

% NF is the LAST record's field count, not the first record's and not a running value.
% "7 disk" has 2 fields where the first record also has 2 -- so use an input whose last
% record differs, or the test cannot distinguish them.
test(nf_is_the_last_records_field_count, [condition(clang_available)]) :-
    run_with("a b c\nd e\n", "{ n++; c[$1]++ } END { print NF, c[\"a\"] }\n", "2 1\n"),
    !,
    run_with("a b\nd e f\n", "{ n++; c[$1]++ } END { print NF, c[\"a\"] }\n", "3 1\n"),
    !.

% --- regressions in the same route --------------------------------------

test(nr_in_the_mixed_route_unchanged, [condition(clang_available)]) :-
    run("{ n++; c[$1]++ } END { print NR, c[\"5\"] }\n", "3 2\n"),
    !.

test(bare_scalar_in_the_mixed_route_unchanged, [condition(clang_available)]) :-
    run("{ n++; c[$1]++ } END { print n, c[\"5\"] }\n", "3 2\n"),
    !.

test(nf_in_the_scalar_only_route_unchanged, [condition(clang_available)]) :-
    run("{ n++ } END { print NF }\n", "2\n"),
    !.

% --- pay-per-use: the retain machinery appears ONLY when a field reads the record

% A mixed program that reads no record field must not carry the retain buffer at all. This is
% what keeps the capability free for every other program -- and it is checked on the
% construct's OWN generated names, never a driver-wide mnemonic.
test(the_retain_buffer_is_absent_when_nothing_reads_the_record) :-
    build_ll("{ n++; c[$1]++ } END { print n, c[\"5\"] }\n", LL),
    assertion(\+ sub_string(LL, _, _, _, "@plawk_lastrec_buf")),
    assertion(\+ sub_string(LL, _, _, _, "@plawk_lastrec_store")),
    !.

% ...and present, with its store, when NF does read it. A projection emitted WITHOUT the
% store would print empty; a store emitted without a definition is the exit-4 clang failure
% this change hit on the way. Both halves asserted.
test(the_retain_buffer_and_its_store_appear_together_for_nf) :-
    build_ll("{ n++; c[$1]++ } END { print NF, c[\"5\"] }\n", LL),
    assertion(sub_string(LL, _, _, _, "@plawk_lastrec_buf")),
    assertion(sub_string(LL, _, _, _, "define void @plawk_lastrec_store")),
    assertion(sub_string(LL, _, _, _, "call void @plawk_lastrec_store")),
    !.

% --- one emitter, both routes ------------------------------------------
%
% NF must lower identically in both routes -- the point of reusing
% plawk_end_lastrec_nf_lines//2 instead of copying it, which is exactly the mistake the
% bare-scalar print made.
test(both_routes_emit_the_same_nf_instructions) :-
    nf_lines("{ n++ } END { print NF }\n", ScalarOnly),
    nf_lines("{ n++; c[$1]++ } END { print NF, c[\"5\"] }\n", Mixed),
    assertion(ScalarOnly \== []),
    assertion(ScalarOnly == Mixed),
    !.

% --- the route boundaries, pinned with their reasons ------------------

% The STATEMENT-LIST form still declines. `plawk_mixed_end_print_body_ir/5` passes
% `no_end_record` on purpose: it is reached through plawk_end_list_bodies/8, which serves the
% assoc chain too and does not carry the token. Widening it means threading the capability
% through that shared dispatcher -- its own change. Pinned so the boundary is stated rather
% than discovered.
test(the_statement_list_form_still_declines) :-
    build_status("{ n++; c[$1]++ } END { print NF; print n }\n", 3),
    !.

% NF with NO assoc field in the END print declines -- a driver-SELECTION boundary, not this
% capability: with nothing reading a table in END, the program does not reach the mixed
% driver at all. Paired with the working form so the difference is visible.
test(nf_without_an_assoc_end_field_declines) :-
    build_status("{ n++; c[$1]++ } END { print NF }\n", 3),
    !.

% The ASSOC-ONLY route (no scalar slots) has no `special` clauses at all -- the whole row of
% the matrix is empty there, for NR as much as NF, so this is not an NF gap. Pinned as a
% trio so the row moves together when it moves.
test(the_assoc_only_route_declines_every_special) :-
    build_status("{ c[$1]++ } END { print NF, c[\"5\"] }\n", 3),
    build_status("{ c[$1]++ } END { print NR, c[\"5\"] }\n", 3),
    build_status("{ c[$1]++ } END { print RT, c[\"5\"] }\n", 3),
    !.

:- end_tests(plawk_mixed_end_nf).

% --- helpers ---------------------------------------------------------------

odir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_mixed_end_nf', Dir),
    ( exists_directory(Dir) -> true ; make_directory_path(Dir) ).

run(Src, Expected) :-
    input(Input),
    run_with(Input, Src, Expected).

run_with(Input, Src, Expected) :-
    odir(Dir),
    directory_file_path(Dir, 'nf_bin', Bin),
    % A build that unexpectedly declines must not silently run a stale binary.
    ( exists_file(Bin) -> delete_file(Bin) ; true ),
    directory_file_path(Dir, 'nf', Prog0),
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
    directory_file_path(Dir, 'nf_reject', Prog0),
    atom_concat(Prog0, '.plawk', Prog),
    setup_call_cleanup(open(Prog, write, S, [encoding(utf8)]),
        write(S, Src), close(S)),
    atom_concat(Prog0, '_bin', Bin),
    cli([build, Prog, '-o', Bin], ExpectedStatus).

build_ll(Src, LL) :-
    plawk_parse_string(Src, Program),
    plawk_program_native_driver_ir(Program, 'input.txt', IR),
    atom_string(IR, LL).

% The instruction lines the NF projection contributes, by their own generated names.
nf_lines(Src, Lines) :-
    build_ll(Src, LL),
    split_string(LL, "\n", " ", All),
    include(nf_line, All, Lines).

nf_line(Line) :-
    (   sub_string(Line, _, _, _, "end_lastrec_nf")
    ;   sub_string(Line, _, _, _, "end_nf_")
    ),
    !.

cli(Args, ExpectedStatus) :-
    process_create(path(swipl), ['examples/plawk/bin/plawk' | Args],
        [stdout(pipe(S)), stderr(std), process(Pid)]),
    read_string(S, _, _), close(S),
    process_wait(Pid, exit(Status)),
    assertion(Status == ExpectedStatus).
