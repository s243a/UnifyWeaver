:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% plawk: field reads in an END block -- `END { print $1 }`.
%
%   { n++ } END { print $1 }        DECLINED (3)
%   { n++ } END { print $0 }        DECLINED (3)
%
% awk keeps `$0` and its fields in END, holding the LAST RECORD READ. gawk on
% `a 1 / b 2 / c 3` prints `c` for `$1` and `c 3` for `$0`.
%
% ---------------------------------------------------------------------------
% THE LAST RECORD IS GONE BY END -- RETENTION MUST BE EXPLICIT
%
% At `end_print` the shared transient record buffer holds the bytes
% `end_of_file`: the EOF sentinel the reader interned on its final call. Proven by
% probe -- before #4100's gate, `END { while (…) print $1 }` printed
% `end_of_file` once per iteration, which is `$1` of the sentinel text.
%
% So this is not a matter of FINDING the record. The record loop now COPIES each
% record into a retained buffer (@plawk_lastrec_store), and END re-materialises
% those bytes in the transient buffer and projects fields from them.
%
% The buffer is a single reused, geometrically grown allocation -- the same shape
% as @wam_rt_set, which is how RT already survives to END. That means CONSTANT
% memory and one memcpy per record however long the stream is. Interning each
% record instead would grow the atom table with every DISTINCT record and break
% the streaming invariant the whole design protects.
%
% ---------------------------------------------------------------------------
% WHERE THE RUNTIME LIVES, AND WHY IT IS NOT IN wam_llvm_target.pl
%
% The globals and the two defines are emitted as PROGRAM-level IR by the codegen,
% not into the shared runtime block. Putting them beside the @wam_rt_* globals was
% tried first and reverted: that block is emitted into EVERY driver, so unused
% globals perturbed the `.ll` of every program for zero functional gain -- and
% unexercised runtime IR is unverified runtime IR.
%
% Emitting them per-program instead makes the feature PAY-PER-USE, which the
% golden-IR test below pins: an END with no field read produces byte-identical IR
% and not one of these symbols.
%
% ---------------------------------------------------------------------------
% ONE GATE, INVERTED -- NOT A SECOND DEFINITION
%
% plawk_end_term_reads_record/1 is the structural term walk #4100 added as the
% END-loop SAFETY gate. This feature INVERTS it rather than restating it: the same
% walk that refuses a record read where none can be projected now decides whether
% to retain. One predicate, two uses, nothing to drift -- which matters given that
% every wrong-output bug in this line was one property implemented twice.
%
% It was called plawk_end_term_mentions_field/1 and matched `field(_)` alone. That
% name described the implementation, and the implementation was NARROWER THAN THE
% PROPERTY: `NF` reads the record without being a `field(_)`, so it walked straight
% past the gate and `END { if (n == 3) print NF }` printed 1 -- NF of the
% `end_of_file` sentinel -- where gawk prints 2. See the NF section below.
%
% plawk_end_record_source/4 returns BOTH the capability token END needs and the
% record-loop IR that makes it true, so a driver cannot emit the projection
% without the store. A projection whose bytes were never retained would print
% EMPTY -- silently wrong, the failure mode this line refuses to ship.
%
% ---------------------------------------------------------------------------
% A PRE-EXISTING WRONG OUTPUT THIS WORK SURFACED
%
% Probing the neighbourhood found `END { if (n == 3) print $1 }` printing
% `end_of_file` -- wrong output that predates this change (confirmed against the
% parent commit, not assumed). plawk_end_if_branch_ir/8 lowers each branch through
% the RULE-BODY print emitter, which projects `$N` from `%line`. That is the
% identical defect #4100 gated for END loops, in a driver nobody re-checked:
% reuse inherits what an emitter ASSUMES, not just what it does.
%
% It is now a clean decline, pinned below, and projecting it is a follow-on that
% needs the prefixed print emitter's record source parameterised.
%
% gawk 5.2 is the oracle for every expectation here.

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(filesex), [make_directory_path/1]).
:- use_module('../examples/plawk/parser/plawk_parser').
:- use_module('../examples/plawk/codegen/llvm/plawk_native_codegen').

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

% Three records, so the last record is `c 3` and a counting rule leaves n = 3.
input("a 1\nb 2\nc 3\n").

:- begin_tests(plawk_end_field_reads).

% --- the feature ---------------------------------------------------------

test(end_print_field_one, [condition(clang_available)]) :-
    run("{ n++ } END { print $1 }\n", "c\n"),
    !.

test(end_print_field_two, [condition(clang_available)]) :-
    run("{ n++ } END { print $2 }\n", "3\n"),
    !.

% `$0` is the whole retained record, printed from its text rather than sliced.
test(end_print_whole_record, [condition(clang_available)]) :-
    run("{ n++ } END { print $0 }\n", "c 3\n"),
    !.

test(end_print_two_fields_with_ofs, [condition(clang_available)]) :-
    run("{ n++ } END { print $1, $2 }\n", "c 3\n"),
    !.

% A field index past NF is the empty string, not garbage -- the slicer returns a
% null slice and `%.*s` with length 0 prints nothing.
test(end_print_field_past_nf_is_empty, [condition(clang_available)]) :-
    run("{ n++ } END { print $9 }\n", "\n"),
    !.

% --- composition with what END already did -------------------------------

test(end_field_mixed_with_a_scalar, [condition(clang_available)]) :-
    run("{ n++ } END { print n, $1 }\n", "3 c\n"),
    !.

% Concatenation: the field goes through the same per-part emitter, so one leading
% separator and no separator between parts.
test(end_field_in_a_concatenation, [condition(clang_available)]) :-
    run("{ n++ } END { print $1 \" / \" $2 }\n", "c / 3\n"),
    !.

test(end_whole_record_in_a_concatenation, [condition(clang_available)]) :-
    run("{ n++ } END { print \"last=\" $0 }\n", "last=c 3\n"),
    !.

% A STATEMENT LIST: each print is lowered independently and its `end_`-prefixed
% names suffixed, so two field reads cannot collide.
test(end_field_in_a_statement_list, [condition(clang_available)]) :-
    run("{ n++ } END { print $1; print $2 }\n", "c\n3\n"),
    !.

test(end_field_then_exit, [condition(clang_available)]) :-
    run_status("{ n++ } END { print $1; exit 2 }\n", "c\n", 2),
    !.

% --- separators ----------------------------------------------------------

% FS is the separator the retained record is sliced with, as in the loop.
test(end_field_honours_fs, [condition(clang_available)]) :-
    run_input("BEGIN { FS = \":\" }\n{ n++ } END { print $2 }\n",
        "x:y:z\np:q:r\n", "q\n"),
    !.

test(end_field_honours_ofs, [condition(clang_available)]) :-
    run("BEGIN { OFS = \"-\" }\n{ n++ } END { print $1, $2 }\n", "c-3\n"),
    !.

% ORS terminates an END field print like every other print.
test(end_field_honours_ors, [condition(clang_available)]) :-
    run("BEGIN { ORS = \"|\" }\n{ n++ } END { print $1 }\n", "c|"),
    !.

% A custom RS: the retained bytes are the record WITHOUT its separator, matching
% what `$0` held inside the loop.
test(end_field_honours_rs, [condition(clang_available)]) :-
    run_input("BEGIN { RS = \";\" }\n{ n++ } END { print $1 }\n",
        "a 1;b 2;c 3", "c\n"),
    !.

% --- the hazards, each probed rather than reasoned about ------------------

% EMPTY INPUT: no record was ever read, so the retained buffer is still null.
% awk gives an empty `$0` in END, not garbage -- the null buffer re-materialises
% as the empty string.
test(empty_input_gives_an_empty_record, [condition(clang_available)]) :-
    run_input("{ n++ } END { print $1 }\n", "", "\n"),
    !.

test(empty_input_gives_an_empty_whole_record, [condition(clang_available)]) :-
    run_input("{ n++ } END { print $0 }\n", "", "\n"),
    !.

% RECORDS LONGER THAN THE INITIAL CAPACITY, so the retained buffer must grow.
% Three records of ~5-11 KB: `$2` of the last one is `tail2`.
test(long_records_grow_the_retained_buffer, [condition(clang_available)]) :-
    long_input(Input),
    run_input("{ n++ } END { print $2 }\n", Input, "tail2\n"),
    !.

% The retain happens on every record READ, before pattern matching -- so a rule
% that matches none of them still leaves the last record available, as in awk.
test(retained_even_when_no_rule_matches, [condition(clang_available)]) :-
    run("/zzz/ { n++ } END { print $1 }\n", "c\n"),
    !.

% ...and `next` skips the rest of the rule body, not the retain.
test(retained_across_next, [condition(clang_available)]) :-
    run("{ n++; next } END { print $1 }\n", "c\n"),
    !.

% --- pay-per-use: the IR of unaffected programs ---------------------------

% An END with NO field read emits none of the retained-record machinery. Pinned on
% the construct's OWN symbols -- `@plawk_lastrec_buf` and `@plawk_lastrec_store`
% appear nowhere else -- rather than on a common LLVM mnemonic, which is how three
% earlier tests in this line passed vacuously.
test(no_end_field_read_emits_no_retained_record_ir) :-
    plawk_parse_string("{ n++ } END { print n }\n", Program),
    plawk_program_native_driver_ir(Program, 'input.txt', DriverIR),
    assertion(\+ sub_atom(DriverIR, _, _, _, '@plawk_lastrec_buf')),
    assertion(\+ sub_atom(DriverIR, _, _, _, '@plawk_lastrec_store')),
    assertion(\+ sub_atom(DriverIR, _, _, _, '@plawk_lastrec_transient')),
    !.

% A program with NO END at all likewise.
test(no_end_block_emits_no_retained_record_ir) :-
    plawk_parse_string("{ print $1 }\n", Program),
    plawk_program_native_driver_ir(Program, 'input.txt', DriverIR),
    assertion(\+ sub_atom(DriverIR, _, _, _, '@plawk_lastrec_buf')),
    !.

% An END field read emits the store, the re-materialisation and the retain call.
test(end_field_read_emits_the_retained_record_ir) :-
    plawk_parse_string("{ n++ } END { print $1 }\n", Program),
    plawk_program_native_driver_ir(Program, 'input.txt', DriverIR),
    assertion(once(sub_atom(DriverIR, _, _, _, '@plawk_lastrec_buf'))),
    assertion(once(sub_atom(DriverIR, _, _, _,
        'define void @plawk_lastrec_store'))),
    % the retain call, in the record loop
    assertion(once(sub_atom(DriverIR, _, _, _,
        'call void @plawk_lastrec_store(i8* %line_s'))),
    % the END-side re-materialisation and slice
    assertion(once(sub_atom(DriverIR, _, _, _, '%end_field_0_id'))),
    assertion(once(sub_atom(DriverIR, _, _, _, '%end_field_0_ptr'))),
    !.

% `$0` re-materialises the record but does NOT slice it.
test(end_whole_record_emits_no_field_slice) :-
    plawk_parse_string("{ n++ } END { print $0 }\n", Program),
    plawk_program_native_driver_ir(Program, 'input.txt', DriverIR),
    assertion(once(sub_atom(DriverIR, _, _, _, '%end_rec_0_id'))),
    assertion(\+ sub_atom(DriverIR, _, _, _, '%end_field_0_ptr')),
    !.

% --- the gate, inverted -------------------------------------------------

% The SAME structural walk that gates END loops decides whether to retain, so the
% two cannot disagree about what counts as a field read.
test(the_retain_gate_is_the_end_loop_field_walk) :-
    assertion(plawk_native_codegen:plawk_end_term_reads_record(
        [print([field(1)])])),
    assertion(\+ plawk_native_codegen:plawk_end_term_reads_record(
        [print([var(n)])])),
    !.

% The rewrite is structural too, and reaches the same places the gate does -- a
% field in a nested loop body, an `if` branch, a print list, a concatenation.
% A per-shape rewriter that missed one would leave a raw field(N) to be lowered
% against `%line`: the `end_of_file` wrong output again.
test(the_rewrite_reaches_a_deeply_nested_field) :-
    plawk_native_codegen:plawk_end_lastrec_rewrite(
        [while_loop(cmp(var(n), gt, int(0)),
             [if(scalar_if(cmp(var(n), eq, int(2))),
                 [print([concat([field(1), string("-"), field(2)])])], [])])],
        Rewritten),
    assertion(\+ plawk_native_codegen:plawk_end_term_reads_record(Rewritten)),
    assertion(Rewritten ==
        [while_loop(cmp(var(n), gt, int(0)),
             [if(scalar_if(cmp(var(n), eq, int(2))),
                 [print([concat([end_lastrec_field(1), string("-"),
                                 end_lastrec_field(2)])])], [])])]),
    !.

% Non-field terms pass through untouched, so the rewrite cannot perturb a
% field-free END.
test(the_rewrite_leaves_a_field_free_term_alone) :-
    Term = [print([var(n), string("x"), special('NR')])],
    plawk_native_codegen:plawk_end_lastrec_rewrite(Term, Rewritten),
    assertion(Rewritten == Term),
    !.

% Under no_end_record nothing is rewritten -- so a driver that did not retain
% cannot end up emitting a projection.
test(no_end_record_rewrites_nothing) :-
    Fields = [field(1)],
    plawk_native_codegen:plawk_end_branch_fields_rewrite(no_end_record, Fields,
        Same),
    assertion(Same == Fields),
    plawk_native_codegen:plawk_end_branch_fields_rewrite(end_record(0' ), Fields,
        Rewritten),
    assertion(Rewritten == [end_lastrec_field(1)]),
    !.

% Text records only: the binary drivers have no `%line_s` to copy from, so the
% source is refused rather than emitting a projection with no store behind it.
test(a_binary_descriptor_gets_no_end_record) :-
    assertion(plawk_native_codegen:plawk_end_record_source(
        binfmt([i64, i64]), [print([field(1)])], no_end_record, '')),
    !.

test(a_text_descriptor_with_a_field_read_gets_one) :-
    plawk_native_codegen:plawk_end_record_source(0' , [print([field(1)])],
        Source, RetainIR),
    assertion(Source == end_record(0' )),
    assertion(RetainIR \== ''),
    !.

test(a_text_descriptor_without_a_field_read_does_not) :-
    assertion(plawk_native_codegen:plawk_end_record_source(
        0' , [print([var(n)])], no_end_record, '')),
    !.

% --- END `if` branches ---------------------------------------------------
%
% These printed `end_of_file` before they were gated, and now project from the
% retained record. The branch goes through the RULE-BODY print emitter, so what
% made this work is the rewrite plus two clauses on that shared emitter -- no
% second print emitter for END.

test(end_if_branch_field_read, [condition(clang_available)]) :-
    run("{ n++ } END { if (n == 3) print $1 }\n", "c\n"),
    !.

test(end_if_else_branch_field_read, [condition(clang_available)]) :-
    run("{ n++ } END { if (n == 3) print \"three\"; else print $2 }\n", "three\n"),
    !.

% Both branches, and the one that runs is the `else`.
test(end_if_both_branches_field_reads, [condition(clang_available)]) :-
    run("{ n++ } END { if (n == 9) print $1; else print $2 }\n", "3\n"),
    !.

test(end_if_branch_whole_record, [condition(clang_available)]) :-
    run("{ n++ } END { if (n == 3) print $0 }\n", "c 3\n"),
    !.

test(end_if_branch_two_fields, [condition(clang_available)]) :-
    run("{ n++ } END { if (n == 3) print $1, $2 }\n", "c 3\n"),
    !.

test(end_if_branch_concatenation, [condition(clang_available)]) :-
    run("{ n++ } END { if (n == 3) print $1 \"/\" $2 }\n", "c/3\n"),
    !.

test(end_if_branch_field_past_nf_is_empty, [condition(clang_available)]) :-
    run("{ n++ } END { if (n == 3) print $9 }\n", "\n"),
    !.

test(end_if_branch_honours_ors, [condition(clang_available)]) :-
    run("BEGIN { ORS = \"|\" }\n{ n++ } END { if (n == 3) print $1 }\n", "c|"),
    !.

% A field-free END-if is untouched.
test(field_free_end_if_unchanged, [condition(clang_available)]) :-
    run("{ n++ } END { if (n == 3) print \"three\"; else print \"other\" }\n",
        "three\n"),
    !.

% The associative END-if supports RECORD-SHAPED KEYS in its CONDITION via a
% synthetic transient %Value. The gate deliberately does not check the condition,
% so this still compiles.
test(assoc_end_if_condition_unchanged, [condition(clang_available)]) :-
    run("{ arr[$1]++ } END { if (\"c\" in arr) print \"yes\"; else print \"no\" }\n",
        "yes\n"),
    !.

% --- END loop bodies ----------------------------------------------------
%
% The body goes through plawk_scalar_action_sequence_pairs//15 -- the same emitter
% a rule body uses -- and needed no change: the rewrite happens before the actions
% reach it, so it still sees an ordinary print of a print-expression term.

test(field_read_in_an_end_while, [condition(clang_available)]) :-
    run("{ n++ } END { while (n > 0) { print $1; n-- } }\n", "c\nc\nc\n"),
    !.

test(field_read_in_an_end_do_while, [condition(clang_available)]) :-
    run("{ n++ } END { do { print $1; n-- } while (n > 0) }\n", "c\nc\nc\n"),
    !.

test(field_read_in_an_end_c_for, [condition(clang_available)]) :-
    run("{ n++ } END { for (i = 0; i < n; i++) print $2 }\n", "3\n3\n3\n"),
    !.

test(whole_record_in_an_end_loop, [condition(clang_available)]) :-
    run("{ n++ } END { while (n > 0) { print $0; n-- } }\n", "c 3\nc 3\nc 3\n"),
    !.

% A field buried in a NESTED loop -- the case that justified the structural walk
% for the gate, and now justifies it for the rewrite.
test(field_read_in_a_nested_end_loop, [condition(clang_available)]) :-
    run("{ n++ } END { i = 0; while (i < 2) { j = 0; while (j < 2) { print $1, i, j; j++ }; i++ } }\n",
        "c 0 0\nc 0 1\nc 1 0\nc 1 1\n"),
    !.

% ...and inside an `if` inside the loop.
test(field_read_in_an_if_inside_an_end_loop, [condition(clang_available)]) :-
    run("{ n++ } END { while (n > 0) { if (n == 2) print $1; n-- } }\n", "c\n"),
    !.

test(field_concatenation_in_an_end_loop, [condition(clang_available)]) :-
    run("{ n++ } END { while (n > 0) { print $1 \"-\" $2; n-- } }\n",
        "c-3\nc-3\nc-3\n"),
    !.

test(field_read_with_break_in_an_end_loop, [condition(clang_available)]) :-
    run("{ n++ } END { while (n > 0) { n--; if (n == 1) break; print $1 } }\n",
        "c\n"),
    !.

% A loop between straight-line prints, both reading fields: the loop's blocks
% rejoin and the trailing statement still runs.
test(end_loop_between_field_prints, [condition(clang_available)]) :-
    run("{ n++ } END { print \"start\"; while (n > 0) { print $1; n-- }; print $2 }\n",
        "start\nc\nc\nc\n3\n"),
    !.

test(end_loop_field_honours_ofs, [condition(clang_available)]) :-
    run("BEGIN { OFS = \"-\" }\n{ n++ } END { while (n > 0) { print $1, $2; n-- } }\n",
        "c-3\nc-3\nc-3\n"),
    !.

% --- declines that stay declines ----------------------------------------

% A field in a loop or `if` CONDITION. The rewrite is structural, so it rewrites
% conditions too -- and no condition emitter has a clause for
% `end_lastrec_field(_)`, so these DECLINE rather than miscompiling. That
% fail-safe is the reason the rewrite is allowed to be indiscriminate.
test(field_in_an_end_if_condition_declines) :-
    build_status("{ n++ } END { if ($1 == \"c\") print \"yes\"; else print \"no\" }\n",
        3),
    !.

% `exit` inside an END loop body still fails clang if admitted, so it stays gated.
test(exit_inside_an_end_loop_still_declines) :-
    build_status("{ n++ } END { while (n > 0) { print $1; if (n == 2) exit 3; n-- } }\n",
        3),
    !.

% The ASSOCIATIVE END-if restricts its branch prints to string literals -- a
% pre-existing restriction unrelated to the record source
% (plawk_assoc_end_if_branch_prints_ok/2), so a field there declines for that
% reason, not this one. Pinned to keep the two apart.
test(assoc_end_if_branch_field_read_declines) :-
    build_status("{ arr[$1]++ } END { if (\"c\" in arr) print $1; else print \"no\" }\n",
        3),
    !.

test(the_assoc_branch_guard_is_what_refuses_it) :-
    assertion(\+ plawk_native_codegen:plawk_assoc_end_if_branch_prints_ok(
        [print([field(1)])], [print([string("no")])])),
    assertion(plawk_native_codegen:plawk_assoc_end_if_branch_prints_ok(
        [print([string("yes")])], [print([string("no")])])),
    !.

% --- NF in END ------------------------------------------------------------
%
% `NF` reads the current record, so in END it must count the RETAINED record. It
% did not: the gate matched `field(_)` only, `NF` is not a `field(_)`, and
% `END { if (n == 3) print NF }` printed 1 -- NF of the `end_of_file` sentinel --
% where gawk prints 2. Wrong output, present since the END-`if` driver existed and
% since #4100 admitted loops into END, and found only by asking what ELSE reads
% `$0`. Straight-line `END { print NF }` declined, which is why it went unnoticed.
%
% The gate is now named for the property (plawk_end_term_reads_record/1) and the
% rewrite covers `special('NF')` as well as `field(N)`.

test(end_print_nf, [condition(clang_available)]) :-
    run("{ n++ } END { print NF }\n", "2\n"),
    !.

% The two that were WRONG, not merely declined.
test(end_if_branch_nf, [condition(clang_available)]) :-
    run("{ n++ } END { if (n == 3) print NF }\n", "2\n"),
    !.

test(nf_in_an_end_loop, [condition(clang_available)]) :-
    run("{ n++ } END { while (n > 0) { print NF; n-- } }\n", "2\n2\n2\n"),
    !.

test(end_nf_with_a_field, [condition(clang_available)]) :-
    run("{ n++ } END { print NF, $1 }\n", "2 c\n"),
    !.

test(end_nf_in_a_concatenation, [condition(clang_available)]) :-
    run("{ n++ } END { print \"nf=\" NF }\n", "nf=2\n"),
    !.

test(end_nf_honours_fs, [condition(clang_available)]) :-
    run_input("BEGIN { FS = \":\" }\n{ n++ } END { print NF }\n",
        "x:y:z\np:q:r\n", "3\n"),
    !.

% Empty input: no record was read, so NF is 0 -- not the field count of whatever
% happens to be in the buffer.
test(end_nf_on_empty_input_is_zero, [condition(clang_available)]) :-
    run_input("{ n++ } END { print NF }\n", "", "0\n"),
    !.

% The gate knows NF reads the record. This is the assertion whose absence was the
% bug: it would have failed before the rename.
test(the_gate_knows_nf_reads_the_record) :-
    assertion(plawk_native_codegen:plawk_end_term_reads_record(
        [print([special('NF')])])),
    % ...and `length`, listed ahead of any END form of it being admitted, so a
    % later change cannot silently measure the EOF sentinel.
    assertion(plawk_native_codegen:plawk_end_term_reads_record(
        [print([special(length)])])),
    % NR and RT are process state that legitimately survives to END.
    assertion(\+ plawk_native_codegen:plawk_end_term_reads_record(
        [print([special('NR')])])),
    assertion(\+ plawk_native_codegen:plawk_end_term_reads_record(
        [print([special('RT')])])),
    !.

test(the_rewrite_covers_nf) :-
    plawk_native_codegen:plawk_end_lastrec_rewrite(
        [print([special('NF'), field(1), special('NR')])], Rewritten),
    assertion(Rewritten ==
        [print([end_lastrec_nf, end_lastrec_field(1), special('NR')])]),
    !.

% --- printf arguments in END ---------------------------------------------
%
% `$0`, `$N` and `NF` as printf arguments produce the SAME call-argument
% vocabulary a record-context printf produces (`string_ptr`, a
% `slice_len`/`slice_ptr` pair, `i64`), so the format rewriter and the call
% renderer needed no new cases.

test(end_printf_field_argument, [condition(clang_available)]) :-
    run("{ n++ } END { printf \"%s\\n\", $1 }\n", "c\n"),
    !.

test(end_printf_whole_record_argument, [condition(clang_available)]) :-
    run("{ n++ } END { printf \"%s\\n\", $0 }\n", "c 3\n"),
    !.

test(end_printf_nf_argument, [condition(clang_available)]) :-
    run("{ n++ } END { printf \"%d\\n\", NF }\n", "2\n"),
    !.

test(end_printf_two_field_arguments, [condition(clang_available)]) :-
    run("{ n++ } END { printf \"[%s] [%s]\\n\", $1, $2 }\n", "[c] [3]\n"),
    !.

test(end_printf_field_and_nf, [condition(clang_available)]) :-
    run("{ n++ } END { printf \"%s=%d\\n\", $1, NF }\n", "c=2\n"),
    !.

% A scalar and a field in one printf: the scalar still reads its final slot.
test(end_printf_scalar_and_field, [condition(clang_available)]) :-
    run("{ n++ } END { printf \"%d %s\\n\", n, $1 }\n", "3 c\n"),
    !.

% Two printfs, so the per-statement `end_` rename has to keep their argument
% temporaries apart.
test(two_end_printfs_with_field_arguments, [condition(clang_available)]) :-
    run("{ n++ } END { printf \"%s|\", $1; printf \"%s\\n\", $2 }\n", "c|3\n"),
    !.

test(end_printf_mixed_with_print, [condition(clang_available)]) :-
    run("{ n++ } END { printf \"%s|\", $1; print $2 }\n", "c|3\n"),
    !.

test(end_printf_field_past_nf_is_empty, [condition(clang_available)]) :-
    run("{ n++ } END { printf \"[%s]\\n\", $9 }\n", "[]\n"),
    !.

test(end_printf_field_on_empty_input, [condition(clang_available)]) :-
    run_input("{ n++ } END { printf \"[%s]\\n\", $1 }\n", "", "[]\n"),
    !.

% --- more declines that stay declines -----------------------------------

% WAS two declines. `length` in END is now admitted -- as a print field, in a
% concatenation, as a printf argument, in an `if` branch and in a loop body -- so both
% pins have been rewritten to assert the behaviour that replaced them rather than
% deleted. See tests/test_plawk_end_length.pl for the row.
%
% The ATTRIBUTION is what these two were protecting, and it held: admitting `length`
% did go through the gate. It went through it for free, in fact, because
% plawk_end_term_reads_record/1 is a STRUCTURAL walk and already admitted
% `length(field(N))` by recursing into its `field(_)` argument -- which is also why the
% `special(length)` clause asserted just above turned out not to be the clause that
% mattered. The bet those pins encoded was that a later change could not admit `length`
% without the gate noticing; the structural walk made that true by construction.
test(end_length_now_builds, [condition(clang_available)]) :-
    run("{ n++ } END { print length }\n", "3\n"),
    !.

test(end_length_in_a_loop_now_builds, [condition(clang_available)]) :-
    run("{ n++ } END { while (n > 0) { print length; n-- } }\n", "3\n3\n3\n"),
    !.

% `NF` in a CONDITION: the rewrite reaches it, no condition emitter has a clause,
% so it declines. It declined before this change too -- nothing narrows.
test(nf_in_an_end_condition_declines) :-
    build_status("{ n++ } END { if (NF == 2) print \"two\"; else print \"no\" }\n",
        3),
    !.

% Builtins over `$0`/`$N` in END are a separate follow-on: they contain a field
% term, so the gate retains the record, but the END emitters have no clause for
% them.
test(end_substr_of_the_record_declines) :-
    build_status("{ n++ } END { print substr($0, 1, 1) }\n", 3),
    !.

test(end_toupper_of_a_field_declines) :-
    build_status("{ n++ } END { print toupper($1) }\n", 3),
    !.

% An END-only program (no rules) uses a different driver, which does not retain.
test(end_only_program_field_read_declines) :-
    build_status("END { print $1 }\n", 3),
    !.

% --- regressions: END without fields is unchanged ------------------------

test(end_print_scalar_unchanged, [condition(clang_available)]) :-
    run("{ n++ } END { print n }\n", "3\n"),
    !.

test(end_print_list_unchanged, [condition(clang_available)]) :-
    run("{ n++ } END { print n; print \"x\" }\n", "3\nx\n"),
    !.

test(end_printf_unchanged, [condition(clang_available)]) :-
    run("{ n++ } END { printf \"%d\\n\", n }\n", "3\n"),
    !.

test(end_rt_unchanged, [condition(clang_available)]) :-
    run("{ n++ } END { print RT }\n", "\n\n"),
    !.

test(end_for_in_unchanged, [condition(clang_available)]) :-
    run("{ c[$1]++ } END { for (k in c) print k }\n", "a\nb\nc\n"),
    !.

test(end_loop_unchanged, [condition(clang_available)]) :-
    run("{ n++ } END { while (n > 0) { print n; n-- } }\n", "3\n2\n1\n"),
    !.

test(rule_body_field_read_unchanged, [condition(clang_available)]) :-
    run("{ print $1 }\n", "a\nb\nc\n"),
    !.

:- end_tests(plawk_end_field_reads).

% --- helpers ---------------------------------------------------------------

odir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_end_field_reads', Dir),
    ( exists_directory(Dir) -> true ; make_directory_path(Dir) ).

% Three records around 5-11 KB, so the retained buffer must realloc past its
% 4096-byte first allocation. `$2` of the last is `tail2`.
long_input(Input) :-
    findall(Line,
        ( between(0, 2, I),
          Width is 5000 + I * 3000,
          length(Codes, Width),
          maplist(=(0'x), Codes),
          atom_codes(Pad, Codes),
          format(atom(Line), "f~w_~w tail~w~n", [I, Pad, I])
        ),
        Lines),
    atomic_list_concat(Lines, Input0),
    atom_string(Input0, Input).

run(Src, Expected) :-
    input(Input),
    run_input_status(Src, Input, Expected, 0).

run_status(Src, Expected, ExpectedRC) :-
    input(Input),
    run_input_status(Src, Input, Expected, ExpectedRC).

run_input(Src, Input, Expected) :-
    run_input_status(Src, Input, Expected, 0).

run_input_status(Src, Input, Expected, ExpectedRC) :-
    odir(Dir),
    directory_file_path(Dir, 'efr_bin', Bin),
    % A build that unexpectedly declines must not silently run a stale binary.
    ( exists_file(Bin) -> delete_file(Bin) ; true ),
    directory_file_path(Dir, 'efr', Prog0),
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
    process_wait(Pid, exit(RC)),
    ( Out == Expected, RC == ExpectedRC
    -> true
    ;  format(user_error, "~n~w~n  got      ~q rc=~w~n  expected ~q rc=~w~n",
           [Src, Out, RC, Expected, ExpectedRC]), fail
    ).

% Build only, asserting the CLI status (2 = parse error, 3 = parses but outside
% the compilable surface, 4 = clang failure).
build_status(Src, ExpectedStatus) :-
    odir(Dir),
    directory_file_path(Dir, 'efr_reject', Prog0),
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
