:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% `length` and `length($N)` in END -- the byte length of the RETAINED last record.
%
% ---------------------------------------------------------------------------
% THE CELL THAT CAME FOR FREE IN FIVE ROUTES AT ONCE
%
% Every END-shaped gap this campaign closed was a missing cell in the print-field
% vocabulary, and until now each cell was paid for per route: the same kind had to be
% added to the scalar walker, the mixed walker, the assoc walker and the shared concat
% part emitter, because each of those wrote the vocabulary down separately. `length`
% cost ONE clause in the shared emitter, and appeared in the single print, in a
% concatenation, in a statement list, in the mixed route and in the assoc route
% together -- because the three walkers were first collapsed onto that emitter.
%
% What the collapse removed was a duplicated LIST, not duplicated code. The clauses
% were three steps each -- separator, one per-kind call, recurse -- and the per-kind
% call was already exactly what plawk_end_field_print_lines/4 makes for that kind.
% So "what may be printed in END" existed as four lists of clause heads with nothing
% keeping them equal, and they had already drifted twice: a string scalar in a concat
% printed its atom id for as long as both copies existed (see
% tests/test_plawk_end_concat_scalar.pl), and NF reached the routes one at a time.
% 24 clauses became 13.
%
% Underneath, the same collapse happened one level down and is the more useful half.
% `NF` had an `end_lastrec_nf` expression row that was its in-loop `nf` row with
% `%line` swapped for the retained record Value; `length` had no such row, which is
% the entire reason every END form of it declined. Both are now entries in ONE table
% of record-reading i64 leaves (plawk_record_i64_read/5) parameterised on WHICH
% record they read, with ONE retained-record wrapper (`end_lastrec_read(Kind)`) that
% works for every entry. So the next record-reading i64 leaf gets its END form for
% free, and no leaf can compute differently in END than it does in a rule body.
%
% THE COST OF ALL THIS WAS ZERO EXISTING IR. 32 golden-corpus programs are
% byte-identical across the collapse; the only changes are the `length` programs
% flipping from decline to build. That is the check that says a collapse was a
% collapse and not a rewrite.
%
% WHAT WAS *NOT* A LENGTH GAP, which is the part that saves the next reader time:
% two END forms of `length` still decline, and in BOTH cases `NF` declines
% identically, so they are shared route boundaries rather than cells of this row.
% They are pinned below as pairs with their NF form, so the attribution is checkable
% instead of asserted.
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

% Three records; the LAST is "7 disk" -- 6 bytes, $1 is "7" (1 byte), $2 is "disk"
% (4 bytes), $3 is absent, NF is 2, NR is 3, c["5"] is 2.
input("5 boot\n5 trace\n7 disk\n").

:- begin_tests(plawk_end_length).

% --- the surface forms --------------------------------------------------

% Bare `length` is awk shorthand for `length($0)`. Both spellings parse to
% `length(field(0))` and field 0 measures the whole record, so the shorthand needed no
% case of its own -- it is already the arity-1 instance of the general form.
test(bare_length_is_the_record_length, [condition(clang_available)]) :-
    run("{ n++ } END { print length }\n", "6\n"),
    !.

test(length_of_field_zero_is_the_same, [condition(clang_available)]) :-
    run("{ n++ } END { print length($0) }\n", "6\n"),
    !.

test(length_of_a_field, [condition(clang_available)]) :-
    run("{ n++ } END { print length($1) }\n", "1\n"),
    !,
    run("{ n++ } END { print length($2) }\n", "4\n"),
    !.

% A field past NF has length 0, not garbage from the buffer's tail.
test(length_of_a_field_past_nf_is_zero, [condition(clang_available)]) :-
    run("{ n++ } END { print length($3) }\n", "0\n"),
    !.

% It is the LAST record's length. The default input's first and last records are both
% 6 bytes, so a test using it cannot tell the two apart -- this one uses an input
% whose records differ in length, which is the only way the claim is checkable.
test(it_is_the_last_records_length_not_the_first, [condition(clang_available)]) :-
    run_with("aa\nbbbb\n", "{ n++ } END { print length }\n", "4\n"),
    !,
    run_with("bbbb\naa\n", "{ n++ } END { print length }\n", "2\n"),
    !.

% --- the routes, all of which got the cell from one clause --------------

test(scalar_route_beside_other_fields, [condition(clang_available)]) :-
    run("{ n++ } END { print length, n }\n", "6 3\n"),
    !,
    run("{ n++ } END { print NR, NF, length, $1, n }\n", "3 2 6 7 3\n"),
    !.

test(in_a_concatenation, [condition(clang_available)]) :-
    run("{ n++ } END { print \"L=\" length }\n", "L=6\n"),
    !,
    run("{ n++ } END { print \"a\" length \"b\" }\n", "a6b\n"),
    !.

% Two lengths in one concat: the per-part indices must not collide.
test(twice_in_one_concatenation, [condition(clang_available)]) :-
    run("{ n++ } END { print length length }\n", "66\n"),
    !.

test(in_a_statement_list, [condition(clang_available)]) :-
    run("{ n++ } END { print length; print n }\n", "6\n3\n"),
    !.

test(mixed_route_beside_an_assoc_read, [condition(clang_available)]) :-
    run("{ n++; c[$1]++ } END { print length, c[\"5\"] }\n", "6 2\n"),
    !.

test(assoc_only_route, [condition(clang_available)]) :-
    run("{ c[$1]++ } END { print length, c[\"5\"] }\n", "6 2\n"),
    !.

% --- printf ------------------------------------------------------------

test(as_a_printf_argument, [condition(clang_available)]) :-
    run("{ n++ } END { printf \"%d\\n\", length }\n", "6\n"),
    !,
    run("{ n++ } END { printf \"len=%d\\n\", length($2) }\n", "len=4\n"),
    !.

% Beside NF, so the two retained-record i64 leaves coexist in one printf without
% their generated temporaries colliding.
test(printf_with_length_and_nf_together, [condition(clang_available)]) :-
    run("{ n++ } END { printf \"%d %d\\n\", length, NF }\n", "6 2\n"),
    !.

test(as_a_printf_argument_in_a_statement_list, [condition(clang_available)]) :-
    run("{ n++; c[$1]++ } END { print c[\"5\"]; printf \"%d\\n\", length }\n",
        "2\n6\n"),
    !,
    run("{ c[$1]++ } END { print c[\"5\"]; printf \"%d\\n\", length }\n", "2\n6\n"),
    !.

% --- the END `if` branch -----------------------------------------------

test(in_an_if_branch, [condition(clang_available)]) :-
    run("{ n++ } END { if (n == 3) print length }\n", "6\n"),
    !.

test(in_an_if_branch_beside_nf, [condition(clang_available)]) :-
    run("{ n++ } END { if (n == 3) print length, NF }\n", "6 2\n"),
    !.

% Both branches, and the branch NOT taken must not evaluate it.
test(in_an_if_else, [condition(clang_available)]) :-
    run("{ n++ } END { if (n == 3) print length; else print \"no\" }\n", "6\n"),
    !,
    run("{ n++ } END { if (n == 9) print length; else print \"no\" }\n", "no\n"),
    !.

% --- the END loop body -------------------------------------------------
%
% A loop body in END reaches the shared sequence emitter, whose print vocabulary is a
% THIRD list (plawk_rule_body_print_field/1) -- it had a row for `end_lastrec_nf` and
% none for the length leaf, so this declined while the identical NF form worked. One row.
test(in_an_end_loop_body, [condition(clang_available)]) :-
    run("{ n++ } END { while (n > 0) { print length; n-- } }\n", "6\n6\n6\n"),
    !,
    run("{ n++ } END { while (n > 0) { print length($2); n-- } }\n", "4\n4\n4\n"),
    !.

test(nf_in_an_end_loop_body_unchanged, [condition(clang_available)]) :-
    run("{ n++ } END { while (n > 0) { print NF; n-- } }\n", "2\n2\n2\n"),
    !.

% --- the known defect this suite pinned has now been FIXED ------------
%
% These two were written to pin the CURRENT WRONG behaviour of `length` as a COMPARISON
% operand -- deliberately red-in-waiting, so the follow-up would have to flip them
% rather than quietly add to a green suite. It did, so they are rewritten to assert what
% replaced them (never deleted, per the campaign's rule for a pin whose cause is gone).
%
% What was wrong: `{ if (length > 3) print $1 }` printed nothing where gawk prints three
% records, and `END { if (length == 6) … }` took the else branch for a 6-byte last
% record -- while the bare-pattern spelling `length > 3 { print $1 }` was correct all
% along. The condition grammar captured bare `length` as an ordinary variable named
% `length`, worth 0.
%
% The handoff worked as intended and is worth noting as a practice: a pin that states a
% known defect AS the current behaviour, with its cause and its sizing in the comment,
% costs one test and makes the follow-up impossible to forget. The alternative -- a note
% in a doc -- does not fail when the code changes.
%
% tests/test_plawk_cond_specials.pl owns the row now (it turned out to be four lists of
% one set, not two, and every special on the RIGHT of a condition was wrong the same
% way, not just `length`).
test(length_as_a_comparison_operand_now_agrees_with_the_bare_pattern,
        [condition(clang_available)]) :-
    run("length > 3 { print $1 }\n", "5\n5\n7\n"),
    !,
    run("{ if (length > 3) print $1 }\n", "5\n5\n7\n"),
    !.

test(the_condition_grammar_now_parses_bare_length_as_the_special) :-
    plawk_parse_string("{ n++ } END { if (length == 6) print \"six\" }\n", Program),
    Program = program(_, _, [end([if(scalar_if(Cond), _, _)])]),
    % WAS cmp(var(length), eq, int(6)) -- the defect, stated at the level it happened.
    assertion(Cond == cmp(special(length), eq, int(6))),
    !.

% ...and in END that special now DECLINES rather than measuring the EOF sentinel, which
% is where `NF` already was. The END print of `length` (every test above) is unaffected:
% it is the CONDITION that cannot reach the retained record, not the branch.
test(length_in_an_end_condition_declines_like_nf) :-
    build_status("{ n++ } END { if (length == 6) print \"six\"; else print \"no\" }\n", 3),
    build_status("{ n++ } END { if (NF == 2) print \"two\"; else print \"no\" }\n", 3),
    !.

% --- the in-loop form is untouched -------------------------------------
%
% `length` in a rule body reads the CURRENT record and always worked. It now goes
% through the same plawk_record_i64_read/5 entry as the END form, so these pin that
% the shared entry did not change what the in-loop form computes.
% The three records are "5 boot" (6), "5 trace" (7), "7 disk" (6) -- the middle one is
% a byte longer, which is what makes this a per-record read rather than a constant.
test(in_loop_length_unchanged, [condition(clang_available)]) :-
    run("{ print length }\n", "6\n7\n6\n"),
    !,
    run("{ print length($1) }\n", "1\n1\n1\n"),
    !.

test(nf_in_end_unchanged, [condition(clang_available)]) :-
    run("{ n++ } END { print NF }\n", "2\n"),
    !,
    run("{ n++ } END { printf \"%d\\n\", NF }\n", "2\n"),
    !,
    run("{ n++ } END { if (n == 3) print NF }\n", "2\n"),
    !.

% --- the retain machinery, on the construct's own names -----------------

% `length` in END reads the retained record, so it must carry the retain buffer AND
% its store. A projection emitted without the store would measure nothing; a store
% called without a definition is the exit-4 clang failure the NF change hit. Both
% halves asserted, as in tests/test_plawk_mixed_end_nf.pl.
test(the_retain_buffer_and_its_store_appear_together) :-
    build_ll("{ n++ } END { print length }\n", LL),
    assertion(sub_string(LL, _, _, _, "@plawk_lastrec_buf")),
    assertion(sub_string(LL, _, _, _, "define void @plawk_lastrec_store")),
    assertion(sub_string(LL, _, _, _, "call void @plawk_lastrec_store")),
    !.

% Pay-per-use is unchanged: a program whose END reads no record carries no buffer.
% `length` had to join plawk_end_term_reads_record/1 for this to be true, and it did
% so for free -- that gate is a STRUCTURAL walk, so it already admitted
% `length(field(N))` by recursing into the `field(_)` argument.
test(no_retain_buffer_when_nothing_reads_the_record) :-
    build_ll("{ n++ } END { print n }\n", LL),
    assertion(\+ sub_string(LL, _, _, _, "@plawk_lastrec_buf")),
    !.

% The length projection must measure the RETAINED record, not `%line` -- which at END
% is the EOF sentinel. Checked on the emitted call: the field-length runtime call takes
% the retained transient's Value, never `%line`.
test(the_length_call_reads_the_retained_record_not_the_line) :-
    build_ll("{ n++ } END { print length }\n", LL),
    length_calls(LL, Calls),
    assertion(Calls \== []),
    assertion(forall(member(C, Calls),
        ( sub_string(C, _, _, _, "@plawk_lastrec_transient")
        ; sub_string(C, _, _, _, "%end_len_")
        ))),
    assertion(forall(member(C, Calls), \+ sub_string(C, _, _, _, "%line"))),
    !.

% --- one table, one computation ----------------------------------------
%
% The property the generalisation buys: the in-loop and END forms of `length` differ
% ONLY in which record Value they are handed. Compared on the runtime call itself,
% with the record operand removed -- if the two ever computed length differently, the
% remaining operands (field index and separator) would differ.
test(in_loop_and_end_length_differ_only_in_the_record) :-
    length_call_tail("{ print length($2) }\n", InLoop),
    length_call_tail("{ n++ } END { print length($2) }\n", InEnd),
    assertion(InLoop \== ""),
    assertion(InLoop == InEnd),
    !.

% --- boundaries that are NOT this row, each paired with its NF form -----

% Arithmetic over a record read in END declines, and `NF + 1` declines identically:
% plawk_end_scalar_expr/1's operand surface is slots / NR / literals, and no record
% read is in it. A shared boundary, so `length + 1` is not a cell of this row.
test(arithmetic_over_a_record_read_declines_for_length_and_nf_alike) :-
    build_status("{ n++ } END { print length + 1 }\n", 3),
    build_status("{ n++ } END { print NF + 1 }\n", 3),
    !.

% A for-in chain body reads no record at all -- `NF` declines there too, and did
% before `length` existed in END. The chain driver carries the EndRecord token (that
% landed separately), so what remains is the chain body's print vocabulary, which is
% its own follow-on rather than anything to do with `length`.
test(a_for_in_chain_body_declines_for_length_and_nf_alike) :-
    build_status("{ c[$1]++ } END { for (k in c) print k, length }\n", 3),
    build_status("{ c[$1]++ } END { for (k in c) print k, NF }\n", 3),
    !,
    % ...and the same chain without a record read still works, so the decline is
    % about the record read and not about the chain.
    run_sorted("{ c[$1]++ } END { for (k in c) print k }\n", "5\n7\n"),
    !.

% --- a bonus cell from the collapse, pinned because it was not the goal --
%
% Delegating the ASSOC walker to the shared emitter passes the EMPTY scalar plan --
% which that walker already did for concat parts -- and that makes the shared generic
% scalar-expression clause reachable, so a CONSTANT expression now prints in the
% assoc route. It was a decline before and it agrees with gawk. Pinned here rather
% than left as an unremarked side effect: an unintended behaviour change that nobody
% wrote down is indistinguishable later from a defect.
test(a_constant_expression_now_prints_in_the_assoc_route,
        [condition(clang_available)]) :-
    run("{ c[$1]++ } END { print 1 + 2, c[\"5\"] }\n", "3 2\n"),
    !.

% A `var` in the assoc route still resolves to nothing -- but note it does not
% DECLINE: a scalar named in an END print makes the program MIXED, so it is routed to
% the walker that has slots and prints unset as empty. Pinned so the empty output is
% not later read as the assoc route silently accepting a var it has no slot for.
test(a_var_in_an_assoc_end_print_routes_to_the_mixed_walker,
        [condition(clang_available)]) :-
    run("{ c[$1]++ } END { print n, c[\"5\"] }\n", " 2\n"),
    !.

:- end_tests(plawk_end_length).

% --- helpers ---------------------------------------------------------------

odir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_end_length', Dir),
    ( exists_directory(Dir) -> true ; make_directory_path(Dir) ).

run(Src, Expected) :-
    input(Input),
    run_with(Input, Src, Expected).

run_with(Input, Src, Expected) :-
    run_raw(Input, Src, Out),
    (   Out == Expected
    ->  true
    ;   format(user_error, "~n~w~n  got      ~q~n  expected ~q~n",
            [Src, Out, Expected]), fail
    ).

% for-in iteration order is unspecified in awk, so compare sorted lines.
run_sorted(Src, Expected) :-
    input(Input),
    run_raw(Input, Src, Out),
    sorted_lines(Out, Got),
    sorted_lines(Expected, Want),
    (   Got == Want
    ->  true
    ;   format(user_error, "~n~w~n  got      ~q~n  expected ~q~n",
            [Src, Got, Want]), fail
    ).

sorted_lines(Text, Sorted) :-
    split_string(Text, "\n", "", Parts0),
    exclude(==(""), Parts0, Parts),
    msort(Parts, Sorted).

run_raw(Input, Src, Out) :-
    odir(Dir),
    directory_file_path(Dir, 'len_bin', Bin),
    % A build that unexpectedly declines must not silently run a stale binary.
    ( exists_file(Bin) -> delete_file(Bin) ; true ),
    directory_file_path(Dir, 'len', Prog0),
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

build_status(Src, ExpectedStatus) :-
    odir(Dir),
    directory_file_path(Dir, 'len_reject', Prog0),
    atom_concat(Prog0, '.plawk', Prog),
    setup_call_cleanup(open(Prog, write, S, [encoding(utf8)]),
        write(S, Src), close(S)),
    atom_concat(Prog0, '_bin', Bin),
    cli([build, Prog, '-o', Bin], ExpectedStatus).

build_ll(Src, LL) :-
    plawk_parse_string(Src, Program),
    plawk_program_native_driver_ir(Program, 'input.txt', IR),
    atom_string(IR, LL).

% The field-length runtime calls a program emits. `@wam_atom_field_length_value` is
% the one runtime entry point both the in-loop and END forms reach (through
% plawk_record_i64_read/5), so these are the instructions that carry the computation.
length_calls(LL, Calls) :-
    split_string(LL, "\n", " ", All),
    include(length_call, All, Calls).

length_call(Line) :-
    sub_string(Line, _, _, _, "@wam_atom_field_length_value"),
    !.

% The field-length call with its RECORD operand dropped, leaving the field index and
% separator. The in-loop and END forms must agree on this remainder: that is the
% property of computing length in one place rather than two.
length_call_tail(Src, Tail) :-
    build_ll(Src, LL),
    length_calls(LL, [Call | _]),
    sub_string(Call, Before, _, _, "%Value "),
    Start is Before + 7,
    sub_string(Call, Start, _, 0, Rest),
    % skip the record operand, keep everything from the following comma
    sub_string(Rest, Comma, _, _, ","),
    !,
    sub_string(Rest, Comma, _, 0, Tail).
length_call_tail(_Src, "").

cli(Args, ExpectedStatus) :-
    process_create(path(swipl), ['examples/plawk/bin/plawk' | Args],
        [stdout(pipe(S)), stderr(std), process(Pid)]),
    read_string(S, _, _), close(S),
    process_wait(Pid, exit(Status)),
    assertion(Status == ExpectedStatus).
