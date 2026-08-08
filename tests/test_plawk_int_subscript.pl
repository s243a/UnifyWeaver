:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% An INTEGER-literal subscript, end to end: `c[5]++`, `c[5] += N`, `c[5]--`,
% `print c[5]` and `END { print c[5] }`.
%
% awk array subscripts are STRINGS, so `c[5]` is the key "5" -- the same key `c["5"]` and a
% field holding "5" produce. That single fact is all the semantics here; everything else is
% about why it took three attempts to be allowed to say it.
%
% ---------------------------------------------------------------------------
% A REFUSAL THAT OUTLIVED ITS CAUSE, TWICE
%
% `c[5]++` has been refused three times for three different reasons, and only the first was
% ever the real one:
%
%   1. THE KEY-SPACE COLLISION. `print arr[N]` read the RAW integer N as the key id
%      regardless of the table's kind, so admitting the update meant
%      `{ c[5]++; print c[5] }` stored to interned "5" and loaded from raw 5 -- a program
%      that BUILT and printed empty. A genuine reason: the surface could not be given a
%      consistent meaning. (tests/test_plawk_literal_assoc_key.pl records the refusal.)
%
%   2. HALF-RESOLVED. With the reads fixed (tests/test_plawk_posarray_keyspace.pl),
%      `END { print arr[N] }` still declined on an awk-semantics table -- a separate,
%      deliberately documented refusal at that emitter, carrying its own copy of the
%      collision reasoning. Admitting the update then would have shipped `c[5]++` compiling
%      while the natural `{ c[5]++ } END { print c[5] }` declined. Not a reason to refuse
%      the feature; a reason to finish it.
%
%   3. Nothing. Both halves resolved, so the refusal is retired.
%
% The general shape, worth carrying: a refusal is a claim about the code at a moment in
% time, and the code moves. When the thing a refusal cited gets fixed elsewhere, nobody
% re-reads the refusal -- it just keeps being cited. Each of the notes above was rewritten
% rather than left standing, precisely so the next reader can tell which reason is still
% load-bearing.
%
% ---------------------------------------------------------------------------
% THE FIFTH SITE, AND WHY A DECLINING GATE HID LONGEST
%
% Lifting (2) meant relaxing plawk_assoc_record_program_ok/3, a SURFACE gate holding a
% private copy of the same two-key-space reasoning -- "integer assoc keys would collide with
% atom ids, so they stay binary-only -- EXCEPT positional-array tables" -- the same sentence
% the END emitter carried. It was the fifth site with that reasoning, after the rule-body
% read, the delete, the membership probe and the END emitter.
%
% The other four each produced a wrong output and were fixed together. This one produced a
% DECLINE, which is exactly why it outlived them: a gate that refuses reads as caution, so
% nobody asks whether its premise is still true. **A conservative gate is a duplicated
% decision wearing a disguise.**
%
% The lift itself is a term REWRITE, not a new emitter: plawk_assoc_end_int_key_rewrite/4
% turns `arr[5]` into `arr["5"]` before either END walker sees it. That matters because the
% key-globals walker declares the c-string constant the print emitter references by index,
% and it takes only the field list -- no plan, no descriptor -- so it cannot tell an
% awk-semantics table from a positional one. A clause on the emitter alone would reference a
% global nobody emitted; a clause on both would put the key-space decision in a walker with
% no way to make it. Rewriting upstream leaves both walkers untouched.
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

% $1 is 5, 5, 7 -- so a field-derived key collides with a plausible positional slot, which
% is the whole point of the fixture.
input("5 boot\n5 trace\n7 disk\n").

:- begin_tests(plawk_int_subscript).

% --- the update -----------------------------------------------------------

test(int_key_counter, [condition(clang_available)]) :-
    run("{ c[5]++ } END { print c[5] }\n", "3\n"),
    !.

test(int_key_add_assign, [condition(clang_available)]) :-
    run("{ c[5] += 2 } END { print c[5] }\n", "6\n"),
    !.

test(int_key_decrement, [condition(clang_available)]) :-
    run("{ c[5]-- } END { print c[5] }\n", "-3\n"),
    !.

test(int_key_subtract_assign, [condition(clang_available)]) :-
    run("{ c[5] -= 2 } END { print c[5] }\n", "-6\n"),
    !.

% --- the END read --------------------------------------------------------
%
% This is the half that declined even after the reads were resolved.

test(int_key_end_read, [condition(clang_available)]) :-
    run("{ c[$1]++ } END { print c[5] }\n", "2\n"),
    !.

test(int_key_end_read_other_key, [condition(clang_available)]) :-
    run("{ c[$1]++ } END { print c[7] }\n", "1\n"),
    !.

% An absent int key is the empty an absent element gives, not a 0 and not a
% coincidental atom-id hit.
test(an_absent_int_key_reads_empty, [condition(clang_available)]) :-
    run("{ c[$1]++ } END { print c[9] }\n", "\n"),
    !.

% --- the rule-body read --------------------------------------------------

test(int_key_rule_body_read, [condition(clang_available)]) :-
    run("{ c[5]++; print c[5] }\n", "1\n2\n3\n"),
    !.

% --- `c[5]` and `c["5"]` are ONE key ------------------------------------
%
% The load-bearing property. Three ways to arrive at the key "5" -- an int literal, a string
% literal, and a field holding "5" -- must all land in the same entry. Two records have
% $1 = 5 and the literal fires on all three, so a shared key gives 2 + 3 = 5; separate key
% spaces would give 3.

test(the_int_and_string_spellings_are_the_same_key, [condition(clang_available)]) :-
    run("{ c[5]++ } END { print c[\"5\"] }\n", "3\n"),
    !,
    run("{ c[\"5\"]++ } END { print c[5] }\n", "3\n"),
    !.

test(an_int_literal_key_and_a_field_holding_that_text_are_the_same_key,
        [condition(clang_available)]) :-
    run("{ c[$1]++; c[5]++ } END { print c[5] }\n", "5\n"),
    !,
    % ...and the for-in view shows TWO entries ("5" and "7"), not three. The sum could be
    % explained away; an extra entry could not.
    run_sorted("{ c[$1]++; c[5]++ } END { for (k in c) print k, c[k] }\n",
        "5 5\n7 1\n"),
    !.

% --- a positional table still refuses the UPDATE ------------------------
%
% Not the retired refusal: a positional table's keys are integer positions and this is an
% interning update, so the program is unrepresentable rather than merely unresolved. That
% kind of refusal does not expire.
test(an_int_key_update_on_a_positional_table_declines) :-
    build_status("{ split($0, a, \" \"); a[5]++; print a[1] }\n", 3),
    !,
    build_status("{ split($0, a, \" \"); a[5] += 2; print a[1] }\n", 3),
    !.

% ...while READING a positional table by an int subscript is the raw position, unchanged.
test(a_positional_int_read_is_still_the_raw_position, [condition(clang_available)]) :-
    run("{ split($0, a, \" \"); print a[1] }\n", "5\n5\n7\n"),
    !,
    run("{ split($0, a, \" \") } END { print a[1] }\n", "7\n"),
    !.

% --- the rewrite, at the IR level ---------------------------------------
%
% `END { print c[5] }` must reach the interned-key path, i.e. emit the key constant and
% intern it -- not pass a bare `i64 5` to the getter, which is what it did before.
test(an_end_int_read_interns_its_decimal_spelling) :-
    build_ll("{ c[$1]++ } END { print c[5] }\n", LL),
    assertion(sub_string(LL, _, _, _, "@.plawk_assoc_print_key_0")),
    assertion(sub_string(LL, _, _, _, "@wam_intern_atom")),
    !.

% ...and it emits exactly what the string spelling emits, which is the check that the
% rewrite happens upstream of BOTH END walkers rather than in one of them.
test(the_int_and_string_end_reads_emit_identical_ir) :-
    build_ll("{ c[$1]++ } END { print c[5] }\n", IntIR),
    build_ll("{ c[$1]++ } END { print c[\"5\"] }\n", StrIR),
    assertion(IntIR == StrIR),
    !.

% A positional END int read must NOT be rewritten -- it keeps the raw-position reading, so
% no key constant is emitted for it.
test(a_positional_end_int_read_is_not_rewritten) :-
    build_ll("{ split($0, a, \" \") } END { print a[1] }\n", LL),
    assertion(\+ sub_string(LL, _, _, _, "@.plawk_assoc_print_key_0")),
    !.

:- end_tests(plawk_int_subscript).

% --- helpers ---------------------------------------------------------------

odir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_int_subscript', Dir),
    ( exists_directory(Dir) -> true ; make_directory_path(Dir) ).

run(Src, Expected) :-
    run_(Src, Expected, plain).

run_sorted(Src, Expected) :-
    run_(Src, Expected, sorted).

run_(Src, Expected, Mode) :-
    odir(Dir),
    input(Input),
    directory_file_path(Dir, 'is_bin', Bin),
    % A build that unexpectedly declines must not silently run a stale binary.
    ( exists_file(Bin) -> delete_file(Bin) ; true ),
    directory_file_path(Dir, 'is', Prog0),
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

% awk's for-in order is unspecified, so for-in output is compared as a multiset of lines.
sort_lines(In, Out) :-
    split_string(In, "\n", "", Parts0),
    ( append(Parts, [""], Parts0) -> true ; Parts = Parts0 ),
    msort(Parts, Sorted),
    atomic_list_concat(Sorted, '\n', Joined),
    ( Sorted == [] -> Out = "" ; format(string(Out), "~w\n", [Joined]) ).

% Build only, asserting the CLI status (2 = parse error, 3 = parses but outside the
% compilable surface).
build_status(Src, ExpectedStatus) :-
    odir(Dir),
    directory_file_path(Dir, 'is_reject', Prog0),
    atom_concat(Prog0, '.plawk', Prog),
    setup_call_cleanup(open(Prog, write, S, [encoding(utf8)]),
        write(S, Src), close(S)),
    atom_concat(Prog0, '_bin', Bin),
    cli([build, Prog, '-o', Bin], ExpectedStatus).

% The emitted LLVM IR, straight from the code generator -- no clang needed.
build_ll(Src, LL) :-
    plawk_parse_string(Src, Program),
    plawk_program_native_driver_ir(Program, 'input.txt', IR),
    atom_string(IR, LL).

cli(Args, ExpectedStatus) :-
    process_create(path(swipl), ['examples/plawk/bin/plawk' | Args],
        [stdout(pipe(S)), stderr(std), process(Pid)]),
    read_string(S, _, _), close(S),
    process_wait(Pid, exit(Status)),
    assertion(Status == ExpectedStatus).
