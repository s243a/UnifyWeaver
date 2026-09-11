:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% TWO KEY SPACES, ONE SURFACE SYNTAX -- and the one rule that resolves them.
%
% plawk has two array conventions, and `arr[SUBSCRIPT]` spells both:
%
%   awk semantics      keys are STRINGS. `c[5]` is the key "5", interned to an atom id.
%                      Produced by `c[$1]++`, `c["x"]++`, row captures, ...
%   positional tables  keys are the RAW INTEGER position 1..n. Produced by `split()`
%                      and `as array` / posarray binds.
%
% Which space a LITERAL subscript means is decided by the TABLE'S KIND, not by how the
% subscript is written. Nothing at the surface or spec level knows the kind; PosArrays is
% a plan-time set. Every site that resolved a literal key was therefore deciding on its
% own, and they did not agree:
%
%   site                     did                        on the wrong table kind
%   -----------------------------------------------------------------------------------
%   lookup_int (read)        raw integer, always        awk-semantics: WRONG OUTPUT
%   assoc_delete_lit         interned text, always      positional:    WRONG OUTPUT
%   membership string key    interned text, always      positional:    WRONG OUTPUT
%   END read, string key     interned text, always      positional:    WRONG OUTPUT
%
% Four sites, and every one of them a SILENT WRONG OUTPUT rather than a decline:
%
%   { c[$1]++; print c[5] }                         empty   gawk: 1/2/2
%   { split($0,a," "); delete a[1]; print a[1] }    5/5/7   gawk: (empty)
%   { split($0,a," "); if ("1" in a) print "y" }    (none)  gawk: y/y/y
%   { split($0,a," ") } END { print a["1"] }        empty   gawk: 7
%
% That is the signature of this defect variant, and worth stating plainly: when two sites
% disagree about a VALUE that both consider valid -- an atom id and a position are both
% just i64 -- nothing declines and nothing crashes. Compare the gate-style variant, where a
% missing row makes a program refuse and announce itself. The END read's own INT clause had
% already worked this out for itself ("a text-mode literal key would collide with an atom
% id -- EXCEPT a positional-array table"), which is the tell that the property wanted one
% home and had none.
%
% All four now go through plawk_assoc_literal_key_space/4, which answers once:
% positional + canonical decimal >= 1 -> that raw position; positional + anything else ->
% raw 0, which is NO position and therefore the absent element awk sees; otherwise the
% interned text.
%
% ---------------------------------------------------------------------------
% WHY raw 0 AND NOT "intern it anyway"
%
% For `a["x"]` on a split table the tempting shortcut is to intern "x" and probe with the
% result. That is not merely useless, it is DANGEROUS: atom ids are small sequential
% integers, so the id of "x" can COINCIDE with a live position and report a hit on an
% element that has nothing to do with the key. Raw 0 is chosen because positional keys
% start at 1, so it is guaranteed absent -- a correct answer rather than a lucky one.
%
% ---------------------------------------------------------------------------
% WHY THE DECIMAL TEST IS A ROUND TRIP
%
% `a["01"]` must NOT alias slot 1: awk's split keys are "1".."n", so "01" is a different,
% absent key. plawk_canonical_decimal_key/2 therefore requires Text to be byte-identical
% to the spelling of its own value, which number_string/2 alone does not -- it happily
% reads "01" as 1. Pinned below for "01", "1.0", "0" and "-1".
%
% ---------------------------------------------------------------------------
% THE UPDATE SIDE DECLINES, IT DOES NOT GET A KEY SPACE
%
% A lone-literal-keyed UPDATE (`a["x"]++`) on a positional table is not a resolution
% problem: that table's keys are integers and it cannot hold "x" at all. So it declines,
% and this suite pins that -- it was briefly a wrong output when the string-literal key
% landed before this rule existed (the inc interned "x", the read resolved raw 0, and the
% program printed empty where gawk prints 1).
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

% $1 is 5, 5, 7 -- numeric-looking, so it exercises the collision: as an awk key it is
% the text "5", and 5 is also a plausible positional slot.
input("5 boot\n5 trace\n7 disk\n").

:- begin_tests(plawk_posarray_keyspace).

% --- the four wrong outputs, each fixed -----------------------------------

% WAS: empty lines. `print c[N]` read the RAW integer N as the key id on an
% awk-semantics table, i.e. an atom-registry position, not the key "N".
test(an_int_subscript_read_on_an_awk_table_is_the_interned_text,
        [condition(clang_available)]) :-
    run("{ c[$1]++; print c[5] }\n", "1\n2\n2\n"),
    !,
    % ...and it agrees with the string spelling, which is the point: awk subscripts are
    % strings, so `c[5]` and `c["5"]` are one key.
    run("{ c[$1]++; print c[\"5\"] }\n", "1\n2\n2\n"),
    !.

% WAS: 5/5/7 -- the delete interned "1" and missed raw key 1, so the element survived.
test(a_delete_on_a_positional_table_uses_the_raw_position,
        [condition(clang_available)]) :-
    run("{ split($0, a, \" \"); delete a[1]; print a[1] }\n", "\n\n\n"),
    !,
    % Both spellings, because on a positional table awk's key "1" and the integer 1 are
    % the same slot.
    run("{ split($0, a, \" \"); delete a[\"1\"]; print a[1] }\n", "\n\n\n"),
    !.

% WAS: nothing printed -- the probe interned "1" and compared an atom id against a
% raw-int-keyed table.
test(membership_with_a_string_key_on_a_positional_table,
        [condition(clang_available)]) :-
    run("{ split($0, a, \" \"); if (\"1\" in a) print \"y\" }\n", "y\ny\ny\n"),
    !,
    run("{ split($0, a, \" \"); if (1 in a) print \"y\" }\n", "y\ny\ny\n"),
    !.

% WAS: an empty line. The END read's STRING clause interned "1" and missed raw key 1,
% while its INT clause had already got this right -- the fourth site, and the one whose
% own comment names the collision ("a text-mode literal key would collide with an atom
% id -- EXCEPT a positional-array table"). Both spellings must now agree.
test(an_end_read_on_a_positional_table_uses_the_raw_position,
        [condition(clang_available)]) :-
    run("{ split($0, a, \" \") } END { print a[\"1\"] }\n", "7\n"),
    !,
    run("{ split($0, a, \" \") } END { print a[1] }\n", "7\n"),
    !,
    run("{ split($0, a, \" \") } END { print a[\"2\"], a[2] }\n", "disk disk\n"),
    !.

% ...and a non-decimal END key on a positional table is the absent element, not a
% coincidental atom-id hit.
test(an_end_read_of_a_non_decimal_key_on_a_positional_table_is_absent,
        [condition(clang_available)]) :-
    run("{ split($0, a, \" \") } END { print a[\"x\"] }\n", "\n"),
    !.

% The END read on an awk-semantics table is unchanged.
test(an_end_read_on_an_awk_table_still_interns, [condition(clang_available)]) :-
    run("{ c[$1]++ } END { print c[\"5\"] }\n", "2\n"),
    !.

% WAS a decline, pinned as "the one piece the plan-time rule has not been wired into yet".
% It is wired in now: the END read rewrites `arr[5]` to the key "5" before either END walker
% sees it, so an int subscript reads the same key the string spelling does. Kept, inverted,
% and PAIRED with the string spelling -- awk subscripts are strings, so the two are one key
% and must never drift apart again.
test(an_int_subscript_end_read_on_an_awk_table_is_the_interned_text,
        [condition(clang_available)]) :-
    run("{ c[$1]++ } END { print c[5] }\n", "2\n"),
    !,
    run("{ c[$1]++ } END { print c[\"5\"] }\n", "2\n"),
    !.

% --- the read, both spellings on both table kinds ------------------------

test(positional_read_int_subscript, [condition(clang_available)]) :-
    run("{ split($0, a, \" \"); print a[1] }\n", "5\n5\n7\n"),
    !.

test(positional_read_string_subscript, [condition(clang_available)]) :-
    run("{ split($0, a, \" \"); print a[\"1\"] }\n", "5\n5\n7\n"),
    !.

test(positional_read_second_slot, [condition(clang_available)]) :-
    run("{ split($0, a, \" \"); print a[2], a[\"2\"] }\n",
        "boot boot\ntrace trace\ndisk disk\n"),
    !.

% A subscript past the end, and a non-decimal subscript, are both absent elements --
% empty, not a coincidental hit on some live slot.
test(positional_read_out_of_range_is_absent, [condition(clang_available)]) :-
    run("{ split($0, a, \" \"); print a[9] }\n", "\n\n\n"),
    !.

test(positional_read_non_decimal_key_is_absent, [condition(clang_available)]) :-
    run("{ split($0, a, \" \"); print a[\"x\"] }\n", "\n\n\n"),
    !.

test(awk_table_read_literal_key, [condition(clang_available)]) :-
    run("{ c[\"x\"]++; print c[\"x\"] }\n", "1\n2\n3\n"),
    !.

% --- canonicality: a decimal-looking key that is not canonical ----------
%
% Each of these must stay its own (absent) key rather than aliasing a slot. This is what
% the round-trip check in plawk_canonical_decimal_key/2 buys; number_string/2 alone reads
% "01" as 1 and would have hit slot 1.

test(leading_zero_is_not_slot_one, [condition(clang_available)]) :-
    run("{ split($0, a, \" \"); print a[\"01\"] }\n", "\n\n\n"),
    !.

test(a_float_spelling_is_not_slot_one, [condition(clang_available)]) :-
    run("{ split($0, a, \" \"); print a[\"1.0\"] }\n", "\n\n\n"),
    !.

test(zero_and_negative_are_not_slots, [condition(clang_available)]) :-
    run("{ split($0, a, \" \"); print a[\"0\"] }\n", "\n\n\n"),
    !,
    run("{ split($0, a, \" \"); print a[\"-1\"] }\n", "\n\n\n"),
    !.

% A non-canonical delete must not remove slot 1 either.
test(a_leading_zero_delete_leaves_slot_one_alone, [condition(clang_available)]) :-
    run("{ split($0, a, \" \"); delete a[\"01\"]; print a[1] }\n", "5\n5\n7\n"),
    !.

test(a_leading_zero_membership_is_false, [condition(clang_available)]) :-
    run("{ split($0, a, \" \"); if (\"01\" in a) print \"y\"; print a[1] }\n",
        "5\n5\n7\n"),
    !.

% ...and on an AWK-semantics table "01" is simply a key, distinct from "1".
test(a_leading_zero_is_its_own_key_on_an_awk_table, [condition(clang_available)]) :-
    run("{ c[\"01\"]++ } END { print c[\"01\"] }\n", "3\n"),
    !,
    run("{ c[\"01\"]++ } END { print c[\"1\"] }\n", "\n"),
    !.

% --- delete and membership on an awk-semantics table are unchanged -------

test(awk_table_delete_both_spellings, [condition(clang_available)]) :-
    run("{ c[$1]++; delete c[5] } END { print c[\"5\"] }\n", "\n"),
    !,
    run("{ c[$1]++; delete c[\"5\"] } END { print c[\"5\"] }\n", "\n"),
    !.

test(awk_table_membership, [condition(clang_available)]) :-
    run("{ c[$1]++; if (\"5\" in c) print \"y\" }\n", "y\ny\ny\n"),
    !,
    run("{ c[$1]++; if (\"zz\" in c) print \"y\"; print \"end\" }\n",
        "end\nend\nend\n"),
    !.

% --- the update side declines on a positional table ---------------------
%
% Not a key-space choice: a positional table's keys are integers, so it cannot hold "x"
% at all. Pinned because it was briefly a WRONG OUTPUT -- when the string-literal key
% update landed before this rule existed, the inc interned "x" while the read resolved
% raw 0, and the program printed empty where gawk prints 1.
test(a_literal_key_update_on_a_positional_table_declines) :-
    build_status("{ split($0, a, \" \"); a[\"x\"]++; print a[\"x\"] }\n", 3),
    build_status("{ split($0, a, \" \"); a[\"x\"]++ } END { print a[\"x\"] }\n", 3),
    !.

% Even the representable spelling declines -- incrementing a split slot would need a
% raw-keyed increment, and it is not an idiom worth one. Pinned so that if it is ever
% added, it is added deliberately.
test(a_decimal_literal_key_update_on_a_positional_table_also_declines) :-
    build_status("{ split($0, a, \" \"); a[\"1\"]++; print a[1] }\n", 3),
    !.

% Only the LONE-LITERAL update shape is gated. A key with a field component is untouched
% and behaves exactly as it did before the gate existed.
test(a_field_keyed_update_on_a_positional_table_is_untouched,
        [condition(clang_available)]) :-
    run("{ split($0, a, \" \"); a[$1]++; print a[1] }\n", "5\n5\n7\n"),
    !,
    run("{ split($0, a, \" \"); a[$1,$2]++; print a[1] }\n", "5\n5\n7\n"),
    !.

% --- two adjacent assoc END fields keep their OFS -----------------------
%
% Adding the positional clause meant rewriting the string clause's head, and the
% `plawk_scalar_end_separator_lines//2` goal was dropped in the process -- so
% `END { print c["a"], c["b"] }` printed `48` instead of `4 8`. A one-space failure from a
% clause-head edit, caught by an unrelated suite (tests/test_plawk_literal_assoc_key.pl),
% not by anything in this one. Pinned here, in both key spaces, because this suite owns the
% clauses that were rewritten.
test(two_adjacent_assoc_end_fields_keep_the_separator,
        [condition(clang_available)]) :-
    run("{ c[$1]++ } END { print c[\"5\"], c[\"7\"] }\n", "2 1\n"),
    !,
    run("{ split($0, a, \" \") } END { print a[\"1\"], a[\"2\"] }\n", "7 disk\n"),
    !,
    run("{ split($0, a, \" \") } END { print a[1], a[2] }\n", "7 disk\n"),
    !,
    % ...and mixed with a literal, which used a different clause and always worked.
    run("{ c[$1]++ } END { print c[\"5\"], \"x\" }\n", "2 x\n"),
    !.

% --- the rule has ONE home ----------------------------------------------
%
% The three sites emit different IR (a value read, an occupancy probe, a delete call), so
% they cannot share an emitter -- but they must share the ANSWER. Checked at the IR level:
% on a positional table a literal subscript must never reach @wam_intern_atom, and on an
% awk-semantics table it must never become a bare raw key.

test(a_positional_literal_subscript_never_interns) :-
    build_ll("{ split($0, a, \" \"); delete a[\"1\"]; print a[\"1\"] }\n", LL),
    assertion(\+ sub_string(LL, _, _, _, "@wam_intern_subsep_key_comp")),
    !.

test(an_awk_table_int_subscript_read_does_intern) :-
    build_ll("{ c[$1]++; print c[5] }\n", LL),
    assertion(sub_string(LL, _, _, _, "@wam_intern_subsep_key_comp")),
    !.

:- end_tests(plawk_posarray_keyspace).

% --- helpers ---------------------------------------------------------------

odir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_posarray_keyspace', Dir),
    ( exists_directory(Dir) -> true ; make_directory_path(Dir) ).

run(Src, Expected) :-
    odir(Dir),
    input(Input),
    directory_file_path(Dir, 'ks_bin', Bin),
    % A build that unexpectedly declines must not silently run a stale binary.
    ( exists_file(Bin) -> delete_file(Bin) ; true ),
    directory_file_path(Dir, 'ks', Prog0),
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

% Build only, asserting the CLI status (2 = parse error, 3 = parses but outside
% the compilable surface).
build_status(Src, ExpectedStatus) :-
    odir(Dir),
    directory_file_path(Dir, 'ks_reject', Prog0),
    atom_concat(Prog0, '.plawk', Prog),
    setup_call_cleanup(open(Prog, write, S, [encoding(utf8)]),
        write(S, Src), close(S)),
    atom_concat(Prog0, '_bin', Bin),
    cli([build, Prog, '-o', Bin], ExpectedStatus).

% The emitted LLVM IR, straight from the code generator -- no clang needed, so these
% pins run everywhere.
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
