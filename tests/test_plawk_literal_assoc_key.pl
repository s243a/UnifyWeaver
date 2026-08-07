:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% A STRING-LITERAL associative key -- `c["total"]++`, `t["all"] += $2`, `c["x"]--`.
% The last column of the key-coverage matrix recorded in
% tests/test_plawk_assoc_decrement.pl and tests/test_plawk_assoc_key_coverage.pl:
%
%              field key $1   literal key "x"   scalar-var key k
%   c[K]++          yes           yes <- was no       yes
%   c[K] += N       yes           yes <- was no       yes
%   c[K]--          yes           yes <- was no       yes
%
% ---------------------------------------------------------------------------
% THE ARITY-1 CASE OF SOMETHING THAT ALREADY WORKED
%
% Sized as "a literal-key sibling of the field-key path", this needs a spec, a
% planned action and two emitters -- a third way to say "intern these bytes and use
% them as a key", next to the field-key path and `assoc_delete_lit`.
%
% It needs none of that, because `c["x"]` is the ARITY-1 case of the multi-dimensional
% key `c[$i,"lit",...]`, which already builds any arity from any mix of field and
% compile-time-literal components. The only thing standing in the way was one guard:
%
%     plawk_subsep_key_components(Subscripts, Comps) :-
%         Subscripts = [_, _ | _],        % <- became [_ | _]
%
% The RUNTIME was already general. @wam_intern_subsep_key_comp computes
% `(N-1) * SUBSEP_len` separator bytes and skips the separator for component 0, so at
% N = 1 it interns exactly the component's own bytes -- the same atom id
% @wam_intern_atom of that literal produces. That equality is what lets the counter,
% the reads, membership and `delete` all agree on the key, and it is pinned below by
% counting a literal key and a FIELD holding the same text into the same entry.
%
% Same shape as the `arr[k]--` sizing error (test_plawk_assoc_decrement.pl): follow the
% nearest SEMANTICS ("a key built by joining N described subscripts, N = 1") instead of
% the nearest spelling ("a literal, like assoc_delete_lit's literal"). Two rows at the
% shape gates, one read row, and one relaxed arity guard; no new builder, planned
% action or emitter.
%
% ---------------------------------------------------------------------------
% WHY THE INTEGER-LITERAL KEY IS *NOT* PART OF THIS
%
% `c[5]++` still declines, and not because a row is missing -- `c[$1,5]++` proves the
% integer literal builds fine as a component. It is a KEY-SPACE COLLISION between two
% table conventions that share the surface syntax `arr[5]`:
%
%   - awk semantics: keys are STRINGS, so `arr[5]` is the key "5", interned.
%   - positional tables (`split()`, posarray binds): keys are the RAW INTEGER
%     position, which is how `lookup_int` reads them.
%
% Which space `arr[5]` means is decided by the TABLE'S KIND, not by the subscript.
% Admitting the update would compile `{ c[5]++; print c[5] }` into a store to interned
% "5" and a load from raw 5 -- it BUILDS and prints four empty lines where gawk prints
% 1..4. Verified before deciding, which is why it is refused here rather than shipped.
% Pinned as a decline, with `c[$1,5]++` beside it so the refusal stays attributed to
% the collision and not to the literal.
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

% Four records; $1 is INFO once, DEBUG twice, ERROR once. $2 is never numeric.
input("INFO boot ok\nDEBUG trace one\nERROR disk full\nDEBUG trace two\n").

:- begin_tests(plawk_literal_assoc_key).

% --- the counter, read back in END -----------------------------------------

test(literal_key_counter, [condition(clang_available)]) :-
    run("{ c[\"total\"]++ } END { print c[\"total\"] }\n", "4\n"),
    !.

test(literal_key_add_assign, [condition(clang_available)]) :-
    run("{ c[\"x\"] += 2 } END { print c[\"x\"] }\n", "8\n"),
    !.

test(literal_key_subtract_assign, [condition(clang_available)]) :-
    run("{ c[\"x\"] -= 2 } END { print c[\"x\"] }\n", "-8\n"),
    !.

test(literal_key_decrement, [condition(clang_available)]) :-
    run("{ c[\"x\"]-- } END { print c[\"x\"] }\n", "-4\n"),
    !.

test(literal_key_negative_add_delta, [condition(clang_available)]) :-
    run("{ c[\"x\"] += -3 } END { print c[\"x\"] }\n", "-12\n"),
    !.

% The grand-total idiom, with a field delta.
test(literal_key_field_delta, [condition(clang_available)]) :-
    run("{ t[\"all\"] += $2 } END { print t[\"all\"] }\n", "0\n"),
    !.

% Two distinct literal keys in one rule body, different spellings.
test(two_literal_keys, [condition(clang_available)]) :-
    run("{ c[\"a\"]++; c[\"b\"] += 2 } END { print c[\"a\"], c[\"b\"] }\n", "4 8\n"),
    !.

% A literal key under a rule guard -- the per-category tally idiom.
test(literal_key_under_guards, [condition(clang_available)]) :-
    run("$1 == \"DEBUG\" { c[\"dbg\"]++ }\n$1 == \"ERROR\" { c[\"err\"]++ }\nEND { print c[\"dbg\"], c[\"err\"] }\n",
        "2 1\n"),
    !.

% A key with a space in it -- the bytes are interned as-is, not tokenised.
test(literal_key_with_a_space, [condition(clang_available)]) :-
    run("{ c[\"a b\"]++ } END { print c[\"a b\"] }\n", "4\n"),
    !.

% --- the key is the literal's BYTES ---------------------------------------
%
% The load-bearing property: the arity-1 subsep intern must produce the same atom id
% @wam_intern_atom of those bytes produces, or the counter and every reader would
% disagree. Counting a FIELD holding "INFO" (once) and the literal "INFO" (on all four
% records) into the same table must land on ONE entry, giving 1 + 4 = 5. Two key spaces
% would give 4 instead -- the literal's own entry, with the field's 1 elsewhere.
test(a_literal_key_and_a_field_holding_that_text_are_the_same_key,
        [condition(clang_available)]) :-
    run("{ c[$1]++; c[\"INFO\"]++ } END { print c[\"INFO\"], c[\"ERROR\"] }\n", "5 1\n"),
    !,
    % ...and the for-in view shows exactly THREE entries. A separate literal key space
    % would show four, which is the failure this pin is really watching for -- the sum
    % above could be explained away, an extra entry cannot.
    run_sorted("{ c[$1]++; c[\"INFO\"]++ } END { for (k in c) print k, c[k] }\n",
        "DEBUG 2\nERROR 1\nINFO 5\n"),
    !.

% --- the for-in END route (a different driver) ----------------------------

test(literal_key_forin_end, [condition(clang_available)]) :-
    run_sorted("{ c[\"x\"]++ } END { for (k in c) print k, c[k] }\n", "x 4\n"),
    !.

test(literal_key_add_assign_forin_end, [condition(clang_available)]) :-
    run_sorted("{ c[$1]++; c[\"ALL\"] += 2 } END { for (k in c) print k, c[k] }\n",
        "ALL 8\nDEBUG 2\nERROR 1\nINFO 1\n"),
    !.

% --- reads, membership and delete on a literal key ------------------------

% A rule-body read (`print c["x"]`) -- the running count each record.
test(literal_key_rule_body_read, [condition(clang_available)]) :-
    run("{ c[\"x\"]++; print c[\"x\"] }\n", "1\n2\n3\n4\n"),
    !.

% ...and inside a concatenation, which spec's its parts recursively.
test(literal_key_read_in_a_concat, [condition(clang_available)]) :-
    run("{ c[\"x\"]++; print \"n=\" c[\"x\"] }\n", "n=1\nn=2\nn=3\nn=4\n"),
    !.

% Membership probes the same key the counter built.
test(literal_key_membership, [condition(clang_available)]) :-
    run("{ c[\"x\"]++ } END { if (\"x\" in c) print \"yes\" }\n", "yes\n"),
    !.

% `delete c["x"]` (which interns the literal directly, via its own older path)
% removes the entry this counter created -- the two agree on the key.
test(literal_key_delete_removes_the_counted_entry, [condition(clang_available)]) :-
    run("{ c[\"x\"]++; delete c[\"x\"] } END { print c[\"x\"] }\n", "\n"),
    !.

% --- it really is the multi-dimensional builder at arity 1 ---------------

% The emitted IR must call the N-ary key builder with n = 1 -- not a separate
% literal-key intern. If a sibling path is ever introduced, this is what fails.
test(the_literal_key_rides_the_n_ary_builder_at_arity_one) :-
    build_ll("{ c[\"x\"]++ } END { print c[\"x\"] }\n", LL),
    assertion(sub_string(LL, _, _, _, "@wam_intern_subsep_key_comp")),
    assertion(sub_string(LL, _, _, _, "i64 1, i8 32)")),
    !.

% The descriptor array has exactly ONE element for a lone literal key (the multi-dim
% form's is [2 x ...] or wider), so the arity travels in the IR and not only in a
% Prolog term.
test(the_descriptor_array_has_one_element) :-
    build_ll("{ c[\"x\"]++ } END { print c[\"x\"] }\n", LL),
    assertion(sub_string(LL, _, _, _, "[1 x {i64, i8*, i64}]")),
    !.

% --- what is still refused, pinned --------------------------------------

% An INTEGER-literal key declines -- pinned as a PAIR with the same integer literal
% used as a multi-dim COMPONENT, which works. So the refusal is attributable: the
% integer literal builds fine, and what blocks it is the raw-integer key space that
% positional tables use for `arr[N]` reads.
test(an_integer_literal_key_declines_while_the_same_component_works) :-
    build_status("{ c[5]++ } END { print c[\"5\"] }\n", 3),
    build_status("{ c[5] += 2 } END { print c[\"5\"] }\n", 3),
    !.

test(an_integer_literal_multi_dim_component_still_works, [condition(clang_available)]) :-
    run_sorted("{ c[$1,5]++ } END { for (k in c) print c[k] }\n", "1\n1\n2\n"),
    !.

% The MIXED route (a scalar counter alongside an assoc update) declines for a literal
% key -- pinned as a pair with the multi-dimensional form, which declines identically
% and always has. The mixed chain has no N-ary key path at all, so this restriction
% belongs to that gap, not to the literal key.
test(the_mixed_route_declines_for_literal_and_multi_dim_keys_alike) :-
    build_status("{ n++; c[\"x\"]++ } END { print n }\n", 3),
    build_status("{ n++; c[$1,\"z\"]++ } END { print n }\n", 3),
    !.

% An EMPTY-STRING subscript is a PARSE error, not a decline -- a pre-existing parser
% restriction (`arr[""]` is legal awk), unrelated to the key builder.
test(an_empty_string_subscript_is_a_parse_error) :-
    build_status("{ c[\"\"]++ } END { print c[\"\"] }\n", 2),
    !.

% --- regressions: the multi-dimensional family is untouched -------------

% Three distinct ($1,$2) pairs over four records -- (DEBUG,trace) repeats.
test(multi_dim_counter_unchanged, [condition(clang_available)]) :-
    run_sorted("{ c[$1,$2]++ } END { for (k in c) print c[k] }\n", "1\n1\n2\n"),
    !.

test(all_literal_multi_dim_unchanged, [condition(clang_available)]) :-
    run_sorted("{ c[\"a\",\"b\"]++ } END { for (k in c) print c[k] }\n", "4\n"),
    !.

test(field_key_counter_unchanged, [condition(clang_available)]) :-
    run("{ c[$1]++ } END { print c[\"DEBUG\"] }\n", "2\n"),
    !.

% A split (positional) table still reads by its raw integer position -- the key space
% this change deliberately did not touch.
test(positional_table_read_unchanged, [condition(clang_available)]) :-
    run("{ split($0, a, \" \"); print a[1] }\n", "INFO\nDEBUG\nERROR\nDEBUG\n"),
    !.

:- end_tests(plawk_literal_assoc_key).

% --- helpers ---------------------------------------------------------------

odir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_literal_assoc_key', Dir),
    ( exists_directory(Dir) -> true ; make_directory_path(Dir) ).

run(Src, Expected) :-
    run_(Src, Expected, plain).

run_sorted(Src, Expected) :-
    run_(Src, Expected, sorted).

run_(Src, Expected, Mode) :-
    odir(Dir),
    input(Input),
    directory_file_path(Dir, 'lk_bin', Bin),
    % A build that unexpectedly declines must not silently run a stale binary.
    ( exists_file(Bin) -> delete_file(Bin) ; true ),
    directory_file_path(Dir, 'lk', Prog0),
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

% awk's for-in order is unspecified, so for-in output is compared as a multiset of
% lines. The trailing newline is preserved.
sort_lines(In, Out) :-
    split_string(In, "\n", "", Parts0),
    ( append(Parts, [""], Parts0) -> true ; Parts = Parts0 ),
    msort(Parts, Sorted),
    atomic_list_concat(Sorted, '\n', Joined),
    ( Sorted == [] -> Out = "" ; format(string(Out), "~w\n", [Joined]) ).

% Build only, asserting the CLI status (2 = parse error, 3 = parses but outside
% the compilable surface).
build_status(Src, ExpectedStatus) :-
    odir(Dir),
    directory_file_path(Dir, 'lk_reject', Prog0),
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
