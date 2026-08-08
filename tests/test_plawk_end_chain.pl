:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% plawk: a MIXED END statement chain -- `for (k in arr)` loops interleaved with
% plain statements.
%
% The all-loops driver chained N for-in loops; the scalar driver chained N plain
% statements. Neither accepted a MIX, so several shapes declined for one
% structural reason:
%
%   END { for (k in c) print k; print "done" }
%   END { print "start"; for (k in c) print k }
%   END { for (k in c) print k; exit 2 }
%   END { for (k in c) print k; print c["a"] }
%   END { for (k in c) print k; print "--"; for (j in d) print j }
%
% A for-in loop is a GROUP OF BASIC BLOCKS (`forin<I>_head` / `_body` /
% `_body_done` / `_after`) wired to a predecessor and a successor, so a plain
% statement joining that chain needs a block of its own: `endstmt<I>`, holding its
% straight-line print IR and the same tail (branch onward, or free the tables and
% return). Modelling both kinds as chain ITEMS with an entry and an exit label is
% what lets them interleave in any order.
%
% Two things had to be got right, both found by running the output rather than
% reading the IR:
%
%   * the assoc END print emitter FREES every table in its base case, which is
%     correct when the print is the whole END block but double-frees in a chain
%     (and leaves a later loop iterating freed memory). The chain uses a
%     non-freeing body variant; its tail frees exactly once.
%   * a plain statement's module globals must carry the SAME per-statement rename
%     as its body, for BOTH string literals (`end_`-prefixed) and literal assoc
%     keys (`plawk_assoc_print_key_N`, which the `end_` rename does not touch --
%     so two literal-key lookups would otherwise share ..._key_0).
%
% for-in iteration order is hash-dependent, so key lines are compared as sorted
% sets. gawk 5.2 is the oracle, exit STATUS included.

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(filesex), [make_directory_path/1]).
:- use_module('../examples/plawk/parser/plawk_parser').
:- use_module('../examples/plawk/codegen/plawk_native_codegen').

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

% Three records: "a 1" / "b 2" / "c 3". So `c[$1]` has keys a,b,c and `d[$2]` has
% keys 1,2,3, each with count 1.
input("a 1\nb 2\nc 3\n").

:- begin_tests(plawk_end_chain).

% --- loop then plain statement --------------------------------------------

test(forin_then_plain_print, [condition(clang_available)]) :-
    run_sorted("{ c[$1]++ } END { for (k in c) print k; print \"done\" }\n",
        ["a", "b", "c", "done"], 0),
    !.

test(forin_then_two_plain_prints, [condition(clang_available)]) :-
    run_sorted("{ c[$1]++ } END { for (k in c) print k; print \"x\"; print \"y\" }\n",
        ["a", "b", "c", "x", "y"], 0),
    !.

% A plain statement reading the table by LITERAL key, after the loop.
test(forin_then_assoc_lookup, [condition(clang_available)]) :-
    run_sorted("{ c[$1]++ } END { for (k in c) print k; print c[\"a\"] }\n",
        ["a", "b", "c", "1"], 0),
    !.

% TWO literal-key lookups in separate statements: both must resolve their own key
% global. Without the key rename they shared ..._key_0.
test(forin_then_two_assoc_lookups, [condition(clang_available)]) :-
    run_sorted("{ c[$1]++ } END { for (k in c) print k; print c[\"a\"]; print c[\"b\"] }\n",
        ["a", "b", "c", "1", "1"], 0),
    !.

% --- plain statement then loop --------------------------------------------

test(plain_print_then_forin, [condition(clang_available)]) :-
    run_sorted("{ c[$1]++ } END { print \"start\"; for (k in c) print k }\n",
        ["start", "a", "b", "c"], 0),
    !.

test(two_plain_prints_then_forin, [condition(clang_available)]) :-
    run_sorted("{ c[$1]++ } END { print \"a\"; print \"b\"; for (k in c) print k }\n",
        ["a", "b", "a", "b", "c"], 0),
    !.

% --- plain statement between two loops ------------------------------------

% The interesting wiring case: the plain block must receive control from loop 0's
% `_after` and branch into loop 2's `_head`.
test(plain_print_between_two_forins, [condition(clang_available)]) :-
    run_sorted("{ c[$1]++; d[$2]++ } END { for (k in c) print k; print \"--\"; \c
                for (j in d) print j }\n",
        ["a", "b", "c", "--", "1", "2", "3"], 0),
    !.

% Ordering is not just a set property: check the separator really lands between
% the two loops' output, not before or after both.
test(plain_print_between_forins_is_positioned, [condition(clang_available)]) :-
    run_lines("{ c[$1]++; d[$2]++ } END { for (k in c) print k; print \"--\"; \c
               for (j in d) print j }\n", Lines, 0),
    nth0(SepIndex, Lines, "--"),
    assertion(SepIndex == 3),
    length(Lines, 7),
    !.

% --- exit in the chain ----------------------------------------------------

test(forin_then_exit, [condition(clang_available)]) :-
    run_sorted("{ c[$1]++ } END { for (k in c) print k; exit 2 }\n",
        ["a", "b", "c"], 2),
    !.

test(forin_print_then_exit, [condition(clang_available)]) :-
    run_sorted("{ c[$1]++ } END { for (k in c) print k; print \"x\"; exit 5 }\n",
        ["a", "b", "c", "x"], 5),
    !.

% Statements after the exit are dead, as everywhere else.
test(exit_truncates_the_chain, [condition(clang_available)]) :-
    run_sorted("{ c[$1]++ } END { for (k in c) print k; exit 1; print \"never\" }\n",
        ["a", "b", "c"], 1),
    !.

% An exit BEFORE the loop suppresses the loop entirely.
test(exit_before_loop_suppresses_it, [condition(clang_available)]) :-
    run_sorted("{ c[$1]++ } END { print \"only\"; exit 4; for (k in c) print k }\n",
        ["only"], 4),
    !.

% A bare `exit` in the chain is status 0.
test(forin_then_bare_exit, [condition(clang_available)]) :-
    run_sorted("{ c[$1]++ } END { for (k in c) print k; exit }\n",
        ["a", "b", "c"], 0),
    !.

% --- regressions: shapes the older drivers own ---------------------------

% All-loops END stays with the multi-for-in driver (byte-identical IR).
test(two_forins_unchanged, [condition(clang_available)]) :-
    run_sorted("{ c[$1]++; d[$2]++ } END { for (k in c) print k; for (j in d) print j }\n",
        ["a", "b", "c", "1", "2", "3"], 0),
    !.

test(single_forin_unchanged, [condition(clang_available)]) :-
    run_sorted("{ c[$1]++ } END { for (k in c) print k, c[k] }\n",
        ["a 1", "b 1", "c 1"], 0),
    !.

% All-plain assoc END stays with its own driver.
test(assoc_end_print_unchanged, [condition(clang_available)]) :-
    run_sorted("{ c[$1]++ } END { print c[\"a\"] }\n", ["1"], 0),
    !.

% --- clean declines ------------------------------------------------------

% `printf` is not in the chain's statement vocabulary yet (the assoc chain has no
% scalar state plan for its arguments) -- declines rather than dropping output.
test(forin_then_printf_declines) :-
    build_status("{ c[$1]++ } END { for (k in c) print k; printf \"n=%d\\n\", 1 }\n", 3),
    !.

% WAS a decline -- "not wired to the record counter yet". It is now, for free: NR needs no
% capability, only a state plan to ask, and plawk_end_nr_value/2 falls back to `%plawk_nr`,
% which this driver already emits when a print mentions NR.
%
% `printf` above still declines, and a field read / NF in this chain still declines, because
% those DO need something threaded (scalar argument state, and the retained-record token
% respectively). The boundary is per-capability, not per-route -- which is why one of these
% pins could flip while its neighbour stayed put.
test(nr_in_plain_chain_statement_now_works, [condition(clang_available)]) :-
    run_sorted("{ c[$1]++ } END { for (k in c) print k; print NR }\n",
        ["3", "a", "b", "c"], 0),
    !.

% A LOOP-FREE assoc END statement list (`END { print c["a"]; print "done" }`) is
% a DIFFERENT driver -- straight-line, no block chain -- and landed as the sibling
% to this PR; its behaviour is covered in tests/test_plawk_assoc_end_list.pl. This
% chain still requires at least one loop, which keeps it from claiming that shape;
% chain_items_require_a_mix/0 below asserts exactly that.

% --- structure -----------------------------------------------------------

% The chain claims a MIXED list only: it needs at least one loop and at least one
% plain statement, so it cannot steal the all-loops or all-plain shapes.
test(chain_items_require_a_mix) :-
    Loop = for_in(var(k), var(c), [print([var(k)])]),
    Plain = print([string("x")]),
    assertion(plawk_native_codegen:plawk_end_chain_items([Loop, Plain], _)),
    assertion(plawk_native_codegen:plawk_end_chain_items([Plain, Loop], _)),
    assertion(\+ plawk_native_codegen:plawk_end_chain_items([Loop, Loop], _)),
    assertion(\+ plawk_native_codegen:plawk_end_chain_items([Plain, Plain], _)),
    assertion(\+ plawk_native_codegen:plawk_end_chain_items([Loop], _)),
    !.

% Entry and exit labels: a loop is entered at its head and left from its
% `_after` block; a plain statement's single block is both.
test(chain_labels) :-
    Loop = loop(k, c, [var(k)]),
    Plain = plain(print([string("x")])),
    plawk_native_codegen:plawk_end_chain_entry_label(Loop, 0, L0Entry),
    plawk_native_codegen:plawk_end_chain_exit_label(Loop, 0, L0Exit),
    assertion(L0Entry == forin_head),
    assertion(L0Exit == forin_after),
    plawk_native_codegen:plawk_end_chain_entry_label(Loop, 2, L2Entry),
    assertion(L2Entry == forin2_head),
    plawk_native_codegen:plawk_end_chain_entry_label(Plain, 1, P1Entry),
    plawk_native_codegen:plawk_end_chain_exit_label(Plain, 1, P1Exit),
    assertion(P1Entry == endstmt1),
    assertion(P1Exit == endstmt1),
    !.

% The non-freeing print body really omits the frees the full emitter appends.
test(chain_print_body_omits_table_frees) :-
    Plan = assoc_plan([c], []),
    % The assoc emitter takes an EndRecord capability token now (it grew field/NF-in-END
    % support). `no_end_record` here: this test is about the table frees, and a string literal
    % reads no record, so the token cannot affect what is asserted.
    plawk_native_codegen:plawk_assoc_end_print_ir([string("x")], Plan, 32, [32],
        no_end_record, WithFrees),
    plawk_native_codegen:plawk_assoc_end_print_body_ir([string("x")], Plan, 32,
        [32], WithoutFrees),
    assertion(once(sub_atom(WithFrees, _, _, _, '@wam_assoc_i64_free'))),
    assertion(\+ sub_atom(WithoutFrees, _, _, _, '@wam_assoc_i64_free')),
    !.

% --- IR shape ------------------------------------------------------------

% The plain statement gets its own block, and the tables are freed exactly ONCE
% for a one-table chain -- a per-statement free was the double-free bug.
test(chain_ir_frees_each_table_once) :-
    plawk_parse_string("{ c[$1]++ } END { for (k in c) print k; print \"done\" }\n",
        Program),
    plawk_program_native_driver_ir(Program, 'input.txt', DriverIR),
    assertion(once(sub_atom(DriverIR, _, _, _, 'endstmt1:'))),
    findall(B, sub_atom(DriverIR, B, _, _,
        '@wam_assoc_i64_free(%WamAssocI64Table* %plawk_assoc_table_0)'), Frees),
    assertion(Frees = [_]),
    !.

% An exit in the chain returns @plawk_exit_code rather than the literal 0 the
% all-loops tail emits.
test(chain_exit_ir_returns_exit_code) :-
    plawk_parse_string("{ c[$1]++ } END { for (k in c) print k; exit 2 }\n",
        Program),
    plawk_program_native_driver_ir(Program, 'input.txt', DriverIR),
    assertion(once(sub_atom(DriverIR, _, _, _,
        'store i32 2, i32* @plawk_exit_code'))),
    assertion(once(sub_atom(DriverIR, _, _, _, 'ret i32 %plawk_exit_ec'))),
    !.

:- end_tests(plawk_end_chain).

% --- helpers ---------------------------------------------------------------

odir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_end_chain', Dir),
    ( exists_directory(Dir) -> true ; make_directory_path(Dir) ).

% Build + run, comparing output lines as a SORTED set and asserting the exit
% status (for-in order is hash-dependent).
run_sorted(Src, ExpectedSortedLines, ExpectedStatus) :-
    run_lines(Src, Lines, Status),
    msort(Lines, SortedLines),
    maplist(atom_string, ExpectedAtoms, ExpectedSortedLines),
    maplist(atom_string, ExpectedAtoms, ExpectedStrings),
    msort(ExpectedStrings, ExpectedSorted),
    assertion(SortedLines == ExpectedSorted),
    assertion(Status == ExpectedStatus).

% Build + run, returning the output lines IN ORDER and the exit status.
run_lines(Src, Lines, Status) :-
    odir(Dir),
    input(Input),
    directory_file_path(Dir, 'ec_bin', Bin),
    % A build that unexpectedly declines must not silently run a stale binary.
    ( exists_file(Bin) -> delete_file(Bin) ; true ),
    directory_file_path(Dir, 'ec', Prog0),
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
    process_wait(Pid, exit(Status)),
    split_string(Out, "\n", "", Lines0),
    exclude(==(""), Lines0, Lines).

% Build only, asserting the CLI status (3 = parses but outside the compilable
% surface -- a clean decline, distinct from 2 = parse error).
build_status(Src, ExpectedStatus) :-
    odir(Dir),
    directory_file_path(Dir, 'ec_decline', Prog0),
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
