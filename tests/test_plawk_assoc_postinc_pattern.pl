:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% awk's dedupe idiom `!seen[$1]++` and its positive twin `seen[$1]++`.
%
% ---------------------------------------------------------------------------
% The pattern reads the OLD value of seen[K] and increments it. It failed to parse
% (exit 2). It is now parsed to assoc_postinc(K, A) and, when exact, desugared by
% plawk_desugar_assoc_postinc_patterns/2 into two membership rules the codegen
% already compiles:
%
%     !A[K]++ { ACT }  ==>  K in A { A[K]++ }       !(K in A) { A[K]++; ACT }
%      A[K]++ { ACT }  ==>  K in A { A[K]++; ACT }  !(K in A) { A[K]++ }
%
% Exactly one of each pair fires per record, and the increment precedes ACT, as in
% awk. It is exact only while membership and truthiness agree, i.e. while every
% write to A is `A[x]++`; so the rewrite requires the rules to touch A only through
% `A[x]++` and `in`, and BEGIN not to mention A at all. Otherwise the raw
% assoc_postinc term stays and codegen declines (exit 3) -- never a guess.
%
% gawk is the oracle (verified against gawk 5.1.0, LC_ALL=C).

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(lists)).
:- use_module(library(filesex), [make_directory_path/1]).
:- use_module('../examples/plawk/parser/plawk_parser').

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

input("alice 30 NY\nbob 25 LA\ncarol 35 NY\ndave 40 SF\nalice 12 LA\n").

:- begin_tests(plawk_assoc_postinc_pattern).

% --- the desugaring ----------------------------------------------------------

test(negated_form_desugars) :-
    plawk_parse_string("!seen[$1]++\n",
        program([],
            [ rule(in_arr(field(1), seen), [inc_assoc(var(seen), field(1))]),
              rule(not_pat(in_arr(field(1), seen)),
                   [inc_assoc(var(seen), field(1)), print([field(0)])])
            ], [])),
    !.

test(positive_form_desugars) :-
    plawk_parse_string("seen[$3]++ { print $3 }\n",
        program([],
            [ rule(in_arr(field(3), seen),
                   [inc_assoc(var(seen), field(3)), print([field(3)])]),
              rule(not_pat(in_arr(field(3), seen)), [inc_assoc(var(seen), field(3))])
            ], [])),
    !.

% The pair replaces the rule IN PLACE: rules before and after keep their order.
test(desugar_keeps_rule_order) :-
    plawk_parse_string("{ n++ }\n!s[$3]++ { print $3 }\nEND { print n }\n",
        program([], [rule(always, _), rule(in_arr(field(3), s), _),
                     rule(not_pat(in_arr(field(3), s)), _)], [end(_)])),
    !.

% --- when it would not be exact, it does not happen -------------------------

% A rule READS A: not an admitted use, so the raw term stays (codegen declines).
test(read_of_the_array_in_a_rule_blocks_it) :-
    plawk_parse_string("!seen[$1]++ { print seen[$1] }\n",
        program([], [rule(not_pat(assoc_postinc(field(1), seen)), _)], [])),
    !.

% Nested inside a combinator: no rule-level pair exists, the term stays.
test(nested_postinc_is_left_alone) :-
    plawk_parse_string("!s[$1]++ && $2 > 20\n",
        program([], [rule(and_pat(not_pat(assoc_postinc(field(1), s)), _), _)], [])),
    !.

% END may do anything with A: it runs after the last pattern.
test(end_use_does_not_block_it) :-
    plawk_parse_string("!s[$1]++\nEND { for (k in s) print k, s[k] }\n",
        program([], [rule(in_arr(field(1), s), _),
                     rule(not_pat(in_arr(field(1), s)), _)], _)),
    !.

% A tagged-union program carries its rules as case_blocks(...), not a list. The
% desugaring must pass it through untouched -- when it did not, every tagged-union
% program became a parse error (caught by the full sweep, 5 union suites red).
test(tagged_union_rules_pass_through) :-
    plawk_parse_string("BEGIN { BINFMT = \"case(i64 | i64)\" } case 0 { { a[$1]++ } } case 1 { { a[$1]++ } } END { print 1 }\n",
        program(_, case_blocks(_), _)),
    !.

% --- runtime parity ------------------------------------------------------------

test(first_occurrence_of_a_key, [condition(clang_available)]) :-
    run("!seen[$1]++\n", "alice 30 NY\nbob 25 LA\ncarol 35 NY\ndave 40 SF\n"),
    !,
    run("!seen[$3]++ { print $3 }\n", "NY\nLA\nSF\n"),
    !.

test(repeats_only, [condition(clang_available)]) :-
    run("seen[$1]++\n", "alice 12 LA\n"),
    !,
    run("seen[$3]++ { print \"dup\", $3 }\n", "dup NY\ndup LA\n"),
    !.

% The counts left in the array are awk's (every record incremented once).
test(counts_are_awks, [condition(clang_available)]) :-
    run("!s[$3]++ { print $3 }\nEND { print s[\"NY\"], s[\"LA\"], s[\"SF\"] }\n",
        "NY\nLA\nSF\n2 2 1\n"),
    !.

% Forms not reached here decline or fail to parse -- never exit 4.
test(unreached_forms_do_not_miscompile) :-
    forall(member(Src,
            [ "!seen[$1]++\nEND { print NR }\n",
              "!a[$0]++\n",
              "!seen[$1]++ { print seen[$1] }\n",
              "!s[$1]++ && $2 > 20\n",
              "{ n++ }\n!s[$3]++ { print $3, n }\n",
              "seen[$1]++ == 0\n"
            ]),
        build_status_not_4(Src)),
    !.

:- end_tests(plawk_assoc_postinc_pattern).

% --- helpers ---------------------------------------------------------------

odir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_assoc_postinc_pattern', Dir),
    ( exists_directory(Dir) -> true ; make_directory_path(Dir) ).

run(Src, Expected) :-
    input(Input),
    odir(Dir),
    directory_file_path(Dir, 'api_bin', Bin),
    % A build that unexpectedly declines must not silently run a stale binary.
    ( exists_file(Bin) -> delete_file(Bin) ; true ),
    directory_file_path(Dir, 'api', Prog0),
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

build_status_not_4(Src) :-
    odir(Dir),
    directory_file_path(Dir, 'api_status', Prog0),
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

cli(Args, ExpectedStatus) :-
    process_create(path(swipl), ['examples/plawk/bin/plawk' | Args],
        [stdout(pipe(S)), stderr(std), process(Pid)]),
    read_string(S, _, _), close(S),
    process_wait(Pid, exit(Status)),
    assertion(Status == ExpectedStatus).
