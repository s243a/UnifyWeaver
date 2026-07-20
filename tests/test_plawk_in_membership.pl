:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% AWK associative-array membership: `key in arr` tests whether an occupied
% entry exists without reading its value and without inserting a missing key.
% The distinction matters for stored zero values and for later for-in counts.
% v1 accepts the established key forms; two-field multi-dimensional keys reuse
% the SUBSEP interning path. Unsupported wider tuples remain a clean codegen
% rejection rather than being approximated or autovivified.

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(filesex), [make_directory_path/1]).
:- use_module('../examples/plawk/parser/plawk_parser').

:- begin_tests(plawk_in_membership).

% --- parsing ----------------------------------------------------------------

test(field_key_parses) :-
    plawk_parse_string("{ if ($1 in seen) print $1 }\n",
        program([], [rule(always,
            [if(in_arr(field(1), seen), [print([field(1)])], [])])], [])),
    !.

test(string_key_parses) :-
    plawk_parse_string("{ if (\"x\" in seen) print $1 }\n",
        program([], [rule(always,
            [if(in_arr(string("x"), seen), [print([field(1)])], [])])], [])),
    !.

test(integer_key_parses) :-
    plawk_parse_string("{ if (2 in parts) print $1 }\n",
        program([], [rule(always,
            [if(in_arr(int(2), parts), [print([field(1)])], [])])], [])),
    !.

test(variable_key_parses) :-
    plawk_parse_string("{ if (k in seen) print k }\n",
        program([], [rule(always,
            [if(in_arr(var(k), seen), [print([var(k)])], [])])], [])),
    !.

test(two_field_key_parses) :-
    plawk_parse_string("{ if (($1,$2) in pairs) print $1 }\n",
        program([], [rule(always,
            [if(in_arr(subsep_key([field(1), field(2)]), pairs),
                [print([field(1)])], [])])], [])),
    !.

test(negated_membership_parses) :-
    plawk_parse_string("{ if (!($1 in seen)) print $1 }\n",
        program([], [rule(always,
            [if(not_pat(in_arr(field(1), seen)),
                [print([field(1)])], [])])], [])),
    !.

% Parentheses can put a membership term in a general value position. Preserve
% that surface in the AST so codegen, rather than the parser, owns the current
% unsupported-value diagnostic.
test(parenthesized_value_membership_parses) :-
    plawk_parse_string("{ print (k in arr) }\n",
        program([], [rule(always,
            [print([in_expr(in_arr(var(k), arr))])])], [])),
    !.

% --- runtime ----------------------------------------------------------------

% This is the exact distinct idiom that motivated the feature. The membership
% check runs before the increment, so only first occurrences print. Critically,
% a miss must not insert: the END for-in cardinality remains exactly three.
test(first_seen_and_exact_cardinality, [condition(clang_available)]) :-
    mdir(Dir),
    Src = "{ if (!($1 in seen)) print $1; seen[$1]++ } END { n=0; for (k in seen) n++; print n }\n",
    build_run(Dir, 'distinct', Src, "a\na\nb\nc\nb\n", Out, St),
    assertion(St == 0),
    assertion(Out == "a\nb\nc\n3\n"),
    !.

% A field key inserted earlier in the same action sequence is present; an
% unrelated string-literal key is absent. This also covers positive membership
% and negation without relying on a zero/nonzero table value.
test(single_present_and_absent, [condition(clang_available)]) :-
    mdir(Dir),
    Src = "{ seen[$1]++; if ($1 in seen) print \"present\"; if (!(\"missing\" in seen)) print \"absent\" } END { n=0; for (k in seen) n++; print n }\n",
    build_run(Dir, 'present_absent', Src, "x\n", Out, St),
    assertion(St == 0),
    assertion(Out == "present\nabsent\n1\n"),
    !.

% The field-slice membership surface is deliberately 1-based in v1. `$0`
% parses as a field but must fail closed rather than being treated as empty.
test(whole_record_membership_rejected, [condition(clang_available)]) :-
    mdir(Dir),
    build_status(Dir, 'whole_record',
        "{ if ($0 in whole) print \"bad\" }\n", St),
    assertion(St == 3),
    !.

% A two-field membership key must be byte-identical to the key made by
% `pairs[$1,$2]++`: both routes join the field slices through SUBSEP.
test(two_field_subsep_membership, [condition(clang_available)]) :-
    mdir(Dir),
    Src = "{ pairs[$1,$2]++; if (($1,$2) in pairs) print \"present\" } END { n=0; for (k in pairs) n++; print n }\n",
    build_run(Dir, 'subsep', Src, "a x\na x\na y\n", Out, St),
    assertion(St == 0),
    assertion(Out == "present\npresent\npresent\n2\n"),
    !.

% A literal membership check becomes true only after that literal has been
% inserted. The first record therefore emits nothing and the second emits x.
test(string_literal_membership, [condition(clang_available)]) :-
    mdir(Dir),
    Src = "{ seen[$1]++; if (\"x\" in seen) print \"x\" } END { n=0; for (k in seen) n++; print n }\n",
    build_run(Dir, 'literal', Src, "y\nx\n", Out, St),
    assertion(St == 0),
    assertion(Out == "x\n2\n"),
    !.

% Empty string is a real awk key. A leading comma makes $1 empty, so the
% literal lookup must find the key written through that field slice.
test(empty_string_literal_membership, [condition(clang_available)]) :-
    mdir(Dir),
    Src = "BEGIN { FS=\",\" } { seen[$1]++; if (\"\" in seen) print \"present\" } END { n=0; for (k in seen) n++; print n }\n",
    build_run(Dir, 'empty_literal', Src, ",x\n", Out, St),
    assertion(St == 0),
    assertion(Out == "present\n1\n"),
    !.

% In a text-keyed table, an integer literal is interned by its decimal spelling:
% literal 2 must find the same key previously inserted from field text "2".
test(integer_literal_matches_text_field_key,
        [condition(clang_available)]) :-
    mdir(Dir),
    Src = "{ seen[$1]++; if (2 in seen) print \"present\" } END { n=0; for (k in seen) n++; print n }\n",
    build_run(Dir, 'integer_text', Src, "1\n2\n", Out, St),
    assertion(St == 0),
    assertion(Out == "present\n2\n"),
    !.

% `+= 0` creates an occupied entry whose value is zero. Membership must query
% the occupied bit, not approximate it as `wam_assoc_i64_get(...) != 0`.
test(stored_zero_is_present, [condition(clang_available)]) :-
    mdir(Dir),
    Src = "{ zero[$1] += 0; if ($1 in zero) print \"present\" } END { n=0; for (k in zero) n++; print n }\n",
    build_run(Dir, 'zero', Src, "z\n", Out, St),
    assertion(St == 0),
    assertion(Out == "present\n1\n"),
    !.

% Positional arrays use raw integer keys. A one-record split creates positions
% 1..3, so 2 exists and 4 does not.
test(integer_membership_on_split_array, [condition(clang_available)]) :-
    mdir(Dir),
    Src = "{ split($0, parts, \",\"); if (2 in parts) print \"present\"; if (!(4 in parts)) print \"absent\" } END { n=0; for (k in parts) n++; print n }\n",
    build_run(Dir, 'integer', Src, "a,b,c\n", Out, St),
    assertion(St == 0),
    assertion(Out == "present\nabsent\n3\n"),
    !.

% Membership is also a rule pattern, not only an if condition. The first rule
% inserts before the second rule's guard is evaluated for the same record.
test(top_level_rule_membership, [condition(clang_available)]) :-
    mdir(Dir),
    Src = "{ seen[$1]++ } $1 in seen { print $1 }\n",
    build_run(Dir, 'top_level', Src, "a\nb\n", Out, St),
    assertion(St == 0),
    assertion(Out == "a\nb\n"),
    !.

% A for-in loop variable is a real table key in its narrow filter scope.
test(for_in_variable_membership, [condition(clang_available)]) :-
    mdir(Dir),
    Src = "{ a[$1]++; b[$2]++ } END { for (k in a) { if (k in b) print k } }\n",
    build_run(Dir, 'for_in_var', Src, "a a\nb x\n", Out, St),
    assertion(St == 0),
    assertion(Out == "a\n"),
    !.

% Negation uses the same scoped loop key without reading or mutating b. The
% single result also makes the assertion independent of hash iteration order.
test(negated_for_in_variable_membership, [condition(clang_available)]) :-
    mdir(Dir),
    Src = "{ a[$1]++; b[$2]++ } END { for (k in a) { if (!(k in b)) print k } }\n",
    build_run(Dir, 'for_in_var_not', Src, "a a\nb x\n", Out, St),
    assertion(St == 0),
    assertion(Out == "b\n"),
    !.

% Planning folds paired negations, so recursive `!` remains valid without
% defining the same LLVM SSA name twice.
test(double_negated_for_in_variable_membership,
        [condition(clang_available)]) :-
    mdir(Dir),
    Src = "{ a[$1]++; b[$2]++ } END { for (k in a) { if (!(!(k in b))) print k } }\n",
    build_run(Dir, 'for_in_var_double_not', Src, "a a\nb x\n", Out, St),
    assertion(St == 0),
    assertion(Out == "a\n"),
    !.

% Boolean wrappers recurse through the membership-aware emitter while ordinary
% comparison leaves retain their existing meaning.
test(boolean_membership_with_ordinary_leaves,
        [condition(clang_available)]) :-
    mdir(Dir),
    Src = "{ seen[$1]++; if (($1 in seen) && $2 == \"yes\") print \"and\"; if ((\"missing\" in seen) || $2 == \"yes\") print \"or\" }\n",
    build_run(Dir, 'boolean', Src, "x yes\ny no\n", Out, St),
    assertion(St == 0),
    assertion(Out == "and\nor\n"),
    !.

% The parser deliberately preserves a wider SUBSEP key in the AST, but v1 only
% lowers exactly two field components. It must fail with compile status 3.
test(three_field_membership_rejected, [condition(clang_available)]) :-
    mdir(Dir),
    Src = "{ seen[$1,$2]++; if (($1,$2,$3) in seen) print \"bad\" } END { n=0; for (k in seen) n++; print n }\n",
    build_status(Dir, 'three_fields', Src, St),
    assertion(St == 3),
    !.

% Membership is a guard in v1, not a printable value. The explicit in_expr AST
% keeps this a clean compilable-surface error (3), not a parse error (2).
test(parenthesized_value_membership_rejected,
        [condition(clang_available)]) :-
    mdir(Dir),
    build_status(Dir, 'value_expr', "{ print (k in arr) }\n", St),
    assertion(St == 3),
    !.

% A split table's loop key is a raw integer position, while an ordinary
% associative table's key is an interned string id. v1 rejects forwarding a
% key between those domains instead of comparing unrelated i64 values.
test(cross_domain_for_in_membership_rejected,
        [condition(clang_available)]) :-
    mdir(Dir),
    Src1 = "{ split($0, parts, \",\"); text[$1]++ } END { for (k in parts) { if (k in text) print k } }\n",
    build_status(Dir, 'for_in_pos_to_text', Src1, St1),
    assertion(St1 == 3),
    Src2 = "{ split($0, parts, \",\"); text[$1]++ } END { for (k in text) { if (k in parts) print k } }\n",
    build_status(Dir, 'for_in_text_to_pos', Src2, St2),
    assertion(St2 == 3),
    !.

% Rule-body for-in uses a separate planner from END and enforces the same
% positional-vs-text key-domain boundary.
test(cross_domain_rule_body_membership_rejected,
        [condition(clang_available)]) :-
    mdir(Dir),
    Src = "{ split($0, parts, \",\"); text[$1]++; for (k in parts) { if (k in text) print k } }\n",
    build_status(Dir, 'rule_for_in_pos_to_text', Src, St),
    assertion(St == 3),
    !.

% The tagged-union for-in driver currently supports only an unguarded print.
% A membership filter must not be silently discarded by that broader driver.
test(tagged_union_for_in_membership_rejected,
        [condition(clang_available)]) :-
    mdir(Dir),
    Src = "BEGIN { BINFMT = \"case(i64 | i64)\" } case 0 { { a[$1]++; b[$1]++ } } case 1 { { a[$1]++; b[$1]++ } } END { for (k in a) { if (k in b) print k } }\n",
    build_status(Dir, 'tagged_for_in_membership', Src, St),
    assertion(St == 3),
    !.

:- end_tests(plawk_in_membership).

% --- helpers ----------------------------------------------------------------

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

mdir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_in_membership', Dir),
    ( exists_directory(Dir) -> true ; make_directory_path(Dir) ).

write_prog(Dir, Name, Src, Bin-Prog) :-
    directory_file_path(Dir, Name, Prog0),
    atom_concat(Prog0, '.plawk', Prog),
    setup_call_cleanup(open(Prog, write, S, [encoding(utf8)]),
        write(S, Src), close(S)),
    atom_concat(Prog0, '_bin', Bin).

build_status(Dir, Name, Src, Status) :-
    write_prog(Dir, Name, Src, Bin-Prog),
    process_create(path(swipl), ['examples/plawk/bin/plawk', build, Prog, '-o', Bin],
        [stdout(null), stderr(null), process(Pid)]),
    process_wait(Pid, exit(Status)).

build_run(Dir, Name, Src, Input, Out, RunStatus) :-
    write_prog(Dir, Name, Src, Bin-Prog),
    process_create(path(swipl), ['examples/plawk/bin/plawk', build, Prog, '-o', Bin],
        [stdout(null), stderr(std), process(BPid)]),
    process_wait(BPid, exit(0)),
    process_create(Bin, [],
        [stdin(pipe(In)), stdout(pipe(RS)), stderr(std), process(RPid)]),
    format(In, "~w", [Input]),
    close(In),
    read_string(RS, _, Out),
    close(RS),
    process_wait(RPid, exit(RunStatus)).
