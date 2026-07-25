:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% Multi-dimensional array subscripts `arr[i,j,...]` and SUBSEP. In awk,
% `arr[i,j]` is sugar for `arr[i SUBSEP j]`: the subscripts are joined by SUBSEP
% (default "\034", the FS/0x1C byte) into one string key. Any arity and any mix
% of FIELD and LITERAL subscripts (`arr[$i,$j]`, `arr[$i,$j,$k]`, `arr[$i,"x"]`,
% `arr[$1,5]`, `arr["a","b"]`, ...) is handled uniformly, covering the write
% counter `arr[...]++`, the element read `arr[...]`, and membership
% `(...) in arr`; for-in iteration sees the joined key.
%
% The join is done by the runtime helper @wam_intern_subsep_key_comp: each
% subscript is described by a constant descriptor {field_index, lit_ptr, lit_len}
% (a null lit_ptr means "slice this field"), the helper resolves each component
% to its bytes and joins them with the SUBSEP bytes (@wam_subsep_ptr /
% @wam_subsep_len, default 0x1C, overridable by `BEGIN { SUBSEP = "…" }`), and
% interns the result to one atom id -- the same key the write, read, and
% membership paths build. An integer literal is keyed by its decimal text, so
% `arr[$1,5]` and `arr[$1,"5"]` are the same key. A scalar-VARIABLE subscript is
% a clean not-yet (compile error), not miscompiled: its value is not a
% compile-time constant.

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(filesex), [make_directory_path/1]).
:- use_module('../examples/plawk/parser/plawk_parser').

:- begin_tests(plawk_subsep).

% --- parsing ----------------------------------------------------------------

% `c[$1,$2]++` parses to a subsep_key of the two field subscripts.
test(multidim_inc_parses) :-
    plawk_parse_string("{ c[$1,$2]++ }\n",
        program([], [rule(always,
            [inc_assoc(var(c), subsep_key([field(1), field(2)]))])], [])),
    !.

% `print c[$1,$2]` parses the read as the same subsep_key subscript.
test(multidim_read_parses) :-
    plawk_parse_string("{ print c[$1,$2] }\n",
        program([], [rule(always,
            [print([assoc(var(c), subsep_key([field(1), field(2)]))])])], [])),
    !.

% A single subscript still collapses to the bare expression (unchanged).
test(single_subscript_unchanged) :-
    plawk_parse_string("{ c[$1]++ }\n",
        program([], [rule(always, [inc_assoc(var(c), field(1))])], [])),
    !.

% --- runtime ----------------------------------------------------------------

% Per-record running count: `c[$1,$2]++` then read it back. The (a,x) pair
% counts 1,2,3 across its three records; (b,y) counts 1.
test(running_count, [condition(clang_available)]) :-
    sdir(Dir),
    build_run(Dir, 'rc', "{ c[$1,$2]++; print $1, $2, c[$1,$2] }\n",
        "a x\na x\nb y\na x\n", Out),
    assertion(Out == "a x 1\na x 2\nb y 1\na x 3\n"), !.

% Histogram via END for-in over the joined keys: (a,x) x3, (b,y) x2 -> {2,3}.
test(histogram_counts, [condition(clang_available)]) :-
    sdir(Dir),
    build_run_sorted(Dir, 'hc', "{ c[$1,$2]++ } END { for (k in c) print c[k] }\n",
        "a x\na x\nb y\na x\nb y\n", Lines),
    assertion(Lines == ["2", "3"]), !.

% SUBSEP separates the subscripts: (a,bc) and (ab,c) are DISTINCT keys (each
% counts 1). A naive separator-less concat would collide both to "abc" (one 2).
test(subsep_separates, [condition(clang_available)]) :-
    sdir(Dir),
    build_run_sorted(Dir, 'sep', "{ c[$1,$2]++ } END { for (k in c) print c[k] }\n",
        "a bc\nab c\n", Lines),
    assertion(Lines == ["1", "1"]), !.

% The default SUBSEP is the 0x1C byte: the key for (a,b) is "a\x1cb".
test(default_subsep_byte, [condition(clang_available)]) :-
    sdir(Dir),
    build_run(Dir, 'db', "{ c[$1,$2]++ } END { for (k in c) print k }\n",
        "a b\n", Out),
    assertion(Out == "a\x1c\b\n"), !.

% `BEGIN { SUBSEP = "-" }` overrides the join byte: the key is "a-b".
test(subsep_override, [condition(clang_available)]) :-
    sdir(Dir),
    build_run(Dir, 'ov',
        "BEGIN { SUBSEP = \"-\" } { c[$1,$2]++ } END { for (k in c) print k }\n",
        "a b\n", Out),
    assertion(Out == "a-b\n"), !.

% An empty SUBSEP joins the subscripts adjacent: the key is "ab".
test(subsep_empty, [condition(clang_available)]) :-
    sdir(Dir),
    build_run(Dir, 'es',
        "BEGIN { SUBSEP = \"\" } { c[$1,$2]++ } END { for (k in c) print k }\n",
        "a b\n", Out),
    assertion(Out == "ab\n"), !.

% Three-subscript keys work (any arity, via @wam_intern_subsep_key_n). The
% running count for a repeated (a,b,c) advances; a distinct third field is its
% own key.
test(three_dim_counter_read, [condition(clang_available)]) :-
    sdir(Dir),
    build_run(Dir, 'c3',
        "{ c[$1,$2,$3]++; print c[$1,$2,$3] }\n",
        "a b c\na b c\na b d\n", Out),
    assertion(Out == "1\n2\n1\n"), !.

% Three-dim histogram via END for-in over the joined keys (compare counts as a
% sorted set, since the joined key contains the raw SUBSEP byte).
test(three_dim_histogram, [condition(clang_available)]) :-
    sdir(Dir),
    build_run_sorted(Dir, 'h3',
        "{ c[$1,$2,$3]++ } END { for (k in c) print c[k] }\n",
        "a b c\na b c\na b d\n", Lines),
    assertion(Lines == ["1", "2"]), !.

% Three-dim membership: `($i,$j,$k) in arr` sees a repeated tuple.
test(three_dim_membership, [condition(clang_available)]) :-
    sdir(Dir),
    build_run(Dir, 'm3',
        "($1,$2,$3) in seen { print \"dup\" } { seen[$1,$2,$3]++ }\n",
        "a b c\na b c\nx y z\n", Out),
    assertion(Out == "dup\n"), !.

% Four subscripts work the same way (no per-arity special case).
test(four_dim_counter_read, [condition(clang_available)]) :-
    sdir(Dir),
    build_run(Dir, 'c4',
        "{ c[$1,$2,$3,$4]++; print c[$1,$2,$3,$4] }\n",
        "a b c d\na b c d\n", Out),
    assertion(Out == "1\n2\n"), !.

% --- literal subscript components -------------------------------------------

% A string-literal component mixes with a field: `c[$1,"x"]`. The literal
% contributes its own bytes, so the key varies only with $1.
test(string_component_counter_read, [condition(clang_available)]) :-
    sdir(Dir),
    build_run(Dir, 'ls', "{ c[$1,\"x\"]++; print c[$1,\"x\"] }\n",
        "a\na\nb\n", Out),
    assertion(Out == "1\n2\n1\n"), !.

% Histogram over literal-bearing keys: (a,x) x2 and (b,x) x1.
test(string_component_histogram, [condition(clang_available)]) :-
    sdir(Dir),
    build_run_sorted(Dir, 'lh', "{ c[$1,\"x\"]++ } END { for (k in c) print c[k] }\n",
        "a\na\nb\n", Lines),
    assertion(Lines == ["1", "2"]), !.

% An INTEGER-literal component is keyed by its decimal text, so `c[$1,5]` and
% `c[$1,"5"]` are the SAME key (awk array keys are strings): both increments land
% on one entry, which reads back as 4 after two records.
test(int_component_is_decimal_text, [condition(clang_available)]) :-
    sdir(Dir),
    build_run(Dir, 'li', "{ c[$1,5]++; c[$1,\"5\"]++; print c[$1,5] }\n",
        "a\na\n", Out),
    assertion(Out == "2\n4\n"), !.

% An all-literal key is one constant key: every record bumps the same entry.
test(all_literal_key, [condition(clang_available)]) :-
    sdir(Dir),
    build_run(Dir, 'la', "{ c[\"a\",\"b\"]++ } END { for (k in c) print c[k] }\n",
        "x\ny\nz\n", Out),
    assertion(Out == "3\n"), !.

% Literals and fields interleave at any position/arity: `c[$1,"m",$2]`.
test(mixed_field_literal_three, [condition(clang_available)]) :-
    sdir(Dir),
    build_run_sorted(Dir, 'lm', "{ c[$1,\"m\",$2]++ } END { for (k in c) print c[k] }\n",
        "a p\na p\na q\n", Lines),
    assertion(Lines == ["1", "2"]), !.

% A literal component builds the same bytes as a field holding that text, so
% `c[$1,"x"]` and `c[$1,$2]` collide when $2 is "x" (one key, count 2).
test(literal_matches_equal_field, [condition(clang_available)]) :-
    sdir(Dir),
    build_run(Dir, 'lf', "{ c[$1,\"x\"]++; c[$1,$2]++ } END { for (k in c) print c[k] }\n",
        "a x\n", Out),
    assertion(Out == "2\n"), !.

% Membership with a literal component probes the same key the counter builds.
test(literal_component_membership, [condition(clang_available)]) :-
    sdir(Dir),
    build_run(Dir, 'lmem',
        "($1,\"x\") in seen { print \"dup\" } { seen[$1,\"x\"]++ }\n",
        "a\na\nb\n", Out),
    assertion(Out == "dup\n"), !.

% A SCALAR-VARIABLE component is a clean not-yet: its value is not a
% compile-time constant, so it cannot ride the constant descriptor array. The
% program is rejected rather than miscompiled.
test(var_subscript_rejected, [condition(clang_available)]) :-
    sdir(Dir),
    build_status(Dir, 'rv', "{ k = \"z\"; c[$1,k]++ }\n", St),
    assertion(St \== 0), !.

:- end_tests(plawk_subsep).

% --- helpers ---------------------------------------------------------------

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

sdir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_subsep', Dir),
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

build_run(Dir, Name, Src, Input, Out) :-
    write_prog(Dir, Name, Src, Bin-Prog),
    process_create(path(swipl), ['examples/plawk/bin/plawk', build, Prog, '-o', Bin],
        [stdout(null), stderr(null), process(BPid)]),
    process_wait(BPid, exit(0)),
    process_create(Bin, ['-'],
        [stdin(pipe(In)), stdout(pipe(RS)), stderr(std), process(RPid)]),
    format(In, "~w", [Input]),
    close(In),
    read_string(RS, _, Out),
    close(RS),
    process_wait(RPid, exit(0)).

build_run_sorted(Dir, Name, Src, Input, SortedLines) :-
    build_run(Dir, Name, Src, Input, Out),
    split_string(Out, "\n", "", Parts0),
    exclude(==(""), Parts0, Parts),
    msort(Parts, SortedLines).
