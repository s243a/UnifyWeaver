:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% Multi-dimensional array subscripts `arr[i,j]` and SUBSEP. In awk, `arr[i,j]`
% is sugar for `arr[i SUBSEP j]`: the subscripts are joined by SUBSEP (default
% "\034", the FS/0x1C byte) into one string key. v1 keys on exactly two field
% subscripts (`arr[$i,$j]`), covering the write counter `arr[$i,$j]++` and the
% element read `arr[$i,$j]`; for-in iteration sees the joined key.
%
% The join is done by the runtime helper @wam_intern_subsep_key2, which slices
% the two fields, joins them with the SUBSEP bytes (@wam_subsep_ptr /
% @wam_subsep_len, default 0x1C, overridable by `BEGIN { SUBSEP = "…" }`), and
% interns the result to one atom id -- the same key both the write and read
% paths build. Three-plus subscripts and non-field subscripts (string / var)
% are a clean not-yet (compile error), not miscompiled.

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

% A three-subscript key is a clean not-yet (v1 keys on two fields): rejected
% with a compile error rather than miscompiled.
test(three_dim_rejected, [condition(clang_available)]) :-
    sdir(Dir),
    build_status(Dir, 'r3', "{ c[$1,$2,$3]++ }\n", St),
    assertion(St == 3), !.

% A string subscript in a multi-dim key is a clean not-yet (v1 keys on fields).
test(string_subscript_rejected, [condition(clang_available)]) :-
    sdir(Dir),
    build_status(Dir, 'rs', "{ c[$1,\"x\"]++ }\n", St),
    assertion(St == 3), !.

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
