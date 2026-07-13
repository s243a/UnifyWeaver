:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% plawk multi-pass, phase 4: the `over TABLE` reader. Instead of re-scanning
% the input, a pass iterates a table's entries as its record source --
% `pass over TABLE as VAR { print ... }`. Fields are NAMED (name lookup): the
% loop key is bound to VAR, and the value is read as TABLE[VAR] (no positional
% $1/$2). So pass 1 accumulates a table and pass 2 emits ONE line per distinct
% key (not per input record), the "process what the previous pass stored"
% shape. Reuses the END for-in body emitter, so the printed key/value/
% separator are identical. Pairs with a cache-backed table (durable across
% runs) just like the input-scanning passes. See PLAWK_MULTIPASS_CACHE.md.

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(filesex), [make_directory_path/1]).
:- use_module('../examples/plawk/parser/plawk_parser').

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

:- begin_tests(plawk_over_table).

% `pass over T as V { ... }` parses to a pass_over/3 node, distinct from a
% plain input-scanning pass.
test(over_parses) :-
    plawk_parse_string(
        "pass { total[$1] += $2 }\npass over total as k { print k, total[k] }\n",
        program_passes([],
            [pass([rule(always, [add_assoc(var(total), field(1), field(2))])]),
             pass_over(var(k), var(total),
                 [print([var(k), assoc(var(total), var(k))])])],
            [])),
    !.

% Pass 1 sums per key; pass 2 iterates the table (one line per KEY, not per
% record) printing key + value. Over a=10,20 / b=5: a 30 / b 5 (two lines,
% not three).
test(over_key_value, [condition(clang_available)]) :-
    odir(Dir),
    Src = "pass { total[$1] += $2 }\npass over total as k { print k, total[k] }\n",
    run_sorted(Dir, 'ov', Src, "a 10\na 20\nb 5\n", S),
    assertion(S == ["a 30", "b 5"]),
    !.

% The key alone (name lookup of just the loop variable).
test(over_key_only, [condition(clang_available)]) :-
    odir(Dir),
    Src = "pass { c[$1]++ }\npass over c as k { print k }\n",
    run_sorted(Dir, 'ovk', Src, "a\na\nb\n", S),
    assertion(S == ["a", "b"]),
    !.

% An `over TABLE` reader over a cache-backed table persists across runs, like
% the input-scanning passes: run 1 -> a 10 / b 5; run 2 -> a 20 / b 10.
test(over_cache_persists, [condition(clang_available)]) :-
    odir(Dir),
    directory_file_path(Dir, 'ovc.db', Store),
    ( exists_file(Store) -> delete_file(Store) ; true ),
    format(atom(Src),
        "BEGIN cache(\"~w\") { declare total }\n\c
         pass { total[$1] += $2 }\n\c
         pass over total as k { print k, total[k] }\n", [Store]),
    build(Dir, 'ovc', Src, Bin, In, "a 10\nb 5\n"),
    run_sorted_bin(Bin, In, S1), assertion(S1 == ["a 10", "b 5"]),
    run_sorted_bin(Bin, In, S2), assertion(S2 == ["a 20", "b 10"]),
    !.

:- end_tests(plawk_over_table).

% --- helpers ---------------------------------------------------------------

odir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_over_table', Dir),
    ( exists_directory(Dir) -> true ; make_directory_path(Dir) ).

run_sorted(Dir, Name, Src, Input, Sorted) :-
    build(Dir, Name, Src, Bin, In, Input),
    run_sorted_bin(Bin, In, Sorted).

build(Dir, Name, Src, Bin, In, Input) :-
    directory_file_path(Dir, Name, Prog0),
    atom_concat(Prog0, '.plawk', Prog),
    setup_call_cleanup(open(Prog, write, S, [encoding(utf8)]),
        write(S, Src), close(S)),
    atom_concat(Prog0, '_bin', Bin),
    cli([build, Prog, '-o', Bin], 0),
    atom_concat(Prog0, '_in.txt', In),
    setup_call_cleanup(open(In, write, SI, [encoding(utf8)]),
        write(SI, Input), close(SI)).

run_sorted_bin(Bin, In, Sorted) :-
    process_create(Bin, [In], [stdout(pipe(PS)), stderr(std), process(Pid)]),
    read_string(PS, _, Out), close(PS), process_wait(Pid, exit(0)),
    split_string(Out, "\n", "", L0), exclude(==(""), L0, L), msort(L, Sorted).

cli(Args, ExpectedStatus) :-
    process_create(path(swipl), ['examples/plawk/bin/plawk' | Args],
        [stdout(pipe(S)), stderr(std), process(Pid)]),
    read_string(S, _, _), close(S),
    process_wait(Pid, exit(Status)),
    assertion(Status == ExpectedStatus).
