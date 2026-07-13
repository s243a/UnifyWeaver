:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% Per-key aggregation across passes: associative add-assign `arr[$k] += $v`
% folds a per-key sum into a table in pass 1; pass 2 reads it back per record
% (`print $1, total[$1]`). This is the per-key counterpart to grand-total
% normalise -- each record is annotated with the total for ITS key, not the
% grand total. `arr[$k] += 1` is the general form of the counted `arr[$k]++`.
% (Fractional per-key normalise -- `$2 / total[$1]` -- rides on top of this
% once arithmetic operands can be table lookups.) See PLAWK_MULTIPASS_CACHE.md.

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(filesex), [make_directory_path/1]).

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

:- begin_tests(plawk_perkey).

% Per-key sum: pass 1 folds `total[$1] += $2`, pass 2 annotates each record
% with its key's total. Over a=10, a=20, b=5: a->30, b->5.
test(perkey_sum, [condition(clang_available)]) :-
    kdir(Dir),
    Src = "pass { total[$1] += $2 }\npass { print $1, total[$1] }\n",
    run_sorted(Dir, 'pks', Src, "a 10\na 20\nb 5\n", S),
    assertion(S == ["a 30", "a 30", "b 5"]),
    !.

% `arr[$k] += 1` is the general form of `arr[$k]++`: a per-key count folded
% in pass 1, annotated per record in pass 2. Over a a b: a->2, b->1.
test(perkey_count_via_add, [condition(clang_available)]) :-
    kdir(Dir),
    Src = "pass { c[$1] += 1 }\npass { print $1, c[$1] }\n",
    run_sorted(Dir, 'pkc', Src, "a\na\nb\n", S),
    assertion(S == ["a 2", "a 2", "b 1"]),
    !.

% Single-pass add-assign folds a per-key sum, dumped in END -- the add-assign
% chain must not break the pure-assoc route.
test(single_pass_add_assign, [condition(clang_available)]) :-
    kdir(Dir),
    Src = "{ total[$1] += $2 }\nEND { for (k in total) print k, total[k] }\n",
    run_sorted(Dir, 'spa', Src, "a 10\na 20\nb 5\n", S),
    assertion(S == ["a 30", "b 5"]),
    !.

% Fractional per-key normalise: a table lookup as an arithmetic operand in a
% print (`$2 / total[$1]`), evaluated in f64. Each record is divided by its
% OWN key's total. Over a=10,30 (total 40) and b=5 (total 5):
% 0.25, 0.75, 1.
test(perkey_normalise, [condition(clang_available)]) :-
    kdir(Dir),
    Src = "pass { total[$1] += $2 }\npass { print $1, $2 / total[$1] }\n",
    run_sorted(Dir, 'pkn', Src, "a 10\na 30\nb 5\n", S),
    assertion(S == ["a 0.25", "a 0.75", "b 1"]),
    !.

:- end_tests(plawk_perkey).

% --- helpers ---------------------------------------------------------------

kdir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_perkey', Dir),
    ( exists_directory(Dir) -> true ; make_directory_path(Dir) ).

run_sorted(Dir, Name, Src, Input, Sorted) :-
    directory_file_path(Dir, Name, Prog0),
    atom_concat(Prog0, '.plawk', Prog),
    setup_call_cleanup(open(Prog, write, S, [encoding(utf8)]),
        write(S, Src), close(S)),
    atom_concat(Prog0, '_bin', Bin),
    cli([build, Prog, '-o', Bin], 0),
    atom_concat(Prog0, '_in.txt', In),
    setup_call_cleanup(open(In, write, SI, [encoding(utf8)]),
        write(SI, Input), close(SI)),
    process_create(Bin, [In], [stdout(pipe(PS)), stderr(std), process(Pid)]),
    read_string(PS, _, Out), close(PS), process_wait(Pid, exit(0)),
    split_string(Out, "\n", "", L0), exclude(==(""), L0, L), msort(L, Sorted).

cli(Args, ExpectedStatus) :-
    process_create(path(swipl), ['examples/plawk/bin/plawk' | Args],
        [stdout(pipe(S)), stderr(std), process(Pid)]),
    read_string(S, _, _), close(S),
    process_wait(Pid, exit(Status)),
    assertion(Status == ExpectedStatus).
