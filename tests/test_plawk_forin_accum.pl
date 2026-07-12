:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% Assoc for-in with per-entry logic, stage 2: SCALAR ACCUMULATION. An END
% `for (k in arr) acc += OPERAND` folds the final hash into a loop-carried
% scalar, and a trailing print reads the total -- the canonical "sum /
% count the histogram" idiom. Two pieces landed here: (1) a multi-action
% END that admits a for-in accumulate followed by a print (the END grammar
% otherwise takes exactly one action), and (2) a second loop phi in the END
% for-in loop that carries the accumulator across iterations. OPERAND is the
% iterated value `arr[k]`, the loop key `k`, or an integer literal. See
% PLAWK_ASSOC_FORIN.md.

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(filesex), [make_directory_path/1]).
:- use_module('../examples/plawk/parser/plawk_parser').
:- use_module('../examples/plawk/codegen/plawk_native_codegen').

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

:- begin_tests(plawk_forin_accum).

% `acc += arr[k]` parses to a two-action END: the for-in accumulate over
% the value, then a print that reads the accumulator.
test(value_accum_parses) :-
    plawk_parse_string(
        "{ c[$1]++ }\n\c
         END { for (k in c) total += c[k] ; print total }\n",
        program(_, _, [end([for_in(var(k), var(c),
            [add(var(total), forin_val(c))]), print([var(total)])])])),
    !.

% `acc += k` folds the keys; `acc += 1` counts entries.
test(key_and_const_accum_parse) :-
    plawk_parse_string(
        "{ c[$1]++ }\nEND { for (k in c) s += k ; print s }\n",
        program(_, _, [end([for_in(var(k), var(c),
            [add(var(s), forin_key)]), print([var(s)])])])),
    plawk_parse_string(
        "{ c[$1]++ }\nEND { for (k in c) n += 1 ; print n }\n",
        program(_, _, [end([for_in(var(k), var(c),
            [add(var(n), int(1))]), print([var(n)])])])),
    !.

% The plain (single-action) END for-in still parses to its own shape --
% the two-action accumulate clause must not shadow it.
test(plain_forin_still_single_action) :-
    plawk_parse_string(
        "{ c[$1]++ }\nEND { for (k in c) print k, c[k] }\n",
        program(_, _, [end([for_in(var(k), var(c),
            [print([var(k), assoc(var(c), var(k))])])])])),
    !.

% Sum the values, end to end. Final hash a=3, b=2, c=1; total = 6.
test(value_sum_runs, [condition(clang_available)]) :-
    fa_dir(Dir),
    Src = "{ c[$1]++ }\nEND { for (k in c) total += c[k] ; print total }\n",
    build_run_text(Dir, 'sum', Src, "a\na\nb\na\nc\nb\n", Out),
    assertion(Out == "6\n"),
    !.

% Count entries: three distinct keys -> 3.
test(count_runs, [condition(clang_available)]) :-
    fa_dir(Dir),
    Src = "{ c[$1]++ }\nEND { for (k in c) n += 1 ; print n }\n",
    build_run_text(Dir, 'cnt', Src, "a\na\nb\na\nc\nb\n", Out),
    assertion(Out == "3\n"),
    !.

% A leading string literal prints beside the accumulator.
test(labeled_sum_runs, [condition(clang_available)]) :-
    fa_dir(Dir),
    Src = "{ c[$1]++ }\nEND { for (k in c) total += c[k] ; print \"sum\", total }\n",
    build_run_text(Dir, 'lbl', Src, "a\na\nb\na\nc\nb\n", Out),
    assertion(Out == "sum 6\n"),
    !.

% Empty hash: the loop runs zero iterations, the accumulator stays 0.
test(empty_hash_sum_zero, [condition(clang_available)]) :-
    fa_dir(Dir),
    Src = "/never/ { c[$1]++ }\nEND { for (k in c) total += c[k] ; print total }\n",
    build_run_text(Dir, 'emp', Src, "a\nb\n", Out),
    assertion(Out == "0\n"),
    !.

:- end_tests(plawk_forin_accum).

% --- helpers ---------------------------------------------------------------

fa_dir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_forin_accum', Dir),
    ( exists_directory(Dir) -> true ; make_directory_path(Dir) ).

build_run_text(Dir, Name, Src, InputText, Out) :-
    directory_file_path(Dir, Name, Prog0),
    atom_concat(Prog0, '.plawk', Prog),
    setup_call_cleanup(open(Prog, write, S, [encoding(utf8)]),
        write(S, Src), close(S)),
    atom_concat(Prog0, '_bin', Bin),
    cli([build, Prog, '-o', Bin], _, 0),
    atom_concat(Prog0, '_in.txt', Input),
    setup_call_cleanup(open(Input, write, SI, [encoding(utf8)]),
        write(SI, InputText), close(SI)),
    run_bin(Bin, [Input], Out, 0).

cli(Args, Out, ExpectedStatus) :-
    process_create(path(swipl), ['examples/plawk/bin/plawk' | Args],
        [stdout(pipe(S)), stderr(std), process(Pid)]),
    read_string(S, _, Out),
    close(S),
    process_wait(Pid, exit(Status)),
    assertion(Status == ExpectedStatus).

run_bin(Bin, Args, Out, ExpectedStatus) :-
    process_create(Bin, Args,
        [stdout(pipe(S)), stderr(std), process(Pid)]),
    read_string(S, _, Out),
    close(S),
    process_wait(Pid, exit(Status)),
    assertion(Status == ExpectedStatus).
