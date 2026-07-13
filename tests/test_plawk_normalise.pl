:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% Grand-total normalise: the canonical two-pass shape that motivated the
% whole multi-pass arc. Pass 1 folds a grand total into a cross-pass scalar
% (`total += $N`); pass 2 prints each record divided by that total
% (`print $1, $N / total`). Two things had to land for this to be faithful:
%   * a pure-scalar (no assoc table) multi-pass program, and
%   * arithmetic in a per-record print evaluated in f64 -- the surface `/`
%     is integer (div_i64), which would truncate every ratio to 0, so a
%     print arithmetic expression is computed as double and printed with %g.
% See PLAWK_MULTIPASS_CACHE.md.

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(filesex), [make_directory_path/1]).

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

:- begin_tests(plawk_normalise).

% Fractional normalise, no assoc table: sum field 2 in pass 1, print each
% record's share in pass 2. Over 10/20/5 (total 35): 0.285714, 0.571429,
% 0.142857.
test(normalise_fraction, [condition(clang_available)]) :-
    ndir(Dir),
    Src = "pass { total += $2 }\npass { print $1, $2 / total }\n",
    run_sorted(Dir, 'norm', Src, "a 10\na 20\nb 5\n", S),
    assertion(S == ["a 0.285714", "a 0.571429", "b 0.142857"]),
    !.

% Pure-scalar grand total (no table, no arithmetic): pass 1 sums, pass 2
% annotates each record with the total. Over 10/20/5: 35 on every line.
test(grand_total_scalar, [condition(clang_available)]) :-
    ndir(Dir),
    Src = "pass { total += $2 }\npass { print $1, total }\n",
    run_sorted(Dir, 'gt', Src, "a 10\na 20\nb 5\n", S),
    assertion(S == ["a 35", "a 35", "b 35"]),
    !.

% Arithmetic against a constant in a print (promotes the int to double):
% `$2 / 2` halves each value with a fractional result.
test(print_arith_const, [condition(clang_available)]) :-
    ndir(Dir),
    Src = "pass { total += $2 }\npass { print $1, $2 / 2 }\n",
    run_sorted(Dir, 'half', Src, "a 3\nb 5\n", S),
    assertion(S == ["a 1.5", "b 2.5"]),
    !.

:- end_tests(plawk_normalise).

% --- helpers ---------------------------------------------------------------

ndir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_normalise', Dir),
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
