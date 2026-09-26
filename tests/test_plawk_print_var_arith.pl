:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% Arithmetic over scalar variables in a rule-body `print`: `{ n++; print n * 2 }`.
%
% ---------------------------------------------------------------------------
% `print n` worked and `y = n * 2; print y` worked, but `print n * 2` declined
% (exit 3): the rule-body print whitelist is checked on the SOURCE tree, where the
% read is still var(n), and the arithmetic operand vocabulary had no variable leaf.
% The reads are substituted with the current slot values before emission anyway
% (plawk_substitute_print_field/3), so admitting the leaf is enough -- in a
% print-only whitelist row, leaving the shared operand predicate variable-free.
%
% Two wrong-output paths surfaced and are closed here:
%   - A double-typed tree (a field or strnum operand makes it double) whose leaves
%     the f64 print path did not recognise fell through to the i64 binary print,
%     which re-reads the field with the strict integer parse: `print n + $2`
%     printed 2 for "7.5". The substituted leaves ssa/ssa_f64/ssa_strnum are now
%     f64 operands, AND the i64 binary print refuses any double-typed tree -- if
%     the f64 path cannot take it, the print declines.
%   - `x = $2; print x * 2`: the arithmetic read was not a strnum-safe read, so x
%     fell back to an i64 counter fed by the strict parse ("7.5" -> 0). Print
%     arithmetic is now a safe strnum read (substituted to ssa_strnum -> strtod).
%
% gawk is the oracle (verified against gawk 5.1.0, LC_ALL=C).

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(filesex), [make_directory_path/1]).

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

input("a 5\nb 7.5\nc 3abc\n").

:- begin_tests(plawk_print_var_arith).

test(counter_arithmetic, [condition(clang_available)]) :-
    run("{ n++; print n * 2 }\n", "2\n4\n6\n"),
    !,
    run("{ n++; print $1, n * 2 }\n", "a 2\nb 4\nc 6\n"),
    !,
    run("{ n++; print (n + 1) * (n - 1) }\n", "0\n3\n8\n"),
    !,
    run("{ n++; m = 3; print n * m }\n", "3\n6\n9\n"),
    !.

test(in_concat_and_printf, [condition(clang_available)]) :-
    run("{ n++; print \"n=\" n * 2 }\n", "n=2\nn=4\nn=6\n"),
    !,
    run("{ n++; printf \"%d|%s\\n\", n * 2, $1 }\n", "2|a\n4|b\n6|c\n"),
    !.

% A field operand makes the tree double: 7.5 is read by strtod, not as 0.
test(counter_plus_field, [condition(clang_available)]) :-
    run("{ n++; print n + $2 }\n", "6\n9.5\n6\n"),
    !.

test(double_scalar_arithmetic, [condition(clang_available)]) :-
    run("{ s += $2; print s / 2 }\n", "2.5\n6.25\n7.75\n"),
    !,
    run("{ s += $2; print s * 1.5 }\n", "7.5\n18.75\n23.25\n"),
    !.

% A strnum copy stays text when printed bare and is numeric under arithmetic.
test(strnum_arithmetic, [condition(clang_available)]) :-
    run("{ x = $2; print x * 2 }\n", "10\n15\n6\n"),
    !,
    run("{ x = $2; print x, x * 2 }\n", "5 10\n7.5 15\n3abc 6\n"),
    !,
    run("{ x = $2; print \"v=\" x + 1 }\n", "v=6\nv=8.5\nv=4\n"),
    !.

:- end_tests(plawk_print_var_arith).

% --- helpers ---------------------------------------------------------------

odir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_print_var_arith', Dir),
    ( exists_directory(Dir) -> true ; make_directory_path(Dir) ).

run(Src, Expected) :-
    input(Input),
    odir(Dir),
    directory_file_path(Dir, 'pva_bin', Bin),
    % A build that unexpectedly declines must not silently run a stale binary.
    ( exists_file(Bin) -> delete_file(Bin) ; true ),
    directory_file_path(Dir, 'pva', Prog0),
    atom_concat(Prog0, '.plawk', Prog),
    setup_call_cleanup(open(Prog, write, S, [encoding(utf8)]),
        write(S, Src), close(S)),
    atom_concat(Prog0, '_in.txt', In),
    setup_call_cleanup(open(In, write, SI, [encoding(utf8)]),
        write(SI, Input), close(SI)),
    process_create(path(swipl), ['examples/plawk/bin/plawk', build, Prog, '-o', Bin],
        [stdout(pipe(BO)), stderr(null), process(BPid)]),
    read_string(BO, _, _), close(BO),
    process_wait(BPid, exit(BuildStatus)),
    assertion(BuildStatus == 0),
    process_create(Bin, [In], [stdout(pipe(PS)), stderr(std), process(Pid)]),
    read_string(PS, _, Out),
    close(PS),
    process_wait(Pid, exit(0)),
    ( Out == Expected
    -> true
    ;  format(user_error, "~n~w~n  got      ~q~n  expected ~q~n",
           [Src, Out, Expected]), fail
    ).
