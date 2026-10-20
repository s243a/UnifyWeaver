:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% `/` and `%` by zero are fatal, as in awk: gawk prints "division by zero
% attempted" and exits 2, with every line printed before the division already
% out. plawk printed inf / -nan for `/` and 0 for `%` and exited 0 -- in a rule
% body, in END (`sum / NR` on empty input) and in BEGIN alike. A divisor that is
% runtime data now goes through @wam_awk_divisor_checked (double) or
% @wam_awk_divisor_checked_i64 (counter `%`); a nonzero literal divisor is not
% checked.
%
% Known difference: in a multi-item `print` whose LATER item divides by zero, the
% items before it are already written (`print $1, 10 / $2` leaves "b " where gawk
% leaves nothing); gawk evaluates every item first. The exit status is the same.
%
% gawk 5.1.0 (LC_ALL=C) gives every expected stdout and exit status below.

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(filesex), [make_directory_path/1]).
:- use_module('../examples/plawk/parser/plawk_parser').
:- use_module('../examples/plawk/codegen/llvm/plawk_native_codegen').

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

input(zero, "a 3\nb 0\nc 5\n").
input(empty, "").

:- begin_tests(plawk_div_zero).

test(rule_body_division_by_zero_is_fatal, [condition(clang_available)]) :-
    run(zero, "{ print 10 / $2 }\n", "3.33333\n", 2),
    run(zero, "{ x = 10 / $2; print x }\n", "3.33333\n", 2),
    run(zero, "{ t += 10 / $2 } END { print t }\n", "", 2),
    run(zero, "{ print $2 / 0 }\n", "", 2),
    run(zero, "{ print $2 % 0 }\n", "", 2),
    run(empty, "{ print 10 / $2 }\n", "", 0),
    !.

test(end_division_by_zero_is_fatal, [condition(clang_available)]) :-
    run(empty, "{ t += $2 } END { print t / NR }\n", "", 2),
    run(empty, "{ t += $2 } END { avg = t / NR; print avg }\n", "", 2),
    run(empty, "{ n++ } END { print 10 / n }\n", "", 2),
    run(empty, "{ t += $2 } END { print t % NR }\n", "", 2),
    run(zero, "{ n++; z = 0 } END { print n % z }\n", "", 2),
    run(zero, "{ s += $2 } END { printf \"%.2f\\n\", s / (NR - 3) }\n", "", 2),
    % a nonzero divisor is unaffected
    run(zero, "{ t += $2 } END { print t / NR, t % NR }\n", "2.66667 2\n", 0),
    run(zero, "{ n++ } END { print 10 / n }\n", "3.33333\n", 0),
    !.

test(begin_division_by_zero_is_fatal, [condition(clang_available)]) :-
    run(empty, "BEGIN { x = 0; print 5 % x }\n", "", 2),
    run(empty, "BEGIN { x = 0; print 5 / x }\n", "", 2),
    run(empty, "BEGIN { x = 4; print 10 / x, 10 % x }\n", "2.5 2\n", 0),
    !.

test(nonzero_divisors_unchanged, [condition(clang_available)]) :-
    run(zero, "{ print ($2 + 1) / ($2 - $2 + 1) }\n", "4\n1\n6\n", 0),
    run(zero, "{ n++; print n % 2 }\n", "1\n0\n1\n", 0),
    !.

% the element is the divisor: the line before it is out, the run stops with 2
test(forin_element_divisor_is_fatal, [condition(clang_available)]) :-
    run_status(zero, "{ c[$1] += $2 } END { for (k in c) print k, 10 / c[k] }\n", 2),
    !.

% a nonzero literal divisor is not checked; a runtime one is
test(literal_divisor_not_checked) :-
    plawk_parse_string("{ print $2 / 2, $2 % 3 }\n", P1),
    plawk_program_native_driver_ir(P1, 'input.txt', IR1),
    assertion(\+ sub_atom(IR1, _, _, _, 'call double @wam_awk_divisor_checked')),
    plawk_parse_string("{ print $2 / $3 }\n", P2),
    plawk_program_native_driver_ir(P2, 'input.txt', IR2),
    assertion(once(sub_atom(IR2, _, _, _, 'call double @wam_awk_divisor_checked'))),
    !.

:- end_tests(plawk_div_zero).

% --- helpers ---------------------------------------------------------------

odir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_div_zero', Dir),
    ( exists_directory(Dir) -> true ; make_directory_path(Dir) ).

build(Src, Bin, In, Input) :-
    odir(Dir),
    directory_file_path(Dir, 'dz_bin', Bin),
    ( exists_file(Bin) -> delete_file(Bin) ; true ),
    directory_file_path(Dir, 'dz.plawk', Prog),
    setup_call_cleanup(open(Prog, write, S, [encoding(utf8)]),
        write(S, Src), close(S)),
    directory_file_path(Dir, 'dz_in.txt', In),
    setup_call_cleanup(open(In, write, SI, [encoding(utf8)]),
        write(SI, Input), close(SI)),
    process_create(path(swipl),
        ['examples/plawk/bin/plawk', build, Prog, '-o', Bin],
        [stdout(null), stderr(null), process(BPid)]),
    process_wait(BPid, BuildStatus),
    assertion(BuildStatus == exit(0)).

% stdout and exit status as gawk gives them; a fatal exit also names the error
run(InputName, Src, Expected, ExpectedStatus) :-
    input(InputName, Input),
    build(Src, Bin, In, Input),
    process_create(Bin, [In], [stdout(pipe(PS)), stderr(pipe(PE)), process(Pid)]),
    read_string(PS, _, Out),
    read_string(PE, _, Err),
    close(PS),
    close(PE),
    process_wait(Pid, Exit),
    (   Out == Expected,
        Exit == exit(ExpectedStatus),
        ( ExpectedStatus =:= 0
        ->  Err == ""
        ;   sub_string(Err, _, _, _, "division by zero attempted")
        )
    ->  true
    ;   format(user_error, "~n~w~n  got      ~q ~w ~q~n  expected ~q exit(~w)~n",
            [Src, Out, Exit, Err, Expected, ExpectedStatus]),
        fail
    ).

run_status(InputName, Src, ExpectedStatus) :-
    input(InputName, Input),
    build(Src, Bin, In, Input),
    process_create(Bin, [In], [stdout(null), stderr(null), process(Pid)]),
    process_wait(Pid, Exit),
    assertion(Exit == exit(ExpectedStatus)).
