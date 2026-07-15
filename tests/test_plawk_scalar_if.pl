:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% plawk scalar `if` conditions (PLAWK_CONTROL_FLOW_PLAN.md 3b). An `if` condition
% may be a SCALAR comparison over variables (`if (i > 2)`, `if (i < n && j > 0)`)
% -- the same `VAR CMP int/VAR` shape as a loop condition, wrapped as
% scalar_if(_) so the lowering reads slot values -- in addition to the existing
% field/pattern guard (`$1 > 2`, `$0 ~ /re/`). This unblocks counter-based
% conditionals inside loops (a prerequisite for loop-local break/continue).

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(filesex), [make_directory_path/1]).
:- use_module('../examples/plawk/parser/plawk_parser').

:- begin_tests(plawk_scalar_if).

% A scalar comparison parses to scalar_if(cmp(...)); a field comparison stays a
% field pattern (the two do not collide).
test(scalar_if_parses) :-
    plawk_parse_string("{ if (i > 2) { print i } }\n",
        program([], [rule(always,
            [if(scalar_if(cmp(var(i), gt, int(2))), [print([var(i)])], [])])], [])),
    !.

test(field_if_unchanged) :-
    plawk_parse_string("{ if ($1 > 2) { print $1 } }\n",
        program([], [rule(always,
            [if(field_cmp(1, gt, 2), [print([field(1)])], [])])], [])),
    !.

test(scalar_if_and_parses) :-
    plawk_parse_string("{ if (i > 2 && j < 5) { print i } }\n",
        program([], [rule(always,
            [if(scalar_if(and(cmp(var(i), gt, int(2)),
                              cmp(var(j), lt, int(5)))),
                [print([var(i)])], [])])], [])),
    !.

% --- runtime ----------------------------------------------------------------

% A scalar `if` on an accumulator, counted into another scalar read from END.
test(scalar_if_counter, [condition(clang_available)]) :-
    sdir(Dir),
    Src = "{ n++; if (n > 2) hits++ }\nEND { print hits }\n",
    build_run(Dir, 'sc', Src, "a\nb\nc\nd\n", Out, St),
    assertion(St == 0),
    assertion(Out == "2\n"),
    !.

% A scalar `if` guarding a print of a field.
test(scalar_if_guarded_field_print, [condition(clang_available)]) :-
    sdir(Dir),
    Src = "{ n++; if (n > 1) print $1 }\n",
    build_run(Dir, 'sp', Src, "x\ny\nz\n", Out, St),
    assertion(St == 0),
    assertion(Out == "y\nz\n"),
    !.

% `&&` over scalar comparisons.
test(scalar_if_and, [condition(clang_available)]) :-
    sdir(Dir),
    Src = "{ n++; if (n > 1 && n < 4) print $1 }\n",
    build_run(Dir, 'sa', Src, "a\nb\nc\nd\n", Out, St),
    assertion(St == 0),
    assertion(Out == "b\nc\n"),
    !.

% The headline enabler: a scalar `if` INSIDE a loop (guard on the loop counter).
test(scalar_if_inside_loop, [condition(clang_available)]) :-
    sdir(Dir),
    Src = "{ i = 0; while (i < 5) { if (i > 2) print i; i++ } }\n",
    build_run(Dir, 'sl', Src, "x\n", Out, St),
    assertion(St == 0),
    assertion(Out == "3\n4\n"),
    !.

% NOTE: a scalar `if` in an END block (`END { if (n > 1) ... }`) uses the
% separate END-print lowering, not the rule-body sequence walker, so it is not
% wired here -- a follow-on. Scalar `if` in rule bodies and loops (the loop
% control enabler) is what this PR delivers.

% A field/regex condition still compiles and runs (no regression).
test(field_regex_if_still_runs, [condition(clang_available)]) :-
    sdir(Dir),
    Src = "{ if ($0 ~ /err/) print $1 }\n",
    build_run(Dir, 'fr', Src, "err 1\nok 2\n", Out, St),
    assertion(St == 0),
    assertion(Out == "err\n"),
    !.

:- end_tests(plawk_scalar_if).

% --- helpers ---------------------------------------------------------------

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

sdir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_scalar_if', Dir),
    ( exists_directory(Dir) -> true ; make_directory_path(Dir) ).

build_run(Dir, Name, Src, Input, Out, RunStatus) :-
    directory_file_path(Dir, Name, Prog0),
    atom_concat(Prog0, '.plawk', Prog),
    setup_call_cleanup(open(Prog, write, S, [encoding(utf8)]),
        write(S, Src), close(S)),
    atom_concat(Prog0, '_bin', Bin),
    process_create(path(swipl), ['examples/plawk/bin/plawk', build, Prog, '-o', Bin],
        [stdout(null), stderr(null), process(BPid)]),
    process_wait(BPid, exit(0)),
    process_create(Bin, [],
        [stdin(pipe(In)), stdout(pipe(RS)), stderr(std), process(RPid)]),
    format(In, "~w", [Input]),
    close(In),
    read_string(RS, _, Out),
    close(RS),
    process_wait(RPid, exit(RunStatus)).
