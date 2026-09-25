:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% BEGIN initial values for user scalars: `BEGIN { n = 1 }`.
%
% ---------------------------------------------------------------------------
% BEGIN accepted only special-variable string assignments (FS, OFS, ...), so
% `BEGIN { n = 10 }` was a parse error. `NAME = INT` now parses, and codegen LIFTS it
% out of BEGIN before any driver sees it (some drivers ignore an unfamiliar BEGIN
% action, which would silently drop the value) and makes it the SEED of the
% scalar's loop-header phi instead of the type zero. A seeded scalar is never
% "unset", so it leaves unset tracking.
%
% The guarantee is checked on the output: each seeded phi carries a
% `; plawk-begin-init(NAME)` marker, and the program is accepted only if EVERY lifted
% name has one. What cannot be seeded declines (exit 3) instead of printing the type
% zero where awk prints the BEGIN value:
%   - a string / strnum scalar (`max = $2`): its value is a runtime atom id;
%   - a driver whose scalars do not start at that phi (mixed scalar+assoc, END-only);
%   - an assignment that cannot be lifted order-independently (assigned twice in
%     BEGIN, or read by another BEGIN statement).
% A name nothing outside BEGIN mentions is unobservable and is simply dropped.
%
% gawk is the oracle (verified against gawk 5.1.0, LC_ALL=C).

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(lists)).
:- use_module(library(filesex), [make_directory_path/1]).
:- use_module('../examples/plawk/parser/plawk_parser').
:- use_module('../examples/plawk/codegen/llvm/plawk_native_codegen').

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

input("a 5\nb 7\nc 2\n").

:- begin_tests(plawk_begin_scalar_init).

% --- parsing ---------------------------------------------------------------

test(user_int_assignment_parses) :-
    plawk_parse_string("BEGIN { n = 10 }\n{ n++ }\n",
        program([begin([set(var(n), int(10))])], _, [])),
    !,
    plawk_parse_string("BEGIN {\n  total = -3\n  FS = \":\"\n}\n{ total++ }\n",
        program([begin([set(var(total), int(-3)), set(var('FS'), string(":"))])],
            _, [])),
    !.

% A special name is never taken as a user scalar.
test(special_names_are_not_user_scalars) :-
    \+ plawk_parse_string("BEGIN { NR = 1 }\n{ print }\n", _),
    !.

% --- runtime parity ----------------------------------------------------------

test(counter_starts_at_the_begin_value, [condition(clang_available)]) :-
    run("BEGIN { n = 10 }\n{ n++ }\nEND { print n }\n", "13\n"),
    !,
    run("BEGIN { total = -3 }\n{ total += $2 }\nEND { print total }\n", "11\n"),
    !.

% No records: END sees the BEGIN value, not 0 and not empty.
test(empty_input_keeps_the_begin_value, [condition(clang_available)]) :-
    run_with("", "BEGIN { n = 10 }\n{ n++ }\nEND { print n }\n", "10\n"),
    !.

test(double_slot, [condition(clang_available)]) :-
    run("BEGIN { s = 1 }\n{ s += $2 * 0.5 }\nEND { print s }\n", "8\n"),
    !.

test(read_in_the_rule_body, [condition(clang_available)]) :-
    run("BEGIN { c = 100 }\n{ print $1, c; c-- }\n", "a 100\nb 99\nc 98\n"),
    !,
    run("BEGIN { n = 2 }\n{ if (n > 0) { print $1; n-- } }\n", "a\nb\n"),
    !.

test(with_special_assignments_and_patterns, [condition(clang_available)]) :-
    run("BEGIN { FS = \" \"; n = 2 }\n$2 > n { n = n + 1 }\nEND { print n }\n",
        "4\n"),
    !.

% A name nothing outside BEGIN mentions is dropped, not required to be seeded.
test(unused_begin_scalar_is_dropped, [condition(clang_available)]) :-
    run("BEGIN { n = 7 }\n{ print $1 }\n", "a\nb\nc\n"),
    !.

% --- the guarantee is on the IR -------------------------------------------------

test(seeded_phi_is_marked) :-
    ir_of("BEGIN { n = 10 }\n{ n++ }\nEND { print n }\n", IR),
    sub_atom(IR, _, _, _,
        'phi i64 [10, %check_handle_value], [%next_slot_0, %continue_loop] ; plawk-begin-init(n)'),
    !.

test(no_marker_without_begin_values) :-
    ir_of("{ n++ }\nEND { print n }\n", IR),
    \+ sub_atom(IR, _, _, _, 'plawk-begin-init'),
    !.

% --- what cannot be seeded declines (never prints the type zero) ------------

% String / strnum slots, which declined until their seed could be an atom id: a
% second compile attempt interns the value's text in the entry block
% (begin_seed/2) and seeds the phi from it. `BEGIN { m = 9 } { m = $1 }` was the
% case that proved the IR check has teeth -- with the check disabled it built and
% printed "" on empty input where gawk prints 9 -- and now prints 9.
test(text_slot_seeds, [condition(clang_available)]) :-
    run("BEGIN { max = 0 }\n$2 > max { max = $2 }\nEND { print max }\n", "7\n"),
    !,
    run_with("", "BEGIN { max = 0 }\n$2 > max { max = $2 }\nEND { print max }\n", "0\n"),
    !,
    run("BEGIN { m = 9 }\n{ m = $1 }\nEND { print m }\n", "c\n"),
    !,
    run_with("", "BEGIN { m = 9 }\n{ m = $1 }\nEND { print m }\n", "9\n"),
    !,
    run_with("", "BEGIN { m = \"none\" }\n{ if ($2 > 100) m = $1 }\nEND { print m }\n",
        "none\n"),
    !,
    run("BEGIN { s = \"x\" }\n{ s = s $1 }\nEND { print s }\n", "xabc\n"),
    !.

% A BEGIN constant the program only prints is substituted as a literal.
test(print_only_constants, [condition(clang_available)]) :-
    run("BEGIN { sep = \"-\" }\n{ print $1 sep $2 }\n", "a-5\nb-7\nc-2\n"),
    !,
    run("BEGIN { label = \"row\" }\n{ print label, NR }\n", "row 1\nrow 2\nrow 3\n"),
    !,
    % reassigned in a rule: not a constant -- it takes the seeded-slot path
    run("BEGIN { sep = \"-\" }\n{ sep = \":\"; print $1 sep $2 }\n",
        "a:5\nb:7\nc:2\n"),
    !.

test(unseedable_declines) :-
    forall(member(Src,
            [ % assigned twice in BEGIN: not order-independent
              "BEGIN { n = 1; n = 2 }\n{ n++ }\nEND { print n }\n",
              % read by another BEGIN statement
              "BEGIN { x = 5; print x }\n",
              % mixed scalar+assoc driver
              "BEGIN { n = 1 }\n{ c[$1]++; n++ }\nEND { print n }\n",
              % END-only
              "BEGIN { n = 0 }\nEND { print n }\n"
            ]),
        build_status(Src, 3)),
    !.

:- end_tests(plawk_begin_scalar_init).

% --- helpers ---------------------------------------------------------------

odir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_begin_scalar_init', Dir),
    ( exists_directory(Dir) -> true ; make_directory_path(Dir) ).

run(Src, Expected) :-
    input(Input),
    run_with(Input, Src, Expected).

run_with(Input, Src, Expected) :-
    odir(Dir),
    directory_file_path(Dir, 'bsi_bin', Bin),
    % A build that unexpectedly declines must not silently run a stale binary.
    ( exists_file(Bin) -> delete_file(Bin) ; true ),
    directory_file_path(Dir, 'bsi', Prog0),
    atom_concat(Prog0, '.plawk', Prog),
    setup_call_cleanup(open(Prog, write, S, [encoding(utf8)]),
        write(S, Src), close(S)),
    atom_concat(Prog0, '_in.txt', In),
    setup_call_cleanup(open(In, write, SI, [encoding(utf8)]),
        write(SI, Input), close(SI)),
    build_status_of(Prog, Bin, 0),
    process_create(Bin, [In], [stdout(pipe(PS)), stderr(std), process(Pid)]),
    read_string(PS, _, Out),
    close(PS),
    process_wait(Pid, exit(0)),
    ( Out == Expected
    -> true
    ;  format(user_error, "~n~w~n  got      ~q~n  expected ~q~n",
           [Src, Out, Expected]), fail
    ).

build_status(Src, ExpectedStatus) :-
    odir(Dir),
    directory_file_path(Dir, 'bsi_status', Prog0),
    atom_concat(Prog0, '.plawk', Prog),
    atom_concat(Prog0, '_bin', Bin),
    setup_call_cleanup(open(Prog, write, S, [encoding(utf8)]),
        write(S, Src), close(S)),
    build_status_of(Prog, Bin, Status),
    ( Status == ExpectedStatus
    -> true
    ;  format(user_error, "~n~w~n  build exit ~w, expected ~w~n",
           [Src, Status, ExpectedStatus]), fail
    ).

build_status_of(Prog, Bin, Status) :-
    process_create(path(swipl), ['examples/plawk/bin/plawk', build, Prog, '-o', Bin],
        [stdout(pipe(O)), stderr(null), process(Pid)]),
    read_string(O, _, _), close(O),
    process_wait(Pid, exit(Status)).

ir_of(Src, IR) :-
    plawk_parse_string(Src, Program),
    plawk_program_native_driver_ir(Program, 'input.txt', IR).
