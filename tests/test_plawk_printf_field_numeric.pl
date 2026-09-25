:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% printf numeric conversions of a FIELD: `printf "%d", $2`, `"%5.1f"`, `"%x"`, ...
%
% ---------------------------------------------------------------------------
% A field reached printf as a slice, which only `%s` accepts, so every numeric
% conversion of a field declined (exit 3) -- `printf "%-8s %6.2f\n", $1, $2`, one of
% the most common printf lines there is. awk converts an argument to what its
% conversion asks for; plawk now does the same for fields:
%
%   - the format is pre-scanned for each argument's conversion class, with the SAME
%     scanner the format rewriter uses (plawk_printf_arg_classes/2);
%   - a field under a numeric conversion is read as a number, awk-style, through the
%     strtod coercion `float($N)` already lowers (plawk_printf_coerce_arg/4);
%   - under `%d`/`%x`/`%o`/`%u`/`%i` that double is truncated toward zero by the
%     runtime's @wam_awk_f64_to_i64 -- FATAL (exit 2) outside the i64 range, where a
%     bare fptosi is poison and gawk prints digits a fixed `%ld` cannot reproduce.
%
% Numeric text is DECIMAL, as in awk: @wam_awk_strtod rejects what C strtod would
% also accept -- "nan", "inf", hex "0x1A" -- reading them as 0.
%
% Only FIELD arguments are coerced. plawk's integer path still reads a non-integer
% field as 0 (`x = $2 + 0` on "30.25"); coercing those values too would turn today's
% clean decline of `x = $2 + 0; printf "%5.1f", x` into silent wrong output, so that
% stays a decline and is pinned here.
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

input("alice 30.25 NY 7\nbob 5 LA -2.9\ncarol 3abc SF x\n").

:- begin_tests(plawk_printf_field_numeric).

test(float_conversions, [condition(clang_available)]) :-
    run("{ printf \"%5.1f|\\n\", $2 }\n", " 30.2|\n  5.0|\n  3.0|\n"),
    !,
    run("{ printf \"%-8s %6.2f\\n\", $1, $2 }\n",
        "alice     30.25\nbob        5.00\ncarol      3.00\n"),
    !,
    run("{ printf \"%e\\n\", $2 }\n", "3.025000e+01\n5.000000e+00\n3.000000e+00\n"),
    !.

% `%d` truncates toward zero: 30.25 -> 30, -2.9 -> -2, "x" -> 0.
test(integer_conversions_truncate, [condition(clang_available)]) :-
    run("{ printf \"%d|\\n\", $2 }\n", "30|\n5|\n3|\n"),
    !,
    run("{ printf \"%5d|%x|%o\\n\", $4, $2, $2 }\n",
        "    7|1e|36\n   -2|5|5\n    0|3|3\n"),
    !.

test(mixed_with_other_arguments, [condition(clang_available)]) :-
    run("{ printf \"%d %s %.1f\\n\", NR, $1, $2 }\n",
        "1 alice 30.2\n2 bob 5.0\n3 carol 3.0\n"),
    !.

test(sprintf_shares_the_coercion, [condition(clang_available)]) :-
    run("{ t = sprintf(\"%d-%d\", $2, $4); print t }\n", "30-7\n5--2\n3-0\n"),
    !,
    run("{ t = sprintf(\"%05.1f\", $2); print t }\n", "030.2\n005.0\n003.0\n"),
    !.

% awk numeric text is decimal: "0x1A", "nan", "inf" read as 0; blanks and a sign
% are allowed; trailing text is ignored.
test(numeric_text_is_decimal, [condition(clang_available)]) :-
    run_with("a 12\nb 0x1A\nc nan\nd inf\ne  -7.9\nf .5x\n",
        "{ printf \"%d|%.2f\\n\", $2, $2 }\n",
        "12|12.00\n0|0.00\n0|0.00\n0|0.00\n-7|-7.90\n0|0.50\n", 0),
    !.

% Outside the i64 range `%d` is fatal (exit 2), never a wrong number.
test(out_of_range_integer_conversion_is_fatal, [condition(clang_available)]) :-
    run_with("a 5\nb 1e30\n", "{ printf \"%d\\n\", $2 }\n", "5\n", 2),
    !.

% --- the coercion is confined to fields ------------------------------------

% An integer-path value is NOT widened: plawk reads "30.25" there as 0, so this must
% keep declining rather than print 0.0.
test(integer_path_values_are_not_coerced) :-
    build_status("{ x = $2 + 0; printf \"%5.1f|\\n\", x }\n", 3),
    !,
    build_status("{ printf \"%.2f|\\n\", $2 * 1 }\n", 3),
    !.

% END printf of a field under a numeric conversion is not reached here (no
% retained-record numeric row): it must decline, not read the EOF sentinel.
test(end_numeric_field_declines) :-
    build_status("END { printf \"%d\\n\", $2 }\n", 3),
    !.

test(ir_uses_the_strtod_read_and_checked_truncation) :-
    ir_of("{ printf \"%d\\n\", $2 }\n", IR),
    sub_atom(IR, _, _, _, '@wam_atom_field_f64_value(%Value %line, i64 2'),
    sub_atom(IR, _, _, _, 'call i64 @wam_awk_f64_to_i64(double'),
    !.

:- end_tests(plawk_printf_field_numeric).

% --- helpers ---------------------------------------------------------------

odir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_printf_field_numeric', Dir),
    ( exists_directory(Dir) -> true ; make_directory_path(Dir) ).

run(Src, Expected) :-
    input(Input),
    run_with(Input, Src, Expected, 0).

run_with(Input, Src, Expected, ExpectedRC) :-
    odir(Dir),
    directory_file_path(Dir, 'pfn_bin', Bin),
    % A build that unexpectedly declines must not silently run a stale binary.
    ( exists_file(Bin) -> delete_file(Bin) ; true ),
    directory_file_path(Dir, 'pfn', Prog0),
    atom_concat(Prog0, '.plawk', Prog),
    setup_call_cleanup(open(Prog, write, S, [encoding(utf8)]),
        write(S, Src), close(S)),
    atom_concat(Prog0, '_in.txt', In),
    setup_call_cleanup(open(In, write, SI, [encoding(utf8)]),
        write(SI, Input), close(SI)),
    build_status_of(Prog, Bin, 0),
    process_create(Bin, [In], [stdout(pipe(PS)), stderr(null), process(Pid)]),
    read_string(PS, _, Out),
    close(PS),
    process_wait(Pid, exit(RC)),
    ( Out == Expected, RC == ExpectedRC
    -> true
    ;  format(user_error, "~n~w~n  got      ~q (exit ~w)~n  expected ~q (exit ~w)~n",
           [Src, Out, RC, Expected, ExpectedRC]), fail
    ).

build_status(Src, ExpectedStatus) :-
    odir(Dir),
    directory_file_path(Dir, 'pfn_status', Prog0),
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
