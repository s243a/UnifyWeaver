:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% Bare field-vs-field comparison patterns (`$N OP $M`).  Record fields are
% awk strnums: comparison is numeric when both operands look numeric and
% lexical otherwise.  These patterns use the ordinary pattern-combinator path,
% so they can be negated and combined with && / || and other base patterns.

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(filesex), [make_directory_path/1]).
:- use_module('../examples/plawk/parser/plawk_parser').
:- use_module('../examples/plawk/codegen/plawk_native_codegen').

:- begin_tests(plawk_field_cmp_pattern).

% --- parsing ----------------------------------------------------------------

test(gt_pattern_parses) :-
    plawk_parse_string("$1 > $2 { print $0 }\n",
        program([], [rule(field_cmp2(1, gt, 2), [print([field(0)])])], [])),
    !.

test(eq_pattern_parses) :-
    plawk_parse_string("$2 == $3 { print $1 }\n",
        program([], [rule(field_cmp2(2, eq, 3), [print([field(1)])])], [])),
    !.

test(le_pattern_parses) :-
    plawk_parse_string("$1 <= $4 { print $4 }\n",
        program([], [rule(field_cmp2(1, le, 4), [print([field(4)])])], [])),
    !.

test(all_comparison_operators_parse) :-
    forall(member(Token-Op,
                  ["=="-eq, "!="-ne, "<"-lt,
                   "<="-le, ">"-gt, ">="-ge]),
        ( format(string(Source), "$7 ~s $8 { print $0 }~n", [Token]),
          plawk_parse_string(Source,
              program([], [rule(field_cmp2(7, Op, 8),
                                [print([field(0)])])], []))
        )),
    !.

test(combinator_precedence_parses) :-
    plawk_parse_string("!($1 <= $2) || $3 != $4 && $5 == $6 { print $0 }\n",
        program([],
            [rule(or_pat(not_pat(field_cmp2(1, le, 2)),
                         and_pat(field_cmp2(3, ne, 4),
                                 field_cmp2(5, eq, 6))),
                  [print([field(0)])])], [])),
    !.

test(zero_left_index_rejected, [fail]) :-
    plawk_parse_string("$0 > $1 { print $0 }\n", _).

test(zero_right_index_rejected, [fail]) :-
    plawk_parse_string("$1 > $0 { print $0 }\n", _).

% Both lowering arities must use the bounded slice helper. Interning each field
% here would retain every distinct input value in the process-wide atom table.
test(guard_ir_uses_bounded_slices_without_interning) :-
    plawk_native_codegen:plawk_pattern_guard_ir(
        field_cmp2(1, gt, 2), 32, ''-SingleIR),
    assertion(sub_atom(SingleIR, _, _, _, '@wam_strnum_cmp_slices')),
    assertion(\+ sub_atom(SingleIR, _, _, _, '@wam_intern_atom')),
    assertion(sub_atom(SingleIR, _, _, _,
        '%plawk_surface_ffcmp_ok = icmp ne i32 %plawk_surface_ffcmp_rc, 2')),
    plawk_native_codegen:plawk_pattern_guard_ir(
        field_cmp2(2, le, 3), 32, cmp_probe, '%cmp_result', ''-MultiIR),
    assertion(sub_atom(MultiIR, _, _, _, '@wam_strnum_cmp_slices')),
    assertion(\+ sub_atom(MultiIR, _, _, _, '@wam_intern_atom')),
    assertion(sub_atom(MultiIR, _, _, _,
        '%cmp_result = and i1 %cmp_probe_ok, %cmp_probe_cmp')),
    !.

% --- runtime ----------------------------------------------------------------

% A single-rule program exercises the compact pattern-guard lowering.  The
% first match distinguishes numeric order (`10 > 9`) from lexical order.
test(single_rule_numeric_order, [condition(toolchain_available)]) :-
    Source = "$1 > $2 { print $0 }\n",
    Input = "10 9\n2 10\n01 1\nz a\n",
    build_run_matches_awk('single_numeric', Source, Input, Out),
    assertion(Out == "10 9\nz a\n"),
    !.

% Numeric-looking strings compare by numeric value (`01 == 1`); if either
% field is nonnumeric the same equality is lexical (`01 != 01x`).
test(strnum_equality, [condition(toolchain_available)]) :-
    Source = "$1 == $2 { print $0 }\n",
    Input = "01 1\n1 1\nabc abc\n01 01x\n",
    build_run_matches_awk('strnum_eq', Source, Input, Out),
    assertion(Out == "01 1\n1 1\nabc abc\n"),
    !.

% `10 > 9x` would be true under forced numeric conversion, but awk compares
% it lexically and rejects it because one side is not a numeric string.
test(lexical_fallback, [condition(toolchain_available)]) :-
    Source = "$1 > $2 { print $0 }\n",
    Input = "10 9x\n9x 10\nabc abd\nabd abc\n",
    build_run_matches_awk('lexical_fallback', Source, Input, Out),
    assertion(Out == "9x 10\nabd abc\n"),
    !.

% Multiple rules take the GlobalBase/MatchValue guard path.  Together these
% rules execute every comparison operator on less/equal/greater pairs.
test(multi_rule_all_operators, [condition(toolchain_available)]) :-
    Source = "$1 == $2 { print \"eq\" }\n\
$1 != $2 { print \"ne\" }\n\
$1 < $2 { print \"lt\" }\n\
$1 <= $2 { print \"le\" }\n\
$1 > $2 { print \"gt\" }\n\
$1 >= $2 { print \"ge\" }\n",
    Input = "2 10\n7 7\n10 2\n",
    build_run_matches_awk('multi_ops', Source, Input, Out),
    assertion(Out == "ne\nlt\nle\neq\nle\nge\nne\ngt\nge\n"),
    !.

% All three pattern combinators operate recursively on field_cmp2 guards.
test(not_and_or_combinators, [condition(toolchain_available)]) :-
    Source = "!($1 > $2) && ($3 == $4 || $5 < $6) { print $0 }\n",
    Input = "2 10 x x 9 1\n10 2 x y 1 9\n2 10 x y 1 9\n2 10 x y 9 1\n",
    build_run_matches_awk('combinators', Source, Input, Out),
    assertion(Out == "2 10 x x 9 1\n2 10 x y 1 9\n"),
    !.

% The new comparison composes with a pre-existing string-literal field guard.
test(combined_with_other_base_pattern, [condition(toolchain_available)]) :-
    Source = "$1 > $2 && $3 == \"ok\" { print $1 }\n",
    Input = "10 9 ok\n10 9 no\n2 10 ok\n",
    build_run_matches_awk('mixed_base', Source, Input, Out),
    assertion(Out == "10\n"),
    !.

% Both slices must use the active FS rather than assuming whitespace fields.
test(active_fs_for_both_fields, [condition(toolchain_available)]) :-
    Source = "BEGIN { FS = \":\" } $1 < $2 { print $0 }\n",
    Input = "2:10\n10:2\na:b\n",
    build_run_matches_awk('active_fs', Source, Input, Out),
    assertion(Out == "2:10\na:b\n"),
    !.

% Regex FS is represented by the FS sentinel 0 in codegen. Both projections
% must take that path, including records with adjacent separator matches.
test(regex_fs_for_both_fields, [condition(toolchain_available)]) :-
    Source = "BEGIN { FS = \"[,:]+\" } $1 < $2 { print $0 }\n",
    Input = "2::10\n10,,2\na,:b\n",
    build_run_matches_awk('regex_fs', Source, Input, Out),
    assertion(Out == "2::10\na,:b\n"),
    !.

% Out-of-range fields are empty strings. In particular, two missing fields
% compare equal without dereferencing either missing slice.
test(missing_fields_are_empty, [condition(toolchain_available)]) :-
    Source = "$2 == $3 { print $0 }\n",
    Input = "one\none two\none two two\n",
    build_run_matches_awk('missing_fields', Source, Input, Out),
    assertion(Out == "one\none two two\n"),
    !.

% The v1 lowering is textual-record-only. BINFMT has no integer FS and must
% fail at the documented compiler boundary instead of emitting a wrong guard.
test(binfmt_field_comparison_rejected, [condition(clang_available)]) :-
    Source = "BEGIN { BINFMT = \"i64 i64\" } $1 > $2 { print $1 }\n",
    sdir(Dir),
    build_status(Dir, 'binfmt_reject', Source, Status),
    assertion(Status == 3),
    !.

:- end_tests(plawk_field_cmp_pattern).

% --- helpers ----------------------------------------------------------------

toolchain_available :-
    clang_available,
    awk_available.

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

awk_available :-
    catch(( process_create(path(awk), ['BEGIN { exit 0 }'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

sdir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_field_cmp_pattern', Dir),
    ( exists_directory(Dir) -> true ; make_directory_path(Dir) ).

build_run_matches_awk(Name, Source, Input, Out) :-
    run_system_awk(Source, Input, AwkOut),
    sdir(Dir),
    build_run(Dir, Name, Source, Input, Out),
    assertion(Out == AwkOut).

run_system_awk(Source, Input, Out) :-
    atom_string(SourceAtom, Source),
    process_create(path(awk), [SourceAtom],
        [stdin(pipe(In)), stdout(pipe(RS)), stderr(std), process(Pid)]),
    format(In, "~s", [Input]),
    close(In),
    read_string(RS, _, Out),
    close(RS),
    process_wait(Pid, exit(0)).

build_run(Dir, Name, Source, Input, Out) :-
    directory_file_path(Dir, Name, Prog0),
    atom_concat(Prog0, '.plawk', Prog),
    setup_call_cleanup(open(Prog, write, S, [encoding(utf8)]),
        format(S, "~s", [Source]), close(S)),
    atom_concat(Prog0, '_bin', Bin),
    process_create(path(swipl), ['examples/plawk/bin/plawk', build, Prog, '-o', Bin],
        [stdout(null), stderr(null), process(BPid)]),
    process_wait(BPid, exit(0)),
    process_create(Bin, ['-'],
        [stdin(pipe(In)), stdout(pipe(RS)), stderr(std), process(RPid)]),
    format(In, "~s", [Input]),
    close(In),
    read_string(RS, _, Out),
    close(RS),
    process_wait(RPid, exit(0)).

build_status(Dir, Name, Source, Status) :-
    directory_file_path(Dir, Name, Prog0),
    atom_concat(Prog0, '.plawk', Prog),
    setup_call_cleanup(open(Prog, write, S, [encoding(utf8)]),
        format(S, "~s", [Source]), close(S)),
    atom_concat(Prog0, '_bin', Bin),
    process_create(path(swipl), ['examples/plawk/bin/plawk', build, Prog, '-o', Bin],
        [stdout(null), stderr(null), process(Pid)]),
    process_wait(Pid, exit(Status)).
