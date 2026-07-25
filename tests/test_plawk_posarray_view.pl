:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% plawk: the POSITIONAL-ARRAY property, asked consistently at every level.
%
% A table filled by `split` (or an `as array` bind) is keyed by integer
% POSITIONS 1..n, not by interned atom ids. Whether a table is positional is
% asked over three different term representations of the same program -- raw
% parser actions, per-rule action specs, and planned actions -- and those three
% views had drifted apart:
%
%   * the multi-pass `over TABLE` reader carried the program's str-valued
%     tables into its plan but not its positional ones, so the reader resolved
%     an integer position as an atom-registry id and printed raw bytes where
%     gawk prints 1, 2, 3;
%   * the raw-action walk behind the text-mode int-key gate recognised only
%     `as array` binds, so an integer END lookup on a split table
%     (`END { print p[1] }`) was declined even though its keys are positions;
%   * with both views agreeing, the two loop-key emitters (binary mode /
%     positional) stopped being mutually exclusive and the multi-pass driver's
%     one-function-per-pass check rejected the program outright.
%
% Producers now live in one table (plawk_posarray_producer/3), so all three
% levels see the same set. gawk 5.2 is the oracle for every expectation here.

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(filesex), [make_directory_path/1]).
:- use_module('../examples/plawk/parser/plawk_parser').
:- use_module('../examples/plawk/codegen/plawk_native_codegen').

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

:- begin_tests(plawk_posarray_view).

% --- the `over` reader over a positional table ------------------------------

% Keys are positions: gawk's `for (k in parts) print k, parts[k]` over a
% 3-piece split gives 1 a / 2 b / 3 c. The reader used to print the raw bytes
% of ids 1..3 instead.
test(over_split_table_prints_positional_keys, [condition(clang_available)]) :-
    odir(Dir),
    Src = "pass { split($0, parts, \",\") }\n\c
           pass over parts as k { print k, parts[k] }\n",
    run_sorted(Dir, 'pv_ovsplit', Src, "a,b,c\n", S),
    assertion(S == ["1 a", "2 b", "3 c"]),
    !.

% The value alone still resolves to the piece's text.
test(over_split_table_prints_values, [condition(clang_available)]) :-
    odir(Dir),
    Src = "pass { split($0, parts, \",\") }\n\c
           pass over parts as k { print parts[k] }\n",
    run_sorted(Dir, 'pv_ovsplitval', Src, "a,b,c\n", S),
    assertion(S == ["a", "b", "c"]),
    !.

% Regression: a table keyed by STRINGS (a counter) must still resolve its key
% ids back to text in the same reader.
test(over_string_keyed_table_still_resolves_keys, [condition(clang_available)]) :-
    odir(Dir),
    Src = "pass { c[$1]++ }\npass over c as k { print k, c[k] }\n",
    run_sorted(Dir, 'pv_ovstr', Src, "a x\nb y\na z\n", S),
    assertion(S == ["a 2", "b 1"]),
    !.

% IR shape: the positional reader prints the key with the i64 format and never
% resolves it through the atom registry.
test(over_split_reader_ir_prints_key_numerically) :-
    plawk_parse_string(
        "pass { split($0, parts, \",\") }\n\c
         pass over parts as k { print k, parts[k] }\n", Program),
    plawk_program_multipass_driver_ir(Program, DriverIR),
    assertion(once(sub_atom(DriverIR, _, _, _,
        '@printf(i8* %forin_key_fmt_0, i64 %forin_key_id)'))),
    assertion(\+ sub_atom(DriverIR, _, _, _,
        '@wam_atom_to_string(i64 %forin_key_id)')),
    !.

% The two loop-key emitters must be mutually exclusive: when both a binary
% descriptor and a positional plan could apply the emitter offered two
% identical solutions, and the multi-pass driver's one-function-per-pass check
% then failed the whole program. One solution, always.
test(over_split_reader_driver_is_deterministic) :-
    plawk_parse_string(
        "pass { split($0, parts, \",\") }\n\c
         pass over parts as k { print k, parts[k] }\n", Program),
    aggregate_all(count,
        plawk_program_multipass_driver_ir(Program, _IR), N),
    assertion(N == 1),
    !.

% --- integer END lookups on a positional table -----------------------------

% `END { print p[N] }` on a split table: the keys are genuine positions, so an
% integer literal cannot collide with an atom id and the lookup is safe in text
% mode. This used to be declined (exit 3) because the raw-action walk behind
% the gate only knew about `as array` binds.
test(end_int_lookup_on_split_table, [condition(clang_available)]) :-
    odir(Dir),
    forall(member(N-Expect, [1-"a", 2-"b", 3-"c"]),
        ( format(atom(Src),
              "{ split($0, p, \",\") }\nEND { print p[~w] }\n", [N]),
          format(atom(Name), 'pv_end~w', [N]),
          run_sorted(Dir, Name, Src, "a,b,c\n", S),
          assertion(S == [Expect])
        )),
    !.

% An ABSENT position is an uninitialised element: empty in string context, as
% gawk prints. (Resolving id 0 through the atom registry printed "(null)".)
test(end_int_lookup_absent_position_is_empty, [condition(clang_available)]) :-
    odir(Dir),
    Src = "{ split($0, p, \",\") }\nEND { print p[9] }\n",
    build(Dir, 'pv_endabsent', Src, Bin, In, "a,b,c\n"),
    run_out(Bin, In, Out),
    assertion(Out == "\n"),
    !.

% --- the shared producer table ---------------------------------------------

% Every level recognises the same producers. A split is positional whether it
% is seen as a raw action, as an action spec, or as a planned action -- these
% three used to be maintained independently.
test(split_is_positional_at_every_level) :-
    Rules = [rule(always, [split_into(field(0), var(parts), string(","))])],
    plawk_native_codegen:plawk_program_posarray_arrays(Rules, Surface),
    assertion(Surface == [parts]),
    maplist(plawk_native_codegen:plawk_assoc_rule_action_specs, Rules, Specs),
    plawk_native_codegen:plawk_assoc_specs_posarray_arrays(Specs, Spec),
    assertion(Spec == [parts]),
    plawk_native_codegen:plawk_assoc_plan_from_specs(Specs, [parts], Plan),
    assertion(plawk_native_codegen:plawk_assoc_plan_posarray_array(Plan, parts)),
    !.

% A plain counter is positional at NO level (its keys are interned strings).
test(counter_is_not_positional_at_any_level) :-
    Rules = [rule(always, [inc_assoc(var(c), field(1))])],
    plawk_native_codegen:plawk_program_posarray_arrays(Rules, Surface),
    assertion(Surface == []),
    maplist(plawk_native_codegen:plawk_assoc_rule_action_specs, Rules, Specs),
    plawk_native_codegen:plawk_assoc_specs_posarray_arrays(Specs, Spec),
    assertion(Spec == []),
    plawk_native_codegen:plawk_assoc_plan_from_specs(Specs, [c], Plan),
    assertion(\+ plawk_native_codegen:plawk_assoc_plan_posarray_array(Plan, c)),
    !.

% A reader plan has no writer of its own, so it carries the program's
% positional table names directly -- the same mechanism str_arrays/1 uses.
test(reader_plan_carries_positional_tables) :-
    Plan = assoc_plan([parts], [str_arrays([parts]), posarrays([parts])]),
    assertion(plawk_native_codegen:plawk_assoc_plan_posarray_array(Plan, parts)),
    assertion(plawk_native_codegen:plawk_assoc_plan_str_array(Plan, parts)),
    Bare = assoc_plan([parts], [str_arrays([parts])]),
    assertion(\+ plawk_native_codegen:plawk_assoc_plan_posarray_array(Bare, parts)),
    !.

:- end_tests(plawk_posarray_view).

% --- helpers ---------------------------------------------------------------

odir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_posarray_view', Dir),
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
    run_out(Bin, In, Out),
    split_string(Out, "\n", "", L0), exclude(==(""), L0, L), msort(L, Sorted).

run_out(Bin, In, Out) :-
    process_create(Bin, [In], [stdout(pipe(PS)), stderr(std), process(Pid)]),
    read_string(PS, _, Out), close(PS), process_wait(Pid, exit(0)).

cli(Args, ExpectedStatus) :-
    process_create(path(swipl), ['examples/plawk/bin/plawk' | Args],
        [stdout(pipe(S)), stderr(std), process(Pid)]),
    read_string(S, _, _), close(S),
    process_wait(Pid, exit(Status)),
    assertion(Status == ExpectedStatus).
