:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% A field KEY past NF is awk's empty string, not "no key".
%
% `c[$3]++` on a two-field record counts c[""] in awk. The field-key emitters
% SKIPPED the action when the field slice was missing (a null pointer) -- the
% counted inc, `+=`, `delete`, row capture `r[$k] = $0`, and the mixed walker's
% inc -- so the "" entry was never created (or never deleted). Wrong output, e.g.
%     { c[$3]++ } END { for (k in c) print k, c[k] }      gawk " 5", plawk nothing
% Each site now swaps the null pointer for a per-site empty constant before the
% intern (the foreign-arg marshal's pattern), so the key is the "" atom.
%
% gawk is the oracle (verified against gawk 5.1.0, LC_ALL=C). for-in order is
% unspecified, so outputs are compared as sorted lines.

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

% an empty record, and a record with one field
input("a 1\nb 2\n\na 3\nb\n").

:- begin_tests(plawk_missing_field_key).

test(counted_inc_keys_the_empty_string, [condition(clang_available)]) :-
    run_sorted("{ c[$3]++ } END { for (k in c) print k, c[k] }\n", " 5\n"),
    !,
    run_sorted("{ c[$2]++ } END { for (k in c) print k, c[k] }\n",
        " 2\n1 1\n2 1\n3 1\n"),
    !.

test(add_assign_keys_the_empty_string, [condition(clang_available)]) :-
    run_sorted("{ c[$2] += 5 } END { for (k in c) print k, c[k] }\n",
        " 10\n1 5\n2 5\n3 5\n"),
    !.

% c[""] is created through a scalar key (`k = $2`, whose missing-field path
% already interned ""), then every record's `delete c[$3]` removes it; the skip
% left c[""] = 2 behind.
test(delete_of_a_missing_field_deletes_the_empty_key, [condition(clang_available)]) :-
    run_sorted("{ k = $2; c[k]++; delete c[$3] } END { for (x in c) print x, c[x] }\n",
        "1 1\n2 1\n3 1\n"),
    !.

test(row_capture_stores_under_the_empty_key, [condition(clang_available)]) :-
    run_sorted("{ r[$2] = $0 } END { for (k in r) print k, r[k] }\n",
        " b\n1 a 1\n2 b 2\n3 a 3\n"),
    !.

% The mixed walker's field-key inc takes the same path: no skip branch, the
% intern reads the safe pointer.
test(mixed_walker_inc_has_no_skip_branch) :-
    build_ll("{ c[$3]++; n++ } END { print n, c[\"1\"] }\n", LL),
    assertion(sub_string(LL, _, _, _, "_assoc_0_key_sptr = select i1")),
    assertion(\+ sub_string(LL, _, _, _, "br i1 %rule_0_body_assoc_0_key_missing")),
    !.

:- end_tests(plawk_missing_field_key).

% --- helpers ---------------------------------------------------------------

odir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_missing_field_key', Dir),
    ( exists_directory(Dir) -> true ; make_directory_path(Dir) ).

run_sorted(Src, Expected) :-
    input(Input),
    odir(Dir),
    directory_file_path(Dir, 'mk_bin', Bin),
    ( exists_file(Bin) -> delete_file(Bin) ; true ),
    directory_file_path(Dir, 'mk', Prog0),
    atom_concat(Prog0, '.plawk', Prog),
    setup_call_cleanup(open(Prog, write, S, [encoding(utf8)]),
        write(S, Src), close(S)),
    atom_concat(Prog0, '_in.txt', In),
    setup_call_cleanup(open(In, write, SI, [encoding(utf8)]),
        write(SI, Input), close(SI)),
    process_create(path(swipl), ['examples/plawk/bin/plawk', build, Prog, '-o', Bin],
        [stdout(pipe(O)), stderr(null), process(BPid)]),
    read_string(O, _, _), close(O),
    process_wait(BPid, exit(BuildStatus)),
    assertion(BuildStatus == 0),
    process_create(Bin, [In], [stdout(pipe(PS)), stderr(std), process(Pid)]),
    read_string(PS, _, Out0),
    close(PS),
    process_wait(Pid, exit(0)),
    sorted_lines(Out0, Out),
    ( Out == Expected
    -> true
    ;  format(user_error, "~n~w~n  got      ~q~n  expected ~q~n",
           [Src, Out, Expected]), fail
    ).

sorted_lines(Text, Sorted) :-
    split_string(Text, "\n", "", Parts0),
    exclude(==(""), Parts0, Parts),
    msort(Parts, SortedParts),
    atomic_list_concat(SortedParts, '\n', Joined0),
    (   SortedParts == []
    ->  Sorted = ""
    ;   atom_string(Joined0, Joined),
        string_concat(Joined, "\n", Sorted)
    ).

build_ll(Src, LL) :-
    plawk_parse_string(Src, Program),
    plawk_program_native_driver_ir(Program, 'input.txt', IR),
    atom_string(IR, LL).
