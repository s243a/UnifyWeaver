:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% `$i` -- a field indexed by a scalar variable: `for (i = 1; i <= NF; i++) print $i`.
%
% ---------------------------------------------------------------------------
% The commonest awk loop failed to parse (exit 2): the grammar knew `$N`, `$NF` and
% `$(NF+K)`, but not a variable index. `$i`, `$(i)`, `$(i-K)`, `$(i+K)` now parse to
% a distinct functor, field_var(Name, Offset) -- every context not taught it
% declines -- and a rule-body print (incl. loop bodies and printf) substitutes the
% variable's CURRENT slot value, giving field_dyn(SlotValue, Offset). That lowers
% like `$NF`: the index is range-checked (a negative index is fatal in awk: exit 2
% after the output so far) and read through the subslicer (index 0 is $0; past NF
% is empty). Only an integer counter is an index; any other kind declines.
%
% gawk is the oracle (verified against gawk 5.1.0, LC_ALL=C), exit status included.

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(lists)).
:- use_module(library(filesex), [make_directory_path/1]).
:- use_module('../examples/plawk/parser/plawk_parser').

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

input("a b c\nx y\n\nlast one here now\n").

:- begin_tests(plawk_dollar_var).

test(parses_to_field_var) :-
    plawk_parse_string("{ print $i, $(i-1), $(j+2) }\n",
        program([], [rule(always, [print([field_var(i, 0), field_var(i, -1),
            field_var(j, 2)])])], [])),
    !,
    % a special is not a variable: $NF keeps its own term
    plawk_parse_string("{ print $NF }\n",
        program([], [rule(always, [print([field_nf(0)])])], [])),
    !.

test(loop_over_fields, [condition(clang_available)]) :-
    run("{ for (i = 1; i <= NF; i++) print $i }\n",
        "a\nb\nc\nx\ny\nlast\none\nhere\nnow\n", 0),
    !,
    run("{ for (i = NF; i > 0; i--) printf \"%s \", $i; print \"\" }\n",
        "c b a \ny x \n\nnow here one last \n", 0),
    !,
    run("{ for (i = 1; i <= NF; i++) if (i > 1) print i, $i }\n",
        "2 b\n3 c\n2 y\n2 one\n3 here\n4 now\n", 0),
    !.

test(index_zero_and_past_nf, [condition(clang_available)]) :-
    run("{ i = 0; print $i }\n", "a b c\nx y\n\nlast one here now\n", 0),
    !,
    run("{ i = 9; print $i \"|\" }\n", "|\n|\n|\n|\n", 0),
    !.

test(offsets, [condition(clang_available)]) :-
    run("{ i = 1; print $(i+1), $i }\n", "b a\ny x\n \none last\n", 0),
    !.

% A negative index is fatal, as in gawk: the third record is empty, so NF-1 = -1.
test(negative_index_is_fatal, [condition(clang_available)]) :-
    run("{ i = NF; print $(i-1) }\n", "b\nx\n", 2),
    !.

:- end_tests(plawk_dollar_var).

% --- helpers ---------------------------------------------------------------

odir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_dollar_var', Dir),
    ( exists_directory(Dir) -> true ; make_directory_path(Dir) ).

run(Src, Expected, ExpectedRC) :-
    input(Input),
    odir(Dir),
    directory_file_path(Dir, 'dv_bin', Bin),
    % A build that unexpectedly declines must not silently run a stale binary.
    ( exists_file(Bin) -> delete_file(Bin) ; true ),
    directory_file_path(Dir, 'dv', Prog0),
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
    process_create(Bin, [In], [stdout(pipe(PS)), stderr(null), process(Pid)]),
    read_string(PS, _, Out),
    close(PS),
    process_wait(Pid, exit(RC)),
    ( Out == Expected, RC == ExpectedRC
    -> true
    ;  format(user_error, "~n~w~n  got      ~q (exit ~w)~n  expected ~q (exit ~w)~n",
           [Src, Out, RC, Expected, ExpectedRC]), fail
    ).
