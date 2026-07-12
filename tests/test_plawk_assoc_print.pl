:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% Per-record output in assoc programs: `print` inside the record loop,
% alongside table updates, with a text field `$N` or a table lookup
% `arr[$N]` (the i64 value at the key interned from field N). This is the
% capability that makes multi-pass useful for the "process each record with
% a global aggregate" (normalise) shape -- pass 1 builds a table, pass 2
% prints each record using it. See PLAWK_MULTIPASS_CACHE.md.

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(filesex), [make_directory_path/1]).

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

:- begin_tests(plawk_assoc_print).

% Single pass, per-record running count: `{ c[$1]++ ; print $1, c[$1] }`
% prints the key and its count SO FAR each record (needs an END to route
% through the mixed driver). Over a a b: a 1 / a 2 / b 1, then END c["a"]=2.
test(single_pass_running_count, [condition(clang_available)]) :-
    ap(Dir),
    Src = "{ c[$1]++ ; print $1, c[$1] }\nEND { print c[\"a\"] }\n",
    run(Dir, 'run', Src, "a\na\nb\n", Out),
    assertion(Out == ["a 1", "a 2", "b 1", "2"]),
    !.

% Multi-pass normalise shape (no END): pass 1 counts, pass 2 prints each
% record with the FINAL count. Over a a b: a 2 / a 2 / b 1.
test(multipass_final_count, [condition(clang_available)]) :-
    ap(Dir),
    Src = "pass { c[$1]++ }\npass { print $1, c[$1] }\n",
    run_sorted(Dir, 'mp', Src, "a\na\nb\n", Srt),
    assertion(Srt == ["a 2", "a 2", "b 1"]),
    !.

% A text field alone in a pass print (no lookup).
test(multipass_field_only, [condition(clang_available)]) :-
    ap(Dir),
    Src = "pass { c[$1]++ }\npass { print $1 }\nEND { for (k in c) print k, c[k] }\n",
    run_sorted(Dir, 'fo', Src, "a\na\nb\n", Srt),
    assertion(Srt == ["a", "a", "a 2", "b", "b 1"]),
    !.

:- end_tests(plawk_assoc_print).

% --- helpers ---------------------------------------------------------------

ap(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_assoc_print', Dir),
    ( exists_directory(Dir) -> true ; make_directory_path(Dir) ).

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

% ordered output (line order matters for the running-count case)
run(Dir, Name, Src, Input, Lines) :-
    build(Dir, Name, Src, Bin, In, Input),
    process_create(Bin, [In], [stdout(pipe(S)), stderr(std), process(Pid)]),
    read_string(S, _, Out), close(S), process_wait(Pid, exit(0)),
    split_string(Out, "\n", "", L0), exclude(==(""), L0, Lines).

% order-independent output
run_sorted(Dir, Name, Src, Input, Sorted) :-
    run(Dir, Name, Src, Input, Lines),
    msort(Lines, Sorted).

cli(Args, ExpectedStatus) :-
    process_create(path(swipl), ['examples/plawk/bin/plawk' | Args],
        [stdout(pipe(S)), stderr(std), process(Pid)]),
    read_string(S, _, _), close(S),
    process_wait(Pid, exit(Status)),
    assertion(Status == ExpectedStatus).
