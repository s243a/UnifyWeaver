:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% plawk multi-pass, phase 2 PR-A: parser + AST for `pass { }` blocks. A
% program with one or more `pass { }` blocks parses to
% program_passes(Begin, [pass(Rules), ...], End); a bare-main program is
% unchanged (program(...)). A SINGLE pass is exactly today's main loop, so
% the plawk CLI normalises it to the ordinary single-pass program and the
% whole existing pipeline compiles+runs it. Two-or-more passes need the
% multi-pass driver (PR-B); until then the CLI reports them as unsupported
% rather than mis-compiling. See PLAWK_MULTIPASS_CACHE.md.

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(filesex), [make_directory_path/1]).
:- use_module('../examples/plawk/parser/plawk_parser').

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

:- begin_tests(plawk_passes).

% One `pass { }` block -> program_passes with a single pass.
test(single_pass_parses) :-
    plawk_parse_string(
        "pass { c[$1]++ }\nEND { for (k in c) print k, c[k] }\n",
        program_passes([], [pass([rule(always, [inc_assoc(var(c), field(1))])])],
            [end([for_in(var(k), var(c),
                [print([var(k), assoc(var(c), var(k))])])])])),
    !.

% Two `pass { }` blocks -> program_passes with two passes, in order.
test(two_passes_parse) :-
    plawk_parse_string(
        "pass { a[$1]++ }\npass { print $1 }\n",
        program_passes([],
            [pass([rule(always, [inc_assoc(var(a), field(1))])]),
             pass([rule(always, [print([field(1)])])])],
            [])),
    !.

% A bare-main program is untouched -- still program(...), not program_passes.
test(bare_main_unchanged) :-
    plawk_parse_string(
        "{ c[$1]++ }\nEND { for (k in c) print k, c[k] }\n",
        program(_, _, _)),
    \+ plawk_parse_string(
        "{ c[$1]++ }\nEND { for (k in c) print k, c[k] }\n",
        program_passes(_, _, _)),
    !.

% A single pass compiles and runs exactly like the equivalent bare main.
test(single_pass_runs_like_main, [condition(clang_available)]) :-
    pdir(Dir),
    Src = "pass { c[$1]++ }\nEND { for (k in c) print k, c[k] }\n",
    build_bin(Dir, 'one', Src, Bin, 0),
    directory_file_path(Dir, 'in.txt', Input),
    setup_call_cleanup(open(Input, write, S, [encoding(utf8)]),
        write(S, "a\na\nb\n"), close(S)),
    run_sorted(Bin, [Input], Out),
    assertion(Out == ["a 2", "b 1"]),
    !.

% Two passes are diagnosed as not-yet-implemented (exit 3), not mis-built.
test(two_passes_reported_unsupported) :-
    pdir(Dir),
    Src = "pass { c[$1]++ }\npass { print $1 }\nEND { for (k in c) print k, c[k] }\n",
    directory_file_path(Dir, 'two.plawk', Prog),
    setup_call_cleanup(open(Prog, write, S, [encoding(utf8)]),
        write(S, Src), close(S)),
    directory_file_path(Dir, 'two_bin', Bin),
    cli([build, Prog, '-o', Bin], 3),
    !.

:- end_tests(plawk_passes).

% --- helpers ---------------------------------------------------------------

pdir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_passes', Dir),
    ( exists_directory(Dir) -> true ; make_directory_path(Dir) ).

build_bin(Dir, Name, Src, Bin, ExpectedStatus) :-
    directory_file_path(Dir, Name, Prog0),
    atom_concat(Prog0, '.plawk', Prog),
    setup_call_cleanup(open(Prog, write, S, [encoding(utf8)]),
        write(S, Src), close(S)),
    atom_concat(Prog0, '_bin', Bin),
    cli([build, Prog, '-o', Bin], ExpectedStatus).

run_sorted(Bin, Args, Sorted) :-
    process_create(Bin, Args,
        [stdout(pipe(S)), stderr(std), process(Pid)]),
    read_string(S, _, Out), close(S),
    process_wait(Pid, exit(Status)),
    assertion(Status == 0),
    split_string(Out, "\n", "", L0), exclude(==(""), L0, L), msort(L, Sorted).

cli(Args, ExpectedStatus) :-
    process_create(path(swipl), ['examples/plawk/bin/plawk' | Args],
        [stdout(pipe(S)), stderr(std), process(Pid)]),
    read_string(S, _, _), close(S),
    process_wait(Pid, exit(Status)),
    assertion(Status == ExpectedStatus).
