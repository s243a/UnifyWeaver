:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% Output redirection: `print ... > "file"`, `>> "file"`, and printf.
%
% ---------------------------------------------------------------------------
% `print $1 > "out.txt"` failed to parse. It now parses to
% redirect(write|append, string(Path), Print) for a rule-body print or printf,
% and lowers as the unredirected print BRACKETED by two runtime calls:
%
%   @plawk_redirect_begin(name, append)  flush stdio; save the real stdout once;
%                                        point fd 1 at the target
%   @plawk_redirect_end()                flush; restore fd 1
%
% so every print emitter works unchanged. Each target is opened ONCE per name and
% kept open, as in awk: the first `>` truncates, later prints append; `>>` opens
% for append. `/dev/stderr` and `/dev/stdout` are the real streams, never reopened.
% A target that cannot be opened is fatal (exit 2), as in gawk. The helpers live
% in the NATIVE external declarations only.
%
% Admitted only where the scalar action-sequence emitter lowers it (rule bodies);
% other contexts (END, BEGIN, the assoc drivers) decline or fail to parse.
%
% gawk is the oracle (verified against gawk 5.1.0, LC_ALL=C): stdout, stderr and
% the files written are all compared.

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(filesex), [make_directory_path/1, delete_directory_and_contents/1]).
:- use_module(library(readutil)).

clang_available :-
    catch(( process_create(path(clang), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

input("a 1\nb 2\nc 3\n").

:- begin_tests(plawk_print_redirect).

test(write_to_a_file, [condition(clang_available)]) :-
    run("{ print $1 > \"out.txt\" }\n", "", "", ["out.txt"-"a\nb\nc\n"], 0),
    !.

% The first `>` truncates an existing file; later prints to it append.
test(truncates_once_then_appends, [condition(clang_available)]) :-
    run("$2 > 1 { print $1 > \"keep.txt\" }\n", "", "", ["keep.txt"-"b\nc\n"], 0),
    !.

test(append_mode, [condition(clang_available)]) :-
    run("{ print $0 >> \"keep.txt\" }\n", "", "",
        ["keep.txt"-"old\na 1\nb 2\nc 3\n"], 0),
    !.

test(mixed_with_stdout_and_several_files, [condition(clang_available)]) :-
    run("{ print $1 > \"out.txt\"; print $2 }\n", "1\n2\n3\n", "",
        ["out.txt"-"a\nb\nc\n"], 0),
    !,
    run("{ print $2 > \"x.txt\"; print $1 > \"y.txt\" }\n", "", "",
        ["x.txt"-"1\n2\n3\n", "y.txt"-"a\nb\nc\n"], 0),
    !,
    run("{ if ($2 > 1) print $1 > \"big.txt\"; else print $1 > \"small.txt\" }\n",
        "", "", ["big.txt"-"b\nc\n", "small.txt"-"a\n"], 0),
    !.

test(printf_redirect, [condition(clang_available)]) :-
    run("{ printf \"%s-%s\\n\", $1, $2 > \"p.txt\" }\n", "", "",
        ["p.txt"-"a-1\nb-2\nc-3\n"], 0),
    !.

test(standard_streams, [condition(clang_available)]) :-
    run("{ print $1 > \"/dev/stderr\" }\n", "", "a\nb\nc\n", [], 0),
    !,
    run("{ print $1 > \"/dev/stdout\" }\n", "a\nb\nc\n", "", [], 0),
    !.

test(unopenable_target_is_fatal, [condition(clang_available)]) :-
    run("{ print $1 > \"nodir/f.txt\" }\n", "", _AnyStderr, [], 2),
    !.

:- end_tests(plawk_print_redirect).

% --- helpers ---------------------------------------------------------------

odir(Dir) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_print_redirect', Dir),
    ( exists_directory(Dir) -> true ; make_directory_path(Dir) ).

% run(+Src, +Stdout, ?Stderr, +Files, +RC): build, run in a fresh working
% directory holding keep.txt ("old\n"), and compare stdout, stderr (when bound),
% the exit status and the named files' contents.
run(Src, ExpectedOut, ExpectedErr, Files, ExpectedRC) :-
    input(Input),
    odir(Dir),
    directory_file_path(Dir, 'pr_bin', Bin),
    ( exists_file(Bin) -> delete_file(Bin) ; true ),
    directory_file_path(Dir, 'pr', Prog0),
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
    directory_file_path(Dir, 'cwd', Cwd),
    ( exists_directory(Cwd) -> delete_directory_and_contents(Cwd) ; true ),
    make_directory_path(Cwd),
    directory_file_path(Cwd, 'keep.txt', Keep),
    setup_call_cleanup(open(Keep, write, SK), write(SK, "old\n"), close(SK)),
    process_create(Bin, [In], [cwd(Cwd), stdout(pipe(PS)), stderr(pipe(PE)),
        process(Pid)]),
    read_string(PS, _, Out), close(PS),
    read_string(PE, _, Err), close(PE),
    process_wait(Pid, exit(RC)),
    findall(F-C, ( member(F-_, Files), directory_file_path(Cwd, F, Path),
                   ( exists_file(Path) -> read_file_to_string(Path, C, []) ; C = missing ) ),
        Got),
    (   Out == ExpectedOut, RC == ExpectedRC, Got == Files,
        ( var(ExpectedErr) -> true ; Err == ExpectedErr )
    ->  true
    ;   format(user_error, "~n~w~n  got      out=~q err=~q files=~q rc=~w~n  expected out=~q err=~q files=~q rc=~w~n",
            [Src, Out, Err, Got, RC, ExpectedOut, ExpectedErr, Files, ExpectedRC]),
        fail
    ).
