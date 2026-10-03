:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (@s243a)
%
% examples/plawk/scripts/ir_corpus.pl -- the IR corpus tool for refactors that
% must not change generated code (docs/design/PLAN_TEMPLATE_REFACTOR.md §5).
% Driven through its CLI, as a refactor PR uses it.

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(filesex), [make_directory_path/1, directory_file_path/3]).

:- begin_tests(plawk_ir_corpus).

% `list` extracts the corpus: every "...{..." literal of tests/test_plawk_*.pl,
% including non-ASCII ones (read as UTF-8, whatever the locale)
test(list_extracts_the_corpus) :-
    tmp_file_path('list.txt', Out),
    tool([list, Out], 0, _),
    read_file_to_string(Out, Text, [encoding(utf8)]),
    split_string(Text, "\n", "", Lines0),
    exclude(==(""), Lines0, Lines),
    length(Lines, N),
    assertion(N > 2000),
    assertion(memberchk("{ c[$1]++ } END { for (k in c) print k, c[k] }\\n", Lines)),
    assertion(( member(L, Lines), sub_string(L, _, _, _, "café") )),
    !.

% diff: identical dumps -> exit 0; a changed IR is listed with its program -> exit 1
test(diff_reports_changes_with_exit_status) :-
    tmp_file_path('a.txt', A),
    tmp_file_path('b.txt', B),
    tmp_file_path('c.txt', C),
    Dump = "=== { print $1 }\\n\nIR 1111\nline\n=== { x }\\n\nDECLINE\n",
    write_utf8(A, Dump),
    write_utf8(B, Dump),
    write_utf8(C, "=== { print $1 }\\n\nIR 2222\nline\n=== { x }\\n\nIR 3333\nmore\n"),
    tool([diff, A, B], 0, Out0),
    assertion(sub_string(Out0, _, _, _, "0 changed")),
    tool([diff, A, C], 1, Out1),
    assertion(sub_string(Out1, _, _, _, "CHANGED compiled -> compiled: { print $1 }")),
    assertion(sub_string(Out1, _, _, _, "CHANGED decline -> compiled: { x }")),
    assertion(sub_string(Out1, _, _, _, "2 changed")),
    !.

:- end_tests(plawk_ir_corpus).

% --- helpers ---------------------------------------------------------------

tmp_file_path(Name, Path) :-
    current_prolog_flag(tmp_dir, Tmp),
    directory_file_path(Tmp, 'uw_plawk_ir_corpus', Dir),
    make_directory_path(Dir),
    directory_file_path(Dir, Name, Path).

write_utf8(Path, Text) :-
    setup_call_cleanup(open(Path, write, S, [encoding(utf8)]), write(S, Text), close(S)).

tool(Args, ExpectedStatus, Out) :-
    process_create(path(swipl), ['examples/plawk/scripts/ir_corpus.pl' | Args],
        [stdout(pipe(O)), stderr(null), process(Pid)]),
    read_string(O, _, Out), close(O),
    process_wait(Pid, exit(Status)),
    assertion(Status == ExpectedStatus).
