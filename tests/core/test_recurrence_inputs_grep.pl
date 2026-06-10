:- encoding(utf8).
%% Grep tripwire for recurrence_inputs.pl: asserts the module body
%% contains no target-specific atoms (fsharp / haskell / c_target etc).
%%
%% This is the faster, less robust of the two target-agnosticism
%% enforcement layers. The stronger layer is the load-isolation test
%% (test_recurrence_inputs_isolated.pl). Both are kept because grep
%% is cheap and catches direct violations even when load-isolation
%% would also catch them transitively.
%%
%% Usage:
%%   swipl -g run_tests -t halt tests/core/test_recurrence_inputs_grep.pl

:- use_module(library(lists)).
:- use_module(library(readutil), [read_file_to_string/3]).
:- use_module(library(filesex), [directory_file_path/3]).

%% Capture this test file's directory at load time so we can build
%% an absolute path to the helper module.
:- prolog_load_context(directory, Dir),
   asserta(test_file_directory(Dir)).

run_tests :-
    format("~n========================================~n"),
    format("recurrence_inputs Grep Tripwire~n"),
    format("========================================~n~n"),
    test_file_directory(Dir),
    directory_file_path(Dir, '../../src/unifyweaver/core/recurrence_inputs.pl',
                        HelperPath),
    read_file_to_string(HelperPath, Source, []),
    %% Strip comments first so module references in design-note
    %% comments don't trigger false positives.
    strip_prolog_comments(Source, CodeOnly),
    Forbidden = [
        "fsharp",
        "wam_fsharp",
        "wam_haskell",
        "wam_c_",
        "haskell_target",
        "c_target",
        "csharp_target"
    ],
    findall(F,
            ( member(F, Forbidden),
              sub_string(CodeOnly, _, _, _, F)
            ),
            Violations),
    (   Violations == []
    ->  format("[PASS] no target-specific atoms in module body (comments stripped)~n"),
        format("Checked forbidden tokens: ~w~n", [Forbidden]),
        format("~nAll tests passed~n"),
        format("========================================~n")
    ;   format("[FAIL] target-specific atoms found in module body:~n"),
        forall(member(V, Violations),
               format("  - ~w~n", [V])),
        format("Tests FAILED~n"),
        halt(1)
    ).

%% strip_prolog_comments(+Source, -CodeOnly)
%
% Naive comment stripper: removes everything between `%` and end of
% line on each line. Good enough for the grep check; doesn't handle
% string-literal % correctly but no string in this module contains
% the forbidden substrings.
strip_prolog_comments(Source, CodeOnly) :-
    split_string(Source, "\n", "", Lines),
    maplist(strip_line_comment, Lines, StrippedLines),
    atomics_to_string(StrippedLines, "\n", CodeOnly).

strip_line_comment(Line, Stripped) :-
    (   sub_string(Line, Idx, _, _, "%")
    ->  sub_string(Line, 0, Idx, _, Stripped)
    ;   Stripped = Line
    ).
