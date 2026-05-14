:- module(generate_wam_c_lowered_helper_benchmark,
          [ main/0,
            generate/2
          ]).
:- initialization(main, main).

:- use_module('../../src/unifyweaver/targets/wam_c_target').
:- use_module(library(filesex), [directory_file_path/3, make_directory_path/1]).

%% generate_wam_c_lowered_helper_benchmark.pl
%%
%% Generates a tiny WAM-C benchmark project that compares interpreted WAM-C
%% facts against the prototype lowered fact-helper path.
%%
%% Usage:
%%   swipl -q -s examples/benchmark/generate_wam_c_lowered_helper_benchmark.pl -- \
%%       <output-dir> [interpreted|lowered]

main :-
    current_prolog_flag(argv, Argv),
    (   Argv == []
    ->  true
    ;   Argv = [OutputDir, ModeAtom]
    ->  generate(OutputDir, ModeAtom),
        halt(0)
    ;   Argv = [OutputDir]
    ->  generate(OutputDir, lowered),
        halt(0)
    ;   format(user_error,
               'Usage: ... -- <output-dir> [interpreted|lowered]~n',
               []),
        halt(1)
    ).

main :-
    format(user_error, 'Error: WAM-C lowered-helper benchmark generation failed~n', []),
    halt(1).

generate(OutputDir, Mode) :-
    parse_mode(Mode, ParsedMode),
    setup_benchmark_facts,
    make_directory_path(OutputDir),
    mode_options(ParsedMode, Options),
    mode_predicates(ParsedMode, Predicates),
    write_wam_c_project(Predicates, Options, OutputDir),
    directory_file_path(OutputDir, 'main.c', MainPath),
    mode_query_predicate(ParsedMode, QueryPred),
    mode_alias_setup(ParsedMode, AliasDecl, AliasSetup),
    lowered_helper_benchmark_main(QueryPred, AliasDecl, AliasSetup, MainCode),
    write_text_file(MainPath, MainCode),
    directory_file_path(OutputDir, 'README.md', ReadmePath),
    format(atom(Readme),
           '# WAM-C lowered-helper benchmark~n~nMode: `~w`~n~nPlanner metadata is emitted as comments in `lib.c`.~n',
           [ParsedMode]),
    write_text_file(ReadmePath, Readme),
    cleanup_benchmark_facts.

parse_mode(lowered, lowered) :- !.
parse_mode('lowered', lowered) :- !.
parse_mode(interpreted, interpreted) :- !.
parse_mode('interpreted', interpreted) :- !.
parse_mode(Mode, _) :-
    throw(error(domain_error(wam_c_lowered_helper_benchmark_mode, Mode), _)).

mode_options(lowered, [lowered_helpers(true), report_lowered_helpers(true)]).
mode_options(interpreted, [report_lowered_helpers(true)]).

mode_predicates(lowered, [user:wam_c_bench_pair/2,
                          user:wam_c_bench_pair_alias/2,
                          user:wam_c_bench_pair_tag/3,
                          user:wam_c_bench_pair_keep/2]).
mode_predicates(interpreted, [user:wam_c_bench_pair/2]).

mode_query_predicate(lowered, 'wam_c_bench_pair_keep/2').
mode_query_predicate(interpreted, 'wam_c_bench_pair/2').

mode_alias_setup(lowered,
                 'void setup_wam_c_bench_pair_alias_2(WamState* state);\nvoid setup_wam_c_bench_pair_tag_3(WamState* state);\nvoid setup_wam_c_bench_pair_keep_2(WamState* state);\n',
                 '    setup_wam_c_bench_pair_alias_2(&state);\n    setup_wam_c_bench_pair_tag_3(&state);\n    setup_wam_c_bench_pair_keep_2(&state);\n').
mode_alias_setup(interpreted, '', '').

setup_benchmark_facts :-
    cleanup_benchmark_facts,
    assertz(user:wam_c_bench_pair(a, b)),
    assertz(user:wam_c_bench_pair(a, c)),
    assertz(user:wam_c_bench_pair(b, d)),
    assertz(user:wam_c_bench_pair_tag(a, b, keep)),
    assertz(user:wam_c_bench_pair_tag(a, c, keep)),
    assertz(user:wam_c_bench_pair_tag(b, d, keep)),
    assertz(user:((wam_c_bench_pair_alias(X, Y) :- wam_c_bench_pair(X, Y)))),
    assertz(user:((wam_c_bench_pair_keep(X, Y) :- wam_c_bench_pair_tag(X, Y, keep)))).

cleanup_benchmark_facts :-
    retractall(user:wam_c_bench_pair(_, _)),
    retractall(user:wam_c_bench_pair_alias(_, _)),
    retractall(user:wam_c_bench_pair_tag(_, _, _)),
    retractall(user:wam_c_bench_pair_keep(_, _)).

lowered_helper_benchmark_main(QueryPred, AliasDecl, AliasSetup, Code) :-
    format(atom(Code),
'#include "wam_runtime.h"
#include <stdio.h>

void setup_wam_c_bench_pair_2(WamState* state);
~w
void setup_lowered_wam_c_helpers(WamState* state);

static int emit_pair(WamState* state, const char* left, const char* right, double score) {
    WamValue args[2] = { val_atom(left), val_atom(right) };
    int rc = wam_run_predicate(state, "~w", args, 2);
    if (rc != 0 || state->P != WAM_HALT) return 1;
    printf("%s\\t%s\\t%.6f\\n", left, right, score);
    return 0;
}

int main(void) {
    WamState state;
    wam_state_init(&state);
    setup_wam_c_bench_pair_2(&state);
~w
    setup_lowered_wam_c_helpers(&state);

    printf("left\\tright\\tscore\\n");

    if (emit_pair(&state, "a", "b", 1.0) != 0) {
        wam_free_state(&state);
        return 10;
    }
    if (emit_pair(&state, "a", "c", 2.0) != 0) {
        wam_free_state(&state);
        return 20;
    }
    if (emit_pair(&state, "b", "d", 3.0) != 0) {
        wam_free_state(&state);
        return 30;
    }

    wam_free_state(&state);
    return 0;
}
',
           [AliasDecl, QueryPred, AliasSetup]).

write_text_file(Path, Content) :-
    setup_call_cleanup(
        open(Path, write, Stream),
        format(Stream, '~w', [Content]),
        close(Stream)
    ).
