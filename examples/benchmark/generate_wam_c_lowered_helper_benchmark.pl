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
    write_wam_c_project([user:wam_c_bench_pair/2], Options, OutputDir),
    directory_file_path(OutputDir, 'main.c', MainPath),
    lowered_helper_benchmark_main(MainCode),
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

setup_benchmark_facts :-
    cleanup_benchmark_facts,
    assertz(user:wam_c_bench_pair(a, b)),
    assertz(user:wam_c_bench_pair(a, c)),
    assertz(user:wam_c_bench_pair(b, d)).

cleanup_benchmark_facts :-
    retractall(user:wam_c_bench_pair(_, _)).

lowered_helper_benchmark_main(
'#include "wam_runtime.h"
#include <stdio.h>

void setup_wam_c_bench_pair_2(WamState* state);
void setup_lowered_wam_c_helpers(WamState* state);

static int emit_pair(WamState* state, const char* left, const char* right, double score) {
    WamValue args[2] = { val_atom(left), val_atom(right) };
    int rc = wam_run_predicate(state, "wam_c_bench_pair/2", args, 2);
    if (rc != 0 || state->P != WAM_HALT) return 1;
    printf("%s\\t%s\\t%.6f\\n", left, right, score);
    return 0;
}

int main(void) {
    WamState state;
    wam_state_init(&state);
    setup_wam_c_bench_pair_2(&state);
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

    WamValue missing_args[2] = { val_atom("z"), val_atom("missing") };
    int missing_rc = wam_run_predicate(&state, "wam_c_bench_pair/2", missing_args, 2);
    if (missing_rc != WAM_HALT) {
        wam_free_state(&state);
        return 40;
    }

    wam_free_state(&state);
    return 0;
}
').

write_text_file(Path, Content) :-
    setup_call_cleanup(
        open(Path, write, Stream),
        format(Stream, '~w', [Content]),
        close(Stream)
    ).
