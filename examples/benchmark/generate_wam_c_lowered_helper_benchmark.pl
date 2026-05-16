:- module(generate_wam_c_lowered_helper_benchmark,
          [ main/0,
            generate/2,
            generate/3
          ]).
:- initialization(main, main).

:- use_module('../../src/unifyweaver/targets/wam_c_target').
:- use_module(library(filesex), [directory_file_path/3, make_directory_path/1]).

%% generate_wam_c_lowered_helper_benchmark.pl
%%
%% Generates a tiny WAM-C benchmark project that compares interpreted WAM-C
%% predicates against the prototype lowered helper paths.
%%
%% Usage:
%%   swipl -q -s examples/benchmark/generate_wam_c_lowered_helper_benchmark.pl -- \
%%       <output-dir> [interpreted|lowered] [scale]

main :-
    current_prolog_flag(argv, Argv),
    (   Argv == []
    ->  true
    ;   Argv = [OutputDir, ModeAtom, ScaleAtom]
    ->  generate(OutputDir, ModeAtom, ScaleAtom),
        halt(0)
    ;   Argv = [OutputDir, ModeAtom]
    ->  generate(OutputDir, ModeAtom, dev),
        halt(0)
    ;   Argv = [OutputDir]
    ->  generate(OutputDir, lowered, dev),
        halt(0)
    ;   format(user_error,
               'Usage: ... -- <output-dir> [interpreted|lowered] [scale]~n',
               []),
        halt(1)
    ).

main :-
    format(user_error, 'Error: WAM-C lowered-helper benchmark generation failed~n', []),
    halt(1).

generate(OutputDir, Mode) :-
    generate(OutputDir, Mode, dev).

generate(OutputDir, Mode, Scale) :-
    parse_mode(Mode, ParsedMode),
    parse_scale(Scale, RowCount),
    setup_benchmark_facts(RowCount),
    make_directory_path(OutputDir),
    mode_options(ParsedMode, Options),
    mode_predicates(ParsedMode, Predicates),
    write_wam_c_project(Predicates, Options, OutputDir),
    directory_file_path(OutputDir, 'main.c', MainPath),
    lowered_helper_benchmark_main(RowCount, MainCode),
    write_text_file(MainPath, MainCode),
    directory_file_path(OutputDir, 'README.md', ReadmePath),
    format(atom(Readme),
           '# WAM-C lowered-helper benchmark~n~nMode: `~w`~n~nScale: `~w` (~w rows per projection variant)~n~nThe generated runner queries direct, reordered, ignored-output, and row-constrained projection shapes. Planner metadata is emitted as comments in `lib.c`.~n',
           [ParsedMode, Scale, RowCount]),
    write_text_file(ReadmePath, Readme),
    cleanup_benchmark_facts.

parse_mode(lowered, lowered) :- !.
parse_mode('lowered', lowered) :- !.
parse_mode(interpreted, interpreted) :- !.
parse_mode('interpreted', interpreted) :- !.
parse_mode(Mode, _) :-
    throw(error(domain_error(wam_c_lowered_helper_benchmark_mode, Mode), _)).

parse_scale(Scale, Count) :-
    atom(Scale),
    parse_scale_atom(Scale, Count),
    !.
parse_scale(Scale, Count) :-
    string(Scale),
    atom_string(ScaleAtom, Scale),
    parse_scale_atom(ScaleAtom, Count),
    !.
parse_scale(Scale, _) :-
    throw(error(domain_error(wam_c_lowered_helper_benchmark_scale, Scale), _)).

parse_scale_atom(dev, 4) :- !.
parse_scale_atom(Scale, Count) :-
    atom_chars(Scale, Chars),
    append(DigitChars, SuffixChars, Chars),
    DigitChars \= [],
    maplist(char_type_digit, DigitChars),
    number_chars(Base, DigitChars),
    scale_suffix_multiplier(SuffixChars, Multiplier),
    Count is Base * Multiplier,
    Count > 0.

char_type_digit(Char) :-
    char_type(Char, digit).

scale_suffix_multiplier([], 1).
scale_suffix_multiplier(['x'], 4).
scale_suffix_multiplier(['k'], 1000).
scale_suffix_multiplier(['K'], 1000).

mode_options(lowered, [lowered_helpers(true), report_lowered_helpers(true)]).
mode_options(interpreted, [report_lowered_helpers(true)]).

mode_predicates(_Mode, [user:wam_c_bench_pair/2,
                        user:wam_c_bench_pair_alias/2,
                        user:wam_c_bench_pair_projected/2,
                        user:wam_c_bench_pair_left/1,
                        user:wam_c_bench_pair_equal/1,
                        user:wam_c_bench_pair_tag/3,
                        user:wam_c_bench_pair_keep/2,
                        user:wam_c_bench_pair_score/3,
                        user:wam_c_bench_pair_small/2]).

setup_benchmark_facts(RowCount) :-
    cleanup_benchmark_facts,
    forall(between(1, RowCount, I), assert_benchmark_fact_row(I)),
    assertz(user:((wam_c_bench_pair_alias(X, Y) :- wam_c_bench_pair(X, Y)))),
    assertz(user:((wam_c_bench_pair_projected(Y, X) :- wam_c_bench_pair(X, Y)))),
    assertz(user:((wam_c_bench_pair_left(X) :- wam_c_bench_pair(X, _Ignored)))),
    assertz(user:((wam_c_bench_pair_equal(X) :- wam_c_bench_pair(X, X)))),
    assertz(user:((wam_c_bench_pair_keep(X, Y) :- wam_c_bench_pair_tag(X, Y, keep)))),
    assertz(user:((wam_c_bench_pair_small(X, Y) :-
        wam_c_bench_pair_score(X, Y, Score),
        Score =< 3))).

assert_benchmark_fact_row(I) :-
    benchmark_atoms(I, Left, Right, Equal),
    assertz(user:wam_c_bench_pair(Left, Right)),
    assertz(user:wam_c_bench_pair(Equal, Equal)),
    assertz(user:wam_c_bench_pair_tag(Left, Right, keep)),
    assertz(user:wam_c_bench_pair_tag(Equal, Equal, keep)),
    assertz(user:wam_c_bench_pair_score(Left, Right, I)),
    assertz(user:wam_c_bench_pair_score(Equal, Equal, I)).

benchmark_atoms(I, Left, Right, Equal) :-
    format(atom(Left), 'l~w', [I]),
    format(atom(Right), 'r~w', [I]),
    format(atom(Equal), 'e~w', [I]).

cleanup_benchmark_facts :-
    retractall(user:wam_c_bench_pair(_, _)),
    retractall(user:wam_c_bench_pair_alias(_, _)),
    retractall(user:wam_c_bench_pair_projected(_, _)),
    retractall(user:wam_c_bench_pair_left(_)),
    retractall(user:wam_c_bench_pair_equal(_)),
    retractall(user:wam_c_bench_pair_tag(_, _, _)),
    retractall(user:wam_c_bench_pair_keep(_, _)),
    retractall(user:wam_c_bench_pair_score(_, _, _)),
    retractall(user:wam_c_bench_pair_small(_, _)).

lowered_helper_benchmark_main(RowCount, Code) :-
    benchmark_atom_array(RowCount, left, LeftArray),
    benchmark_atom_array(RowCount, right, RightArray),
    benchmark_atom_array(RowCount, equal, EqualArray),
    format(atom(Code),
'#include "wam_runtime.h"
#include <stdio.h>

void setup_wam_c_bench_pair_2(WamState* state);
void setup_wam_c_bench_pair_alias_2(WamState* state);
void setup_wam_c_bench_pair_projected_2(WamState* state);
void setup_wam_c_bench_pair_left_1(WamState* state);
void setup_wam_c_bench_pair_equal_1(WamState* state);
void setup_wam_c_bench_pair_tag_3(WamState* state);
void setup_wam_c_bench_pair_keep_2(WamState* state);
void setup_wam_c_bench_pair_score_3(WamState* state);
void setup_wam_c_bench_pair_small_2(WamState* state);
void setup_lowered_wam_c_helpers(WamState* state);

static int emit_pair(WamState* state, const char* variant, const char* pred, const char* left, const char* right, double score) {
    WamValue args[2] = { val_atom(left), val_atom(right) };
    int rc = wam_run_predicate(state, pred, args, 2);
    if (rc != 0 || state->P != WAM_HALT) return 1;
    printf("%s:%s\\t%s\\t%.6f\\n", variant, left, right, score);
    return 0;
}

static int emit_left(WamState* state, const char* variant, const char* pred, const char* left, double score) {
    WamValue args[1] = { val_atom(left) };
    int rc = wam_run_predicate(state, pred, args, 1);
    if (rc != 0 || state->P != WAM_HALT) return 1;
    printf("%s:%s\\t_\\t%.6f\\n", variant, left, score);
    return 0;
}

int main(void) {
    WamState state;
    static const char* lefts[] = { ~w };
    static const char* rights[] = { ~w };
    static const char* equals[] = { ~w };
    const int row_count = ~w;
    wam_state_init(&state);
    setup_wam_c_bench_pair_2(&state);
    setup_wam_c_bench_pair_alias_2(&state);
    setup_wam_c_bench_pair_projected_2(&state);
    setup_wam_c_bench_pair_left_1(&state);
    setup_wam_c_bench_pair_equal_1(&state);
    setup_wam_c_bench_pair_tag_3(&state);
    setup_wam_c_bench_pair_keep_2(&state);
    setup_wam_c_bench_pair_score_3(&state);
    setup_wam_c_bench_pair_small_2(&state);
    setup_lowered_wam_c_helpers(&state);

    printf("left\\tright\\tscore\\n");

    for (int i = 0; i < row_count; i++) {
        double base = (double)(i + 1);
        if (emit_pair(&state, "direct", "wam_c_bench_pair_alias/2", lefts[i], rights[i], base) != 0) {
            wam_free_state(&state);
            return 10;
        }
        if (emit_pair(&state, "reordered", "wam_c_bench_pair_projected/2", rights[i], lefts[i], base + 0.25) != 0) {
            wam_free_state(&state);
            return 20;
        }
        if (emit_left(&state, "ignored-output", "wam_c_bench_pair_left/1", lefts[i], base + 0.50) != 0) {
            wam_free_state(&state);
            return 30;
        }
        if (emit_left(&state, "row-constrained", "wam_c_bench_pair_equal/1", equals[i], base + 0.75) != 0) {
            wam_free_state(&state);
            return 40;
        }
    }

    wam_free_state(&state);
    return 0;
}
',
           [LeftArray, RightArray, EqualArray, RowCount]).

benchmark_atom_array(RowCount, Kind, Code) :-
    findall(Literal,
            (   between(1, RowCount, I),
                benchmark_atoms(I, Left, Right, Equal),
                benchmark_array_atom(Kind, Left, Right, Equal, Atom),
                format(atom(Literal), '"~w"', [Atom])
            ),
            Literals),
    atomic_list_concat(Literals, ', ', Code).

benchmark_array_atom(left, Left, _Right, _Equal, Left).
benchmark_array_atom(right, _Left, Right, _Equal, Right).
benchmark_array_atom(equal, _Left, _Right, Equal, Equal).

write_text_file(Path, Content) :-
    setup_call_cleanup(
        open(Path, write, Stream),
        format(Stream, '~w', [Content]),
        close(Stream)
    ).
