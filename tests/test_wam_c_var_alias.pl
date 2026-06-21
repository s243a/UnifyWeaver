:- encoding(utf8).
% Var-var aliasing regression tests for the WAM C target (M143).
%
% PR #2976's cross-target sweep found the C runtime's wam_unify copied
% one unbound cell over the other instead of installing a VAL_REF link,
% so `X = Y` created no alias: `X = Y, X = 1, Y = 2` SUCCEEDED, and
% `X = Y, X = 42, Y =:= 42` failed. wam_runtime.h's unbound-unbound
% case now installs a VAL_REF to the partner's heap slot.
%
% Gated on gcc; compiles the probes + runtime with gcc and runs them,
% mirroring run_real_prolog_builtin_executable_smoke in
% tests/test_wam_c_target.pl.

:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(filesex)).
:- use_module('../src/unifyweaver/targets/wam_c_target').
:- use_module('../src/unifyweaver/targets/wam_target').

:- dynamic user:c_alias_chain/1.
:- dynamic user:c_alias_conflict/1.
:- dynamic user:c_alias_chain3/1.
:- dynamic user:c_alias_backtrack/1.

% Binding through a var-var alias must propagate to both.
user:c_alias_chain(R)    :- X = Y, X = 42, ( Y =:= 42 -> R is 1 ; R is 0 ).
% Conflicting bindings through an alias must FAIL (the smoking gun:
% this succeeded before the fix).
user:c_alias_conflict(R) :- X = Y, X = 1, Y = 2, R is 1.
% Three-deep chain.
user:c_alias_chain3(R)   :- X = Y, Y = Z, X = 42, ( Z =:= 42 -> R is 1 ; R is 0 ).
% Alias must dissolve on backtrack out of the aliasing goal.
user:c_alias_backtrack(R) :-
    ( X = Y, X = 1, Y = 2 -> R is 0
    ; Y = 7, X = 5, ( Y =:= 7, X =:= 5 -> R is 1 ; R is 0 )
    ).

gcc_available :-
    catch(( process_create(path(gcc), ['--version'],
                           [stdout(null), stderr(null), process(Pid)]),
            process_wait(Pid, exit(0)) ), _, fail).

:- begin_tests(wam_c_var_alias, [condition(gcc_available)]).

test(var_alias_exec_parity) :-
    Dir = 'output/test_wam_c_var_alias',
    ( exists_directory(Dir) -> delete_directory_and_contents(Dir) ; true ),
    make_directory_path(Dir),
    Preds = [c_alias_chain, c_alias_conflict, c_alias_chain3, c_alias_backtrack],
    findall(PredCode,
            ( member(P, Preds),
              compile_predicate_to_wam(user:P/1, [], WamCode),
              compile_wam_predicate_to_c(user:P/1, WamCode, [], PredCode)
            ),
            PredCodes),
    compile_wam_runtime_to_c([], RuntimeCode),
    atomic_list_concat([Dir, '/runtime.c'], RuntimePath),
    write_text_file(RuntimePath, RuntimeCode),
    atomic_list_concat(PredCodes, '\n\n', AllPreds),
    format(atom(PredTU), '#include "wam_runtime.h"~n~n~w', [AllPreds]),
    atomic_list_concat([Dir, '/pred.c'], PredPath),
    write_text_file(PredPath, PredTU),
    driver_source(DriverSrc),
    atomic_list_concat([Dir, '/main.c'], MainPath),
    write_text_file(MainPath, DriverSrc),
    % Locate the runtime header next to the C target module.
    absolute_file_name('../src/unifyweaver/targets/wam_c_runtime', HdrDir,
                       [relative_to('tests/'), file_type(directory)]),
    atomic_list_concat([Dir, '/alias_bin'], ExePath),
    format(atom(Cmd),
        'gcc -O0 -I~w -o ~w ~w/main.c ~w/pred.c ~w/runtime.c -lm 2>&1 && ~w',
        [HdrDir, ExePath, Dir, Dir, Dir, ExePath]),
    process_create(path(sh), ['-c', Cmd],
                   [stdout(pipe(Out)), stderr(std), process(Pid)]),
    read_string(Out, _, OutStr), close(Out),
    process_wait(Pid, Status),
    ( Status == exit(0), sub_string(OutStr, _, _, _, "ALL 6 PASS")
    ->  true
    ;   format(user_error, "~n[c var-alias test output]~n~w~n", [OutStr]),
        throw(c_var_alias_test_failed(Status))
    ).

:- end_tests(wam_c_var_alias).

write_text_file(Path, Content) :-
    setup_call_cleanup(
        open(Path, write, Stream, [encoding(utf8)]),
        format(Stream, '~w', [Content]),
        close(Stream)).

% C driver: each row is (pred, arg, expected_success). The R=1 / R=0
% pairs read the actual value of R back through success/failure.
driver_source('#include <stdio.h>
#include "wam_runtime.h"

void setup_c_alias_chain_1(WamState* state);
void setup_c_alias_conflict_1(WamState* state);
void setup_c_alias_chain3_1(WamState* state);
void setup_c_alias_backtrack_1(WamState* state);

static int check(WamState *state, const char *pred, int argval, int expect_true) {
    WamValue args[1] = { val_int(argval) };
    int rc = wam_run_predicate(state, pred, args, 1);
    int is_true = (rc == 0 && state->P == WAM_HALT);
    if (is_true == expect_true) { printf("PASS %s(%d)\\n", pred, argval); return 1; }
    printf("FAIL %s(%d): got %s\\n", pred, argval, is_true ? "true" : "false");
    return 0;
}

int main(void) {
    WamState state;
    wam_state_init(&state);
    setup_c_alias_chain_1(&state);
    setup_c_alias_conflict_1(&state);
    setup_c_alias_chain3_1(&state);
    setup_c_alias_backtrack_1(&state);
    int n = 0;
    n += check(&state, "c_alias_chain/1", 1, 1);
    n += check(&state, "c_alias_chain/1", 0, 0);
    n += check(&state, "c_alias_conflict/1", 1, 0);
    n += check(&state, "c_alias_chain3/1", 1, 1);
    n += check(&state, "c_alias_chain3/1", 0, 0);
    n += check(&state, "c_alias_backtrack/1", 1, 1);
    if (n == 6) printf("ALL 6 PASS\\n");
    wam_free_state(&state);
    return n == 6 ? 0 : 1;
}
').
