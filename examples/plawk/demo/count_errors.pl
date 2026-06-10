% SPDX-License-Identifier: MIT
% Copyright (c) 2026 John William Creighton (s243a)

:- initialization(main, main).

:- use_module('../core/plawk_core').

main :-
    Reader = text_file_reader('examples/plawk/demo/sample_log.txt', " "),
    Handler = collect_error_lines,
    process_all(Reader, Handler, state([], [], 0, none), StateN),
    state_counter(StateN, Count),
    state_outputs(StateN, ErrorLines),
    format('records=~w~n', [Count]),
    forall(member(Line, ErrorLines), format('~s~n', [Line])).

collect_error_lines(Item, State0, StateN, yes) :-
    increment_counter(State0, State1),
    (   item_field(1, Item, "ERROR")
    ->  item_field(0, Item, Line),
        append_output(Line, State1, StateN)
    ;   StateN = State1
    ).
