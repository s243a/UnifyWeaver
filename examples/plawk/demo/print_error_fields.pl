% SPDX-License-Identifier: MIT
% Copyright (c) 2026 John William Creighton (s243a)

:- initialization(main, main).

:- use_module('../core/plawk_core').

main :-
    Reader = text_file_reader('examples/plawk/demo/error_fields.txt', " "),
    Handler = print_error_fields,
    State0 = state([], [], 0, plawk_options(" ", ",")),
    process_all(Reader, Handler, State0, StateN),
    state_outputs(StateN, Lines),
    forall(member(Line, Lines), format('~s~n', [Line])).

print_error_fields(Item, State0, StateN, yes) :-
    increment_counter(State0, State1),
    (   item_field(1, Item, "ERROR")
    ->  item_field(2, Item, Component),
        item_field(3, Item, Message),
        nr(State1, RecordNumber),
        print_fields([RecordNumber, Component, Message], State1, StateN)
    ;   StateN = State1
    ).
