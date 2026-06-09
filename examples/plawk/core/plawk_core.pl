% SPDX-License-Identifier: MIT
% Copyright (c) 2026 John William Creighton (s243a)

:- module(plawk_core, [
    process_all/4,
    item_field/3,
    item_field_count/2,
    state_counter/2,
    increment_counter/2,
    text_file_reader/5,
    append_output/3,
    state_outputs/2,
    nr/2,
    nf/2,
    fs/2,
    ofs/2,
    print_fields/3,
    print_item/3
]).

:- meta_predicate process_all(3, 4, +, -).

mode(process_all(+, +, +, -)).
mode(item_field(+, +, -)).
mode(item_field_count(+, -)).
mode(state_counter(+, -)).
mode(increment_counter(+, -)).
mode(text_file_reader(+, +, -, +, -)).
mode(append_output(+, +, -)).
mode(state_outputs(+, -)).
mode(nr(+, -)).
mode(nf(+, -)).
mode(fs(+, -)).
mode(ofs(+, -)).
mode(print_fields(+, +, -)).
mode(print_item(+, +, -)).

process_all(Reader, Handler, State0, StateN) :-
    call(Reader, Item, State0, State1),
    (   Item == end_of_file
    ->  StateN = State1
    ;   call(Handler, Item, State1, State2, Continue),
        (   Continue == yes
        ->  process_all(Reader, Handler, State2, StateN)
        ;   StateN = State2
        )
    ).

item_field(0, record(text, Line, _Fields), Line) :-
    !.

item_field(Index, record(text, _Line, Fields), Value) :-
    integer(Index),
    Index > 0,
    nth1(Index, Fields, Value).

item_field(Index, record(text, Fields), Value) :-
    integer(Index),
    Index > 0,
    nth1(Index, Fields, Value).

item_field_count(record(text, _Line, Fields), Count) :-
    length(Fields, Count).

item_field_count(record(text, Fields), Count) :-
    length(Fields, Count).

state_counter(state(_InputStreams, _OutputStreams, Counter, _UserFields), Counter).

increment_counter(
    state(InputStreams, OutputStreams, Counter0, UserFields),
    state(InputStreams, OutputStreams, CounterN, UserFields)
) :-
    CounterN is Counter0 + 1.
text_file_reader(Path, FieldSeparator, Item, State0, StateN) :-
    State0 = state(InputStreams0, OutputStreams, Counter, UserFields),
    (   InputStreams0 = text_reader(Path, [Item | Rest])
    ->  InputStreamsN = text_reader(Path, Rest)
    ;   InputStreams0 = []
    ->  read_file_to_string(Path, Content, []),
        text_records(Content, FieldSeparator, Records),
        Records = [Item | Rest],
        InputStreamsN = text_reader(Path, Rest)
    ),
    StateN = state(InputStreamsN, OutputStreams, Counter, UserFields).

text_records(Content, FieldSeparator, Records) :-
    split_string(Content, "\n", "\r", Lines0),
    drop_final_empty_line(Lines0, Lines),
    maplist(text_record(FieldSeparator), Lines, DataRecords),
    append(DataRecords, [end_of_file], Records).

drop_final_empty_line([], []).
drop_final_empty_line([""], []) :-
    !.
drop_final_empty_line([Line | Lines0], [Line | Lines]) :-
    drop_final_empty_line(Lines0, Lines).

text_record(FieldSeparator, Line, record(text, Line, Fields)) :-
    split_string(Line, FieldSeparator, "", Fields).

append_output(Output, state(InputStreams, OutputStreams0, Counter, UserFields),
              state(InputStreams, OutputStreamsN, Counter, UserFields)) :-
    normalize_outputs(OutputStreams0, Outputs0),
    append(Outputs0, [Output], OutputsN),
    OutputStreamsN = outputs(OutputsN).

state_outputs(state(_InputStreams, OutputStreams, _Counter, _UserFields), Outputs) :-
    normalize_outputs(OutputStreams, Outputs).

normalize_outputs(outputs(Outputs), Outputs) :-
    !.
normalize_outputs([], []) :-
    !.
normalize_outputs(Outputs, Outputs).
nr(State, Number) :-
    state_counter(State, Number).

nf(Item, Count) :-
    item_field_count(Item, Count).

fs(state(_InputStreams, _OutputStreams, _Counter, UserFields), Separator) :-
    (   UserFields = plawk_options(Separator0, _OutputSeparator)
    ->  Separator = Separator0
    ;   Separator = " "
    ).

ofs(state(_InputStreams, _OutputStreams, _Counter, UserFields), Separator) :-
    (   UserFields = plawk_options(_FieldSeparator, Separator0)
    ->  Separator = Separator0
    ;   Separator = " "
    ).

print_fields(Fields, State0, StateN) :-
    ofs(State0, Separator),
    atomic_list_concat(Fields, Separator, LineAtom),
    atom_string(LineAtom, Line),
    append_output(Line, State0, StateN).

print_item(Item, State0, StateN) :-
    item_field(0, Item, Line),
    append_output(Line, State0, StateN).
