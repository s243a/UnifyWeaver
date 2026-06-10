% SPDX-License-Identifier: MIT
% Copyright (c) 2026 John William Creighton (s243a)

:- begin_tests(plawk_core).

:- use_module(plawk_core).

test(process_all_stops_on_explicit_eof) :-
    Reader = test_reader([record(text, ["one"]), end_of_file]),
    Handler = count_handler,
    process_all(Reader, Handler, state([], [], 0, none), StateN),
    state_counter(StateN, 1).

test(process_all_honors_break) :-
    Reader = test_reader([
        record(text, ["one"]),
        record(text, ["two"]),
        end_of_file
    ]),
    Handler = break_after_one_handler,
    process_all(Reader, Handler, state([], [], 0, none), StateN),
    state_counter(StateN, 1).

test(item_field_uses_awk_style_one_based_indexing) :-
    item_field(2, record(text, ["alpha", "beta", "gamma"]), "beta").

test(item_field_zero_returns_original_text_record) :-
    item_field(0, record(text, "alpha beta gamma", ["alpha", "beta", "gamma"]),
               "alpha beta gamma").

test(item_field_count_returns_text_field_count) :-
    item_field_count(record(text, ["alpha", "beta", "gamma"]), 3).

test(increment_counter_updates_state_counter) :-
    increment_counter(state([], [], 41, data), StateN),
    StateN = state([], [], 42, data).

test(nr_reflects_state_counter) :-
    nr(state([], [], 7, none), 7).

test(nf_reflects_current_item_field_count) :-
    nf(record(text, "a b c", ["a", "b", "c"]), 3).

test(field_separator_defaults_to_space) :-
    fs(state([], [], 0, none), " ").

test(output_field_separator_defaults_to_space) :-
    ofs(state([], [], 0, none), " ").

test(field_separators_can_be_read_from_user_fields) :-
    State = state([], [], 0, plawk_options("|", ",")),
    fs(State, "|"),
    ofs(State, ",").

test(print_fields_joins_values_with_ofs) :-
    State0 = state([], [], 0, plawk_options(" ", ",")),
    print_fields(["alpha", "beta", "gamma"], State0, StateN),
    state_outputs(StateN, ["alpha,beta,gamma"]).

test(print_item_emits_original_record_text) :-
    print_item(record(text, "alpha beta", ["alpha", "beta"]),
               state([], [], 0, none), StateN),
    state_outputs(StateN, ["alpha beta"]).

test(text_file_reader_yields_records_then_eof) :-
    Reader = text_file_reader('examples/plawk/core/testdata/sample_log.txt', " "),
    call(Reader, Item1, state([], [], 0, none), State1),
    call(Reader, Item2, State1, State2),
    call(Reader, Item3, State2, _State3),
    Item1 = record(text, "INFO boot", ["INFO", "boot"]),
    Item2 = record(text, "ERROR disk", ["ERROR", "disk"]),
    Item3 = end_of_file.

test(append_output_accumulates_in_state_order) :-
    append_output("first", state([], [], 0, none), State1),
    append_output("second", State1, State2),
    state_outputs(State2, ["first", "second"]).

test(process_all_collects_error_lines) :-
    Reader = text_file_reader('examples/plawk/core/testdata/sample_log.txt', " "),
    Handler = collect_errors_handler,
    process_all(Reader, Handler, state([], [], 0, none), StateN),
    state_counter(StateN, 2),
    state_outputs(StateN, ["ERROR disk"]).

test_reader(Items, Item, State0, StateN) :-
    State0 = state(InputStreams0, OutputStreams, Counter, UserFields),
    (   InputStreams0 = reader_queue([Item | Rest])
    ->  InputStreamsN = reader_queue(Rest)
    ;   InputStreams0 = []
    ->  Items = [Item | Rest],
        InputStreamsN = reader_queue(Rest)
    ),
    StateN = state(InputStreamsN, OutputStreams, Counter, UserFields).

count_handler(_Item, State0, StateN, yes) :-
    increment_counter(State0, StateN).

break_after_one_handler(_Item, State0, StateN, Continue) :-
    increment_counter(State0, StateN),
    state_counter(StateN, Counter),
    (   Counter >= 1
    ->  Continue = no
    ;   Continue = yes
    ).

collect_errors_handler(Item, State0, StateN, yes) :-
    increment_counter(State0, State1),
    (   item_field(1, Item, "ERROR")
    ->  item_field(0, Item, Line),
        append_output(Line, State1, StateN)
    ;   StateN = State1
    ).

:- end_tests(plawk_core).
