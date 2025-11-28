:- encoding(utf8).
% Generate person/1 AWK script

:- use_module('src/unifyweaver/targets/awk_target').

% Simple facts for testing
person(alice).
person(bob).
person(charlie).

main :-
    write('Generating person/1 AWK script...'), nl,
    awk_target:compile_predicate_to_awk(person/1,
        [record_format(tsv), unique(true)], AwkCode),
    awk_target:write_awk_script(AwkCode, 'person_filter.awk'),
    write('Done! Script written to person_filter.awk'), nl,
    halt.

:- initialization(main).
