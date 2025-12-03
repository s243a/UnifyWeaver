:- use_module('src/unifyweaver/targets/go_target').

adults(Name, Age) :-
    json_record([name-Name, age-Age]),
    Age >= 30.

test :-
    catch(
        (compile_predicate_to_go(adults/2, [
            db_backend(bbolt),
            db_file('test.db'),
            db_bucket(users),
            db_mode(read),
            package(main)
        ], Code),
        format('SUCCESS: Generated ~w chars~n', [Code])),
        E,
        format('ERROR: ~w~n', [E])
    ).

:- initialization(test, main).
