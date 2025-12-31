:- use_module('../../src/unifyweaver/sources/yaml_source').

% Define the test predicate
test_yaml_users :-
    yaml_source:compile_source(get_users/2, [
        yaml_file('examples/yaml_test/data.yaml'),
        yaml_filter('data["users"]')
    ], [], Code),
    write(Code).

:- initialization(test_yaml_users, main).
