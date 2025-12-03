
:- use_module('src/unifyweaver/targets/go_target').

% Mock data for testing
test_json_data('{"users": [{"name": "alice", "tags": ["admin", "dev"]}, {"name": "bob", "tags": ["user"]}]}').

% Predicate using array iteration and nested extraction
user_tag(Name, Tag) :-
    json_get([users], UserList),
    json_array_member(UserList, User),
    json_get(User, [name], Name),
    json_get(User, [tags], Tags),
    json_array_member(Tags, Tag).

test_go_json_arrays :-
    % Compile the predicate
    compile_predicate_to_go(user_tag/2, [json_input(true)], Code),
    
    % Write to file
    write_go_program(Code, 'output/user_tag.go'),
    
    % Run it
    shell('go build -o output/user_tag output/user_tag.go'),
    shell('echo \'{"users": [{"name": "alice", "tags": ["admin", "dev"]}, {"name": "bob", "tags": ["user"]}]}\' | ./output/user_tag > output/user_tag.txt'),
    
    % Verify output
    read_file_to_string('output/user_tag.txt', Output, []),
    (   sub_string(Output, _, _, _, "alice:admin"),
        sub_string(Output, _, _, _, "alice:dev"),
        sub_string(Output, _, _, _, "bob:user")
    ->  format('Test passed: JSON array iteration~n')
    ;   format('Test failed: Output was:~n~s~n', [Output]),
        fail
    ).
