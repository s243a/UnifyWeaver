% Test Go code generation from Phase 7-8 Prolog predicates
% Usage: swipl -g 'run_tests' -t halt test_go_codegen.pl

% Import the actual go_target module
:- ['../../src/unifyweaver/targets/go_target.pl'].

% Generate small-world Go code with custom options
generate_small_world_go :-
    format('Generating Go small-world code...~n'),
    Options = [k_local(10), k_long(5), alpha(2.0)],
    compile_small_world_proper_go(Options, Code),

    % Write to file
    open('generated/smallworld/smallworld.go', write, Stream),
    write(Stream, Code),
    close(Stream),
    format('  Written to generated/smallworld/smallworld.go~n').

% Generate multi-interface Go code
generate_multi_interface_go :-
    format('Generating Go multi-interface code...~n'),
    Options = [gamma(2.5), min_interfaces(1), max_interfaces(100)],
    compile_multi_interface_node_go(Options, Code),

    % Write to file
    open('generated/multiinterface/multiinterface.go', write, Stream),
    write(Stream, Code),
    close(Stream),
    format('  Written to generated/multiinterface/multiinterface.go~n').

% Run all generation tests
run_tests :-
    format('~n=== Phase 7-8 Go Code Generation ===~n~n'),
    % Create output directories
    (exists_directory('generated') -> true ; make_directory('generated')),
    (exists_directory('generated/smallworld') -> true ; make_directory('generated/smallworld')),
    (exists_directory('generated/multiinterface') -> true ; make_directory('generated/multiinterface')),

    % Generate code
    (generate_small_world_go -> format('[PASS] Small-world generation~n') ; format('[FAIL] Small-world generation~n')),
    (generate_multi_interface_go -> format('[PASS] Multi-interface generation~n') ; format('[FAIL] Multi-interface generation~n')),
    format('~n=== Generation Complete ===~n').
