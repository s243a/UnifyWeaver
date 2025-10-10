:- encoding(utf8).
% test_policy.pl - Tests for the firewall and preference system.

:- module(test_policy, [test_policy/0]).

:- use_module(library(recursive_compiler)).
:- use_module(library(firewall)).
:- use_module(library(preferences)).
:- use_module(library(stream_compiler)).

% --- Test Definitions ---

% Define some dummy predicates for testing.
:- assertz(user:(allowed_pred(a) :- true)).
:- assertz(user:(denied_pred(a) :- true)).
:- assertz(user:(pref_pred(a) :- true)).
:- assertz(user:(default_pred(a) :- true)).

% --- Test Runner ---

test_policy :- 
    writeln('=== Testing Firewall and Preference Policies ==='),
    setup_policies,
    run_test(test_allowed_succeeds),
    run_test(test_denied_fails),
    run_test(test_rule_preference_overrides_default),
    run_test(test_runtime_overrides_all),
    run_test(test_compile_facts_direct),
    writeln('\n=== All Policy Tests Complete ===').

%% test_compile_facts_direct
% Directly calls compile_facts for allowed_pred/1 and checks if BashCode is instantiated.
test_compile_facts_direct :-
    format('~n--- Running test_compile_facts_direct ---~n'),
    
    % Define a simple test predicate
    catch(abolish(direct_test_pred/1), _, true),
    assertz(direct_test_pred(test_value)),
    
    % Call compile_facts directly (bypassing policy_validator)
    % This should work regardless of firewall settings
    (   stream_compiler:compile_facts(direct_test_pred, 1, [], BashCode),
        nonvar(BashCode),
        BashCode \= '' ->
        format('  ✓ PASS: test_compile_facts_direct~n')
    ;   format('  ✗ FAIL: test_compile_facts_direct - BashCode not instantiated~n'),
        fail
    ).

%% setup_policies is det.
%  Asserts all firewall and preference rules at runtime.
setup_policies :-
    assertz(firewall:rule_firewall(allowed_pred/1, [execution([bash, python]), services([sql])])),
    assertz(firewall:rule_firewall(denied_pred/1, [denied([bash])])),
    assertz(preferences:rule_preferences(pref_pred/1, [prefer([python]), optimization(speed)])),
    assertz(preferences:preferences_default([prefer([bash]), optimization(balance)])).

run_test(Test) :-
    format('\n--- Running ~w ---\n', [Test]),
    (
        call(Test) ->
        format('  \u2705 PASS: ~w\n', [Test])
    ;   format('  \u274c FAIL: ~w\n', [Test])
    ).


% --- Test Cases ---

%% test_allowed_succeeds
% This should succeed because 'bash' is in the allowed list for allowed_pred/1.
test_allowed_succeeds :-
    compile_recursive(allowed_pred/1, [], _).

%% test_denied_fails
% This should fail because 'bash' is in the denied list for denied_pred/1.
% The compilation predicate should fail, so the negation should succeed.
test_denied_fails :-
    \+ compile_recursive(denied_pred/1, [], _).

%% test_rule_preference_overrides_default
% Checks that the rule-specific preference for 'python' is chosen over
% the global default of 'bash'.
test_rule_preference_overrides_default :-
    preferences:get_final_options(pref_pred/1, [], FinalOptions),
    member(prefer([python]), FinalOptions).

%% test_runtime_overrides_all
% Checks that a runtime option overrides both the rule-specific and
% default preferences.
test_runtime_overrides_all :-
    preferences:get_final_options(pref_pred/1, [prefer([csharp])], FinalOptions),
    member(prefer([csharp]), FinalOptions).
