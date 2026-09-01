% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (s243a)
%
% test_cli_args.pl -- a plunit port of peerhailer's `test/cliArgs.test.mjs`
% (the 17-test contract matrix from docs/cli-arg-parsing.md), driving the
% Prolog reference parser in cli_args.pl.
%
% Test names and numbering follow the corpus one-for-one; the JS `assert.throws`
% cases become `error(Message)` assertions, with the corpus's regex rendered as
% a substring check (every corpus regex is a plain literal).
%
%   swipl -q -g test_cli_args -t halt examples/cli_args/test_cli_args.pl

:- module(test_cli_args, [test_cli_args/0]).

:- use_module(library(plunit)).
:- use_module(cli_args).

% ---------------------------------------------------------------------------
% Harness helpers
% ---------------------------------------------------------------------------

%!  of(+Line, -Result) is det.
%
%   The corpus's `const of = (line) => parseArgs(line.split(" ").filter(Boolean))`.
of(Line, Result) :-
    split_string(Line, " ", "", Raw),
    exclude(==(""), Raw, Tokens),
    parse_args(Tokens, Result).

%!  ok_of(+Line, -Positional, -Flags) is semidet.
ok_of(Line, Positional, Flags) :-
    of(Line, ok(Positional, Flags)).

%!  flag(+Flags, +Key, -Value) is semidet.
flag([K-V|Rest], Key, Value) :-
    (   K == Key
    ->  Value = V
    ;   flag(Rest, Key, Value)
    ).

%!  no_flag(+Flags, +Key) is semidet.   % JS `flags.x === undefined`
no_flag(Flags, Key) :-
    \+ flag(Flags, Key, _).

%!  error_matching(+Line, +Needle) is semidet.
%
%   The corpus asserts `assert.throws(..., /needle/)`; every corpus pattern is a
%   literal, so a substring test is the faithful rendering.
error_matching(Line, Needle) :-
    of(Line, error(Message)),
    contains_substring(Message, Needle).

contains_substring(Haystack, Needle) :-
    sub_string(Haystack, _, _, _, Needle),
    !.

%!  contains_token(+List, +Token) is semidet.
contains_token([X|Xs], Token) :-
    (   X == Token
    ->  true
    ;   contains_token(Xs, Token)
    ).

% ---------------------------------------------------------------------------
% The contract matrix
% ---------------------------------------------------------------------------

:- begin_tests(cli_args).

% 1 & 2: a boolean flag works before or after the positional
test('1 & 2: a boolean flag works before or after the positional') :-
    forall(member(Line, ["block --include-key bob", "block bob --include-key"]),
           (   ok_of(Line, Positional, Flags),
               assertion(Positional == ["block", "bob"]),
               assertion(flag(Flags, "include-key", true))
           )).

% 3: an unknown option is refused against the command schema
test('3: an unknown option is refused against the command schema') :-
    assertion(of("block bob --include-keey", error(_))),
    assertion(error_matching("block bob --include-keey",
                             "unknown option --include-keey")).

% 4: --include-key=false is off; =anything-else is on
test('4: --include-key=false is off; =anything-else is on') :-
    ok_of("block bob --include-key=false", _, F1),
    assertion(flag(F1, "include-key", false)),
    ok_of("block bob --include-key=yes", _, F2),
    assertion(flag(F2, "include-key", true)).

% 5: a value given to a boolean flag lands as an extra positional and fails
test('5: a value given to a boolean flag lands as an extra positional and fails') :-
    assertion(error_matching("block bob --include-key yes", "extra argument: yes")).

% 6: --state selects the same state before or after the command
test('6: --state selects the same state before or after the command') :-
    ok_of("--state P block bob", _, F1),
    assertion(flag(F1, "state", "P")),
    ok_of("block bob --state P", _, F2),
    assertion(flag(F2, "state", "P")).

% 7: `--` preserves a forwarded command with its own flags
test('7: `--` preserves a forwarded command with its own flags') :-
    ok_of("commands add deploy -- ./run.sh --env prod", Positional, _),
    % The handler reads rest.slice(2).join(" "); rest is positional after the command.
    length(Prefix, 3),
    append(Prefix, Tail, Positional),
    assertion(Tail == ["./run.sh", "--env", "prod"]).

% 8: after `--`, a --state token is payload, not the global option
test('8: after `--`, a --state token is payload, not the global option') :-
    ok_of("commands add deploy -- ./run.sh --state child.json", Positional, Flags),
    assertion(no_flag(Flags, "state")),                       % not swallowed as the global
    assertion(contains_token(Positional, "--state")),         % kept as payload
    assertion(contains_token(Positional, "child.json")).

% 9: --key keeps an inline dash-leading PEM as its value
test('9: --key keeps an inline dash-leading PEM as its value') :-
    Pem = "-----BEGIN-PUBLIC-KEY-----",
    parse_args(["add", "bob", "--key", Pem], ok(_, Flags)),
    assertion(flag(Flags, "key", Pem)).

% 10: --debug is bare-true, spaced, or =valued
test('10: --debug is bare-true, spaced, or =valued') :-
    ok_of("daemon --debug", _, F1),
    assertion(flag(F1, "debug", true)),
    ok_of("daemon --debug 2", _, F2),
    assertion(flag(F2, "debug", "2")),
    ok_of("daemon --debug=2", _, F3),
    assertion(flag(F3, "debug", "2")).

% 11: a value handed to a boolean daemon flag fails loudly
test('11: a value handed to a boolean daemon flag fails loudly') :-
    assertion(error_matching("daemon --require-target-binding yes",
                             "extra argument: yes")).

% 12: --force is refused on `profiles pin` -- it belongs to `profiles remove`
test('12: --force is refused on `profiles pin` - it belongs to `profiles remove`') :-
    assertion(error_matching("profiles pin trusted --force", "unknown option --force")),
    % ...and is accepted on remove.
    ok_of("profiles remove temp --force", _, Flags),
    assertion(flag(Flags, "force", true)).

% 13: a string option with no value fails
test('13: a string option with no value fails') :-
    assertion(error_matching("add bob --profile", "--profile needs a value")).

% a missing required positional fails with the argument name
test('a missing required positional fails with the argument name') :-
    assertion(error_matching("block", "missing argument: name")).

% an unmigrated command falls back to the lenient parse (no schema, no error)
test('an unmigrated command falls back to the lenient parse (no schema, no error)') :-
    % `tunnels` has no schema yet: unknown flags are accepted, booleans stay
    % greedy -- exactly the legacy behaviour.
    ok_of("tunnels add acp 127.0.0.1:9100 --anything here", Positional, Flags),
    Positional = [First|_],
    assertion(First == "tunnels"),
    assertion(flag(Flags, "anything", "here")).

% regression: daemon accepts --hail-on-encrypted / --hail-on-tls (Fable)
test('regression: daemon accepts --hail-on-encrypted / --hail-on-tls (Fable)') :-
    ok_of("daemon --hail-on-encrypted tailscale0", _, F1),
    assertion(flag(F1, "hail-on-encrypted", "tailscale0")),
    ok_of("daemon --hail-on-tls eth0", _, F2),
    assertion(flag(F2, "hail-on-tls", "eth0")).

% regression: unblock --key with no name is valid (Fable)
test('regression: unblock --key with no name is valid (Fable)') :-
    ok_of("unblock --key ABCDEF12", Positional, Flags),
    assertion(flag(Flags, "key", "ABCDEF12")),
    assertion(Positional == ["unblock"]),          % no name required
    % ...and the name form still works.
    ok_of("unblock bob", Positional2, _),
    assertion(Positional2 == ["unblock", "bob"]).

% the lenient fallback is verbatim: --a --b=c keeps the old greedy reading
test('the lenient fallback is verbatim: --a --b=c keeps the old greedy reading') :-
    % An unmigrated command must behave exactly as before: the old parser read
    % `--b=c` (which is not a bare --word) as the value of --a.
    ok_of("tunnels --a --b=c", _, Flags),
    assertion(flag(Flags, "a", "--b=c")).

:- end_tests(cli_args).

% ---------------------------------------------------------------------------
% Entry point
% ---------------------------------------------------------------------------

%!  test_cli_args is det.
%
%   Runs the suite and prints an explicit count on stdout, so the pass total is
%   visible even under `swipl -q` (which silences plunit's own summary).
test_cli_args :-
    test_count(Total),
    (   run_tests(cli_args)
    ->  format("cli_args contract matrix: ~w/~w tests passed~n", [Total, Total])
    ;   format(user_error, "cli_args contract matrix: FAILED (of ~w tests)~n", [Total]),
        halt(1)
    ).

test_count(Total) :-
    findall(Name, plunit:current_test(cli_args, Name, _, _, _), Names),
    length(Names, Total).
