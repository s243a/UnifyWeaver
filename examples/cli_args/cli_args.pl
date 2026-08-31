% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2026 John William Creighton (s243a)
%
% cli_args.pl -- a Prolog reference implementation of peerhailer's CLI argument
% parser (`src/cliArgs.js`, peerhailer @ 08ad35e), byte-for-byte behaviour-
% compatible with the JavaScript oracle vendored at `oracle/cliArgs.js`.
%
% This is step A1 of the UnifyWeaver transpilation maturity demo: the Prolog
% here is the source of truth that later steps push through the JS targets, so
% it mirrors the oracle exactly rather than "cleaning it up".
%
% Result convention
% -----------------
%   parse_args(+Argv, -Result)
%     Argv   : list of SWI *strings* (the raw tokens; `process.argv.slice(2)`).
%     Result : ok(Positional, Flags)  |  error(Message)
%       Positional : list of strings, in the oracle's `positional` order.
%       Flags      : list of Key-Value pairs. Key is a string; Value is a
%                    string or one of the atoms `true` / `false`. The list is
%                    in the oracle's JS object key-insertion order (leading
%                    globals first, then parse order; a later assignment to an
%                    existing key overwrites the value but keeps its position).
%       Message    : the exact string the oracle's `CliError` carries.
%     parse_args/2 never throws for a usage error and never fails; the JS
%     transpile maps error(M) -> `throw new CliError(M)` at the module edge.
%
% Everything is pure: no assert/retract, no exceptions, no library(pcre) (both
% oracle regexes are re-expressed as character logic so the compilers only ever
% see first-order Prolog).

:- module(cli_args, [
    parse_args/2,               % +Argv, -Result
    parse_args/3,               % +Argv, +Registry, -Result
    default_registry/1,         % -Registry
    global_options/1,           % -Options
    is_long_flag/1,             % +Token          (the strict parser's flag test)
    looks_like_legacy_flag/1    % +Token          (the lenient parser's flag test)
]).

% ===========================================================================
% 1. The grammar, as data
% ===========================================================================
%
% Registry = list of Name-Entry pairs (a plain ordered assoc; the oracle's
% `COMMANDS` object, with JS key-insertion order preserved).
%
%   Entry  ::= schema(Options, Positionals)        % a leaf command
%            | group(Actions)                      % grouped: keyed by 1st positional
%   Actions::= list of ActionName-schema(Options, Positionals)
%   Options::= list of OptionName-Kind
%   Kind   ::= boolean | string | optional
%   Positionals ::= list of names; "[name]" is optional, "...name" is variadic.
%
% `route` reads exactly as the oracle's grouped entry does:
%
%   "route"-group([
%       "discover"-schema(["dest"-string, "dest-file"-string,
%                          "control"-string, "port"-string], []),
%       ... ])

%!  global_options(-Options) is det.
%
%   The oracle's GLOBAL_OPTIONS: accepted by every command, wherever they appear.
global_options([
    "state"-string,
    "name"-string
]).

%!  default_registry(-Registry) is det.
%
%   The oracle's COMMANDS. parse_args/3 accepts any registry of this shape,
%   mirroring the JS `registry?` parameter.
default_registry([
    "block"-schema(["include-key"-boolean], ["name"]),

    % name is optional: `unblock --key <pem|fingerprint>` lifts a key with no name.
    "unblock"-schema(["key"-string], ["[name]"]),

    "add"-schema([ "profile"-string,
                   "until"-string,
                   "key"-string,
                   "key-file"-string,
                   "transport"-string ],
                 ["name", "[address]"]),

    "daemon"-schema([ "debug"-optional,
                      "port"-string,
                      "host"-string,
                      "hail-on"-string,
                      "hail-on-encrypted"-string,
                      "hail-on-tls"-string,
                      "tls-cert"-string,
                      "tls-key"-string,
                      "allow-origin"-string,
                      "chat"-boolean,
                      "route"-boolean,
                      "ui"-boolean,
                      "require-target-binding"-boolean,
                      "require-sealed"-boolean ],
                    []),

    "commands"-group([
        "add"-schema([], ["name", "...command"]),
        "remove"-schema([], ["name"]),
        "list"-schema([], [])
    ]),

    % Live operations on a running daemon's in-memory routed key store (M3b).
    "route"-group([
        "discover"-schema([ "dest"-string, "dest-file"-string,
                            "control"-string, "port"-string ], []),
        "status"-schema([ "dest"-string, "dest-file"-string,
                          "control"-string, "port"-string ], []),
        "approve"-schema([ "dest"-string, "dest-file"-string,
                           "seal-key"-string, "seal-key-file"-string,
                           "control"-string, "port"-string ], []),
        "send"-schema([ "dest"-string, "dest-file"-string, "public"-boolean,
                        "ttl"-string, "budget"-string,
                        "control"-string, "port"-string ],
                      ["...message"])
    ]),

    "profiles"-group([
        "add"-schema(["allows"-string, "description"-string], ["name"]),
        "remove"-schema(["force"-boolean, "reassign"-string], ["name"]),
        "pin"-schema([], ["name"]),
        "unpin"-schema([], ["name"]),
        "list"-schema([], [])
    ])
]).

% ---------------------------------------------------------------------------
% JS `Object.prototype` keys.
%
% The oracle looks options and commands up with plain property access on object
% literals (`key in GLOBAL_OPTIONS`, `options[key]`, `registry[command]`), so an
% inherited Object.prototype member answers as a *present, truthy* entry. That
% is observable behaviour, so it is modelled here rather than papered over.
% See README "Oracle subtleties" #7.
% ---------------------------------------------------------------------------
js_object_prototype_keys([
    "constructor",
    "hasOwnProperty",
    "isPrototypeOf",
    "propertyIsEnumerable",
    "toLocaleString",
    "toString",
    "valueOf",
    "__defineGetter__",
    "__defineSetter__",
    "__lookupGetter__",
    "__lookupSetter__",
    "__proto__"
]).

js_object_prototype_key(Key) :-
    js_object_prototype_keys(Keys),
    string_member(Key, Keys).

% ===========================================================================
% 2. The two flag regexes, as character logic
% ===========================================================================

%!  is_long_flag(+Token) is semidet.
%
%   The strict parser's notion: JS `/^--[a-z][a-z0-9-]*(=|$)/i`.
%   After `--` the first character must be a letter, then the maximal run of
%   [A-Za-z0-9-] must be followed by `=` or by end-of-string. (Backtracking to
%   a shorter run can never help: the character after a shorter run is itself a
%   run character, hence neither `=` nor the end.)
%
%   Deliberately false for a dash-leading PEM such as
%   `-----BEGIN-PUBLIC-KEY-----`, whose third character is `-`, not a letter --
%   which is why `--key <inline PEM>` keeps working.
is_long_flag(Token) :-
    string_chars(Token, Chars),
    Chars = ['-', '-', First | Rest],
    js_alpha(First),
    long_flag_tail(Rest).

long_flag_tail([]).
long_flag_tail([C|Cs]) :-
    (   C == '='
    ->  true
    ;   js_flag_char(C),
        long_flag_tail(Cs)
    ).

%!  looks_like_legacy_flag(+Token) is semidet.
%
%   The *legacy* parser's flag test: JS `/^--[a-z][a-z0-9-]*$/i` -- a bare
%   `--word` only. Used solely by parse_lenient/3, and the reason
%   `tunnels --a --b=c` reads `--b=c` as the value of `--a`.
looks_like_legacy_flag(Token) :-
    string_chars(Token, Chars),
    Chars = ['-', '-', First | Rest],
    js_alpha(First),
    legacy_flag_tail(Rest).

legacy_flag_tail([]).
legacy_flag_tail([C|Cs]) :-
    js_flag_char(C),
    legacy_flag_tail(Cs).

js_alpha(C) :-
    char_code(C, X),
    (   X >= 0'a, X =< 0'z
    ->  true
    ;   X >= 0'A, X =< 0'Z
    ).

js_flag_char(C) :-
    char_code(C, X),
    (   X >= 0'a, X =< 0'z
    ->  true
    ;   X >= 0'A, X =< 0'Z
    ->  true
    ;   X >= 0'0, X =< 0'9
    ->  true
    ;   X =:= 0'-
    ).

% ===========================================================================
% 3. Small pure string / list helpers
% ===========================================================================

%!  starts_with(+String, +Prefix) is semidet.
starts_with(String, Prefix) :-
    string_length(String, L),
    string_length(Prefix, N),
    L >= N,
    sub_string(String, 0, N, _, Sub),
    Sub == Prefix.

%!  substring_from(+String, +Start, -Sub) is det.  % JS String#slice(Start)
substring_from(String, Start, Sub) :-
    string_length(String, L),
    Len is L - Start,
    sub_string(String, Start, Len, 0, Sub).

%!  substring_range(+String, +Start, +End, -Sub) is det.  % JS slice(Start, End)
substring_range(String, Start, End, Sub) :-
    Len is End - Start,
    sub_string(String, Start, Len, _, Sub).

%!  first_equals_index(+String, -Index) is det.
%
%   JS `String#indexOf("=")`: the 0-based index, or -1 when absent.
first_equals_index(String, Index) :-
    string_chars(String, Chars),
    first_char_index(Chars, '=', 0, Index).

first_char_index([], _Target, _I, -1).
first_char_index([C|Cs], Target, I, Index) :-
    (   C == Target
    ->  Index = I
    ;   I1 is I + 1,
        first_char_index(Cs, Target, I1, Index)
    ).

%!  split_flag_token(+Token, -Key, -Inline) is det.
%
%   Mirrors the oracle's `equals`/`key`/`inlineValue` triple.
%   Inline is `none` or `some(Value)`. Token is known to start with `--`, so the
%   `=` (if any) is always at index >= 2.
split_flag_token(Token, Key, Inline) :-
    first_equals_index(Token, Eq),
    (   Eq >= 0
    ->  substring_range(Token, 2, Eq, Key),
        ValueStart is Eq + 1,
        substring_from(Token, ValueStart, Value),
        Inline = some(Value)
    ;   substring_from(Token, 2, Key),
        Inline = none
    ).

%!  string_member(+String, +List) is semidet.
string_member(S, [X|Xs]) :-
    (   S == X
    ->  true
    ;   string_member(S, Xs)
    ).

%!  pair_lookup(+Pairs, +Key, -Value) is semidet.
%
%   Own-property lookup in an ordered assoc list (fails when absent).
pair_lookup([K-V|Rest], Key, Value) :-
    (   K == Key
    ->  Value = V
    ;   pair_lookup(Rest, Key, Value)
    ).

%!  nth0_default(+Index, +List, +Default, -Elem) is det.  % JS `xs[i] ?? d`
%
%   One clause with an explicit conditional so the call stays deterministic
%   (the index, not the list, is the first argument).
nth0_default(I, List, Default, Elem) :-
    (   List = [X|Xs]
    ->  (   I =:= 0
        ->  Elem = X
        ;   I1 is I - 1,
            nth0_default(I1, Xs, Default, Elem)
        )
    ;   Elem = Default
    ).

%!  last_element(+List, -Last) is semidet.  % fails on []
last_element([X|Xs], Last) :-
    (   Xs == []
    ->  Last = X
    ;   last_element(Xs, Last)
    ).

% ---------------------------------------------------------------------------
% Flag maps: ordered Key-Value lists with JS object-assignment semantics.
% ---------------------------------------------------------------------------

%!  flags_set(+Flags0, +Key, +Value, -Flags) is det.
%
%   `obj[Key] = Value`: overwrite in place (keeping the original position) or
%   append at the end. Assigning to `__proto__` on an object literal invokes the
%   inherited accessor, which ignores a primitive -- so no property is created.
flags_set(Flags0, Key, Value, Flags) :-
    (   Key == "__proto__"
    ->  Flags = Flags0
    ;   flags_put(Flags0, Key, Value, Flags)
    ).

flags_put([], Key, Value, [Key-Value]).
flags_put([K-V|Rest], Key, Value, Out) :-
    (   K == Key
    ->  Out = [K-Value|Rest]
    ;   Out = [K-V|Rest1],
        flags_put(Rest, Key, Value, Rest1)
    ).

%!  merge_flags(+Base, +Overlay, -Merged) is det.
%
%   JS `{ ...Base, ...Overlay }`: Overlay's value wins, Base's position wins.
%   The accumulator loop puts the driving list first so first-argument indexing
%   keeps it deterministic.
merge_flags(Base, Overlay, Merged) :-
    merge_flags_(Overlay, Base, Merged).

merge_flags_([], Base, Base).
merge_flags_([K-V|Rest], Base, Merged) :-
    flags_set(Base, K, V, Base1),
    merge_flags_(Rest, Base1, Merged).

% ===========================================================================
% 4. schemaFor
% ===========================================================================

%!  schema_for(+Command, +MaybeAction, +Registry, -Schema, -ActionConsumed) is semidet.
%
%   Fails where the oracle returns `null` (unmigrated command, or a grouped
%   command with an unknown/absent action) -- which signals the lenient fallback.
%   MaybeAction is `none` or `some(String)`; JS treats both `undefined` and the
%   empty string as falsy, hence the `A \== ""` guard.
schema_for(Command, MaybeAction, Registry, Schema, ActionConsumed) :-
    registry_entry(Registry, Command, Entry),
    (   Entry = group(Actions)
    ->  MaybeAction = some(Action),
        Action \== "",
        action_entry(Actions, Action, Schema),
        ActionConsumed = true
    ;   Schema = Entry,
        ActionConsumed = false
    ).

registry_entry(Registry, Command, Entry) :-
    (   pair_lookup(Registry, Command, E)
    ->  Entry = E
    ;   js_object_prototype_key(Command),
        Entry = schema([], [])          % an inherited member has no .actions,
    ).                                  % no .options and no .positionals

action_entry(Actions, Action, Schema) :-
    (   pair_lookup(Actions, Action, S)
    ->  Schema = S
    ;   js_object_prototype_key(Action),
        Schema = schema([], [])
    ).

% ===========================================================================
% 5. parseLenient -- the legacy parse, kept verbatim
% ===========================================================================
%
% Greedy, untyped, no `--` handling, and its own narrower flag test.

%!  parse_lenient(+Argv, -Positional, -Flags) is det.
parse_lenient(Argv, Positional, Flags) :-
    lenient_loop(Argv, [], [], PositionalRev, Flags),
    reverse(PositionalRev, Positional).

lenient_loop([], PosAcc, FlagsAcc, PosAcc, FlagsAcc).
lenient_loop([Token|Rest], PosAcc, FlagsAcc, PosOut, FlagsOut) :-
    (   starts_with(Token, "--")
    ->  split_flag_token(Token, Key, Inline),
        (   Inline = some(Value)
        ->  flags_set(FlagsAcc, Key, Value, Flags1),
            lenient_loop(Rest, PosAcc, Flags1, PosOut, FlagsOut)
        ;   Rest = [Next|Rest1],
            \+ looks_like_legacy_flag(Next)
        ->  flags_set(FlagsAcc, Key, Next, Flags1),
            lenient_loop(Rest1, PosAcc, Flags1, PosOut, FlagsOut)
        ;   flags_set(FlagsAcc, Key, true, Flags1),
            lenient_loop(Rest, PosAcc, Flags1, PosOut, FlagsOut)
        )
    ;   lenient_loop(Rest, [Token|PosAcc], FlagsAcc, PosOut, FlagsOut)
    ).

% ===========================================================================
% 6. parseStrict
% ===========================================================================

%!  parse_strict(+Tokens, +Schema, +Positional0, -Outcome) is det.
%
%   Tokens are the argv tail after the command (and action, if consumed);
%   Positional0 already holds those. Outcome is ok(Positional, Flags) | err(Msg).
parse_strict(Tokens, schema(Options, Positionals), Positional0, Outcome) :-
    strict_loop(Tokens, false, Options, [], [], ValuesRev, Flags, Status),
    (   Status = err(Message)
    ->  Outcome = err(Message)
    ;   reverse(ValuesRev, Values),
        check_arity(Values, Positionals, ArityStatus),
        (   ArityStatus = err(Message2)
        ->  Outcome = err(Message2)
        ;   append(Positional0, Values, Positional),
            Outcome = ok(Positional, Flags)
        )
    ).

% strict_loop(+Tokens, +AfterTerminator, +Options, +ValuesAcc, +FlagsAcc,
%             -ValuesRev, -Flags, -Status)
strict_loop([], _After, _Options, ValuesAcc, FlagsAcc, ValuesAcc, FlagsAcc, ok).
strict_loop([Token|Rest], After, Options, ValuesAcc, FlagsAcc,
            ValuesRev, Flags, Status) :-
    (   After == true
    ->  strict_loop(Rest, true, Options, [Token|ValuesAcc], FlagsAcc,
                    ValuesRev, Flags, Status)
    ;   Token == "--"
    ->  % its tail is payload, never options
        strict_loop(Rest, true, Options, ValuesAcc, FlagsAcc,
                    ValuesRev, Flags, Status)
    ;   \+ starts_with(Token, "--")
    ->  strict_loop(Rest, After, Options, [Token|ValuesAcc], FlagsAcc,
                    ValuesRev, Flags, Status)
    ;   split_flag_token(Token, Key, Inline),
        (   option_kind(Options, Key, Kind)
        ->  strict_option(Kind, Key, Inline, Rest, After, Options,
                          ValuesAcc, FlagsAcc, ValuesRev, Flags, Status)
        ;   string_concat("unknown option --", Key, Message),
            ValuesRev = ValuesAcc, Flags = FlagsAcc, Status = err(Message)
        )
    ).

% strict_option(+Kind, +Key, +Inline, +Rest, +After, +Options, ...)
strict_option(Kind, Key, Inline, Rest, After, Options,
              ValuesAcc, FlagsAcc, ValuesRev, Flags, Status) :-
    (   Kind == boolean
    ->  % `--x=false` is off, anything else (including bare) is on.
        (   Inline = some(Inline1)
        ->  (   Inline1 == "false" -> Bool = false ; Bool = true )
        ;   Bool = true
        ),
        flags_set(FlagsAcc, Key, Bool, Flags1),
        strict_loop(Rest, After, Options, ValuesAcc, Flags1,
                    ValuesRev, Flags, Status)
    ;   Inline = some(Inline2)
    ->  flags_set(FlagsAcc, Key, Inline2, Flags1),
        strict_loop(Rest, After, Options, ValuesAcc, Flags1,
                    ValuesRev, Flags, Status)
    ;   % A value that is missing or looks like the next option is not a value.
        next_value(Rest, MaybeValue),
        (   Kind == string
        ->  (   MaybeValue = some(Value)
            ->  flags_set(FlagsAcc, Key, Value, Flags1),
                Rest = [_Consumed|Rest1],
                strict_loop(Rest1, After, Options, ValuesAcc, Flags1,
                            ValuesRev, Flags, Status)
            ;   string_concat("--", Key, Prefix),
                string_concat(Prefix, " needs a value", Message),
                ValuesRev = ValuesAcc, Flags = FlagsAcc, Status = err(Message)
            )
        ;   % `optional` (and the untyped-but-truthy inherited kind): a
            % following value if one is there, else the bare `true`.
            (   MaybeValue = some(Value2)
            ->  flags_set(FlagsAcc, Key, Value2, Flags1),
                Rest = [_Consumed2|Rest2],
                strict_loop(Rest2, After, Options, ValuesAcc, Flags1,
                            ValuesRev, Flags, Status)
            ;   flags_set(FlagsAcc, Key, true, Flags1),
                strict_loop(Rest, After, Options, ValuesAcc, Flags1,
                            ValuesRev, Flags, Status)
            )
        )
    ).

%!  next_value(+Rest, -MaybeValue) is det.
%
%   JS: `next !== undefined && next !== "--" && !isLongFlag(next) ? next : null`.
next_value([], none).
next_value([Next|_], MaybeValue) :-
    (   Next \== "--",
        \+ is_long_flag(Next)
    ->  MaybeValue = some(Next)
    ;   MaybeValue = none
    ).

%!  option_kind(+SchemaOptions, +Key, -Kind) is semidet.
%
%   `{ ...GLOBAL_OPTIONS, ...schema.options }[key]`, truthiness included: a
%   command option shadows a global, and an inherited Object.prototype member
%   answers truthy-but-untyped (`other`), which the oracle then treats exactly
%   like `optional`. Fails only where the oracle sees a falsy kind.
option_kind(SchemaOptions, Key, Kind) :-
    (   pair_lookup(SchemaOptions, Key, K1)
    ->  Kind = K1
    ;   global_options(Globals),
        pair_lookup(Globals, Key, K2)
    ->  Kind = K2
    ;   js_object_prototype_key(Key),
        Kind = other
    ).

% ---------------------------------------------------------------------------
% Positional arity, against the schema's names.
% ---------------------------------------------------------------------------

%!  check_arity(+Values, +Names, -Status) is det.
check_arity(Values, Names, Status) :-
    length(Values, ValueCount),
    length(Names, NameCount),
    (   last_element(Names, Last),
        starts_with(Last, "...")
    ->  Variadic = true
    ;   Variadic = false
    ),
    count_required(Names, 0, Required),
    (   ValueCount < Required
    ->  nth0_default(ValueCount, Names, "argument", RawName),
        strip_brackets(RawName, Name),
        string_concat("missing argument: ", Name, Message),
        Status = err(Message)
    ;   Variadic == false,
        ValueCount > NameCount
    ->  nth0_default(NameCount, Values, "", Extra),
        string_concat("unexpected extra argument: ", Extra, Message2),
        Status = err(Message2)
    ;   Status = ok
    ).

count_required([], Acc, Acc).
count_required([N|Ns], Acc, Required) :-
    (   starts_with(N, "[")
    ->  Acc1 = Acc
    ;   starts_with(N, "...")
    ->  Acc1 = Acc
    ;   Acc1 is Acc + 1
    ),
    count_required(Ns, Acc1, Required).

%!  strip_brackets(+String, -Stripped) is det.  % JS `.replace(/[[\]]/g, "")`
strip_brackets(String, Stripped) :-
    string_chars(String, Chars),
    drop_brackets(Chars, Kept),
    string_chars(Stripped, Kept).

drop_brackets([], []).
drop_brackets([C|Cs], Kept) :-
    (   ( C == '[' ; C == ']' )
    ->  Kept = Kept1
    ;   Kept = [C|Kept1]
    ),
    drop_brackets(Cs, Kept1).

% ===========================================================================
% 7. parseArgs
% ===========================================================================

%!  parse_args(+Argv, -Result) is det.
parse_args(Argv, Result) :-
    default_registry(Registry),
    parse_args(Argv, Registry, Result).

%!  parse_args(+Argv, +Registry, -Result) is det.
%
%   Strict for a scheduled command, lenient otherwise.
parse_args(Argv, Registry, Result) :-
    % Find the command: the first bare token, skipping any leading global option
    % and its value (globals may appear before the command, e.g.
    % `--state P block bob`).
    scan_leading_globals(Argv, [], Rest, LeadingGlobals),
    (   Rest = [Command|AfterCommand],
        \+ starts_with(Command, "--")
    ->  (   AfterCommand = [Action0|_]
        ->  MaybeAction = some(Action0)
        ;   MaybeAction = none
        ),
        (   schema_for(Command, MaybeAction, Registry, Schema, ActionConsumed)
        ->  (   ActionConsumed == true
            ->  MaybeAction = some(Action),
                AfterCommand = [_|Tail],
                Positional0 = [Command, Action]
            ;   Tail = AfterCommand,
                Positional0 = [Command]
            ),
            parse_strict(Tail, Schema, Positional0, Outcome),
            (   Outcome = ok(Positional, Flags)
            ->  merge_flags(LeadingGlobals, Flags, Merged),
                Result = ok(Positional, Merged)
            ;   Outcome = err(Message),
                Result = error(Message)
            )
        ;   lenient_result(Argv, Result)
        )
    ;   lenient_result(Argv, Result)
    ).

lenient_result(Argv, ok(Positional, Flags)) :-
    parse_lenient(Argv, Positional, Flags).

%!  scan_leading_globals(+Argv, +Acc, -Rest, -Globals) is det.
%
%   Consumes leading `--global [value]` tokens. Bails (leaving the token in
%   Rest) on the first `--token` whose key is not in GLOBAL_OPTIONS, and never
%   crosses a bare `--`.
scan_leading_globals([], Acc, [], Acc).
scan_leading_globals([Token|Rest], Acc, RestOut, Globals) :-
    (   starts_with(Token, "--"),
        Token \== "--"
    ->  split_flag_token(Token, Key, Inline),
        (   is_global_key(Key)
        ->  (   Inline = some(Value)
            ->  flags_set(Acc, Key, Value, Acc1),
                scan_leading_globals(Rest, Acc1, RestOut, Globals)
            ;   next_value(Rest, MaybeValue),
                (   MaybeValue = some(Value2)
                ->  flags_set(Acc, Key, Value2, Acc1),
                    Rest = [_Consumed|Rest1],
                    scan_leading_globals(Rest1, Acc1, RestOut, Globals)
                ;   flags_set(Acc, Key, true, Acc1),
                    scan_leading_globals(Rest, Acc1, RestOut, Globals)
                )
            )
        ;   % a non-global before the command -> leave it for the parse below
            RestOut = [Token|Rest], Globals = Acc
        )
    ;   RestOut = [Token|Rest], Globals = Acc
    ).

%!  is_global_key(+Key) is semidet.  % JS `key in GLOBAL_OPTIONS`
is_global_key(Key) :-
    (   global_options(Globals),
        pair_lookup(Globals, Key, _)
    ->  true
    ;   js_object_prototype_key(Key)
    ).
