:- encoding(utf8).
% constraints_demo.pl - Demonstration of UnifyWeaver's constraint system

:- use_module('../src/unifyweaver/core/constraint_analyzer').
:- use_module('../src/unifyweaver/core/stream_compiler').

demo :-
    format('~n╔═══════════════════════════════════════════════════════════════╗~n'),
    format('║  UnifyWeaver Constraint System Demo                         ║~n'),
    format('╚═══════════════════════════════════════════════════════════════╝~n'),

    % Setup demo predicates
    setup_demo_data,

    % Show default constraints
    format('~n┌─ Default Constraints ───────────────────────────────────────┐~n'),
    get_default_constraints(Defaults),
    format('│ Defaults: ~w~n', [Defaults]),
    format('│ Rationale: Most Prolog queries are order-independent       │~n'),
    format('│            Efficient: uses sort -u instead of hash tables  │~n'),
    format('└─────────────────────────────────────────────────────────────┘~n'),

    % Demo 1: Standard predicate with defaults
    format('~n┌─ Demo 1: Standard Predicate (uses defaults) ───────────────┐~n'),
    format('│ Predicate: family_relation/2                               │~n'),
    format('│ Constraints: [default]                                      │~n'),
    format('│ Strategy: sort -u (efficient deduplication)                │~n'),
    format('└─────────────────────────────────────────────────────────────┘~n'),
    compile_predicate(family_relation/2, [], Code1),
    write_demo_file('output/demo_default.sh', Code1),
    format('✓ Generated: output/demo_default.sh~n'),

    % Demo 2: Temporal/ordered data
    format('~n┌─ Demo 2: Temporal Query (preserves order) ─────────────────┐~n'),
    format('│ Predicate: event_sequence/2                                │~n'),
    format('│ Constraints: [unique, ordered]                             │~n'),
    format('│ Strategy: hash-based dedup (preserves insertion order)    │~n'),
    format('└─────────────────────────────────────────────────────────────┘~n'),
    declare_constraint(event_sequence/2, [unique, ordered]),
    compile_predicate(event_sequence/2, [], Code2),
    write_demo_file('output/demo_ordered.sh', Code2),
    format('✓ Generated: output/demo_ordered.sh~n'),

    % Demo 3: Allow duplicates
    format('~n┌─ Demo 3: Path Finding (allows duplicates) ─────────────────┐~n'),
    format('│ Predicate: all_connections/2                               │~n'),
    format('│ Constraints: [unique(false)]                               │~n'),
    format('│ Strategy: no deduplication (shows all paths)              │~n'),
    format('└─────────────────────────────────────────────────────────────┘~n'),
    declare_constraint(all_connections/2, [unique(false)]),
    compile_predicate(all_connections/2, [], Code3),
    write_demo_file('output/demo_no_dedup.sh', Code3),
    format('✓ Generated: output/demo_no_dedup.sh~n'),

    % Show generated file snippets
    format('~n┌─ Generated Code Snippets ───────────────────────────────────┐~n'),
    format('│                                                             │~n'),
    format('│ Demo 1 (sort -u):                                          │~n'),
    show_snippet(Code1, 'sort -u'),
    format('│                                                             │~n'),
    format('│ Demo 2 (hash dedup):                                       │~n'),
    show_snippet(Code2, 'declare -A seen'),
    format('│                                                             │~n'),
    format('│ Demo 3 (no dedup):                                         │~n'),
    show_snippet(Code3, 'no deduplication'),
    format('└─────────────────────────────────────────────────────────────┘~n'),

    % Cleanup
    cleanup_demo_data,

    format('~n✓ Demo complete!~n'),
    format('  Check output/demo_*.sh for generated code~n~n').

% Setup demo data
setup_demo_data :-
    % Clear existing
    catch(abolish(user:person/1), _, true),
    catch(abolish(user:parent/2), _, true),
    catch(abolish(user:event/2), _, true),
    catch(abolish(user:connection/2), _, true),

    % Define predicates for demos
    assertz(user:person(alice)),
    assertz(user:person(bob)),
    assertz(user:person(charlie)),

    assertz(user:parent(alice, bob)),
    assertz(user:parent(alice, charlie)),
    assertz(user:parent(bob, dave)),

    assertz(user:event(morning, wake_up)),
    assertz(user:event(morning, breakfast)),
    assertz(user:event(noon, lunch)),

    assertz(user:connection(a, b)),
    assertz(user:connection(b, c)),
    assertz(user:connection(a, c)),

    % Define rules
    assertz(user:(family_relation(X, Y) :- parent(X, Y))),
    assertz(user:(family_relation(X, Y) :- parent(Y, X))),

    assertz(user:(event_sequence(T, E) :- event(T, E))),

    assertz(user:(all_connections(X, Y) :- connection(X, Y))),
    assertz(user:(all_connections(X, Y) :- connection(Y, X))).

% Cleanup
cleanup_demo_data :-
    catch(abolish(user:person/1), _, true),
    catch(abolish(user:parent/2), _, true),
    catch(abolish(user:event/2), _, true),
    catch(abolish(user:connection/2), _, true),
    catch(abolish(user:family_relation/2), _, true),
    catch(abolish(user:event_sequence/2), _, true),
    catch(abolish(user:all_connections/2), _, true),
    clear_constraints(family_relation/2),
    clear_constraints(event_sequence/2),
    clear_constraints(all_connections/2).

% Helper to write demo files
write_demo_file(Path, Code) :-
    open(Path, write, Stream, [newline(posix)]),
    write(Stream, Code),
    close(Stream).

% Helper to show code snippet
show_snippet(Code, Marker) :-
    (   sub_string(Code, Before, Len, _After, Marker) ->
        Start is max(0, Before - 20),
        Length is min(60, Before - Start + Len + 20),
        sub_string(Code, Start, Length, _, Snippet),
        split_string(Snippet, "\n", "", Lines),
        (   Lines = [First|_] ->
            format('│   ...~w...~n', [First])
        ;   format('│   [snippet]~n')
        )
    ;   format('│   [marker not found]~n')
    ).
