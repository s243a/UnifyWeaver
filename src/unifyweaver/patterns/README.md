# UI Patterns Module

Generalizable, composable UI patterns for multi-target code generation.

## Overview

The `ui_patterns.pl` module provides abstract UI patterns that compile to multiple targets (React Native, Vue, etc.). Patterns are:

- **Generalizable**: Same pattern specification works across targets
- **Composable**: Patterns can be combined together
- **Integrated**: Uses binding registry and component registry

## Pattern Categories

### Navigation Patterns

```prolog
%% Create stack navigator
navigation_pattern(stack, [
    screen(home, 'HomeScreen', [title('Home')]),
    screen(detail, 'DetailScreen', [])
], Pattern).

%% Create tab navigator
navigation_pattern(tab, [
    screen(feed, 'FeedScreen', []),
    screen(profile, 'ProfileScreen', [])
], Pattern).

%% Create drawer navigator
navigation_pattern(drawer, [
    screen(home, 'HomeScreen', []),
    screen(settings, 'SettingsScreen', [])
], Pattern).
```

### State Patterns

```prolog
%% Local state (useState)
local_state(count, 0).

%% Global state (Zustand store)
global_state(appStore, [
    slice(user, [field(name, string), field(email, string)], [
        action(setName, "(set, name) => set({ name })")
    ])
], Pattern).

%% Derived state (useMemo)
derived_state(fullName, [firstName, lastName], "firstName + ' ' + lastName").
```

### Data Fetching Patterns

```prolog
%% Query pattern (useQuery)
query_pattern(fetch_users, '/api/users', [stale_time(60000)], Pattern).

%% Mutation pattern (useMutation)
mutation_pattern(create_user, '/api/users', [method('POST')], Pattern).

%% Paginated/infinite pattern (useInfiniteQuery)
paginated_pattern(load_feed, '/api/feed', [page_param(page)], Pattern).
```

### Persistence Patterns

```prolog
%% AsyncStorage
local_storage(user_prefs, '{ theme: string }', Pattern).

%% MMKV (faster)
persistence_pattern(mmkv, [key(cache), schema(object)], Pattern).
```

## Compilation

Compile patterns to target-specific code:

```prolog
%% Define and compile a navigation pattern
?- navigation_pattern(stack, [screen(home, 'HomeScreen', [])], P),
   define_pattern(my_nav, P, []),
   compile_pattern(my_nav, react_native, [], Code).
```

## Pattern Composition

Combine multiple patterns:

```prolog
?- compose_patterns([navigation_pattern, state_pattern], [], Composed).
```

Check pattern compatibility:

```prolog
?- pattern_compatible(nav_pattern, state_pattern).
```

## Testing

```bash
# Run all pattern tests (38 tests)
swipl -g "run_tests" -t halt src/unifyweaver/patterns/test_ui_patterns.pl

# Run inline tests
swipl -g "test_ui_patterns" -t halt src/unifyweaver/patterns/ui_patterns.pl
```

## Integration with React Native Target

The React Native target provides convenience predicates:

```prolog
%% Generate navigation
generate_rn_navigation([
    screen(home, 'HomeScreen', []),
    screen(settings, 'SettingsScreen', [])
], [type(tab)], Code).

%% Generate query hook
generate_rn_query_hook(fetch_data, '/api/data', Code).

%% Generate storage hook
generate_rn_storage_hook(user_prefs, '{ theme: string }', Code).
```

## Files

| File | Description |
|------|-------------|
| `ui_patterns.pl` | Main patterns module |
| `test_ui_patterns.pl` | 38 plunit tests |
| `README.md` | This documentation |

## Extending to New Targets

To add a new target (e.g., Flutter):

1. Add `compile_pattern_spec/4` clauses for the target
2. Implement target-specific code generators
3. Register target capabilities in `target_has_capability/2`

```prolog
compile_pattern_spec(navigation(Type, Screens, Config), flutter, Options, Code) :-
    compile_flutter_navigation(Type, Screens, Config, Options, Code).
```
