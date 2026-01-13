# UI Patterns Module

Generalizable, composable UI patterns for multi-target code generation.

## Overview

The `ui_patterns.pl` module provides abstract UI patterns that compile to multiple targets:

- **React Native** - React Navigation, Zustand, React Query, AsyncStorage
- **Vue 3** - Vue Router, Pinia, Vue Query, localStorage
- **Flutter** - GoRouter, Riverpod, FutureProvider, SharedPreferences
- **SwiftUI** - NavigationStack, ObservableObject, async/await, AppStorage

Patterns are:

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
%% Define a navigation pattern once
?- navigation_pattern(stack, [screen(home, 'HomeScreen', [])], P),
   define_pattern(my_nav, P, []).

%% Compile to different targets
?- compile_pattern(my_nav, react_native, [], RNCode).   % React Navigation
?- compile_pattern(my_nav, vue, [], VueCode).           % Vue Router
?- compile_pattern(my_nav, flutter, [], FlutterCode).   % GoRouter
?- compile_pattern(my_nav, swiftui, [], SwiftCode).     % NavigationStack
```

### Multi-Target Compilation

```prolog
%% Load target modules
:- use_module('targets/vue_target', []).
:- use_module('targets/flutter_target', []).
:- use_module('targets/swiftui_target', []).

%% Compile state pattern to all targets
?- test_state([store(appStore), slices([...])]),
   ui_patterns:compile_state_pattern(global, Shape, [], react_native, [], RN),
   vue_target:compile_state_pattern(global, Shape, [], vue, [], Vue),
   flutter_target:compile_state_pattern(global, Shape, [], flutter, [], Flutter),
   swiftui_target:compile_state_pattern(global, Shape, [], swiftui, [], SwiftUI).
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

# Run multi-target tests (34 tests)
swipl -g "run_tests" -t halt src/unifyweaver/targets/test_multi_target.pl

# Run inline tests
swipl -g "test_ui_patterns" -t halt src/unifyweaver/patterns/ui_patterns.pl
```

## Target-Specific Features

### React Native
- React Navigation (stack, tab, drawer)
- Zustand for global state
- React Query for data fetching
- AsyncStorage/MMKV for persistence

### Vue 3
- Vue Router with TypeScript
- Pinia stores with actions
- Vue Query composables
- localStorage with reactive refs

### Flutter
- GoRouter for declarative navigation
- Riverpod StateNotifier/StateNotifierProvider
- FutureProvider for async data
- SharedPreferences and Hive

### SwiftUI
- NavigationStack and TabView
- ObservableObject with @Published
- async/await with URLSession
- @AppStorage and Keychain

## Files

| File | Description |
|------|-------------|
| `ui_patterns.pl` | Main patterns module |
| `test_ui_patterns.pl` | 38 plunit tests |
| `../targets/target_interface.pl` | Target contract documentation |
| `../targets/vue_target.pl` | Vue 3 code generation |
| `../targets/flutter_target.pl` | Flutter/Dart code generation |
| `../targets/swiftui_target.pl` | SwiftUI/Swift code generation |
| `../targets/test_multi_target.pl` | 34 cross-framework tests |

## Extending to New Targets

To add a new target, create a module implementing:

1. `target_capabilities/1` - List supported features
2. `compile_navigation_pattern/6` - Generate navigation code
3. `compile_state_pattern/6` - Generate state management code
4. `compile_data_pattern/5` - Generate data fetching code
5. `compile_persistence_pattern/5` - Generate persistence code

See `target_interface.pl` for the full contract.
