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

### Basic Composition

Combine multiple patterns:

```prolog
?- compose_patterns([navigation_pattern, state_pattern], [], Composed).
```

Check pattern compatibility:

```prolog
?- pattern_compatible(nav_pattern, state_pattern).
```

### Advanced Composition (pattern_composition.pl)

The `pattern_composition` module provides dependency resolution, conflict detection, and validation:

```prolog
:- use_module('pattern_composition').

%% Compose with automatic dependency resolution
?- compose_with_deps([screen_user_profile], react_native, [], Result).
% Result includes all dependencies (store_user, query_users, etc.)

%% Detect conflicts between patterns
?- detect_conflicts([store1, store2], Conflicts).
% Finds: name clashes, singleton conflicts, exclusions

%% Validate for a specific target
?- validate_for_target([patterns], flutter, Errors).
% Checks if all required capabilities are available

%% Get composition summary
?- composition_summary([nav, store, query], Summary).
% Returns: pattern count, type breakdown, required capabilities
```

#### Pattern Dependencies

Patterns can declare dependencies:

```prolog
define_pattern(user_profile_screen,
    navigation(stack, [screen(profile, 'Profile', [])], []),
    [depends_on([user_store, fetch_user_query])]).
```

#### Conflict Types

- **name_clash**: Two patterns generate the same artifact name
- **singleton**: Multiple singletons of same type with same name
- **exclusion**: Patterns that explicitly exclude each other
- **requirements**: Incompatible library versions

## Extended Patterns (ui_patterns_extended.pl)

The `ui_patterns_extended` module provides additional high-level patterns for common UI scenarios.

### Form Patterns

```prolog
%% Create a form with validation
form_pattern(login_form, [
    field(email, email, [required], []),
    field(password, password, [required, min_length(8)], [])
], Pattern).

%% Create individual fields
form_field(username, text, [required, min_length(3)], Spec).

%% Validation rules
validation_rule(required, [], required).
validation_rule(min_length, [5], min_length(5)).
validation_rule(pattern, ['^[a-z]+$'], pattern('^[a-z]+$')).
validation_rule(matches, [password], matches(password)).
```

Generated code includes:
- **React Native**: react-hook-form + Zod validation + TextInput components
- **Vue 3**: vee-validate + Zod schema + native form elements
- **Flutter**: Form widget + TextFormField + validators
- **SwiftUI**: @State bindings + TextField/SecureField

### List Patterns

```prolog
%% Infinite scroll list
infinite_list(items_list, '/api/items', Pattern).

%% Selectable list (single or multi)
selectable_list(select_list, [mode(single)], Pattern).
selectable_list(multi_list, [mode(multi)], Pattern).

%% Grouped list
grouped_list(contacts, category, Pattern).
```

Generated code includes:
- **React Native**: FlatList + useInfiniteQuery + onEndReached
- **Vue 3**: IntersectionObserver + useInfiniteQuery + virtual scrolling
- **Flutter**: infinite_scroll_pagination + PagingController + RefreshIndicator
- **SwiftUI**: List + refreshable + .onAppear pagination

### Modal Patterns

```prolog
%% Alert modal
alert_modal(my_alert, [title('Alert'), message('Something happened')], Pattern).

%% Confirm modal with custom buttons
confirm_modal(delete_confirm, [
    title('Delete Item'),
    message('Are you sure?'),
    confirm_text('Delete'),
    cancel_text('Keep')
], Pattern).

%% Bottom sheet
bottom_sheet(my_sheet, 'SheetContent', Pattern).

%% Action sheet with options
action_sheet(actions, [
    action(edit, 'Edit'),
    action(delete, 'Delete')
], Pattern).
```

Generated code includes:
- **React Native**: Alert.alert + @gorhom/bottom-sheet + ActionSheetIOS
- **Vue 3**: Teleport + transition + modal composables
- **Flutter**: showDialog + showModalBottomSheet + AlertDialog
- **SwiftUI**: .alert modifier + .sheet + .confirmationDialog

### Auth Flow Patterns

```prolog
%% Login flow
login_flow([endpoint('/api/login')], Pattern).

%% Registration with password confirmation
register_flow([endpoint('/api/register')], Pattern).

%% Forgot password
forgot_password_flow([endpoint('/api/forgot-password')], Pattern).

%% OAuth provider
oauth_flow(google, [client_id('your-client-id')], Pattern).

%% MFA verification
auth_flow(mfa, [endpoint('/api/mfa/verify')], Pattern).

%% Password reset
auth_flow(reset_password, [endpoint('/api/reset-password')], Pattern).
```

Generated code includes:
- Form with appropriate fields (email, password, confirm_password)
- useMutation for API calls with error handling
- Loading states and validation feedback
- OAuth integration patterns

### Extended Pattern Compilation

```prolog
%% Compile form pattern
ui_patterns_extended:compile_form_pattern(login, Fields, react_native, [], Code).

%% Compile list pattern
ui_patterns_extended:compile_list_pattern(infinite, items, '/api/items', vue, [], Code).

%% Compile modal pattern
ui_patterns_extended:compile_modal_pattern(confirm, my_confirm, Config, flutter, [], Code).

%% Compile auth pattern
ui_patterns_extended:compile_auth_pattern(login, [endpoint('/api/login')], swiftui, [], Code).
```

## Testing

```bash
# Run all pattern tests (38 tests)
swipl -g "run_tests" -t halt src/unifyweaver/patterns/test_ui_patterns.pl

# Run extended pattern tests (50 tests)
swipl -g "run_tests" -t halt src/unifyweaver/patterns/test_ui_patterns_extended.pl

# Run composition tests (30 tests)
swipl -g "run_tests" -t halt src/unifyweaver/patterns/test_pattern_composition.pl

# Run multi-target tests (34 tests)
swipl -g "run_tests" -t halt src/unifyweaver/targets/test_multi_target.pl

# Run inline tests
swipl -g "test_ui_patterns" -t halt src/unifyweaver/patterns/ui_patterns.pl
swipl -g "test_extended_patterns" -t halt src/unifyweaver/patterns/ui_patterns_extended.pl
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
| `ui_patterns_extended.pl` | Extended patterns (forms, lists, modals, auth) |
| `pattern_composition.pl` | Dependency resolution and conflict detection |
| `test_ui_patterns.pl` | 38 plunit tests |
| `test_ui_patterns_extended.pl` | 50 extended pattern tests |
| `test_pattern_composition.pl` | 30 composition tests |
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
