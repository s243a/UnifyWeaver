% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% react_native_bindings.pl - React Native-specific bindings
%
% Maps Prolog predicates to React Native components and APIs.
% Designed for composability with the binding registry system.
%
% Categories:
%   - Core Components (View, Text, Pressable, etc.)
%   - List Components (FlatList, SectionList, etc.)
%   - Input Components (TextInput, Switch, etc.)
%   - Navigation (React Navigation)
%   - State Management (hooks, context)
%   - Data Fetching (React Query patterns)
%   - Storage (AsyncStorage, MMKV)
%   - Animation (Reanimated, Animated)
%   - Gestures (gesture-handler)

:- module(react_native_bindings, [
    init_react_native_bindings/0,
    rn_binding/5,               % Convenience: rn_binding(Pred, TargetName, Inputs, Outputs, Options)
    rn_binding_import/2,        % rn_binding_import(Pred, Import)
    rn_hook_binding/4,          % rn_hook_binding(HookName, Inputs, Outputs, Options)
    test_react_native_bindings/0
]).

:- use_module('../core/binding_registry').

% ============================================================================
% INITIALIZATION
% ============================================================================

%% init_react_native_bindings
%
%  Initialize all React Native bindings. Call this before using the compiler.
%
init_react_native_bindings :-
    register_core_component_bindings,
    register_list_component_bindings,
    register_input_component_bindings,
    register_navigation_bindings,
    register_state_bindings,
    register_data_fetching_bindings,
    register_storage_bindings,
    register_animation_bindings,
    register_gesture_bindings.

% ============================================================================
% CONVENIENCE PREDICATES
% ============================================================================

%% rn_binding(?Pred, ?TargetName, ?Inputs, ?Outputs, ?Options)
%
%  Query React Native bindings with reduced arity (Target=react_native implied).
%
rn_binding(Pred, TargetName, Inputs, Outputs, Options) :-
    binding(react_native, Pred, TargetName, Inputs, Outputs, Options).

%% rn_binding_import(?Pred, ?Import)
%
%  Get the import required for a React Native binding.
%
rn_binding_import(Pred, Import) :-
    rn_binding(Pred, _, _, _, Options),
    member(import(Import), Options).

%% rn_hook_binding(?HookName, ?Inputs, ?Outputs, ?Options)
%
%  Query React hook bindings specifically (hooks have is_hook(true) option).
%
rn_hook_binding(HookName, Inputs, Outputs, Options) :-
    rn_binding(_Pred, HookName, Inputs, Outputs, Options),
    member(is_hook(true), Options),
    atom_concat(use_, _, HookName).

% ============================================================================
% DIRECTIVE SUPPORT
% ============================================================================

:- multifile user:term_expansion/2.

user:term_expansion(
    (:- rn_binding(Pred, TargetName, Inputs, Outputs, Options)),
    (:- initialization(binding_registry:declare_binding(react_native, Pred, TargetName, Inputs, Outputs, Options)))
).

% ============================================================================
% CORE COMPONENT BINDINGS
% ============================================================================

register_core_component_bindings :-
    % View - basic container
    declare_binding(react_native, view/2, 'View',
        [props, children], [jsx_element],
        [pure, deterministic, import('react-native')]),

    % Text - text display
    declare_binding(react_native, text/2, 'Text',
        [props, content], [jsx_element],
        [pure, deterministic, import('react-native')]),

    % Pressable - touchable with feedback
    declare_binding(react_native, pressable/3, 'Pressable',
        [props, on_press, children], [jsx_element],
        [effect(ui_event), deterministic, import('react-native')]),

    % Image - image display
    declare_binding(react_native, image/2, 'Image',
        [source, props], [jsx_element],
        [effect(network), deterministic, import('react-native')]),

    % ScrollView - scrollable container
    declare_binding(react_native, scroll_view/2, 'ScrollView',
        [props, children], [jsx_element],
        [pure, deterministic, import('react-native')]),

    % SafeAreaView - respects safe areas
    declare_binding(react_native, safe_area_view/2, 'SafeAreaView',
        [props, children], [jsx_element],
        [pure, deterministic, import('react-native-safe-area-context')]),

    % ActivityIndicator - loading spinner
    declare_binding(react_native, activity_indicator/1, 'ActivityIndicator',
        [props], [jsx_element],
        [pure, deterministic, import('react-native')]),

    % Modal - overlay modal
    declare_binding(react_native, modal/3, 'Modal',
        [visible, props, children], [jsx_element],
        [effect(ui_state), deterministic, import('react-native')]).

% ============================================================================
% LIST COMPONENT BINDINGS
% ============================================================================

register_list_component_bindings :-
    % FlatList - virtualized list
    declare_binding(react_native, flat_list/4, 'FlatList',
        [data, render_item, key_extractor, props], [jsx_element],
        [pure, deterministic, import('react-native')]),

    % SectionList - grouped list
    declare_binding(react_native, section_list/4, 'SectionList',
        [sections, render_item, render_section_header, props], [jsx_element],
        [pure, deterministic, import('react-native')]),

    % VirtualizedList - low-level virtualized list
    declare_binding(react_native, virtualized_list/4, 'VirtualizedList',
        [data, get_item, get_item_count, props], [jsx_element],
        [pure, deterministic, import('react-native')]).

% ============================================================================
% INPUT COMPONENT BINDINGS
% ============================================================================

register_input_component_bindings :-
    % TextInput - text entry
    declare_binding(react_native, text_input/3, 'TextInput',
        [value, on_change, props], [jsx_element],
        [effect(ui_input), deterministic, import('react-native')]),

    % Switch - boolean toggle
    declare_binding(react_native, switch/3, 'Switch',
        [value, on_value_change, props], [jsx_element],
        [effect(ui_input), deterministic, import('react-native')]),

    % Picker (community)
    declare_binding(react_native, picker/4, 'Picker',
        [selected_value, on_value_change, items, props], [jsx_element],
        [effect(ui_input), deterministic, import('@react-native-picker/picker')]).

% ============================================================================
% NAVIGATION BINDINGS (React Navigation)
% ============================================================================

register_navigation_bindings :-
    % NavigationContainer - root navigator
    declare_binding(react_native, navigation_container/2, 'NavigationContainer',
        [props, children], [jsx_element],
        [effect(navigation), deterministic, import('@react-navigation/native')]),

    % Stack Navigator
    declare_binding(react_native, create_stack_navigator/0, 'createStackNavigator',
        [], [navigator],
        [pure, deterministic, import('@react-navigation/stack')]),

    % Bottom Tab Navigator
    declare_binding(react_native, create_bottom_tab_navigator/0, 'createBottomTabNavigator',
        [], [navigator],
        [pure, deterministic, import('@react-navigation/bottom-tabs')]),

    % Drawer Navigator
    declare_binding(react_native, create_drawer_navigator/0, 'createDrawerNavigator',
        [], [navigator],
        [pure, deterministic, import('@react-navigation/drawer')]),

    % useNavigation hook
    declare_binding(react_native, use_navigation/1, 'useNavigation',
        [], [navigation_object],
        [effect(navigation), deterministic, is_hook(true), import('@react-navigation/native')]),

    % useRoute hook
    declare_binding(react_native, use_route/1, 'useRoute',
        [], [route_object],
        [pure, deterministic, is_hook(true), import('@react-navigation/native')]),

    % useFocusEffect hook
    declare_binding(react_native, use_focus_effect/2, 'useFocusEffect',
        [callback], [void],
        [effect(lifecycle), deterministic, is_hook(true), import('@react-navigation/native')]).

% ============================================================================
% STATE MANAGEMENT BINDINGS
% ============================================================================

register_state_bindings :-
    % useState hook
    declare_binding(react_native, use_state/2, 'useState',
        [initial_value], [state_tuple],
        [effect(state), deterministic, is_hook(true), import('react')]),

    % useReducer hook
    declare_binding(react_native, use_reducer/3, 'useReducer',
        [reducer, initial_state], [state_dispatch_tuple],
        [effect(state), deterministic, is_hook(true), import('react')]),

    % useContext hook
    declare_binding(react_native, use_context/2, 'useContext',
        [context], [context_value],
        [pure, deterministic, is_hook(true), import('react')]),

    % useMemo hook
    declare_binding(react_native, use_memo/3, 'useMemo',
        [factory, deps], [memoized_value],
        [pure, deterministic, is_hook(true), import('react')]),

    % useCallback hook
    declare_binding(react_native, use_callback/3, 'useCallback',
        [callback, deps], [memoized_callback],
        [pure, deterministic, is_hook(true), import('react')]),

    % useRef hook
    declare_binding(react_native, use_ref/2, 'useRef',
        [initial_value], [ref_object],
        [effect(ref), deterministic, is_hook(true), import('react')]),

    % useEffect hook
    declare_binding(react_native, use_effect/2, 'useEffect',
        [effect_fn, deps], [void],
        [effect(lifecycle), deterministic, is_hook(true), import('react')]),

    % createContext
    declare_binding(react_native, create_context/1, 'createContext',
        [default_value], [context],
        [pure, deterministic, import('react')]),

    % Zustand store (popular state management)
    declare_binding(react_native, create_zustand_store/1, 'create',
        [store_fn], [use_store_hook],
        [effect(state), deterministic, import('zustand')]).

% ============================================================================
% DATA FETCHING BINDINGS (React Query / TanStack Query)
% ============================================================================

register_data_fetching_bindings :-
    % QueryClient
    declare_binding(react_native, query_client/1, 'QueryClient',
        [options], [query_client],
        [effect(cache), deterministic, import('@tanstack/react-query')]),

    % QueryClientProvider
    declare_binding(react_native, query_client_provider/2, 'QueryClientProvider',
        [client, children], [jsx_element],
        [pure, deterministic, import('@tanstack/react-query')]),

    % useQuery hook
    declare_binding(react_native, use_query/2, 'useQuery',
        [options], [query_result],
        [effect(network), nondeterministic, is_hook(true), import('@tanstack/react-query')]),

    % useMutation hook
    declare_binding(react_native, use_mutation/2, 'useMutation',
        [options], [mutation_result],
        [effect(network), nondeterministic, is_hook(true), import('@tanstack/react-query')]),

    % useInfiniteQuery hook
    declare_binding(react_native, use_infinite_query/2, 'useInfiniteQuery',
        [options], [infinite_query_result],
        [effect(network), nondeterministic, is_hook(true), import('@tanstack/react-query')]),

    % useQueryClient hook
    declare_binding(react_native, use_query_client/1, 'useQueryClient',
        [], [query_client],
        [pure, deterministic, is_hook(true), import('@tanstack/react-query')]).

% ============================================================================
% STORAGE BINDINGS
% ============================================================================

register_storage_bindings :-
    % AsyncStorage
    declare_binding(react_native, async_storage_get/2, 'AsyncStorage.getItem',
        [key], [promise_string],
        [effect(storage), nondeterministic, import('@react-native-async-storage/async-storage')]),

    declare_binding(react_native, async_storage_set/3, 'AsyncStorage.setItem',
        [key, value], [promise_void],
        [effect(storage), nondeterministic, import('@react-native-async-storage/async-storage')]),

    declare_binding(react_native, async_storage_remove/2, 'AsyncStorage.removeItem',
        [key], [promise_void],
        [effect(storage), nondeterministic, import('@react-native-async-storage/async-storage')]),

    declare_binding(react_native, async_storage_clear/1, 'AsyncStorage.clear',
        [], [promise_void],
        [effect(storage), nondeterministic, import('@react-native-async-storage/async-storage')]),

    % MMKV (faster alternative)
    declare_binding(react_native, mmkv_create/1, 'MMKV',
        [options], [mmkv_instance],
        [effect(storage), deterministic, import('react-native-mmkv')]),

    declare_binding(react_native, mmkv_get_string/3, 'mmkv.getString',
        [instance, key], [string_or_undefined],
        [effect(storage), deterministic]),

    declare_binding(react_native, mmkv_set_string/4, 'mmkv.set',
        [instance, key, value], [void],
        [effect(storage), deterministic]).

% ============================================================================
% ANIMATION BINDINGS (Reanimated)
% ============================================================================

register_animation_bindings :-
    % useSharedValue
    declare_binding(react_native, use_shared_value/2, 'useSharedValue',
        [initial_value], [shared_value],
        [effect(animation), deterministic, is_hook(true), import('react-native-reanimated')]),

    % useAnimatedStyle
    declare_binding(react_native, use_animated_style/2, 'useAnimatedStyle',
        [style_fn], [animated_style],
        [effect(animation), deterministic, is_hook(true), import('react-native-reanimated')]),

    % withSpring
    declare_binding(react_native, with_spring/2, 'withSpring',
        [to_value, config], [animation],
        [pure, deterministic, import('react-native-reanimated')]),

    % withTiming
    declare_binding(react_native, with_timing/2, 'withTiming',
        [to_value, config], [animation],
        [pure, deterministic, import('react-native-reanimated')]),

    % withDecay
    declare_binding(react_native, with_decay/1, 'withDecay',
        [config], [animation],
        [pure, deterministic, import('react-native-reanimated')]),

    % Animated.View
    declare_binding(react_native, animated_view/2, 'Animated.View',
        [style, children], [jsx_element],
        [pure, deterministic, import('react-native-reanimated')]),

    % runOnJS
    declare_binding(react_native, run_on_js/2, 'runOnJS',
        [fn, args], [void],
        [effect(js_thread), deterministic, import('react-native-reanimated')]).

% ============================================================================
% GESTURE BINDINGS (Gesture Handler)
% ============================================================================

register_gesture_bindings :-
    % GestureHandlerRootView
    declare_binding(react_native, gesture_handler_root/1, 'GestureHandlerRootView',
        [children], [jsx_element],
        [pure, deterministic, import('react-native-gesture-handler')]),

    % GestureDetector
    declare_binding(react_native, gesture_detector/2, 'GestureDetector',
        [gesture, children], [jsx_element],
        [effect(gesture), deterministic, import('react-native-gesture-handler')]),

    % Gesture.Pan
    declare_binding(react_native, gesture_pan/0, 'Gesture.Pan',
        [], [gesture],
        [pure, deterministic, import('react-native-gesture-handler')]),

    % Gesture.Pinch
    declare_binding(react_native, gesture_pinch/0, 'Gesture.Pinch',
        [], [gesture],
        [pure, deterministic, import('react-native-gesture-handler')]),

    % Gesture.Tap
    declare_binding(react_native, gesture_tap/0, 'Gesture.Tap',
        [], [gesture],
        [pure, deterministic, import('react-native-gesture-handler')]),

    % Gesture.Simultaneous
    declare_binding(react_native, gesture_simultaneous/1, 'Gesture.Simultaneous',
        [gestures], [gesture],
        [pure, deterministic, import('react-native-gesture-handler')]).

% ============================================================================
% TESTING
% ============================================================================

test_react_native_bindings :-
    format('~n=== React Native Bindings Tests ===~n~n'),

    % Test 1: Core bindings exist
    format('Test 1: Core component bindings...~n'),
    (   rn_binding(view/2, 'View', _, _, _)
    ->  format('  PASS: View binding exists~n')
    ;   format('  FAIL: View binding missing~n')
    ),

    % Test 2: Hook bindings exist
    format('~nTest 2: Hook bindings...~n'),
    (   rn_binding(use_state/2, 'useState', _, _, _)
    ->  format('  PASS: useState binding exists~n')
    ;   format('  FAIL: useState binding missing~n')
    ),

    % Test 3: Import extraction
    format('~nTest 3: Import extraction...~n'),
    (   rn_binding_import(flat_list/4, Import),
        Import == 'react-native'
    ->  format('  PASS: FlatList import is react-native~n')
    ;   format('  FAIL: FlatList import incorrect~n')
    ),

    % Test 4: Navigation bindings
    format('~nTest 4: Navigation bindings...~n'),
    (   rn_binding(use_navigation/1, 'useNavigation', _, _, _)
    ->  format('  PASS: useNavigation binding exists~n')
    ;   format('  FAIL: useNavigation binding missing~n')
    ),

    % Test 5: Data fetching bindings
    format('~nTest 5: Data fetching bindings...~n'),
    (   rn_binding(use_query/2, 'useQuery', _, _, Opts),
        member(effect(network), Opts)
    ->  format('  PASS: useQuery has network effect~n')
    ;   format('  FAIL: useQuery effect incorrect~n')
    ),

    % Test 6: Storage bindings
    format('~nTest 6: Storage bindings...~n'),
    (   rn_binding(async_storage_get/2, _, _, _, _)
    ->  format('  PASS: AsyncStorage.getItem binding exists~n')
    ;   format('  FAIL: AsyncStorage binding missing~n')
    ),

    format('~n=== Tests Complete ===~n').

% ============================================================================
% AUTO-INITIALIZE ON LOAD
% ============================================================================

:- initialization((
    init_react_native_bindings,
    format('React Native bindings initialized~n', [])
), now).
