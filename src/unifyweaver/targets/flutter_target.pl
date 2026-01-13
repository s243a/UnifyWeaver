% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% flutter_target.pl - Flutter/Dart Code Generation Target
%
% Generates Flutter widgets and Dart code from UI patterns. Supports:
%   - Navigation (Navigator 2.0, GoRouter)
%   - State management (Provider, Riverpod, setState)
%   - Data fetching (FutureBuilder, http package)
%   - Persistence (shared_preferences, Hive)
%
% Flutter uses a widget-based architecture where everything is a widget.
% State management patterns differ from React hooks - Flutter uses
% StatefulWidget, Provider, or Riverpod for reactive state.
%
% Usage:
%   ?- compile_navigation_pattern(tab, Screens, [], flutter, [], Code).
%   ?- compile_state_pattern(global, Shape, [], flutter, [], Code).

:- module(flutter_target, [
    % Target capabilities
    target_capabilities/1,
    flutter_capabilities/1,

    % Pattern compilation
    compile_navigation_pattern/6,
    compile_state_pattern/6,
    compile_data_pattern/5,
    compile_persistence_pattern/5,

    % Testing
    test_flutter_target/0
]).

:- use_module(library(lists)).

% ============================================================================
% TARGET CAPABILITIES
% ============================================================================

%% flutter_capabilities(-Capabilities)
%
%  Lists what Flutter target can do.
%
flutter_capabilities([
    % Direct capabilities
    supports(material_design),
    supports(cupertino_design),
    supports(custom_widgets),
    supports(animations),
    supports(gestures),
    supports(platform_channels),

    % Libraries
    library('flutter/material.dart'),
    library('go_router'),
    library('flutter_riverpod'),
    library('shared_preferences'),
    library('hive'),
    library('dio'),

    % Limitations
    limitation(no_hot_reload_for_native),
    limitation(large_app_size)
]).

target_capabilities(Caps) :- flutter_capabilities(Caps).

% ============================================================================
% HELPER PREDICATES
% ============================================================================

option_value(Options, Key, Value, Default) :-
    Opt =.. [Key, Value],
    (   member(Opt, Options)
    ->  true
    ;   Value = Default
    ).

capitalize_first(Str, Cap) :-
    (   atom(Str) -> atom_string(Str, S) ; S = Str ),
    string_chars(S, [H|T]),
    upcase_atom(H, HU),
    atom_chars(HU, [HUC]),
    string_chars(Cap, [HUC|T]).

to_pascal_case(Name, Pascal) :-
    atom_string(Name, Str),
    split_string(Str, "_", "", Parts),
    maplist(capitalize_first, Parts, CapParts),
    atomics_to_string(CapParts, Pascal).

% ============================================================================
% PATTERN COMPILATION - Navigation
% ============================================================================
%
% Flutter uses Navigator 2.0 or GoRouter for declarative routing.
% Unlike web frameworks, Flutter navigation is imperative by default.

%% compile_navigation_pattern(+Type, +Screens, +Config, +Target, +Options, -Code)
compile_navigation_pattern(stack, Screens, _Config, flutter, Options, Code) :-
    option_value(Options, component_name, Name, 'AppRouter'),
    generate_flutter_go_router(Screens, stack, Name, Code).
compile_navigation_pattern(tab, Screens, _Config, flutter, Options, Code) :-
    option_value(Options, component_name, Name, 'TabScaffold'),
    generate_flutter_tab_scaffold(Screens, Name, Code).
compile_navigation_pattern(drawer, Screens, _Config, flutter, Options, Code) :-
    option_value(Options, component_name, Name, 'DrawerScaffold'),
    generate_flutter_drawer_scaffold(Screens, Name, Code).

generate_flutter_go_router(Screens, _Type, Name, Code) :-
    generate_flutter_route_definitions(Screens, RouteDefs),
    generate_flutter_route_imports(Screens, Imports),
    format(string(Code),
"// ~w - GoRouter Configuration
import 'package:flutter/material.dart';
import 'package:go_router/go_router.dart';
~w

final ~w = GoRouter(
  initialLocation: '/',
  routes: [
~w
  ],
);

// Usage: MaterialApp.router(routerConfig: ~w)
", [Name, Imports, Name, RouteDefs, Name]).

generate_flutter_route_definitions(Screens, Defs) :-
    findall(RouteDef, (
        member(screen(ScreenName, Component, _Opts), Screens),
        atom_string(ScreenName, NameStr),
        to_pascal_case(Component, WidgetName),
        format(string(RouteDef),
"    GoRoute(
      path: '/~w',
      name: '~w',
      builder: (context, state) => const ~w(),
    )", [NameStr, NameStr, WidgetName])
    ), RouteDefList),
    atomic_list_concat(RouteDefList, ',\n', Defs).

generate_flutter_route_imports(Screens, Imports) :-
    findall(ImportLine, (
        member(screen(_, Component, _), Screens),
        atom_string(Component, CompStr),
        string_lower(CompStr, LowerComp),
        format(string(ImportLine), "import 'screens/~w.dart';", [LowerComp])
    ), ImportLines),
    atomic_list_concat(ImportLines, '\n', Imports).

generate_flutter_tab_scaffold(Screens, Name, Code) :-
    generate_flutter_tab_items(Screens, TabItems),
    generate_flutter_tab_views(Screens, TabViews),
    format(string(Code),
"// ~w - Bottom Navigation Tab Scaffold
import 'package:flutter/material.dart';

class ~w extends StatefulWidget {
  const ~w({super.key});

  @override
  State<~w> createState() => _~wState();
}

class _~wState extends State<~w> {
  int _currentIndex = 0;

  final List<Widget> _screens = [
~w
  ];

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: IndexedStack(
        index: _currentIndex,
        children: _screens,
      ),
      bottomNavigationBar: BottomNavigationBar(
        currentIndex: _currentIndex,
        onTap: (index) => setState(() => _currentIndex = index),
        type: BottomNavigationBarType.fixed,
        items: const [
~w
        ],
      ),
    );
  }
}
", [Name, Name, Name, Name, Name, Name, Name, TabViews, TabItems]).

generate_flutter_tab_items(Screens, Items) :-
    findall(TabItem, (
        member(screen(ScreenName, _, Opts), Screens),
        (   member(title(Title), Opts) -> true ; atom_string(ScreenName, Title) ),
        (   member(icon(Icon), Opts) -> true ; Icon = 'Icons.circle' ),
        format(string(TabItem),
"          BottomNavigationBarItem(
            icon: Icon(~w),
            label: '~w',
          )", [Icon, Title])
    ), TabItemList),
    atomic_list_concat(TabItemList, ',\n', Items).

generate_flutter_tab_views(Screens, Views) :-
    findall(View, (
        member(screen(_, Component, _), Screens),
        to_pascal_case(Component, WidgetName),
        format(string(View), "    const ~w(),", [WidgetName])
    ), ViewList),
    atomic_list_concat(ViewList, '\n', Views).

generate_flutter_drawer_scaffold(Screens, Name, Code) :-
    generate_flutter_drawer_items(Screens, DrawerItems),
    format(string(Code),
"// ~w - Drawer Navigation Scaffold
import 'package:flutter/material.dart';

class ~w extends StatefulWidget {
  const ~w({super.key});

  @override
  State<~w> createState() => _~wState();
}

class _~wState extends State<~w> {
  int _selectedIndex = 0;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('App'),
      ),
      drawer: Drawer(
        child: ListView(
          padding: EdgeInsets.zero,
          children: [
            const DrawerHeader(
              decoration: BoxDecoration(
                color: Colors.blue,
              ),
              child: Text(
                'Navigation',
                style: TextStyle(
                  color: Colors.white,
                  fontSize: 24,
                ),
              ),
            ),
~w
          ],
        ),
      ),
      body: _buildBody(),
    );
  }

  Widget _buildBody() {
    // Return the selected screen based on _selectedIndex
    switch (_selectedIndex) {
      default:
        return const Center(child: Text('Select a screen'));
    }
  }
}
", [Name, Name, Name, Name, Name, Name, Name, DrawerItems]).

generate_flutter_drawer_items(Screens, Items) :-
    findall(DrawerItem, (
        nth0(Index, Screens, screen(ScreenName, _, Opts)),
        (   member(title(Title), Opts) -> true ; atom_string(ScreenName, Title) ),
        (   member(icon(Icon), Opts) -> true ; Icon = 'Icons.circle' ),
        format(string(DrawerItem),
"            ListTile(
              leading: Icon(~w),
              title: Text('~w'),
              selected: _selectedIndex == ~w,
              onTap: () {
                setState(() => _selectedIndex = ~w);
                Navigator.pop(context);
              },
            )", [Icon, Title, Index, Index])
    ), DrawerItemList),
    atomic_list_concat(DrawerItemList, ',\n', Items).

% ============================================================================
% PATTERN COMPILATION - State
% ============================================================================
%
% Flutter state management options:
% - Local: StatefulWidget with setState
% - Global: Riverpod (StateNotifier, StateNotifierProvider)
% - Derived: Computed values via Provider

%% compile_state_pattern(+Type, +Shape, +Config, +Target, +Options, -Code)
compile_state_pattern(local, Shape, _Config, flutter, _Options, Code) :-
    generate_flutter_stateful_widget(Shape, Code).
compile_state_pattern(global, Shape, _Config, flutter, _Options, Code) :-
    generate_flutter_riverpod_provider(Shape, Code).
compile_state_pattern(derived, Shape, _Config, flutter, _Options, Code) :-
    generate_flutter_computed_provider(Shape, Code).

generate_flutter_stateful_widget(Shape, Code) :-
    findall(FieldDef, (
        member(field(Name, Initial), Shape),
        format(string(FieldDef), "  var _~w = ~w;", [Name, Initial])
    ), FieldDefs),
    atomic_list_concat(FieldDefs, '\n', FieldsCode),
    findall(SetterDef, (
        member(field(Name, _), Shape),
        capitalize_first(Name, CapName),
        format(string(SetterDef),
"  void set~w(dynamic value) {
    setState(() => _~w = value);
  }", [CapName, Name])
    ), SetterDefs),
    atomic_list_concat(SetterDefs, '\n\n', SettersCode),
    format(string(Code),
"// Flutter StatefulWidget - Local State
import 'package:flutter/material.dart';

class MyWidget extends StatefulWidget {
  const MyWidget({super.key});

  @override
  State<MyWidget> createState() => _MyWidgetState();
}

class _MyWidgetState extends State<MyWidget> {
  // State fields
~w

  // Setters with setState
~w

  @override
  Widget build(BuildContext context) {
    return Container(
      // Use state fields here
    );
  }
}
", [FieldsCode, SettersCode]).

generate_flutter_riverpod_provider(Shape, Code) :-
    member(store(StoreName), Shape),
    member(slices(Slices), Shape),
    to_pascal_case(StoreName, StoreClass),
    generate_riverpod_state_class(Slices, StoreClass, StateClass),
    generate_riverpod_notifier(Slices, StoreClass, NotifierCode),
    format(string(Code),
"// Riverpod State Management - ~w
import 'package:flutter_riverpod/flutter_riverpod.dart';

~w

~w

final ~wProvider = StateNotifierProvider<~wNotifier, ~wState>((ref) {
  return ~wNotifier();
});

// Usage:
// final state = ref.watch(~wProvider);
// ref.read(~wProvider.notifier).toggleTheme();
", [StoreName, StateClass, NotifierCode, StoreName, StoreClass, StoreClass, StoreClass, StoreName, StoreName]).

generate_riverpod_state_class(Slices, StoreClass, Code) :-
    findall(FieldDef, (
        member(slice(_, Fields, _), Slices),
        member(field(FName, FType), Fields),
        dart_type(FType, DartType),
        format(string(FieldDef), "  final ~w ~w;", [DartType, FName])
    ), FieldDefs),
    atomic_list_concat(FieldDefs, '\n', FieldsStr),
    findall(ParamDef, (
        member(slice(_, Fields, _), Slices),
        member(field(FName, _), Fields),
        format(string(ParamDef), "    required this.~w,", [FName])
    ), ParamDefs),
    atomic_list_concat(ParamDefs, '\n', ParamsStr),
    findall(CopyParam, (
        member(slice(_, Fields, _), Slices),
        member(field(FName, _), Fields),
        format(string(CopyParam), "    ~w? ~w,", [FName, FName])
    ), CopyParams),
    atomic_list_concat(CopyParams, '\n', CopyParamsStr),
    findall(CopyField, (
        member(slice(_, Fields, _), Slices),
        member(field(FName, _), Fields),
        format(string(CopyField), "      ~w: ~w ?? this.~w,", [FName, FName, FName])
    ), CopyFields),
    atomic_list_concat(CopyFields, '\n', CopyFieldsStr),
    format(string(Code),
"class ~wState {
~w

  const ~wState({
~w
  });

  ~wState copyWith({
~w
  }) {
    return ~wState(
~w
    );
  }
}", [StoreClass, FieldsStr, StoreClass, ParamsStr, StoreClass, CopyParamsStr, StoreClass, CopyFieldsStr]).

dart_type("'light' | 'dark'", "String").
dart_type("boolean", "bool").
dart_type("string | null", "String?").
dart_type("number", "int").
dart_type(_, "dynamic").

generate_riverpod_notifier(Slices, StoreClass, Code) :-
    findall(ActionDef, (
        member(slice(_, _, Actions), Slices),
        member(action(AName, _), Actions),
        generate_riverpod_action(AName, ActionDef)
    ), ActionDefs),
    (   ActionDefs = []
    ->  ActionsStr = "  // Add actions here"
    ;   atomic_list_concat(ActionDefs, '\n\n', ActionsStr)
    ),
    findall(InitField, (
        member(slice(_, Fields, _), Slices),
        member(field(FName, FType), Fields),
        dart_default_value(FType, Default),
        format(string(InitField), "      ~w: ~w,", [FName, Default])
    ), InitFields),
    atomic_list_concat(InitFields, '\n', InitStr),
    format(string(Code),
"class ~wNotifier extends StateNotifier<~wState> {
  ~wNotifier() : super(const ~wState(
~w
  ));

~w
}", [StoreClass, StoreClass, StoreClass, StoreClass, InitStr, ActionsStr]).

dart_default_value("'light' | 'dark'", "'light'").
dart_default_value("boolean", "false").
dart_default_value("string | null", "null").
dart_default_value("number", "0").
dart_default_value(_, "null").

generate_riverpod_action(toggleTheme, Code) :-
    format(string(Code),
"  void toggleTheme() {
    state = state.copyWith(
      theme: state.theme == 'light' ? 'dark' : 'light',
    );
  }", []).
generate_riverpod_action(toggleSidebar, Code) :-
    format(string(Code),
"  void toggleSidebar() {
    state = state.copyWith(
      sidebarOpen: !state.sidebarOpen,
    );
  }", []).
generate_riverpod_action(Name, Code) :-
    \+ member(Name, [toggleTheme, toggleSidebar]),
    format(string(Code),
"  void ~w(dynamic value) {
    // TODO: Implement ~w action
  }", [Name, Name]).

generate_flutter_computed_provider(Shape, Code) :-
    member(deps(Deps), Shape),
    member(derive(Derivation), Shape),
    atomic_list_concat(Deps, ', ', DepsStr),
    format(string(Code),
"// Riverpod Computed Provider
import 'package:flutter_riverpod/flutter_riverpod.dart';

// Assumes providers for: ~w
final derivedProvider = Provider<dynamic>((ref) {
  // Watch dependencies
  // final dep1 = ref.watch(dep1Provider);
  // final dep2 = ref.watch(dep2Provider);

  // Compute derived value
  return ~w;
});
", [DepsStr, Derivation]).

% ============================================================================
% PATTERN COMPILATION - Data Fetching
% ============================================================================
%
% Flutter uses FutureBuilder/StreamBuilder with http or Dio for data fetching.
% Riverpod provides FutureProvider and AsyncNotifierProvider for async state.

%% compile_data_pattern(+Type, +Config, +Target, +Options, -Code)
compile_data_pattern(query, Config, flutter, _Options, Code) :-
    generate_flutter_future_provider(Config, Code).
compile_data_pattern(mutation, Config, flutter, _Options, Code) :-
    generate_flutter_mutation_notifier(Config, Code).
compile_data_pattern(infinite, Config, flutter, _Options, Code) :-
    generate_flutter_paginated_provider(Config, Code).

generate_flutter_future_provider(Config, Code) :-
    member(name(Name), Config),
    member(endpoint(Endpoint), Config),
    to_pascal_case(Name, ProviderName),
    format(string(Code),
"// Flutter FutureProvider - ~w
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';

class ~wData {
  // TODO: Define response type
  final Map<String, dynamic> data;

  ~wData(this.data);

  factory ~wData.fromJson(Map<String, dynamic> json) {
    return ~wData(json);
  }
}

final ~wProvider = FutureProvider.autoDispose<~wData>((ref) async {
  final response = await http.get(Uri.parse('~w'));

  if (response.statusCode != 200) {
    throw Exception('Failed to load data');
  }

  return ~wData.fromJson(jsonDecode(response.body));
});

// Usage in widget:
// final asyncValue = ref.watch(~wProvider);
// asyncValue.when(
//   data: (data) => Text(data.toString()),
//   loading: () => CircularProgressIndicator(),
//   error: (e, st) => Text('Error: \\$e'),
// );
", [Name, ProviderName, ProviderName, ProviderName, ProviderName, Name, ProviderName, Endpoint, ProviderName, Name]).

generate_flutter_mutation_notifier(Config, Code) :-
    member(name(Name), Config),
    member(endpoint(Endpoint), Config),
    (   member(method(Method), Config) -> true ; Method = 'POST' ),
    to_pascal_case(Name, NotifierName),
    format(string(Code),
"// Flutter Mutation Notifier - ~w
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';

enum MutationStatus { idle, loading, success, error }

class ~wState {
  final MutationStatus status;
  final dynamic data;
  final String? error;

  const ~wState({
    this.status = MutationStatus.idle,
    this.data,
    this.error,
  });

  ~wState copyWith({
    MutationStatus? status,
    dynamic data,
    String? error,
  }) {
    return ~wState(
      status: status ?? this.status,
      data: data ?? this.data,
      error: error ?? this.error,
    );
  }
}

class ~wNotifier extends StateNotifier<~wState> {
  ~wNotifier() : super(const ~wState());

  Future<void> mutate(Map<String, dynamic> variables) async {
    state = state.copyWith(status: MutationStatus.loading);

    try {
      final response = await http.~w(
        Uri.parse('~w'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode(variables),
      );

      if (response.statusCode != 200) {
        throw Exception('Mutation failed');
      }

      state = state.copyWith(
        status: MutationStatus.success,
        data: jsonDecode(response.body),
      );
    } catch (e) {
      state = state.copyWith(
        status: MutationStatus.error,
        error: e.toString(),
      );
    }
  }

  void reset() {
    state = const ~wState();
  }
}

final ~wProvider = StateNotifierProvider<~wNotifier, ~wState>((ref) {
  return ~wNotifier();
});

// Usage:
// ref.read(~wProvider.notifier).mutate({'key': 'value'});
", [Name, NotifierName, NotifierName, NotifierName, NotifierName, NotifierName, NotifierName, NotifierName, NotifierName, Method, Endpoint, NotifierName, Name, NotifierName, NotifierName, NotifierName, Name]).

generate_flutter_paginated_provider(Config, Code) :-
    member(name(Name), Config),
    member(endpoint(Endpoint), Config),
    (   member(page_param(PageParam), Config) -> true ; PageParam = 'page' ),
    to_pascal_case(Name, NotifierName),
    format(string(Code),
"// Flutter Paginated Provider - ~w
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';

class ~wState {
  final List<dynamic> items;
  final int currentPage;
  final bool hasMore;
  final bool isLoading;
  final String? error;

  const ~wState({
    this.items = const [],
    this.currentPage = 1,
    this.hasMore = true,
    this.isLoading = false,
    this.error,
  });

  ~wState copyWith({
    List<dynamic>? items,
    int? currentPage,
    bool? hasMore,
    bool? isLoading,
    String? error,
  }) {
    return ~wState(
      items: items ?? this.items,
      currentPage: currentPage ?? this.currentPage,
      hasMore: hasMore ?? this.hasMore,
      isLoading: isLoading ?? this.isLoading,
      error: error ?? this.error,
    );
  }
}

class ~wNotifier extends StateNotifier<~wState> {
  ~wNotifier() : super(const ~wState());

  Future<void> loadMore() async {
    if (state.isLoading || !state.hasMore) return;

    state = state.copyWith(isLoading: true);

    try {
      final response = await http.get(
        Uri.parse('~w?~w=${state.currentPage}'),
      );

      if (response.statusCode != 200) {
        throw Exception('Failed to load page');
      }

      final data = jsonDecode(response.body);
      final newItems = data['data'] as List<dynamic>;
      final hasMore = data['hasMore'] as bool? ?? false;

      state = state.copyWith(
        items: [...state.items, ...newItems],
        currentPage: state.currentPage + 1,
        hasMore: hasMore,
        isLoading: false,
      );
    } catch (e) {
      state = state.copyWith(
        isLoading: false,
        error: e.toString(),
      );
    }
  }

  void refresh() {
    state = const ~wState();
    loadMore();
  }
}

final ~wProvider = StateNotifierProvider<~wNotifier, ~wState>((ref) {
  return ~wNotifier();
});

// Usage with ListView:
// NotificationListener<ScrollNotification>(
//   onNotification: (notification) {
//     if (notification.metrics.pixels >= notification.metrics.maxScrollExtent - 200) {
//       ref.read(~wProvider.notifier).loadMore();
//     }
//     return false;
//   },
//   child: ListView.builder(...),
// )
", [Name, NotifierName, NotifierName, NotifierName, NotifierName, NotifierName, NotifierName, NotifierName, NotifierName, Endpoint, PageParam, NotifierName, Name, NotifierName, NotifierName, NotifierName, Name]).

% ============================================================================
% PATTERN COMPILATION - Persistence
% ============================================================================
%
% Flutter uses shared_preferences for simple key-value storage,
% or Hive for more complex local storage needs.

%% compile_persistence_pattern(+Type, +Config, +Target, +Options, -Code)
compile_persistence_pattern(local, Config, flutter, _Options, Code) :-
    generate_flutter_shared_prefs(Config, Code).
compile_persistence_pattern(secure, Config, flutter, _Options, Code) :-
    generate_flutter_secure_storage(Config, Code).

generate_flutter_shared_prefs(Config, Code) :-
    member(key(Key), Config),
    to_pascal_case(Key, ClassName),
    format(string(Code),
"// Flutter SharedPreferences - ~w
import 'package:shared_preferences/shared_preferences.dart';
import 'dart:convert';

class ~wStorage {
  static const String _key = '~w';

  static Future<Map<String, dynamic>?> load() async {
    final prefs = await SharedPreferences.getInstance();
    final stored = prefs.getString(_key);
    if (stored == null) return null;
    return jsonDecode(stored) as Map<String, dynamic>;
  }

  static Future<void> save(Map<String, dynamic> data) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(_key, jsonEncode(data));
  }

  static Future<void> remove() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.remove(_key);
  }
}

// Riverpod provider for reactive storage
final ~wProvider = FutureProvider<Map<String, dynamic>?>((ref) async {
  return ~wStorage.load();
});

// Usage:
// final data = await ~wStorage.load();
// await ~wStorage.save({'theme': 'dark'});
", [Key, ClassName, Key, Key, ClassName, ClassName, ClassName]).

generate_flutter_secure_storage(Config, Code) :-
    member(key(Key), Config),
    to_pascal_case(Key, ClassName),
    format(string(Code),
"// Flutter Secure Storage - ~w
import 'package:flutter_secure_storage/flutter_secure_storage.dart';
import 'dart:convert';

class ~wSecureStorage {
  static const String _key = '~w';
  static const _storage = FlutterSecureStorage();

  static Future<Map<String, dynamic>?> load() async {
    final stored = await _storage.read(key: _key);
    if (stored == null) return null;
    return jsonDecode(stored) as Map<String, dynamic>;
  }

  static Future<void> save(Map<String, dynamic> data) async {
    await _storage.write(key: _key, value: jsonEncode(data));
  }

  static Future<void> remove() async {
    await _storage.delete(key: _key);
  }
}

// Riverpod provider for reactive secure storage
final ~wSecureProvider = FutureProvider<Map<String, dynamic>?>((ref) async {
  return ~wSecureStorage.load();
});

// Usage:
// final data = await ~wSecureStorage.load();
// await ~wSecureStorage.save({'token': 'secret'});
", [Key, ClassName, Key, Key, ClassName, ClassName, ClassName]).

% ============================================================================
% TESTING
% ============================================================================

test_flutter_target :-
    format('~n=== Flutter Target Tests ===~n~n'),

    % Test 1: Navigation pattern
    format('Test 1: Tab navigation generation...~n'),
    (   compile_navigation_pattern(tab,
            [screen(home, 'HomeScreen', [title('Home')]),
             screen(profile, 'ProfileScreen', [title('Profile')])],
            [], flutter, [], Code1),
        sub_string(Code1, _, _, _, "BottomNavigationBar"),
        sub_string(Code1, _, _, _, "StatefulWidget")
    ->  format('  PASS: Tab scaffold generated~n')
    ;   format('  FAIL: Tab navigation incorrect~n')
    ),

    % Test 2: State pattern
    format('~nTest 2: Riverpod state generation...~n'),
    (   compile_state_pattern(global,
            [store(appStore), slices([slice(ui, [field(theme, "'light' | 'dark'")], [action(toggleTheme, _)])])],
            [], flutter, [], Code2),
        sub_string(Code2, _, _, _, "StateNotifier"),
        sub_string(Code2, _, _, _, "StateNotifierProvider")
    ->  format('  PASS: Riverpod provider generated~n')
    ;   format('  FAIL: State pattern incorrect~n')
    ),

    % Test 3: Data pattern
    format('~nTest 3: FutureProvider generation...~n'),
    (   compile_data_pattern(query,
            [name(fetchUsers), endpoint('/api/users')],
            flutter, [], Code3),
        sub_string(Code3, _, _, _, "FutureProvider"),
        sub_string(Code3, _, _, _, "http.get")
    ->  format('  PASS: FutureProvider generated~n')
    ;   format('  FAIL: Data pattern incorrect~n')
    ),

    % Test 4: Persistence pattern
    format('~nTest 4: SharedPreferences generation...~n'),
    (   compile_persistence_pattern(local,
            [key(userPrefs)],
            flutter, [], Code4),
        sub_string(Code4, _, _, _, "SharedPreferences"),
        sub_string(Code4, _, _, _, "jsonEncode")
    ->  format('  PASS: SharedPreferences generated~n')
    ;   format('  FAIL: Persistence pattern incorrect~n')
    ),

    format('~n=== Tests Complete ===~n').

% ============================================================================
% INITIALIZATION
% ============================================================================

:- initialization((
    format('Flutter target module loaded~n', [])
), now).
