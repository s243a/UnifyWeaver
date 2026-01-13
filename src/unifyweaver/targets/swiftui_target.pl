% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% swiftui_target.pl - SwiftUI Code Generation Target
%
% Generates SwiftUI views and Swift code from UI patterns. Supports:
%   - Navigation (NavigationStack, TabView)
%   - State management (@State, @StateObject, @EnvironmentObject)
%   - Data fetching (async/await, URLSession)
%   - Persistence (@AppStorage, UserDefaults)
%
% SwiftUI uses a declarative approach with property wrappers for state.
% The View protocol defines the body property that returns the UI.
%
% Usage:
%   ?- compile_navigation_pattern(tab, Screens, [], swiftui, [], Code).
%   ?- compile_state_pattern(global, Shape, [], swiftui, [], Code).

:- module(swiftui_target, [
    % Target capabilities
    target_capabilities/1,
    swiftui_capabilities/1,

    % Pattern compilation
    compile_navigation_pattern/6,
    compile_state_pattern/6,
    compile_data_pattern/5,
    compile_persistence_pattern/5,

    % Testing
    test_swiftui_target/0
]).

:- use_module(library(lists)).

% ============================================================================
% TARGET CAPABILITIES
% ============================================================================

%% swiftui_capabilities(-Capabilities)
%
%  Lists what SwiftUI target can do.
%
swiftui_capabilities([
    % Direct capabilities
    supports(declarative_ui),
    supports(property_wrappers),
    supports(combine_framework),
    supports(async_await),
    supports(previews),
    supports(accessibility),

    % Libraries/Frameworks
    library('SwiftUI'),
    library('Combine'),
    library('Foundation'),

    % Limitations
    limitation(ios_14_minimum),
    limitation(no_uikit_direct),
    limitation(limited_customization)
]).

target_capabilities(Caps) :- swiftui_capabilities(Caps).

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
% SwiftUI uses NavigationStack (iOS 16+) or NavigationView for navigation,
% and TabView for tab-based navigation.

%% compile_navigation_pattern(+Type, +Screens, +Config, +Target, +Options, -Code)
compile_navigation_pattern(stack, Screens, _Config, swiftui, Options, Code) :-
    option_value(Options, component_name, Name, 'AppNavigation'),
    generate_swiftui_navigation_stack(Screens, Name, Code).
compile_navigation_pattern(tab, Screens, _Config, swiftui, Options, Code) :-
    option_value(Options, component_name, Name, 'MainTabView'),
    generate_swiftui_tab_view(Screens, Name, Code).
compile_navigation_pattern(drawer, Screens, _Config, swiftui, Options, Code) :-
    option_value(Options, component_name, Name, 'SidebarNavigation'),
    generate_swiftui_sidebar(Screens, Name, Code).

generate_swiftui_navigation_stack(Screens, Name, Code) :-
    generate_swiftui_navigation_links(Screens, NavLinks),
    generate_swiftui_navigation_destinations(Screens, Destinations),
    format(string(Code),
"// ~w - SwiftUI Navigation Stack
import SwiftUI

enum AppRoute: Hashable {
~w
}

struct ~w: View {
    var body: some View {
        NavigationStack {
            List {
~w
            }
            .navigationTitle(\"Navigation\")
            .navigationDestination(for: AppRoute.self) { route in
                switch route {
~w
                }
            }
        }
    }
}

#Preview {
    ~w()
}
", [Name, Destinations, Name, NavLinks, Destinations, Name]).

generate_swiftui_navigation_links(Screens, Links) :-
    findall(Link, (
        member(screen(ScreenName, _, Opts), Screens),
        (   member(title(Title), Opts) -> true ; atom_string(ScreenName, Title) ),
        atom_string(ScreenName, NameStr),
        format(string(Link),
"                NavigationLink(value: AppRoute.~w) {
                    Text(\"~w\")
                }", [NameStr, Title])
    ), LinkList),
    atomic_list_concat(LinkList, '\n', Links).

generate_swiftui_navigation_destinations(Screens, Destinations) :-
    findall(Switch, (
        member(screen(ScreenName, Component, _), Screens),
        atom_string(ScreenName, NameStr),
        to_pascal_case(Component, ViewName),
        format(string(Switch),
"                case .~w:
                    ~w()", [NameStr, ViewName])
    ), SwitchList),
    atomic_list_concat(SwitchList, '\n', Destinations).

generate_swiftui_tab_view(Screens, Name, Code) :-
    generate_swiftui_tab_items(Screens, TabItems),
    format(string(Code),
"// ~w - SwiftUI Tab View
import SwiftUI

struct ~w: View {
    @State private var selectedTab = 0

    var body: some View {
        TabView(selection: $selectedTab) {
~w
        }
    }
}

#Preview {
    ~w()
}
", [Name, Name, TabItems, Name]).

generate_swiftui_tab_items(Screens, Items) :-
    findall(TabItem, (
        nth0(Index, Screens, screen(ScreenName, Component, Opts)),
        (   member(title(Title), Opts) -> true ; atom_string(ScreenName, Title) ),
        (   member(icon(Icon), Opts) -> true ; Icon = 'circle' ),
        to_pascal_case(Component, ViewName),
        format(string(TabItem),
"            ~w()
                .tabItem {
                    Label(\"~w\", systemImage: \"~w\")
                }
                .tag(~w)", [ViewName, Title, Icon, Index])
    ), TabItemList),
    atomic_list_concat(TabItemList, '\n', Items).

generate_swiftui_sidebar(Screens, Name, Code) :-
    Screens = [screen(FirstScreen, _, _)|_],
    atom_string(FirstScreen, FirstItem),
    generate_swiftui_sidebar_items(Screens, SidebarItems),
    generate_swiftui_sidebar_detail(Screens, DetailView),
    generate_swiftui_icon_cases(Screens, IconCases),
    format(string(Code),
"// ~w - SwiftUI Sidebar Navigation
import SwiftUI

enum SidebarItem: String, CaseIterable, Identifiable {
~w

    var id: String { rawValue }
}

struct ~w: View {
    @State private var selectedItem: SidebarItem? = .~w

    var body: some View {
        NavigationSplitView {
            List(SidebarItem.allCases, selection: $selectedItem) { item in
                NavigationLink(value: item) {
                    Label(item.rawValue.capitalized, systemImage: iconFor(item))
                }
            }
            .navigationTitle(\"Menu\")
        } detail: {
            if let item = selectedItem {
~w
            } else {
                Text(\"Select an item\")
            }
        }
    }

    private func iconFor(_ item: SidebarItem) -> String {
        switch item {
~w
        }
    }
}

#Preview {
    ~w()
}
", [Name, SidebarItems, Name, FirstItem, DetailView, IconCases, Name]).

generate_swiftui_sidebar_items(Screens, Items) :-
    findall(Item, (
        member(screen(ScreenName, _, _), Screens),
        atom_string(ScreenName, NameStr),
        format(string(Item), "    case ~w", [NameStr])
    ), ItemList),
    atomic_list_concat(ItemList, '\n', Items).

generate_swiftui_sidebar_detail(Screens, DetailView) :-
    findall(Case, (
        member(screen(ScreenName, Component, _), Screens),
        atom_string(ScreenName, NameStr),
        to_pascal_case(Component, ViewName),
        format(string(Case),
"                switch item {
                case .~w:
                    ~w()
                }", [NameStr, ViewName])
    ), _CaseList),
    % Simplified for now - just switch on item
    format(string(DetailView),
"                switch item {
                default:
                    Text(item.rawValue.capitalized)
                }", []).

generate_swiftui_icon_cases(Screens, Cases) :-
    findall(Case, (
        member(screen(ScreenName, _, Opts), Screens),
        atom_string(ScreenName, NameStr),
        (   member(icon(Icon), Opts) -> true ; Icon = 'circle' ),
        format(string(Case), "        case .~w: return \"~w\"", [NameStr, Icon])
    ), CaseList),
    atomic_list_concat(CaseList, '\n', Cases).

% ============================================================================
% PATTERN COMPILATION - State
% ============================================================================
%
% SwiftUI state management uses property wrappers:
% - @State for local view state
% - @StateObject for owning ObservableObject instances
% - @ObservedObject for passed-in ObservableObject references
% - @EnvironmentObject for dependency injection

%% compile_state_pattern(+Type, +Shape, +Config, +Target, +Options, -Code)
compile_state_pattern(local, Shape, _Config, swiftui, _Options, Code) :-
    generate_swiftui_local_state(Shape, Code).
compile_state_pattern(global, Shape, _Config, swiftui, _Options, Code) :-
    generate_swiftui_observable_object(Shape, Code).
compile_state_pattern(derived, Shape, _Config, swiftui, _Options, Code) :-
    generate_swiftui_computed_property(Shape, Code).

generate_swiftui_local_state(Shape, Code) :-
    findall(StateDef, (
        member(field(Name, Initial), Shape),
        swift_type_and_default(Initial, SwiftType, Default),
        format(string(StateDef), "    @State private var ~w: ~w = ~w", [Name, SwiftType, Default])
    ), StateDefs),
    atomic_list_concat(StateDefs, '\n', StatesCode),
    format(string(Code),
"// SwiftUI Local State
import SwiftUI

struct MyView: View {
    // State properties
~w

    var body: some View {
        VStack {
            // Use state properties here
        }
    }
}
", [StatesCode]).

swift_type_and_default("'light' | 'dark'", "String", "\"light\"").
swift_type_and_default("boolean", "Bool", "false").
swift_type_and_default("string | null", "String?", "nil").
swift_type_and_default("number", "Int", "0").
swift_type_and_default(Val, "String", Quoted) :-
    atom(Val),
    format(string(Quoted), "\"~w\"", [Val]).
swift_type_and_default(_, "Any", "nil").

generate_swiftui_observable_object(Shape, Code) :-
    member(store(StoreName), Shape),
    member(slices(Slices), Shape),
    to_pascal_case(StoreName, ClassName),
    generate_swift_published_properties(Slices, Properties),
    generate_swift_methods(Slices, Methods),
    format(string(Code),
"// SwiftUI ObservableObject - ~w
import SwiftUI
import Combine

@MainActor
class ~w: ObservableObject {
    // Published properties
~w

    // Singleton instance (optional)
    static let shared = ~w()

    private init() {}

    // Methods
~w
}

// Usage in View:
// @StateObject private var store = ~w.shared
// or
// @EnvironmentObject var store: ~w

// In App:
// ContentView()
//     .environmentObject(~w.shared)
", [StoreName, ClassName, Properties, ClassName, Methods, ClassName, ClassName, ClassName]).

generate_swift_published_properties(Slices, Properties) :-
    findall(PropDef, (
        member(slice(_, Fields, _), Slices),
        member(field(FName, FType), Fields),
        swift_type_and_default(FType, SwiftType, Default),
        format(string(PropDef), "    @Published var ~w: ~w = ~w", [FName, SwiftType, Default])
    ), PropDefs),
    atomic_list_concat(PropDefs, '\n', Properties).

generate_swift_methods(Slices, Methods) :-
    findall(MethodDef, (
        member(slice(_, _, Actions), Slices),
        member(action(AName, _), Actions),
        generate_swift_method(AName, MethodDef)
    ), MethodDefs),
    (   MethodDefs = []
    ->  Methods = "    // Add methods here"
    ;   atomic_list_concat(MethodDefs, '\n\n', Methods)
    ).

generate_swift_method(toggleTheme, Code) :-
    format(string(Code),
"    func toggleTheme() {
        theme = theme == \"light\" ? \"dark\" : \"light\"
    }", []).
generate_swift_method(toggleSidebar, Code) :-
    format(string(Code),
"    func toggleSidebar() {
        sidebarOpen.toggle()
    }", []).
generate_swift_method(Name, Code) :-
    \+ member(Name, [toggleTheme, toggleSidebar]),
    format(string(Code),
"    func ~w(_ value: Any) {
        // TODO: Implement ~w
    }", [Name, Name]).

generate_swiftui_computed_property(Shape, Code) :-
    member(deps(Deps), Shape),
    member(derive(Derivation), Shape),
    atomic_list_concat(Deps, ', ', DepsStr),
    format(string(Code),
"// SwiftUI Computed Property
import SwiftUI

// In a View or ObservableObject:
// Assumes properties exist for: ~w

var derivedValue: some View {
    // Computed based on dependencies
    // return ~w
}

// Or as a computed property in ObservableObject:
// var derivedValue: SomeType {
//     // Combine dependencies
// }
", [DepsStr, Derivation]).

% ============================================================================
% PATTERN COMPILATION - Data Fetching
% ============================================================================
%
% SwiftUI uses async/await with URLSession for data fetching.
% State is managed with @State for loading/error states.

%% compile_data_pattern(+Type, +Config, +Target, +Options, -Code)
compile_data_pattern(query, Config, swiftui, _Options, Code) :-
    generate_swiftui_async_fetch(Config, Code).
compile_data_pattern(mutation, Config, swiftui, _Options, Code) :-
    generate_swiftui_mutation(Config, Code).
compile_data_pattern(infinite, Config, swiftui, _Options, Code) :-
    generate_swiftui_paginated_fetch(Config, Code).

generate_swiftui_async_fetch(Config, Code) :-
    member(name(Name), Config),
    member(endpoint(Endpoint), Config),
    to_pascal_case(Name, ClassName),
    format(string(Code),
"// SwiftUI Async Data Fetch - ~w
import SwiftUI

// Response model
struct ~wResponse: Codable {
    // TODO: Define response properties
    let data: [String: AnyCodable]?
}

// View Model
@MainActor
class ~wViewModel: ObservableObject {
    @Published var data: ~wResponse?
    @Published var isLoading = false
    @Published var error: Error?

    func fetch() async {
        isLoading = true
        error = nil

        do {
            guard let url = URL(string: \"~w\") else {
                throw URLError(.badURL)
            }

            let (data, response) = try await URLSession.shared.data(from: url)

            guard let httpResponse = response as? HTTPURLResponse,
                  httpResponse.statusCode == 200 else {
                throw URLError(.badServerResponse)
            }

            self.data = try JSONDecoder().decode(~wResponse.self, from: data)
        } catch {
            self.error = error
        }

        isLoading = false
    }
}

// Usage in View:
struct ~wView: View {
    @StateObject private var viewModel = ~wViewModel()

    var body: some View {
        Group {
            if viewModel.isLoading {
                ProgressView()
            } else if let error = viewModel.error {
                Text(\"Error: \\(error.localizedDescription)\")
            } else if let data = viewModel.data {
                Text(\"Data loaded\")
            } else {
                Text(\"No data\")
            }
        }
        .task {
            await viewModel.fetch()
        }
    }
}
", [Name, ClassName, ClassName, ClassName, Endpoint, ClassName, ClassName, ClassName]).

generate_swiftui_mutation(Config, Code) :-
    member(name(Name), Config),
    member(endpoint(Endpoint), Config),
    (   member(method(Method), Config) -> true ; Method = 'POST' ),
    to_pascal_case(Name, ClassName),
    string_upper(Method, MethodUpper),
    format(string(Code),
"// SwiftUI Mutation - ~w
import SwiftUI

// Request/Response models
struct ~wRequest: Codable {
    // TODO: Define request properties
}

struct ~wResponse: Codable {
    // TODO: Define response properties
    let success: Bool
}

// View Model
@MainActor
class ~wViewModel: ObservableObject {
    @Published var isLoading = false
    @Published var result: ~wResponse?
    @Published var error: Error?

    func mutate(_ input: ~wRequest) async {
        isLoading = true
        error = nil

        do {
            guard let url = URL(string: \"~w\") else {
                throw URLError(.badURL)
            }

            var request = URLRequest(url: url)
            request.httpMethod = \"~w\"
            request.setValue(\"application/json\", forHTTPHeaderField: \"Content-Type\")
            request.httpBody = try JSONEncoder().encode(input)

            let (data, response) = try await URLSession.shared.data(for: request)

            guard let httpResponse = response as? HTTPURLResponse,
                  httpResponse.statusCode == 200 else {
                throw URLError(.badServerResponse)
            }

            self.result = try JSONDecoder().decode(~wResponse.self, from: data)
        } catch {
            self.error = error
        }

        isLoading = false
    }

    func reset() {
        result = nil
        error = nil
    }
}

// Usage:
// @StateObject private var viewModel = ~wViewModel()
// Button(\"Submit\") {
//     Task { await viewModel.mutate(input) }
// }
", [Name, ClassName, ClassName, ClassName, ClassName, ClassName, Endpoint, MethodUpper, ClassName, ClassName]).

generate_swiftui_paginated_fetch(Config, Code) :-
    member(name(Name), Config),
    member(endpoint(Endpoint), Config),
    (   member(page_param(PageParam), Config) -> true ; PageParam = 'page' ),
    to_pascal_case(Name, ClassName),
    format(string(Code),
"// SwiftUI Paginated Fetch - ~w
import SwiftUI

// Page response model
struct ~wPage: Codable {
    let data: [~wItem]
    let hasMore: Bool
    let nextPage: Int?
}

struct ~wItem: Codable, Identifiable {
    let id: String
    // TODO: Add item properties
}

// View Model
@MainActor
class ~wViewModel: ObservableObject {
    @Published var items: [~wItem] = []
    @Published var isLoading = false
    @Published var hasMore = true
    @Published var error: Error?

    private var currentPage = 1

    func loadMore() async {
        guard !isLoading, hasMore else { return }

        isLoading = true
        error = nil

        do {
            guard let url = URL(string: \"~w?~w=\\(currentPage)\") else {
                throw URLError(.badURL)
            }

            let (data, response) = try await URLSession.shared.data(from: url)

            guard let httpResponse = response as? HTTPURLResponse,
                  httpResponse.statusCode == 200 else {
                throw URLError(.badServerResponse)
            }

            let page = try JSONDecoder().decode(~wPage.self, from: data)

            items.append(contentsOf: page.data)
            hasMore = page.hasMore
            currentPage += 1
        } catch {
            self.error = error
        }

        isLoading = false
    }

    func refresh() async {
        items = []
        currentPage = 1
        hasMore = true
        await loadMore()
    }
}

// Usage with List:
struct ~wListView: View {
    @StateObject private var viewModel = ~wViewModel()

    var body: some View {
        List {
            ForEach(viewModel.items) { item in
                Text(item.id)
            }

            if viewModel.hasMore {
                ProgressView()
                    .onAppear {
                        Task { await viewModel.loadMore() }
                    }
            }
        }
        .refreshable {
            await viewModel.refresh()
        }
        .task {
            await viewModel.loadMore()
        }
    }
}
", [Name, ClassName, ClassName, ClassName, ClassName, ClassName, Endpoint, PageParam, ClassName, ClassName, ClassName]).

% ============================================================================
% PATTERN COMPILATION - Persistence
% ============================================================================
%
% SwiftUI uses @AppStorage for simple UserDefaults binding,
% or custom persistence for complex data.

%% compile_persistence_pattern(+Type, +Config, +Target, +Options, -Code)
compile_persistence_pattern(local, Config, swiftui, _Options, Code) :-
    generate_swiftui_app_storage(Config, Code).
compile_persistence_pattern(secure, Config, swiftui, _Options, Code) :-
    generate_swiftui_keychain(Config, Code).

generate_swiftui_app_storage(Config, Code) :-
    member(key(Key), Config),
    to_pascal_case(Key, ClassName),
    format(string(Code),
"// SwiftUI AppStorage - ~w
import SwiftUI

// For simple values, use @AppStorage directly in views:
struct SettingsView: View {
    @AppStorage(\"~w\") private var storedValue: String = \"\"

    var body: some View {
        TextField(\"Value\", text: $storedValue)
    }
}

// For complex objects, use a manager:
@MainActor
class ~wStorage: ObservableObject {
    private let key = \"~w\"

    @Published var data: [String: Any]? {
        didSet {
            save()
        }
    }

    init() {
        load()
    }

    private func load() {
        if let stored = UserDefaults.standard.dictionary(forKey: key) {
            data = stored
        }
    }

    private func save() {
        if let data = data {
            UserDefaults.standard.set(data, forKey: key)
        } else {
            UserDefaults.standard.removeObject(forKey: key)
        }
    }

    func remove() {
        data = nil
    }
}

// Usage:
// @StateObject private var storage = ~wStorage()
", [Key, Key, ClassName, Key, ClassName]).

generate_swiftui_keychain(Config, Code) :-
    member(key(Key), Config),
    to_pascal_case(Key, ClassName),
    format(string(Code),
"// SwiftUI Keychain Storage - ~w
import SwiftUI
import Security

@MainActor
class ~wSecureStorage: ObservableObject {
    private let key = \"~w\"

    @Published var data: Data?

    init() {
        load()
    }

    private func load() {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrAccount as String: key,
            kSecReturnData as String: true,
            kSecMatchLimit as String: kSecMatchLimitOne
        ]

        var result: AnyObject?
        let status = SecItemCopyMatching(query as CFDictionary, &result)

        if status == errSecSuccess {
            data = result as? Data
        }
    }

    func save(_ value: Data) {
        // Delete existing
        let deleteQuery: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrAccount as String: key
        ]
        SecItemDelete(deleteQuery as CFDictionary)

        // Add new
        let addQuery: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrAccount as String: key,
            kSecValueData as String: value
        ]

        let status = SecItemAdd(addQuery as CFDictionary, nil)
        if status == errSecSuccess {
            data = value
        }
    }

    func remove() {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrAccount as String: key
        ]
        SecItemDelete(query as CFDictionary)
        data = nil
    }
}

// Usage:
// @StateObject private var secureStorage = ~wSecureStorage()
// secureStorage.save(\"secret\".data(using: .utf8)!)
", [Key, ClassName, Key, ClassName]).

% ============================================================================
% TESTING
% ============================================================================

test_swiftui_target :-
    format('~n=== SwiftUI Target Tests ===~n~n'),

    % Test 1: Navigation pattern
    format('Test 1: Tab navigation generation...~n'),
    (   compile_navigation_pattern(tab,
            [screen(home, 'HomeView', [title('Home'), icon('house')]),
             screen(profile, 'ProfileView', [title('Profile'), icon('person')])],
            [], swiftui, [], Code1),
        sub_string(Code1, _, _, _, "TabView"),
        sub_string(Code1, _, _, _, "@State")
    ->  format('  PASS: TabView generated~n')
    ;   format('  FAIL: Tab navigation incorrect~n')
    ),

    % Test 2: State pattern
    format('~nTest 2: ObservableObject generation...~n'),
    (   compile_state_pattern(global,
            [store(appStore), slices([slice(ui, [field(theme, "'light' | 'dark'")], [action(toggleTheme, _)])])],
            [], swiftui, [], Code2),
        sub_string(Code2, _, _, _, "ObservableObject"),
        sub_string(Code2, _, _, _, "@Published")
    ->  format('  PASS: ObservableObject generated~n')
    ;   format('  FAIL: State pattern incorrect~n')
    ),

    % Test 3: Data pattern
    format('~nTest 3: Async fetch generation...~n'),
    (   compile_data_pattern(query,
            [name(fetchUsers), endpoint('/api/users')],
            swiftui, [], Code3),
        sub_string(Code3, _, _, _, "URLSession"),
        sub_string(Code3, _, _, _, "async")
    ->  format('  PASS: Async fetch generated~n')
    ;   format('  FAIL: Data pattern incorrect~n')
    ),

    % Test 4: Persistence pattern
    format('~nTest 4: AppStorage generation...~n'),
    (   compile_persistence_pattern(local,
            [key(userPrefs)],
            swiftui, [], Code4),
        sub_string(Code4, _, _, _, "@AppStorage"),
        sub_string(Code4, _, _, _, "UserDefaults")
    ->  format('  PASS: AppStorage generated~n')
    ;   format('  FAIL: Persistence pattern incorrect~n')
    ),

    format('~n=== Tests Complete ===~n').

% ============================================================================
% INITIALIZATION
% ============================================================================

:- initialization((
    format('SwiftUI target module loaded~n', [])
), now).
