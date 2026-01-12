%% generate_app.pl - Generate PT_Explorer React Native App
%%
%% Demonstrates composable UI patterns by generating a complete mobile app:
%%   - Tab navigation (Home, Search, Favorites, Profile)
%%   - React Query hooks for API data fetching
%%   - Zustand store for global state
%%   - AsyncStorage for offline persistence
%%   - Mindmap visualization component
%%
%% Usage:
%%   ?- generate_all_app_code.
%%   ?- generate_app_component(navigation, Code).

:- module(generate_app, [
    generate_all_app_code/0,
    generate_app_component/2,
    generate_backend_api/1,
    show_app_structure/0
]).

:- use_module('../../../patterns/ui_patterns').
:- use_module('../../../glue/pattern_glue').
:- use_module('../../../targets/react_native_target').

% ============================================================================
% APP STRUCTURE
% ============================================================================

%% show_app_structure/0
%
%  Display the app structure being generated.
%
show_app_structure :-
    format('~n=== PT_Explorer App Structure ===~n~n'),
    format('src/~n'),
    format('  navigation/~n'),
    format('    AppNavigator.tsx      - Tab navigation~n'),
    format('  screens/~n'),
    format('    HomeScreen.tsx        - Tree list view~n'),
    format('    SearchScreen.tsx      - Search trees~n'),
    format('    FavoritesScreen.tsx   - Saved favorites~n'),
    format('    ProfileScreen.tsx     - User settings~n'),
    format('    TreeDetailScreen.tsx  - Mindmap view~n'),
    format('  hooks/~n'),
    format('    useTrees.ts           - Fetch trees query~n'),
    format('    useTreeDetail.ts      - Fetch tree detail~n'),
    format('    useSearch.ts          - Search query~n'),
    format('    useFavorites.ts       - Favorites mutation~n'),
    format('  store/~n'),
    format('    useAppStore.ts        - Zustand global state~n'),
    format('  storage/~n'),
    format('    useUserPrefs.ts       - AsyncStorage hook~n'),
    format('    useFavoritesCache.ts  - Offline favorites~n'),
    format('  components/~n'),
    format('    TreeCard.tsx          - Tree list item~n'),
    format('    MindMapView.tsx       - Mindmap visualization~n'),
    format('  api/~n'),
    format('    client.ts             - API client~n'),
    format('~n').

% ============================================================================
% PATTERN DEFINITIONS
% ============================================================================

%% Define all app patterns
define_app_patterns :-
    % Navigation
    navigation_pattern(tab, [
        screen(home, 'HomeScreen', [title('Home')]),
        screen(search, 'SearchScreen', [title('Search')]),
        screen(favorites, 'FavoritesScreen', [title('Favorites')]),
        screen(profile, 'ProfileScreen', [title('Profile')])
    ], NavPattern),
    define_pattern(app_navigation, NavPattern, [requires([navigation])]),

    % Detail navigation (stack within tab)
    navigation_pattern(stack, [
        screen(tree_list, 'TreeListScreen', []),
        screen(tree_detail, 'TreeDetailScreen', [])
    ], DetailNavPattern),
    define_pattern(detail_navigation, DetailNavPattern, []),

    % Data fetching patterns
    query_pattern(fetch_trees, '/api/trees', [stale_time(300000)], _),
    query_pattern(fetch_tree_detail, '/api/trees/:id', [stale_time(60000)], _),
    query_pattern(search_trees, '/api/search', [stale_time(30000)], _),
    mutation_pattern(toggle_favorite, '/api/favorites', [method('POST')], _),
    paginated_pattern(load_tree_feed, '/api/feed', [page_param(page)], _),

    % Global state
    global_state(appStore, [
        slice(ui, [
            field(theme, "'light' | 'dark'"),
            field(sidebarOpen, boolean)
        ], [
            action(toggleTheme, "(set) => set((state) => ({ theme: state.theme === 'light' ? 'dark' : 'light' }))"),
            action(toggleSidebar, "(set) => set((state) => ({ sidebarOpen: !state.sidebarOpen }))")
        ]),
        slice(filters, [
            field(sortBy, "'date' | 'name' | 'size'"),
            field(filterType, "string | null")
        ], [
            action(setSortBy, "(set, sortBy) => set({ sortBy })"),
            action(setFilterType, "(set, filterType) => set({ filterType })")
        ])
    ], _),

    % Persistence patterns
    local_storage(user_prefs, '{ theme: "light" | "dark"; fontSize: number }', _),
    local_storage(favorites_cache, '{ ids: string[]; lastSync: number }', _),

    format('App patterns defined~n').

% ============================================================================
% CODE GENERATION
% ============================================================================

%% generate_all_app_code/0
%
%  Generate all app components and display them.
%
generate_all_app_code :-
    format('~n=== Generating PT_Explorer App ===~n~n'),
    define_app_patterns,

    format('~n--- Navigation ---~n'),
    generate_app_component(navigation, NavCode),
    format('~w~n', [NavCode]),

    format('~n--- Data Hooks ---~n'),
    generate_app_component(fetch_trees, TreesHook),
    format('~w~n', [TreesHook]),

    format('~n--- Store ---~n'),
    generate_app_component(store, StoreCode),
    format('~w~n', [StoreCode]),

    format('~n--- Persistence ---~n'),
    generate_app_component(user_prefs, PrefsCode),
    format('~w~n', [PrefsCode]),

    format('~n--- Backend API ---~n'),
    generate_backend_api(BackendCode),
    format('~w~n', [BackendCode]),

    format('~n=== Generation Complete ===~n').

%% generate_app_component(+Component, -Code)
%
%  Generate a specific app component.
%
generate_app_component(navigation, Code) :-
    compile_pattern(app_navigation, react_native, [component_name('AppNavigator')], Code).

generate_app_component(detail_navigation, Code) :-
    compile_pattern(detail_navigation, react_native, [component_name('DetailNavigator')], Code).

generate_app_component(fetch_trees, Code) :-
    compile_pattern(fetch_trees, react_native, [], Code).

generate_app_component(fetch_tree_detail, Code) :-
    compile_pattern(fetch_tree_detail, react_native, [], Code).

generate_app_component(search_trees, Code) :-
    compile_pattern(search_trees, react_native, [], Code).

generate_app_component(toggle_favorite, Code) :-
    compile_pattern(toggle_favorite, react_native, [], Code).

generate_app_component(load_tree_feed, Code) :-
    compile_pattern(load_tree_feed, react_native, [], Code).

generate_app_component(store, Code) :-
    compile_pattern(appStore, react_native, [], Code).

generate_app_component(user_prefs, Code) :-
    compile_pattern(user_prefs, react_native, [], Code).

generate_app_component(favorites_cache, Code) :-
    compile_pattern(favorites_cache, react_native, [], Code).

%% generate_backend_api(-Code)
%
%  Generate Express backend for app patterns.
%
generate_backend_api(Code) :-
    generate_express_routes([
        fetch_trees,
        fetch_tree_detail,
        search_trees,
        toggle_favorite,
        load_tree_feed
    ], [router_name('pearltreesRouter')], Code).

% ============================================================================
% FULL APP GENERATION TO FILES
% ============================================================================

%% generate_app_to_files(+OutputDir)
%
%  Generate all app files to a directory.
%
generate_app_to_files(OutputDir) :-
    define_app_patterns,

    % Generate each component
    generate_and_save(app_navigation, OutputDir, 'navigation/AppNavigator.tsx'),
    generate_and_save(fetch_trees, OutputDir, 'hooks/useTrees.ts'),
    generate_and_save(fetch_tree_detail, OutputDir, 'hooks/useTreeDetail.ts'),
    generate_and_save(search_trees, OutputDir, 'hooks/useSearch.ts'),
    generate_and_save(toggle_favorite, OutputDir, 'hooks/useFavorites.ts'),
    generate_and_save(appStore, OutputDir, 'store/useAppStore.ts'),
    generate_and_save(user_prefs, OutputDir, 'storage/useUserPrefs.ts'),
    generate_and_save(favorites_cache, OutputDir, 'storage/useFavoritesCache.ts'),

    format('Generated app files to ~w~n', [OutputDir]).

generate_and_save(Pattern, OutputDir, RelPath) :-
    (   compile_pattern(Pattern, react_native, [], _Code)
    ->  format(string(FullPath), '~w/~w', [OutputDir, RelPath]),
        format('Would write to: ~w~n', [FullPath])
        % In real usage: open(FullPath, write, Stream), write(Stream, _Code), close(Stream)
    ;   format('Failed to generate: ~w~n', [Pattern])
    ).

% ============================================================================
% TESTING
% ============================================================================

test_generate_app :-
    format('~n=== App Generation Tests ===~n~n'),

    % Test 1: Define patterns
    format('Test 1: Define app patterns...~n'),
    define_app_patterns,
    (   pattern(app_navigation, _, _),
        pattern(fetch_trees, _, _),
        pattern(appStore, _, _)
    ->  format('  PASS: All patterns defined~n')
    ;   format('  FAIL: Some patterns missing~n')
    ),

    % Test 2: Generate navigation
    format('~nTest 2: Generate navigation...~n'),
    (   generate_app_component(navigation, NavCode),
        sub_string(NavCode, _, _, _, "createBottomTabNavigator")
    ->  format('  PASS: Tab navigator generated~n')
    ;   format('  FAIL: Navigation generation failed~n')
    ),

    % Test 3: Generate query hook
    format('~nTest 3: Generate query hook...~n'),
    (   generate_app_component(fetch_trees, QueryCode),
        sub_string(QueryCode, _, _, _, "useQuery")
    ->  format('  PASS: Query hook generated~n')
    ;   format('  FAIL: Query hook generation failed~n')
    ),

    % Test 4: Generate backend
    format('~nTest 4: Generate backend API...~n'),
    (   generate_backend_api(BackendCode),
        sub_string(BackendCode, _, _, _, "router.get")
    ->  format('  PASS: Express routes generated~n')
    ;   format('  FAIL: Backend generation failed~n')
    ),

    format('~n=== Tests Complete ===~n').

:- initialization((
    format('PT_Explorer generator loaded~n'),
    format('Run show_app_structure/0 to see app structure~n'),
    format('Run generate_all_app_code/0 to generate all code~n')
), now).
