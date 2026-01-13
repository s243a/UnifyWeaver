% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% ecommerce_app.pl - Full-Featured E-commerce Example
%
% Demonstrates all UI patterns working together in a complete application:
%   - Product catalog with infinite scroll
%   - Shopping cart with global state
%   - Checkout form with validation
%   - User authentication flows
%   - Order history
%   - Search functionality
%
% Generates complete project files for any frontend target with Python backend.
%
% Usage:
%   ?- generate_ecommerce_project(react_native, '/output/ecommerce-rn').
%   ?- generate_ecommerce_full_stack(vue, fastapi, '/output/ecommerce-vue').
%   ?- generate_ecommerce_all_frontends('/output/ecommerce').

:- module(ecommerce_app, [
    % App definition
    define_ecommerce_app/0,
    ecommerce_patterns/1,

    % Project generation
    generate_ecommerce_project/2,          % +Target, +OutputDir
    generate_ecommerce_full_stack/3,       % +FrontendTarget, +BackendTarget, +OutputDir
    generate_ecommerce_all_frontends/1,    % +BaseOutputDir

    % Compilation
    compile_ecommerce_app/2,               % +Target, -Code

    % Testing
    test_ecommerce_app/0
]).

:- use_module('../../patterns/ui_patterns').
:- use_module('../../patterns/ui_patterns_extended').
:- use_module('../../glue/pattern_glue').
:- use_module('../../glue/project_generator').
:- use_module('../../glue/fastapi_generator').
:- use_module('../../glue/flask_generator').

% Load target modules
:- catch(use_module('../../targets/vue_target', []), _, true).
:- catch(use_module('../../targets/flutter_target', []), _, true).
:- catch(use_module('../../targets/swiftui_target', []), _, true).

% ============================================================================
% APP SPECIFICATION
% ============================================================================

%% define_ecommerce_app/0
%
%  Define all patterns for the E-commerce application.
%
define_ecommerce_app :-
    % Clean up any existing patterns
    cleanup_ecommerce_patterns,

    % === NAVIGATION PATTERNS ===

    % Main tab navigation
    ui_patterns:define_pattern(main_tabs,
        navigation(tab, [
            screen(home, 'HomeScreen', [title('Home'), icon('home')]),
            screen(search, 'SearchScreen', [title('Search'), icon('search')]),
            screen(cart, 'CartScreen', [title('Cart'), icon('cart')]),
            screen(profile, 'ProfileScreen', [title('Profile'), icon('user')])
        ], []),
        [requires([navigation])]),

    % Product stack navigation
    ui_patterns:define_pattern(product_stack,
        navigation(stack, [
            screen(product_list, 'ProductListScreen', []),
            screen(product_detail, 'ProductDetailScreen', []),
            screen(reviews, 'ReviewsScreen', [])
        ], []),
        [requires([navigation]), depends_on([main_tabs])]),

    % Checkout stack navigation
    ui_patterns:define_pattern(checkout_stack,
        navigation(stack, [
            screen(checkout, 'CheckoutScreen', []),
            screen(payment, 'PaymentScreen', []),
            screen(confirmation, 'ConfirmationScreen', [])
        ], []),
        [requires([navigation])]),

    % Order stack navigation
    ui_patterns:define_pattern(order_stack,
        navigation(stack, [
            screen(order_list, 'OrderListScreen', []),
            screen(order_detail, 'OrderDetailScreen', [])
        ], []),
        [requires([navigation])]),

    % === STATE PATTERNS ===

    % Shopping cart state
    ui_patterns:define_pattern(cart_store,
        state(global, [
            store(cartStore),
            slices([
                slice(cart, [
                    field(items, 'CartItem[]'),
                    field(total, number),
                    field(itemCount, number)
                ], [
                    action(addItem, "(set, item) => set(s => {
                        const existing = s.items.find(i => i.id === item.id);
                        if (existing) {
                            return { items: s.items.map(i => i.id === item.id ? {...i, quantity: i.quantity + 1} : i) };
                        }
                        return { items: [...s.items, {...item, quantity: 1}] };
                    })"),
                    action(removeItem, "(set, id) => set(s => ({ items: s.items.filter(i => i.id !== id) }))"),
                    action(updateQuantity, "(set, id, quantity) => set(s => ({
                        items: s.items.map(i => i.id === id ? {...i, quantity} : i)
                    }))"),
                    action(clearCart, "(set) => set({ items: [], total: 0, itemCount: 0 })"),
                    action(calculateTotal, "(set) => set(s => ({
                        total: s.items.reduce((sum, i) => sum + i.price * i.quantity, 0),
                        itemCount: s.items.reduce((sum, i) => sum + i.quantity, 0)
                    }))")
                ])
            ])
        ], []),
        [requires([zustand]), singleton(true)]),

    % User auth state
    ui_patterns:define_pattern(user_store,
        state(global, [
            store(userStore),
            slices([
                slice(user, [
                    field(isAuthenticated, boolean),
                    field(token, 'string | null'),
                    field(profile, 'User | null'),
                    field(loading, boolean)
                ], [
                    action(setUser, "(set, user, token) => set({ isAuthenticated: true, profile: user, token })"),
                    action(logout, "(set) => set({ isAuthenticated: false, profile: null, token: null })"),
                    action(setLoading, "(set, loading) => set({ loading })")
                ])
            ])
        ], []),
        [requires([zustand]), singleton(true)]),

    % Search/filter state
    ui_patterns:define_pattern(filter_store,
        state(global, [
            store(filterStore),
            slices([
                slice(filters, [
                    field(query, string),
                    field(category, 'string | null'),
                    field(minPrice, 'number | null'),
                    field(maxPrice, 'number | null'),
                    field(sortBy, 'price | name | rating')
                ], [
                    action(setQuery, "(set, query) => set({ query })"),
                    action(setCategory, "(set, category) => set({ category })"),
                    action(setPriceRange, "(set, min, max) => set({ minPrice: min, maxPrice: max })"),
                    action(setSortBy, "(set, sortBy) => set({ sortBy })"),
                    action(clearFilters, "(set) => set({ query: '', category: null, minPrice: null, maxPrice: null })")
                ])
            ])
        ], []),
        [requires([zustand])]),

    % === DATA FETCHING PATTERNS ===

    % Products query (paginated)
    ui_patterns:define_pattern(fetch_products,
        data(query, [
            name(fetchProducts),
            endpoint('/api/products'),
            stale_time(60000),
            retry(3)
        ]),
        [requires([react_query])]),

    % Single product query
    ui_patterns:define_pattern(fetch_product,
        data(query, [
            name(fetchProduct),
            endpoint('/api/products/:id'),
            stale_time(30000)
        ]),
        [requires([react_query])]),

    % Product search
    ui_patterns:define_pattern(search_products,
        data(query, [
            name(searchProducts),
            endpoint('/api/search'),
            stale_time(30000)
        ]),
        [requires([react_query])]),

    % Categories
    ui_patterns:define_pattern(fetch_categories,
        data(query, [
            name(fetchCategories),
            endpoint('/api/categories'),
            stale_time(300000)
        ]),
        [requires([react_query])]),

    % Orders query
    ui_patterns:define_pattern(fetch_orders,
        data(query, [
            name(fetchOrders),
            endpoint('/api/orders'),
            stale_time(60000)
        ]),
        [requires([react_query])]),

    % Single order
    ui_patterns:define_pattern(fetch_order,
        data(query, [
            name(fetchOrder),
            endpoint('/api/orders/:id'),
            stale_time(30000)
        ]),
        [requires([react_query])]),

    % === MUTATION PATTERNS ===

    % Create order
    ui_patterns:define_pattern(create_order,
        data(mutation, [
            name(createOrder),
            endpoint('/api/orders'),
            method('POST'),
            invalidates([fetch_orders])
        ]),
        [requires([react_query])]),

    % Add to cart (API sync)
    ui_patterns:define_pattern(sync_cart,
        data(mutation, [
            name(syncCart),
            endpoint('/api/cart'),
            method('POST')
        ]),
        [requires([react_query])]),

    % === PERSISTENCE PATTERNS ===

    % Cart persistence
    ui_patterns:define_pattern(cart_persistence,
        persistence(local, [
            key(cart),
            schema('{ items: CartItem[], total: number }')
        ]),
        [requires([async_storage])]),

    % User preferences
    ui_patterns:define_pattern(user_prefs,
        persistence(local, [
            key(userPrefs),
            schema('{ theme: "light" | "dark", currency: string, recentlyViewed: string[] }')
        ]),
        [requires([async_storage])]),

    % Auth token storage
    ui_patterns:define_pattern(auth_storage,
        persistence(secure, [
            key(authToken),
            schema('string')
        ]),
        [requires([secure_storage])]),

    % === FORM PATTERNS ===

    % Checkout form
    define_checkout_form,

    % === AUTH PATTERNS ===

    % Login flow
    define_auth_patterns.

%% define_checkout_form/0
%
%  Define the checkout form pattern.
%
define_checkout_form :-
    ui_patterns_extended:form_pattern(checkout_form, [
        field(email, email, [required], [placeholder('Email address')]),
        field(firstName, text, [required, min_length(2)], [placeholder('First name')]),
        field(lastName, text, [required, min_length(2)], [placeholder('Last name')]),
        field(address, text, [required], [placeholder('Street address')]),
        field(city, text, [required], [placeholder('City')]),
        field(state, text, [required], [placeholder('State/Province')]),
        field(zipCode, text, [required, pattern('^[0-9]{5}(-[0-9]{4})?$')], [placeholder('ZIP code')]),
        field(country, select, [required], [options(['US', 'CA', 'UK', 'AU'])]),
        field(cardNumber, text, [required, pattern('^[0-9]{16}$')], [placeholder('Card number')]),
        field(cardExpiry, text, [required, pattern('^(0[1-9]|1[0-2])/[0-9]{2}$')], [placeholder('MM/YY')]),
        field(cardCvv, password, [required, min_length(3), max_length(4)], [placeholder('CVV')])
    ], _).

%% define_auth_patterns/0
%
%  Define authentication flow patterns.
%
define_auth_patterns :-
    % Login
    ui_patterns_extended:login_flow([endpoint('/api/auth/login')], _),

    % Register
    ui_patterns_extended:register_flow([endpoint('/api/auth/register')], _),

    % Forgot password
    ui_patterns_extended:forgot_password_flow([endpoint('/api/auth/forgot-password')], _).

%% cleanup_ecommerce_patterns/0
%
%  Remove existing patterns before redefining.
%
cleanup_ecommerce_patterns :-
    Patterns = [
        main_tabs, product_stack, checkout_stack, order_stack,
        cart_store, user_store, filter_store,
        fetch_products, fetch_product, search_products, fetch_categories,
        fetch_orders, fetch_order, create_order, sync_cart,
        cart_persistence, user_prefs, auth_storage
    ],
    forall(member(P, Patterns),
           retractall(ui_patterns:stored_pattern(P, _, _))).

% ============================================================================
% APP PATTERNS LIST
% ============================================================================

%% ecommerce_patterns(-Patterns)
%
%  List of all patterns in the E-commerce app.
%
ecommerce_patterns([
    % Navigation
    main_tabs, product_stack, checkout_stack, order_stack,
    % State
    cart_store, user_store, filter_store,
    % Data queries
    fetch_products, fetch_product, search_products, fetch_categories,
    fetch_orders, fetch_order,
    % Mutations
    create_order, sync_cart,
    % Persistence
    cart_persistence, user_prefs, auth_storage
]).

% ============================================================================
% PROJECT GENERATION
% ============================================================================

%% generate_ecommerce_project(+Target, +OutputDir)
%
%  Generate complete e-commerce project for a frontend target.
%
generate_ecommerce_project(Target, OutputDir) :-
    define_ecommerce_app,
    ecommerce_patterns(Patterns),
    project_generator:generate_project(
        app(ecommerce, Patterns, []),
        Target,
        OutputDir,
        Result
    ),
    format('Generated ~w project at ~w~n', [Target, OutputDir]),
    Result = project_result(_, _, _, Dirs, Files),
    format('  Directories: ~w~n', [Dirs]),
    format('  Files: ~w~n', [Files]).

%% generate_ecommerce_full_stack(+FrontendTarget, +BackendTarget, +OutputDir)
%
%  Generate full-stack e-commerce project.
%
generate_ecommerce_full_stack(FrontendTarget, BackendTarget, OutputDir) :-
    define_ecommerce_app,
    ecommerce_patterns(Patterns),
    project_generator:generate_full_stack_project(
        app(ecommerce, Patterns, []),
        FrontendTarget,
        BackendTarget,
        OutputDir,
        Result
    ),
    format('Generated full-stack project at ~w~n', [OutputDir]),
    format('  Frontend: ~w~n', [FrontendTarget]),
    format('  Backend: ~w~n', [BackendTarget]),
    Result = full_stack_result(FrontendResult, BackendResult),
    format('  Frontend result: ~w~n', [FrontendResult]),
    format('  Backend result: ~w~n', [BackendResult]).

%% generate_ecommerce_all_frontends(+BaseOutputDir)
%
%  Generate e-commerce for all frontend targets.
%
generate_ecommerce_all_frontends(BaseOutputDir) :-
    Targets = [react_native, vue, flutter, swiftui],
    forall(member(T, Targets), (
        atom_concat(BaseOutputDir, '/', Prefix),
        atom_concat(Prefix, T, OutputDir),
        catch(
            generate_ecommerce_project(T, OutputDir),
            Error,
            format('Error generating ~w: ~w~n', [T, Error])
        )
    )).

% ============================================================================
% COMPILATION
% ============================================================================

%% compile_ecommerce_app(+Target, -Code)
%
%  Compile the e-commerce app patterns to target code.
%
compile_ecommerce_app(Target, Code) :-
    define_ecommerce_app,
    ecommerce_patterns(Patterns),
    compile_patterns_to_target(Patterns, Target, [], Codes),
    atomic_list_concat(Codes, '\n\n// ============\n\n', Code).

compile_patterns_to_target([], _, _, []).
compile_patterns_to_target([P|Ps], Target, Options, [Code|Codes]) :-
    (   catch(ui_patterns:compile_pattern(P, Target, Options, Code), _, fail)
    ->  true
    ;   format(string(Code), "// Pattern ~w not compilable for ~w", [P, Target])
    ),
    compile_patterns_to_target(Ps, Target, Options, Codes).

% ============================================================================
% BACKEND ENDPOINTS
% ============================================================================

%% ecommerce_endpoints(-Endpoints)
%
%  List of all API endpoints for the e-commerce backend.
%
ecommerce_endpoints([
    % Products
    endpoint(list_products, get, '/api/products', list_products_handler),
    endpoint(get_product, get, '/api/products/:id', get_product_handler),
    endpoint(search_products, get, '/api/search', search_products_handler),

    % Categories
    endpoint(list_categories, get, '/api/categories', list_categories_handler),

    % Cart
    endpoint(get_cart, get, '/api/cart', get_cart_handler),
    endpoint(add_to_cart, post, '/api/cart/items', add_to_cart_handler),
    endpoint(update_cart_item, put, '/api/cart/items/:id', update_cart_item_handler),
    endpoint(remove_from_cart, delete, '/api/cart/items/:id', remove_from_cart_handler),

    % Orders
    endpoint(list_orders, get, '/api/orders', list_orders_handler),
    endpoint(get_order, get, '/api/orders/:id', get_order_handler),
    endpoint(create_order, post, '/api/orders', create_order_handler),

    % Auth
    endpoint(login, post, '/api/auth/login', login_handler),
    endpoint(register, post, '/api/auth/register', register_handler),
    endpoint(forgot_password, post, '/api/auth/forgot-password', forgot_password_handler),

    % User
    endpoint(get_profile, get, '/api/user/profile', get_profile_handler),
    endpoint(update_profile, put, '/api/user/profile', update_profile_handler)
]).

% ============================================================================
% TESTING
% ============================================================================

test_ecommerce_app :-
    format('~n=== E-commerce App Tests ===~n~n'),

    % Test 1: Pattern definition
    format('Test 1: Pattern definition...~n'),
    define_ecommerce_app,
    (   ui_patterns:stored_pattern(main_tabs, _, _),
        ui_patterns:stored_pattern(cart_store, _, _)
    ->  format('  PASS: Patterns defined~n')
    ;   format('  FAIL: Patterns not defined~n')
    ),

    % Test 2: Pattern count
    format('~nTest 2: Pattern count...~n'),
    ecommerce_patterns(Patterns),
    length(Patterns, Count),
    (   Count >= 15
    ->  format('  PASS: ~w patterns defined~n', [Count])
    ;   format('  FAIL: Only ~w patterns~n', [Count])
    ),

    % Test 3: Cart store has actions
    format('~nTest 3: Cart store actions...~n'),
    (   ui_patterns:stored_pattern(cart_store, Spec, _),
        Spec = state(global, Shape, _),
        member(slices(Slices), Shape),
        member(slice(cart, _, Actions), Slices),
        length(Actions, ActionCount),
        ActionCount >= 4
    ->  format('  PASS: Cart store has ~w actions~n', [ActionCount])
    ;   format('  FAIL: Cart store missing actions~n')
    ),

    % Test 4: Data queries have endpoints
    format('~nTest 4: Data query endpoints...~n'),
    (   ui_patterns:stored_pattern(fetch_products, data(query, Config), _),
        member(endpoint('/api/products'), Config)
    ->  format('  PASS: Products query has endpoint~n')
    ;   format('  FAIL: Products query missing endpoint~n')
    ),

    % Test 5: Endpoints list
    format('~nTest 5: Backend endpoints...~n'),
    ecommerce_endpoints(Endpoints),
    length(Endpoints, EndpointCount),
    (   EndpointCount >= 15
    ->  format('  PASS: ~w endpoints defined~n', [EndpointCount])
    ;   format('  FAIL: Only ~w endpoints~n', [EndpointCount])
    ),

    format('~n=== Tests Complete ===~n').

% ============================================================================
% INITIALIZATION
% ============================================================================

:- initialization((
    format('E-commerce app module loaded~n', [])
), now).
