% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% flask_generator.pl - Flask Backend Code Generator
%
% Generates Flask routes and handlers from UI pattern specifications.
% Simpler alternative to FastAPI for legacy compatibility.
%
% Features:
%   - Flask route handlers with request validation
%   - Blueprint organization for modular APIs
%   - CORS support via flask-cors
%   - Error handling with JSON responses
%
% Usage:
%   ?- generate_flask_handler(fetch_tasks, data(query, [...]), [], Code).
%   ?- generate_flask_app([pattern1, pattern2], [], Code).

:- module(flask_generator, [
    % Handler generation
    generate_flask_handler/4,          % +Name, +Spec, +Options, -Code
    generate_flask_query_handler/4,    % +Name, +Config, +Options, -Code
    generate_flask_mutation_handler/4, % +Name, +Config, +Options, -Code
    generate_flask_infinite_handler/4, % +Name, +Config, +Options, -Code

    % App generation
    generate_flask_app/3,              % +Patterns, +Options, -Code
    generate_flask_routes/3,           % +Patterns, +Options, -Code
    generate_flask_blueprint/4,        % +Name, +Patterns, +Options, -Code

    % Utilities
    generate_flask_imports/2,          % +Options, -Code
    generate_flask_config/2,           % +Options, -Code

    % Testing
    test_flask_generator/0
]).

:- use_module(library(lists)).

% ============================================================================
% HANDLER GENERATION
% ============================================================================

%% generate_flask_handler(+Name, +Spec, +Options, -Code)
%
%  Generate Flask handler for a pattern specification.
%
generate_flask_handler(Name, data(query, Config), Options, Code) :-
    generate_flask_query_handler(Name, Config, Options, Code).
generate_flask_handler(Name, data(mutation, Config), Options, Code) :-
    generate_flask_mutation_handler(Name, Config, Options, Code).
generate_flask_handler(Name, data(infinite, Config), Options, Code) :-
    generate_flask_infinite_handler(Name, Config, Options, Code).
generate_flask_handler(Name, _, _Options, Code) :-
    atom_string(Name, NameStr),
    format(string(Code),
"# Handler for ~w (generic)
@app.route('/api/~w', methods=['GET'])
def ~w():
    return jsonify({'message': 'Not implemented'})
", [NameStr, NameStr, NameStr]).

%% generate_flask_query_handler(+Name, +Config, +Options, -Code)
%
%  Generate Flask GET handler for query patterns.
%
generate_flask_query_handler(Name, Config, _Options, Code) :-
    member(endpoint(Endpoint), Config),
    atom_string(Name, NameStr),
    atom_string(Endpoint, EndpointStr),
    snake_case(NameStr, FuncName),
    format(string(Code),
"@app.route('~w', methods=['GET'])
def ~w():
    \"\"\"Fetch ~w with pagination.\"\"\"
    try:
        page = request.args.get('page', 1, type=int)
        limit = request.args.get('limit', 20, type=int)

        # Validate pagination params
        page = max(1, page)
        limit = min(max(1, limit), 100)

        offset = (page - 1) * limit

        # TODO: Implement data fetching logic
        data = []  # Replace with actual data fetch
        total = 0  # Replace with actual count

        return jsonify({
            'success': True,
            'data': data,
            'pagination': {
                'page': page,
                'limit': limit,
                'total': total,
                'hasMore': page * limit < total
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500
", [EndpointStr, FuncName, NameStr]).

%% generate_flask_mutation_handler(+Name, +Config, +Options, -Code)
%
%  Generate Flask POST/PUT/DELETE handler for mutation patterns.
%
generate_flask_mutation_handler(Name, Config, _Options, Code) :-
    member(endpoint(Endpoint), Config),
    (   member(method(Method), Config) -> true ; Method = 'POST' ),
    atom_string(Name, NameStr),
    atom_string(Endpoint, EndpointStr),
    atom_string(Method, MethodStr),
    string_upper(MethodStr, MethodUpper),
    snake_case(NameStr, FuncName),
    generate_flask_mutation_body(MethodUpper, EndpointStr, FuncName, NameStr, Code).

generate_flask_mutation_body("DELETE", Endpoint, FuncName, NameStr, Code) :-
    format(string(Code),
"@app.route('~w', methods=['DELETE'])
def ~w():
    \"\"\"Delete ~w.\"\"\"
    try:
        # TODO: Implement delete logic
        return jsonify({
            'success': True,
            'message': 'Deleted successfully'
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500
", [Endpoint, FuncName, NameStr]).

generate_flask_mutation_body(Method, Endpoint, FuncName, NameStr, Code) :-
    Method \= "DELETE",
    format(string(Code),
"@app.route('~w', methods=['~w'])
def ~w():
    \"\"\"Create/update ~w.\"\"\"
    try:
        data = request.get_json()

        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400

        # TODO: Implement mutation logic
        result = data

        return jsonify({
            'success': True,
            'data': result
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500
", [Endpoint, Method, FuncName, NameStr]).

%% generate_flask_infinite_handler(+Name, +Config, +Options, -Code)
%
%  Generate Flask handler for infinite scroll/pagination patterns.
%
generate_flask_infinite_handler(Name, Config, _Options, Code) :-
    member(endpoint(Endpoint), Config),
    (   member(page_param(PageParam), Config) -> true ; PageParam = 'cursor' ),
    atom_string(Name, NameStr),
    atom_string(Endpoint, EndpointStr),
    atom_string(PageParam, PageParamStr),
    snake_case(NameStr, FuncName),
    format(string(Code),
"@app.route('~w', methods=['GET'])
def ~w():
    \"\"\"Fetch ~w with cursor-based pagination (infinite scroll).\"\"\"
    try:
        ~w = request.args.get('~w', None)
        limit = request.args.get('limit', 20, type=int)
        limit = min(max(1, limit), 100)

        # TODO: Implement cursor-based pagination
        data = []  # Replace with actual data fetch
        next_cursor = None  # Set to next item's cursor if more items exist

        return jsonify({
            'success': True,
            'data': data,
            'nextCursor': next_cursor,
            'hasMore': next_cursor is not None
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500
", [EndpointStr, FuncName, NameStr, PageParamStr, PageParamStr]).

% ============================================================================
% APP GENERATION
% ============================================================================

%% generate_flask_app(+Patterns, +Options, -Code)
%
%  Generate complete Flask application with all handlers.
%
generate_flask_app(Patterns, Options, Code) :-
    generate_flask_imports(Options, Imports),
    generate_flask_config(Options, Config),
    generate_flask_routes(Patterns, Options, Routes),
    format(string(Code),
"~w

~w

~w

# Health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'})

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'success': False, 'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'success': False, 'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
", [Imports, Config, Routes]).

%% generate_flask_routes(+Patterns, +Options, -Code)
%
%  Generate route handlers for all patterns.
%
generate_flask_routes(Patterns, Options, Code) :-
    findall(Handler, (
        member(P, Patterns),
        catch(ui_patterns:pattern(P, Spec, _), _, fail),
        generate_flask_handler(P, Spec, Options, Handler)
    ), Handlers),
    (   Handlers \= []
    ->  atomic_list_concat(Handlers, '\n\n', Code)
    ;   Code = "# No patterns to generate routes for"
    ).

%% generate_flask_blueprint(+Name, +Patterns, +Options, -Code)
%
%  Generate a Flask Blueprint for modular organization.
%
generate_flask_blueprint(Name, Patterns, Options, Code) :-
    atom_string(Name, NameStr),
    snake_case(NameStr, BpName),
    findall(Handler, (
        member(P, Patterns),
        catch(ui_patterns:pattern(P, Spec, _), _, fail),
        generate_flask_bp_handler(BpName, P, Spec, Options, Handler)
    ), Handlers),
    atomic_list_concat(Handlers, '\n\n', HandlersStr),
    format(string(Code),
"from flask import Blueprint, jsonify, request

~w_bp = Blueprint('~w', __name__, url_prefix='/api/~w')

~w
", [BpName, BpName, BpName, HandlersStr]).

generate_flask_bp_handler(BpName, Name, data(query, Config), _Options, Code) :-
    member(endpoint(Endpoint), Config),
    atom_string(Name, NameStr),
    atom_string(Endpoint, EndpointStr),
    snake_case(NameStr, FuncName),
    % Remove /api prefix for blueprint routes
    (   sub_string(EndpointStr, 0, 4, _, "/api")
    ->  sub_string(EndpointStr, 4, _, 0, Route)
    ;   Route = EndpointStr
    ),
    format(string(Code),
"@~w_bp.route('~w', methods=['GET'])
def ~w():
    page = request.args.get('page', 1, type=int)
    limit = request.args.get('limit', 20, type=int)
    # TODO: Implement
    return jsonify({'success': True, 'data': []})
", [BpName, Route, FuncName]).

generate_flask_bp_handler(BpName, Name, _, _Options, Code) :-
    atom_string(Name, NameStr),
    snake_case(NameStr, FuncName),
    format(string(Code),
"@~w_bp.route('/~w', methods=['GET'])
def ~w():
    return jsonify({'message': 'Not implemented'})
", [BpName, NameStr, FuncName]).

%% generate_flask_imports(+Options, -Code)
%
%  Generate Flask import statements.
%
generate_flask_imports(_Options, Code) :-
    Code = "from flask import Flask, jsonify, request
from flask_cors import CORS
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)".

%% generate_flask_config(+Options, -Code)
%
%  Generate Flask app configuration.
%
generate_flask_config(Options, Code) :-
    option_value(Options, app_name, 'FlaskAPI', AppName),
    format(string(Code),
"app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = True

# Enable CORS
CORS(app, resources={r\"/api/*\": {\"origins\": \"*\"}})

# App name: ~w
", [AppName]).

% ============================================================================
% AUTH HANDLERS
% ============================================================================

%% generate_flask_auth_handlers(+Options, -Code)
%
%  Generate authentication-related handlers.
%
generate_flask_auth_handlers(_Options, Code) :-
    Code = "@app.route('/api/auth/login', methods=['POST'])
def login():
    \"\"\"User login endpoint.\"\"\"
    try:
        data = request.get_json()

        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400

        email = data.get('email')
        password = data.get('password')

        if not email or not password:
            return jsonify({'success': False, 'error': 'Email and password required'}), 400

        # TODO: Implement authentication logic
        # Verify credentials against database
        # Generate JWT token

        return jsonify({
            'success': True,
            'token': 'jwt_token_here',
            'user': {'email': email}
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 401

@app.route('/api/auth/register', methods=['POST'])
def register():
    \"\"\"User registration endpoint.\"\"\"
    try:
        data = request.get_json()

        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400

        email = data.get('email')
        password = data.get('password')
        confirm_password = data.get('confirm_password')

        if not email or not password:
            return jsonify({'success': False, 'error': 'Email and password required'}), 400

        if password != confirm_password:
            return jsonify({'success': False, 'error': 'Passwords do not match'}), 400

        if len(password) < 8:
            return jsonify({'success': False, 'error': 'Password must be at least 8 characters'}), 400

        # TODO: Implement registration logic
        # Create user in database
        # Generate JWT token

        return jsonify({
            'success': True,
            'token': 'jwt_token_here',
            'user': {'email': email},
            'message': 'Registration successful'
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400".

% ============================================================================
% UTILITIES
% ============================================================================

option_value(Options, Key, Default, Value) :-
    Opt =.. [Key, Value],
    (   member(Opt, Options)
    ->  true
    ;   Value = Default
    ).

snake_case(Str, Snake) :-
    (   atom(Str) -> atom_string(Str, S) ; S = Str ),
    string_chars(S, Chars),
    snake_chars(Chars, SnakeChars),
    string_chars(Snake, SnakeChars).

snake_chars([], []).
snake_chars([C|Cs], [L|Rs]) :-
    char_type(C, upper),
    !,
    downcase_char(C, L),
    snake_chars(Cs, Rs).
snake_chars([C|Cs], [C|Rs]) :-
    snake_chars(Cs, Rs).

downcase_char(C, L) :-
    char_code(C, Code),
    (   Code >= 65, Code =< 90
    ->  LCode is Code + 32,
        char_code(L, LCode)
    ;   L = C
    ).

% ============================================================================
% TESTING
% ============================================================================

test_flask_generator :-
    format('~n=== Flask Generator Tests ===~n~n'),

    % Test 1: Query handler generation
    format('Test 1: Query handler generation...~n'),
    generate_flask_query_handler(fetch_items, [endpoint('/api/items')], [], Code1),
    (   sub_string(Code1, _, _, _, "@app.route"),
        sub_string(Code1, _, _, _, "GET")
    ->  format('  PASS: Generated GET handler~n')
    ;   format('  FAIL: Missing route decorator~n')
    ),

    % Test 2: Mutation handler generation
    format('~nTest 2: Mutation handler generation...~n'),
    generate_flask_mutation_handler(create_item, [endpoint('/api/items'), method('POST')], [], Code2),
    (   sub_string(Code2, _, _, _, "POST")
    ->  format('  PASS: Generated POST handler~n')
    ;   format('  FAIL: Missing POST method~n')
    ),

    % Test 3: Delete handler
    format('~nTest 3: Delete handler...~n'),
    generate_flask_mutation_handler(delete_item, [endpoint('/api/items'), method('DELETE')], [], Code3),
    (   sub_string(Code3, _, _, _, "DELETE")
    ->  format('  PASS: Generated DELETE handler~n')
    ;   format('  FAIL: Missing DELETE method~n')
    ),

    % Test 4: Infinite scroll handler
    format('~nTest 4: Infinite scroll handler...~n'),
    generate_flask_infinite_handler(load_feed, [endpoint('/api/feed')], [], Code4),
    (   sub_string(Code4, _, _, _, "cursor")
    ->  format('  PASS: Generated cursor-based pagination~n')
    ;   format('  FAIL: Missing cursor parameter~n')
    ),

    % Test 5: Full app generation
    format('~nTest 5: Full app generation...~n'),
    generate_flask_app([], [app_name('TestAPI')], Code5),
    (   sub_string(Code5, _, _, _, "Flask"),
        sub_string(Code5, _, _, _, "CORS")
    ->  format('  PASS: Generated full Flask app~n')
    ;   format('  FAIL: App generation failed~n')
    ),

    format('~n=== Tests Complete ===~n').

% ============================================================================
% INITIALIZATION
% ============================================================================

:- initialization((
    format('Flask generator module loaded~n', [])
), now).
