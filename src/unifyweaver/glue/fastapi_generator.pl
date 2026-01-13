% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% fastapi_generator.pl - FastAPI Backend Code Generator
%
% Generates FastAPI routes, Pydantic models, and async handlers
% from UI pattern specifications.
%
% Features:
%   - Pydantic model generation for request/response validation
%   - Async route handlers with type hints
%   - OpenAPI schema generation
%   - Pagination support for list endpoints
%   - CRUD operations for data patterns
%
% Usage:
%   ?- generate_fastapi_handler(fetch_tasks, data(query, [...]), [], Code).
%   ?- generate_fastapi_app([pattern1, pattern2], [], Code).

:- module(fastapi_generator, [
    % Handler generation
    generate_fastapi_handler/4,        % +Name, +Spec, +Options, -Code
    generate_fastapi_query_handler/4,  % +Name, +Config, +Options, -Code
    generate_fastapi_mutation_handler/4, % +Name, +Config, +Options, -Code
    generate_fastapi_infinite_handler/4, % +Name, +Config, +Options, -Code

    % App generation
    generate_fastapi_app/3,            % +Patterns, +Options, -Code
    generate_fastapi_routes/3,         % +Patterns, +Options, -Code

    % Pydantic models
    generate_pydantic_models/3,        % +Schemas, +Options, -Code
    generate_pydantic_model/3,         % +Name, +Fields, -Code

    % Utilities
    generate_fastapi_imports/2,        % +Options, -Code
    generate_fastapi_config/2,         % +Options, -Code

    % Testing
    test_fastapi_generator/0
]).

:- use_module(library(lists)).

% ============================================================================
% HANDLER GENERATION
% ============================================================================

%% generate_fastapi_handler(+Name, +Spec, +Options, -Code)
%
%  Generate FastAPI handler for a pattern specification.
%
generate_fastapi_handler(Name, data(query, Config), Options, Code) :-
    generate_fastapi_query_handler(Name, Config, Options, Code).
generate_fastapi_handler(Name, data(mutation, Config), Options, Code) :-
    generate_fastapi_mutation_handler(Name, Config, Options, Code).
generate_fastapi_handler(Name, data(infinite, Config), Options, Code) :-
    generate_fastapi_infinite_handler(Name, Config, Options, Code).
generate_fastapi_handler(Name, _, _Options, Code) :-
    atom_string(Name, NameStr),
    format(string(Code),
"# Handler for ~w (generic)
@app.get(\"/api/~w\")
async def ~w():
    return {\"message\": \"Not implemented\"}
", [NameStr, NameStr, NameStr]).

%% generate_fastapi_query_handler(+Name, +Config, +Options, -Code)
%
%  Generate FastAPI GET handler for query patterns.
%
generate_fastapi_query_handler(Name, Config, _Options, Code) :-
    member(endpoint(Endpoint), Config),
    atom_string(Name, NameStr),
    atom_string(Endpoint, EndpointStr),
    snake_case(NameStr, FuncName),
    format(string(Code),
"@app.get(\"~w\")
async def ~w(
    page: int = Query(1, ge=1, description=\"Page number\"),
    limit: int = Query(20, ge=1, le=100, description=\"Items per page\")
) -> Dict[str, Any]:
    \"\"\"
    Fetch ~w with pagination.
    \"\"\"
    try:
        # TODO: Implement data fetching logic
        offset = (page - 1) * limit
        data = []  # Replace with actual data fetch
        total = 0  # Replace with actual count

        return {
            \"success\": True,
            \"data\": data,
            \"pagination\": {
                \"page\": page,
                \"limit\": limit,
                \"total\": total,
                \"hasMore\": page * limit < total
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
", [EndpointStr, FuncName, NameStr]).

%% generate_fastapi_mutation_handler(+Name, +Config, +Options, -Code)
%
%  Generate FastAPI POST/PUT/DELETE handler for mutation patterns.
%
generate_fastapi_mutation_handler(Name, Config, _Options, Code) :-
    member(endpoint(Endpoint), Config),
    (   member(method(Method), Config) -> true ; Method = 'POST' ),
    atom_string(Name, NameStr),
    atom_string(Endpoint, EndpointStr),
    atom_string(Method, MethodStr),
    string_lower(MethodStr, MethodLower),
    snake_case(NameStr, FuncName),
    capitalize_first(NameStr, ModelName),
    generate_mutation_decorator(MethodLower, EndpointStr, Decorator),
    generate_mutation_body(MethodLower, ModelName, FuncName, NameStr, Body),
    format(string(Code), "~w~w", [Decorator, Body]).

generate_mutation_decorator("post", Endpoint, Code) :-
    format(string(Code), "@app.post(\"~w\")~n", [Endpoint]).
generate_mutation_decorator("put", Endpoint, Code) :-
    format(string(Code), "@app.put(\"~w\")~n", [Endpoint]).
generate_mutation_decorator("patch", Endpoint, Code) :-
    format(string(Code), "@app.patch(\"~w\")~n", [Endpoint]).
generate_mutation_decorator("delete", Endpoint, Code) :-
    format(string(Code), "@app.delete(\"~w\")~n", [Endpoint]).
generate_mutation_decorator(_, Endpoint, Code) :-
    format(string(Code), "@app.post(\"~w\")~n", [Endpoint]).

generate_mutation_body("delete", _ModelName, FuncName, NameStr, Code) :-
    format(string(Code),
"async def ~w(id: str = Path(..., description=\"Item ID\")) -> Dict[str, Any]:
    \"\"\"
    Delete ~w by ID.
    \"\"\"
    try:
        # TODO: Implement delete logic
        return {\"success\": True, \"message\": \"Deleted successfully\"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
", [FuncName, NameStr]).

generate_mutation_body(_, ModelName, FuncName, NameStr, Code) :-
    format(string(Code),
"async def ~w(data: ~wInput) -> Dict[str, Any]:
    \"\"\"
    Create/update ~w.
    \"\"\"
    try:
        # TODO: Implement mutation logic
        result = data.dict()
        return {\"success\": True, \"data\": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
", [FuncName, ModelName, NameStr]).

%% generate_fastapi_infinite_handler(+Name, +Config, +Options, -Code)
%
%  Generate FastAPI handler for infinite scroll/pagination patterns.
%
generate_fastapi_infinite_handler(Name, Config, _Options, Code) :-
    member(endpoint(Endpoint), Config),
    (   member(page_param(PageParam), Config) -> true ; PageParam = 'cursor' ),
    atom_string(Name, NameStr),
    atom_string(Endpoint, EndpointStr),
    atom_string(PageParam, PageParamStr),
    snake_case(NameStr, FuncName),
    format(string(Code),
"@app.get(\"~w\")
async def ~w(
    ~w: Optional[str] = Query(None, description=\"Pagination cursor\"),
    limit: int = Query(20, ge=1, le=100, description=\"Items per page\")
) -> Dict[str, Any]:
    \"\"\"
    Fetch ~w with cursor-based pagination (infinite scroll).
    \"\"\"
    try:
        # TODO: Implement cursor-based pagination
        data = []  # Replace with actual data fetch
        next_cursor = None  # Set to next item's cursor if more items exist

        return {
            \"success\": True,
            \"data\": data,
            \"nextCursor\": next_cursor,
            \"hasMore\": next_cursor is not None
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
", [EndpointStr, FuncName, PageParamStr, NameStr]).

% ============================================================================
% APP GENERATION
% ============================================================================

%% generate_fastapi_app(+Patterns, +Options, -Code)
%
%  Generate complete FastAPI application with all handlers.
%
generate_fastapi_app(Patterns, Options, Code) :-
    generate_fastapi_imports(Options, Imports),
    generate_fastapi_config(Options, Config),
    generate_fastapi_routes(Patterns, Options, Routes),
    option_value(Options, app_name, 'API', AppName),
    format(string(Code),
"~w

~w

app = FastAPI(
    title=\"~w\",
    description=\"Auto-generated API from UnifyWeaver patterns\",
    version=\"1.0.0\"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[\"*\"],
    allow_credentials=True,
    allow_methods=[\"*\"],
    allow_headers=[\"*\"],
)

~w

# Health check endpoint
@app.get(\"/health\")
async def health_check():
    return {\"status\": \"healthy\"}

if __name__ == \"__main__\":
    import uvicorn
    uvicorn.run(app, host=\"0.0.0.0\", port=8000)
", [Imports, Config, AppName, Routes]).

%% generate_fastapi_routes(+Patterns, +Options, -Code)
%
%  Generate route handlers for all patterns.
%
generate_fastapi_routes(Patterns, Options, Code) :-
    findall(Handler, (
        member(P, Patterns),
        catch(ui_patterns:pattern(P, Spec, _), _, fail),
        generate_fastapi_handler(P, Spec, Options, Handler)
    ), Handlers),
    (   Handlers \= []
    ->  atomic_list_concat(Handlers, '\n\n', Code)
    ;   Code = "# No patterns to generate routes for"
    ).

%% generate_fastapi_imports(+Options, -Code)
%
%  Generate FastAPI import statements.
%
generate_fastapi_imports(_Options, Code) :-
    Code = "from fastapi import FastAPI, HTTPException, Query, Path, Body, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)".

%% generate_fastapi_config(+Options, -Code)
%
%  Generate FastAPI configuration and base models.
%
generate_fastapi_config(_Options, Code) :-
    Code = "# Base response models
class SuccessResponse(BaseModel):
    success: bool = True
    data: Optional[Any] = None
    message: Optional[str] = None

class ErrorResponse(BaseModel):
    success: bool = False
    error: str
    detail: Optional[str] = None

class PaginatedResponse(BaseModel):
    success: bool = True
    data: List[Any]
    pagination: Dict[str, Any]".

% ============================================================================
% PYDANTIC MODEL GENERATION
% ============================================================================

%% generate_pydantic_models(+Schemas, +Options, -Code)
%
%  Generate Pydantic models from schema specifications.
%
generate_pydantic_models(Schemas, _Options, Code) :-
    findall(Model, (
        member(schema(Name, Fields), Schemas),
        generate_pydantic_model(Name, Fields, Model)
    ), Models),
    atomic_list_concat(Models, '\n\n', Code).

%% generate_pydantic_model(+Name, +Fields, -Code)
%
%  Generate a single Pydantic model.
%
generate_pydantic_model(Name, Fields, Code) :-
    atom_string(Name, NameStr),
    capitalize_first(NameStr, ClassName),
    findall(FieldDef, (
        member(field(FName, FType), Fields),
        generate_pydantic_field(FName, FType, FieldDef)
    ), FieldDefs),
    atomic_list_concat(FieldDefs, '\n    ', FieldsStr),
    format(string(Code),
"class ~w(BaseModel):
    ~w

    class Config:
        from_attributes = True
", [ClassName, FieldsStr]).

generate_pydantic_field(Name, string, Code) :-
    atom_string(Name, NameStr),
    format(string(Code), "~w: str", [NameStr]).
generate_pydantic_field(Name, number, Code) :-
    atom_string(Name, NameStr),
    format(string(Code), "~w: float", [NameStr]).
generate_pydantic_field(Name, integer, Code) :-
    atom_string(Name, NameStr),
    format(string(Code), "~w: int", [NameStr]).
generate_pydantic_field(Name, boolean, Code) :-
    atom_string(Name, NameStr),
    format(string(Code), "~w: bool", [NameStr]).
generate_pydantic_field(Name, datetime, Code) :-
    atom_string(Name, NameStr),
    format(string(Code), "~w: datetime", [NameStr]).
generate_pydantic_field(Name, optional(Type), Code) :-
    atom_string(Name, NameStr),
    type_to_python(Type, PyType),
    format(string(Code), "~w: Optional[~w] = None", [NameStr, PyType]).
generate_pydantic_field(Name, list(Type), Code) :-
    atom_string(Name, NameStr),
    type_to_python(Type, PyType),
    format(string(Code), "~w: List[~w] = []", [NameStr, PyType]).
generate_pydantic_field(Name, _, Code) :-
    atom_string(Name, NameStr),
    format(string(Code), "~w: Any", [NameStr]).

type_to_python(string, "str").
type_to_python(number, "float").
type_to_python(integer, "int").
type_to_python(boolean, "bool").
type_to_python(datetime, "datetime").
type_to_python(_, "Any").

% ============================================================================
% AUTH HANDLERS
% ============================================================================

%% generate_fastapi_auth_handlers(+Options, -Code)
%
%  Generate authentication-related handlers.
%
generate_fastapi_auth_handlers(_Options, Code) :-
    Code = "# Authentication models
class UserLogin(BaseModel):
    email: str = Field(..., description=\"User email\")
    password: str = Field(..., description=\"User password\")

class UserRegister(BaseModel):
    email: str = Field(..., description=\"User email\")
    password: str = Field(..., min_length=8, description=\"User password\")
    confirm_password: str = Field(..., description=\"Confirm password\")

class AuthResponse(BaseModel):
    success: bool
    token: Optional[str] = None
    user: Optional[Dict[str, Any]] = None
    message: Optional[str] = None

@app.post(\"/api/auth/login\")
async def login(credentials: UserLogin) -> AuthResponse:
    \"\"\"
    User login endpoint.
    \"\"\"
    try:
        # TODO: Implement authentication logic
        # Verify credentials against database
        # Generate JWT token
        return AuthResponse(
            success=True,
            token=\"jwt_token_here\",
            user={\"email\": credentials.email}
        )
    except Exception as e:
        raise HTTPException(status_code=401, detail=str(e))

@app.post(\"/api/auth/register\")
async def register(data: UserRegister) -> AuthResponse:
    \"\"\"
    User registration endpoint.
    \"\"\"
    if data.password != data.confirm_password:
        raise HTTPException(status_code=400, detail=\"Passwords do not match\")

    try:
        # TODO: Implement registration logic
        # Create user in database
        # Generate JWT token
        return AuthResponse(
            success=True,
            token=\"jwt_token_here\",
            user={\"email\": data.email},
            message=\"Registration successful\"
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))".

% ============================================================================
% UTILITIES
% ============================================================================

option_value(Options, Key, Default, Value) :-
    Opt =.. [Key, Value],
    (   member(Opt, Options)
    ->  true
    ;   Value = Default
    ).

capitalize_first(Str, Cap) :-
    (   atom(Str) -> atom_string(Str, S) ; S = Str ),
    string_chars(S, [H|T]),
    upcase_atom(H, HU),
    atom_chars(HU, [HC]),
    string_chars(Cap, [HC|T]).

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

test_fastapi_generator :-
    format('~n=== FastAPI Generator Tests ===~n~n'),

    % Test 1: Query handler generation
    format('Test 1: Query handler generation...~n'),
    generate_fastapi_query_handler(fetch_items, [endpoint('/api/items')], [], Code1),
    (   sub_string(Code1, _, _, _, "@app.get")
    ->  format('  PASS: Generated GET handler~n')
    ;   format('  FAIL: Missing GET decorator~n')
    ),

    % Test 2: Mutation handler generation
    format('~nTest 2: Mutation handler generation...~n'),
    generate_fastapi_mutation_handler(create_item, [endpoint('/api/items'), method('POST')], [], Code2),
    (   sub_string(Code2, _, _, _, "@app.post")
    ->  format('  PASS: Generated POST handler~n')
    ;   format('  FAIL: Missing POST decorator~n')
    ),

    % Test 3: Infinite scroll handler
    format('~nTest 3: Infinite scroll handler...~n'),
    generate_fastapi_infinite_handler(load_feed, [endpoint('/api/feed')], [], Code3),
    (   sub_string(Code3, _, _, _, "cursor")
    ->  format('  PASS: Generated cursor-based pagination~n')
    ;   format('  FAIL: Missing cursor parameter~n')
    ),

    % Test 4: Pydantic model generation
    format('~nTest 4: Pydantic model generation...~n'),
    generate_pydantic_model(product, [field(name, string), field(price, number)], Code4),
    (   sub_string(Code4, _, _, _, "class Product"),
        sub_string(Code4, _, _, _, "name: str")
    ->  format('  PASS: Generated Pydantic model~n')
    ;   format('  FAIL: Model generation failed~n')
    ),

    % Test 5: Full app generation
    format('~nTest 5: Full app generation...~n'),
    generate_fastapi_app([], [app_name('TestAPI')], Code5),
    (   sub_string(Code5, _, _, _, "FastAPI"),
        sub_string(Code5, _, _, _, "CORSMiddleware")
    ->  format('  PASS: Generated full FastAPI app~n')
    ;   format('  FAIL: App generation failed~n')
    ),

    format('~n=== Tests Complete ===~n').

% ============================================================================
% INITIALIZATION
% ============================================================================

:- initialization((
    format('FastAPI generator module loaded~n', [])
), now).
