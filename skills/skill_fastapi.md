# Skill: FastAPI

Generate FastAPI REST APIs with Pydantic models, async handlers, and automatic OpenAPI schemas.

## When to Use

- User asks "how do I create a FastAPI app?"
- User needs type-validated Python APIs
- User wants automatic OpenAPI documentation
- User needs async REST endpoints

## Quick Start

```prolog
:- use_module('src/unifyweaver/glue/fastapi_generator').

% Generate complete FastAPI app
generate_fastapi_app([list_products, create_product], [app_name('ProductAPI')], Code).

% Generate Pydantic model
generate_pydantic_model(product, [
    field(name, string),
    field(price, number),
    field(in_stock, boolean)
], Model).
```

## Handler Generation

### Query Handler (GET with Type Hints)

```prolog
generate_fastapi_query_handler(fetch_items, [endpoint('/api/items')], [], Code).
```

**Output:**
```python
@app.get("/api/items")
async def fetch_items(
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(20, ge=1, le=100, description="Items per page")
) -> Dict[str, Any]:
    """
    Fetch fetch_items with pagination.
    """
    try:
        offset = (page - 1) * limit
        data = []  # Replace with actual data fetch
        total = 0
        return {
            "success": True,
            "data": data,
            "pagination": {
                "page": page,
                "limit": limit,
                "total": total,
                "hasMore": page * limit < total
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

### Mutation Handler (POST/PUT/DELETE)

```prolog
% POST with Pydantic input
generate_fastapi_mutation_handler(create_item, [endpoint('/api/items'), method('POST')], [], Code).

% DELETE with path parameter
generate_fastapi_mutation_handler(delete_item, [endpoint('/api/items/{id}'), method('DELETE')], [], Code).
```

**POST Output:**
```python
@app.post("/api/items")
async def create_item(data: CreateItemInput) -> Dict[str, Any]:
    """
    Create/update create_item.
    """
    try:
        result = data.dict()
        return {"success": True, "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

### Infinite Scroll Handler

```prolog
generate_fastapi_infinite_handler(load_feed, [endpoint('/api/feed')], [], Code).
```

**Output:**
```python
@app.get("/api/feed")
async def load_feed(
    cursor: Optional[str] = Query(None, description="Pagination cursor"),
    limit: int = Query(20, ge=1, le=100, description="Items per page")
) -> Dict[str, Any]:
    """
    Fetch load_feed with cursor-based pagination (infinite scroll).
    """
```

## Pydantic Models

### Generate Model

```prolog
generate_pydantic_model(Name, Fields, Code).
```

**Field Types:**
- `string` - `str`
- `number` - `float`
- `integer` - `int`
- `boolean` - `bool`
- `datetime` - `datetime`
- `optional(Type)` - `Optional[Type] = None`
- `list(Type)` - `List[Type] = []`

**Example:**
```prolog
generate_pydantic_model(user, [
    field(name, string),
    field(email, string),
    field(age, optional(integer)),
    field(roles, list(string))
], Code).
```

**Output:**
```python
class User(BaseModel):
    name: str
    email: str
    age: Optional[int] = None
    roles: List[str] = []

    class Config:
        from_attributes = True
```

### Generate Multiple Models

```prolog
generate_pydantic_models([
    schema(product, [field(name, string), field(price, number)]),
    schema(order, [field(product_id, string), field(quantity, integer)])
], [], Code).
```

## App Generation

### Complete App

```prolog
generate_fastapi_app(Patterns, Options, Code).
```

**Options:**
- `app_name(Name)` - API title for OpenAPI (default: 'API')

**Generates:**
- FastAPI imports (Query, Path, HTTPException, etc.)
- CORS middleware configuration
- Base response models (SuccessResponse, ErrorResponse, PaginatedResponse)
- All route handlers
- Health check endpoint
- Uvicorn runner

### Generated Base Models

```python
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
    pagination: Dict[str, Any]
```

## Auth Handlers

```prolog
generate_fastapi_auth_handlers([], Code).
```

**Generates:**
- `UserLogin` model with email/password
- `UserRegister` model with password confirmation
- `AuthResponse` model
- POST `/api/auth/login` endpoint
- POST `/api/auth/register` endpoint with validation

## Utilities

### Generate Imports

```prolog
generate_fastapi_imports([], Code).
```

### Generate Config

```prolog
generate_fastapi_config([], Code).
```

## Testing

```prolog
test_fastapi_generator.
```

Runs tests for:
- Query handler generation
- Mutation handler generation
- Infinite scroll handler
- Pydantic model generation
- Full app generation

## Related

**Parent Skill:**
- `skill_web_frameworks.md` - Web frameworks sub-master

**Sibling Skills:**
- `skill_flask_api.md` - Flask (simpler)
- `skill_express_api.md` - Express.js

**Code:**
- `src/unifyweaver/glue/fastapi_generator.pl`
