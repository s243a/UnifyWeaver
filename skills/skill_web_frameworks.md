# Skill: Web Frameworks (Sub-Master)

Generate REST API servers from Prolog specifications using Flask, FastAPI, or Express.

## When to Use

- User asks "how do I create a REST API?"
- User needs to expose Prolog predicates as HTTP endpoints
- User wants CRUD operations for data patterns
- User needs pagination, authentication, or CORS support

## Skill Hierarchy

```
skill_server_tools.md (parent)
└── skill_web_frameworks.md (this file)
    ├── skill_flask_api.md - Flask routes and blueprints
    ├── skill_fastapi.md - FastAPI with Pydantic models
    └── skill_express_api.md - Express.js routers
```

## Quick Start

### Flask (Python - Simple)

```prolog
:- use_module('src/unifyweaver/glue/flask_generator').

% Generate complete Flask app
generate_flask_app([fetch_items, create_item], [app_name('ItemAPI')], Code).

% Generate individual handler
generate_flask_query_handler(fetch_items, [endpoint('/api/items')], [], Handler).
```

### FastAPI (Python - Async + Validation)

```prolog
:- use_module('src/unifyweaver/glue/fastapi_generator').

% Generate complete FastAPI app
generate_fastapi_app([list_products], [app_name('ProductAPI')], Code).

% Generate Pydantic model for validation
generate_pydantic_model(product, [
    field(name, string),
    field(price, number),
    field(quantity, optional(integer))
], Model).
```

### Express (Node.js)

```prolog
:- use_module('src/unifyweaver/glue/express_generator').

% Generate Express router
generate_express_router(api, [endpoints([math_endpoints, numpy_endpoints])], Code).

% Generate complete Express app
generate_express_app(my_api, [port(3001)], AppCode).
```

## Framework Comparison

| Feature | Flask | FastAPI | Express |
|---------|-------|---------|---------|
| Language | Python | Python | Node.js |
| Async | Optional | Native | Native |
| Validation | Manual | Pydantic | Manual |
| OpenAPI | Flask-RESTX | Automatic | Manual |
| CORS | flask-cors | Middleware | cors package |
| Best For | Simple APIs | Modern Python | JS ecosystem |

## Common Patterns

### Query Handler (Pagination)

All frameworks generate handlers with pagination support:

```python
# Flask/FastAPI style
@app.get("/api/items")
def list_items(page: int = 1, limit: int = 20):
    offset = (page - 1) * limit
    return {"data": items, "pagination": {...}}
```

### Mutation Handler (CRUD)

```python
# POST/PUT/DELETE handlers
@app.post("/api/items")
def create_item(data: ItemInput):
    return {"success": True, "data": result}
```

### Infinite Scroll (Cursor-based)

```python
@app.get("/api/feed")
def list_feed(cursor: str = None, limit: int = 20):
    return {"data": items, "nextCursor": next_cursor, "hasMore": True}
```

## Child Skills

- `skill_flask_api.md` - Flask routes, blueprints, auth handlers
- `skill_fastapi.md` - FastAPI, Pydantic models, async handlers
- `skill_express_api.md` - Express routers, security middleware

## Related

**Parent Skill:**
- `skill_server_tools.md` - Backend services master

**Sibling Skills:**
- `skill_ipc.md` - Inter-process communication
- `skill_infrastructure.md` - Deployment and ops

**Code:**
- `src/unifyweaver/glue/flask_generator.pl`
- `src/unifyweaver/glue/fastapi_generator.pl`
- `src/unifyweaver/glue/express_generator.pl`
