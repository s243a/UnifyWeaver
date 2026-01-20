# Skill: Flask API

Generate Flask REST API routes and handlers from Prolog specifications.

## When to Use

- User asks "how do I create a Flask API?"
- User needs simple Python REST endpoints
- User wants blueprint-organized APIs
- User needs Flask auth handlers

## Quick Start

```prolog
:- use_module('src/unifyweaver/glue/flask_generator').

% Generate complete Flask app
generate_flask_app([fetch_tasks, create_task], [app_name('TaskAPI')], Code).
```

## Handler Generation

### Query Handler (GET with Pagination)

```prolog
generate_flask_query_handler(fetch_items, [endpoint('/api/items')], [], Code).
```

**Output:**
```python
@app.route('/api/items', methods=['GET'])
def fetch_items():
    """Fetch fetch_items with pagination."""
    try:
        page = request.args.get('page', 1, type=int)
        limit = request.args.get('limit', 20, type=int)
        page = max(1, page)
        limit = min(max(1, limit), 100)
        offset = (page - 1) * limit
        # TODO: Implement data fetching logic
        data = []
        total = 0
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
```

### Mutation Handler (POST/PUT/DELETE)

```prolog
% POST handler
generate_flask_mutation_handler(create_item, [endpoint('/api/items'), method('POST')], [], Code).

% DELETE handler
generate_flask_mutation_handler(delete_item, [endpoint('/api/items/<id>'), method('DELETE')], [], Code).
```

### Infinite Scroll Handler

```prolog
generate_flask_infinite_handler(load_feed, [endpoint('/api/feed'), page_param('cursor')], [], Code).
```

**Output:**
```python
@app.route('/api/feed', methods=['GET'])
def load_feed():
    """Fetch load_feed with cursor-based pagination (infinite scroll)."""
    cursor = request.args.get('cursor', None)
    limit = request.args.get('limit', 20, type=int)
    # Returns nextCursor and hasMore
```

## App Generation

### Complete App

```prolog
generate_flask_app(Patterns, Options, Code).
```

**Options:**
- `app_name(Name)` - Application name (default: 'FlaskAPI')

**Generates:**
- Flask imports with CORS
- App configuration
- All route handlers
- Health check endpoint
- Error handlers (404, 500)
- Main entry point

### Blueprint Organization

```prolog
generate_flask_blueprint(tasks, [fetch_tasks, create_task, delete_task], [], Code).
```

**Output:**
```python
from flask import Blueprint, jsonify, request

tasks_bp = Blueprint('tasks', __name__, url_prefix='/api/tasks')

@tasks_bp.route('/fetch_tasks', methods=['GET'])
def fetch_tasks():
    # ...
```

## Auth Handlers

```prolog
generate_flask_auth_handlers([], Code).
```

**Generates:**
- `/api/auth/login` - Email/password login
- `/api/auth/register` - User registration with validation

## Utilities

### Generate Imports

```prolog
generate_flask_imports([], Code).
```

**Output:**
```python
from flask import Flask, jsonify, request
from flask_cors import CORS
import logging
```

### Generate Config

```prolog
generate_flask_config([app_name('MyAPI')], Code).
```

## Testing

```prolog
test_flask_generator.
```

Runs tests for:
- Query handler generation
- Mutation handler generation
- Delete handler generation
- Infinite scroll handler
- Full app generation

## Related

**Parent Skill:**
- `skill_web_frameworks.md` - Web frameworks sub-master

**Sibling Skills:**
- `skill_fastapi.md` - FastAPI with Pydantic
- `skill_express_api.md` - Express.js

**Code:**
- `src/unifyweaver/glue/flask_generator.pl`
