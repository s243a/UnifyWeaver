# Skill: Express API

Generate Express.js routers and apps with declarative endpoint definitions and security middleware.

## When to Use

- User asks "how do I create an Express API?"
- User needs Node.js REST endpoints
- User wants to expose Python functions via HTTP
- User needs TypeScript types for API endpoints

## Quick Start

```prolog
:- use_module('src/unifyweaver/glue/express_generator').

% Generate Express router
generate_express_router(my_api, [], Code).

% Generate complete Express app
generate_express_app(my_api, [port(3001)], AppCode).
```

## Endpoint Declaration

### Define Endpoints

```prolog
api_endpoint(Path, Config).
```

**Config Options:**
- `method(Method)` - HTTP method (get, post, put, delete)
- `module(Module)` - Python module to call
- `function(Func)` - Python function to call
- `attr(Attr)` - Python attribute to access
- `input_schema(Schema)` - Input field types
- `output_schema(Type)` - Output type
- `description(Text)` - Endpoint description

**Example:**
```prolog
api_endpoint('/numpy/mean', [
    method(post),
    module(numpy),
    function(mean),
    input_schema([data: array(number)]),
    output_schema(number),
    description("Calculate arithmetic mean")
]).

api_endpoint('/math/pi', [
    method(get),
    module(math),
    attr(pi),
    output_schema(number),
    description("Get pi constant")
]).
```

### Endpoint Groups

```prolog
api_endpoint_group(math_endpoints, [
    '/math/sqrt',
    '/math/pow',
    '/math/pi',
    '/math/e'
]).
```

### Pre-defined Endpoints

| Group | Endpoints |
|-------|-----------|
| `math_endpoints` | sqrt, pow, pi, e |
| `numpy_endpoints` | mean, std, sum, min, max |
| `statistics_endpoints` | mean, median, stdev |

## Router Generation

### Basic Router

```prolog
generate_express_router(Name, Options, Code).
```

**Options:**
- `endpoints(List)` - Endpoint paths or group names
- `bridge_import(Path)` - RPyC bridge import path

**Output:**
```typescript
import { Router, Request, Response } from 'express';
import { bridge } from './rpyc_bridge';
import { validateCall, validateAttr } from './validator';

export const myApiRouter = Router();

// Calculate arithmetic mean
myApiRouter.post('/numpy/mean', async (req: Request, res: Response) => {
  try {
    const { data } = req.body;
    const validation = validateCall('numpy', 'mean', data);
    if (!validation.valid) {
      return res.status(400).json({ success: false, error: validation.error });
    }
    const result = await bridge.call('numpy', 'mean', data);
    res.json({ success: true, result });
  } catch (error) {
    res.status(500).json({ success: false, error: error.message });
  }
});
```

### Secure Router

```prolog
generate_secure_router(Name, Code).
```

Generates router with security middleware:
- Rate limiting
- Request timeout
- Input validation

```typescript
import { rateLimiter, timeoutMiddleware, securityMiddleware } from './security_middleware';

export const myApiRouter = Router();
myApiRouter.use(securityMiddleware);
```

## App Generation

### Complete App

```prolog
generate_express_app(Name, Options, Code).
```

**Options:**
- `port(Port)` - Server port (default: 3001)
- `endpoints(List)` - Endpoints to include

**Generates:**
- Express imports
- CORS middleware
- JSON body parser
- Health check endpoint
- Router mounting
- Error handler
- Server startup

## TypeScript Types

### Generate Types

```prolog
generate_endpoint_types(Name, Code).
```

**Output:**
```typescript
// /numpy/mean
export interface NumpyMeanInput {
  data: number[];
}

export type NumpyMeanOutput = number;

// Generic API response wrapper
export interface ApiResponse<T> {
  success: boolean;
  result?: T;
  error?: string;
}
```

## Endpoint Management

### Dynamic Declaration

```prolog
% Add endpoint
declare_endpoint('/custom/endpoint', [
    method(post),
    module(custom),
    function(process),
    description("Custom processing")
]).

% Add group
declare_endpoint_group(custom_endpoints, ['/custom/endpoint']).

% Clear all
clear_endpoints.
```

### Query Endpoints

```prolog
% Get all endpoints
all_endpoints(Endpoints).

% Get endpoints for module
endpoints_for_module(numpy, NumpyEndpoints).
```

## Integration with RPyC Security

The Express generator integrates with `rpyc_security.pl`:

```prolog
% Load security module
:- use_module('./rpyc_security').

% Validation in handlers uses:
% - validateCall(module, func, args)
% - validateAttr(module, attr)
```

## Testing

```prolog
test_express_generator.
```

Runs tests for:
- Endpoint queries
- Router code generation
- App code generation
- Types code generation

## Related

**Parent Skill:**
- `skill_web_frameworks.md` - Web frameworks sub-master

**Sibling Skills:**
- `skill_flask_api.md` - Flask (Python)
- `skill_fastapi.md` - FastAPI (Python)
- `skill_frontend_security.md` - RPyC security rules

**Code:**
- `src/unifyweaver/glue/express_generator.pl`
- `src/unifyweaver/glue/rpyc_security.pl`
