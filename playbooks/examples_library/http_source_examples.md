# HTTP Source Examples

This file contains executable records for the HTTP data source playbook.

## Basic HTTP Source

:::unifyweaver.execution.http_source_basic
```bash
#!/bin/bash
# HTTP Source Demo - Basic Usage
# Demonstrates fetching data from HTTP endpoints

set -euo pipefail
cd /root/UnifyWeaver

echo "=== HTTP Source Demo: Basic Usage ==="

# Create test directory
mkdir -p tmp/http_demo

# Create the Prolog script that uses http_source
cat > tmp/http_demo/generate_http_source.pl << 'PROLOG'
:- use_module('src/unifyweaver/core/dynamic_source_compiler').
:- use_module('src/unifyweaver/sources/http_source').

main :-
    format("~n=== HTTP Source Configuration ===~n"),

    % Example 1: Simple JSON API source
    format("~nGenerating bash for JSON API source...~n"),
    Config1 = [
        url('https://jsonplaceholder.typicode.com/users'),
        method(get),
        cache_duration(300),  % Cache for 5 minutes
        headers(['Accept: application/json'])
    ],

    (   http_source:compile_source(fetch_users/1, Config1, [], BashCode1)
    ->  open('tmp/http_demo/fetch_users.sh', write, S1),
        write(S1, BashCode1),
        close(S1),
        format("Generated: tmp/http_demo/fetch_users.sh~n")
    ;   format("Failed to compile JSON API source~n")
    ),

    % Example 2: API with POST data
    format("~nGenerating bash for POST API source...~n"),
    Config2 = [
        url('https://jsonplaceholder.typicode.com/posts'),
        method(post),
        post_data('{"title": "test", "body": "content", "userId": 1}'),
        headers(['Content-Type: application/json', 'Accept: application/json']),
        cache_duration(0)  % No caching for POST
    ],

    (   http_source:compile_source(create_post/2, Config2, [], BashCode2)
    ->  open('tmp/http_demo/create_post.sh', write, S2),
        write(S2, BashCode2),
        close(S2),
        format("Generated: tmp/http_demo/create_post.sh~n")
    ;   format("Failed to compile POST API source~n")
    ),

    % Example 3: Wget-based source
    format("~nGenerating bash for wget-based source...~n"),
    Config3 = [
        url('https://api.github.com'),
        method(get),
        tool(wget),
        timeout(10),
        cache_duration(600)  % Cache for 10 minutes
    ],

    (   http_source:compile_source(github_api/1, Config3, [], BashCode3)
    ->  open('tmp/http_demo/github_api.sh', write, S3),
        write(S3, BashCode3),
        close(S3),
        format("Generated: tmp/http_demo/github_api.sh~n")
    ;   format("Failed to compile wget source~n")
    ),

    format("~n=== All sources generated ===~n"),
    halt(0).

:- initialization(main, main).
PROLOG

echo ""
echo "Running Prolog to generate HTTP source scripts..."
swipl tmp/http_demo/generate_http_source.pl

echo ""
echo "=== Generated Scripts ==="

echo ""
echo "--- fetch_users.sh (first 50 lines) ---"
head -50 tmp/http_demo/fetch_users.sh 2>/dev/null || echo "File not found"

echo ""
echo "--- create_post.sh (first 50 lines) ---"
head -50 tmp/http_demo/create_post.sh 2>/dev/null || echo "File not found"

echo ""
echo "=== Testing fetch_users (if curl available) ==="
if command -v curl &> /dev/null; then
    chmod +x tmp/http_demo/fetch_users.sh
    # Just test the function definition was generated
    if grep -q "fetch_users()" tmp/http_demo/fetch_users.sh; then
        echo "SUCCESS: fetch_users function generated correctly"
    else
        echo "WARNING: fetch_users function not found in output"
    fi
else
    echo "curl not found - skipping HTTP test"
fi

echo ""
echo "Success: HTTP source demo complete"
```
:::

## HTTP Source with Caching

:::unifyweaver.execution.http_source_cached
```bash
#!/bin/bash
# HTTP Source Demo - Caching Features
# Demonstrates HTTP caching and cache management

set -euo pipefail
cd /root/UnifyWeaver

echo "=== HTTP Source Demo: Caching Features ==="

mkdir -p tmp/http_demo

# Create Prolog script demonstrating caching
cat > tmp/http_demo/demo_caching.pl << 'PROLOG'
:- use_module('src/unifyweaver/core/dynamic_source_compiler').
:- use_module('src/unifyweaver/sources/http_source').

main :-
    format("~n=== Generating Cached HTTP Source ===~n"),

    Config = [
        url('https://httpbin.org/json'),
        method(get),
        cache_file('/tmp/httpbin_cache'),
        cache_duration(60),  % Cache for 1 minute
        headers(['Accept: application/json']),
        timeout(15)
    ],

    (   http_source:compile_source(httpbin_data/1, Config, [], BashCode)
    ->  open('tmp/http_demo/httpbin_cached.sh', write, S),
        write(S, BashCode),
        close(S),
        format("Generated: tmp/http_demo/httpbin_cached.sh~n"),
        format("~nThe script includes:~n"),
        format("  - httpbin_data() - main function (with caching)~n"),
        format("  - httpbin_data_cache_clear() - clear cache~n"),
        format("  - httpbin_data_cache_info() - show cache status~n")
    ;   format("Failed to compile source~n")
    ),
    halt(0).

:- initialization(main, main).
PROLOG

echo ""
echo "Running Prolog to generate cached HTTP source..."
swipl tmp/http_demo/demo_caching.pl

echo ""
echo "=== Cached Source Script ==="
head -60 tmp/http_demo/httpbin_cached.sh

echo ""
echo "Success: HTTP caching demo complete"
```
:::

## HTTP Source Module Info

:::unifyweaver.execution.http_source_info
```bash
#!/bin/bash
# HTTP Source Demo - Module Information
# Shows HTTP source plugin capabilities

set -euo pipefail
cd /root/UnifyWeaver

echo "=== HTTP Source Plugin Information ==="

# Show source_info
swipl -g "
    use_module('src/unifyweaver/sources/http_source'),
    http_source:source_info(Info),
    format('~nHTTP Source Plugin Info:~n'),
    format('  ~w~n', [Info]),
    halt.
" 2>&1

echo ""
echo "=== Configuration Options ==="
echo ""
echo "Required:"
echo "  url(URL)           - HTTP endpoint URL"
echo ""
echo "Optional:"
echo "  method(get|post|put|delete|head) - HTTP method (default: get)"
echo "  timeout(Seconds)   - Request timeout (default: 30)"
echo "  tool(curl|wget)    - HTTP client tool (default: curl)"
echo "  cache_file(Path)   - Cache file location"
echo "  cache_duration(Sec) - Cache validity in seconds (default: 300)"
echo "  headers([...])     - List of HTTP headers"
echo "  post_data(Data)    - POST/PUT body data"
echo "  user_agent(UA)     - Custom User-Agent string"

echo ""
echo "=== Example Configurations ==="
echo ""
echo "Simple GET:"
echo '  [url("https://api.example.com/data")]'
echo ""
echo "POST with JSON:"
echo '  [url("https://api.example.com/create"),'
echo '   method(post),'
echo '   post_data("{\"key\": \"value\"}"),'
echo '   headers(["Content-Type: application/json"])]'
echo ""
echo "Cached with wget:"
echo '  [url("https://api.example.com/data"),'
echo '   tool(wget),'
echo '   cache_duration(600)]'

echo ""
echo "Success: HTTP source info displayed"
```
:::
