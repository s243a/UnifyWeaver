# Playbook: HTTP Data Source

## Audience
This playbook is a high-level guide for coding agents (Gemini CLI, Claude Code, etc.). Agents orchestrate UnifyWeaver to generate bash functions that fetch data from HTTP/REST APIs with caching support.


## Finding Examples

There are two ways to find the correct example record for this task:

### Method 1: Manual Extraction
Search the documentation using grep:
```bash
grep -r "http_source" playbooks/examples_library/
```

### Method 2: Semantic Search (Recommended)
Use the LDA-based semantic search skill to find relevant examples by intent:
```bash
./unifyweaver search "how to use http source"


## Workflow Overview
Use UnifyWeaver's http_source plugin to:
1. Define HTTP endpoints as Prolog predicates
2. Compile predicates to bash functions using curl or wget
3. Automatically handle caching, timeouts, and headers
4. Generate both GET and POST requests

## Agent Inputs
Reference the following artifacts:
1. **Executable Records** - `playbooks/examples_library/http_source_examples.md`
2. **Environment Setup Skill** - `skills/skill_unifyweaver_environment.md`
3. **Extraction Skill** - `skills/skill_extract_records.md`

## Execution Guidance

### Step 1: Navigate to project root
```bash
cd /root/UnifyWeaver
```

### Step 2: Extract the basic HTTP source demo
```bash
perl scripts/utils/extract_records.pl \
  -f content \
  -q "unifyweaver.execution.http_source_basic" \
  playbooks/examples_library/http_source_examples.md \
  > tmp/run_http_basic.sh
```

### Step 3: Make it executable and run
```bash
chmod +x tmp/run_http_basic.sh
bash tmp/run_http_basic.sh
```

**Expected Output**:
```
=== HTTP Source Demo: Basic Usage ===

Running Prolog to generate HTTP source scripts...

=== HTTP Source Configuration ===

Generating bash for JSON API source...
  Compiling HTTP source: fetch_users/1
Generated: tmp/http_demo/fetch_users.sh

Generating bash for POST API source...
  Compiling HTTP source: create_post/2
Generated: tmp/http_demo/create_post.sh
...
SUCCESS: fetch_users function generated correctly

Success: HTTP source demo complete
```

### Step 4: View module info (optional)
```bash
perl scripts/utils/extract_records.pl \
  -f content \
  -q "unifyweaver.execution.http_source_info" \
  playbooks/examples_library/http_source_examples.md \
  > tmp/run_http_info.sh
chmod +x tmp/run_http_info.sh
bash tmp/run_http_info.sh
```

### Step 5: Test caching features (optional)
```bash
perl scripts/utils/extract_records.pl \
  -f content \
  -q "unifyweaver.execution.http_source_cached" \
  playbooks/examples_library/http_source_examples.md \
  > tmp/run_http_cached.sh
chmod +x tmp/run_http_cached.sh
bash tmp/run_http_cached.sh
```

## What This Playbook Demonstrates

1. **http_source plugin** (`src/unifyweaver/sources/http_source.pl`):
   - `compile_source/4` - Compile HTTP source to bash
   - `validate_config/1` - Validate configuration options
   - `source_info/1` - Plugin metadata

2. **Configuration options**:
   - `url(URL)` - HTTP endpoint (required)
   - `method(get|post|put|delete|head)` - HTTP method
   - `timeout(Seconds)` - Request timeout
   - `tool(curl|wget)` - HTTP client tool
   - `cache_file(Path)` - Cache location
   - `cache_duration(Seconds)` - Cache validity
   - `headers([...])` - Custom HTTP headers
   - `post_data(Data)` - Request body for POST/PUT

3. **Generated bash functions**:
   - `predicate()` - Main function
   - `predicate_stream()` - Streaming alias
   - `predicate_raw()` - Raw output mode
   - `predicate_cache_clear()` - Clear cache (if caching enabled)
   - `predicate_cache_info()` - Show cache status (if caching enabled)

## Example Configurations

### Simple GET request:
```prolog
Config = [
    url('https://api.example.com/users'),
    method(get)
].
```

### POST with JSON body:
```prolog
Config = [
    url('https://api.example.com/users'),
    method(post),
    post_data('{"name": "test"}'),
    headers(['Content-Type: application/json'])
].
```

### Cached API with custom timeout:
```prolog
Config = [
    url('https://api.example.com/data'),
    timeout(60),
    cache_duration(3600),  % 1 hour cache
    cache_file('/tmp/api_cache')
].
```

### Using wget instead of curl:
```prolog
Config = [
    url('https://api.example.com/data'),
    tool(wget),
    user_agent('MyApp/1.0')
].
```

## Common Mistakes to Avoid

- **DO NOT** run extracted scripts with `swipl` - they are bash scripts
- **DO** ensure curl or wget is installed before testing
- **DO** use proper quoting for URLs with special characters
- **DO** set appropriate cache_duration for frequently changing APIs

## Expected Outcome
- Generated bash scripts that fetch from HTTP endpoints
- Scripts include proper error handling and timeout support
- Cached scripts avoid redundant network requests

## Citations
[1] playbooks/examples_library/http_source_examples.md
[2] src/unifyweaver/sources/http_source.pl
[3] src/unifyweaver/core/dynamic_source_compiler.pl
[4] skills/skill_unifyweaver_environment.md
