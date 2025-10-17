:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% http_pipeline_demo.pl - HTTP source pipeline demonstration
% Shows API fetching with caching and header management

:- initialization(main, main).

%% Load infrastructure
:- use_module('src/unifyweaver/sources').
:- load_files('src/unifyweaver/sources/http_source', [imports([])]).
:- load_files('src/unifyweaver/core/dynamic_source_compiler', [imports([])]).

%% Define HTTP sources

% HTTP source 1: Fetch from JSONPlaceholder (free test API)
:- source(http, api_posts, [
    url('https://jsonplaceholder.typicode.com/posts/1'),
    headers(['User-Agent: UnifyWeaver/0.0.2']),
    cache_duration(3600),
    cache_file('output/cache/api_posts.json')
]).

% HTTP source 2: Fetch user data
:- source(http, api_user, [
    url('https://jsonplaceholder.typicode.com/users/1'),
    headers(['User-Agent: UnifyWeaver/0.0.2']),
    cache_duration(3600),
    cache_file('output/cache/api_user.json')
]).

%% ============================================
%% MAIN PIPELINE
%% ============================================

main :-
    format('🌐 HTTP Source Pipeline Demo~n', []),
    format('===================================~n~n', []),
    
    % Step 1: Setup
    format('📝 Step 1: Setting up cache directory...~n', []),
    setup_cache,
    
    % Step 2: Compile HTTP sources
    format('~n🔨 Step 2: Compiling HTTP sources...~n', []),
    compile_http_sources,
    
    % Step 3: Execute pipeline
    format('~n🚀 Step 3: Fetching data from API...~n', []),
    execute_http_pipeline,
    
    format('~n✅ HTTP Pipeline Complete!~n~n', []),
    format('Generated files:~n', []),
    format('  - output/api_posts.sh (API fetch script)~n', []),
    format('  - output/api_user.sh (User fetch script)~n', []),
    format('  - output/cache/*.json (cached responses)~n', []),
    !.

main :-
    format('~n❌ Pipeline failed!~n', []),
    halt(1).

%% ============================================
%% PIPELINE STEPS
%% ============================================

setup_cache :-
    (exists_directory('output') -> true ; make_directory('output')),
    (exists_directory('output/cache') -> true ; make_directory('output/cache')),
    format('   ✓ Created cache directory~n', []).

compile_http_sources :-
    % Compile api_posts
    format('   Compiling api_posts/1...~n', []),
    dynamic_source_compiler:compile_dynamic_source(api_posts/1, [], BashCode1),
    open('output/api_posts.sh', write, S1),
    write(S1, BashCode1),
    close(S1),
    shell('chmod +x output/api_posts.sh', _),
    
    % Compile api_user
    format('   Compiling api_user/1...~n', []),
    dynamic_source_compiler:compile_dynamic_source(api_user/1, [], BashCode2),
    open('output/api_user.sh', write, S2),
    write(S2, BashCode2),
    close(S2),
    shell('chmod +x output/api_user.sh', _),
    
    format('   ✓ Generated HTTP source scripts~n', []).

execute_http_pipeline :-
    % Fetch post data
    format('   Fetching post from API...~n', []),
    shell('bash output/api_posts.sh > output/post_data.json 2>/dev/null', Status1),
    (Status1 = 0 -> 
        format('   ✓ Post data fetched~n', []),
        format('~n   Post Content:~n', []),
        shell('cat output/post_data.json | head -5', _)
    ; 
        format('   ℹ API unavailable or network error (this is OK for offline testing)~n', [])
    ),
    
    % Fetch user data  
    format('~n   Fetching user from API...~n', []),
    shell('bash output/api_user.sh > output/user_data.json 2>/dev/null', Status2),
    (Status2 = 0 -> 
        format('   ✓ User data fetched~n', []),
        format('~n   User Info:~n', []),
        shell('cat output/user_data.json | head -5', _)
    ; 
        format('   ℹ API unavailable or network error (this is OK for offline testing)~n', [])
    ),
    
    % Show cache info
    format('~n   Cache Status:~n', []),
    shell('ls -lh output/cache/ 2>/dev/null | grep -v "^total" || echo "   No cached files yet"', _).

%% ============================================
%% USAGE
%% ============================================

/*
To run this HTTP pipeline demo:

cd scripts/testing/test_env5
swipl -g main -t halt examples/http_pipeline_demo.pl

This demonstrates:
✅ HTTP GET requests to REST APIs
✅ Response caching (1 hour duration)
✅ Custom headers (User-Agent)
✅ Multiple API endpoints
✅ Cache file management

The pipeline:
1. Sets up cache directory
2. Compiles HTTP sources to bash
3. Fetches post data from JSONPlaceholder API
4. Fetches user data from same API
5. Displays fetched content
6. Shows cache status

Note: Requires curl or wget and internet connection.
If offline, the demo explains caching would work when online.
*/
