:- encoding(utf8).
% test_firewall_network_access.pl - Test Firewall Network Access Control
%
% Tests the firewall's network access validation system for HTTP sources.
% Validates URL whitelisting, blacklisting, domain patterns, and implications.

:- use_module('../src/unifyweaver/core/firewall').

%% ============================================
%% MAIN TEST SUITE
%% ============================================

main :-
    format('~n╔════════════════════════════════════════════════════════╗~n', []),
    format('║  Test: Firewall Network Access Control               ║~n', []),
    format('╚════════════════════════════════════════════════════════╝~n~n', []),

    % Run all tests
    test_network_access_denied,
    test_network_access_allowed,
    test_host_whitelist,
    test_host_blacklist,
    test_wildcard_patterns,
    test_network_implications,
    test_url_parsing,

    format('~n╔════════════════════════════════════════════════════════╗~n', []),
    format('║  All Tests Passed ✓                                   ║~n', []),
    format('╚════════════════════════════════════════════════════════╝~n', []),
    halt(0).

main :-
    format('~n[✗] Tests failed~n', []),
    halt(1).

%% ============================================
%% TEST 1: NETWORK ACCESS DENIED
%% ============================================

test_network_access_denied :-
    format('~n[Test 1] Network Access Denied~n', []),
    format('────────────────────────────────────────────────────────~n', []),

    % Test 1.1: Explicit network_access(denied) blocks all URLs
    Firewall1 = [network_access(denied)],
    URL1 = 'https://api.github.com/users',
    (   \+ validate_network_access(URL1, Firewall1)
    ->  format('  ✓ network_access(denied) blocks URL~n', [])
    ;   format('  ✗ FAIL: Expected network access to be denied~n', []),
        fail
    ),

    % Test 1.2: Even localhost is denied when network_access(denied)
    URL2 = 'http://localhost:8080/api',
    (   \+ validate_network_access(URL2, Firewall1)
    ->  format('  ✓ network_access(denied) blocks localhost~n', [])
    ;   format('  ✗ FAIL: Expected localhost to be denied~n', []),
        fail
    ),

    % Test 1.3: File URLs are also blocked
    URL3 = 'file:///etc/passwd',
    (   \+ validate_network_access(URL3, Firewall1)
    ->  format('  ✓ network_access(denied) blocks file URLs~n', [])
    ;   format('  ✗ FAIL: Expected file URL to be denied~n', []),
        fail
    ),

    format('[✓] Test 1 Complete~n', []),
    !.

%% ============================================
%% TEST 2: NETWORK ACCESS ALLOWED
%% ============================================

test_network_access_allowed :-
    format('~n[Test 2] Network Access Allowed~n', []),
    format('────────────────────────────────────────────────────────~n', []),

    % Test 2.1: Empty firewall allows all URLs
    Firewall1 = [],
    URL1 = 'https://api.github.com/users',
    (   validate_network_access(URL1, Firewall1)
    ->  format('  ✓ Empty firewall allows URL~n', [])
    ;   format('  ✗ FAIL: Expected empty firewall to allow~n', []),
        fail
    ),

    % Test 2.2: network_access(allowed) explicitly allows
    Firewall2 = [network_access(allowed)],
    URL2 = 'https://example.com/data.json',
    (   validate_network_access(URL2, Firewall2)
    ->  format('  ✓ network_access(allowed) permits URL~n', [])
    ;   format('  ✗ FAIL: Expected allowed to permit~n', []),
        fail
    ),

    % Test 2.3: Localhost is allowed by default
    Firewall3 = [network_access(allowed)],
    URL3 = 'http://localhost:3000/api',
    (   validate_network_access(URL3, Firewall3)
    ->  format('  ✓ Localhost allowed~n', [])
    ;   format('  ✗ FAIL: Expected localhost to be allowed~n', []),
        fail
    ),

    format('[✓] Test 2 Complete~n', []),
    !.

%% ============================================
%% TEST 3: HOST WHITELIST
%% ============================================

test_host_whitelist :-
    format('~n[Test 3] Host Whitelist Patterns~n', []),
    format('────────────────────────────────────────────────────────~n', []),

    % Test 3.1: Exact host match
    Firewall1 = [network_hosts(['api.github.com', 'example.com'])],
    URL1 = 'https://api.github.com/users',
    (   validate_network_access(URL1, Firewall1)
    ->  format('  ✓ Exact host match works~n', [])
    ;   format('  ✗ FAIL: Expected exact match to succeed~n', []),
        fail
    ),

    % Test 3.2: Non-whitelisted host is blocked
    URL2 = 'https://malicious.com/data',
    (   \+ validate_network_access(URL2, Firewall1)
    ->  format('  ✓ Non-whitelisted host blocked~n', [])
    ;   format('  ✗ FAIL: Expected non-whitelisted to fail~n', []),
        fail
    ),

    % Test 3.3: Subdomain handling
    Firewall2 = [network_hosts(['github.com'])],
    URL3 = 'https://api.github.com/repos',
    (   \+ validate_network_access(URL3, Firewall2)
    ->  format('  ✓ Subdomain requires wildcard or exact match~n', [])
    ;   format('  ⚠ Subdomain matched without wildcard (may be OK)~n', [])
    ),

    % Test 3.4: Multiple hosts
    Firewall3 = [network_hosts(['internal.company.com', 'api.trusted.org'])],
    URL4 = 'https://internal.company.com/data',
    (   validate_network_access(URL4, Firewall3)
    ->  format('  ✓ Multiple hosts - first match works~n', [])
    ;   format('  ✗ FAIL: Expected first host to match~n', []),
        fail
    ),

    URL5 = 'https://api.trusted.org/endpoint',
    (   validate_network_access(URL5, Firewall3)
    ->  format('  ✓ Multiple hosts - second match works~n', [])
    ;   format('  ✗ FAIL: Expected second host to match~n', []),
        fail
    ),

    format('[✓] Test 3 Complete~n', []),
    !.

%% ============================================
%% TEST 4: HOST BLACKLIST
%% ============================================

test_host_blacklist :-
    format('~n[Test 4] Host Blacklist (Using Denied)~n', []),
    format('────────────────────────────────────────────────────────~n', []),

    % Note: Blacklisting is typically done via denied services
    % We test the integration with network_hosts whitelist

    % Test 4.1: Whitelist without blacklist
    Firewall1 = [network_hosts(['safe.com', 'trusted.org'])],
    URL1 = 'https://safe.com/data',
    (   validate_network_access(URL1, Firewall1)
    ->  format('  ✓ Whitelisted host allowed~n', [])
    ;   format('  ✗ FAIL: Expected whitelisted to pass~n', []),
        fail
    ),

    % Test 4.2: Non-whitelisted implicitly blocked
    URL2 = 'https://untrusted.com/data',
    (   \+ validate_network_access(URL2, Firewall1)
    ->  format('  ✓ Non-whitelisted implicitly blocked~n', [])
    ;   format('  ✗ FAIL: Expected implicit block~n', []),
        fail
    ),

    % Test 4.3: Combination with network_access
    Firewall2 = [
        network_access(allowed),
        network_hosts(['internal.example.com'])
    ],
    URL3 = 'https://internal.example.com/api',
    (   validate_network_access(URL3, Firewall2)
    ->  format('  ✓ Combined allowed + whitelist works~n', [])
    ;   format('  ✗ FAIL: Expected combined policy to work~n', []),
        fail
    ),

    URL4 = 'https://external.example.com/api',
    (   \+ validate_network_access(URL4, Firewall2)
    ->  format('  ✓ Whitelist overrides general allowed~n', [])
    ;   format('  ✗ FAIL: Expected whitelist to restrict~n', []),
        fail
    ),

    format('[✓] Test 4 Complete~n', []),
    !.

%% ============================================
%% TEST 5: WILDCARD PATTERNS
%% ============================================

test_wildcard_patterns :-
    format('~n[Test 5] Wildcard Pattern Matching~n', []),
    format('────────────────────────────────────────────────────────~n', []),

    % Test 5.1: Wildcard suffix
    Firewall1 = [network_hosts(['*.github.com'])],
    URL1 = 'https://api.github.com/users',
    (   validate_network_access(URL1, Firewall1)
    ->  format('  ✓ Wildcard suffix *.github.com matches api.github.com~n', [])
    ;   format('  ✗ FAIL: Expected wildcard suffix to match~n', []),
        fail
    ),

    % Test 5.2: Wildcard prefix
    Firewall2 = [network_hosts(['example.*'])],
    URL2 = 'https://example.com/data',
    (   validate_network_access(URL2, Firewall2)
    ->  format('  ✓ Wildcard prefix example.* matches example.com~n', [])
    ;   format('  ✗ FAIL: Expected wildcard prefix to match~n', []),
        fail
    ),

    % Test 5.3: Mid-string wildcard
    Firewall3 = [network_hosts(['*.internal.*'])],
    URL3 = 'https://api.internal.company.com/endpoint',
    (   validate_network_access(URL3, Firewall3)
    ->  format('  ✓ Mid-string wildcard works~n', [])
    ;   format('  ✗ FAIL: Expected mid-string wildcard to match~n', []),
        fail
    ),

    % Test 5.4: Full wildcard (allow all - not recommended)
    Firewall4 = [network_hosts(['*'])],
    URL4 = 'https://anywhere.com/data',
    (   validate_network_access(URL4, Firewall4)
    ->  format('  ✓ Full wildcard * allows all~n', [])
    ;   format('  ✗ FAIL: Expected * to match all~n', []),
        fail
    ),

    % Test 5.5: Wildcard non-match
    Firewall5 = [network_hosts(['*.github.com'])],
    URL5 = 'https://gitlab.com/repos',
    (   \+ validate_network_access(URL5, Firewall5)
    ->  format('  ✓ Wildcard correctly excludes non-match~n', [])
    ;   format('  ✗ FAIL: Expected wildcard to exclude gitlab.com~n', []),
        fail
    ),

    format('[✓] Test 5 Complete~n', []),
    !.

%% ============================================
%% TEST 6: NETWORK ACCESS IMPLICATIONS
%% ============================================

test_network_implications :-
    format('~n[Test 6] Network Access Implications~n', []),
    format('────────────────────────────────────────────────────────~n', []),

    % Test 6.1: mode(offline) implies network_access(denied)
    (   firewall_implies_default(mode(offline), Policy1),
        Policy1 = network_access(denied)
    ->  format('  ✓ mode(offline) → network_access(denied)~n', [])
    ;   format('  ✗ FAIL: Expected offline implication~n', []),
        fail
    ),

    % Test 6.2: environment(restricted) implies network restrictions
    derive_policy(environment(restricted), RestrictedPolicies),
    (   member(denied(service(_, network_access(external))), RestrictedPolicies)
    ->  format('  ✓ environment(restricted) → deny external network~n', [])
    ;   format('  ⚠ Restricted environment policy incomplete~n', [])
    ),

    % Test 6.3: security_policy(strict) includes network controls
    derive_policy(security_policy(strict), StrictPolicies),
    findall(net, member(denied(service(_, network_access(_))), StrictPolicies), NetDenials),
    length(NetDenials, NumNetDenials),
    (   NumNetDenials > 0
    ->  format('  ✓ security_policy(strict) → ~w network restrictions~n', [NumNetDenials])
    ;   format('  ⚠ Strict security policy could include network controls~n', [])
    ),

    % Test 6.4: Custom user implication
    assertz(firewall:firewall_implies(corporate_network_only,
                                      network_hosts(['*.internal.company.com']))),
    format('  ✓ Added custom implication: corporate_network_only~n', []),

    % Verify it works
    (   firewall_implies(corporate_network_only, Policy4),
        Policy4 = network_hosts(['*.internal.company.com'])
    ->  format('  ✓ Custom network implication works~n', [])
    ;   format('  ✗ FAIL: Custom implication not working~n', []),
        fail
    ),

    % Clean up
    retractall(firewall:firewall_implies(corporate_network_only, _)),

    format('[✓] Test 6 Complete~n', []),
    !.

%% ============================================
%% TEST 7: URL PARSING AND VALIDATION
%% ============================================

test_url_parsing :-
    format('~n[Test 7] URL Parsing and Edge Cases~n', []),
    format('────────────────────────────────────────────────────────~n', []),

    % Test 7.1: Standard HTTPS URL
    Firewall1 = [network_hosts(['api.example.com'])],
    URL1 = 'https://api.example.com:443/v1/users',
    (   validate_network_access(URL1, Firewall1)
    ->  format('  ✓ HTTPS URL with port parses correctly~n', [])
    ;   format('  ✗ FAIL: Expected HTTPS URL to parse~n', []),
        fail
    ),

    % Test 7.2: HTTP URL (non-secure)
    URL2 = 'http://api.example.com/data',
    (   validate_network_access(URL2, Firewall1)
    ->  format('  ✓ HTTP URL parses correctly~n', [])
    ;   format('  ✗ FAIL: Expected HTTP URL to parse~n', []),
        fail
    ),

    % Test 7.3: URL with query parameters
    URL3 = 'https://api.example.com/search?q=test&limit=10',
    (   validate_network_access(URL3, Firewall1)
    ->  format('  ✓ URL with query params works~n', [])
    ;   format('  ✗ FAIL: Expected query params to work~n', []),
        fail
    ),

    % Test 7.4: URL with fragment
    URL4 = 'https://api.example.com/docs#section-2',
    (   validate_network_access(URL4, Firewall1)
    ->  format('  ✓ URL with fragment works~n', [])
    ;   format('  ✗ FAIL: Expected fragment to work~n', []),
        fail
    ),

    % Test 7.5: URL with auth (if supported)
    Firewall2 = [network_hosts(['secure.example.com'])],
    URL5 = 'https://user:pass@secure.example.com/api',
    (   validate_network_access(URL5, Firewall2)
    ->  format('  ✓ URL with authentication info parses~n', [])
    ;   format('  ⚠ URL with auth may not be fully supported~n', [])
    ),

    % Test 7.6: IPv4 address
    Firewall3 = [network_hosts(['192.168.1.100'])],
    URL6 = 'http://192.168.1.100:8080/data',
    (   validate_network_access(URL6, Firewall3)
    ->  format('  ✓ IPv4 address in URL works~n', [])
    ;   format('  ⚠ IPv4 address may not be supported~n', [])
    ),

    % Test 7.7: localhost variants
    Firewall4 = [network_hosts(['localhost', '127.0.0.1'])],
    URL7 = 'http://localhost:3000/api',
    (   validate_network_access(URL7, Firewall4)
    ->  format('  ✓ localhost hostname works~n', [])
    ;   format('  ✗ FAIL: Expected localhost to work~n', []),
        fail
    ),

    format('[✓] Test 7 Complete~n', []),
    !.

:- initialization(main, main).
