:- encoding(utf8).
% test_network_firewall.pl - Test network access control in firewall

:- use_module('../src/unifyweaver/core/firewall_v2').

main :-
    format('~n╔════════════════════════════════════════════════════════╗~n', []),
    format('║  Network Access Control Tests                          ║~n', []),
    format('╚════════════════════════════════════════════════════════╝~n~n', []),

    test_url_extraction,
    test_domain_matching,
    test_deny_all_policy,
    test_whitelist_policy,
    test_denied_domains,
    test_url_patterns,

    format('~n╔════════════════════════════════════════════════════════╗~n', []),
    format('║  All Network Tests Passed ✓                            ║~n', []),
    format('╚════════════════════════════════════════════════════════╝~n', []),
    halt(0).

main :-
    format('~n[✗] Network tests failed~n', []),
    halt(1).

%% Test 1: URL extraction
test_url_extraction :-
    format('~n[Test 1] URL domain extraction~n', []),
    format('────────────────────────────────────────────────────────~n', []),

    firewall_v2:extract_domain('https://api.example.com/data', Domain1),
    format('  https://api.example.com/data → ~w~n', [Domain1]),
    Domain1 = 'api.example.com',

    firewall_v2:extract_domain('http://example.com', Domain2),
    format('  http://example.com → ~w~n', [Domain2]),
    Domain2 = 'example.com',

    firewall_v2:extract_domain('example.com/path', Domain3),
    format('  example.com/path → ~w~n', [Domain3]),
    Domain3 = 'example.com',

    format('[✓] URL extraction works~n', []),
    !.

%% Test 2: Domain pattern matching
test_domain_matching :-
    format('~n[Test 2] Domain pattern matching~n', []),
    format('────────────────────────────────────────────────────────~n', []),

    % Exact match
    (   firewall_v2:domain_matches_pattern('example.com', 'example.com')
    ->  format('  ✓ Exact match: example.com = example.com~n', [])
    ;   format('  ✗ Exact match failed~n', []), fail
    ),

    % Subdomain match
    (   firewall_v2:domain_matches_pattern('api.example.com', 'example.com')
    ->  format('  ✓ Subdomain match: api.example.com matches example.com~n', [])
    ;   format('  ✗ Subdomain match failed~n', []), fail
    ),

    % Wildcard match
    (   firewall_v2:domain_matches_pattern('api.example.com', '*.example.com')
    ->  format('  ✓ Wildcard match: api.example.com matches *.example.com~n', [])
    ;   format('  ✗ Wildcard match failed~n', []), fail
    ),

    format('[✓] Domain matching works~n', []),
    !.

%% Test 3: Deny all policy
test_deny_all_policy :-
    format('~n[Test 3] Deny all network access policy~n', []),
    format('────────────────────────────────────────────────────────~n', []),

    % Clean state
    retractall(network_access_policy(_)),

    % Load no_network policy
    load_firewall_policy(no_network),

    % Check URL access
    check_url_access('https://example.com', Result),
    format('  Access to https://example.com: ~w~n', [Result]),
    Result = deny(_),

    % Clean up
    retractall(network_access_policy(_)),
    retractall(denied_service(_, _)),

    format('[✓] Deny all policy works~n', []),
    !.

%% Test 4: Whitelist policy
test_whitelist_policy :-
    format('~n[Test 4] Domain whitelist policy~n', []),
    format('────────────────────────────────────────────────────────~n', []),

    % Clean state
    retractall(network_access_policy(_)),
    retractall(allowed_domain(_)),

    % Load whitelist policy
    load_firewall_policy(whitelist_domains(['example.com', 'trusted.org'])),

    % Check allowed domain
    check_url_access('https://api.example.com/data', Result1),
    format('  Access to api.example.com: ~w~n', [Result1]),
    Result1 = allow,

    % Check denied domain
    check_url_access('https://untrusted.com/data', Result2),
    format('  Access to untrusted.com: ~w~n', [Result2]),
    Result2 = deny(_),

    % Clean up
    retractall(network_access_policy(_)),
    retractall(allowed_domain(_)),

    format('[✓] Whitelist policy works~n', []),
    !.

%% Test 5: Denied domains
test_denied_domains :-
    format('~n[Test 5] Denied domains~n', []),
    format('────────────────────────────────────────────────────────~n', []),

    % Clean state
    retractall(denied_domain(_)),
    retractall(network_access_policy(_)),
    retractall(firewall_mode(_)),

    % Set permissive mode
    set_firewall_mode(permissive),

    % Deny specific domain
    assertz(denied_domain('malicious.com')),

    % Check denied domain
    check_url_access('https://malicious.com/bad', Result1),
    format('  Access to malicious.com: ~w~n', [Result1]),
    Result1 = deny(_),

    % Check allowed domain
    check_url_access('https://good.com/api', Result2),
    format('  Access to good.com: ~w~n', [Result2]),
    Result2 = allow,

    % Clean up
    retractall(denied_domain(_)),
    retractall(firewall_mode(_)),

    format('[✓] Denied domains work~n', []),
    !.

%% Test 6: URL patterns
test_url_patterns :-
    format('~n[Test 6] URL pattern matching~n', []),
    format('────────────────────────────────────────────────────────~n', []),

    % Clean state
    retractall(denied_url_pattern(_)),
    retractall(network_access_policy(_)),
    retractall(firewall_mode(_)),

    % Set permissive mode
    set_firewall_mode(permissive),

    % Deny URLs containing 'admin'
    assertz(denied_url_pattern('admin')),

    % Check denied pattern
    check_url_access('https://example.com/admin/users', Result1),
    format('  Access to /admin/users: ~w~n', [Result1]),
    Result1 = deny(_),

    % Check allowed URL
    check_url_access('https://example.com/api/users', Result2),
    format('  Access to /api/users: ~w~n', [Result2]),
    Result2 = allow,

    % Clean up
    retractall(denied_url_pattern(_)),
    retractall(firewall_mode(_)),

    format('[✓] URL patterns work~n', []),
    !.

:- initialization(main, main).
