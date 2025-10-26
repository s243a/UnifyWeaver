:- encoding(utf8).
% generate_test_scripts.pl - Generate PowerShell test scripts with real data

:- use_module('../src/unifyweaver/core/powershell_compiler').
:- use_module('../src/unifyweaver/core/firewall_v2').

main :-
    format('~n╔════════════════════════════════════════════════════╗~n', []),
    format('║  Generating PowerShell Test Scripts               ║~n', []),
    format('╚════════════════════════════════════════════════════╝~n~n', []),

    mkdir_p('test_output'),

    generate_csv_pure,
    generate_json_pure,
    % Skip BaaS - needs actual Prolog clauses

    format('~n╔════════════════════════════════════════════════════╗~n', []),
    format('║  All Scripts Generated ✓                          ║~n', []),
    format('║  Check test_output/ directory                     ║~n', []),
    format('╚════════════════════════════════════════════════════╝~n', []),
    halt(0).

main :-
    format('[✗] Script generation failed~n', []),
    halt(1).

mkdir_p(Dir) :-
    (   exists_directory(Dir)
    ->  true
    ;   make_directory(Dir)
    ).

%% Generate CSV pure PowerShell script
generate_csv_pure :-
    format('[Generating] CSV Pure PowerShell: test_output/csv_pure.ps1~n', []),
    compile_to_powershell(user_data/3, [
        source_type(csv),
        csv_file('test_data/test_users.csv'),
        has_header(true),
        powershell_mode(pure),
        output_file('test_output/csv_pure.ps1')
    ], _),
    format('[✓] Generated csv_pure.ps1~n', []).

%% Generate CSV BaaS script
generate_csv_baas :-
    format('[Generating] CSV BaaS: test_output/csv_baas.ps1~n', []),
    compile_to_powershell(user_data/3, [
        source_type(csv),
        csv_file('test_data/test_users.csv'),
        has_header(true),
        powershell_mode(baas),
        output_file('test_output/csv_baas.ps1')
    ], _),
    format('[✓] Generated csv_baas.ps1~n', []).

%% Generate JSON pure PowerShell script
generate_json_pure :-
    format('[Generating] JSON Pure PowerShell: test_output/json_pure.ps1~n', []),
    compile_to_powershell(product_data/3, [
        source_type(json),
        json_file('test_data/test_products.json'),
        jq_filter('.[]'),
        powershell_mode(pure),
        output_file('test_output/json_pure.ps1')
    ], _),
    format('[✓] Generated json_pure.ps1~n', []).

%% Generate JSON BaaS script
generate_json_baas :-
    format('[Generating] JSON BaaS: test_output/json_baas.ps1~n', []),
    compile_to_powershell(product_data/3, [
        source_type(json),
        json_file('test_data/test_products.json'),
        jq_filter('.[]'),
        powershell_mode(baas),
        output_file('test_output/json_baas.ps1')
    ], _),
    format('[✓] Generated json_baas.ps1~n', []).

:- initialization(main, main).
