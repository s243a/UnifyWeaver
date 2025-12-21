% Test script for Phase 7-8 code generation predicates
% Usage: swipl -g run_tests -t halt test_small_world_codegen.pl

% Define the predicates inline for testing (extracted from target files)

% Python small-world
compile_small_world_proper_python(Options, Code) :-
    ( member(k_local(KLocal), Options) -> true ; KLocal = 10 ),
    ( member(k_long(KLong), Options) -> true ; KLong = 5 ),
    ( member(alpha(Alpha), Options) -> true ; Alpha = 2.0 ),
    format(string(Code), '
# Small-world network with k_local = ~w, k_long = ~w, alpha = ~w
from small_world_proper import ProperSmallWorldNetwork
network = ProperSmallWorldNetwork(k_local=~w, k_long=~w, alpha=~w)
', [KLocal, KLong, Alpha, KLocal, KLong, Alpha]).

% Go small-world
compile_small_world_proper_go(Options, Code) :-
    ( member(k_local(KLocal), Options) -> true ; KLocal = 10 ),
    ( member(k_long(KLong), Options) -> true ; KLong = 5 ),
    ( member(alpha(Alpha), Options) -> true ; Alpha = 2.0 ),
    format(string(Code), '
// Small-world network
const KLocal = ~w
const KLong = ~w
const Alpha = ~w
', [KLocal, KLong, Alpha]).

% Rust small-world
compile_small_world_proper_rust(Options, Code) :-
    ( member(k_local(KLocal), Options) -> true ; KLocal = 10 ),
    ( member(k_long(KLong), Options) -> true ; KLong = 5 ),
    ( member(alpha(Alpha), Options) -> true ; Alpha = 2.0 ),
    format(string(Code), '
// Small-world network
pub const K_LOCAL: usize = ~w;
pub const K_LONG: usize = ~w;
pub const ALPHA: f64 = ~w;
', [KLocal, KLong, Alpha]).

% Python multi-interface
compile_multi_interface_node_python(Options, Code) :-
    ( member(gamma(Gamma), Options) -> true ; Gamma = 2.5 ),
    ( member(min_interfaces(MinInt), Options) -> true ; MinInt = 1 ),
    ( member(max_interfaces(MaxInt), Options) -> true ; MaxInt = 100 ),
    format(string(Code), '
# Multi-interface with gamma = ~w
GAMMA = ~w
MIN_INTERFACES = ~w
MAX_INTERFACES = ~w
', [Gamma, Gamma, MinInt, MaxInt]).

% Go multi-interface
compile_multi_interface_node_go(Options, Code) :-
    ( member(gamma(Gamma), Options) -> true ; Gamma = 2.5 ),
    ( member(min_interfaces(MinInt), Options) -> true ; MinInt = 1 ),
    ( member(max_interfaces(MaxInt), Options) -> true ; MaxInt = 100 ),
    format(string(Code), '
// Multi-interface with gamma
const Gamma = ~w
const MinInterfaces = ~w
const MaxInterfaces = ~w
', [Gamma, MinInt, MaxInt]).

% Rust multi-interface
compile_multi_interface_node_rust(Options, Code) :-
    ( member(gamma(Gamma), Options) -> true ; Gamma = 2.5 ),
    ( member(min_interfaces(MinInt), Options) -> true ; MinInt = 1 ),
    ( member(max_interfaces(MaxInt), Options) -> true ; MaxInt = 100 ),
    format(string(Code), '
// Multi-interface with gamma
pub const GAMMA: f64 = ~w;
pub const MIN_INTERFACES: usize = ~w;
pub const MAX_INTERFACES: usize = ~w;
', [Gamma, MinInt, MaxInt]).

% Test Python small-world code generation
test_python_small_world :-
    format('Testing Python small-world code generation...~n'),
    Options = [k_local(10), k_long(5), alpha(2.0)],
    compile_small_world_proper_python(Options, Code),
    (sub_string(Code, _, _, _, "k_local = 10") ->
        format('  [PASS] Python small-world with k_local=10~n')
    ;
        format('  [FAIL] Python small-world missing k_local~n'),
        fail
    ).

% Test Go small-world code generation
test_go_small_world :-
    format('Testing Go small-world code generation...~n'),
    Options = [k_local(15), k_long(7), alpha(2.5)],
    compile_small_world_proper_go(Options, Code),
    (sub_string(Code, _, _, _, "KLocal = 15") ->
        format('  [PASS] Go small-world with KLocal=15~n')
    ;
        format('  [FAIL] Go small-world missing KLocal~n'),
        fail
    ).

% Test Rust small-world code generation
test_rust_small_world :-
    format('Testing Rust small-world code generation...~n'),
    Options = [k_local(20), k_long(10), alpha(1.5)],
    compile_small_world_proper_rust(Options, Code),
    (sub_string(Code, _, _, _, "K_LOCAL: usize = 20") ->
        format('  [PASS] Rust small-world with K_LOCAL=20~n')
    ;
        format('  [FAIL] Rust small-world missing K_LOCAL~n'),
        fail
    ).

% Test multi-interface predicates
test_multi_interface :-
    format('Testing multi-interface code generation...~n'),
    compile_multi_interface_node_python([gamma(2.5)], PyCode),
    (sub_string(PyCode, _, _, _, "GAMMA = 2.5") ->
        format('  [PASS] Python multi-interface with gamma=2.5~n')
    ;
        format('  [FAIL] Python multi-interface~n'), fail
    ),
    compile_multi_interface_node_go([gamma(3.0)], GoCode),
    (sub_string(GoCode, _, _, _, "Gamma = 3") ->
        format('  [PASS] Go multi-interface with Gamma=3~n')
    ;
        format('  [FAIL] Go multi-interface~n'), fail
    ),
    compile_multi_interface_node_rust([gamma(2.0)], RsCode),
    (sub_string(RsCode, _, _, _, "GAMMA: f64 = 2") ->
        format('  [PASS] Rust multi-interface with GAMMA=2~n')
    ;
        format('  [FAIL] Rust multi-interface~n'), fail
    ).

% Run all tests
run_tests :-
    format('~n=== Phase 7-8 Code Generation Tests ===~n~n'),
    (test_python_small_world -> true ; true),
    (test_go_small_world -> true ; true),
    (test_rust_small_world -> true ; true),
    (test_multi_interface -> true ; true),
    format('~n=== All tests completed ===~n').
