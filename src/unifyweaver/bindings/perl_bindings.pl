:- module(perl_bindings, [
    init_perl_bindings/0
]).

:- use_module('../core/binding_registry').

init_perl_bindings :-
    % Arithmetic
    % Target, Pred, TargetName (Template), Inputs, Outputs, Options
    declare_binding(perl, plus/3, 'my ~w = ~w + ~w;', [int, int], [int], [pattern(assignment)]),
    declare_binding(perl, minus/3, 'my ~w = ~w - ~w;', [int, int], [int], [pattern(assignment)]),
    declare_binding(perl, mult/3, 'my ~w = ~w * ~w;', [int, int], [int], [pattern(assignment)]),
    declare_binding(perl, div/3, 'my ~w = ~w / ~w;', [int, int], [int], [pattern(assignment)]),
    
    % String operations
    declare_binding(perl, concat/3, 'my ~w = ~w . ~w;', [string, string], [string], [pattern(assignment)]),
    declare_binding(perl, length/2, 'my ~w = length(~w);', [string], [int], [pattern(assignment)]),
    
    % Comparison
    declare_binding(perl, eq/2, 'if (~w eq ~w) {', [string, string], [], [pattern(block_start)]),
    declare_binding(perl, neq/2, 'if (~w ne ~w) {', [string, string], [], [pattern(block_start)]),
    declare_binding(perl, gt/2, 'if (~w > ~w) {', [int, int], [], [pattern(block_start)]),
    declare_binding(perl, lt/2, 'if (~w < ~w) {', [int, int], [], [pattern(block_start)]).

:- initialization(init_perl_bindings).