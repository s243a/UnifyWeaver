#!/usr/bin/env swipl
% Manual C# code generation script for section 4.1

:- initialization(main, main).

main :-
    % Setup test data
    asserta(user:test_parity_input(0)),
    asserta(user:test_parity_input(1)),
    asserta(user:test_parity_input(2)),
    asserta(user:test_parity_input(3)),
    asserta(user:test_parity_input(4)),
    asserta(user:test_even(0)),
    asserta(user:test_odd(1)),
    asserta(user:(test_even(N) :- test_parity_input(N), N > 0, N1 is N - 1, test_odd(N1))),
    asserta(user:(test_odd(N) :- test_parity_input(N), N > 1, N1 is N - 1, test_even(N1))),

    % Load C# query target
    use_module('src/unifyweaver/targets/csharp_query_target'),

    % Build plan
    writeln('Building query plan for test_even/1...'),
    csharp_query_target:build_query_plan(test_even/1, [target(csharp_query)], Plan),

    % Create output directory
    Dir = '/tmp/test_even_manual',
    (exists_directory(Dir) -> delete_directory_and_contents(Dir) ; true),
    make_directory(Dir),

    % Create dotnet project
    writeln('Creating dotnet console project...'),
    process_create(path(dotnet), ['new', 'console', '--force', '--framework', 'net9.0'],
                   [cwd(Dir), stdout(null), stderr(null)]),

    % Copy QueryRuntime.cs
    writeln('Copying QueryRuntime.cs...'),
    absolute_file_name('src/unifyweaver/targets/csharp_query_runtime/QueryRuntime.cs', RuntimePath, []),
    atom_concat(Dir, '/QueryRuntime.cs', RuntimeCopy),
    copy_file(RuntimePath, RuntimeCopy),

    % Generate and write query module
    writeln('Generating C# query module...'),
    csharp_query_target:render_plan_to_csharp(Plan, ModuleSource),
    csharp_query_target:plan_module_name(Plan, ModuleClass),
    atom_concat(ModuleClass, '.cs', ModuleFile),
    atom_concat(Dir, '/', DirSlash),
    atom_concat(DirSlash, ModuleFile, ModulePath),
    open(ModulePath, write, Stream1),
    write(Stream1, ModuleSource),
    close(Stream1),

    % Write harness (Program.cs)
    writeln('Writing Program.cs harness...'),
    format(atom(HarnessSource),
'using System;

class Program {
    static void Main() {
        foreach (var row in ~w.Query()) {
            Console.WriteLine(string.Join(",", row));
        }
    }
}', [ModuleClass]),
    atom_concat(Dir, '/Program.cs', ProgramPath),
    open(ProgramPath, write, Stream2),
    write(Stream2, HarnessSource),
    close(Stream2),

    format('~n✓ Generated C# code in: ~w~n', [Dir]),
    format('~nNext steps:~n'),
    format('  cd ~w~n', [Dir]),
    format('  dotnet build~n'),
    format('  dotnet run~n'),
    format('~nExpected output: 0, 2, 4~n~n'),
    halt(0).
