:- use_module('../src/unifyweaver/targets/python_target').

compile_ancestor :-
    % Load facts and rules
    consult('family.pl'),
    
    % Compile in generator mode
    compile_predicate_to_python(
        ancestor/2,
        [mode(generator), record_format(jsonl)],
        PythonCode
    ),
    
    % Save to file
    open('ancestor.py', write, Stream),
    write(Stream, PythonCode),
    close(Stream),
    
    writeln('✓ Compiled ancestor.py in generator mode').

% Run automatically
:- compile_ancestor, halt.
