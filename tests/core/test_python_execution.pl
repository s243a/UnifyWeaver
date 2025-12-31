:- module(test_python_execution, [test_python_execution/0]).
:- use_module(library(plunit)).
:- use_module(library(process)).
:- use_module(library(http/json)).
:- use_module(unifyweaver(targets/python_target)).

test_python_execution :-
    run_tests([python_execution]).

:- begin_tests(python_execution).

test(streaming_behavior, [setup(setup_python_script(ScriptFile)), cleanup(delete_file(ScriptFile))]) :-
    % 1. Compile a simple pass-through predicate
    % pass(R) :- true.
    assertz(pass(_)),
    compile_predicate_to_python(pass/1, [], Code),
    retract(pass(_)),
    
    % 2. Save to file
    open(ScriptFile, write, Stream),
    write(Stream, Code),
    close(Stream),
    
    % 3. Run Python process
    % We need to find python executable. Assume 'python' or 'python3' is on PATH.
    (   catch(process_create(path(python), ['--version'], [stdout(null)]), _, fail)
    ->  Python = path(python)
    ;   Python = path(python3)
    ),
    
    process_create(Python, ['-u', ScriptFile], [stdin(pipe(In)), stdout(pipe(Out))]),
    
    % 4. Feed data and check output incrementally
    % Send Record 1
    format(In, '{"id": 1}\n', []),
    flush_output(In),
    
    % Read Record 1
    read_line_to_string(Out, Line1),
    assertion(Line1 \== end_of_file),
    atom_json_dict(Line1, Json1, []),
    assertion(Json1.id == 1),
    
    % Send Record 2
    format(In, '{"id": 2}\n', []),
    flush_output(In),
    
    % Read Record 2
    read_line_to_string(Out, Line2),
    assertion(Line2 \== end_of_file),
    atom_json_dict(Line2, Json2, []),
    assertion(Json2.id == 2),
    
    close(In),
    close(Out).

setup_python_script(File) :-
    tmp_file('test_script', File).

test(multi_clause_execution, [setup(setup_python_script(ScriptFile)), cleanup(delete_file(ScriptFile))]) :-
    % parent(R, c1) :- R.id = 1.
    % parent(R, c2) :- R.id = 1.
    assertz((parent(R, c1) :- get_dict(id, R, 1))),
    assertz((parent(R, c2) :- get_dict(id, R, 1))),
    
    compile_predicate_to_python(parent/2, [], Code),
    retractall(parent(_, _)),
    
    % Save and run
    open(ScriptFile, write, Stream),
    write(Stream, Code),
    close(Stream),
    
    % Find python
    (   catch(process_create(path(python), ['--version'], [stdout(null)]), _, fail)
    ->  Python = path(python)
    ;   Python = path(python3)
    ),
    
    process_create(Python, ['-u', ScriptFile], [stdin(pipe(In)), stdout(pipe(Out))]),
    
    % Send {"id": 1}
    format(In, '{"id": 1}\n', []),
    flush_output(In),
    
    % Expect c1
    read_line_to_string(Out, Line1),
    assertion(Line1 \== end_of_file),
    % Expect "c1"
    % atom_json_dict expects a JSON object (dict) or we can use json_read_term
    % But json.dumps("c1") -> "c1" (with quotes).
    % atom_json_dict might fail if it's not a dict.
    % Let's use term_string for simple values if they are JSON compatible?
    % Or just check the string.
    assertion(Line1 == "\"c1\""),
    
    % Expect c2
    read_line_to_string(Out, Line2),
    assertion(Line2 \== end_of_file),
    assertion(Line2 == "\"c2\""),
    
    close(In),
    close(Out).

test(factorial_execution, [setup(setup_python_script(ScriptFile)), cleanup(delete_file(ScriptFile))]) :-
    % Define factorial
    assertz((factorial(0, 1))),
    assertz((factorial(N, F) :- N > 0, N1 is N - 1, factorial(N1, F1), F is N * F1)),
    
    compile_predicate_to_python(factorial/2, [], Code),
    retractall(factorial(_,_)),
    
    % Save and run
    open(ScriptFile, write, Stream),
    write(Stream, Code),
    close(Stream),
    
    % Find python
    (   catch(process_create(path(python), ['--version'], [stdout(null)]), _, fail)
    ->  Python = path(python)
    ;   Python = path(python3)
    ),
    
    process_create(Python, ['-u', ScriptFile], [stdin(pipe(In)), stdout(pipe(Out))]),
    
    % Test factorial(5) = 120
    format(In, '{"n": 5}\n', []),
    flush_output(In),
    
    read_line_to_string(Out, Line1),
    assertion(Line1 \== end_of_file),
    atom_json_dict(Line1, Json1, []),
    assertion(Json1.get(result) == 120),
    
    close(In),
    close(Out).

test(sum_execution, [setup(setup_python_script(ScriptFile)), cleanup(delete_file(ScriptFile))]) :-
    % Define sum with accumulator (tail recursive, arity 3)
    assertz((sum(0, Acc, Acc))),
    assertz((sum(N, Acc, S) :- N > 0, N1 is N - 1, Acc1 is Acc + N, sum(N1, Acc1, S))),
    
    compile_predicate_to_python(sum/3, [], Code),
    retractall(sum(_,_,_)),
    
    % Save and run
    open(ScriptFile, write, Stream),
    write(Stream, Code),
    close(Stream),
    
    % Find python
    (   catch(process_create(path(python), ['--version'], [stdout(null)]), _, fail)
    ->  Python = path(python)
    ;   Python = path(python3)
    ),
    
    process_create(Python, ['-u', ScriptFile], [stdin(pipe(In)), stdout(pipe(Out))]),
    
    % Test sum(5, 0) = 15 (5+4+3+2+1 = 15)
    format(In, '{"n": 5, "acc": 0}\n', []),
    flush_output(In),
    
    read_line_to_string(Out, Line1),
    assertion(Line1 \== end_of_file),
    atom_json_dict(Line1, Json1, []),
    assertion(Json1.get(result) == 15),
    
    close(In),
    close(Out).

:- end_tests(python_execution).
