:- module(test_ps_xml, [main/0]).
:- use_module('../src/unifyweaver/targets/powershell_target').

main :-
    % Compile a predicate 'stream_pearls' that reads 'data.xml' and extracts 'pt:Pearl' tags
    compile_predicate_to_powershell(stream_pearls/1, 
        [input_source(xml('context/PT/pearltrees_export.rdf', ['pt:PagePearl', 'pt:Tree']))], 
        Code),
    
    open('output/test_xml.ps1', write, S),
    write(S, Code),
    close(S),
    format('Generated output/test_xml.ps1~n', []).
