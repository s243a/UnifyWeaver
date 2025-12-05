:- module(test_ps_vector, [main/0]).
:- use_module('../src/unifyweaver/targets/powershell_target').

main :-
    % Generate vector ops code
    powershell_target:compile_vector_ops([], Code),
    
    % Add a test harness
    TestCode = '
$v1 = @(1.0, 0.0, 1.0)
$v2 = @(1.0, 0.0, 1.0)
$v3 = @(0.0, 1.0, 0.0)

$score1 = Get-CosineSimilarity $v1 $v2
Write-Host "Similarity (Identical): $score1"

$score2 = Get-CosineSimilarity $v1 $v3
Write-Host "Similarity (Orthogonal): $score2"
',
    
    format(string(FullCode), "~s~n~s", [Code, TestCode]),
    
    open('output/test_vector.ps1', write, S),
    write(S, FullCode),
    close(S),
    format('Generated output/test_vector.ps1~n', []).
