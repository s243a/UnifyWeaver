:- encoding(utf8).
% Emit wam_runtime.c for local C selftests / diff tooling.
:- use_module('../../../src/unifyweaver/targets/wam_c_target').
:- use_module(library(filesex),
              [make_directory_path/1, directory_file_path/3, copy_file/2]).

main :-
    current_prolog_flag(argv, [OutDir|_]),
    (   OutDir == []
    ->  OutDir2 = 'diff/runtime'
    ;   OutDir2 = OutDir
    ),
    make_directory_path(OutDir2),
    compile_wam_runtime_to_c([], RuntimeCode),
    directory_file_path(OutDir2, 'wam_runtime.c', RtPath),
    setup_call_cleanup(open(RtPath, write, S),
                       format(S, '~w', [RuntimeCode]),
                       close(S)),
    module_property(wam_c_target, file(TargetFile)),
    file_directory_name(TargetFile, TargetsDir),
    directory_file_path(TargetsDir, 'wam_c_runtime/wam_runtime.h', HdrSrc),
    directory_file_path(OutDir2, 'wam_runtime.h', HdrDst),
    copy_file(HdrSrc, HdrDst),
    format('gen_runtime.pl: wrote ~w and ~w~n', [RtPath, HdrDst]).
