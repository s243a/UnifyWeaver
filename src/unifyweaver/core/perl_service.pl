:- module(perl_service, [
    check_perl_available/0,
    generate_inline_perl_call/4
]).

% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% perl_service.pl - Provides an interface for calling inline Perl scripts.

:- use_module(library(process)).
:- use_module(library(readutil)).

%% check_perl_available/0
%  Succeeds if a `perl` interpreter is found.
check_perl_available :-
    perl_candidates(Candidates),
    member(Exec-Args, Candidates),
    perl_check(Exec, Args),
    !.
check_perl_available :-
    format('Perl check failed using all configured candidates.~n', []),
    fail.

perl_candidates([
    path(perl)-['-v'],
    '/usr/bin/perl'-['-v'],
    '/bin/perl'-['-v']
]).

perl_check(Exec, Args) :-
    catch(
        process_create(Exec, Args, [stdout(null), stderr(null), process(Process)]),
        Error,
        (
            (   Error = error(existence_error(_, _), _)
            ->  fail
            ;   format('Perl check via ~q raised exception: ~q~n', [Exec, Error]),
                fail
            )
        )
    ),
    process_wait(Process, exit(0)).


%% generate_inline_perl_call(+PerlCode, +Args, +InputVar, -BashCode)
%  Generates bash code to execute an inline Perl script.
%
%  The generated snippet embeds the Perl code via a single-quoted heredoc that
%  feeds `/dev/fd/3`. If `InputVar` is the atom `stdin`, the caller is expected
%  to redirect stdin (for example using a pipeline or `< file`). Otherwise the
%  snippet will append a here-string `<<< "$InputVar"` so the Perl script
%  receives data from the provided bash variable.
%
%  @param +PerlCode  Atom containing the Perl program.
%  @param +Args      List of atoms representing arguments passed to Perl.
%  @param +InputVar  Atom with bash variable name, or `stdin`.
%  @param -BashCode  Atom with the generated bash snippet.
generate_inline_perl_call(PerlCode, Args, InputVar, BashCode) :-
    maplist(shell_quote_arg, Args, QuotedArgs),
    args_segment_codes(QuotedArgs, ArgSegmentCodes),
    choose_heredoc_label(PerlCode, Label),
    perl_body_codes(PerlCode, BodyCodes),
    input_redirect_codes(InputVar, InputRedirectCodes),
    atom_codes(Label, LabelCodes),
    phrase(perl_call_codes(ArgSegmentCodes, LabelCodes, InputRedirectCodes, BodyCodes), Codes),
    atom_codes(BashCode, Codes).

%% shell_quote_arg(+Atom, -Quoted)
%  Quote an argument for safe use in bash.
shell_quote_arg(Atom, Quoted) :-
    atom_codes(Atom, Codes),
    phrase(shell_quote_codes(Codes), QuotedCodes),
    atom_codes(Quoted, QuotedCodes).

shell_quote_codes(Codes) -->
    "'", shell_quote_body(Codes), "'".

shell_quote_body([]) --> [].
shell_quote_body([39|Rest]) --> "'", "\\", "'", "'", shell_quote_body(Rest).
shell_quote_body([C|Rest]) --> [C], shell_quote_body(Rest).

args_segment_codes([], []).
args_segment_codes(QuotedArgs, Codes) :-
    QuotedArgs \= [],
    atomic_list_concat(QuotedArgs, ' ', Joined),
    atom_codes(Joined, JoinedCodes),
    Codes = [32|JoinedCodes]. % leading space before joined args

choose_heredoc_label(PerlCode, Label) :-
    choose_heredoc_label(PerlCode, 'PERL', Label).

choose_heredoc_label(PerlCode, Candidate, Label) :-
    (   sub_atom(PerlCode, _, _, _, Candidate)
    ->  atom_concat(Candidate, '_END', Next),
        choose_heredoc_label(PerlCode, Next, Label)
    ;   Label = Candidate
    ).

perl_body_codes(PerlCode, BodyCodes) :-
    atom_codes(PerlCode, Codes),
    (   Codes = []
    ->  BodyCodes = []
    ;   (   append(_, [10], Codes)
        ->  BodyCodes = Codes
        ;   append(Codes, [10], BodyCodes)
        )
    ).

input_redirect_codes(stdin, []) :- !.
input_redirect_codes(InputVar, Codes) :-
    format(atom(Redirect), ' <<< "$~w"', [InputVar]),
    atom_codes(Redirect, Codes).

perl_call_codes(ArgSegment, LabelCodes, InputRedirect, Body) -->
    "perl /dev/fd/3",
    arg_segment(ArgSegment),
    " 3<<'",
    LabelCodes,
    "'",
    InputRedirect,
    "\n",
    Body,
    LabelCodes,
    "\n".

arg_segment([]) --> [].
arg_segment(Codes) --> Codes.
