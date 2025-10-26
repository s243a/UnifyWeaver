:- encoding(utf8).
% Quick emoji test - verify emoji rendering in Windows Terminal
%
% Usage:
%   swipl -l init.pl -g quick_test -t halt examples/quick_emoji_test.pl
%
% To test different emoji levels:
%   UNIFYWEAVER_EMOJI_LEVEL=full swipl -l init.pl -g quick_test -t halt examples/quick_emoji_test.pl
%   UNIFYWEAVER_EMOJI_LEVEL=bmp swipl -l init.pl -g quick_test -t halt examples/quick_emoji_test.pl
%   UNIFYWEAVER_EMOJI_LEVEL=ascii swipl -l init.pl -g quick_test -t halt examples/quick_emoji_test.pl

:- use_module(unifyweaver(core/platform_compat)).

quick_test :-
    % Set emoji level from environment if provided
    (   getenv('UNIFYWEAVER_EMOJI_LEVEL', EnvLevel),
        atom_string(EmojiLevel, EnvLevel),
        memberchk(EmojiLevel, [ascii, bmp, full])
    ->  set_emoji_level(EmojiLevel)
    ;   true  % Keep default
    ),

    get_emoji_level(Level),
    format('~n=== Quick Emoji Test ===~n~n', []),
    format('Current emoji level: ~w~n~n', [Level]),

    % Test with actual emoji characters (single backslash escapes)
    format('Direct format/2 tests:~n', []),
    format('  ✅ Checkmark (BMP)~n', []),
    format('  🚀 Rocket (non-BMP)~n', []),
    format('  📊 Chart (non-BMP)~n', []),
    format('  ⚠ Warning (BMP)~n~n', []),

    % Test with safe_format (should adapt based on level)
    format('safe_format/2 tests:~n', []),
    safe_format('  ✅ Checkmark via safe_format~n', []),
    safe_format('  🚀 Rocket via safe_format~n', []),
    safe_format('  📊 Chart via safe_format~n', []),
    safe_format('  ⚠ Warning via safe_format~n', []),

    format('~nTest complete!~n~n', []).
