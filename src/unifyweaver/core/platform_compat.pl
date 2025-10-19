:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (@s243a)
%
% platform_compat.pl - Cross-platform compatibility utilities
%
% Provides safe_format/2 and safe_format/3 predicates that gracefully
% degrade emoji and Unicode output on platforms with limited support.

:- module(platform_compat, [
    safe_format/2,
    safe_format/3,
    set_emoji_support/1,
    set_emoji_level/1,
    emoji_supported/0,
    get_emoji_level/1,
    detect_terminal/1,
    auto_detect_and_set_emoji_level/0
]).

%% ============================================================================
%% EMOJI SUPPORT CONFIGURATION
%% ============================================================================

:- dynamic emoji_level/1.

% Default: BMP symbols only (safe for ConEmu)
% Levels:
%   - ascii:  No emoji, use ASCII fallbacks only ([OK], [STEP], etc.)
%   - bmp:    BMP Unicode symbols only (✅ ❌ ⚠ ℹ ⚡) without variation selectors
%   - full:   Full emoji support including non-BMP (🚀 📊 📈 etc.)
emoji_level(bmp).

%! set_emoji_level(+Level:atom) is det.
%
%  Set the emoji support level based on terminal capabilities.
%
%  @arg Level One of: ascii, bmp, full

set_emoji_level(Level) :-
    must_be(oneof([ascii, bmp, full]), Level),
    retractall(emoji_level(_)),
    assertz(emoji_level(Level)).

%! get_emoji_level(-Level:atom) is det.
%
%  Get the current emoji support level.

get_emoji_level(Level) :-
    emoji_level(Level).

%! set_emoji_support(+Supported:boolean) is det.
%
%  Legacy interface: Convert boolean to emoji level.
%  true = full, false = ascii
%
%  @arg Supported true to enable full emoji, false for ASCII only

set_emoji_support(true) :-
    set_emoji_level(full).
set_emoji_support(false) :-
    set_emoji_level(ascii).

%! emoji_supported is semidet.
%
%  Legacy interface: Check if any emoji are supported (bmp or full).

emoji_supported :-
    emoji_level(Level),
    Level \= ascii.

%% ============================================================================
%% EMOJI MAPPING TABLE
%% ============================================================================

%! emoji_to_bmp(+FullEmoji:atom, -BmpSymbol:atom) is semidet.
%
%  Maps full emoji (with U+FE0F) to BMP-only versions (without variation selector).
%  For non-BMP emoji, maps to closest BMP equivalent or keeps as-is.

% Status indicators - remove U+FE0F variation selector for BMP compatibility
emoji_to_bmp('⚠️', '⚠').   % U+26A0 U+FE0F → U+26A0
emoji_to_bmp('ℹ️', 'ℹ').   % U+2139 U+FE0F → U+2139
emoji_to_bmp('🏗️', '🏗').  % U+1F3D7 U+FE0F → U+1F3D7

% Non-BMP emoji have no BMP equivalent, but we list them for documentation
% These will fall through to ASCII in BMP mode
% (The emoji_fallback rules below handle the ASCII conversion)

%! emoji_fallback(+Emoji:atom, -Fallback:atom) is semidet.
%
%  Maps emoji characters to their ASCII fallback representations.
%  Used when emoji_level is 'ascii' or when 'bmp' mode encounters non-BMP emoji.

% Status indicators (BMP range U+2000-U+27FF)
emoji_fallback('✅', '[OK]').      % U+2705
emoji_fallback('❌', '[FAIL]').    % U+274C
emoji_fallback('⚠️', '[WARN]').    % U+26A0 U+FE0F
emoji_fallback('⚠', '[WARN]').     % U+26A0 (BMP version)
emoji_fallback('ℹ️', '[INFO]').    % U+2139 U+FE0F
emoji_fallback('ℹ', '[INFO]').     % U+2139 (BMP version)

% Progress indicators (non-BMP range U+1F300-U+1F9FF)
emoji_fallback('🚀', '[STEP]').    % U+1F680
emoji_fallback('📡', '[LOAD]').    % U+1F4E1
emoji_fallback('📊', '[DATA]').    % U+1F4CA
emoji_fallback('📈', '[PROC]').    % U+1F4C8
emoji_fallback('💾', '[SAVE]').    % U+1F4BE
emoji_fallback('🔄', '[SYNC]').    % U+1F504

% Task indicators (non-BMP range)
emoji_fallback('👥', '[USER]').    % U+1F465
emoji_fallback('🎯', '[GOAL]').    % U+1F3AF
emoji_fallback('🎉', '[DONE]').    % U+1F389
emoji_fallback('🔧', '[TOOL]').    % U+1F527
emoji_fallback('🔍', '[FIND]').    % U+1F50D
emoji_fallback('📝', '[NOTE]').    % U+1F4DD

% Category indicators (mixed BMP and non-BMP)
emoji_fallback('🏗️', '[BUILD]').  % U+1F3D7 U+FE0F
emoji_fallback('🏗', '[BUILD]').   % U+1F3D7 (without variation selector)
emoji_fallback('🧪', '[TEST]').    % U+1F9EA
emoji_fallback('📦', '[PKG]').     % U+1F4E6
emoji_fallback('🌐', '[WEB]').     % U+1F310
emoji_fallback('🔐', '[SEC]').     % U+1F510
emoji_fallback('⚡', '[FAST]').    % U+26A1 (BMP)

%% ============================================================================
%% SAFE FORMAT PREDICATES
%% ============================================================================

%! safe_format(+Format:atom, +Args:list) is det.
%
%  Like format/2, but replaces emoji characters with ASCII fallbacks
%  when emoji_supported/0 fails.
%
%  @arg Format Format string (may contain emoji)
%  @arg Args Arguments for format/2

safe_format(Format, Args) :-
    safe_format(current_output, Format, Args).

%! safe_format(+Stream, +Format:atom, +Args:list) is det.
%
%  Like format/3, but adapts emoji based on terminal capability level.
%
%  @arg Stream Output stream
%  @arg Format Format string (may contain emoji)
%  @arg Args Arguments for format/3

safe_format(Stream, Format, Args) :-
    get_emoji_level(Level),
    adapt_format(Level, Format, AdaptedFormat),
    format(Stream, AdaptedFormat, Args).

%! adapt_format(+Level:atom, +Format:atom, -AdaptedFormat:atom) is det.
%
%  Adapt format string based on emoji support level.
%
%  @arg Level Emoji level: ascii, bmp, or full
%  @arg Format Original format string
%  @arg AdaptedFormat Adapted format string

adapt_format(full, Format, Format) :-
    % Full emoji support - use as-is
    !.
adapt_format(bmp, Format, AdaptedFormat) :-
    % BMP only - remove variation selectors and convert non-BMP to ASCII
    !,
    atom_chars(Format, Chars),
    convert_to_bmp(Chars, BmpChars),
    atom_chars(AdaptedFormat, BmpChars).
adapt_format(ascii, Format, AdaptedFormat) :-
    % ASCII only - replace all emoji with ASCII fallbacks
    !,
    atom_chars(Format, Chars),
    replace_emoji_chars(Chars, AsciiChars),
    atom_chars(AdaptedFormat, AsciiChars).

%! convert_to_bmp(+Chars:list, -BmpChars:list) is det.
%
%  Convert emoji to BMP-safe versions:
%  - Remove U+FE0F variation selectors
%  - Replace non-BMP emoji with ASCII fallbacks
%  - Keep BMP symbols as-is

convert_to_bmp([], []).
convert_to_bmp([Char|Rest], Result) :-
    (   is_variation_selector(Char)
    ->  % Skip variation selectors (U+FE0F, U+FE0E)
        Result = BmpRest
    ;   emoji_to_bmp(Char, BmpVersion)
    ->  % Has BMP equivalent - use it
        atom_chars(BmpVersion, BmpChars),
        append(BmpChars, BmpRest, Result)
    ;   emoji_fallback(Char, Fallback),
        \+ char_is_bmp(Char)
    ->  % Non-BMP emoji - use ASCII fallback
        atom_chars(Fallback, FallbackChars),
        append(FallbackChars, BmpRest, Result)
    ;   % BMP symbol or regular character - keep as-is
        Result = [Char|BmpRest]
    ),
    convert_to_bmp(Rest, BmpRest).

%! char_is_bmp(+Char:atom) is semidet.
%
%  Check if a character is in the Basic Multilingual Plane (U+0000-U+FFFF).

char_is_bmp(Char) :-
    char_code(Char, Code),
    Code >= 0x0000,
    Code =< 0xFFFF.

%% ============================================================================
%% TERMINAL DETECTION
%% ============================================================================

%! detect_terminal(-Terminal:atom) is det.
%
%  Detect the current terminal emulator by checking environment variables.
%  Returns one of: windows_terminal, conemu, wsl, unknown
%
%  @arg Terminal The detected terminal type

detect_terminal(Terminal) :-
    (   getenv('WT_SESSION', _)
    ->  Terminal = windows_terminal
    ;   getenv('ConEmuPID', _)
    ->  Terminal = conemu
    ;   getenv('WSL_DISTRO_NAME', _)
    ->  Terminal = wsl
    ;   Terminal = unknown
    ).

%! terminal_emoji_level(+Terminal:atom, -Level:atom) is det.
%
%  Map terminal type to recommended emoji support level.
%
%  @arg Terminal Terminal type from detect_terminal/1
%  @arg Level Recommended emoji level: ascii, bmp, or full

terminal_emoji_level(windows_terminal, full).
terminal_emoji_level(wsl, full).
terminal_emoji_level(conemu, bmp).
terminal_emoji_level(unknown, bmp).  % Conservative default

%! auto_detect_and_set_emoji_level is det.
%
%  Automatically detect terminal and set appropriate emoji level.
%  Call this during initialization for automatic configuration.

auto_detect_and_set_emoji_level :-
    detect_terminal(Terminal),
    terminal_emoji_level(Terminal, Level),
    set_emoji_level(Level),
    format('[Platform Compat] Detected terminal: ~w, emoji level: ~w~n', [Terminal, Level]).

%! replace_emoji(+Format:atom, -SafeFormat:atom) is det.
%
%  Replace all emoji characters in Format with their ASCII fallbacks.
%
%  @arg Format Original format string
%  @arg SafeFormat Format string with emoji replaced

replace_emoji(Format, SafeFormat) :-
    atom_chars(Format, Chars),
    replace_emoji_chars(Chars, SafeChars),
    atom_chars(SafeFormat, SafeChars).

%! replace_emoji_chars(+Chars:list, -SafeChars:list) is det.
%
%  Replace emoji characters in a character list.
%  Also strips U+FE0F variation selectors.

replace_emoji_chars([], []).
replace_emoji_chars([Char|Rest], Result) :-
    (   is_variation_selector(Char)
    ->  % Skip variation selectors (U+FE0F, U+FE0E)
        Result = SafeRest
    ;   emoji_fallback(Char, Fallback)
    ->  % Found emoji - replace with fallback
        atom_chars(Fallback, FallbackChars),
        append(FallbackChars, SafeRest, Result)
    ;   % Not an emoji - keep as-is
        Result = [Char|SafeRest]
    ),
    replace_emoji_chars(Rest, SafeRest).

%! is_variation_selector(+Char:atom) is semidet.
%
%  Check if character is a Unicode variation selector.

is_variation_selector(Char) :-
    char_code(Char, Code),
    (   Code =:= 0xFE0F    % Variation Selector-16 (emoji presentation)
    ;   Code =:= 0xFE0E    % Variation Selector-15 (text presentation)
    ).

%% ============================================================================
%% USAGE EXAMPLES
%% ============================================================================

/*
% Example 1: Basic usage with emoji support enabled
?- set_emoji_support(true).
?- safe_format('✅ Test passed~n', []).
✅ Test passed

% Example 2: Basic usage with emoji support disabled
?- set_emoji_support(false).
?- safe_format('✅ Test passed~n', []).
[OK] Test passed

% Example 3: Multiple emoji in one format string
?- set_emoji_support(false).
?- safe_format('🚀 Starting process... 📊 Loading data... ✅ Complete!~n', []).
[STEP] Starting process... [DATA] Loading data... [OK] Complete!

% Example 4: With format arguments
?- set_emoji_support(false).
?- safe_format('📡 Loading ~w from ~w... ✅~n', [module, 'src/core.pl']).
[LOAD] Loading module from src/core.pl... [OK]

% Example 5: Stream output
?- set_emoji_support(false).
?- open('output.txt', write, Stream).
?- safe_format(Stream, '✅ Success~n', []).
?- close(Stream).
*/
