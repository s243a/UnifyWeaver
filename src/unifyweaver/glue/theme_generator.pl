% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% Theme Generator - Centralized Theme Definitions for Visualizations
%
% This module provides declarative theme specifications that generate
% CSS custom properties and theme switching functionality.
%
% Usage:
%   % Define a theme
%   theme(corporate, [
%       colors([primary('#1e40af'), secondary('#3b82f6')]),
%       typography([font_family('Inter, sans-serif'), base_size(16)])
%   ]).
%
%   % Generate theme CSS
%   ?- generate_theme_css(corporate, CSS).

:- module(theme_generator, [
    % Theme specifications
    theme/2,                        % theme(+Name, +Options)
    theme_extends/2,                % theme_extends(+Child, +Parent)

    % Color palette specifications
    color_palette/2,                % color_palette(+Name, +Colors)
    semantic_colors/2,              % semantic_colors(+Theme, +Mappings)

    % Typography specifications
    typography/2,                   % typography(+Name, +Options)
    font_scale/2,                   % font_scale(+Name, +Scale)

    % Spacing specifications
    spacing_scale/2,                % spacing_scale(+Name, +Values)

    % Generation predicates
    generate_theme_css/2,           % generate_theme_css(+Theme, -CSS)
    generate_theme_variables/2,     % generate_theme_variables(+Theme, -Vars)
    generate_theme_provider/2,      % generate_theme_provider(+Themes, -JSX)
    generate_theme_hook/1,          % generate_theme_hook(-Hook)
    generate_theme_context/1,       % generate_theme_context(-Context)
    generate_theme_toggle/2,        % generate_theme_toggle(+Themes, -Toggle)
    generate_theme_types/2,         % generate_theme_types(+Themes, -Types)

    % Utility predicates
    get_theme_colors/2,             % get_theme_colors(+Theme, -Colors)
    get_theme_typography/2,         % get_theme_typography(+Theme, -Typography)
    get_theme_spacing/2,            % get_theme_spacing(+Theme, -Spacing)
    resolve_theme/2,                % resolve_theme(+Theme, -ResolvedOptions)

    % Management
    declare_theme/2,                % declare_theme(+Name, +Options)
    clear_themes/0,                 % clear_themes

    % Testing
    test_theme_generator/0
]).

:- use_module(library(lists)).

% ============================================================================
% DYNAMIC PREDICATES
% ============================================================================

:- dynamic theme/2.
:- dynamic theme_extends/2.
:- dynamic color_palette/2.
:- dynamic semantic_colors/2.
:- dynamic typography/2.
:- dynamic font_scale/2.
:- dynamic spacing_scale/2.

:- discontiguous theme/2.
:- discontiguous color_palette/2.

% ============================================================================
% DEFAULT COLOR PALETTES
% ============================================================================

color_palette(slate, [
    c50('#f8fafc'), c100('#f1f5f9'), c200('#e2e8f0'), c300('#cbd5e1'),
    c400('#94a3b8'), c500('#64748b'), c600('#475569'), c700('#334155'),
    c800('#1e293b'), c900('#0f172a'), c950('#020617')
]).

color_palette(blue, [
    c50('#eff6ff'), c100('#dbeafe'), c200('#bfdbfe'), c300('#93c5fd'),
    c400('#60a5fa'), c500('#3b82f6'), c600('#2563eb'), c700('#1d4ed8'),
    c800('#1e40af'), c900('#1e3a8a'), c950('#172554')
]).

color_palette(emerald, [
    c50('#ecfdf5'), c100('#d1fae5'), c200('#a7f3d0'), c300('#6ee7b7'),
    c400('#34d399'), c500('#10b981'), c600('#059669'), c700('#047857'),
    c800('#065f46'), c900('#064e3b'), c950('#022c22')
]).

color_palette(red, [
    c50('#fef2f2'), c100('#fee2e2'), c200('#fecaca'), c300('#fca5a5'),
    c400('#f87171'), c500('#ef4444'), c600('#dc2626'), c700('#b91c1c'),
    c800('#991b1b'), c900('#7f1d1d'), c950('#450a0a')
]).

color_palette(amber, [
    c50('#fffbeb'), c100('#fef3c7'), c200('#fde68a'), c300('#fcd34d'),
    c400('#fbbf24'), c500('#f59e0b'), c600('#d97706'), c700('#b45309'),
    c800('#92400e'), c900('#78350f'), c950('#451a03')
]).

color_palette(violet, [
    c50('#f5f3ff'), c100('#ede9fe'), c200('#ddd6fe'), c300('#c4b5fd'),
    c400('#a78bfa'), c500('#8b5cf6'), c600('#7c3aed'), c700('#6d28d9'),
    c800('#5b21b6'), c900('#4c1d95'), c950('#2e1065')
]).

% ============================================================================
% DEFAULT THEMES
% ============================================================================

theme(light, [
    colors([
        primary('#3b82f6'),
        secondary('#64748b'),
        accent('#8b5cf6'),
        success('#10b981'),
        warning('#f59e0b'),
        error('#ef4444'),
        background('#ffffff'),
        surface('#f8fafc'),
        text_primary('#0f172a'),
        text_secondary('#475569'),
        text_muted('#94a3b8'),
        border('#e2e8f0'),
        shadow('rgba(0, 0, 0, 0.1)')
    ]),
    typography([
        font_family('"Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif'),
        font_mono('"JetBrains Mono", "Fira Code", monospace'),
        base_size(16),
        line_height(1.5),
        heading_weight(600),
        body_weight(400)
    ]),
    spacing([
        unit(4),
        xs(4), sm(8), md(16), lg(24), xl(32), xxl(48)
    ]),
    borders([
        radius_sm(4), radius_md(8), radius_lg(12), radius_full(9999),
        width(1)
    ]),
    shadows([
        sm('0 1px 2px 0 rgba(0, 0, 0, 0.05)'),
        md('0 4px 6px -1px rgba(0, 0, 0, 0.1)'),
        lg('0 10px 15px -3px rgba(0, 0, 0, 0.1)'),
        xl('0 20px 25px -5px rgba(0, 0, 0, 0.1)')
    ]),
    transitions([
        duration_fast(150),
        duration_normal(300),
        duration_slow(500),
        easing('cubic-bezier(0.4, 0, 0.2, 1)')
    ])
]).

theme(dark, [
    colors([
        primary('#60a5fa'),
        secondary('#94a3b8'),
        accent('#a78bfa'),
        success('#34d399'),
        warning('#fbbf24'),
        error('#f87171'),
        background('#0f172a'),
        surface('#1e293b'),
        text_primary('#f8fafc'),
        text_secondary('#cbd5e1'),
        text_muted('#64748b'),
        border('#334155'),
        shadow('rgba(0, 0, 0, 0.3)')
    ]),
    typography([
        font_family('"Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif'),
        font_mono('"JetBrains Mono", "Fira Code", monospace'),
        base_size(16),
        line_height(1.5),
        heading_weight(600),
        body_weight(400)
    ]),
    spacing([
        unit(4),
        xs(4), sm(8), md(16), lg(24), xl(32), xxl(48)
    ]),
    borders([
        radius_sm(4), radius_md(8), radius_lg(12), radius_full(9999),
        width(1)
    ]),
    shadows([
        sm('0 1px 2px 0 rgba(0, 0, 0, 0.2)'),
        md('0 4px 6px -1px rgba(0, 0, 0, 0.3)'),
        lg('0 10px 15px -3px rgba(0, 0, 0, 0.3)'),
        xl('0 20px 25px -5px rgba(0, 0, 0, 0.3)')
    ]),
    transitions([
        duration_fast(150),
        duration_normal(300),
        duration_slow(500),
        easing('cubic-bezier(0.4, 0, 0.2, 1)')
    ])
]).

theme(high_contrast, [
    colors([
        primary('#0000ff'),
        secondary('#000000'),
        accent('#ff00ff'),
        success('#008000'),
        warning('#ffff00'),
        error('#ff0000'),
        background('#ffffff'),
        surface('#ffffff'),
        text_primary('#000000'),
        text_secondary('#000000'),
        text_muted('#333333'),
        border('#000000'),
        shadow('none')
    ]),
    typography([
        font_family('Arial, sans-serif'),
        font_mono('Courier New, monospace'),
        base_size(18),
        line_height(1.6),
        heading_weight(700),
        body_weight(400)
    ]),
    spacing([
        unit(4),
        xs(4), sm(8), md(16), lg(24), xl(32), xxl(48)
    ]),
    borders([
        radius_sm(0), radius_md(0), radius_lg(0), radius_full(0),
        width(2)
    ]),
    shadows([
        sm('none'), md('none'), lg('none'), xl('none')
    ]),
    transitions([
        duration_fast(0),
        duration_normal(0),
        duration_slow(0),
        easing('linear')
    ])
]).

% Corporate theme extending light
theme(corporate, [
    extends(light),
    colors([
        primary('#1e40af'),
        secondary('#475569'),
        accent('#0891b2')
    ])
]).

% Nature theme extending light
theme(nature, [
    extends(light),
    colors([
        primary('#047857'),
        secondary('#365314'),
        accent('#84cc16')
    ])
]).

% ============================================================================
% FONT SCALES
% ============================================================================

font_scale(default, [
    xs(0.75),      % 12px at 16px base
    sm(0.875),     % 14px
    base(1),       % 16px
    lg(1.125),     % 18px
    xl(1.25),      % 20px
    xxl(1.5),      % 24px
    xxxl(1.875),   % 30px
    display(2.25), % 36px
    giant(3)       % 48px
]).

font_scale(compact, [
    xs(0.6875),    % 11px at 16px base
    sm(0.75),      % 12px
    base(0.875),   % 14px
    lg(1),         % 16px
    xl(1.125),     % 18px
    xxl(1.25),     % 20px
    xxxl(1.5),     % 24px
    display(1.875),% 30px
    giant(2.25)    % 36px
]).

font_scale(large, [
    xs(0.875),     % 14px at 16px base
    sm(1),         % 16px
    base(1.125),   % 18px
    lg(1.25),      % 20px
    xl(1.5),       % 24px
    xxl(1.875),    % 30px
    xxxl(2.25),    % 36px
    display(3),    % 48px
    giant(3.75)    % 60px
]).

% ============================================================================
% SPACING SCALES
% ============================================================================

spacing_scale(default, [
    px(1), s0_5(2), s1(4), s1_5(6), s2(8), s2_5(10), s3(12), s3_5(14),
    s4(16), s5(20), s6(24), s7(28), s8(32), s9(36), s10(40),
    s11(44), s12(48), s14(56), s16(64), s20(80), s24(96),
    s28(112), s32(128), s36(144), s40(160), s44(176), s48(192),
    s52(208), s56(224), s60(240), s64(256), s72(288), s80(320), s96(384)
]).

spacing_scale(compact, [
    px(1), s0_5(1), s1(2), s1_5(3), s2(4), s2_5(5), s3(6), s3_5(7),
    s4(8), s5(10), s6(12), s7(14), s8(16), s9(18), s10(20),
    s11(22), s12(24), s14(28), s16(32), s20(40), s24(48)
]).

% ============================================================================
% THEME CSS GENERATION
% ============================================================================

%% generate_theme_css(+Theme, -CSS)
%  Generate complete CSS for a theme with custom properties.
generate_theme_css(Theme, CSS) :-
    resolve_theme(Theme, Options),
    generate_color_vars(Options, ColorVars),
    generate_typography_vars(Options, TypoVars),
    generate_spacing_vars(Options, SpacingVars),
    generate_border_vars(Options, BorderVars),
    generate_shadow_vars(Options, ShadowVars),
    generate_transition_vars(Options, TransitionVars),
    atom_string(Theme, ThemeStr),
    format(atom(CSS), '/* Theme: ~w */
[data-theme="~w"] {
  /* Colors */
~w

  /* Typography */
~w

  /* Spacing */
~w

  /* Borders */
~w

  /* Shadows */
~w

  /* Transitions */
~w
}

/* Theme-specific utility classes */
[data-theme="~w"] .bg-primary { background-color: var(--color-primary); }
[data-theme="~w"] .bg-secondary { background-color: var(--color-secondary); }
[data-theme="~w"] .bg-surface { background-color: var(--color-surface); }
[data-theme="~w"] .text-primary { color: var(--color-text-primary); }
[data-theme="~w"] .text-secondary { color: var(--color-text-secondary); }
[data-theme="~w"] .border-default { border-color: var(--color-border); }
', [ThemeStr, ThemeStr, ColorVars, TypoVars, SpacingVars, BorderVars,
    ShadowVars, TransitionVars, ThemeStr, ThemeStr, ThemeStr, ThemeStr,
    ThemeStr, ThemeStr]).

generate_color_vars(Options, Vars) :-
    (member(colors(Colors), Options) -> true ; Colors = []),
    findall(Line, (
        member(Color, Colors),
        Color =.. [Name, Value],
        atom_string(Name, NameStr),
        format(atom(Line), '  --color-~w: ~w;', [NameStr, Value])
    ), Lines),
    atomic_list_concat(Lines, '\n', Vars).

generate_typography_vars(Options, Vars) :-
    (member(typography(Typo), Options) -> true ; Typo = []),
    findall(Line, (
        member(T, Typo),
        T =.. [Name, Value],
        atom_string(Name, NameStr),
        format_typography_value(Value, FormattedValue),
        format(atom(Line), '  --typography-~w: ~w;', [NameStr, FormattedValue])
    ), Lines),
    atomic_list_concat(Lines, '\n', Vars).

format_typography_value(Value, Value) :- atom(Value), !.
format_typography_value(Value, Formatted) :-
    number(Value),
    (Value < 10 -> format(atom(Formatted), '~wpx', [Value]) ; format(atom(Formatted), '~w', [Value])).

generate_spacing_vars(Options, Vars) :-
    (member(spacing(Spacing), Options) -> true ; Spacing = []),
    findall(Line, (
        member(S, Spacing),
        S =.. [Name, Value],
        atom_string(Name, NameStr),
        format(atom(Line), '  --spacing-~w: ~wpx;', [NameStr, Value])
    ), Lines),
    atomic_list_concat(Lines, '\n', Vars).

generate_border_vars(Options, Vars) :-
    (member(borders(Borders), Options) -> true ; Borders = []),
    findall(Line, (
        member(B, Borders),
        B =.. [Name, Value],
        atom_string(Name, NameStr),
        format(atom(Line), '  --border-~w: ~wpx;', [NameStr, Value])
    ), Lines),
    atomic_list_concat(Lines, '\n', Vars).

generate_shadow_vars(Options, Vars) :-
    (member(shadows(Shadows), Options) -> true ; Shadows = []),
    findall(Line, (
        member(S, Shadows),
        S =.. [Name, Value],
        atom_string(Name, NameStr),
        format(atom(Line), '  --shadow-~w: ~w;', [NameStr, Value])
    ), Lines),
    atomic_list_concat(Lines, '\n', Vars).

generate_transition_vars(Options, Vars) :-
    (member(transitions(Transitions), Options) -> true ; Transitions = []),
    findall(Line, (
        member(T, Transitions),
        T =.. [Name, Value],
        atom_string(Name, NameStr),
        format_transition_value(Name, Value, FormattedValue),
        format(atom(Line), '  --transition-~w: ~w;', [NameStr, FormattedValue])
    ), Lines),
    atomic_list_concat(Lines, '\n', Vars).

format_transition_value(Name, Value, Formatted) :-
    atom_string(Name, NameStr),
    sub_string(NameStr, _, _, _, "duration"),
    format(atom(Formatted), '~wms', [Value]), !.
format_transition_value(_, Value, Value).

% ============================================================================
% THEME VARIABLES GENERATION
% ============================================================================

%% generate_theme_variables(+Theme, -Vars)
%  Generate just the CSS custom property declarations.
generate_theme_variables(Theme, Vars) :-
    resolve_theme(Theme, Options),
    generate_color_vars(Options, ColorVars),
    generate_typography_vars(Options, TypoVars),
    generate_spacing_vars(Options, SpacingVars),
    format(atom(Vars), '~w\n~w\n~w', [ColorVars, TypoVars, SpacingVars]).

% ============================================================================
% REACT PROVIDER GENERATION
% ============================================================================

%% generate_theme_provider(+Themes, -JSX)
%  Generate a React theme provider component.
generate_theme_provider(Themes, JSX) :-
    findall(ThemeStr, (member(T, Themes), atom_string(T, ThemeStr)), ThemeStrs),
    atomic_list_concat(ThemeStrs, ' | ', ThemeUnion),
    format(atom(JSX), 'import React, { createContext, useContext, useState, useEffect, ReactNode } from "react";

type ThemeName = "~w";

interface ThemeContextType {
  theme: ThemeName;
  setTheme: (theme: ThemeName) => void;
  toggleTheme: () => void;
  isDark: boolean;
}

const ThemeContext = createContext<ThemeContextType | undefined>(undefined);

interface ThemeProviderProps {
  children: ReactNode;
  defaultTheme?: ThemeName;
  storageKey?: string;
}

export const ThemeProvider: React.FC<ThemeProviderProps> = ({
  children,
  defaultTheme = "light",
  storageKey = "app-theme"
}) => {
  const [theme, setThemeState] = useState<ThemeName>(() => {
    if (typeof window !== "undefined") {
      const stored = localStorage.getItem(storageKey);
      if (stored && isValidTheme(stored)) return stored as ThemeName;

      // Check system preference
      if (window.matchMedia("(prefers-color-scheme: dark)").matches) {
        return "dark";
      }
    }
    return defaultTheme;
  });

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    localStorage.setItem(storageKey, theme);
  }, [theme, storageKey]);

  // Listen for system theme changes
  useEffect(() => {
    const mediaQuery = window.matchMedia("(prefers-color-scheme: dark)");
    const handleChange = (e: MediaQueryListEvent) => {
      const stored = localStorage.getItem(storageKey);
      if (!stored) {
        setThemeState(e.matches ? "dark" : "light");
      }
    };
    mediaQuery.addEventListener("change", handleChange);
    return () => mediaQuery.removeEventListener("change", handleChange);
  }, [storageKey]);

  const setTheme = (newTheme: ThemeName) => {
    setThemeState(newTheme);
  };

  const toggleTheme = () => {
    setThemeState(prev => prev === "dark" ? "light" : "dark");
  };

  const isDark = theme === "dark";

  return (
    <ThemeContext.Provider value={{ theme, setTheme, toggleTheme, isDark }}>
      {children}
    </ThemeContext.Provider>
  );
};

export const useTheme = () => {
  const context = useContext(ThemeContext);
  if (!context) {
    throw new Error("useTheme must be used within a ThemeProvider");
  }
  return context;
};

function isValidTheme(value: string): value is ThemeName {
  return ["~w"].includes(value);
}
', [ThemeUnion, ThemeUnion]).

% ============================================================================
% THEME HOOK GENERATION
% ============================================================================

%% generate_theme_hook(-Hook)
%  Generate a standalone useTheme hook.
generate_theme_hook(Hook) :-
    format(atom(Hook), 'import { useState, useEffect, useCallback } from "react";

type ThemeName = "light" | "dark" | "system";

interface UseThemeResult {
  theme: ThemeName;
  resolvedTheme: "light" | "dark";
  setTheme: (theme: ThemeName) => void;
  toggleTheme: () => void;
  themes: ThemeName[];
}

export const useTheme = (storageKey = "theme"): UseThemeResult => {
  const [theme, setThemeState] = useState<ThemeName>(() => {
    if (typeof window === "undefined") return "system";
    return (localStorage.getItem(storageKey) as ThemeName) || "system";
  });

  const [resolvedTheme, setResolvedTheme] = useState<"light" | "dark">(() => {
    if (typeof window === "undefined") return "light";
    return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  });

  useEffect(() => {
    const root = document.documentElement;
    const mediaQuery = window.matchMedia("(prefers-color-scheme: dark)");

    const updateTheme = () => {
      let resolved: "light" | "dark";
      if (theme === "system") {
        resolved = mediaQuery.matches ? "dark" : "light";
      } else {
        resolved = theme as "light" | "dark";
      }
      setResolvedTheme(resolved);
      root.setAttribute("data-theme", resolved);
    };

    updateTheme();
    localStorage.setItem(storageKey, theme);

    const handleChange = () => {
      if (theme === "system") updateTheme();
    };

    mediaQuery.addEventListener("change", handleChange);
    return () => mediaQuery.removeEventListener("change", handleChange);
  }, [theme, storageKey]);

  const setTheme = useCallback((newTheme: ThemeName) => {
    setThemeState(newTheme);
  }, []);

  const toggleTheme = useCallback(() => {
    setThemeState(prev => {
      if (prev === "light") return "dark";
      if (prev === "dark") return "light";
      return resolvedTheme === "light" ? "dark" : "light";
    });
  }, [resolvedTheme]);

  return {
    theme,
    resolvedTheme,
    setTheme,
    toggleTheme,
    themes: ["light", "dark", "system"]
  };
};
', []).

% ============================================================================
% THEME CONTEXT GENERATION
% ============================================================================

%% generate_theme_context(-Context)
%  Generate theme context with types.
generate_theme_context(Context) :-
    format(atom(Context), 'import { createContext } from "react";

export interface ThemeColors {
  primary: string;
  secondary: string;
  accent: string;
  success: string;
  warning: string;
  error: string;
  background: string;
  surface: string;
  textPrimary: string;
  textSecondary: string;
  textMuted: string;
  border: string;
}

export interface ThemeTypography {
  fontFamily: string;
  fontMono: string;
  baseSize: number;
  lineHeight: number;
}

export interface ThemeSpacing {
  unit: number;
  xs: number;
  sm: number;
  md: number;
  lg: number;
  xl: number;
}

export interface Theme {
  name: string;
  colors: ThemeColors;
  typography: ThemeTypography;
  spacing: ThemeSpacing;
}

export interface ThemeContextValue {
  theme: Theme;
  themeName: string;
  setTheme: (name: string) => void;
  availableThemes: string[];
}

export const ThemeContext = createContext<ThemeContextValue | null>(null);
', []).

% ============================================================================
% THEME TOGGLE GENERATION
% ============================================================================

%% generate_theme_toggle(+Themes, -Toggle)
%  Generate a theme toggle component.
generate_theme_toggle(Themes, Toggle) :-
    findall(Opt, (
        member(T, Themes),
        atom_string(T, TStr),
        format(atom(Opt), '        <option value="~w">~w</option>', [TStr, TStr])
    ), Options),
    atomic_list_concat(Options, '\n', OptionsStr),
    format(atom(Toggle), 'import React from "react";
import { useTheme } from "./useTheme";

interface ThemeToggleProps {
  showLabel?: boolean;
  className?: string;
}

export const ThemeToggle: React.FC<ThemeToggleProps> = ({
  showLabel = true,
  className = ""
}) => {
  const { theme, setTheme, toggleTheme } = useTheme();

  return (
    <div className={`theme-toggle ${className}`}>
      {showLabel && <span className="theme-toggle__label">Theme:</span>}
      <select
        value={theme}
        onChange={(e) => setTheme(e.target.value as any)}
        className="theme-toggle__select"
        aria-label="Select theme"
      >
~w
      </select>
      <button
        onClick={toggleTheme}
        className="theme-toggle__button"
        aria-label="Toggle theme"
      >
        {theme === "dark" ? "☀️" : "🌙"}
      </button>
    </div>
  );
};

export const ThemeToggleCSS = `
.theme-toggle {
  display: flex;
  align-items: center;
  gap: var(--spacing-sm, 8px);
}

.theme-toggle__label {
  font-size: var(--typography-base_size, 14px);
  color: var(--color-text-secondary);
}

.theme-toggle__select {
  padding: var(--spacing-xs, 4px) var(--spacing-sm, 8px);
  border: 1px solid var(--color-border);
  border-radius: var(--border-radius_sm, 4px);
  background: var(--color-surface);
  color: var(--color-text-primary);
  font-size: var(--typography-base_size, 14px);
  cursor: pointer;
}

.theme-toggle__button {
  padding: var(--spacing-xs, 4px) var(--spacing-sm, 8px);
  border: 1px solid var(--color-border);
  border-radius: var(--border-radius_sm, 4px);
  background: var(--color-surface);
  cursor: pointer;
  font-size: 1.2em;
  transition: transform var(--transition-duration_fast, 150ms);
}

.theme-toggle__button:hover {
  transform: scale(1.1);
}
`;
', [OptionsStr]).

% ============================================================================
% TYPE GENERATION
% ============================================================================

%% generate_theme_types(+Themes, -Types)
%  Generate TypeScript types for themes.
generate_theme_types(Themes, Types) :-
    findall(TStr, (member(T, Themes), atom_string(T, TStr)), ThemeStrs),
    atomic_list_concat(ThemeStrs, '" | "', ThemeUnion),
    format(atom(Types), 'export type ThemeName = "~w";

export interface ThemeColors {
  primary: string;
  secondary: string;
  accent: string;
  success: string;
  warning: string;
  error: string;
  background: string;
  surface: string;
  textPrimary: string;
  textSecondary: string;
  textMuted: string;
  border: string;
  shadow: string;
}

export interface ThemeTypography {
  fontFamily: string;
  fontMono: string;
  baseSize: number;
  lineHeight: number;
  headingWeight: number;
  bodyWeight: number;
}

export interface ThemeSpacing {
  unit: number;
  xs: number;
  sm: number;
  md: number;
  lg: number;
  xl: number;
  xxl: number;
}

export interface ThemeBorders {
  radiusSm: number;
  radiusMd: number;
  radiusLg: number;
  radiusFull: number;
  width: number;
}

export interface ThemeShadows {
  sm: string;
  md: string;
  lg: string;
  xl: string;
}

export interface ThemeTransitions {
  durationFast: number;
  durationNormal: number;
  durationSlow: number;
  easing: string;
}

export interface Theme {
  name: ThemeName;
  colors: ThemeColors;
  typography: ThemeTypography;
  spacing: ThemeSpacing;
  borders: ThemeBorders;
  shadows: ThemeShadows;
  transitions: ThemeTransitions;
}

export const themeNames: ThemeName[] = ["~w"];
', [ThemeUnion, ThemeUnion]).

% ============================================================================
% UTILITY PREDICATES
% ============================================================================

%% get_theme_colors(+Theme, -Colors)
%  Get color definitions for a theme.
get_theme_colors(Theme, Colors) :-
    resolve_theme(Theme, Options),
    member(colors(Colors), Options), !.
get_theme_colors(_, []).

%% get_theme_typography(+Theme, -Typography)
%  Get typography definitions for a theme.
get_theme_typography(Theme, Typography) :-
    resolve_theme(Theme, Options),
    member(typography(Typography), Options), !.
get_theme_typography(_, []).

%% get_theme_spacing(+Theme, -Spacing)
%  Get spacing definitions for a theme.
get_theme_spacing(Theme, Spacing) :-
    resolve_theme(Theme, Options),
    member(spacing(Spacing), Options), !.
get_theme_spacing(_, []).

%% resolve_theme(+Theme, -ResolvedOptions)
%  Resolve theme with inheritance.
resolve_theme(Theme, ResolvedOptions) :-
    theme(Theme, Options),
    (member(extends(Parent), Options) ->
        resolve_theme(Parent, ParentOptions),
        merge_theme_options(ParentOptions, Options, ResolvedOptions)
    ;
        ResolvedOptions = Options
    ).

merge_theme_options(Parent, Child, Merged) :-
    findall(Opt, (
        member(Opt, Child),
        Opt \= extends(_)
    ), ChildOpts),
    findall(POpt, (
        member(POpt, Parent),
        POpt =.. [Key|_],
        \+ (member(COpt, ChildOpts), COpt =.. [Key|_])
    ), InheritedOpts),
    append(ChildOpts, InheritedOpts, Merged).

% ============================================================================
% MANAGEMENT PREDICATES
% ============================================================================

%% declare_theme(+Name, +Options)
%  Declare a new theme.
declare_theme(Name, Options) :-
    retractall(theme(Name, _)),
    assertz(theme(Name, Options)).

%% clear_themes/0
%  Clear all dynamic themes.
clear_themes :-
    retractall(theme(_, _)),
    retractall(theme_extends(_, _)).

% ============================================================================
% TESTING
% ============================================================================

test_theme_generator :-
    writeln('Testing theme generator...'),

    % Test theme existence
    (theme(light, _) -> writeln('  [PASS] light theme exists') ; writeln('  [FAIL] light theme')),
    (theme(dark, _) -> writeln('  [PASS] dark theme exists') ; writeln('  [FAIL] dark theme')),

    % Test color palettes
    (color_palette(blue, _) -> writeln('  [PASS] blue palette exists') ; writeln('  [FAIL] blue palette')),

    % Test theme resolution
    (resolve_theme(light, Opts), member(colors(_), Opts) ->
        writeln('  [PASS] resolve_theme returns colors') ;
        writeln('  [FAIL] resolve_theme')),

    % Test CSS generation
    (generate_theme_css(light, CSS), atom_length(CSS, L), L > 500 ->
        writeln('  [PASS] generate_theme_css produces CSS') ;
        writeln('  [FAIL] generate_theme_css')),

    % Test provider generation
    (generate_theme_provider([light, dark], Provider), atom_length(Provider, PL), PL > 500 ->
        writeln('  [PASS] generate_theme_provider produces code') ;
        writeln('  [FAIL] generate_theme_provider')),

    % Test hook generation
    (generate_theme_hook(Hook), atom_length(Hook, HL), HL > 500 ->
        writeln('  [PASS] generate_theme_hook produces code') ;
        writeln('  [FAIL] generate_theme_hook')),

    writeln('Theme generator tests complete.').
