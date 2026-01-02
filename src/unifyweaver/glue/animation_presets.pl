% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% Animation Presets - Library of Reusable Animation Patterns
%
% This module provides a curated library of animation presets that can
% be applied to visualization components.
%
% Usage:
%   % Apply a preset animation
%   animation_preset(my_chart, fade_in, [duration(300)]).
%
%   % Generate animation CSS
%   ?- generate_preset_css(fade_in, CSS).

:- module(animation_presets, [
    % Preset definitions
    preset/2,                       % preset(+Name, +Definition)
    preset_category/2,              % preset_category(+Name, +Category)

    % Preset queries
    get_preset/2,                   % get_preset(+Name, -Definition)
    list_presets/1,                 % list_presets(-Presets)
    list_presets_by_category/2,     % list_presets_by_category(+Category, -Presets)

    % Generation predicates
    generate_preset_css/2,          % generate_preset_css(+Preset, -CSS)
    generate_preset_keyframes/2,    % generate_preset_keyframes(+Preset, -Keyframes)
    generate_preset_class/3,        % generate_preset_class(+Preset, +Options, -Class)
    generate_all_presets_css/1,     % generate_all_presets_css(-CSS)
    generate_preset_hook/1,         % generate_preset_hook(-Hook)
    generate_preset_component/2,    % generate_preset_component(+Preset, -Component)

    % Composition
    compose_presets/3,              % compose_presets(+Presets, +Options, -Combined)
    sequence_presets/3,             % sequence_presets(+Presets, +Options, -Sequence)

    % Utility
    preset_duration/2,              % preset_duration(+Preset, -Duration)
    preset_easing/2,                % preset_easing(+Preset, -Easing)

    % Management
    declare_preset/2,               % declare_preset(+Name, +Definition)
    clear_presets/0,                % clear_presets

    % Testing
    test_animation_presets/0
]).

:- use_module(library(lists)).

% ============================================================================
% DYNAMIC PREDICATES
% ============================================================================

:- dynamic preset/2.
:- dynamic preset_category/2.

:- discontiguous preset/2.
:- discontiguous preset_category/2.

% ============================================================================
% ENTRY ANIMATIONS
% ============================================================================

preset_category(fade_in, entry).
preset(fade_in, [
    keyframes([
        frame(0, [opacity(0)]),
        frame(100, [opacity(1)])
    ]),
    duration(300),
    easing('ease-out'),
    fill_mode(forwards)
]).

preset_category(fade_in_up, entry).
preset(fade_in_up, [
    keyframes([
        frame(0, [opacity(0), transform('translateY(20px)')]),
        frame(100, [opacity(1), transform('translateY(0)')])
    ]),
    duration(400),
    easing('ease-out'),
    fill_mode(forwards)
]).

preset_category(fade_in_down, entry).
preset(fade_in_down, [
    keyframes([
        frame(0, [opacity(0), transform('translateY(-20px)')]),
        frame(100, [opacity(1), transform('translateY(0)')])
    ]),
    duration(400),
    easing('ease-out'),
    fill_mode(forwards)
]).

preset_category(fade_in_left, entry).
preset(fade_in_left, [
    keyframes([
        frame(0, [opacity(0), transform('translateX(-20px)')]),
        frame(100, [opacity(1), transform('translateX(0)')])
    ]),
    duration(400),
    easing('ease-out'),
    fill_mode(forwards)
]).

preset_category(fade_in_right, entry).
preset(fade_in_right, [
    keyframes([
        frame(0, [opacity(0), transform('translateX(20px)')]),
        frame(100, [opacity(1), transform('translateX(0)')])
    ]),
    duration(400),
    easing('ease-out'),
    fill_mode(forwards)
]).

preset_category(slide_in_up, entry).
preset(slide_in_up, [
    keyframes([
        frame(0, [transform('translateY(100%)')]),
        frame(100, [transform('translateY(0)')])
    ]),
    duration(500),
    easing('cubic-bezier(0.16, 1, 0.3, 1)'),
    fill_mode(forwards)
]).

preset_category(slide_in_down, entry).
preset(slide_in_down, [
    keyframes([
        frame(0, [transform('translateY(-100%)')]),
        frame(100, [transform('translateY(0)')])
    ]),
    duration(500),
    easing('cubic-bezier(0.16, 1, 0.3, 1)'),
    fill_mode(forwards)
]).

preset_category(slide_in_left, entry).
preset(slide_in_left, [
    keyframes([
        frame(0, [transform('translateX(-100%)')]),
        frame(100, [transform('translateX(0)')])
    ]),
    duration(500),
    easing('cubic-bezier(0.16, 1, 0.3, 1)'),
    fill_mode(forwards)
]).

preset_category(slide_in_right, entry).
preset(slide_in_right, [
    keyframes([
        frame(0, [transform('translateX(100%)')]),
        frame(100, [transform('translateX(0)')])
    ]),
    duration(500),
    easing('cubic-bezier(0.16, 1, 0.3, 1)'),
    fill_mode(forwards)
]).

preset_category(scale_in, entry).
preset(scale_in, [
    keyframes([
        frame(0, [opacity(0), transform('scale(0.9)')]),
        frame(100, [opacity(1), transform('scale(1)')])
    ]),
    duration(300),
    easing('cubic-bezier(0.16, 1, 0.3, 1)'),
    fill_mode(forwards)
]).

preset_category(scale_in_center, entry).
preset(scale_in_center, [
    keyframes([
        frame(0, [opacity(0), transform('scale(0)')]),
        frame(100, [opacity(1), transform('scale(1)')])
    ]),
    duration(400),
    easing('cubic-bezier(0.34, 1.56, 0.64, 1)'),
    fill_mode(forwards)
]).

preset_category(zoom_in, entry).
preset(zoom_in, [
    keyframes([
        frame(0, [opacity(0), transform('scale(0.3)')]),
        frame(50, [opacity(1)]),
        frame(100, [transform('scale(1)')])
    ]),
    duration(500),
    easing('cubic-bezier(0.16, 1, 0.3, 1)'),
    fill_mode(forwards)
]).

preset_category(bounce_in, entry).
preset(bounce_in, [
    keyframes([
        frame(0, [opacity(0), transform('scale(0.3)')]),
        frame(50, [transform('scale(1.05)')]),
        frame(70, [transform('scale(0.9)')]),
        frame(100, [opacity(1), transform('scale(1)')])
    ]),
    duration(600),
    easing('cubic-bezier(0.34, 1.56, 0.64, 1)'),
    fill_mode(forwards)
]).

preset_category(flip_in_x, entry).
preset(flip_in_x, [
    keyframes([
        frame(0, [opacity(0), transform('perspective(400px) rotateX(90deg)')]),
        frame(40, [transform('perspective(400px) rotateX(-10deg)')]),
        frame(70, [transform('perspective(400px) rotateX(10deg)')]),
        frame(100, [opacity(1), transform('perspective(400px) rotateX(0)')])
    ]),
    duration(600),
    easing('ease-in-out'),
    fill_mode(forwards)
]).

preset_category(flip_in_y, entry).
preset(flip_in_y, [
    keyframes([
        frame(0, [opacity(0), transform('perspective(400px) rotateY(90deg)')]),
        frame(40, [transform('perspective(400px) rotateY(-10deg)')]),
        frame(70, [transform('perspective(400px) rotateY(10deg)')]),
        frame(100, [opacity(1), transform('perspective(400px) rotateY(0)')])
    ]),
    duration(600),
    easing('ease-in-out'),
    fill_mode(forwards)
]).

% ============================================================================
% EXIT ANIMATIONS
% ============================================================================

preset_category(fade_out, exit).
preset(fade_out, [
    keyframes([
        frame(0, [opacity(1)]),
        frame(100, [opacity(0)])
    ]),
    duration(300),
    easing('ease-in'),
    fill_mode(forwards)
]).

preset_category(fade_out_up, exit).
preset(fade_out_up, [
    keyframes([
        frame(0, [opacity(1), transform('translateY(0)')]),
        frame(100, [opacity(0), transform('translateY(-20px)')])
    ]),
    duration(400),
    easing('ease-in'),
    fill_mode(forwards)
]).

preset_category(fade_out_down, exit).
preset(fade_out_down, [
    keyframes([
        frame(0, [opacity(1), transform('translateY(0)')]),
        frame(100, [opacity(0), transform('translateY(20px)')])
    ]),
    duration(400),
    easing('ease-in'),
    fill_mode(forwards)
]).

preset_category(scale_out, exit).
preset(scale_out, [
    keyframes([
        frame(0, [opacity(1), transform('scale(1)')]),
        frame(100, [opacity(0), transform('scale(0.9)')])
    ]),
    duration(300),
    easing('ease-in'),
    fill_mode(forwards)
]).

preset_category(zoom_out, exit).
preset(zoom_out, [
    keyframes([
        frame(0, [opacity(1), transform('scale(1)')]),
        frame(50, [opacity(0.5), transform('scale(1.1)')]),
        frame(100, [opacity(0), transform('scale(0.3)')])
    ]),
    duration(400),
    easing('ease-in'),
    fill_mode(forwards)
]).

% ============================================================================
% ATTENTION ANIMATIONS
% ============================================================================

preset_category(pulse, attention).
preset(pulse, [
    keyframes([
        frame(0, [transform('scale(1)')]),
        frame(50, [transform('scale(1.05)')]),
        frame(100, [transform('scale(1)')])
    ]),
    duration(600),
    easing('ease-in-out'),
    iteration_count(infinite)
]).

preset_category(bounce, attention).
preset(bounce, [
    keyframes([
        frame(0, [transform('translateY(0)')]),
        frame(50, [transform('translateY(-10px)')]),
        frame(100, [transform('translateY(0)')])
    ]),
    duration(500),
    easing('cubic-bezier(0.34, 1.56, 0.64, 1)'),
    iteration_count(infinite)
]).

preset_category(shake, attention).
preset(shake, [
    keyframes([
        frame(0, [transform('translateX(0)')]),
        frame(25, [transform('translateX(-5px)')]),
        frame(50, [transform('translateX(5px)')]),
        frame(75, [transform('translateX(-5px)')]),
        frame(100, [transform('translateX(0)')])
    ]),
    duration(400),
    easing('ease-in-out'),
    iteration_count(1)
]).

preset_category(wiggle, attention).
preset(wiggle, [
    keyframes([
        frame(0, [transform('rotate(0deg)')]),
        frame(25, [transform('rotate(-3deg)')]),
        frame(50, [transform('rotate(3deg)')]),
        frame(75, [transform('rotate(-3deg)')]),
        frame(100, [transform('rotate(0deg)')])
    ]),
    duration(300),
    easing('ease-in-out'),
    iteration_count(1)
]).

preset_category(flash, attention).
preset(flash, [
    keyframes([
        frame(0, [opacity(1)]),
        frame(25, [opacity(0)]),
        frame(50, [opacity(1)]),
        frame(75, [opacity(0)]),
        frame(100, [opacity(1)])
    ]),
    duration(800),
    easing('linear'),
    iteration_count(1)
]).

preset_category(heartbeat, attention).
preset(heartbeat, [
    keyframes([
        frame(0, [transform('scale(1)')]),
        frame(14, [transform('scale(1.3)')]),
        frame(28, [transform('scale(1)')]),
        frame(42, [transform('scale(1.3)')]),
        frame(70, [transform('scale(1)')])
    ]),
    duration(1000),
    easing('ease-in-out'),
    iteration_count(infinite)
]).

preset_category(jello, attention).
preset(jello, [
    keyframes([
        frame(0, [transform('skewX(0deg) skewY(0deg)')]),
        frame(22, [transform('skewX(-12.5deg) skewY(-12.5deg)')]),
        frame(33, [transform('skewX(6.25deg) skewY(6.25deg)')]),
        frame(44, [transform('skewX(-3.125deg) skewY(-3.125deg)')]),
        frame(55, [transform('skewX(1.5625deg) skewY(1.5625deg)')]),
        frame(66, [transform('skewX(-0.78125deg) skewY(-0.78125deg)')]),
        frame(77, [transform('skewX(0.390625deg) skewY(0.390625deg)')]),
        frame(88, [transform('skewX(-0.1953125deg) skewY(-0.1953125deg)')]),
        frame(100, [transform('skewX(0deg) skewY(0deg)')])
    ]),
    duration(1000),
    easing('ease-in-out'),
    iteration_count(1)
]).

% ============================================================================
% TRANSITION PRESETS
% ============================================================================

preset_category(smooth, transition).
preset(smooth, [
    properties([all]),
    duration(300),
    easing('cubic-bezier(0.4, 0, 0.2, 1)')
]).

preset_category(snappy, transition).
preset(snappy, [
    properties([all]),
    duration(150),
    easing('cubic-bezier(0.4, 0, 1, 1)')
]).

preset_category(elastic, transition).
preset(elastic, [
    properties([transform]),
    duration(500),
    easing('cubic-bezier(0.68, -0.55, 0.265, 1.55)')
]).

preset_category(spring, transition).
preset(spring, [
    properties([transform, opacity]),
    duration(400),
    easing('cubic-bezier(0.34, 1.56, 0.64, 1)')
]).

% ============================================================================
% CHART-SPECIFIC PRESETS
% ============================================================================

preset_category(chart_draw, chart).
preset(chart_draw, [
    keyframes([
        frame(0, [stroke_dashoffset(1000)]),
        frame(100, [stroke_dashoffset(0)])
    ]),
    duration(1500),
    easing('ease-out'),
    fill_mode(forwards)
]).

preset_category(bar_grow, chart).
preset(bar_grow, [
    keyframes([
        frame(0, [transform('scaleY(0)'), transform_origin('bottom')]),
        frame(100, [transform('scaleY(1)'), transform_origin('bottom')])
    ]),
    duration(600),
    easing('cubic-bezier(0.16, 1, 0.3, 1)'),
    fill_mode(forwards)
]).

preset_category(pie_reveal, chart).
preset(pie_reveal, [
    keyframes([
        frame(0, [stroke_dasharray('0 100')]),
        frame(100, [stroke_dasharray('100 0')])
    ]),
    duration(1000),
    easing('ease-out'),
    fill_mode(forwards)
]).

preset_category(data_point_pop, chart).
preset(data_point_pop, [
    keyframes([
        frame(0, [transform('scale(0)'), opacity(0)]),
        frame(60, [transform('scale(1.2)')]),
        frame(100, [transform('scale(1)'), opacity(1)])
    ]),
    duration(400),
    easing('cubic-bezier(0.34, 1.56, 0.64, 1)'),
    fill_mode(forwards)
]).

preset_category(tooltip_appear, chart).
preset(tooltip_appear, [
    keyframes([
        frame(0, [opacity(0), transform('translateY(5px) scale(0.95)')]),
        frame(100, [opacity(1), transform('translateY(0) scale(1)')])
    ]),
    duration(200),
    easing('cubic-bezier(0.16, 1, 0.3, 1)'),
    fill_mode(forwards)
]).

% ============================================================================
% CSS GENERATION
% ============================================================================

%% generate_preset_css(+Preset, -CSS)
%  Generate CSS for a single preset.
generate_preset_css(Preset, CSS) :-
    preset(Preset, Definition),
    atom_string(Preset, PresetStr),
    (member(keyframes(Keyframes), Definition) ->
        generate_keyframes_css(Preset, Keyframes, KeyframeCSS),
        generate_animation_class(Preset, Definition, ClassCSS),
        format(atom(CSS), '~w\n\n~w', [KeyframeCSS, ClassCSS])
    ;
        % Transition preset
        generate_transition_class(Preset, Definition, CSS)
    ).

generate_keyframes_css(Preset, Keyframes, CSS) :-
    atom_string(Preset, PresetStr),
    findall(FrameCSS, (
        member(frame(Percent, Props), Keyframes),
        generate_frame_css(Percent, Props, FrameCSS)
    ), FrameList),
    atomic_list_concat(FrameList, '\n', FramesStr),
    format(atom(CSS), '@keyframes ~w {\n~w\n}', [PresetStr, FramesStr]).

generate_frame_css(Percent, Props, CSS) :-
    findall(PropCSS, (
        member(Prop, Props),
        Prop =.. [Name, Value],
        css_property_name(Name, CSSName),
        format(atom(PropCSS), '    ~w: ~w;', [CSSName, Value])
    ), PropList),
    atomic_list_concat(PropList, '\n', PropsStr),
    format(atom(CSS), '  ~w% {\n~w\n  }', [Percent, PropsStr]).

css_property_name(opacity, 'opacity') :- !.
css_property_name(transform, 'transform') :- !.
css_property_name(transform_origin, 'transform-origin') :- !.
css_property_name(stroke_dashoffset, 'stroke-dashoffset') :- !.
css_property_name(stroke_dasharray, 'stroke-dasharray') :- !.
css_property_name(Name, CSSName) :-
    atom_string(Name, NameStr),
    string_replace(NameStr, "_", "-", CSSName).

string_replace(String, From, To, Result) :-
    split_string(String, From, "", Parts),
    atomic_list_concat(Parts, To, Result).

generate_animation_class(Preset, Definition, CSS) :-
    atom_string(Preset, PresetStr),
    (member(duration(D), Definition) -> true ; D = 300),
    (member(easing(E), Definition) -> true ; E = 'ease'),
    (member(fill_mode(F), Definition) -> true ; F = 'none'),
    (member(iteration_count(I), Definition) -> true ; I = 1),
    format(atom(CSS), '.animate-~w {
  animation-name: ~w;
  animation-duration: ~wms;
  animation-timing-function: ~w;
  animation-fill-mode: ~w;
  animation-iteration-count: ~w;
}', [PresetStr, PresetStr, D, E, F, I]).

generate_transition_class(Preset, Definition, CSS) :-
    atom_string(Preset, PresetStr),
    (member(properties(Props), Definition) -> true ; Props = [all]),
    (member(duration(D), Definition) -> true ; D = 300),
    (member(easing(E), Definition) -> true ; E = 'ease'),
    findall(P, member(P, Props), PropList),
    atomic_list_concat(PropList, ', ', PropsStr),
    format(atom(CSS), '.transition-~w {
  transition-property: ~w;
  transition-duration: ~wms;
  transition-timing-function: ~w;
}', [PresetStr, PropsStr, D, E]).

%% generate_preset_keyframes(+Preset, -Keyframes)
%  Generate just the @keyframes rule.
generate_preset_keyframes(Preset, Keyframes) :-
    preset(Preset, Definition),
    member(keyframes(KFs), Definition),
    generate_keyframes_css(Preset, KFs, Keyframes).

%% generate_preset_class(+Preset, +Options, -Class)
%  Generate animation class with custom options.
generate_preset_class(Preset, Options, Class) :-
    preset(Preset, Definition),
    atom_string(Preset, PresetStr),
    (member(duration(D), Options) -> true ;
     member(duration(D), Definition) -> true ; D = 300),
    (member(easing(E), Options) -> true ;
     member(easing(E), Definition) -> true ; E = 'ease'),
    (member(delay(Del), Options) -> DelayStr = Del ; DelayStr = 0),
    format(atom(Class), '.animate-~w-custom {
  animation-name: ~w;
  animation-duration: ~wms;
  animation-timing-function: ~w;
  animation-delay: ~wms;
  animation-fill-mode: forwards;
}', [PresetStr, PresetStr, D, E, DelayStr]).

%% generate_all_presets_css(-CSS)
%  Generate CSS for all presets.
generate_all_presets_css(CSS) :-
    findall(PresetCSS, (
        preset(P, _),
        generate_preset_css(P, PresetCSS)
    ), AllCSS),
    atomic_list_concat(AllCSS, '\n\n', CSS).

% ============================================================================
% REACT HOOK GENERATION
% ============================================================================

%% generate_preset_hook(-Hook)
%  Generate useAnimation hook for applying presets.
generate_preset_hook(Hook) :-
    format(atom(Hook), 'import { useState, useCallback, useRef, useEffect } from "react";

type AnimationPreset =
  | "fade_in" | "fade_in_up" | "fade_in_down" | "fade_in_left" | "fade_in_right"
  | "slide_in_up" | "slide_in_down" | "slide_in_left" | "slide_in_right"
  | "scale_in" | "scale_in_center" | "zoom_in" | "bounce_in"
  | "flip_in_x" | "flip_in_y"
  | "fade_out" | "fade_out_up" | "fade_out_down" | "scale_out" | "zoom_out"
  | "pulse" | "bounce" | "shake" | "wiggle" | "flash" | "heartbeat" | "jello"
  | "chart_draw" | "bar_grow" | "pie_reveal" | "data_point_pop" | "tooltip_appear";

interface AnimationOptions {
  duration?: number;
  delay?: number;
  easing?: string;
  onComplete?: () => void;
}

interface UseAnimationResult {
  ref: React.RefObject<HTMLElement>;
  animate: (preset: AnimationPreset, options?: AnimationOptions) => void;
  isAnimating: boolean;
  stop: () => void;
}

export const useAnimation = (): UseAnimationResult => {
  const ref = useRef<HTMLElement>(null);
  const [isAnimating, setIsAnimating] = useState(false);
  const animationRef = useRef<Animation | null>(null);

  const animate = useCallback((preset: AnimationPreset, options: AnimationOptions = {}) => {
    if (!ref.current) return;

    const element = ref.current;
    const className = `animate-${preset.replace(/_/g, "-")}`;

    // Apply custom duration if provided
    if (options.duration) {
      element.style.animationDuration = `${options.duration}ms`;
    }
    if (options.delay) {
      element.style.animationDelay = `${options.delay}ms`;
    }
    if (options.easing) {
      element.style.animationTimingFunction = options.easing;
    }

    // Remove existing animation classes
    element.classList.forEach(cls => {
      if (cls.startsWith("animate-")) {
        element.classList.remove(cls);
      }
    });

    // Trigger reflow
    void element.offsetWidth;

    // Add animation class
    element.classList.add(className);
    setIsAnimating(true);

    const handleEnd = () => {
      setIsAnimating(false);
      options.onComplete?.();
      element.removeEventListener("animationend", handleEnd);
    };

    element.addEventListener("animationend", handleEnd);
  }, []);

  const stop = useCallback(() => {
    if (!ref.current) return;
    ref.current.style.animationPlayState = "paused";
    setIsAnimating(false);
  }, []);

  return { ref, animate, isAnimating, stop };
};

// Utility hook for staggered animations
export const useStaggeredAnimation = (
  count: number,
  preset: AnimationPreset,
  staggerDelay = 50
) => {
  const refs = useRef<(HTMLElement | null)[]>([]);

  const setRef = useCallback((index: number) => (el: HTMLElement | null) => {
    refs.current[index] = el;
  }, []);

  const animateAll = useCallback(() => {
    refs.current.forEach((el, index) => {
      if (el) {
        setTimeout(() => {
          el.classList.add(`animate-${preset.replace(/_/g, "-")}`);
        }, index * staggerDelay);
      }
    });
  }, [preset, staggerDelay]);

  return { setRef, animateAll };
};
', []).

% ============================================================================
% COMPONENT GENERATION
% ============================================================================

%% generate_preset_component(+Preset, -Component)
%  Generate an animated component wrapper.
generate_preset_component(Preset, Component) :-
    atom_string(Preset, PresetStr),
    format(atom(Component), 'import React, { forwardRef, useEffect, useRef, useState } from "react";

interface Animated~wProps extends React.HTMLAttributes<HTMLDivElement> {
  children: React.ReactNode;
  trigger?: boolean | "onMount" | "onHover" | "onClick";
  duration?: number;
  delay?: number;
  onAnimationEnd?: () => void;
}

export const Animated~w = forwardRef<HTMLDivElement, Animated~wProps>(
  ({ children, trigger = "onMount", duration, delay = 0, onAnimationEnd, className = "", style, ...props }, ref) => {
    const innerRef = useRef<HTMLDivElement>(null);
    const [isAnimating, setIsAnimating] = useState(false);

    useEffect(() => {
      if (trigger === "onMount" || trigger === true) {
        const timer = setTimeout(() => setIsAnimating(true), delay);
        return () => clearTimeout(timer);
      }
    }, [trigger, delay]);

    const handleAnimationEnd = () => {
      onAnimationEnd?.();
    };

    const handleHover = () => {
      if (trigger === "onHover") setIsAnimating(true);
    };

    const handleClick = () => {
      if (trigger === "onClick") {
        setIsAnimating(false);
        requestAnimationFrame(() => setIsAnimating(true));
      }
    };

    const animationStyle: React.CSSProperties = {
      ...style,
      ...(duration && { animationDuration: `${duration}ms` }),
    };

    return (
      <div
        ref={ref || innerRef}
        className={`${isAnimating ? "animate-~w" : ""} ${className}`}
        style={animationStyle}
        onAnimationEnd={handleAnimationEnd}
        onMouseEnter={handleHover}
        onClick={handleClick}
        {...props}
      >
        {children}
      </div>
    );
  }
);

Animated~w.displayName = "Animated~w";
', [PresetStr, PresetStr, PresetStr, PresetStr, PresetStr, PresetStr]).

% ============================================================================
% COMPOSITION
% ============================================================================

%% compose_presets(+Presets, +Options, -Combined)
%  Combine multiple presets into one.
compose_presets(Presets, Options, Combined) :-
    findall(KF, (
        member(P, Presets),
        preset(P, Def),
        member(keyframes(KF), Def)
    ), AllKeyframes),
    flatten(AllKeyframes, FlatKeyframes),
    (member(duration(D), Options) -> true ; D = 500),
    (member(easing(E), Options) -> true ; E = 'ease'),
    Combined = [
        keyframes(FlatKeyframes),
        duration(D),
        easing(E),
        fill_mode(forwards)
    ].

%% sequence_presets(+Presets, +Options, -Sequence)
%  Create a sequence of animations.
sequence_presets(Presets, Options, Sequence) :-
    (member(stagger(S), Options) -> true ; S = 100),
    findall(seq(P, Delay), (
        nth0(I, Presets, P),
        Delay is I * S
    ), Sequence).

% ============================================================================
% UTILITY PREDICATES
% ============================================================================

%% get_preset(+Name, -Definition)
%  Get preset definition.
get_preset(Name, Definition) :-
    preset(Name, Definition).

%% list_presets(-Presets)
%  List all available presets.
list_presets(Presets) :-
    findall(P, preset(P, _), Presets).

%% list_presets_by_category(+Category, -Presets)
%  List presets in a category.
list_presets_by_category(Category, Presets) :-
    findall(P, preset_category(P, Category), Presets).

%% preset_duration(+Preset, -Duration)
%  Get preset duration.
preset_duration(Preset, Duration) :-
    preset(Preset, Def),
    member(duration(Duration), Def), !.
preset_duration(_, 300).

%% preset_easing(+Preset, -Easing)
%  Get preset easing.
preset_easing(Preset, Easing) :-
    preset(Preset, Def),
    member(easing(Easing), Def), !.
preset_easing(_, 'ease').

% ============================================================================
% MANAGEMENT PREDICATES
% ============================================================================

%% declare_preset(+Name, +Definition)
%  Declare a custom preset.
declare_preset(Name, Definition) :-
    retractall(preset(Name, _)),
    assertz(preset(Name, Definition)).

%% clear_presets/0
%  Clear custom presets.
clear_presets :-
    retractall(preset(_, _)),
    retractall(preset_category(_, _)).

% ============================================================================
% TESTING
% ============================================================================

test_animation_presets :-
    writeln('Testing animation presets...'),

    % Test preset existence
    (preset(fade_in, _) -> writeln('  [PASS] fade_in preset exists') ; writeln('  [FAIL] fade_in')),
    (preset(bounce, _) -> writeln('  [PASS] bounce preset exists') ; writeln('  [FAIL] bounce')),

    % Test categories
    (preset_category(fade_in, entry) -> writeln('  [PASS] fade_in is entry') ; writeln('  [FAIL] category')),
    (list_presets_by_category(entry, EntryList), length(EntryList, EL), EL > 5 ->
        writeln('  [PASS] has entry presets') ; writeln('  [FAIL] entry list')),

    % Test CSS generation
    (generate_preset_css(fade_in, CSS), atom_length(CSS, L), L > 100 ->
        writeln('  [PASS] generate_preset_css produces CSS') ;
        writeln('  [FAIL] generate_preset_css')),

    % Test all presets CSS
    (generate_all_presets_css(AllCSS), atom_length(AllCSS, AL), AL > 1000 ->
        writeln('  [PASS] generate_all_presets_css works') ;
        writeln('  [FAIL] generate_all_presets_css')),

    % Test hook generation
    (generate_preset_hook(Hook), atom_length(Hook, HL), HL > 500 ->
        writeln('  [PASS] generate_preset_hook works') ;
        writeln('  [FAIL] generate_preset_hook')),

    writeln('Animation presets tests complete.').
