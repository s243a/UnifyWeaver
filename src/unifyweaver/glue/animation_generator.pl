% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% Animation Generator - Declarative Animations and Transitions
%
% This module provides declarative animation specifications that generate
% CSS keyframes, transitions, and animation properties.
%
% Usage:
%   % Define a keyframe animation
%   animation(fade_in, [
%       duration(300),
%       easing(ease_out),
%       keyframes([
%           frame(0, [opacity(0)]),
%           frame(100, [opacity(1)])
%       ])
%   ]).
%
%   % Define a transition
%   transition(button_hover, [
%       property(transform),
%       duration(200),
%       easing(ease_in_out)
%   ]).
%
%   % Generate CSS
%   ?- generate_animation_css(fade_in, CSS).

:- module(animation_generator, [
    % Animation definitions
    animation/2,                    % animation(+Name, +Options)
    transition/2,                   % transition(+Name, +Options)

    % Easing functions
    easing/2,                       % easing(+Name, +CSSValue)

    % Generation predicates
    generate_animation_css/2,       % generate_animation_css(+Name, -CSS)
    generate_keyframes_css/2,       % generate_keyframes_css(+Name, -CSS)
    generate_transition_css/2,      % generate_transition_css(+Name, -CSS)
    generate_animation_class/3,     % generate_animation_class(+Name, +ClassName, -CSS)
    generate_all_animations_css/1,  % generate_all_animations_css(-CSS)

    % Animation utilities
    animation_property/3,           % animation_property(+Animation, +Property, -Value)
    get_animation_duration/2,       % get_animation_duration(+Animation, -Duration)
    get_animation_easing/2,         % get_animation_easing(+Animation, -Easing)

    % React/JSX generation
    generate_animation_hook/2,      % generate_animation_hook(+Name, -Hook)
    generate_animation_component/2, % generate_animation_component(+Name, -JSX)

    % Management
    declare_animation/2,            % declare_animation(+Name, +Options)
    declare_transition/2,           % declare_transition(+Name, +Options)
    clear_animations/0,             % clear_animations

    % Testing
    test_animation_generator/0
]).

:- use_module(library(lists)).

% ============================================================================
% DYNAMIC PREDICATES
% ============================================================================

:- dynamic animation/2.
:- dynamic transition/2.

:- discontiguous animation/2.
:- discontiguous transition/2.

% ============================================================================
% EASING FUNCTIONS
% ============================================================================

% Standard easing functions
easing(linear, 'linear').
easing(ease, 'ease').
easing(ease_in, 'ease-in').
easing(ease_out, 'ease-out').
easing(ease_in_out, 'ease-in-out').

% Cubic bezier easings
easing(ease_in_quad, 'cubic-bezier(0.55, 0.085, 0.68, 0.53)').
easing(ease_out_quad, 'cubic-bezier(0.25, 0.46, 0.45, 0.94)').
easing(ease_in_out_quad, 'cubic-bezier(0.455, 0.03, 0.515, 0.955)').

easing(ease_in_cubic, 'cubic-bezier(0.55, 0.055, 0.675, 0.19)').
easing(ease_out_cubic, 'cubic-bezier(0.215, 0.61, 0.355, 1)').
easing(ease_in_out_cubic, 'cubic-bezier(0.645, 0.045, 0.355, 1)').

easing(ease_in_quart, 'cubic-bezier(0.895, 0.03, 0.685, 0.22)').
easing(ease_out_quart, 'cubic-bezier(0.165, 0.84, 0.44, 1)').
easing(ease_in_out_quart, 'cubic-bezier(0.77, 0, 0.175, 1)').

easing(ease_in_back, 'cubic-bezier(0.6, -0.28, 0.735, 0.045)').
easing(ease_out_back, 'cubic-bezier(0.175, 0.885, 0.32, 1.275)').
easing(ease_in_out_back, 'cubic-bezier(0.68, -0.55, 0.265, 1.55)').

easing(ease_in_elastic, 'cubic-bezier(0.5, -0.25, 0.25, 1.5)').
easing(ease_out_elastic, 'cubic-bezier(0.75, -0.5, 0.5, 1.25)').

% Spring-like animations
easing(spring, 'cubic-bezier(0.5, 1.6, 0.4, 0.7)').
easing(bounce, 'cubic-bezier(0.68, -0.55, 0.265, 1.55)').

% ============================================================================
% DEFAULT ANIMATIONS
% ============================================================================

% Fade animations
animation(fade_in, [
    duration(300),
    easing(ease_out),
    fill_mode(forwards),
    keyframes([
        frame(0, [opacity(0)]),
        frame(100, [opacity(1)])
    ])
]).

animation(fade_out, [
    duration(300),
    easing(ease_in),
    fill_mode(forwards),
    keyframes([
        frame(0, [opacity(1)]),
        frame(100, [opacity(0)])
    ])
]).

animation(fade_in_up, [
    duration(400),
    easing(ease_out),
    fill_mode(forwards),
    keyframes([
        frame(0, [opacity(0), transform('translateY(20px)')]),
        frame(100, [opacity(1), transform('translateY(0)')])
    ])
]).

animation(fade_in_down, [
    duration(400),
    easing(ease_out),
    fill_mode(forwards),
    keyframes([
        frame(0, [opacity(0), transform('translateY(-20px)')]),
        frame(100, [opacity(1), transform('translateY(0)')])
    ])
]).

% Scale animations
animation(scale_in, [
    duration(300),
    easing(ease_out_back),
    fill_mode(forwards),
    keyframes([
        frame(0, [opacity(0), transform('scale(0.9)')]),
        frame(100, [opacity(1), transform('scale(1)')])
    ])
]).

animation(scale_out, [
    duration(200),
    easing(ease_in),
    fill_mode(forwards),
    keyframes([
        frame(0, [opacity(1), transform('scale(1)')]),
        frame(100, [opacity(0), transform('scale(0.9)')])
    ])
]).

animation(pop_in, [
    duration(400),
    easing(spring),
    fill_mode(forwards),
    keyframes([
        frame(0, [opacity(0), transform('scale(0.5)')]),
        frame(70, [transform('scale(1.05)')]),
        frame(100, [opacity(1), transform('scale(1)')])
    ])
]).

% Slide animations
animation(slide_in_left, [
    duration(300),
    easing(ease_out),
    fill_mode(forwards),
    keyframes([
        frame(0, [transform('translateX(-100%)')]),
        frame(100, [transform('translateX(0)')])
    ])
]).

animation(slide_in_right, [
    duration(300),
    easing(ease_out),
    fill_mode(forwards),
    keyframes([
        frame(0, [transform('translateX(100%)')]),
        frame(100, [transform('translateX(0)')])
    ])
]).

animation(slide_out_left, [
    duration(300),
    easing(ease_in),
    fill_mode(forwards),
    keyframes([
        frame(0, [transform('translateX(0)')]),
        frame(100, [transform('translateX(-100%)')])
    ])
]).

animation(slide_out_right, [
    duration(300),
    easing(ease_in),
    fill_mode(forwards),
    keyframes([
        frame(0, [transform('translateX(0)')]),
        frame(100, [transform('translateX(100%)')])
    ])
]).

% Rotation animations
animation(rotate_in, [
    duration(400),
    easing(ease_out),
    fill_mode(forwards),
    keyframes([
        frame(0, [opacity(0), transform('rotate(-180deg) scale(0.5)')]),
        frame(100, [opacity(1), transform('rotate(0) scale(1)')])
    ])
]).

animation(spin, [
    duration(1000),
    easing(linear),
    iteration_count(infinite),
    keyframes([
        frame(0, [transform('rotate(0deg)')]),
        frame(100, [transform('rotate(360deg)')])
    ])
]).

% Pulse animations
animation(pulse, [
    duration(1000),
    easing(ease_in_out),
    iteration_count(infinite),
    keyframes([
        frame(0, [transform('scale(1)')]),
        frame(50, [transform('scale(1.05)')]),
        frame(100, [transform('scale(1)')])
    ])
]).

animation(heartbeat, [
    duration(1200),
    easing(ease_in_out),
    iteration_count(infinite),
    keyframes([
        frame(0, [transform('scale(1)')]),
        frame(14, [transform('scale(1.3)')]),
        frame(28, [transform('scale(1)')]),
        frame(42, [transform('scale(1.3)')]),
        frame(70, [transform('scale(1)')])
    ])
]).

% Shake animation
animation(shake, [
    duration(500),
    easing(ease_in_out),
    keyframes([
        frame(0, [transform('translateX(0)')]),
        frame(20, [transform('translateX(-10px)')]),
        frame(40, [transform('translateX(10px)')]),
        frame(60, [transform('translateX(-10px)')]),
        frame(80, [transform('translateX(10px)')]),
        frame(100, [transform('translateX(0)')])
    ])
]).

% Chart-specific animations
animation(draw_line, [
    duration(1500),
    easing(ease_out),
    fill_mode(forwards),
    keyframes([
        frame(0, [stroke_dashoffset(1000)]),
        frame(100, [stroke_dashoffset(0)])
    ])
]).

animation(bar_grow, [
    duration(800),
    easing(ease_out),
    fill_mode(forwards),
    keyframes([
        frame(0, [transform('scaleY(0)'), transform_origin('bottom')]),
        frame(100, [transform('scaleY(1)'), transform_origin('bottom')])
    ])
]).

animation(pie_slice, [
    duration(600),
    easing(ease_out),
    fill_mode(forwards),
    keyframes([
        frame(0, [opacity(0), transform('scale(0.8)')]),
        frame(100, [opacity(1), transform('scale(1)')])
    ])
]).

% ============================================================================
% DEFAULT TRANSITIONS
% ============================================================================

transition(hover_lift, [
    properties([transform, box_shadow]),
    duration(200),
    easing(ease_out),
    on_hover([
        transform('translateY(-2px)'),
        box_shadow('0 4px 12px rgba(0,0,0,0.15)')
    ])
]).

transition(hover_glow, [
    properties([box_shadow]),
    duration(300),
    easing(ease_in_out),
    on_hover([
        box_shadow('0 0 20px var(--accent, #00d4ff)')
    ])
]).

transition(hover_scale, [
    properties([transform]),
    duration(150),
    easing(ease_out),
    on_hover([
        transform('scale(1.05)')
    ])
]).

transition(color_fade, [
    properties([color, background_color]),
    duration(200),
    easing(ease_in_out)
]).

transition(opacity_fade, [
    properties([opacity]),
    duration(200),
    easing(ease_in_out)
]).

transition(all_smooth, [
    properties([all]),
    duration(300),
    easing(ease_in_out)
]).

% Focus transitions
transition(focus_ring, [
    properties([outline, outline_offset]),
    duration(150),
    easing(ease_out),
    on_focus([
        outline('2px solid var(--accent, #00d4ff)'),
        outline_offset('2px')
    ])
]).

% ============================================================================
% CSS GENERATION
% ============================================================================

%% generate_keyframes_css(+Name, -CSS)
%  Generate @keyframes CSS from animation definition.
generate_keyframes_css(Name, CSS) :-
    animation(Name, Options),
    member(keyframes(Frames), Options),
    atom_string(Name, NameStr),
    generate_keyframes_body(Frames, BodyStr),
    format(atom(CSS), '@keyframes ~w {\n~w}', [NameStr, BodyStr]).

%% generate_keyframes_body(+Frames, -Body)
generate_keyframes_body(Frames, Body) :-
    maplist(generate_frame_css, Frames, FrameStrs),
    atomic_list_concat(FrameStrs, '\n', Body).

%% generate_frame_css(+Frame, -CSS)
generate_frame_css(frame(Percent, Properties), CSS) :-
    generate_properties_css(Properties, PropsStr),
    format(atom(CSS), '  ~w% {\n~w  }\n', [Percent, PropsStr]).

%% generate_properties_css(+Properties, -CSS)
generate_properties_css(Properties, CSS) :-
    maplist(generate_property_css, Properties, PropStrs),
    atomic_list_concat(PropStrs, CSS).

%% generate_property_css(+Property, -CSS)
generate_property_css(Property, CSS) :-
    Property =.. [PropName, Value],
    css_property_name(PropName, CSSName),
    format(atom(CSS), '    ~w: ~w;\n', [CSSName, Value]).

%% css_property_name(+PrologName, -CSSName)
css_property_name(opacity, 'opacity').
css_property_name(transform, 'transform').
css_property_name(transform_origin, 'transform-origin').
css_property_name(background_color, 'background-color').
css_property_name(background, 'background').
css_property_name(color, 'color').
css_property_name(box_shadow, 'box-shadow').
css_property_name(border_radius, 'border-radius').
css_property_name(stroke_dashoffset, 'stroke-dashoffset').
css_property_name(stroke_dasharray, 'stroke-dasharray').
css_property_name(outline, 'outline').
css_property_name(outline_offset, 'outline-offset').
css_property_name(filter, 'filter').
css_property_name(clip_path, 'clip-path').
css_property_name(Name, Name) :- atom(Name).

%% generate_animation_css(+Name, -CSS)
%  Generate complete animation CSS (keyframes + class).
generate_animation_css(Name, CSS) :-
    generate_keyframes_css(Name, KeyframesCSS),
    generate_animation_class(Name, Name, ClassCSS),
    format(atom(CSS), '~w\n\n~w', [KeyframesCSS, ClassCSS]).

%% generate_animation_class(+AnimationName, +ClassName, -CSS)
%  Generate animation utility class.
generate_animation_class(AnimationName, ClassName, CSS) :-
    animation(AnimationName, Options),
    get_duration_value(Options, Duration),
    get_easing_value(Options, Easing),
    get_fill_mode_value(Options, FillMode),
    get_iteration_count_value(Options, IterCount),
    get_delay_value(Options, Delay),
    format(atom(CSS), '.~w {\n  animation-name: ~w;\n  animation-duration: ~wms;\n  animation-timing-function: ~w;\n  animation-fill-mode: ~w;\n  animation-iteration-count: ~w;\n  animation-delay: ~wms;\n}\n',
        [ClassName, AnimationName, Duration, Easing, FillMode, IterCount, Delay]).

%% Helper predicates for animation options
get_duration_value(Options, Duration) :-
    (member(duration(Duration), Options) -> true ; Duration = 300).

get_easing_value(Options, EasingCSS) :-
    (member(easing(EasingName), Options), easing(EasingName, EasingCSS) -> true
    ; EasingCSS = 'ease').

get_fill_mode_value(Options, FillMode) :-
    (member(fill_mode(FillMode), Options) -> true ; FillMode = 'none').

get_iteration_count_value(Options, IterCount) :-
    (member(iteration_count(IterCount), Options) -> true ; IterCount = 1).

get_delay_value(Options, Delay) :-
    (member(delay(Delay), Options) -> true ; Delay = 0).

%% generate_transition_css(+Name, -CSS)
%  Generate transition CSS.
generate_transition_css(Name, CSS) :-
    transition(Name, Options),
    get_transition_properties(Options, Properties),
    get_duration_value(Options, Duration),
    get_easing_value(Options, Easing),
    get_delay_value(Options, Delay),
    format_property_list(Properties, PropStr),
    format(atom(BaseCSS), '.~w {\n  transition-property: ~w;\n  transition-duration: ~wms;\n  transition-timing-function: ~w;\n  transition-delay: ~wms;\n}\n',
        [Name, PropStr, Duration, Easing, Delay]),
    (member(on_hover(HoverProps), Options) ->
        generate_hover_css(Name, HoverProps, HoverCSS),
        format(atom(CSS), '~w\n~w', [BaseCSS, HoverCSS])
    ; member(on_focus(FocusProps), Options) ->
        generate_focus_css(Name, FocusProps, FocusCSS),
        format(atom(CSS), '~w\n~w', [BaseCSS, FocusCSS])
    ; CSS = BaseCSS
    ).

%% get_transition_properties(+Options, -Properties)
get_transition_properties(Options, Properties) :-
    (member(properties(Properties), Options) -> true
    ; member(property(Prop), Options) -> Properties = [Prop]
    ; Properties = [all]).

%% format_property_list(+Properties, -String)
format_property_list(Properties, String) :-
    maplist(format_css_property, Properties, PropStrs),
    atomic_list_concat(PropStrs, ', ', String).

format_css_property(all, 'all') :- !.
format_css_property(Prop, CSS) :-
    css_property_name(Prop, CSS).

%% generate_hover_css(+Name, +Properties, -CSS)
generate_hover_css(Name, Properties, CSS) :-
    generate_properties_css(Properties, PropsStr),
    format(atom(CSS), '.~w:hover {\n~w}\n', [Name, PropsStr]).

%% generate_focus_css(+Name, +Properties, -CSS)
generate_focus_css(Name, Properties, CSS) :-
    generate_properties_css(Properties, PropsStr),
    format(atom(CSS), '.~w:focus-visible {\n~w}\n', [Name, PropsStr]).

%% generate_all_animations_css(-CSS)
%  Generate CSS for all defined animations.
generate_all_animations_css(CSS) :-
    findall(AnimCSS, (animation(Name, _), generate_animation_css(Name, AnimCSS)), AnimCSSList),
    findall(TransCSS, (transition(Name, _), generate_transition_css(Name, TransCSS)), TransCSSList),
    append(AnimCSSList, TransCSSList, AllCSS),
    atomic_list_concat(AllCSS, '\n', CSS).

% ============================================================================
% ANIMATION UTILITIES
% ============================================================================

%% animation_property(+Animation, +Property, -Value)
animation_property(AnimName, Property, Value) :-
    animation(AnimName, Options),
    member(Term, Options),
    Term =.. [Property, Value].

%% get_animation_duration(+Animation, -Duration)
get_animation_duration(AnimName, Duration) :-
    animation(AnimName, Options),
    get_duration_value(Options, Duration).

%% get_animation_easing(+Animation, -Easing)
get_animation_easing(AnimName, Easing) :-
    animation(AnimName, Options),
    get_easing_value(Options, Easing).

% ============================================================================
% REACT/JSX GENERATION
% ============================================================================

%% generate_animation_hook(+Name, -Hook)
%  Generate a React hook for animation control.
generate_animation_hook(Name, Hook) :-
    animation(Name, Options),
    get_duration_value(Options, Duration),
    atom_string(Name, NameStr),
    to_camel_case(NameStr, CamelName),
    format(atom(Hook), 'const use~wAnimation = (onComplete?: () => void) => {
  const [isAnimating, setIsAnimating] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  const trigger = useCallback(() => {
    setIsAnimating(true);
    if (ref.current) {
      ref.current.classList.add(styles.~w);
    }
    setTimeout(() => {
      setIsAnimating(false);
      if (ref.current) {
        ref.current.classList.remove(styles.~w);
      }
      onComplete?.();
    }, ~w);
  }, [onComplete]);

  return { ref, isAnimating, trigger };
};', [CamelName, Name, Name, Duration]).

%% generate_animation_component(+Name, -JSX)
%  Generate a reusable animation wrapper component.
generate_animation_component(Name, JSX) :-
    animation(Name, Options),
    get_duration_value(Options, Duration),
    atom_string(Name, NameStr),
    to_camel_case(NameStr, CamelName),
    format(atom(JSX), 'interface ~wAnimationProps {
  children: React.ReactNode;
  show?: boolean;
  onAnimationEnd?: () => void;
}

export const ~wAnimation: React.FC<~wAnimationProps> = ({
  children,
  show = true,
  onAnimationEnd
}) => {
  const [shouldRender, setShouldRender] = useState(show);

  useEffect(() => {
    if (show) setShouldRender(true);
  }, [show]);

  const handleAnimationEnd = () => {
    if (!show) setShouldRender(false);
    onAnimationEnd?.();
  };

  if (!shouldRender) return null;

  return (
    <div
      className={show ? styles.~w : styles.~wOut}
      style={{ animationDuration: "~wms" }}
      onAnimationEnd={handleAnimationEnd}
    >
      {children}
    </div>
  );
};', [CamelName, CamelName, CamelName, Name, Name, Duration]).

%% to_camel_case(+SnakeCase, -CamelCase)
to_camel_case(Snake, Camel) :-
    atom_string(Snake, SnakeStr),
    split_string(SnakeStr, "_", "", Parts),
    maplist(capitalize_first, Parts, CapParts),
    atomic_list_concat(CapParts, Camel).

capitalize_first(Str, Cap) :-
    string_chars(Str, [First|Rest]),
    upcase_atom(First, UpperFirst),
    atom_chars(Cap, [UpperFirst|Rest]).

% ============================================================================
% MANAGEMENT
% ============================================================================

%% declare_animation(+Name, +Options)
declare_animation(Name, Options) :-
    retractall(animation(Name, _)),
    assertz(animation(Name, Options)).

%% declare_transition(+Name, +Options)
declare_transition(Name, Options) :-
    retractall(transition(Name, _)),
    assertz(transition(Name, Options)).

%% clear_animations/0
clear_animations :-
    retractall(animation(_, _)),
    retractall(transition(_, _)).

% ============================================================================
% TESTING
% ============================================================================

test_animation_generator :-
    format('========================================~n'),
    format('Animation Generator Tests~n'),
    format('========================================~n~n'),

    % Test 1: Keyframes generation
    format('Test 1: Keyframes generation~n'),
    generate_keyframes_css(fade_in, KeyframesCSS),
    (sub_atom(KeyframesCSS, _, _, _, '@keyframes')
    -> format('  PASS: Has @keyframes~n')
    ; format('  FAIL: Missing @keyframes~n')),
    (sub_atom(KeyframesCSS, _, _, _, 'opacity')
    -> format('  PASS: Has opacity property~n')
    ; format('  FAIL: Missing opacity~n')),

    % Test 2: Animation class generation
    format('~nTest 2: Animation class generation~n'),
    generate_animation_class(fade_in, fade_in, ClassCSS),
    (sub_atom(ClassCSS, _, _, _, 'animation-name')
    -> format('  PASS: Has animation-name~n')
    ; format('  FAIL: Missing animation-name~n')),
    (sub_atom(ClassCSS, _, _, _, 'animation-duration')
    -> format('  PASS: Has animation-duration~n')
    ; format('  FAIL: Missing animation-duration~n')),

    % Test 3: Transition CSS generation
    format('~nTest 3: Transition CSS generation~n'),
    generate_transition_css(hover_lift, TransCSS),
    (sub_atom(TransCSS, _, _, _, 'transition-property')
    -> format('  PASS: Has transition-property~n')
    ; format('  FAIL: Missing transition-property~n')),
    (sub_atom(TransCSS, _, _, _, ':hover')
    -> format('  PASS: Has hover state~n')
    ; format('  FAIL: Missing hover state~n')),

    % Test 4: Easing functions
    format('~nTest 4: Easing functions~n'),
    easing(ease_out_back, BackEasing),
    (sub_atom(BackEasing, _, _, _, 'cubic-bezier')
    -> format('  PASS: ease_out_back has cubic-bezier~n')
    ; format('  FAIL: ease_out_back missing cubic-bezier~n')),

    % Test 5: Animation utilities
    format('~nTest 5: Animation utilities~n'),
    get_animation_duration(fade_in, Duration),
    (Duration =:= 300
    -> format('  PASS: fade_in duration is 300ms~n')
    ; format('  FAIL: Wrong duration~n')),

    % Test 6: React hook generation
    format('~nTest 6: React hook generation~n'),
    generate_animation_hook(scale_in, Hook),
    (sub_atom(Hook, _, _, _, 'useState')
    -> format('  PASS: Hook uses useState~n')
    ; format('  FAIL: Hook missing useState~n')),
    (sub_atom(Hook, _, _, _, 'useCallback')
    -> format('  PASS: Hook uses useCallback~n')
    ; format('  FAIL: Hook missing useCallback~n')),

    % Test 7: Chart animations
    format('~nTest 7: Chart-specific animations~n'),
    animation(draw_line, DrawOpts),
    (member(keyframes(_), DrawOpts)
    -> format('  PASS: draw_line has keyframes~n')
    ; format('  FAIL: draw_line missing keyframes~n')),
    animation(bar_grow, BarOpts),
    (member(keyframes(_), BarOpts)
    -> format('  PASS: bar_grow has keyframes~n')
    ; format('  FAIL: bar_grow missing keyframes~n')),

    format('~nAll tests completed.~n').
