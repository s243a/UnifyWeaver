:- encoding(utf8).
% SPDX-License-Identifier: MIT OR Apache-2.0
%
% csharp_query_target.pl - DEPRECATED: Compatibility shim for csharp_target.pl
% This module exists only for backward compatibility. All functionality has moved to csharp_target.pl
% 
% MIGRATION: Replace use_module(csharp_query_target) with use_module(csharp_target)

:- module(csharp_query_target, [
    build_query_plan/3,     % +PredIndicator, +Options, -PlanDict
    build_query_plans/3,    % +PredIndicator, +Options, -PlanDicts
    build_query_plan_for_inputs/4, % +PredIndicator, +Options, +InputPositions, -PlanDict
    render_plan_to_csharp/2,% +PlanDict, -CSharpSource
    plan_module_name/2      % +PlanDict, -ModuleName
]).

:- use_module(csharp_target, [
    build_query_plan/3,
    build_query_plans/3,
    build_query_plan_for_inputs/4,
    render_plan_to_csharp/2,
    plan_module_name/2
]).

% All predicates are re-exported from csharp_target
% No additional implementation needed
