/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2025 John William Creighton (s243a)
 *
 * Example: .NET Data Processing Pipeline with Python Integration
 *
 * This example demonstrates a four-stage data processing pipeline:
 * 1. C#: Load and validate data
 * 2. PowerShell: Filter and transform
 * 3. IronPython: Enrich with business logic
 * 4. CPython (fallback): ML-based scoring using numpy
 */

:- use_module('../../src/unifyweaver/glue/dotnet_glue').

%% ============================================
%% Pipeline Definition
%% ============================================

%% Define the pipeline stages with their target runtime
pipeline_steps([
    step(validate, csharp, validate_logic, []),
    step(filter, powershell, filter_logic, []),
    step(enrich, ironpython, enrich_logic, [imports([json, collections])]),
    step(score, cpython, score_logic, [imports([numpy])])
]).

%% Stage 1: C# Validation (native)
validate_logic('
    // Validate required fields
    if (string.IsNullOrEmpty(record["id"]))
        return null;
    if (!int.TryParse(record["age"]?.ToString(), out _))
        return null;
    return record;
').

%% Stage 2: PowerShell Filter (in-process via bridge)
filter_logic('
    param($InputObject)

    # Filter: Only active users over 18
    if ($InputObject.status -eq "active" -and [int]$InputObject.age -ge 18) {
        # Transform: Add computed field
        $InputObject | Add-Member -NotePropertyName "adult" -NotePropertyValue $true -PassThru
    }
').

%% Stage 3: IronPython Enrichment (in-process via bridge)
%% Note: Uses only IronPython-compatible modules
enrich_logic('
# Enrich record with business category
from collections import OrderedDict

age = int(record["age"])
if age < 25:
    category = "young_adult"
elif age < 45:
    category = "middle_adult"
elif age < 65:
    category = "senior"
else:
    category = "elderly"

result = OrderedDict(record)
result["category"] = category
result["risk_tier"] = "low" if age < 50 else "medium" if age < 70 else "high"
').

%% Stage 4: CPython ML Scoring (pipe-based fallback)
%% Note: Uses numpy, which is NOT IronPython compatible
score_logic('
import numpy as np

# Simulate ML-based risk scoring
features = np.array([
    float(record.get("age", 0)),
    1.0 if record.get("category") == "young_adult" else 0.0,
    1.0 if record.get("category") == "middle_adult" else 0.0,
    1.0 if record.get("risk_tier") == "low" else 0.5 if record.get("risk_tier") == "medium" else 0.0
])

# Simple scoring model (would be trained model in production)
weights = np.array([0.3, 0.4, 0.2, 0.1])
score = np.dot(features, weights) / 100.0

result = dict(record)
result["ml_score"] = round(float(score), 4)
result["recommendation"] = "approve" if score > 0.5 else "review"
').

%% ============================================
%% Code Generation
%% ============================================

generate_all :-
    format('Generating .NET data pipeline...~n~n'),

    % Generate bridges
    generate_powershell_bridge([], PSBridge),
    generate_ironpython_bridge([], IPyBridge),
    generate_cpython_bridge([], CPyBridge),

    % Generate pipeline
    pipeline_steps(Steps),
    generate_dotnet_pipeline(Steps, [namespace('DataPipeline'), class('UserScorer')], Pipeline),

    % Write files
    open('PowerShellBridge.cs', write, S1),
    write(S1, PSBridge),
    close(S1),
    format('  Created: PowerShellBridge.cs~n'),

    open('IronPythonBridge.cs', write, S2),
    write(S2, IPyBridge),
    close(S2),
    format('  Created: IronPythonBridge.cs~n'),

    open('CPythonBridge.cs', write, S3),
    write(S3, CPyBridge),
    close(S3),
    format('  Created: CPythonBridge.cs~n'),

    open('UserScorerPipeline.cs', write, S4),
    write(S4, Pipeline),
    close(S4),
    format('  Created: UserScorerPipeline.cs~n'),

    % Generate project file
    generate_csproj(Csproj),
    open('DataPipeline.csproj', write, S5),
    write(S5, Csproj),
    close(S5),
    format('  Created: DataPipeline.csproj~n'),

    % Generate sample data
    generate_sample_data(SampleData),
    open('sample_users.json', write, S6),
    write(S6, SampleData),
    close(S6),
    format('  Created: sample_users.json~n'),

    format('~nDone! Build with:~n'),
    format('  dotnet build~n'),
    format('  dotnet run~n').

%% ============================================
%% Project File Generation
%% ============================================

generate_csproj(Csproj) :-
    Csproj = '
<Project Sdk="Microsoft.NET.Sdk">

  <PropertyGroup>
    <OutputType>Exe</OutputType>
    <TargetFramework>net8.0</TargetFramework>
    <ImplicitUsings>enable</ImplicitUsings>
    <Nullable>enable</Nullable>
  </PropertyGroup>

  <ItemGroup>
    <!-- PowerShell hosting -->
    <PackageReference Include="Microsoft.PowerShell.SDK" Version="7.4.0" />

    <!-- IronPython hosting -->
    <PackageReference Include="IronPython" Version="3.4.1" />

    <!-- JSON handling -->
    <PackageReference Include="System.Text.Json" Version="8.0.0" />
  </ItemGroup>

</Project>
'.

%% ============================================
%% Sample Data Generation
%% ============================================

generate_sample_data(Data) :-
    Data = '[
  {"id": "001", "name": "Alice", "age": "28", "status": "active"},
  {"id": "002", "name": "Bob", "age": "45", "status": "active"},
  {"id": "003", "name": "Charlie", "age": "17", "status": "active"},
  {"id": "004", "name": "Diana", "age": "62", "status": "inactive"},
  {"id": "005", "name": "Eve", "age": "35", "status": "active"},
  {"id": "006", "name": "Frank", "age": "72", "status": "active"},
  {"id": "007", "name": "Grace", "age": "22", "status": "active"},
  {"id": "", "name": "Invalid", "age": "30", "status": "active"},
  {"id": "009", "name": "Henry", "age": "bad", "status": "active"},
  {"id": "010", "name": "Ivy", "age": "55", "status": "active"}
]
'.

%% ============================================
%% Runtime Choice Example
%% ============================================

show_runtime_choices :-
    format('~n=== Runtime Choice Examples ===~n~n'),

    % IronPython-compatible imports
    Imports1 = [json, collections, re],
    python_runtime_choice(Imports1, R1),
    format('Imports ~w -> ~w~n', [Imports1, R1]),

    % Requires CPython due to numpy
    Imports2 = [numpy, json],
    python_runtime_choice(Imports2, R2),
    format('Imports ~w -> ~w~n', [Imports2, R2]),

    % Requires CPython due to pandas
    Imports3 = [pandas, sys, os],
    python_runtime_choice(Imports3, R3),
    format('Imports ~w -> ~w~n', [Imports3, R3]),

    % Pure IronPython
    Imports4 = [sys, os, math, datetime],
    python_runtime_choice(Imports4, R4),
    format('Imports ~w -> ~w~n', [Imports4, R4]).

%% ============================================
%% Main
%% ============================================

:- initialization((generate_all, show_runtime_choices), main).
