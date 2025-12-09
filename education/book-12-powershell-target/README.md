<!--
SPDX-License-Identifier: MIT AND CC-BY-4.0
Copyright (c) 2025 John William Creighton (s243a)

This documentation is dual-licensed under MIT and CC-BY-4.0.
-->

# Book 12: PowerShell Target

**Windows Automation and .NET Scripting**

*Part of the [UnifyWeaver Education Series](../README.md)*

This book covers compiling Prolog predicates to PowerShell scripts. PowerShell's deep Windows integration and .NET access make this target ideal for Windows automation, system administration, and enterprise environments.

## Prerequisites

**Required:**
- [Book 1: Foundations](../book-01-foundations/README.md)

**Recommended:**
- [Book 3: C# Target](../book-03-csharp-target/README.md) - shares .NET concepts
- [Book 9: Rust Target](../book-09-rust-target/README.md) - first book in the Specialized Targets section

**Technical:**
- PowerShell 7+ (cross-platform) or Windows PowerShell 5.1
- Basic PowerShell knowledge (helpful)

## Implementation Status

This table shows the current implementation status of features described in each chapter.

| Chapter | Feature | Status | Notes |
|---------|---------|--------|-------|
| [Ch 1](01_introduction) | Basic compilation | **Implemented** | `compile_to_powershell/3` works |
| [Ch 1](01_introduction) | Module loading | **Implemented** | `use_module(unifyweaver(core/powershell_compiler))` |
| [Ch 2](02_facts_rules) | Facts to arrays | **Implemented** | Facts compile to `PSCustomObject` arrays |
| [Ch 2](02_facts_rules) | Rules to joins | Partial | Rules generate join code, but don't auto-include dependent facts |
| [Ch 3](03_cmdlet_generation) | CmdletBinding | Not yet | Advanced function attributes not generated |
| [Ch 3](03_cmdlet_generation) | Parameter validation | Not yet | `[ValidateSet()]`, `[Mandatory]` not generated |
| [Ch 3](03_cmdlet_generation) | Begin/Process/End | Partial | Only in `.NET` mode via `dotnet_source` |
| [Ch 4](04_dotnet_integration) | dotnet_source | **Implemented** | Inline C# compilation works |
| [Ch 4](04_dotnet_integration) | DLL caching | **Implemented** | `pre_compile(true)` generates caching |
| [Ch 4](04_dotnet_integration) | NuGet references | **Implemented** | `references(['LiteDB'])` works |
| [Ch 5](05_windows_automation) | Windows automation | Design only | Examples show patterns, not auto-generated |

**Legend:** **Implemented** = tested and working, Partial = works with limitations, Not yet = documented but not implemented, Design only = conceptual/aspirational

## Learning Path

**[1. Introduction](01_introduction)** - *Implemented*
- Why use the PowerShell target?
- PowerShell vs Bash for automation
- Compilation modes (BaaS, Pure, Inline .NET)
- Your first PowerShell compilation

**[2. Facts and Rules](02_facts_rules)** - *Mostly Implemented*
- Compiling facts to PowerShell arrays
- PSCustomObject for binary facts
- Translating rules to functions with joins
- Pipeline integration

**[3. Cmdlet Generation](03_cmdlet_generation)** - *Design Document*
- Creating advanced functions with CmdletBinding
- Parameter attributes and validation
- Begin/Process/End blocks
- Verbose and Debug output

**[4. .NET Integration](04_dotnet_integration)** - *Fully Implemented*
- Inline C# with Add-Type
- The dotnet_source plugin
- DLL caching for 138x speedup
- NuGet package integration

**[5. Windows Automation](05_windows_automation)** - *Design Document*
- File system operations
- Windows services management
- Registry access
- Event logs and WMI/CIM queries

### Part 3: Advanced Topics (Planned)

**Chapter 6: In-Process Hosting** *(coming soon)*
- Sharing runtime with C# target
- Cross-target glue via .NET
- Performance considerations

**Chapter 7: Active Directory** *(coming soon)*
- LDAP queries from Prolog
- User and group management
- Permission handling

**Chapter 8: Enterprise Patterns** *(coming soon)*
- Remote execution (PSRemoting)
- Scheduled tasks
- Credential management

## Quick Example

```prolog
% Define a Windows service checker
service_running(Name) :-
    get_service(Name, Status),
    Status == 'Running'.

% Compile to PowerShell
?- compile_to_powershell(service_running/1, [], Code).

% Generated PowerShell:
% function Test-ServiceRunning {
%     param([string]$Name)
%     $service = Get-Service -Name $Name -ErrorAction SilentlyContinue
%     return $service.Status -eq 'Running'
% }
```

## What's Next?

After completing Book 12, continue to:
- [Book 7: Cross-Target Glue](../book-07-cross-target-glue/README.md) - .NET bridge integration
- [Book 8: Security & Firewall](../book-08-security-firewall/README.md) - Enterprise security
- [Book 13: Semantic Search](../book-13-semantic-search/README.md) - AI capabilities

## License

This educational content is licensed under CC BY 4.0.
Code examples are dual-licensed under MIT OR Apache-2.0.
