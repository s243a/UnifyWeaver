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

## What You'll Learn

By completing this book, you will be able to:

- Compile Prolog predicates to PowerShell cmdlets
- Generate standalone PowerShell scripts
- Access .NET types from generated code
- Build Windows automation pipelines
- Integrate with Active Directory and Windows services

## Chapter Overview (Planned)

### Part 1: Basic Compilation

**Chapter 1: Introduction**
- Why use the PowerShell target?
- PowerShell vs Bash for automation
- Cross-platform PowerShell (pwsh)

**Chapter 2: Facts and Rules**
- Compiling facts to PowerShell arrays
- Translating rules to functions
- Pipeline integration (|)

**Chapter 3: Cmdlet Generation**
- Creating advanced functions
- Parameter attributes
- Pipeline input/output

### Part 2: .NET Integration

**Chapter 4: .NET Types**
- Accessing .NET from PowerShell
- Type conversions
- Working with collections

**Chapter 5: In-Process with C#**
- Sharing runtime with C# target
- Cross-target glue via .NET
- Performance considerations

### Part 3: Windows Automation

**Chapter 6: System Administration**
- File system operations
- Registry access
- Windows services
- Event logs

**Chapter 7: Active Directory**
- LDAP queries from Prolog
- User and group management
- Permission handling

**Chapter 8: Enterprise Patterns**
- Remote execution (PSRemoting)
- Scheduled tasks
- Credential management

## Content Status

This book is planned. Chapters are not yet written.

**For .NET concepts now**, see:
- [Book 3: C# Target](../book-03-csharp-target/README.md) - Core .NET compilation
- [Book 7: Cross-Target Glue](../book-07-cross-target-glue/README.md) - PowerShell in-process hosting

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
