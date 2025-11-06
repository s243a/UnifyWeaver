<!--
SPDX-License-Identifier: MIT AND CC-BY-4.0
Copyright (c) 2025 John William Creighton (s243a)

This documentation is dual-licensed under MIT and CC-BY-4.0.
-->

# Video Script: Quick Start - Your First C# Compilation

**Duration:** 5-7 minutes
**Target Audience:** Developers familiar with Prolog basics
**Goal:** Compile and run first C# program in under 5 minutes

---

## Opening (30 seconds)

**[Screen: Terminal with UnifyWeaver logo]**

> "In this tutorial, we'll compile your first Prolog program to C# and run it.
> Start to finish in under 5 minutes."

**Show:**
- Prerequisites checklist on screen
- .NET SDK 9.0 installed
- SWI-Prolog 9.0+
- UnifyWeaver cloned

---

## Section 1: The Simplest Example (1 minute)

**[Screen: Create new file]**

> "Let's start with family relationships."

**Type in `family.pl`:**
```prolog
parent(alice, bob).
parent(bob, charlie).

grandparent(X, Z) :- parent(X, Y), parent(Y, Z).
```

**Voiceover:**
> "Three lines of Prolog.
> Two facts: alice is bob's parent, bob is charlie's parent.
> One rule: grandparent is parent of parent."

---

## Section 2: Compile to C# (1.5 minutes)

**[Screen: Terminal]**

**Type:**
```bash
swipl -q
```

**Then load:**
```prolog
?- ['family.pl'].
?- use_module('src/unifyweaver/targets/csharp_stream_target').
?- compile_predicate_to_csharp(grandparent/2, [], Code).
```

**Voiceover:**
> "Load UnifyWeaver's C# Stream Target.
> Compile grandparent to C#.
> Watch what happens."

**[Screen: Show Code variable output scrolling]**

> "UnifyWeaver generated complete C# code.
> Let's save it to a file."

**Type:**
```prolog
?- open('Grandparent.cs', write, Stream),
   write(Stream, Code),
   close(Stream).
```

---

## Section 3: Build and Run (1.5 minutes)

**[Screen: Terminal, new tab]**

**Type:**
```bash
dotnet new console -n FamilyQuery
mv Grandparent.cs FamilyQuery/Program.cs
cd FamilyQuery
dotnet build
dotnet run
```

**[Screen: Output appears]**
```
alice:charlie
```

**Voiceover:**
> "One result: alice is charlie's grandparent.
> From Prolog to executable C# in 30 seconds."

---

## Section 4: What Just Happened? (1.5 minutes)

**[Screen: Split view - Prolog on left, C# on right]**

**Highlight Prolog rule:**
```prolog
grandparent(X, Z) :- parent(X, Y), parent(Y, Z).
```

**Show corresponding C# (simplified):**
```csharp
ParentData
    .SelectMany(xy => ParentData
        .Where(yz => xy.Item2 == yz.Item1)
        .Select(yz => (xy.Item1, yz.Item2)))
    .Distinct();
```

**Voiceover:**
> "UnifyWeaver translated the Prolog join into LINQ.
> SelectMany creates the cross product.
> Where filters matching relationships.
> Select projects the result.
> Distinct removes duplicates."

**[Highlight each operation as it's mentioned]**

---

## Section 5: Try It Yourself (30 seconds)

**[Screen: Challenge text]**

**Show:**
```
Challenge: Add more parents
parent(alice, diane).
parent(diane, eve).

What grandparents will we find?
```

**Voiceover:**
> "Add more parents.
> Recompile.
> What changes in the output?"

---

## Closing (30 seconds)

**[Screen: Summary]**

**Show bullet points appearing:**
- ✓ Wrote 3 lines of Prolog
- ✓ Compiled to C#
- ✓ Built and ran in seconds
- ✓ Got correct results

**Voiceover:**
> "You just compiled Prolog to production C# code.
> Next video: handling larger datasets with the Stream Target.
> See you there."

**[Screen: Links to next videos]**

---

## Notes for Recording

**Visual Style:**
- Dark theme terminal
- Large font (18-20pt)
- Syntax highlighting enabled
- Side-by-side splits when comparing code

**Pacing:**
- Pause 2 seconds after each command
- Let output fully appear before continuing
- Highlight matching code sections simultaneously

**Common Questions to Address:**
- Q: "Why C#?" → A: "Type safety, performance, LINQ makes query translation natural"
- Q: "Can I use other .NET languages?" → A: "Yes, F# and VB.NET work too"
- Q: "What about large data?" → A: "Stream Target handles modest data, Query Runtime for big data"

**Files to Prepare:**
- `family.pl` - demo file
- `Grandparent.cs` - pre-generated for reference
- Working directory clean before recording

**Troubleshooting Callouts:**
- If dotnet not found: "Install .NET SDK 9.0 from dot.net"
- If compilation fails: "Make sure UnifyWeaver is in your Prolog library path"

---

## Follow-up Video Ideas

After this quick start, viewers should watch:
1. **Stream Target Deep Dive** - Join strategies, performance tips
2. **Query Runtime Tutorial** - Recursive queries, fixpoint iteration
3. **Real-World Example** - CSV to C# analysis pipeline
