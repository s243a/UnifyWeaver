# Quickstart Agent Philosophy

Design principles and rationale for the quickstart agent and task-oriented Q&A database.

## Core Philosophy

### Meet Users Where They Are

Users don't arrive knowing the system's structure. They arrive with:
- A problem to solve ("I have mindmaps everywhere")
- A vague goal ("make my logic deployable")
- Curiosity ("what can this do?")

The quickstart agent translates user intent into system capabilities, not the reverse.

**Anti-pattern:** "UnifyWeaver has 10 compilation targets" (system-centric)
**Pattern:** "You want a standalone binary? Use the Go target." (user-centric)

### General Before Specific

Teach patterns before instances. A user who understands:

```
compile(Source, Target, Options)
```

Can apply that knowledge to any target. A user who only learned:

```
compile_bash(Source, Options)
```

Must re-learn for each new target.

This mirrors how experts think: they have mental models that instantiate to specifics, not a collection of unrelated facts.

### Progressive Disclosure

Don't overwhelm. Reveal complexity in layers:

1. **Layer 0:** What is this? (one sentence)
2. **Layer 1:** What can it do? (3-5 capabilities)
3. **Layer 2:** How do I do X? (commands, prerequisites)
4. **Layer 3:** What are the details? (flags, options, edge cases)

Users pull information as needed. The agent offers, but doesn't push.

### Task Completion Over Feature Education

The goal is **helping users accomplish things**, not teaching them the system.

| Feature Education | Task Completion |
|-------------------|-----------------|
| "Here's how MST clustering works..." | "Your 500 mindmaps are now in 12 topic folders." |
| "The hierarchy objective is J = D/(1+H)..." | "This mindmap best fits the 'Physics' folder (87% confidence)." |

Explain only what's needed for the current task. Offer deeper dives, don't require them.

## Question Design Principles

### Start With Verbs

Good questions start with what users want to **do**:

- "How do I compile..."
- "I want to organize..."
- "Can I generate..."
- "Where should I put..."

These are naturally task-oriented and map to actions.

### Avoid Jargon in Questions

Questions should use the user's vocabulary, not the system's:

| System Jargon | User Language |
|---------------|---------------|
| "Procrustes projection" | "find similar items" |
| "federated W matrices" | "trained model" |
| "cloudmapref attributes" | "links between mindmaps" |
| "semi-naive evaluation" | "handle complex recursion" |

Answers can introduce terminology, but questions meet users where they are.

### Questions Should Be Discoverable

A new user should be able to formulate any question in our database without prior knowledge.

**Test:** Could someone who just cloned the repo and read only the README ask this question?

If the question requires knowing internal terminology, it belongs at Layer 3+, not Layer 0-2.

## Answer Design Principles

### Answers Are Entry Points

Each answer should provide:
1. A direct response to the question
2. A concrete next action (command, file to read, script to run)
3. Optionally, pointers to go deeper

**Example:**
```
Q: How do I compile Prolog to Bash?

A: Use compiler_driver.pl with --target bash:

   swipl -g "compile('input.pl', bash, 'output.sh')" -t halt src/compiler_driver.pl

   This generates a shell script with Unix pipes for data flow.

   → For recursion handling: see docs/ADVANCED_RECURSION.md
   → For all targets: compiler_driver.pl --help
```

### Answers Should Be Testable

Every answer should include something the user can run to verify it works:

- A command that produces output
- A file path they can check exists
- A predicate they can query

This builds confidence and catches documentation rot.

### Acknowledge Prerequisites

If a task requires setup, say so upfront:

**Bad:** "Run `link_pearltrees.py --input mindmap.smmx`"

**Good:** "First, ensure you have a trained model (see 'How do I train a model?'). Then run..."

Users waste hours debugging when prerequisites are buried.

## Database Design Principles

### Separation for Curation

Build the quickstart database separately from the general Q&A database:

- **Focused review:** Can audit all entries systematically
- **Coverage tracking:** Easy to see gaps in the capability tree
- **Quality gate:** Entries must pass review before inclusion

Merge later, or keep as a priority tier in retrieval.

### Questions Have Levels

```
Level 0: "What is UnifyWeaver?"           (identity)
Level 1: "What can it do?"                (capabilities)
Level 2: "How do I compile?"              (general task)
Level 3: "How do I compile to Bash?"      (specific task)
Level 4: "What flags does --target bash support?" (details)
```

Lower levels should have high-quality, canonical answers. Higher levels can be more numerous and specific.

### Answers Reference the Tree

Each answer knows its position in the capability tree:

```jsonl
{
  "question": "How do I compile to Bash?",
  "level": 3,
  "tree_path": ["Compilation", "Targets", "bash"],
  "answer": "...",
  "prerequisites": ["How do I compile?"],
  "go_deeper": ["What recursion patterns does Bash support?"]
}
```

This enables:
- Coverage analysis (which tree nodes lack questions?)
- Navigation (user asked L3, suggest L4)
- Prerequisite checking (did user see L2 first?)

## Agent Behavior Principles

### Clarify Before Answering

If a question is ambiguous, ask:

**User:** "How do I organize things?"

**Agent:** "Are you organizing:
- Mindmaps (.smmx files)?
- Bookmarks in Pearltrees?
- Compiled output files?
- Something else?"

Don't guess wrong and waste the user's time.

### Offer Depth, Don't Force It

After answering, offer to go deeper:

**Agent:** "Your mindmaps are now grouped. Want to:
- See how the clustering algorithm works?
- Adjust the number of groups?
- Move specific mindmaps to different folders?"

Let users pull information as needed.

### Remember Context

Within a session, remember what the user has learned:

- Don't re-explain prerequisites they've already seen
- Build on previous answers
- Notice patterns ("You've asked about 3 mindmap tasks—want a full overview?")

### Admit Limitations

If a question isn't covered:

**Agent:** "I don't have a specific answer for that. Here's what might help:
- Related question: [X]
- Documentation: [Y]
- Or describe what you're trying to accomplish and I'll try to help."

Never hallucinate answers.

## Success Metrics

### For the Database

- **Coverage:** % of capability tree nodes with at least one question
- **Balance:** Questions distributed across levels (not all L4)
- **Freshness:** Answers reference current code/docs

### For the Agent

- **Task completion rate:** Did the user accomplish their goal?
- **Turns to answer:** Fewer clarification rounds = better questions
- **Escalation rate:** How often does the agent need to fall back to general search?

### For Users

- **Time to first success:** How long until they run something that works?
- **Return rate:** Do they come back, or give up?
- **Question evolution:** Do questions get more sophisticated over time? (learning signal)

## Related Documents

- `QUICKSTART_AGENT_PROPOSAL.md` - Problem statement and capability tree
- `QUICKSTART_AGENT_SPECIFICATION.md` - Database schema and entry format
- `QUICKSTART_AGENT_IMPLEMENTATION.md` - Build plan and tooling
