# The Economic Agent: A Philosophy for AI-Driven Development

## 1. Introduction

This document outlines a philosophy for developing advanced AI agents. The core idea is to move beyond creating simple instruction-following bots and towards creating autonomous, resource-aware **economic agents**.

An economic agent does not just execute a pre-defined plan. It makes strategic decisions by weighing the costs and benefits of its potential actions against a given set of resources (time, API costs, computational power). Its goal is to achieve a task not just correctly, but *efficiently* and *intelligently*.

## 2. Core Principles

### Principle 1: The Agent as an Economic Strategist

The agent's primary role is that of a strategist. Given a goal, it must analyze the resources available and the constraints of the task to choose the most effective course of action from a set of possible strategies. This moves the agent from a "doer" to a "decision-maker."

### Principle 2: The Playbook over the Recipe

We do not provide the agent with a single, rigid recipe. Instead, we provide it with a **playbook**. This playbook (our `program.md` artifact) contains:

*   A clear statement of the overall goal.
*   A menu of different strategies to achieve that goal.
*   The tools required for each strategy.
*   The relative costs (time, API, etc.) and benefits (accuracy, detail) of each strategy.
*   Heuristic guidance to help the agent choose the best strategy.

This empowers the agent with choice and the context to make that choice wisely.

### Principle 3: Justification as a Core Task

An autonomous agent that makes its own decisions must be transparent. Therefore, a core task for the agent is to articulate its chosen strategy and justify *why* it was chosen over the other options. For example: "The files are numerous, so I will start with a fast, cheap triage strategy (local indexing) to identify promising candidates before committing to more expensive API calls."

### Principle 4: Dynamic, Composable Workflows

The agent is not bound by the explicit plans in its playbook. The playbook is a guide, not a straitjacket. The agent should be empowered to dynamically compose its tools in novel ways to handle unexpected opportunities, failures, or real-time user feedback. This includes:

*   **Recovery:** Re-planning a new course of action when a tool fails.
*   **Opportunism:** Using a tool that wasn't in the original plan because an opportunity to improve the outcome arises.
*   **Interaction:** Incorporating user feedback to modify its approach mid-task.

### Principle 5: Efficient World-Knowledge Management

The agent's knowledge of its environment (the codebase, file system, etc.) should be persistent and updated efficiently. It should not re-discover the world from scratch on every run. This principle, inspired by our discussion of indexing, includes:

*   **Persistent State:** The agent should assume that processes like indexing can create a persistent state (e.g., an embeddings database).
*   **Smart Updates:** The agent should not blindly re-index everything. It should have strategies for updating its knowledge based on what has changed (e.g., re-indexing only modified files).
*   **Strategic Indexing:** If a large number of files have changed, the agent might even decide to only re-index a "promising subset" to save time, making a tactical decision about the trade-off between the completeness of its knowledge and the cost of updating it.

## 3. Conclusion

By following these principles, we aim to create not just AI tools, but true AI **collaborators**. These agents will be able to reason about their own actions, manage resources effectively, and engage in sophisticated, goal-oriented problem-solving, making them powerful partners in any complex task.
