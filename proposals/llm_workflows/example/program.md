---
name: text_chunker
version: 1.0.0
description: A tool to chunk text for a RAG pipeline.
---

# Program: Text Chunker

## Goal
The primary goal is to take a large piece of text and break it down into smaller, overlapping chunks suitable for retrieval.

## Tool: `chunk_text.sh`

### Description
This tool takes text from standard input and outputs chunked text to standard output, with each chunk separated by "--- CHUNK ---".

### Usage Example
```bash
cat my_document.txt | ./chunk_text.sh
```

## Plan of Action

This is the plan for the LLM to follow to complete the goal.

```todo
- [ ] Get the source text from the user (e.g., ask for a file path).
- [ ] Execute the `chunk_text.sh` tool with the content of the user's file.
- [ ] Count the number of chunks generated.
- [ ] Report the final count to the user.
```

## Declarative Logic

This is a more formal, machine-readable representation of the plan. An LLM could use this for more rigorous reasoning about the workflow.

```pseudocode_prolog
% The plan to chunk text and report the result.
plan(chunk_text, [
    step(get_text, [
        description('Get a file path from the user.'),
        output(to_variable('FilePath'))
    ]),
    step(run_chunker, [
        tool('chunk_text.sh'),
        input(file_content(from_variable('FilePath'))),
        output(to_variable('ChunkOutput'))
    ]),
    step(count_chunks, [
        description('Count the number of "--- CHUNK ---" separators in the output.'),
        input(from_variable('ChunkOutput')),
        output(to_variable('ChunkCount'))
    ]),
    step(report, [
        description('Report the final count to the user.'),
        input(from_variable('ChunkCount'))
    ])
]).
```
