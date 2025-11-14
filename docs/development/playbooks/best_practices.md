# Playbook Authoring Best Practices

## Do's
- Use clear section headers and concise summaries
- Structure playbooks for skimmability
- Provide numbered procedural steps where possible
- Reference external code samples, databases, or docs
- Use troubleshooting and validation tables
- Limit inline code to essentials; link out for details
- **Reference, Don't Repeat**: Link to external documentation, skill definitions, or example records instead of embedding large blocks of code or explanation. Keep the playbook as a concise guide.
- **Encapsulate Complex Execution**: For any execution sequence that involves more than one or two simple commands, encapsulate the entire workflow into a single, executable script within the example library. The playbook's execution direction should then be a single step: "Extract and execute record X". This makes the agent's job trivial and the process far more robust.

## Don'ts
- Embed extensive code or scripts in playbook files
- Rely on narrative or unstructured explanation
- Omit validation and troubleshooting steps
- Assume agent or human will infer workflow intent
- Create single-purpose or non-reusable playbooks

## Tips
- Keep Markdown formatting consistent throughout
- Provide worked-out examples as separate files in an example database
- Iterate playbook drafts based on agent or peer feedback
- Regularly review playbooks for outdated details or unclear sections
- Make references and links explicit and easy to follow

## Example Template Reference
- See [specification.md](specification.md)
- See [philosophy.md](philosophy.md)

## Notes
Best practices should be updated as project evolves and LLM workflows change. Community suggestions are encouraged!