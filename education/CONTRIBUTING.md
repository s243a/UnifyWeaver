# Contributing to UnifyWeaver

Thank you for your interest in contributing to UnifyWeaver!

## License

By submitting a pull request, you agree to license your contribution under both the MIT and Apache-2.0 licenses, without any additional terms or conditions.

## How to Contribute

### Reporting Issues

- Check if the issue already exists before creating a new one
- Include relevant information: OS, SWI-Prolog version, bash version
- Provide minimal reproducible examples when possible

### Pull Requests

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Add the SPDX license header to any new **source code files** (.pl, .py, .sh):
   ```prolog
   % SPDX-License-Identifier: MIT OR Apache-2.0
   % Copyright (c) 2025 [Your Name]
   ```
   For bash files:
   ```bash
   # SPDX-License-Identifier: MIT OR Apache-2.0
   # Copyright (c) 2025 [Your Name]
   ```
   **Note:** If adding documentation or papers in a new directory, include a README.md with licensing information instead of adding headers to each file.
4. Test your changes by running the built-in tests in the core modules
5. Commit with clear, descriptive messages
6. Push to your fork and submit a pull request

### Code Style

- Follow existing Prolog conventions in the codebase
- Use meaningful predicate names
- Document complex predicates with comments
- Keep functions focused and single-purpose

### Testing

UnifyWeaver includes built-in tests in the core modules:

```prolog
?- use_module(unifyweaver(core/template_system)).
?- test_template_system.

?- use_module(unifyweaver(core/stream_compiler)).
?- test_stream_compiler.

?- use_module(unifyweaver(core/recursive_compiler)).
?- test_recursive_compiler.
```

Please ensure existing tests pass and add tests for new features.

## Questions?

Feel free to open an issue for questions or discussion about potential contributions.