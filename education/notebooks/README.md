<!--
SPDX-License-Identifier: MIT AND CC-BY-4.0
Copyright (c) 2025 John William Creighton (s243a)

This documentation is dual-licensed under MIT and CC-BY-4.0.
-->

# UnifyWeaver Jupyter Notebooks

Interactive notebooks for learning UnifyWeaver through hands-on exploration.

## Notebooks

1. **01_family_tree_tutorial.ipynb** - Introduction to UnifyWeaver compilation
   - Define Prolog facts and rules
   - Compile to Bash scripts
   - Test generated code
   - Understand transitive closure

2. **02_recursion_patterns.ipynb** - Advanced recursion patterns
   - Tail recursion with accumulators
   - Linear recursion with fold
   - Tree recursion for structures
   - Mutual recursion detection
   - Pattern comparison and performance

3. **03_call_graph_analysis.ipynb** - Code analysis and visualization
   - Build call graphs
   - Detect Strongly Connected Components (SCCs)
   - Pattern detection
   - Visualize dependencies with DOT

## Setup

### Prerequisites

1. **SWI-Prolog** (version 8.4.0 or later)
   ```bash
   # Ubuntu/Debian
   sudo apt-add-repository ppa:swi-prolog/stable
   sudo apt-get update
   sudo apt-get install swi-prolog

   # macOS
   brew install swi-prolog

   # Windows
   # Download from https://www.swi-prolog.org/Download.html
   ```

2. **Python** (version 3.8 or later)
   ```bash
   python --version  # Should be 3.8+
   ```

3. **Jupyter** (JupyterLab or Jupyter Notebook)
   ```bash
   pip install jupyterlab
   # OR
   pip install notebook
   ```

### Install Prolog Jupyter Kernel

There are several options for running Prolog in Jupyter. We recommend **prolog-jupyter-kernel**:

#### Option 1: prolog-jupyter-kernel (Recommended)

```bash
# Install the kernel
pip install prolog-jupyter-kernel

# Install the kernel spec
python -m prolog_kernel.install

# Verify installation
jupyter kernelspec list
# Should show 'prolog' in the list
```

#### Option 2: Calysto Prolog (Alternative)

```bash
pip install calysto_prolog
python -m calysto_prolog install
```

#### Option 3: jupyter-swi-prolog (Alternative)

```bash
pip install jupyter-swi-prolog
```

### Setup UnifyWeaver

1. **Clone/Download UnifyWeaver**
   ```bash
   git clone https://github.com/yourusername/UnifyWeaver.git
   cd UnifyWeaver/education/notebooks
   ```

2. **Verify UnifyWeaver is accessible**
   ```bash
   swipl
   ?- ['../init'].
   % Should load successfully
   ```

## Usage

### Start Jupyter

```bash
# From the notebooks directory
cd education/notebooks

# Start JupyterLab
jupyter lab

# OR start Jupyter Notebook
jupyter notebook
```

### Open a Notebook

1. In the Jupyter interface, click on one of the `.ipynb` files
2. Make sure the kernel is set to "SWI-Prolog" (check top-right corner)
3. Run cells with `Shift+Enter`

### Running Cells

- **Prolog cells**: Execute Prolog code directly
- **Bash cells**: Use `%%bash` magic to run shell commands
- **Markdown cells**: Documentation and explanations

### Example Session

```prolog
% Load UnifyWeaver
['../init'].

% Define a simple predicate
:- dynamic parent/2.
parent(abraham, isaac).
parent(isaac, jacob).

% Compile to Bash
use_module(unifyweaver(core/stream_compiler)).
stream_compiler:compile_facts(parent, 2, [], BashCode),
writeln(BashCode).
```

## Troubleshooting

### Kernel not found

If Jupyter doesn't show the Prolog kernel:

```bash
# Reinstall kernel spec
python -m prolog_kernel.install --user

# Check installation
jupyter kernelspec list
```

### UnifyWeaver modules not loading

Make sure you're running notebooks from the `education/notebooks` directory and that `../init.pl` exists:

```bash
ls ../init.pl  # Should exist
```

### SWI-Prolog version issues

Some kernels require specific SWI-Prolog versions. Check compatibility:

```bash
swipl --version
```

### Alternative: Use Bash Cells Only

If Prolog kernel installation is problematic, you can still use the notebooks by:

1. Running Prolog code in separate terminal
2. Using only the `%%bash` cells to test generated scripts
3. Following along with the markdown documentation

## Features

### Interactive Execution

Run Prolog queries and see results immediately:

```prolog
?- factorial(5, F).
F = 120.
```

### Inline Compilation

Compile predicates and view generated Bash code in-place:

```prolog
compile_recursive(factorial/2, [], BashCode),
writeln(BashCode).
```

### Testing Generated Scripts

Test Bash scripts directly in the notebook:

```bash
%%bash
source ../output/factorial.sh
factorial 10 ""
```

### Visualization

Generate DOT graphs for call graph visualization:

```prolog
build_call_graph([is_even/1, is_odd/1], Graph),
generate_dot(Graph, DotCode).
```

## Learning Path

**Recommended order:**

1. **Start with Notebook 1** (Family Tree Tutorial)
   - Get familiar with basic compilation
   - Understand the workflow
   - Test simple examples

2. **Continue to Notebook 2** (Recursion Patterns)
   - Learn about different recursion types
   - See pattern detection in action
   - Compare optimizations

3. **Finish with Notebook 3** (Call Graph Analysis)
   - Explore advanced analysis
   - Understand code structure
   - Visualize dependencies

## Tips

- **Save often**: Jupyter auto-saves, but manual saves (`Ctrl+S`) are good practice
- **Restart kernel**: If things get confused, use `Kernel → Restart` in the menu
- **Clear output**: Use `Cell → All Output → Clear` to clean up
- **Run all**: Use `Run → Run All Cells` to execute the entire notebook
- **Experiment**: Modify examples and see what happens!

## Additional Resources

- [UnifyWeaver Documentation](../README)
- [Chapter 4: Your First Program](../04_your_first_program)
- [Chapter 9: Advanced Recursion Patterns](../09_advanced_recursion)
- [Chapter 10: Prolog Introspection](../10_prolog_introspection)
- [SWI-Prolog Documentation](https://www.swi-prolog.org/pldoc/doc_for?object=manual)
- [Prolog Jupyter Kernel GitHub](https://github.com/hhu-stups/prolog-jupyter-kernel)

## Contributing

Found an issue or have an improvement? Please:

1. Check existing issues
2. Create a new issue with details
3. Or submit a pull request!

## License

These notebooks are part of the UnifyWeaver education materials and are licensed under:
- Code examples: MIT OR Apache-2.0
- Documentation/text: CC-BY-4.0

See parent directory LICENSE files for details.

## Support

For questions or help:
- Open an issue on GitHub
- Check the UnifyWeaver documentation
- Review the education chapters

Happy learning! 🎓
