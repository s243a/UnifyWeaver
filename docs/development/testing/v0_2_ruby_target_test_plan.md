# Ruby Target Test Plan - v0.2

**Version**: 0.2
**Date**: January 2026
**Status**: Draft
**Scope**: Ruby code generation target testing

## Overview

This test plan covers the Ruby target for UnifyWeaver, which generates idiomatic Ruby code for scripting, web backends, and data processing.

## Prerequisites

### System Requirements

- Ruby 3.0+ (3.2+ recommended)
- Bundler 2.0+
- SWI-Prolog 9.0+
- UnifyWeaver repository cloned

### Verification

```bash
# Verify Ruby installation
ruby --version
gem --version
bundle --version

# Verify Prolog
swipl --version
```

## Test Categories

### 1. Unit Tests (Code Generation Only)

#### 1.1 Basic Generator Tests

```bash
swipl -g "use_module('tests/core/test_ruby_generator'), run_tests" -t halt
```

**Test Cases**:
| Test | Description | Expected |
|------|-------------|----------|
| `class_generation` | Class definitions | Correct class syntax |
| `method_generation` | def/end blocks | Method definitions |
| `block_syntax` | do/end and {} | Block handling |
| `symbol_usage` | :symbol literals | Symbol generation |
| `hash_rocket` | => vs : syntax | Modern hash syntax |

#### 1.2 Ruby Idioms Tests

```bash
swipl -g "use_module('tests/core/test_ruby_idioms'), run_tests" -t halt
```

**Test Cases**:
| Test | Description | Expected |
|------|-------------|----------|
| `enumerable_methods` | map, select, reduce | Enumerable usage |
| `safe_navigation` | &. operator | Nil-safe calls |
| `string_interpolation` | #{} syntax | Interpolation |
| `heredocs` | <<~HEREDOC | Multi-line strings |

### 2. Integration Tests

#### 2.1 Script Execution

```bash
./tests/integration/test_ruby_execution.sh
```

**Test Cases**:
| Test | Description | Expected |
|------|-------------|----------|
| `basic_execution` | Run generated script | Exit code 0 |
| `stdin_processing` | ARGF/STDIN | Input handling |
| `json_output` | JSON.generate | Valid JSON |
| `require_handling` | require/require_relative | Module loading |

#### 2.2 Gem Integration

```bash
./tests/integration/test_ruby_gems.sh
```

**Test Cases**:
| Test | Description | Expected |
|------|-------------|----------|
| `bundler_install` | bundle install | Dependencies resolved |
| `gem_require` | External gems | Gems load correctly |

### 3. Generated Code Structure

```ruby
#!/usr/bin/env ruby
# frozen_string_literal: true

require 'json'
require 'set'

class GeneratedQuery
  def initialize
    @facts = Set.new
    @delta = Set.new
    init_facts
  end

  def init_facts
    @facts << { relation: :parent, args: %w[john mary] }
    @delta = @facts.dup
  end

  def solve
    until @delta.empty?
      new_facts = apply_rules
      @delta = new_facts - @facts
      @facts.merge(@delta)
    end
    @facts
  end

  def apply_rules
    Set.new.tap do |result|
      # Rule implementations
    end
  end
end

puts GeneratedQuery.new.solve.map(&:to_json)
```

## Test Commands Reference

### Quick Smoke Test

```bash
swipl -g "
    use_module('src/unifyweaver/targets/ruby_target'),
    compile_to_ruby(test_query, Code),
    format('~w~n', [Code])
" -t halt
```

### Full Test Suite

```bash
./tests/run_ruby_tests.sh
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `RUBY_VERSION` | Target Ruby version | (system) |
| `BUNDLE_PATH` | Gem installation path | (default) |
| `SKIP_RUBY_EXECUTION` | Skip runtime tests | `0` |
| `KEEP_RUBY_ARTIFACTS` | Preserve generated code | `0` |

## Known Issues

1. **Ruby 2.x compatibility**: Some syntax requires Ruby 3.0+
2. **JRuby differences**: Some features differ on JRuby
3. **Frozen string literals**: May affect string mutation
