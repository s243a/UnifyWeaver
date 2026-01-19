#!/usr/bin/env ts-node
/**
 * Constraint System Demo
 *
 * Demonstrates the constraint-based validation system.
 * Run with: ts-node constraint-demo.ts
 */

import {
  C,
  registerConstraint,
  clearConstraints,
  analyzeCommand,
  validateConstraints,
  loadConstraintsFromJson,
  EXAMPLE_CONSTRAINT_FILE
} from './index';

// ANSI colors
const GREEN = '\x1b[32m';
const RED = '\x1b[31m';
const YELLOW = '\x1b[33m';
const CYAN = '\x1b[36m';
const RESET = '\x1b[0m';

function log(msg: string): void {
  console.log(msg);
}

function section(title: string): void {
  console.log(`\n${CYAN}═══ ${title} ${'═'.repeat(50 - title.length)}${RESET}`);
}

function testCase(
  desc: string,
  command: string,
  args: string[],
  expectValid: boolean
): void {
  const result = validateConstraints(command, args);
  const passed = result.ok === expectValid;
  const status = passed ? `${GREEN}✓${RESET}` : `${RED}✗${RESET}`;

  console.log(`${status} ${desc}`);
  console.log(`  Command: ${command} ${args.join(' ')}`);

  if (!result.ok) {
    console.log(`  ${YELLOW}Violation: ${result.error}${RESET}`);
    if (result.violatingArgs) {
      console.log(`  ${YELLOW}Args: ${result.violatingArgs.join(', ')}${RESET}`);
    }
  }

  if (!passed) {
    console.log(`  ${RED}Expected ${expectValid ? 'valid' : 'invalid'}, got ${result.ok ? 'valid' : 'invalid'}${RESET}`);
  }
}

async function main(): Promise<void> {
  console.log(`${CYAN}Constraint-Based Command Validation Demo${RESET}`);
  console.log('==========================================\n');

  // -------------------------------------------------------------------------
  section('1. TypeScript DSL Constraints');
  // -------------------------------------------------------------------------

  log('Registering constraints for "cp" command using TypeScript DSL...\n');

  clearConstraints();
  registerConstraint('cp', C.minArgs(2), { message: 'cp requires source and dest' });
  registerConstraint('cp', C.noShellOps());
  registerConstraint('cp', C.noTraversal());
  registerConstraint('cp', C.withinSandbox());

  testCase('Valid copy', 'cp', ['file.txt', 'backup.txt'], true);
  testCase('Missing dest', 'cp', ['file.txt'], false);
  testCase('Shell injection', 'cp', ['file.txt', '$(rm -rf /)'], false);
  testCase('Parent traversal', 'cp', ['../etc/passwd', 'stolen.txt'], false);

  // -------------------------------------------------------------------------
  section('2. JSON-Loaded Constraints (LLM-Generated)');
  // -------------------------------------------------------------------------

  log('Loading constraints from JSON (simulating DB load)...\n');

  clearConstraints();
  loadConstraintsFromJson(JSON.stringify(EXAMPLE_CONSTRAINT_FILE));

  testCase('Valid backup', './backup.sh', ['docs', 'src'], true);
  testCase('Too many args', './backup.sh', ['a', 'b', 'c', 'd', 'e', 'f'], false);
  testCase('Shell operator', './backup.sh', ['docs; rm -rf /'], false);
  testCase('Absolute path', './backup.sh', ['/etc/passwd'], false);
  testCase('Parent traversal', './backup.sh', ['../../../etc'], false);
  testCase('Binary file', './backup.sh', ['malware.exe'], false);

  // -------------------------------------------------------------------------
  section('3. Full Analysis Report');
  // -------------------------------------------------------------------------

  log('Running full constraint analysis...\n');

  clearConstraints();
  registerConstraint('rm', C.minArgs(1));
  registerConstraint('rm', C.maxArgs(3));
  registerConstraint('rm', C.noShellOps());
  registerConstraint('rm', C.noTraversal());
  registerConstraint('rm', C.withinSandbox());
  registerConstraint('rm', C.blockedFlags(['-r', '-R', '--recursive', '-f', '--force']));

  const analysis = analyzeCommand('rm', ['-rf', '../important', '$(whoami)']);

  log(`Command: rm -rf ../important $(whoami)`);
  log(`Valid: ${analysis.valid ? GREEN + 'yes' : RED + 'no'}${RESET}`);
  log(`\nConstraint Results:`);

  for (const r of analysis.results) {
    const icon = r.satisfied ? `${GREEN}✓${RESET}` : `${RED}✗${RESET}`;
    log(`  ${icon} ${r.constraint.type}: ${r.satisfied ? 'passed' : r.message}`);
    if (r.violatingArgs?.length) {
      log(`    Violating: ${r.violatingArgs.join(', ')}`);
    }
  }

  // -------------------------------------------------------------------------
  section('4. Pattern Matching Demo');
  // -------------------------------------------------------------------------

  log('Demonstrating functor pattern matching...\n');

  clearConstraints();

  // Positional constraints with pattern matching
  registerConstraint('git', C.argIn(0, ['add', 'commit', 'push', 'pull', 'status', 'log']));
  registerConstraint('git', C.argMatches(0, '^(add|commit|status)$'), {
    message: 'Only add, commit, status allowed for now'
  });

  testCase('git status (allowed)', 'git', ['status'], true);
  testCase('git add file (allowed)', 'git', ['add', 'file.txt'], true);
  testCase('git push (blocked)', 'git', ['push'], false);

  // -------------------------------------------------------------------------
  section('5. Multi-Arg Relational Constraints');
  // -------------------------------------------------------------------------

  log('Demonstrating constraints that operate on multiple args...\n');

  clearConstraints();
  registerConstraint('mv', C.minArgs(2));
  registerConstraint('mv', C.sameDir([0, 1]), {
    message: 'Source and dest must be in same directory'
  });

  testCase('mv in same dir', 'mv', ['./docs/a.txt', './docs/b.txt'], true);
  testCase('mv across dirs', 'mv', ['./docs/a.txt', './backup/a.txt'], false);

  // -------------------------------------------------------------------------
  section('Summary');
  // -------------------------------------------------------------------------

  log(`
The constraint system provides:
  • Declarative constraint facts (can be stored in DB/JSON)
  • Pattern-matching functor types via TypeScript discriminated unions
  • Generic analyzer that works for all commands
  • DSL for easy constraint definition
  • Support for multi-arg relational constraints
  • LLM-friendly JSON format for generating new constraints
`);
}

main().catch(err => {
  console.error('Error:', err);
  process.exit(1);
});
