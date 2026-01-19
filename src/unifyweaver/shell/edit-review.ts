/**
 * Edit Review - Constraint-based file edit validation
 *
 * Validates proposed file edits against declarative constraints.
 * Works alongside LLM review for defense in depth.
 *
 * @module unifyweaver/shell/edit-review
 */

import { EditConstraintFunctor, EditConstraint, reviewEdit as llmReviewEdit, GeneratorOptions } from './llm-constraint-generator';

// ============================================================================
// Edit Constraint Store
// ============================================================================

/**
 * In-memory store for edit constraints.
 * Could be extended to use database like command constraints.
 */
const editConstraintStore: EditConstraint[] = [];

/**
 * Register an edit constraint.
 */
export function registerEditConstraint(
  filePattern: string,
  constraint: EditConstraintFunctor,
  options?: { priority?: number; message?: string; source?: string }
): void {
  editConstraintStore.push({
    filePattern,
    constraint,
    priority: options?.priority ?? 0,
    message: options?.message,
    source: options?.source
  });
}

/**
 * Get constraints for a file path.
 */
export function getEditConstraints(filePath: string): EditConstraint[] {
  const { minimatch } = require('minimatch');

  return editConstraintStore
    .filter(c => minimatch(filePath, c.filePattern))
    .sort((a, b) => (b.priority ?? 0) - (a.priority ?? 0));
}

/**
 * Clear all edit constraints.
 */
export function clearEditConstraints(): void {
  editConstraintStore.length = 0;
}

// ============================================================================
// Diff Analysis
// ============================================================================

export interface LineDiff {
  type: 'add' | 'delete' | 'modify' | 'unchanged';
  lineNumber: number;
  originalContent?: string;
  newContent?: string;
}

/**
 * Compute a simple line-by-line diff.
 */
export function computeDiff(original: string, proposed: string): LineDiff[] {
  const originalLines = original.split('\n');
  const proposedLines = proposed.split('\n');
  const diffs: LineDiff[] = [];

  const maxLen = Math.max(originalLines.length, proposedLines.length);

  for (let i = 0; i < maxLen; i++) {
    const origLine = originalLines[i];
    const newLine = proposedLines[i];

    if (origLine === undefined && newLine !== undefined) {
      diffs.push({
        type: 'add',
        lineNumber: i + 1,
        newContent: newLine
      });
    } else if (origLine !== undefined && newLine === undefined) {
      diffs.push({
        type: 'delete',
        lineNumber: i + 1,
        originalContent: origLine
      });
    } else if (origLine !== newLine) {
      diffs.push({
        type: 'modify',
        lineNumber: i + 1,
        originalContent: origLine,
        newContent: newLine
      });
    } else {
      diffs.push({
        type: 'unchanged',
        lineNumber: i + 1,
        originalContent: origLine,
        newContent: newLine
      });
    }
  }

  return diffs;
}

// ============================================================================
// Constraint Evaluators for Edits
// ============================================================================

export interface EditConstraintResult {
  satisfied: boolean;
  constraint: EditConstraintFunctor;
  message?: string;
  violatingLines?: number[];
}

/**
 * Evaluate an edit constraint against a diff.
 */
export function evaluateEditConstraint(
  diff: LineDiff[],
  original: string,
  proposed: string,
  constraint: EditConstraintFunctor
): EditConstraintResult {
  const result: EditConstraintResult = {
    satisfied: true,
    constraint
  };

  switch (constraint.type) {
    case 'no_delete_lines_containing': {
      const deletedLines = diff.filter(d => d.type === 'delete' || d.type === 'modify');
      const violating: number[] = [];

      for (const line of deletedLines) {
        const content = line.originalContent || '';
        for (const pattern of constraint.patterns) {
          if (content.includes(pattern) || new RegExp(pattern).test(content)) {
            violating.push(line.lineNumber);
            break;
          }
        }
      }

      if (violating.length > 0) {
        result.satisfied = false;
        result.violatingLines = violating;
        result.message = `Cannot delete lines containing: ${constraint.patterns.join(', ')}`;
      }
      break;
    }

    case 'no_modify_lines_containing': {
      const modifiedLines = diff.filter(d => d.type === 'modify');
      const violating: number[] = [];

      for (const line of modifiedLines) {
        const content = line.originalContent || '';
        for (const pattern of constraint.patterns) {
          if (content.includes(pattern) || new RegExp(pattern).test(content)) {
            violating.push(line.lineNumber);
            break;
          }
        }
      }

      if (violating.length > 0) {
        result.satisfied = false;
        result.violatingLines = violating;
        result.message = `Cannot modify lines containing: ${constraint.patterns.join(', ')}`;
      }
      break;
    }

    case 'no_add_patterns': {
      const addedLines = diff.filter(d => d.type === 'add' || d.type === 'modify');
      const violating: number[] = [];

      for (const line of addedLines) {
        const content = line.newContent || '';
        for (const pattern of constraint.patterns) {
          if (content.includes(pattern) || new RegExp(pattern).test(content)) {
            violating.push(line.lineNumber);
            break;
          }
        }
      }

      if (violating.length > 0) {
        result.satisfied = false;
        result.violatingLines = violating;
        result.message = `Cannot add content matching: ${constraint.patterns.join(', ')}`;
      }
      break;
    }

    case 'max_lines_changed': {
      const changedCount = diff.filter(d => d.type !== 'unchanged').length;
      if (changedCount > constraint.count) {
        result.satisfied = false;
        result.message = `Too many lines changed: ${changedCount} > ${constraint.count}`;
      }
      break;
    }

    case 'no_change_imports': {
      const importPattern = /^(import|from|require)\s/;
      const changedLines = diff.filter(d => d.type === 'modify' || d.type === 'delete' || d.type === 'add');
      const violating: number[] = [];

      for (const line of changedLines) {
        const origContent = line.originalContent || '';
        const newContent = line.newContent || '';
        if (importPattern.test(origContent) || importPattern.test(newContent)) {
          violating.push(line.lineNumber);
        }
      }

      if (violating.length > 0) {
        result.satisfied = false;
        result.violatingLines = violating;
        result.message = 'Cannot modify import statements';
      }
      break;
    }

    case 'no_change_exports': {
      const exportPattern = /^export\s/;
      const changedLines = diff.filter(d => d.type === 'modify' || d.type === 'delete');
      const violating: number[] = [];

      for (const line of changedLines) {
        const content = line.originalContent || '';
        if (exportPattern.test(content)) {
          violating.push(line.lineNumber);
        }
      }

      if (violating.length > 0) {
        result.satisfied = false;
        result.violatingLines = violating;
        result.message = 'Cannot modify export statements';
      }
      break;
    }

    case 'no_remove_security_checks': {
      const securityPatterns = [
        /validate/i,
        /sanitize/i,
        /escape/i,
        /auth/i,
        /permission/i,
        /check/i,
        /verify/i,
        /assert/i
      ];

      // Check both deleted and modified lines
      const changedLines = diff.filter(d => d.type === 'delete' || d.type === 'modify');
      const violating: number[] = [];

      for (const line of changedLines) {
        const origContent = line.originalContent || '';
        const newContent = line.newContent || '';

        // Check if original had security patterns
        const origHasSecurity = securityPatterns.some(p => p.test(origContent));

        if (line.type === 'delete' && origHasSecurity) {
          // Line was deleted entirely
          violating.push(line.lineNumber);
        } else if (line.type === 'modify' && origHasSecurity) {
          // Line was modified - check if security pattern was removed
          const newHasSecurity = securityPatterns.some(p => p.test(newContent));
          if (!newHasSecurity) {
            violating.push(line.lineNumber);
          }
        }
      }

      if (violating.length > 0) {
        result.satisfied = false;
        result.violatingLines = violating;
        result.message = 'Cannot remove security-related code';
      }
      break;
    }

    case 'preserve_function': {
      const funcPattern = new RegExp(
        `(function\\s+${constraint.name}|const\\s+${constraint.name}\\s*=|${constraint.name}\\s*[:=]\\s*(async\\s+)?function)`,
        'g'
      );

      const originalHas = funcPattern.test(original);
      const proposedHas = new RegExp(funcPattern.source, 'g').test(proposed);

      if (originalHas && !proposedHas) {
        result.satisfied = false;
        result.message = `Cannot remove function: ${constraint.name}`;
      }
      break;
    }

    case 'file_must_parse': {
      // Try to parse as JavaScript/TypeScript
      try {
        // Use Function constructor as a simple syntax check
        // This won't catch all errors but catches basic syntax issues
        new Function(proposed);
      } catch (e) {
        result.satisfied = false;
        result.message = `Invalid syntax: ${e instanceof Error ? e.message : String(e)}`;
      }
      break;
    }

    case 'custom_edit':
      // Custom constraints pass by default
      result.satisfied = true;
      break;

    default:
      const _exhaustive: never = constraint;
      result.satisfied = false;
      result.message = 'Unknown constraint type';
  }

  return result;
}

// ============================================================================
// Edit Review Functions
// ============================================================================

export interface EditReviewOptions extends GeneratorOptions {
  useLLM?: boolean;  // Also run LLM review (default: false)
  requireAllPass?: boolean;  // All constraints must pass (default: true)
}

export interface EditReviewResult {
  allowed: boolean;
  constraintResults: EditConstraintResult[];
  llmResult?: {
    allowed: boolean;
    reason: string;
    risks: string[];
    suggestions: string[];
  };
  summary: string;
}

/**
 * Review a proposed file edit against registered constraints.
 */
export async function reviewFileEdit(
  filePath: string,
  originalContent: string,
  proposedContent: string,
  options: EditReviewOptions = {}
): Promise<EditReviewResult> {
  const { useLLM = false, requireAllPass = true } = options;

  // Get applicable constraints
  const constraints = getEditConstraints(filePath);

  // Compute diff
  const diff = computeDiff(originalContent, proposedContent);

  // Evaluate each constraint
  const constraintResults: EditConstraintResult[] = [];
  for (const { constraint, message } of constraints) {
    const result = evaluateEditConstraint(diff, originalContent, proposedContent, constraint);

    // Override message if custom message provided
    if (!result.satisfied && message) {
      result.message = message;
    }

    constraintResults.push(result);
  }

  // Check constraint results
  const constraintsPass = requireAllPass
    ? constraintResults.every(r => r.satisfied)
    : constraintResults.some(r => r.satisfied) || constraintResults.length === 0;

  // Optional LLM review
  let llmResult: EditReviewResult['llmResult'];
  if (useLLM) {
    try {
      const llm = await llmReviewEdit(filePath, originalContent, proposedContent, options);
      llmResult = {
        allowed: llm.allowed,
        reason: llm.reason,
        risks: llm.risks,
        suggestions: llm.suggestions
      };
    } catch (e) {
      llmResult = {
        allowed: false,
        reason: `LLM review failed: ${e instanceof Error ? e.message : String(e)}`,
        risks: ['LLM review failed'],
        suggestions: []
      };
    }
  }

  // Final decision
  const allowed = useLLM
    ? constraintsPass && (llmResult?.allowed ?? false)
    : constraintsPass;

  // Build summary
  const failedConstraints = constraintResults.filter(r => !r.satisfied);
  let summary: string;

  if (allowed) {
    summary = `Edit allowed. ${constraintResults.length} constraints checked.`;
  } else {
    const reasons: string[] = [];
    if (failedConstraints.length > 0) {
      reasons.push(`${failedConstraints.length} constraint(s) violated`);
    }
    if (llmResult && !llmResult.allowed) {
      reasons.push(`LLM rejected: ${llmResult.reason}`);
    }
    summary = `Edit blocked. ${reasons.join('. ')}`;
  }

  return {
    allowed,
    constraintResults,
    llmResult,
    summary
  };
}

// ============================================================================
// DSL for Edit Constraints
// ============================================================================

/**
 * Edit constraint builder helpers.
 */
export const EC = {
  noDeleteContaining: (patterns: string[]): EditConstraintFunctor =>
    ({ type: 'no_delete_lines_containing', patterns }),

  noModifyContaining: (patterns: string[]): EditConstraintFunctor =>
    ({ type: 'no_modify_lines_containing', patterns }),

  noAddPatterns: (patterns: string[]): EditConstraintFunctor =>
    ({ type: 'no_add_patterns', patterns }),

  maxLinesChanged: (count: number): EditConstraintFunctor =>
    ({ type: 'max_lines_changed', count }),

  noChangeImports: (): EditConstraintFunctor =>
    ({ type: 'no_change_imports' }),

  noChangeExports: (): EditConstraintFunctor =>
    ({ type: 'no_change_exports' }),

  noRemoveSecurityChecks: (): EditConstraintFunctor =>
    ({ type: 'no_remove_security_checks' }),

  preserveFunction: (name: string): EditConstraintFunctor =>
    ({ type: 'preserve_function', name }),

  fileMustParse: (): EditConstraintFunctor =>
    ({ type: 'file_must_parse' }),

  custom: (name: string, params: Record<string, unknown> = {}): EditConstraintFunctor =>
    ({ type: 'custom_edit', name, params })
};

// ============================================================================
// Default Edit Constraints
// ============================================================================

/**
 * Register default edit constraints for common security patterns.
 */
export function registerDefaultEditConstraints(): void {
  // Protect security-critical patterns in all TypeScript/JavaScript files
  registerEditConstraint('**/*.{ts,js}', EC.noRemoveSecurityChecks(), {
    priority: 10,
    message: 'Cannot remove security validation code'
  });

  // Don't allow adding shell operators in scripts
  registerEditConstraint('**/*.{sh,bash}', EC.noAddPatterns(['eval ', '`', '$(', 'rm -rf']), {
    priority: 10,
    message: 'Dangerous patterns not allowed in shell scripts'
  });

  // Protect package.json scripts
  registerEditConstraint('**/package.json', EC.noModifyContaining(['"scripts"']), {
    priority: 5,
    message: 'Cannot modify package.json scripts section'
  });

  // Limit changes to config files
  registerEditConstraint('**/*.config.{js,ts,json}', EC.maxLinesChanged(20), {
    priority: 5,
    message: 'Too many changes to config file'
  });
}
