/**
 * Constraint Store - Storage Abstraction Layer
 *
 * Provides a database/cache agnostic interface for constraint storage.
 * Implementations can use SQLite, Redis, file-based, or in-memory backends.
 *
 * @module unifyweaver/shell/constraint-store
 */

import { ConstraintFunctor, CommandConstraint } from './constraints';

// ============================================================================
// Storage Interface
// ============================================================================

/**
 * Abstract interface for constraint storage.
 * Implementations must provide these methods.
 */
export interface ConstraintStore {
  /**
   * Initialize the store (create tables, connect, etc.)
   */
  init(): Promise<void>;

  /**
   * Close the store (cleanup connections)
   */
  close(): Promise<void>;

  /**
   * Add a constraint for a command.
   */
  add(constraint: StoredConstraint): Promise<number>;

  /**
   * Add multiple constraints in a batch.
   */
  addBatch(constraints: StoredConstraint[]): Promise<number[]>;

  /**
   * Get all constraints for a command.
   * Should also include global constraints (command = '*').
   */
  getForCommand(command: string): Promise<StoredConstraint[]>;

  /**
   * Get all constraints.
   */
  getAll(): Promise<StoredConstraint[]>;

  /**
   * Delete a constraint by ID.
   */
  delete(id: number): Promise<boolean>;

  /**
   * Delete all constraints for a command.
   */
  deleteForCommand(command: string): Promise<number>;

  /**
   * Clear all constraints.
   */
  clear(): Promise<void>;

  /**
   * Check if the store has any constraints for a command.
   */
  hasConstraints(command: string): Promise<boolean>;

  /**
   * Count constraints for a command (or all if command is undefined).
   */
  count(command?: string): Promise<number>;
}

/**
 * Stored constraint with ID and metadata.
 */
export interface StoredConstraint extends CommandConstraint {
  id?: number;
  createdAt?: Date;
  updatedAt?: Date;
  source?: string;  // Where the constraint came from (e.g., 'llm', 'manual', 'default')
}

/**
 * Convert CommandConstraint to StoredConstraint.
 */
export function toStoredConstraint(
  c: CommandConstraint,
  source?: string
): StoredConstraint {
  return {
    ...c,
    source,
    createdAt: new Date()
  };
}

// ============================================================================
// In-Memory Implementation
// ============================================================================

/**
 * In-memory constraint store.
 * Fast but not persistent. Good for testing and development.
 */
export class MemoryConstraintStore implements ConstraintStore {
  private constraints: Map<number, StoredConstraint> = new Map();
  private nextId = 1;

  async init(): Promise<void> {
    // No-op for memory store
  }

  async close(): Promise<void> {
    this.constraints.clear();
  }

  async add(constraint: StoredConstraint): Promise<number> {
    const id = this.nextId++;
    this.constraints.set(id, { ...constraint, id, createdAt: new Date() });
    return id;
  }

  async addBatch(constraints: StoredConstraint[]): Promise<number[]> {
    const ids: number[] = [];
    for (const c of constraints) {
      ids.push(await this.add(c));
    }
    return ids;
  }

  async getForCommand(command: string): Promise<StoredConstraint[]> {
    const results: StoredConstraint[] = [];
    for (const c of this.constraints.values()) {
      if (c.command === command || c.command === '*') {
        results.push(c);
      }
    }
    return results.sort((a, b) => (b.priority ?? 0) - (a.priority ?? 0));
  }

  async getAll(): Promise<StoredConstraint[]> {
    return Array.from(this.constraints.values());
  }

  async delete(id: number): Promise<boolean> {
    return this.constraints.delete(id);
  }

  async deleteForCommand(command: string): Promise<number> {
    let count = 0;
    for (const [id, c] of this.constraints.entries()) {
      if (c.command === command) {
        this.constraints.delete(id);
        count++;
      }
    }
    return count;
  }

  async clear(): Promise<void> {
    this.constraints.clear();
    this.nextId = 1;
  }

  async hasConstraints(command: string): Promise<boolean> {
    for (const c of this.constraints.values()) {
      if (c.command === command || c.command === '*') {
        return true;
      }
    }
    return false;
  }

  async count(command?: string): Promise<number> {
    if (!command) {
      return this.constraints.size;
    }
    let count = 0;
    for (const c of this.constraints.values()) {
      if (c.command === command) {
        count++;
      }
    }
    return count;
  }
}

// ============================================================================
// SQLite Implementation
// ============================================================================

/**
 * SQLite-backed constraint store.
 * Persistent storage with good performance.
 */
export class SQLiteConstraintStore implements ConstraintStore {
  private db: any;  // better-sqlite3 Database
  private dbPath: string;

  constructor(dbPath: string = ':memory:') {
    this.dbPath = dbPath;
  }

  async init(): Promise<void> {
    const Database = require('better-sqlite3');
    this.db = new Database(this.dbPath);

    // Create tables
    this.db.exec(`
      CREATE TABLE IF NOT EXISTS constraints (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        command TEXT NOT NULL,
        functor_type TEXT NOT NULL,
        functor_args TEXT,  -- JSON-encoded functor arguments
        priority INTEGER DEFAULT 0,
        message TEXT,
        source TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        updated_at TEXT DEFAULT CURRENT_TIMESTAMP
      );

      CREATE INDEX IF NOT EXISTS idx_constraints_command ON constraints(command);
      CREATE INDEX IF NOT EXISTS idx_constraints_priority ON constraints(priority DESC);
    `);
  }

  async close(): Promise<void> {
    if (this.db) {
      this.db.close();
      this.db = null;
    }
  }

  async add(constraint: StoredConstraint): Promise<number> {
    const { functorType, functorArgs } = this.serializeFunctor(constraint.constraint);

    const stmt = this.db.prepare(`
      INSERT INTO constraints (command, functor_type, functor_args, priority, message, source)
      VALUES (?, ?, ?, ?, ?, ?)
    `);

    const result = stmt.run(
      constraint.command,
      functorType,
      functorArgs,
      constraint.priority ?? 0,
      constraint.message ?? null,
      constraint.source ?? null
    );

    return result.lastInsertRowid as number;
  }

  async addBatch(constraints: StoredConstraint[]): Promise<number[]> {
    const ids: number[] = [];
    const insert = this.db.transaction((constraints: StoredConstraint[]) => {
      for (const c of constraints) {
        const { functorType, functorArgs } = this.serializeFunctor(c.constraint);
        const stmt = this.db.prepare(`
          INSERT INTO constraints (command, functor_type, functor_args, priority, message, source)
          VALUES (?, ?, ?, ?, ?, ?)
        `);
        const result = stmt.run(
          c.command,
          functorType,
          functorArgs,
          c.priority ?? 0,
          c.message ?? null,
          c.source ?? null
        );
        ids.push(result.lastInsertRowid as number);
      }
    });
    insert(constraints);
    return ids;
  }

  async getForCommand(command: string): Promise<StoredConstraint[]> {
    const stmt = this.db.prepare(`
      SELECT * FROM constraints
      WHERE command = ? OR command = '*'
      ORDER BY priority DESC
    `);

    const rows = stmt.all(command);
    return rows.map((row: any) => this.rowToConstraint(row));
  }

  async getAll(): Promise<StoredConstraint[]> {
    const stmt = this.db.prepare(`SELECT * FROM constraints ORDER BY priority DESC`);
    const rows = stmt.all();
    return rows.map((row: any) => this.rowToConstraint(row));
  }

  async delete(id: number): Promise<boolean> {
    const stmt = this.db.prepare(`DELETE FROM constraints WHERE id = ?`);
    const result = stmt.run(id);
    return result.changes > 0;
  }

  async deleteForCommand(command: string): Promise<number> {
    const stmt = this.db.prepare(`DELETE FROM constraints WHERE command = ?`);
    const result = stmt.run(command);
    return result.changes;
  }

  async clear(): Promise<void> {
    this.db.exec(`DELETE FROM constraints`);
  }

  async hasConstraints(command: string): Promise<boolean> {
    const stmt = this.db.prepare(`
      SELECT 1 FROM constraints
      WHERE command = ? OR command = '*'
      LIMIT 1
    `);
    const row = stmt.get(command);
    return row !== undefined;
  }

  async count(command?: string): Promise<number> {
    if (!command) {
      const stmt = this.db.prepare(`SELECT COUNT(*) as count FROM constraints`);
      return stmt.get().count;
    }
    const stmt = this.db.prepare(`SELECT COUNT(*) as count FROM constraints WHERE command = ?`);
    return stmt.get(command).count;
  }

  /**
   * Serialize a ConstraintFunctor to type + JSON args.
   */
  private serializeFunctor(functor: ConstraintFunctor): { functorType: string; functorArgs: string | null } {
    const { type, ...rest } = functor as any;
    const args = Object.keys(rest).length > 0 ? JSON.stringify(rest) : null;
    return { functorType: type, functorArgs: args };
  }

  /**
   * Deserialize a database row to StoredConstraint.
   */
  private rowToConstraint(row: any): StoredConstraint {
    const args = row.functor_args ? JSON.parse(row.functor_args) : {};
    const constraint: ConstraintFunctor = { type: row.functor_type, ...args } as ConstraintFunctor;

    return {
      id: row.id,
      command: row.command,
      constraint,
      priority: row.priority,
      message: row.message,
      source: row.source,
      createdAt: row.created_at ? new Date(row.created_at) : undefined,
      updatedAt: row.updated_at ? new Date(row.updated_at) : undefined
    };
  }
}

// ============================================================================
// File-based Implementation (JSON)
// ============================================================================

/**
 * File-based constraint store using JSON.
 * Simple persistence without external dependencies.
 */
export class FileConstraintStore implements ConstraintStore {
  private filePath: string;
  private constraints: Map<number, StoredConstraint> = new Map();
  private nextId = 1;

  constructor(filePath: string) {
    this.filePath = filePath;
  }

  async init(): Promise<void> {
    const { existsSync, readFileSync } = require('fs');

    if (existsSync(this.filePath)) {
      try {
        const data = JSON.parse(readFileSync(this.filePath, 'utf-8'));
        this.constraints = new Map(data.constraints.map((c: any) => [c.id, c]));
        this.nextId = data.nextId || 1;
      } catch {
        // File exists but invalid - start fresh
        this.constraints = new Map();
        this.nextId = 1;
      }
    }
  }

  async close(): Promise<void> {
    await this.save();
  }

  private async save(): Promise<void> {
    const { writeFileSync, mkdirSync } = require('fs');
    const { dirname } = require('path');

    mkdirSync(dirname(this.filePath), { recursive: true });

    const data = {
      nextId: this.nextId,
      constraints: Array.from(this.constraints.values())
    };

    writeFileSync(this.filePath, JSON.stringify(data, null, 2));
  }

  async add(constraint: StoredConstraint): Promise<number> {
    const id = this.nextId++;
    this.constraints.set(id, { ...constraint, id, createdAt: new Date() });
    await this.save();
    return id;
  }

  async addBatch(constraints: StoredConstraint[]): Promise<number[]> {
    const ids: number[] = [];
    for (const c of constraints) {
      const id = this.nextId++;
      this.constraints.set(id, { ...c, id, createdAt: new Date() });
      ids.push(id);
    }
    await this.save();
    return ids;
  }

  async getForCommand(command: string): Promise<StoredConstraint[]> {
    const results: StoredConstraint[] = [];
    for (const c of this.constraints.values()) {
      if (c.command === command || c.command === '*') {
        results.push(c);
      }
    }
    return results.sort((a, b) => (b.priority ?? 0) - (a.priority ?? 0));
  }

  async getAll(): Promise<StoredConstraint[]> {
    return Array.from(this.constraints.values());
  }

  async delete(id: number): Promise<boolean> {
    const deleted = this.constraints.delete(id);
    if (deleted) await this.save();
    return deleted;
  }

  async deleteForCommand(command: string): Promise<number> {
    let count = 0;
    for (const [id, c] of this.constraints.entries()) {
      if (c.command === command) {
        this.constraints.delete(id);
        count++;
      }
    }
    if (count > 0) await this.save();
    return count;
  }

  async clear(): Promise<void> {
    this.constraints.clear();
    this.nextId = 1;
    await this.save();
  }

  async hasConstraints(command: string): Promise<boolean> {
    for (const c of this.constraints.values()) {
      if (c.command === command || c.command === '*') {
        return true;
      }
    }
    return false;
  }

  async count(command?: string): Promise<number> {
    if (!command) {
      return this.constraints.size;
    }
    let count = 0;
    for (const c of this.constraints.values()) {
      if (c.command === command) {
        count++;
      }
    }
    return count;
  }
}

// ============================================================================
// Factory Function
// ============================================================================

export type StoreType = 'memory' | 'sqlite' | 'file';

export interface StoreOptions {
  type: StoreType;
  path?: string;  // For sqlite and file stores
}

/**
 * Create a constraint store based on options.
 */
export function createConstraintStore(options: StoreOptions): ConstraintStore {
  switch (options.type) {
    case 'memory':
      return new MemoryConstraintStore();

    case 'sqlite':
      return new SQLiteConstraintStore(options.path || ':memory:');

    case 'file':
      if (!options.path) {
        throw new Error('File store requires a path');
      }
      return new FileConstraintStore(options.path);

    default:
      throw new Error(`Unknown store type: ${options.type}`);
  }
}

// ============================================================================
// Global Store Instance
// ============================================================================

let globalStore: ConstraintStore | null = null;

/**
 * Get the global constraint store.
 * Creates a memory store if none is set.
 */
export function getStore(): ConstraintStore {
  if (!globalStore) {
    globalStore = new MemoryConstraintStore();
  }
  return globalStore;
}

/**
 * Set the global constraint store.
 */
export function setStore(store: ConstraintStore): void {
  globalStore = store;
}

/**
 * Initialize the global store with options.
 */
export async function initStore(options: StoreOptions): Promise<ConstraintStore> {
  const store = createConstraintStore(options);
  await store.init();
  setStore(store);
  return store;
}
