% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% WebWorker Generator - Offload Heavy Computation to Background Threads
%
% This module provides WebWorker patterns for offloading heavy
% data processing from the main thread, keeping visualizations
% responsive even with large datasets.
%
% Usage:
%   % Define a worker configuration
%   worker_config(data_processor, [
%       operations([sort, filter, aggregate]),
%       timeout(30000)
%   ]).
%
%   % Generate worker code
%   ?- generate_worker(data_processor, Worker).

:- module(webworker_generator, [
    % Configuration
    worker_config/2,                    % worker_config(+Name, +Options)

    % Generation predicates
    generate_worker/2,                  % generate_worker(+Name, -Worker)
    generate_worker_hook/2,             % generate_worker_hook(+Name, -Hook)
    generate_worker_pool/2,             % generate_worker_pool(+Name, -Pool)
    generate_data_processor_worker/1,   % generate_data_processor_worker(-Worker)
    generate_chart_worker/1,            % generate_chart_worker(-Worker)

    % Utility predicates
    worker_operations/2,                % worker_operations(+Name, -Operations)

    % Management
    declare_worker_config/2,            % declare_worker_config(+Name, +Options)
    clear_worker_configs/0,             % clear_worker_configs

    % Testing
    test_webworker_generator/0
]).

:- use_module(library(lists)).

% ============================================================================
% DYNAMIC PREDICATES
% ============================================================================

:- dynamic worker_config/2.

:- discontiguous worker_config/2.

% ============================================================================
% DEFAULT CONFIGURATIONS
% ============================================================================

worker_config(default, [
    operations([process]),
    timeout(30000),
    transferable(true),
    error_handling(true)
]).

worker_config(data_processor, [
    operations([sort, filter, aggregate, transform, paginate]),
    timeout(60000),
    transferable(true),
    batch_size(10000)
]).

worker_config(chart_calculator, [
    operations([calculate_points, interpolate, smooth, downsample]),
    timeout(30000),
    transferable(true)
]).

worker_config(statistics, [
    operations([mean, median, stddev, percentile, correlation, regression]),
    timeout(30000)
]).

% ============================================================================
% WORKER GENERATION
% ============================================================================

%% generate_worker(+Name, -Worker)
%  Generate a WebWorker script.
generate_worker(Name, Worker) :-
    (worker_config(Name, Config) -> true ; worker_config(default, Config)),
    (member(operations(Operations), Config) -> true ; Operations = [process]),
    atom_string(Name, NameStr),
    generate_operations_code(Operations, OpsCode),
    format(atom(Worker), '// ~w Worker
// Auto-generated WebWorker for background processing

const operations = {
~w
};

// Message handler
self.onmessage = async (event) => {
  const { id, operation, data, options = {} } = event.data;

  try {
    if (!operations[operation]) {
      throw new Error(`Unknown operation: ${operation}`);
    }

    const startTime = performance.now();
    const result = await operations[operation](data, options);
    const duration = performance.now() - startTime;

    // Use transferable objects for ArrayBuffers
    const transfer: Transferable[] = [];
    if (result instanceof ArrayBuffer) {
      transfer.push(result);
    } else if (Array.isArray(result) && result[0] instanceof ArrayBuffer) {
      transfer.push(...result);
    }

    self.postMessage({
      id,
      success: true,
      result,
      duration
    }, { transfer });

  } catch (error) {
    self.postMessage({
      id,
      success: false,
      error: error instanceof Error ? error.message : String(error)
    });
  }
};

// Utility functions
function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}

function createTypedArray<T extends Float64Array | Float32Array | Int32Array>(
  data: number[],
  type: "float64" | "float32" | "int32" = "float64"
): T {
  switch (type) {
    case "float32":
      return new Float32Array(data) as T;
    case "int32":
      return new Int32Array(data) as T;
    default:
      return new Float64Array(data) as T;
  }
}
', [NameStr, OpsCode]).

generate_operations_code(Operations, Code) :-
    findall(OpCode, (
        member(Op, Operations),
        generate_operation_code(Op, OpCode)
    ), OpCodes),
    atomic_list_concat(OpCodes, ',\n\n', Code).

generate_operation_code(sort, Code) :-
    format(atom(Code), '  sort: (data: unknown[], options: { key?: string; direction?: "asc" | "desc" } = {}) => {
    const { key, direction = "asc" } = options;
    const sorted = [...data];

    sorted.sort((a, b) => {
      const aVal = key ? (a as Record<string, unknown>)[key] : a;
      const bVal = key ? (b as Record<string, unknown>)[key] : b;

      if (aVal < bVal) return direction === "asc" ? -1 : 1;
      if (aVal > bVal) return direction === "asc" ? 1 : -1;
      return 0;
    });

    return sorted;
  }', []).

generate_operation_code(filter, Code) :-
    format(atom(Code), '  filter: (data: unknown[], options: { predicate?: string; conditions?: Record<string, unknown> } = {}) => {
    const { predicate, conditions } = options;

    if (predicate) {
      const filterFn = new Function("item", "index", `return ${predicate}`);
      return data.filter((item, index) => filterFn(item, index));
    }

    if (conditions) {
      return data.filter(item => {
        const record = item as Record<string, unknown>;
        return Object.entries(conditions).every(([key, value]) => {
          if (typeof value === "object" && value !== null) {
            const { $gt, $gte, $lt, $lte, $eq, $ne, $in } = value as Record<string, unknown>;
            const itemValue = record[key];
            if ($gt !== undefined) return (itemValue as number) > ($gt as number);
            if ($gte !== undefined) return (itemValue as number) >= ($gte as number);
            if ($lt !== undefined) return (itemValue as number) < ($lt as number);
            if ($lte !== undefined) return (itemValue as number) <= ($lte as number);
            if ($eq !== undefined) return itemValue === $eq;
            if ($ne !== undefined) return itemValue !== $ne;
            if ($in !== undefined) return ($in as unknown[]).includes(itemValue);
          }
          return record[key] === value;
        });
      });
    }

    return data;
  }', []).

generate_operation_code(aggregate, Code) :-
    format(atom(Code), '  aggregate: (data: unknown[], options: { groupBy?: string; aggregations: Record<string, { field: string; fn: string }> }) => {
    const { groupBy, aggregations } = options;
    const groups = new Map<string, unknown[]>();

    // Group data
    data.forEach(item => {
      const record = item as Record<string, unknown>;
      const key = groupBy ? String(record[groupBy]) : "__all__";
      if (!groups.has(key)) groups.set(key, []);
      groups.get(key)!.push(item);
    });

    // Apply aggregations
    const result: Record<string, Record<string, unknown>>[] = [];

    groups.forEach((items, key) => {
      const aggregated: Record<string, unknown> = groupBy ? { [groupBy]: key } : {};

      Object.entries(aggregations).forEach(([name, { field, fn }]) => {
        const values = items.map(item => (item as Record<string, unknown>)[field] as number);

        switch (fn) {
          case "sum":
            aggregated[name] = values.reduce((a, b) => a + b, 0);
            break;
          case "avg":
            aggregated[name] = values.reduce((a, b) => a + b, 0) / values.length;
            break;
          case "min":
            aggregated[name] = Math.min(...values);
            break;
          case "max":
            aggregated[name] = Math.max(...values);
            break;
          case "count":
            aggregated[name] = values.length;
            break;
        }
      });

      result.push(aggregated);
    });

    return result;
  }', []).

generate_operation_code(transform, Code) :-
    format(atom(Code), '  transform: (data: unknown[], options: { mapper: string }) => {
    const mapperFn = new Function("item", "index", `return ${options.mapper}`);
    return data.map((item, index) => mapperFn(item, index));
  }', []).

generate_operation_code(paginate, Code) :-
    format(atom(Code), '  paginate: (data: unknown[], options: { page: number; pageSize: number }) => {
    const { page, pageSize } = options;
    const start = (page - 1) * pageSize;
    const end = start + pageSize;
    return {
      data: data.slice(start, end),
      total: data.length,
      page,
      pageSize,
      totalPages: Math.ceil(data.length / pageSize)
    };
  }', []).

generate_operation_code(calculate_points, Code) :-
    format(atom(Code), '  calculate_points: (data: number[], options: { xMin?: number; xMax?: number; steps?: number } = {}) => {
    const { xMin = 0, xMax = data.length - 1, steps = data.length } = options;
    const points: [number, number][] = [];
    const step = (xMax - xMin) / (steps - 1);

    for (let i = 0; i < steps; i++) {
      const x = xMin + i * step;
      const idx = Math.round(x);
      const y = data[Math.min(idx, data.length - 1)];
      points.push([x, y]);
    }

    return points;
  }', []).

generate_operation_code(interpolate, Code) :-
    format(atom(Code), '  interpolate: (data: [number, number][], options: { method?: "linear" | "cubic"; points?: number } = {}) => {
    const { method = "linear", points = data.length * 2 } = options;

    if (method === "linear") {
      const result: [number, number][] = [];
      const xMin = data[0][0];
      const xMax = data[data.length - 1][0];
      const step = (xMax - xMin) / (points - 1);

      for (let i = 0; i < points; i++) {
        const x = xMin + i * step;
        // Find surrounding points
        let j = 0;
        while (j < data.length - 1 && data[j + 1][0] < x) j++;

        if (j === data.length - 1) {
          result.push([x, data[j][1]]);
        } else {
          const t = (x - data[j][0]) / (data[j + 1][0] - data[j][0]);
          const y = data[j][1] + t * (data[j + 1][1] - data[j][1]);
          result.push([x, y]);
        }
      }

      return result;
    }

    return data;
  }', []).

generate_operation_code(smooth, Code) :-
    format(atom(Code), '  smooth: (data: number[], options: { windowSize?: number; method?: "moving_avg" | "gaussian" } = {}) => {
    const { windowSize = 5, method = "moving_avg" } = options;
    const result: number[] = [];
    const half = Math.floor(windowSize / 2);

    for (let i = 0; i < data.length; i++) {
      let sum = 0;
      let count = 0;

      for (let j = -half; j <= half; j++) {
        const idx = i + j;
        if (idx >= 0 && idx < data.length) {
          if (method === "gaussian") {
            const weight = Math.exp(-(j * j) / (2 * half * half));
            sum += data[idx] * weight;
            count += weight;
          } else {
            sum += data[idx];
            count++;
          }
        }
      }

      result.push(sum / count);
    }

    return result;
  }', []).

generate_operation_code(downsample, Code) :-
    format(atom(Code), '  downsample: (data: [number, number][], options: { targetPoints?: number; method?: "lttb" | "average" } = {}) => {
    const { targetPoints = Math.min(1000, data.length), method = "lttb" } = options;

    if (data.length <= targetPoints) return data;

    if (method === "lttb") {
      // Largest Triangle Three Buckets algorithm
      const sampled: [number, number][] = [data[0]];
      const bucketSize = (data.length - 2) / (targetPoints - 2);

      let a = 0;
      for (let i = 0; i < targetPoints - 2; i++) {
        const bucketStart = Math.floor((i + 1) * bucketSize) + 1;
        const bucketEnd = Math.min(Math.floor((i + 2) * bucketSize) + 1, data.length - 1);

        // Calculate average of next bucket
        let avgX = 0, avgY = 0;
        for (let j = bucketStart; j < bucketEnd; j++) {
          avgX += data[j][0];
          avgY += data[j][1];
        }
        avgX /= (bucketEnd - bucketStart);
        avgY /= (bucketEnd - bucketStart);

        // Find point with largest triangle area
        let maxArea = -1;
        let maxIdx = bucketStart;
        for (let j = bucketStart; j < bucketEnd; j++) {
          const area = Math.abs(
            (data[a][0] - avgX) * (data[j][1] - data[a][1]) -
            (data[a][0] - data[j][0]) * (avgY - data[a][1])
          );
          if (area > maxArea) {
            maxArea = area;
            maxIdx = j;
          }
        }

        sampled.push(data[maxIdx]);
        a = maxIdx;
      }

      sampled.push(data[data.length - 1]);
      return sampled;
    }

    // Average-based downsampling
    const result: [number, number][] = [];
    const step = data.length / targetPoints;

    for (let i = 0; i < targetPoints; i++) {
      const start = Math.floor(i * step);
      const end = Math.floor((i + 1) * step);
      let sumX = 0, sumY = 0;
      for (let j = start; j < end; j++) {
        sumX += data[j][0];
        sumY += data[j][1];
      }
      result.push([sumX / (end - start), sumY / (end - start)]);
    }

    return result;
  }', []).

generate_operation_code(mean, Code) :-
    format(atom(Code), '  mean: (data: number[]) => {
    return data.reduce((a, b) => a + b, 0) / data.length;
  }', []).

generate_operation_code(median, Code) :-
    format(atom(Code), '  median: (data: number[]) => {
    const sorted = [...data].sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);
    return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
  }', []).

generate_operation_code(stddev, Code) :-
    format(atom(Code), '  stddev: (data: number[]) => {
    const mean = data.reduce((a, b) => a + b, 0) / data.length;
    const squareDiffs = data.map(x => Math.pow(x - mean, 2));
    return Math.sqrt(squareDiffs.reduce((a, b) => a + b, 0) / data.length);
  }', []).

generate_operation_code(percentile, Code) :-
    format(atom(Code), '  percentile: (data: number[], options: { p: number }) => {
    const sorted = [...data].sort((a, b) => a - b);
    const idx = (options.p / 100) * (sorted.length - 1);
    const lower = Math.floor(idx);
    const upper = Math.ceil(idx);
    const weight = idx - lower;
    return sorted[lower] * (1 - weight) + sorted[upper] * weight;
  }', []).

generate_operation_code(correlation, Code) :-
    format(atom(Code), '  correlation: (data: { x: number[]; y: number[] }) => {
    const { x, y } = data;
    const n = x.length;
    const sumX = x.reduce((a, b) => a + b, 0);
    const sumY = y.reduce((a, b) => a + b, 0);
    const sumXY = x.reduce((acc, xi, i) => acc + xi * y[i], 0);
    const sumX2 = x.reduce((a, b) => a + b * b, 0);
    const sumY2 = y.reduce((a, b) => a + b * b, 0);

    const numerator = n * sumXY - sumX * sumY;
    const denominator = Math.sqrt(
      (n * sumX2 - sumX * sumX) * (n * sumY2 - sumY * sumY)
    );

    return denominator === 0 ? 0 : numerator / denominator;
  }', []).

generate_operation_code(regression, Code) :-
    format(atom(Code), '  regression: (data: { x: number[]; y: number[] }) => {
    const { x, y } = data;
    const n = x.length;
    const sumX = x.reduce((a, b) => a + b, 0);
    const sumY = y.reduce((a, b) => a + b, 0);
    const sumXY = x.reduce((acc, xi, i) => acc + xi * y[i], 0);
    const sumX2 = x.reduce((a, b) => a + b * b, 0);

    const slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
    const intercept = (sumY - slope * sumX) / n;

    return { slope, intercept };
  }', []).

generate_operation_code(process, Code) :-
    format(atom(Code), '  process: (data: unknown) => {
    return data;
  }', []).

% ============================================================================
% WORKER HOOK GENERATION
% ============================================================================

%% generate_worker_hook(+Name, -Hook)
%  Generate a useWorker React hook.
generate_worker_hook(Name, Hook) :-
    (worker_config(Name, Config) -> true ; worker_config(default, Config)),
    (member(timeout(Timeout), Config) -> true ; Timeout = 30000),
    atom_string(Name, NameStr),
    format(atom(Hook), 'import { useRef, useCallback, useEffect, useState } from "react";

interface WorkerMessage<T> {
  id: string;
  operation: string;
  data: T;
  options?: Record<string, unknown>;
}

interface WorkerResponse<R> {
  id: string;
  success: boolean;
  result?: R;
  error?: string;
  duration?: number;
}

interface UseWorkerOptions {
  timeout?: number;
  onError?: (error: Error) => void;
}

interface UseWorkerResult<T, R> {
  execute: (operation: string, data: T, options?: Record<string, unknown>) => Promise<R>;
  isProcessing: boolean;
  error: Error | null;
  lastDuration: number | null;
  terminate: () => void;
}

export const useWorker~w = <T, R>(
  workerFactory: () => Worker,
  options: UseWorkerOptions = {}
): UseWorkerResult<T, R> => {
  const { timeout = ~w, onError } = options;

  const workerRef = useRef<Worker | null>(null);
  const pendingRef = useRef<Map<string, {
    resolve: (value: R) => void;
    reject: (error: Error) => void;
    timeoutId: NodeJS.Timeout;
  }>>(new Map());

  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const [lastDuration, setLastDuration] = useState<number | null>(null);

  // Initialize worker
  useEffect(() => {
    workerRef.current = workerFactory();

    workerRef.current.onmessage = (event: MessageEvent<WorkerResponse<R>>) => {
      const { id, success, result, error: errorMsg, duration } = event.data;

      const pending = pendingRef.current.get(id);
      if (!pending) return;

      clearTimeout(pending.timeoutId);
      pendingRef.current.delete(id);

      if (pendingRef.current.size === 0) {
        setIsProcessing(false);
      }

      if (duration !== undefined) {
        setLastDuration(duration);
      }

      if (success && result !== undefined) {
        pending.resolve(result);
      } else {
        const error = new Error(errorMsg || "Worker operation failed");
        setError(error);
        onError?.(error);
        pending.reject(error);
      }
    };

    workerRef.current.onerror = (event) => {
      const error = new Error(event.message);
      setError(error);
      onError?.(error);

      // Reject all pending operations
      pendingRef.current.forEach(({ reject }) => reject(error));
      pendingRef.current.clear();
      setIsProcessing(false);
    };

    return () => {
      workerRef.current?.terminate();
      pendingRef.current.forEach(({ timeoutId }) => clearTimeout(timeoutId));
      pendingRef.current.clear();
    };
  }, [workerFactory, onError]);

  const execute = useCallback(
    (operation: string, data: T, options?: Record<string, unknown>): Promise<R> => {
      return new Promise((resolve, reject) => {
        if (!workerRef.current) {
          reject(new Error("Worker not initialized"));
          return;
        }

        const id = Math.random().toString(36).substring(2, 15);

        const timeoutId = setTimeout(() => {
          pendingRef.current.delete(id);
          if (pendingRef.current.size === 0) {
            setIsProcessing(false);
          }
          const error = new Error(`Worker operation timed out after ${timeout}ms`);
          setError(error);
          onError?.(error);
          reject(error);
        }, timeout);

        pendingRef.current.set(id, { resolve, reject, timeoutId });
        setIsProcessing(true);
        setError(null);

        const message: WorkerMessage<T> = { id, operation, data, options };
        workerRef.current.postMessage(message);
      });
    },
    [timeout, onError]
  );

  const terminate = useCallback(() => {
    workerRef.current?.terminate();
    workerRef.current = null;
    pendingRef.current.forEach(({ timeoutId, reject }) => {
      clearTimeout(timeoutId);
      reject(new Error("Worker terminated"));
    });
    pendingRef.current.clear();
    setIsProcessing(false);
  }, []);

  return {
    execute,
    isProcessing,
    error,
    lastDuration,
    terminate
  };
};
', [NameStr, Timeout]).

% ============================================================================
% WORKER POOL GENERATION
% ============================================================================

%% generate_worker_pool(+Name, -Pool)
%  Generate a worker pool manager.
generate_worker_pool(Name, Pool) :-
    atom_string(Name, NameStr),
    format(atom(Pool), 'interface WorkerPoolOptions {
  workerFactory: () => Worker;
  maxWorkers?: number;
  idleTimeout?: number;
}

interface PooledTask<T, R> {
  id: string;
  operation: string;
  data: T;
  options?: Record<string, unknown>;
  resolve: (value: R) => void;
  reject: (error: Error) => void;
}

interface PooledWorker {
  worker: Worker;
  busy: boolean;
  idleTimer?: NodeJS.Timeout;
}

export class WorkerPool~w<T, R> {
  private workers: PooledWorker[] = [];
  private taskQueue: PooledTask<T, R>[] = [];
  private options: Required<WorkerPoolOptions>;

  constructor(options: WorkerPoolOptions) {
    this.options = {
      maxWorkers: navigator.hardwareConcurrency || 4,
      idleTimeout: 60000,
      ...options
    };
  }

  async execute(operation: string, data: T, options?: Record<string, unknown>): Promise<R> {
    return new Promise((resolve, reject) => {
      const task: PooledTask<T, R> = {
        id: Math.random().toString(36).substring(2, 15),
        operation,
        data,
        options,
        resolve,
        reject
      };

      this.taskQueue.push(task);
      this.processQueue();
    });
  }

  private processQueue(): void {
    while (this.taskQueue.length > 0) {
      const worker = this.getAvailableWorker();
      if (!worker) break;

      const task = this.taskQueue.shift()!;
      this.executeTask(worker, task);
    }
  }

  private getAvailableWorker(): PooledWorker | null {
    // Find idle worker
    let worker = this.workers.find(w => !w.busy);

    if (!worker && this.workers.length < this.options.maxWorkers) {
      // Create new worker
      worker = {
        worker: this.options.workerFactory(),
        busy: false
      };
      this.setupWorkerHandlers(worker);
      this.workers.push(worker);
    }

    return worker || null;
  }

  private setupWorkerHandlers(pooledWorker: PooledWorker): void {
    pooledWorker.worker.onmessage = (event) => {
      // Task completion handled in executeTask
    };

    pooledWorker.worker.onerror = (event) => {
      console.error("Worker error:", event.message);
      pooledWorker.busy = false;
      this.processQueue();
    };
  }

  private executeTask(pooledWorker: PooledWorker, task: PooledTask<T, R>): void {
    if (pooledWorker.idleTimer) {
      clearTimeout(pooledWorker.idleTimer);
    }

    pooledWorker.busy = true;

    const handler = (event: MessageEvent) => {
      const { id, success, result, error } = event.data;

      if (id !== task.id) return;

      pooledWorker.worker.removeEventListener("message", handler);
      pooledWorker.busy = false;

      if (success) {
        task.resolve(result);
      } else {
        task.reject(new Error(error));
      }

      // Set idle timer
      pooledWorker.idleTimer = setTimeout(() => {
        this.terminateWorker(pooledWorker);
      }, this.options.idleTimeout);

      this.processQueue();
    };

    pooledWorker.worker.addEventListener("message", handler);
    pooledWorker.worker.postMessage({
      id: task.id,
      operation: task.operation,
      data: task.data,
      options: task.options
    });
  }

  private terminateWorker(pooledWorker: PooledWorker): void {
    if (pooledWorker.busy) return;

    pooledWorker.worker.terminate();
    const index = this.workers.indexOf(pooledWorker);
    if (index !== -1) {
      this.workers.splice(index, 1);
    }
  }

  terminate(): void {
    this.workers.forEach(w => w.worker.terminate());
    this.workers = [];
    this.taskQueue.forEach(t => t.reject(new Error("Pool terminated")));
    this.taskQueue = [];
  }

  get activeWorkers(): number {
    return this.workers.length;
  }

  get pendingTasks(): number {
    return this.taskQueue.length;
  }
}
', [NameStr]).

% ============================================================================
% SPECIALIZED WORKERS
% ============================================================================

%% generate_data_processor_worker(-Worker)
%  Generate a data processor worker.
generate_data_processor_worker(Worker) :-
    generate_worker(data_processor, Worker).

%% generate_chart_worker(-Worker)
%  Generate a chart calculation worker.
generate_chart_worker(Worker) :-
    generate_worker(chart_calculator, Worker).

% ============================================================================
% UTILITY PREDICATES
% ============================================================================

%% worker_operations(+Name, -Operations)
%  Get the operations for a worker configuration.
worker_operations(Name, Operations) :-
    worker_config(Name, Config),
    member(operations(Operations), Config), !.
worker_operations(_, [process]).

% ============================================================================
% MANAGEMENT PREDICATES
% ============================================================================

%% declare_worker_config(+Name, +Options)
%  Declare a worker configuration.
declare_worker_config(Name, Options) :-
    retractall(worker_config(Name, _)),
    assertz(worker_config(Name, Options)).

%% clear_worker_configs/0
%  Clear all worker configurations.
clear_worker_configs :-
    retractall(worker_config(_, _)).

% ============================================================================
% TESTING
% ============================================================================

test_webworker_generator :-
    writeln('Testing webworker generator...'),

    % Test config existence
    (worker_config(default, _) ->
        writeln('  [PASS] default config exists') ;
        writeln('  [FAIL] default config')),

    (worker_config(data_processor, _) ->
        writeln('  [PASS] data_processor config exists') ;
        writeln('  [FAIL] data_processor config')),

    % Test worker generation
    (generate_worker(data_processor, Worker), atom_length(Worker, WL), WL > 1000 ->
        writeln('  [PASS] generate_worker produces code') ;
        writeln('  [FAIL] generate_worker')),

    % Test hook generation
    (generate_worker_hook(default, Hook), atom_length(Hook, HL), HL > 1000 ->
        writeln('  [PASS] generate_worker_hook produces code') ;
        writeln('  [FAIL] generate_worker_hook')),

    % Test pool generation
    (generate_worker_pool(default, Pool), atom_length(Pool, PL), PL > 500 ->
        writeln('  [PASS] generate_worker_pool produces code') ;
        writeln('  [FAIL] generate_worker_pool')),

    % Test specialized workers
    (generate_data_processor_worker(DPW), atom_length(DPW, DL), DL > 1000 ->
        writeln('  [PASS] generate_data_processor_worker produces code') ;
        writeln('  [FAIL] generate_data_processor_worker')),

    (generate_chart_worker(CW), atom_length(CW, CL), CL > 500 ->
        writeln('  [PASS] generate_chart_worker produces code') ;
        writeln('  [FAIL] generate_chart_worker')),

    writeln('WebWorker generator tests complete.').
