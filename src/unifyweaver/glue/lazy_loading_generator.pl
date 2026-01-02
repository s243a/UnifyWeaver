% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% Lazy Loading Generator - On-demand Data Loading for Large Datasets
%
% This module provides lazy loading patterns for visualizations dealing
% with large datasets, including pagination, infinite scroll, and
% incremental loading.
%
% Usage:
%   % Define a lazy loading configuration
%   lazy_config(my_chart, [
%       strategy(pagination),
%       page_size(50),
%       prefetch(1)
%   ]).
%
%   % Generate lazy loading hook
%   ?- generate_lazy_hook(my_chart, Hook).

:- module(lazy_loading_generator, [
    % Configuration
    lazy_config/2,                  % lazy_config(+Name, +Options)

    % Generation predicates
    generate_lazy_hook/2,           % generate_lazy_hook(+Name, -Hook)
    generate_lazy_loader/2,         % generate_lazy_loader(+Name, -Loader)
    generate_pagination_hook/2,     % generate_pagination_hook(+Name, -Hook)
    generate_infinite_scroll/2,     % generate_infinite_scroll(+Name, -Hook)
    generate_lazy_component/2,      % generate_lazy_component(+Name, -Component)

    % Utility predicates
    lazy_strategy/2,                % lazy_strategy(+Name, -Strategy)

    % Management
    declare_lazy_config/2,          % declare_lazy_config(+Name, +Options)
    clear_lazy_configs/0,           % clear_lazy_configs

    % Testing
    test_lazy_loading_generator/0
]).

:- use_module(library(lists)).

% ============================================================================
% DYNAMIC PREDICATES
% ============================================================================

:- dynamic lazy_config/2.

:- discontiguous lazy_config/2.

% ============================================================================
% DEFAULT CONFIGURATIONS
% ============================================================================

lazy_config(default, [
    strategy(pagination),
    page_size(50),
    initial_load(true),
    prefetch(1),
    cache(true),
    cache_size(5),
    loading_indicator(true),
    error_retry(3),
    retry_delay(1000)
]).

lazy_config(infinite_scroll, [
    strategy(infinite),
    page_size(20),
    initial_load(true),
    threshold(200),
    loading_indicator(true),
    end_message("No more items"),
    error_retry(3)
]).

lazy_config(windowed, [
    strategy(windowed),
    window_size(100),
    overscan(10),
    initial_load(true)
]).

lazy_config(chunked, [
    strategy(chunked),
    chunk_size(1000),
    parallel_chunks(3),
    initial_load(true)
]).

% ============================================================================
% LAZY HOOK GENERATION
% ============================================================================

%% generate_lazy_hook(+Name, -Hook)
%  Generate a useLazyData hook for the configuration.
generate_lazy_hook(Name, Hook) :-
    (lazy_config(Name, Config) -> true ; lazy_config(default, Config)),
    (member(strategy(Strategy), Config) -> true ; Strategy = pagination),
    (member(page_size(PageSize), Config) -> true ; PageSize = 50),
    (member(prefetch(Prefetch), Config) -> true ; Prefetch = 1),
    (member(cache_size(CacheSize), Config) -> true ; CacheSize = 5),
    (member(error_retry(Retry), Config) -> true ; Retry = 3),
    atom_string(Name, NameStr),
    format(atom(Hook), 'import { useState, useCallback, useRef, useEffect, useMemo } from "react";

interface LazyDataOptions<T> {
  fetchFn: (page: number, pageSize: number) => Promise<{ data: T[]; total: number }>;
  pageSize?: number;
  initialPage?: number;
  prefetch?: number;
  cacheSize?: number;
  onError?: (error: Error) => void;
}

interface LazyDataResult<T> {
  data: T[];
  loading: boolean;
  error: Error | null;
  page: number;
  totalPages: number;
  totalItems: number;
  hasMore: boolean;
  loadPage: (page: number) => Promise<void>;
  loadNext: () => Promise<void>;
  loadPrev: () => Promise<void>;
  refresh: () => Promise<void>;
}

export const useLazyData~w = <T>(options: LazyDataOptions<T>): LazyDataResult<T> => {
  const {
    fetchFn,
    pageSize = ~w,
    initialPage = 1,
    prefetch = ~w,
    cacheSize = ~w,
    onError
  } = options;

  const [data, setData] = useState<T[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const [page, setPage] = useState(initialPage);
  const [totalItems, setTotalItems] = useState(0);

  const cache = useRef<Map<number, T[]>>(new Map());
  const retryCount = useRef(0);
  const maxRetries = ~w;

  const totalPages = useMemo(() => Math.ceil(totalItems / pageSize), [totalItems, pageSize]);
  const hasMore = useMemo(() => page < totalPages, [page, totalPages]);

  const loadPage = useCallback(async (targetPage: number) => {
    if (targetPage < 1 || (totalItems > 0 && targetPage > totalPages)) {
      return;
    }

    // Check cache first
    const cached = cache.current.get(targetPage);
    if (cached) {
      setData(cached);
      setPage(targetPage);
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const result = await fetchFn(targetPage, pageSize);

      // Update cache
      cache.current.set(targetPage, result.data);

      // Trim cache if needed
      if (cache.current.size > cacheSize) {
        const oldestKey = cache.current.keys().next().value;
        cache.current.delete(oldestKey);
      }

      setData(result.data);
      setTotalItems(result.total);
      setPage(targetPage);
      retryCount.current = 0;

      // Prefetch adjacent pages
      for (let i = 1; i <= prefetch; i++) {
        const nextPage = targetPage + i;
        const prevPage = targetPage - i;

        if (nextPage <= Math.ceil(result.total / pageSize) && !cache.current.has(nextPage)) {
          fetchFn(nextPage, pageSize).then(r => cache.current.set(nextPage, r.data)).catch(() => {});
        }
        if (prevPage >= 1 && !cache.current.has(prevPage)) {
          fetchFn(prevPage, pageSize).then(r => cache.current.set(prevPage, r.data)).catch(() => {});
        }
      }
    } catch (err) {
      const error = err instanceof Error ? err : new Error("Failed to load data");

      if (retryCount.current < maxRetries) {
        retryCount.current++;
        setTimeout(() => loadPage(targetPage), 1000 * retryCount.current);
      } else {
        setError(error);
        onError?.(error);
      }
    } finally {
      setLoading(false);
    }
  }, [fetchFn, pageSize, totalPages, totalItems, cacheSize, prefetch, onError]);

  const loadNext = useCallback(async () => {
    if (hasMore) {
      await loadPage(page + 1);
    }
  }, [loadPage, page, hasMore]);

  const loadPrev = useCallback(async () => {
    if (page > 1) {
      await loadPage(page - 1);
    }
  }, [loadPage, page]);

  const refresh = useCallback(async () => {
    cache.current.clear();
    await loadPage(page);
  }, [loadPage, page]);

  // Initial load
  useEffect(() => {
    loadPage(initialPage);
  }, []);

  return {
    data,
    loading,
    error,
    page,
    totalPages,
    totalItems,
    hasMore,
    loadPage,
    loadNext,
    loadPrev,
    refresh
  };
};
', [NameStr, PageSize, Prefetch, CacheSize, Retry]).

% ============================================================================
% PAGINATION HOOK GENERATION
% ============================================================================

%% generate_pagination_hook(+Name, -Hook)
%  Generate a usePagination hook with controls.
generate_pagination_hook(Name, Hook) :-
    (lazy_config(Name, Config) -> true ; lazy_config(default, Config)),
    (member(page_size(PageSize), Config) -> true ; PageSize = 50),
    atom_string(Name, NameStr),
    format(atom(Hook), 'import { useState, useCallback, useMemo } from "react";

interface PaginationState {
  page: number;
  pageSize: number;
  total: number;
}

interface UsePaginationResult {
  page: number;
  pageSize: number;
  totalPages: number;
  total: number;
  offset: number;
  setPage: (page: number) => void;
  setPageSize: (size: number) => void;
  setTotal: (total: number) => void;
  nextPage: () => void;
  prevPage: () => void;
  firstPage: () => void;
  lastPage: () => void;
  canGoNext: boolean;
  canGoPrev: boolean;
  pageNumbers: number[];
}

export const usePagination~w = (
  initialPageSize = ~w,
  initialPage = 1
): UsePaginationResult => {
  const [state, setState] = useState<PaginationState>({
    page: initialPage,
    pageSize: initialPageSize,
    total: 0
  });

  const totalPages = useMemo(
    () => Math.ceil(state.total / state.pageSize) || 1,
    [state.total, state.pageSize]
  );

  const offset = useMemo(
    () => (state.page - 1) * state.pageSize,
    [state.page, state.pageSize]
  );

  const canGoNext = useMemo(() => state.page < totalPages, [state.page, totalPages]);
  const canGoPrev = useMemo(() => state.page > 1, [state.page]);

  const pageNumbers = useMemo(() => {
    const pages: number[] = [];
    const maxVisible = 7;

    if (totalPages <= maxVisible) {
      for (let i = 1; i <= totalPages; i++) pages.push(i);
    } else {
      const start = Math.max(1, state.page - 2);
      const end = Math.min(totalPages, start + maxVisible - 1);

      if (start > 1) pages.push(1);
      if (start > 2) pages.push(-1); // ellipsis marker

      for (let i = Math.max(start, 2); i <= Math.min(end, totalPages - 1); i++) {
        pages.push(i);
      }

      if (end < totalPages - 1) pages.push(-1);
      if (end < totalPages) pages.push(totalPages);
    }

    return pages;
  }, [state.page, totalPages]);

  const setPage = useCallback((page: number) => {
    setState(prev => ({
      ...prev,
      page: Math.max(1, Math.min(page, Math.ceil(prev.total / prev.pageSize) || 1))
    }));
  }, []);

  const setPageSize = useCallback((pageSize: number) => {
    setState(prev => ({
      ...prev,
      pageSize,
      page: 1 // Reset to first page when changing page size
    }));
  }, []);

  const setTotal = useCallback((total: number) => {
    setState(prev => ({
      ...prev,
      total,
      page: Math.min(prev.page, Math.ceil(total / prev.pageSize) || 1)
    }));
  }, []);

  const nextPage = useCallback(() => setPage(state.page + 1), [setPage, state.page]);
  const prevPage = useCallback(() => setPage(state.page - 1), [setPage, state.page]);
  const firstPage = useCallback(() => setPage(1), [setPage]);
  const lastPage = useCallback(() => setPage(totalPages), [setPage, totalPages]);

  return {
    page: state.page,
    pageSize: state.pageSize,
    totalPages,
    total: state.total,
    offset,
    setPage,
    setPageSize,
    setTotal,
    nextPage,
    prevPage,
    firstPage,
    lastPage,
    canGoNext,
    canGoPrev,
    pageNumbers
  };
};

// Pagination Controls Component
interface PaginationControlsProps {
  page: number;
  totalPages: number;
  pageNumbers: number[];
  canGoNext: boolean;
  canGoPrev: boolean;
  onPageChange: (page: number) => void;
  onFirst: () => void;
  onLast: () => void;
  onNext: () => void;
  onPrev: () => void;
  className?: string;
}

export const PaginationControls: React.FC<PaginationControlsProps> = ({
  page,
  totalPages,
  pageNumbers,
  canGoNext,
  canGoPrev,
  onPageChange,
  onFirst,
  onLast,
  onNext,
  onPrev,
  className = ""
}) => {
  return (
    <nav className={`pagination ${className}`} aria-label="Pagination">
      <button
        onClick={onFirst}
        disabled={!canGoPrev}
        className="pagination__btn pagination__btn--first"
        aria-label="First page"
      >
        &laquo;
      </button>
      <button
        onClick={onPrev}
        disabled={!canGoPrev}
        className="pagination__btn pagination__btn--prev"
        aria-label="Previous page"
      >
        &lsaquo;
      </button>

      {pageNumbers.map((num, idx) =>
        num === -1 ? (
          <span key={`ellipsis-${idx}`} className="pagination__ellipsis">&hellip;</span>
        ) : (
          <button
            key={num}
            onClick={() => onPageChange(num)}
            className={`pagination__btn pagination__btn--page ${page === num ? "active" : ""}`}
            aria-current={page === num ? "page" : undefined}
          >
            {num}
          </button>
        )
      )}

      <button
        onClick={onNext}
        disabled={!canGoNext}
        className="pagination__btn pagination__btn--next"
        aria-label="Next page"
      >
        &rsaquo;
      </button>
      <button
        onClick={onLast}
        disabled={!canGoNext}
        className="pagination__btn pagination__btn--last"
        aria-label="Last page"
      >
        &raquo;
      </button>
    </nav>
  );
};
', [NameStr, PageSize]).

% ============================================================================
% INFINITE SCROLL GENERATION
% ============================================================================

%% generate_infinite_scroll(+Name, -Hook)
%  Generate an infinite scroll hook.
generate_infinite_scroll(Name, Hook) :-
    (lazy_config(Name, Config) -> true ; lazy_config(infinite_scroll, Config)),
    (member(page_size(PageSize), Config) -> true ; PageSize = 20),
    (member(threshold(Threshold), Config) -> true ; Threshold = 200),
    atom_string(Name, NameStr),
    format(atom(Hook), 'import { useState, useCallback, useRef, useEffect } from "react";

interface InfiniteScrollOptions<T> {
  fetchFn: (page: number, pageSize: number) => Promise<{ data: T[]; hasMore: boolean }>;
  pageSize?: number;
  threshold?: number;
  initialData?: T[];
}

interface InfiniteScrollResult<T> {
  items: T[];
  loading: boolean;
  error: Error | null;
  hasMore: boolean;
  loadMore: () => Promise<void>;
  reset: () => void;
  containerRef: React.RefObject<HTMLDivElement>;
  sentinelRef: React.RefObject<HTMLDivElement>;
}

export const useInfiniteScroll~w = <T>(
  options: InfiniteScrollOptions<T>
): InfiniteScrollResult<T> => {
  const {
    fetchFn,
    pageSize = ~w,
    threshold = ~w,
    initialData = []
  } = options;

  const [items, setItems] = useState<T[]>(initialData);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const [hasMore, setHasMore] = useState(true);
  const [page, setPage] = useState(1);

  const containerRef = useRef<HTMLDivElement>(null);
  const sentinelRef = useRef<HTMLDivElement>(null);
  const loadingRef = useRef(false);

  const loadMore = useCallback(async () => {
    if (loadingRef.current || !hasMore) return;

    loadingRef.current = true;
    setLoading(true);
    setError(null);

    try {
      const result = await fetchFn(page, pageSize);
      setItems(prev => [...prev, ...result.data]);
      setHasMore(result.hasMore);
      setPage(prev => prev + 1);
    } catch (err) {
      setError(err instanceof Error ? err : new Error("Failed to load more items"));
    } finally {
      setLoading(false);
      loadingRef.current = false;
    }
  }, [fetchFn, page, pageSize, hasMore]);

  const reset = useCallback(() => {
    setItems(initialData);
    setPage(1);
    setHasMore(true);
    setError(null);
  }, [initialData]);

  // Set up Intersection Observer
  useEffect(() => {
    const sentinel = sentinelRef.current;
    if (!sentinel) return;

    const observer = new IntersectionObserver(
      (entries) => {
        const [entry] = entries;
        if (entry.isIntersecting && hasMore && !loadingRef.current) {
          loadMore();
        }
      },
      {
        root: containerRef.current,
        rootMargin: `${threshold}px`,
        threshold: 0
      }
    );

    observer.observe(sentinel);

    return () => {
      observer.unobserve(sentinel);
    };
  }, [loadMore, hasMore, threshold]);

  // Initial load
  useEffect(() => {
    if (items.length === 0 && hasMore) {
      loadMore();
    }
  }, []);

  return {
    items,
    loading,
    error,
    hasMore,
    loadMore,
    reset,
    containerRef,
    sentinelRef
  };
};

// Infinite Scroll Container Component
interface InfiniteScrollContainerProps {
  items: unknown[];
  loading: boolean;
  hasMore: boolean;
  containerRef: React.RefObject<HTMLDivElement>;
  sentinelRef: React.RefObject<HTMLDivElement>;
  renderItem: (item: unknown, index: number) => React.ReactNode;
  loadingComponent?: React.ReactNode;
  endMessage?: React.ReactNode;
  className?: string;
}

export const InfiniteScrollContainer: React.FC<InfiniteScrollContainerProps> = ({
  items,
  loading,
  hasMore,
  containerRef,
  sentinelRef,
  renderItem,
  loadingComponent,
  endMessage,
  className = ""
}) => {
  return (
    <div ref={containerRef} className={`infinite-scroll ${className}`}>
      {items.map((item, index) => renderItem(item, index))}

      <div ref={sentinelRef} className="infinite-scroll__sentinel" />

      {loading && (
        <div className="infinite-scroll__loading">
          {loadingComponent || (
            <div className="loading-spinner">Loading...</div>
          )}
        </div>
      )}

      {!hasMore && !loading && endMessage && (
        <div className="infinite-scroll__end">
          {endMessage}
        </div>
      )}
    </div>
  );
};
', [NameStr, PageSize, Threshold]).

% ============================================================================
% LAZY LOADER GENERATION
% ============================================================================

%% generate_lazy_loader(+Name, -Loader)
%  Generate a lazy loader utility class.
generate_lazy_loader(Name, Loader) :-
    (lazy_config(Name, Config) -> true ; lazy_config(default, Config)),
    (member(chunk_size(ChunkSize), Config) -> true ; ChunkSize = 1000),
    (member(parallel_chunks(Parallel), Config) -> true ; Parallel = 3),
    atom_string(Name, NameStr),
    format(atom(Loader), 'interface LazyLoaderOptions<T> {
  fetchChunk: (offset: number, limit: number) => Promise<T[]>;
  totalCount: () => Promise<number>;
  chunkSize?: number;
  parallelChunks?: number;
  onProgress?: (loaded: number, total: number) => void;
}

export class LazyLoader~w<T> {
  private cache: Map<number, T[]> = new Map();
  private loading: Set<number> = new Set();
  private total: number = 0;
  private options: Required<LazyLoaderOptions<T>>;

  constructor(options: LazyLoaderOptions<T>) {
    this.options = {
      chunkSize: ~w,
      parallelChunks: ~w,
      onProgress: () => {},
      ...options
    };
  }

  async initialize(): Promise<number> {
    this.total = await this.options.totalCount();
    return this.total;
  }

  async loadRange(start: number, end: number): Promise<T[]> {
    const { chunkSize } = this.options;

    const startChunk = Math.floor(start / chunkSize);
    const endChunk = Math.floor(end / chunkSize);

    const chunksToLoad: number[] = [];
    for (let i = startChunk; i <= endChunk; i++) {
      if (!this.cache.has(i) && !this.loading.has(i)) {
        chunksToLoad.push(i);
      }
    }

    if (chunksToLoad.length > 0) {
      await this.loadChunks(chunksToLoad);
    }

    // Assemble result
    const result: T[] = [];
    for (let i = start; i < end && i < this.total; i++) {
      const chunkIndex = Math.floor(i / chunkSize);
      const chunkOffset = i % chunkSize;
      const chunk = this.cache.get(chunkIndex);
      if (chunk && chunk[chunkOffset] !== undefined) {
        result.push(chunk[chunkOffset]);
      }
    }

    return result;
  }

  private async loadChunks(chunkIndices: number[]): Promise<void> {
    const { chunkSize, parallelChunks, fetchChunk, onProgress } = this.options;

    // Mark as loading
    chunkIndices.forEach(i => this.loading.add(i));

    // Load in parallel batches
    for (let i = 0; i < chunkIndices.length; i += parallelChunks) {
      const batch = chunkIndices.slice(i, i + parallelChunks);

      await Promise.all(
        batch.map(async (chunkIndex) => {
          const offset = chunkIndex * chunkSize;
          const data = await fetchChunk(offset, chunkSize);
          this.cache.set(chunkIndex, data);
          this.loading.delete(chunkIndex);

          const loadedCount = this.cache.size * chunkSize;
          onProgress(Math.min(loadedCount, this.total), this.total);
        })
      );
    }
  }

  getItem(index: number): T | undefined {
    const { chunkSize } = this.options;
    const chunkIndex = Math.floor(index / chunkSize);
    const chunkOffset = index % chunkSize;
    const chunk = this.cache.get(chunkIndex);
    return chunk?.[chunkOffset];
  }

  isLoaded(index: number): boolean {
    const { chunkSize } = this.options;
    const chunkIndex = Math.floor(index / chunkSize);
    return this.cache.has(chunkIndex);
  }

  isLoading(index: number): boolean {
    const { chunkSize } = this.options;
    const chunkIndex = Math.floor(index / chunkSize);
    return this.loading.has(chunkIndex);
  }

  getTotal(): number {
    return this.total;
  }

  clear(): void {
    this.cache.clear();
    this.loading.clear();
    this.total = 0;
  }
}
', [NameStr, ChunkSize, Parallel]).

% ============================================================================
% LAZY COMPONENT GENERATION
% ============================================================================

%% generate_lazy_component(+Name, -Component)
%  Generate a lazy loading wrapper component.
generate_lazy_component(Name, Component) :-
    atom_string(Name, NameStr),
    format(atom(Component), 'import React, { Suspense, lazy, ComponentType, useState, useEffect } from "react";

interface LazyComponentProps<P> {
  loader: () => Promise<{ default: ComponentType<P> }>;
  fallback?: React.ReactNode;
  errorFallback?: React.ReactNode | ((error: Error) => React.ReactNode);
  onLoad?: () => void;
  onError?: (error: Error) => void;
}

interface ErrorBoundaryState {
  hasError: boolean;
  error: Error | null;
}

class LazyErrorBoundary extends React.Component<
  { children: React.ReactNode; fallback?: React.ReactNode | ((error: Error) => React.ReactNode) },
  ErrorBoundaryState
> {
  constructor(props: any) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { hasError: true, error };
  }

  render() {
    if (this.state.hasError && this.state.error) {
      const { fallback } = this.props;
      if (typeof fallback === "function") {
        return fallback(this.state.error);
      }
      return fallback || <div className="lazy-error">Failed to load component</div>;
    }

    return this.props.children;
  }
}

export function createLazyComponent~w<P extends object>(
  options: LazyComponentProps<P>
): React.FC<P> {
  const { loader, fallback, errorFallback, onLoad, onError } = options;

  const LazyComponent = lazy(async () => {
    try {
      const module = await loader();
      onLoad?.();
      return module;
    } catch (error) {
      const err = error instanceof Error ? error : new Error("Failed to load");
      onError?.(err);
      throw err;
    }
  });

  return (props: P) => (
    <LazyErrorBoundary fallback={errorFallback}>
      <Suspense fallback={fallback || <div className="lazy-loading">Loading...</div>}>
        <LazyComponent {...props} />
      </Suspense>
    </LazyErrorBoundary>
  );
}

// Preload helper
export function preloadComponent~w<P>(
  loader: () => Promise<{ default: ComponentType<P> }>
): void {
  loader();
}

// Loading placeholder component
export const LoadingPlaceholder: React.FC<{
  width?: string | number;
  height?: string | number;
  className?: string;
}> = ({ width = "100%", height = 200, className = "" }) => (
  <div
    className={`loading-placeholder ${className}`}
    style={{
      width,
      height,
      background: "linear-gradient(90deg, #f0f0f0 25%, #e0e0e0 50%, #f0f0f0 75%)",
      backgroundSize: "200% 100%",
      animation: "shimmer 1.5s infinite"
    }}
  />
);

// CSS for shimmer animation
export const LazyLoadingCSS = `
@keyframes shimmer {
  0% { background-position: -200% 0; }
  100% { background-position: 200% 0; }
}

.lazy-loading {
  display: flex;
  align-items: center;
  justify-content: center;
  min-height: 100px;
  color: var(--color-text-secondary, #666);
}

.lazy-error {
  display: flex;
  align-items: center;
  justify-content: center;
  min-height: 100px;
  color: var(--color-error, #dc2626);
  background: var(--color-error-bg, #fef2f2);
  border-radius: var(--border-radius-md, 8px);
  padding: var(--spacing-md, 16px);
}

.loading-placeholder {
  border-radius: var(--border-radius-md, 8px);
}
`;
', [NameStr, NameStr]).

% ============================================================================
% UTILITY PREDICATES
% ============================================================================

%% lazy_strategy(+Name, -Strategy)
%  Get the lazy loading strategy for a configuration.
lazy_strategy(Name, Strategy) :-
    lazy_config(Name, Config),
    member(strategy(Strategy), Config), !.
lazy_strategy(_, pagination).

% ============================================================================
% MANAGEMENT PREDICATES
% ============================================================================

%% declare_lazy_config(+Name, +Options)
%  Declare a lazy loading configuration.
declare_lazy_config(Name, Options) :-
    retractall(lazy_config(Name, _)),
    assertz(lazy_config(Name, Options)).

%% clear_lazy_configs/0
%  Clear all lazy loading configurations.
clear_lazy_configs :-
    retractall(lazy_config(_, _)).

% ============================================================================
% TESTING
% ============================================================================

test_lazy_loading_generator :-
    writeln('Testing lazy loading generator...'),

    % Test config existence
    (lazy_config(default, _) ->
        writeln('  [PASS] default config exists') ;
        writeln('  [FAIL] default config')),

    (lazy_config(infinite_scroll, _) ->
        writeln('  [PASS] infinite_scroll config exists') ;
        writeln('  [FAIL] infinite_scroll config')),

    % Test lazy hook generation
    (generate_lazy_hook(default, Hook), atom_length(Hook, HL), HL > 1000 ->
        writeln('  [PASS] generate_lazy_hook produces code') ;
        writeln('  [FAIL] generate_lazy_hook')),

    % Test pagination hook
    (generate_pagination_hook(default, PagHook), atom_length(PagHook, PL), PL > 1000 ->
        writeln('  [PASS] generate_pagination_hook produces code') ;
        writeln('  [FAIL] generate_pagination_hook')),

    % Test infinite scroll
    (generate_infinite_scroll(infinite_scroll, InfHook), atom_length(InfHook, IL), IL > 1000 ->
        writeln('  [PASS] generate_infinite_scroll produces code') ;
        writeln('  [FAIL] generate_infinite_scroll')),

    % Test lazy loader
    (generate_lazy_loader(chunked, Loader), atom_length(Loader, LL), LL > 500 ->
        writeln('  [PASS] generate_lazy_loader produces code') ;
        writeln('  [FAIL] generate_lazy_loader')),

    writeln('Lazy loading generator tests complete.').
