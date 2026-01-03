import React, { useState, useRef, useCallback, useEffect, useMemo } from "react";

export interface VirtualListProps {
  items: Array<{ id: string; content: string }>;
  itemHeight: number;
  containerHeight: number;
  overscan?: number;
  onVisibleRangeChange?: (start: number, end: number) => void;
}

/**
 * VirtualList - Generated from performance_generator.pl
 *
 * Demonstrates virtualized rendering for efficient display of large lists.
 * Only renders items currently visible in the viewport plus overscan buffer.
 */
export const VirtualList: React.FC<VirtualListProps> = ({
  items,
  itemHeight,
  containerHeight,
  overscan = 3,
  onVisibleRangeChange,
}) => {
  const [scrollTop, setScrollTop] = useState(0);
  const containerRef = useRef<HTMLDivElement>(null);

  const totalHeight = items.length * itemHeight;

  const startIndex = Math.max(0, Math.floor(scrollTop / itemHeight) - overscan);
  const endIndex = Math.min(
    items.length,
    Math.ceil((scrollTop + containerHeight) / itemHeight) + overscan
  );

  const visibleItems = items.slice(startIndex, endIndex);
  const offsetY = startIndex * itemHeight;

  useEffect(() => {
    onVisibleRangeChange?.(startIndex, endIndex);
  }, [startIndex, endIndex, onVisibleRangeChange]);

  const handleScroll = useCallback((e: React.UIEvent<HTMLDivElement>) => {
    setScrollTop(e.currentTarget.scrollTop);
  }, []);

  return (
    <div
      ref={containerRef}
      onScroll={handleScroll}
      style={{
        height: containerHeight,
        overflow: "auto",
        border: "1px solid var(--color-border)",
        borderRadius: "var(--border-radius-md)",
        background: "var(--color-surface)",
      }}
    >
      <div style={{ height: totalHeight, position: "relative" }}>
        <div
          style={{
            position: "absolute",
            top: offsetY,
            left: 0,
            right: 0,
          }}
        >
          {visibleItems.map((item, index) => (
            <div
              key={item.id}
              style={{
                height: itemHeight,
                padding: "0 1rem",
                display: "flex",
                alignItems: "center",
                borderBottom: "1px solid var(--color-border)",
                background:
                  (startIndex + index) % 2 === 0
                    ? "var(--color-surface)"
                    : "var(--color-background)",
              }}
            >
              <span
                style={{
                  width: "60px",
                  color: "var(--color-text-secondary)",
                  fontSize: "0.75rem",
                }}
              >
                #{startIndex + index + 1}
              </span>
              <span style={{ color: "var(--color-text-primary)" }}>
                {item.content}
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export interface LazyLoadGridProps {
  totalItems: number;
  pageSize: number;
  renderItem: (index: number) => React.ReactNode;
  columns?: number;
  gap?: string;
}

/**
 * LazyLoadGrid - Demonstrates lazy loading with intersection observer
 */
export const LazyLoadGrid: React.FC<LazyLoadGridProps> = ({
  totalItems,
  pageSize,
  renderItem,
  columns = 3,
  gap = "1rem",
}) => {
  const [loadedCount, setLoadedCount] = useState(pageSize);
  const [isLoading, setIsLoading] = useState(false);
  const loaderRef = useRef<HTMLDivElement>(null);

  const loadMore = useCallback(() => {
    if (loadedCount >= totalItems || isLoading) return;

    setIsLoading(true);
    // Simulate network delay
    setTimeout(() => {
      setLoadedCount((prev) => Math.min(prev + pageSize, totalItems));
      setIsLoading(false);
    }, 500);
  }, [loadedCount, totalItems, pageSize, isLoading]);

  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        if (entries[0].isIntersecting) {
          loadMore();
        }
      },
      { threshold: 0.1 }
    );

    if (loaderRef.current) {
      observer.observe(loaderRef.current);
    }

    return () => observer.disconnect();
  }, [loadMore]);

  return (
    <div>
      <div
        style={{
          display: "grid",
          gridTemplateColumns: `repeat(${columns}, 1fr)`,
          gap,
        }}
      >
        {Array.from({ length: loadedCount }, (_, i) => (
          <div key={i}>{renderItem(i)}</div>
        ))}
      </div>

      {loadedCount < totalItems && (
        <div
          ref={loaderRef}
          style={{
            padding: "2rem",
            textAlign: "center",
          }}
        >
          {isLoading ? (
            <div
              style={{
                display: "inline-flex",
                alignItems: "center",
                gap: "0.5rem",
                color: "var(--color-text-secondary)",
              }}
            >
              <div
                className="spinner"
                style={{
                  width: "20px",
                  height: "20px",
                  border: "2px solid var(--color-border)",
                  borderTopColor: "var(--color-primary)",
                  borderRadius: "50%",
                  animation: "spin 1s linear infinite",
                }}
              />
              <style>{`
                @keyframes spin {
                  to { transform: rotate(360deg); }
                }
              `}</style>
              Loading...
            </div>
          ) : (
            <span style={{ color: "var(--color-text-secondary)", fontSize: "0.875rem" }}>
              Scroll to load more ({loadedCount} / {totalItems})
            </span>
          )}
        </div>
      )}

      {loadedCount >= totalItems && (
        <div
          style={{
            padding: "1rem",
            textAlign: "center",
            color: "var(--color-text-secondary)",
            fontSize: "0.875rem",
          }}
        >
          All {totalItems} items loaded
        </div>
      )}
    </div>
  );
};

export interface ChunkedDataLoaderProps {
  totalChunks: number;
  chunkSize: number;
  renderChunk: (chunkIndex: number, data: number[]) => React.ReactNode;
}

/**
 * ChunkedDataLoader - Demonstrates chunked/batched data processing
 */
export const ChunkedDataLoader: React.FC<ChunkedDataLoaderProps> = ({
  totalChunks,
  chunkSize,
  renderChunk,
}) => {
  const [loadedChunks, setLoadedChunks] = useState<Map<number, number[]>>(
    new Map()
  );
  const [loadingChunk, setLoadingChunk] = useState<number | null>(null);

  const loadChunk = useCallback(
    (chunkIndex: number) => {
      if (loadedChunks.has(chunkIndex) || loadingChunk !== null) return;

      setLoadingChunk(chunkIndex);

      // Simulate data fetching
      setTimeout(() => {
        const data = Array.from(
          { length: chunkSize },
          (_, i) => chunkIndex * chunkSize + i + 1
        );
        setLoadedChunks((prev) => new Map(prev).set(chunkIndex, data));
        setLoadingChunk(null);
      }, 300);
    },
    [loadedChunks, loadingChunk, chunkSize]
  );

  const progress = (loadedChunks.size / totalChunks) * 100;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>
      {/* Progress bar */}
      <div>
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            marginBottom: "0.25rem",
            fontSize: "0.75rem",
            color: "var(--color-text-secondary)",
          }}
        >
          <span>Data chunks loaded</span>
          <span>
            {loadedChunks.size} / {totalChunks} ({Math.round(progress)}%)
          </span>
        </div>
        <div
          style={{
            height: "8px",
            background: "var(--color-border)",
            borderRadius: "var(--border-radius-full)",
            overflow: "hidden",
          }}
        >
          <div
            style={{
              width: `${progress}%`,
              height: "100%",
              background: "var(--color-primary)",
              transition: "width 0.3s ease-out",
            }}
          />
        </div>
      </div>

      {/* Chunk buttons */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fill, minmax(80px, 1fr))",
          gap: "0.5rem",
        }}
      >
        {Array.from({ length: totalChunks }, (_, i) => {
          const isLoaded = loadedChunks.has(i);
          const isLoading = loadingChunk === i;

          return (
            <button
              key={i}
              onClick={() => loadChunk(i)}
              disabled={isLoaded || isLoading}
              style={{
                padding: "0.5rem",
                fontSize: "0.75rem",
                border: "1px solid var(--color-border)",
                borderRadius: "var(--border-radius-sm)",
                background: isLoaded
                  ? "var(--color-primary)"
                  : "var(--color-surface)",
                color: isLoaded ? "white" : "var(--color-text-primary)",
                cursor: isLoaded || isLoading ? "default" : "pointer",
                opacity: isLoading ? 0.5 : 1,
                transition: "all var(--transition-fast)",
              }}
            >
              {isLoading ? "..." : isLoaded ? "✓" : `#${i + 1}`}
            </button>
          );
        })}
      </div>

      {/* Loaded data */}
      <div
        style={{
          display: "flex",
          flexDirection: "column",
          gap: "0.5rem",
          maxHeight: "200px",
          overflow: "auto",
        }}
      >
        {Array.from(loadedChunks.entries())
          .sort(([a], [b]) => a - b)
          .map(([index, data]) => (
            <div key={index}>{renderChunk(index, data)}</div>
          ))}
      </div>

      {loadedChunks.size === 0 && (
        <div
          style={{
            padding: "2rem",
            textAlign: "center",
            color: "var(--color-text-secondary)",
            border: "1px dashed var(--color-border)",
            borderRadius: "var(--border-radius-md)",
          }}
        >
          Click a chunk button to load data
        </div>
      )}
    </div>
  );
};

export interface PerformanceMetricsProps {
  itemCount: number;
  renderTime: number;
  memoryUsage?: number;
}

/**
 * PerformanceMetrics - Displays performance metrics for demonstrations
 */
export const PerformanceMetrics: React.FC<PerformanceMetricsProps> = ({
  itemCount,
  renderTime,
  memoryUsage,
}) => {
  return (
    <div
      style={{
        display: "flex",
        gap: "1.5rem",
        padding: "0.75rem 1rem",
        background: "var(--color-surface)",
        borderRadius: "var(--border-radius-md)",
        border: "1px solid var(--color-border)",
        fontSize: "0.75rem",
      }}
    >
      <div>
        <span style={{ color: "var(--color-text-secondary)" }}>Items: </span>
        <span style={{ color: "var(--color-text-primary)", fontWeight: 600 }}>
          {itemCount.toLocaleString()}
        </span>
      </div>
      <div>
        <span style={{ color: "var(--color-text-secondary)" }}>Render: </span>
        <span
          style={{
            color:
              renderTime < 16
                ? "#22c55e"
                : renderTime < 50
                ? "#f59e0b"
                : "#ef4444",
            fontWeight: 600,
          }}
        >
          {renderTime.toFixed(1)}ms
        </span>
      </div>
      {memoryUsage !== undefined && (
        <div>
          <span style={{ color: "var(--color-text-secondary)" }}>Memory: </span>
          <span style={{ color: "var(--color-text-primary)", fontWeight: 600 }}>
            {(memoryUsage / 1024 / 1024).toFixed(1)}MB
          </span>
        </div>
      )}
    </div>
  );
};

export interface MemoizedListProps {
  items: Array<{ id: string; value: number }>;
  highlightThreshold?: number;
}

/**
 * MemoizedList - Demonstrates React.memo optimization patterns
 */
const MemoizedListItem = React.memo<{
  id: string;
  value: number;
  isHighlighted: boolean;
}>(({ id, value, isHighlighted }) => {
  // Track renders
  const renderCountRef = useRef(0);
  renderCountRef.current++;

  return (
    <div
      style={{
        display: "flex",
        justifyContent: "space-between",
        alignItems: "center",
        padding: "0.5rem 1rem",
        background: isHighlighted
          ? "var(--color-primary)"
          : "var(--color-surface)",
        color: isHighlighted ? "white" : "var(--color-text-primary)",
        borderRadius: "var(--border-radius-sm)",
        border: "1px solid var(--color-border)",
        marginBottom: "0.25rem",
      }}
    >
      <span>
        {id}: {value}
      </span>
      <span
        style={{
          fontSize: "0.625rem",
          opacity: 0.7,
        }}
      >
        renders: {renderCountRef.current}
      </span>
    </div>
  );
});

MemoizedListItem.displayName = "MemoizedListItem";

export const MemoizedList: React.FC<MemoizedListProps> = ({
  items,
  highlightThreshold = 50,
}) => {
  const [filter, setFilter] = useState("");

  const filteredItems = useMemo(
    () =>
      items.filter(
        (item) =>
          filter === "" || item.id.toLowerCase().includes(filter.toLowerCase())
      ),
    [items, filter]
  );

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "0.5rem" }}>
      <input
        type="text"
        placeholder="Filter items..."
        value={filter}
        onChange={(e) => setFilter(e.target.value)}
        style={{
          padding: "0.5rem 1rem",
          border: "1px solid var(--color-border)",
          borderRadius: "var(--border-radius-sm)",
          background: "var(--color-surface)",
          color: "var(--color-text-primary)",
          fontSize: "0.875rem",
        }}
      />
      <div
        style={{
          fontSize: "0.75rem",
          color: "var(--color-text-secondary)",
        }}
      >
        Showing {filteredItems.length} of {items.length} items | Items with
        value &gt; {highlightThreshold} are highlighted
      </div>
      <div
        style={{
          maxHeight: "300px",
          overflow: "auto",
        }}
      >
        {filteredItems.map((item) => (
          <MemoizedListItem
            key={item.id}
            id={item.id}
            value={item.value}
            isHighlighted={item.value > highlightThreshold}
          />
        ))}
      </div>
    </div>
  );
};
