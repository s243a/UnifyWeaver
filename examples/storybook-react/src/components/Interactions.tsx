import React, { useState, useRef, useCallback, useEffect } from "react";

export interface TooltipProps {
  content: React.ReactNode;
  children: React.ReactElement;
  position?: "top" | "bottom" | "left" | "right";
  delay?: number;
}

/**
 * Tooltip - Generated from interaction_generator.pl
 *
 * Demonstrates hover-triggered contextual information display.
 */
export const Tooltip: React.FC<TooltipProps> = ({
  content,
  children,
  position = "top",
  delay = 100,
}) => {
  const [isVisible, setIsVisible] = useState(false);
  const timeoutRef = useRef<ReturnType<typeof setTimeout>>();

  const handleMouseEnter = () => {
    timeoutRef.current = setTimeout(() => setIsVisible(true), delay);
  };

  const handleMouseLeave = () => {
    if (timeoutRef.current) clearTimeout(timeoutRef.current);
    setIsVisible(false);
  };

  const positionStyles: Record<string, React.CSSProperties> = {
    top: { bottom: "100%", left: "50%", transform: "translateX(-50%)", marginBottom: "8px" },
    bottom: { top: "100%", left: "50%", transform: "translateX(-50%)", marginTop: "8px" },
    left: { right: "100%", top: "50%", transform: "translateY(-50%)", marginRight: "8px" },
    right: { left: "100%", top: "50%", transform: "translateY(-50%)", marginLeft: "8px" },
  };

  return (
    <div
      style={{ position: "relative", display: "inline-block" }}
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
    >
      {children}
      {isVisible && (
        <div
          style={{
            position: "absolute",
            ...positionStyles[position],
            padding: "0.5rem 0.75rem",
            background: "var(--color-text-primary)",
            color: "var(--color-background)",
            borderRadius: "var(--border-radius-sm)",
            fontSize: "0.75rem",
            whiteSpace: "nowrap",
            zIndex: 1000,
            animation: "tooltipFadeIn 0.2s ease-out",
          }}
        >
          <style>{`
            @keyframes tooltipFadeIn {
              from { opacity: 0; transform: translateX(-50%) scale(0.95); }
              to { opacity: 1; transform: translateX(-50%) scale(1); }
            }
          `}</style>
          {content}
        </div>
      )}
    </div>
  );
};

export interface PanZoomCanvasProps {
  children?: React.ReactNode;
  minZoom?: number;
  maxZoom?: number;
  onTransformChange?: (transform: { x: number; y: number; scale: number }) => void;
}

/**
 * PanZoomCanvas - Demonstrates pan and zoom interactions
 */
export const PanZoomCanvas: React.FC<PanZoomCanvasProps> = ({
  children,
  minZoom = 0.5,
  maxZoom = 3,
  onTransformChange,
}) => {
  const [transform, setTransform] = useState({ x: 0, y: 0, scale: 1 });
  const [isPanning, setIsPanning] = useState(false);
  const lastPositionRef = useRef({ x: 0, y: 0 });
  const containerRef = useRef<HTMLDivElement>(null);

  const handleMouseDown = (e: React.MouseEvent) => {
    if (e.button === 0) {
      setIsPanning(true);
      lastPositionRef.current = { x: e.clientX, y: e.clientY };
    }
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!isPanning) return;
    const dx = e.clientX - lastPositionRef.current.x;
    const dy = e.clientY - lastPositionRef.current.y;
    lastPositionRef.current = { x: e.clientX, y: e.clientY };
    setTransform((prev) => {
      const newTransform = { ...prev, x: prev.x + dx, y: prev.y + dy };
      onTransformChange?.(newTransform);
      return newTransform;
    });
  };

  const handleMouseUp = () => setIsPanning(false);

  const handleWheel = useCallback(
    (e: WheelEvent) => {
      e.preventDefault();
      const delta = e.deltaY > 0 ? 0.9 : 1.1;
      setTransform((prev) => {
        const newScale = Math.min(maxZoom, Math.max(minZoom, prev.scale * delta));
        const newTransform = { ...prev, scale: newScale };
        onTransformChange?.(newTransform);
        return newTransform;
      });
    },
    [minZoom, maxZoom, onTransformChange]
  );

  useEffect(() => {
    const container = containerRef.current;
    if (container) {
      container.addEventListener("wheel", handleWheel, { passive: false });
      return () => container.removeEventListener("wheel", handleWheel);
    }
  }, [handleWheel]);

  const resetTransform = () => {
    const newTransform = { x: 0, y: 0, scale: 1 };
    setTransform(newTransform);
    onTransformChange?.(newTransform);
  };

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        gap: "0.5rem",
      }}
    >
      <div
        ref={containerRef}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
        style={{
          width: "100%",
          height: "300px",
          overflow: "hidden",
          background: "var(--color-surface)",
          border: "1px solid var(--color-border)",
          borderRadius: "var(--border-radius-md)",
          cursor: isPanning ? "grabbing" : "grab",
          position: "relative",
        }}
      >
        <div
          style={{
            transform: `translate(${transform.x}px, ${transform.y}px) scale(${transform.scale})`,
            transformOrigin: "center center",
            transition: isPanning ? "none" : "transform 0.1s ease-out",
            width: "100%",
            height: "100%",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
          }}
        >
          {children || (
            <div
              style={{
                width: "200px",
                height: "200px",
                background: "linear-gradient(135deg, var(--color-primary), var(--color-secondary, #7c3aed))",
                borderRadius: "var(--border-radius-lg)",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                color: "white",
                fontWeight: 600,
                fontSize: "0.875rem",
              }}
            >
              Pan & Zoom Me
            </div>
          )}
        </div>
      </div>
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          fontSize: "0.75rem",
          color: "var(--color-text-secondary)",
        }}
      >
        <span>
          Position: ({Math.round(transform.x)}, {Math.round(transform.y)}) | Zoom:{" "}
          {Math.round(transform.scale * 100)}%
        </span>
        <button
          onClick={resetTransform}
          style={{
            padding: "0.25rem 0.5rem",
            fontSize: "0.75rem",
            border: "1px solid var(--color-border)",
            borderRadius: "var(--border-radius-sm)",
            background: "var(--color-surface)",
            cursor: "pointer",
          }}
        >
          Reset
        </button>
      </div>
    </div>
  );
};

export interface DrillDownItem {
  id: string;
  label: string;
  value: number;
  children?: DrillDownItem[];
}

export interface DrillDownChartProps {
  data: DrillDownItem[];
  onDrillDown?: (path: DrillDownItem[]) => void;
}

/**
 * DrillDownChart - Demonstrates hierarchical drill-down interaction
 */
export const DrillDownChart: React.FC<DrillDownChartProps> = ({
  data,
  onDrillDown,
}) => {
  const [path, setPath] = useState<DrillDownItem[]>([]);
  const currentItems = path.length === 0 ? data : path[path.length - 1].children || [];

  const handleClick = (item: DrillDownItem) => {
    if (item.children && item.children.length > 0) {
      const newPath = [...path, item];
      setPath(newPath);
      onDrillDown?.(newPath);
    }
  };

  const navigateToLevel = (index: number) => {
    const newPath = path.slice(0, index);
    setPath(newPath);
    onDrillDown?.(newPath);
  };

  const maxValue = Math.max(...currentItems.map((i) => i.value));

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>
      {/* Breadcrumb */}
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: "0.5rem",
          fontSize: "0.875rem",
        }}
      >
        <button
          onClick={() => navigateToLevel(0)}
          style={{
            background: "none",
            border: "none",
            color: "var(--color-primary)",
            cursor: "pointer",
            padding: "0.25rem",
            fontWeight: path.length === 0 ? 600 : 400,
          }}
        >
          Root
        </button>
        {path.map((item, index) => (
          <React.Fragment key={item.id}>
            <span style={{ color: "var(--color-text-secondary)" }}>/</span>
            <button
              onClick={() => navigateToLevel(index + 1)}
              style={{
                background: "none",
                border: "none",
                color: "var(--color-primary)",
                cursor: "pointer",
                padding: "0.25rem",
                fontWeight: index === path.length - 1 ? 600 : 400,
              }}
            >
              {item.label}
            </button>
          </React.Fragment>
        ))}
      </div>

      {/* Bar Chart */}
      <div style={{ display: "flex", flexDirection: "column", gap: "0.5rem" }}>
        {currentItems.map((item) => (
          <div
            key={item.id}
            onClick={() => handleClick(item)}
            style={{
              display: "flex",
              alignItems: "center",
              gap: "1rem",
              padding: "0.75rem",
              background: "var(--color-surface)",
              borderRadius: "var(--border-radius-md)",
              cursor: item.children ? "pointer" : "default",
              border: "1px solid var(--color-border)",
              transition: "all var(--transition-fast)",
            }}
            onMouseOver={(e) => {
              if (item.children) {
                e.currentTarget.style.borderColor = "var(--color-primary)";
              }
            }}
            onMouseOut={(e) => {
              e.currentTarget.style.borderColor = "var(--color-border)";
            }}
          >
            <div
              style={{
                width: "120px",
                display: "flex",
                alignItems: "center",
                gap: "0.5rem",
              }}
            >
              <span style={{ color: "var(--color-text-primary)", fontWeight: 500 }}>
                {item.label}
              </span>
              {item.children && (
                <span style={{ fontSize: "0.75rem", color: "var(--color-text-secondary)" }}>
                  →
                </span>
              )}
            </div>
            <div
              style={{
                flex: 1,
                height: "24px",
                background: "var(--color-border)",
                borderRadius: "var(--border-radius-sm)",
                overflow: "hidden",
              }}
            >
              <div
                style={{
                  width: `${(item.value / maxValue) * 100}%`,
                  height: "100%",
                  background: "var(--color-primary)",
                  transition: "width 0.3s ease-out",
                }}
              />
            </div>
            <div
              style={{
                width: "60px",
                textAlign: "right",
                color: "var(--color-text-secondary)",
                fontSize: "0.875rem",
              }}
            >
              {item.value}
            </div>
          </div>
        ))}
      </div>

      {currentItems.length === 0 && (
        <div
          style={{
            padding: "2rem",
            textAlign: "center",
            color: "var(--color-text-secondary)",
          }}
        >
          No sub-items at this level
        </div>
      )}
    </div>
  );
};

export interface SelectableItemsProps {
  items: Array<{ id: string; label: string }>;
  multiSelect?: boolean;
  onSelectionChange?: (selectedIds: string[]) => void;
}

/**
 * SelectableItems - Demonstrates selection interactions with keyboard support
 */
export const SelectableItems: React.FC<SelectableItemsProps> = ({
  items,
  multiSelect = true,
  onSelectionChange,
}) => {
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const [focusedIndex, setFocusedIndex] = useState(-1);

  const handleClick = (id: string, e: React.MouseEvent) => {
    setSelectedIds((prev) => {
      const newSet = new Set(prev);
      if (multiSelect && (e.ctrlKey || e.metaKey)) {
        if (newSet.has(id)) {
          newSet.delete(id);
        } else {
          newSet.add(id);
        }
      } else if (multiSelect && e.shiftKey && focusedIndex >= 0) {
        const clickedIndex = items.findIndex((i) => i.id === id);
        const start = Math.min(focusedIndex, clickedIndex);
        const end = Math.max(focusedIndex, clickedIndex);
        for (let i = start; i <= end; i++) {
          newSet.add(items[i].id);
        }
      } else {
        newSet.clear();
        newSet.add(id);
      }
      onSelectionChange?.(Array.from(newSet));
      return newSet;
    });
    setFocusedIndex(items.findIndex((i) => i.id === id));
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "ArrowDown") {
      e.preventDefault();
      setFocusedIndex((prev) => Math.min(prev + 1, items.length - 1));
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setFocusedIndex((prev) => Math.max(prev - 1, 0));
    } else if (e.key === " " || e.key === "Enter") {
      e.preventDefault();
      if (focusedIndex >= 0) {
        const id = items[focusedIndex].id;
        setSelectedIds((prev) => {
          const newSet = new Set(prev);
          if (multiSelect) {
            if (newSet.has(id)) {
              newSet.delete(id);
            } else {
              newSet.add(id);
            }
          } else {
            newSet.clear();
            newSet.add(id);
          }
          onSelectionChange?.(Array.from(newSet));
          return newSet;
        });
      }
    }
  };

  return (
    <div
      tabIndex={0}
      onKeyDown={handleKeyDown}
      style={{
        display: "flex",
        flexDirection: "column",
        gap: "0.25rem",
        outline: "none",
      }}
    >
      <div
        style={{
          marginBottom: "0.5rem",
          fontSize: "0.75rem",
          color: "var(--color-text-secondary)",
        }}
      >
        {multiSelect
          ? "Ctrl+Click: toggle | Shift+Click: range | Arrow keys + Space: navigate"
          : "Click to select | Arrow keys + Enter: navigate"}
      </div>
      {items.map((item, index) => {
        const isSelected = selectedIds.has(item.id);
        const isFocused = index === focusedIndex;
        return (
          <div
            key={item.id}
            onClick={(e) => handleClick(item.id, e)}
            style={{
              padding: "0.75rem 1rem",
              background: isSelected
                ? "var(--color-primary)"
                : "var(--color-surface)",
              color: isSelected ? "white" : "var(--color-text-primary)",
              borderRadius: "var(--border-radius-sm)",
              border: `2px solid ${
                isFocused ? "var(--color-primary)" : "transparent"
              }`,
              cursor: "pointer",
              transition: "all var(--transition-fast)",
              userSelect: "none",
            }}
          >
            {item.label}
          </div>
        );
      })}
      <div
        style={{
          marginTop: "0.5rem",
          fontSize: "0.875rem",
          color: "var(--color-text-secondary)",
        }}
      >
        Selected: {selectedIds.size} item(s)
      </div>
    </div>
  );
};
