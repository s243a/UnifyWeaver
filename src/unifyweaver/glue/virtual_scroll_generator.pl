% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% Virtual Scroll Generator - Efficient Rendering for Large Lists/Tables
%
% This module provides virtual scrolling patterns for visualizations
% dealing with large numbers of items, rendering only visible items
% while maintaining scroll behavior.
%
% Usage:
%   % Define a virtual scroll configuration
%   virtual_config(my_list, [
%       item_height(40),
%       overscan(5),
%       buffer(100)
%   ]).
%
%   % Generate virtual scroll hook
%   ?- generate_virtual_scroll_hook(my_list, Hook).

:- module(virtual_scroll_generator, [
    % Configuration
    virtual_config/2,                   % virtual_config(+Name, +Options)

    % Generation predicates
    generate_virtual_scroll_hook/2,     % generate_virtual_scroll_hook(+Name, -Hook)
    generate_virtual_list/2,            % generate_virtual_list(+Name, -Component)
    generate_virtual_table/2,           % generate_virtual_table(+Name, -Component)
    generate_virtual_grid/2,            % generate_virtual_grid(+Name, -Component)
    generate_virtual_scroll_css/1,      % generate_virtual_scroll_css(-CSS)

    % Utility predicates
    get_item_height/2,                  % get_item_height(+Name, -Height)

    % Management
    declare_virtual_config/2,           % declare_virtual_config(+Name, +Options)
    clear_virtual_configs/0,            % clear_virtual_configs

    % Testing
    test_virtual_scroll_generator/0
]).

:- use_module(library(lists)).

% ============================================================================
% DYNAMIC PREDICATES
% ============================================================================

:- dynamic virtual_config/2.

:- discontiguous virtual_config/2.

% ============================================================================
% DEFAULT CONFIGURATIONS
% ============================================================================

virtual_config(default, [
    item_height(40),
    overscan(5),
    buffer(100),
    scroll_debounce(16),
    smooth_scroll(true)
]).

virtual_config(compact_list, [
    item_height(32),
    overscan(10),
    buffer(150),
    scroll_debounce(16)
]).

virtual_config(large_table, [
    item_height(48),
    header_height(56),
    overscan(3),
    buffer(50),
    sticky_header(true)
]).

virtual_config(card_grid, [
    item_height(200),
    item_width(250),
    gap(16),
    overscan(2),
    buffer(20)
]).

% ============================================================================
% VIRTUAL SCROLL HOOK GENERATION
% ============================================================================

%% generate_virtual_scroll_hook(+Name, -Hook)
%  Generate a useVirtualScroll hook.
generate_virtual_scroll_hook(Name, Hook) :-
    (virtual_config(Name, Config) -> true ; virtual_config(default, Config)),
    (member(item_height(ItemHeight), Config) -> true ; ItemHeight = 40),
    (member(overscan(Overscan), Config) -> true ; Overscan = 5),
    (member(scroll_debounce(Debounce), Config) -> true ; Debounce = 16),
    atom_string(Name, NameStr),
    format(atom(Hook), 'import { useState, useCallback, useRef, useEffect, useMemo } from "react";

interface VirtualScrollOptions {
  itemCount: number;
  itemHeight: number | ((index: number) => number);
  overscan?: number;
  scrollDebounce?: number;
  onRangeChange?: (startIndex: number, endIndex: number) => void;
}

interface VirtualScrollResult {
  virtualItems: VirtualItem[];
  totalHeight: number;
  containerRef: React.RefObject<HTMLDivElement>;
  scrollToIndex: (index: number, align?: "start" | "center" | "end") => void;
  scrollToOffset: (offset: number) => void;
  isScrolling: boolean;
}

interface VirtualItem {
  index: number;
  start: number;
  size: number;
  measureRef?: (node: HTMLElement | null) => void;
}

export const useVirtualScroll~w = (options: VirtualScrollOptions): VirtualScrollResult => {
  const {
    itemCount,
    itemHeight,
    overscan = ~w,
    scrollDebounce = ~w,
    onRangeChange
  } = options;

  const containerRef = useRef<HTMLDivElement>(null);
  const [scrollTop, setScrollTop] = useState(0);
  const [containerHeight, setContainerHeight] = useState(0);
  const [isScrolling, setIsScrolling] = useState(false);
  const scrollingTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  // Calculate item positions
  const { itemPositions, totalHeight } = useMemo(() => {
    const positions: { start: number; size: number }[] = [];
    let offset = 0;

    for (let i = 0; i < itemCount; i++) {
      const size = typeof itemHeight === "function" ? itemHeight(i) : itemHeight;
      positions.push({ start: offset, size });
      offset += size;
    }

    return { itemPositions: positions, totalHeight: offset };
  }, [itemCount, itemHeight]);

  // Find visible range using binary search
  const findStartIndex = useCallback((scrollTop: number): number => {
    let low = 0;
    let high = itemCount - 1;

    while (low <= high) {
      const mid = Math.floor((low + high) / 2);
      const pos = itemPositions[mid];

      if (pos.start <= scrollTop && pos.start + pos.size > scrollTop) {
        return mid;
      } else if (pos.start + pos.size <= scrollTop) {
        low = mid + 1;
      } else {
        high = mid - 1;
      }
    }

    return Math.max(0, Math.min(low, itemCount - 1));
  }, [itemPositions, itemCount]);

  // Calculate visible items
  const virtualItems = useMemo((): VirtualItem[] => {
    if (itemCount === 0 || containerHeight === 0) return [];

    const startIndex = Math.max(0, findStartIndex(scrollTop) - overscan);
    let endIndex = startIndex;

    let currentOffset = itemPositions[startIndex]?.start || 0;
    const maxOffset = scrollTop + containerHeight;

    while (endIndex < itemCount && currentOffset < maxOffset + overscan * (typeof itemHeight === "number" ? itemHeight : ~w)) {
      currentOffset += itemPositions[endIndex]?.size || 0;
      endIndex++;
    }

    endIndex = Math.min(itemCount - 1, endIndex + overscan);

    onRangeChange?.(startIndex, endIndex);

    const items: VirtualItem[] = [];
    for (let i = startIndex; i <= endIndex; i++) {
      items.push({
        index: i,
        start: itemPositions[i].start,
        size: itemPositions[i].size
      });
    }

    return items;
  }, [scrollTop, containerHeight, itemCount, itemPositions, overscan, findStartIndex, onRangeChange, itemHeight]);

  // Scroll handler with debounce
  const handleScroll = useCallback((e: Event) => {
    const target = e.target as HTMLDivElement;
    setScrollTop(target.scrollTop);
    setIsScrolling(true);

    if (scrollingTimeoutRef.current) {
      clearTimeout(scrollingTimeoutRef.current);
    }

    scrollingTimeoutRef.current = setTimeout(() => {
      setIsScrolling(false);
    }, scrollDebounce);
  }, [scrollDebounce]);

  // Resize observer
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const resizeObserver = new ResizeObserver((entries) => {
      for (const entry of entries) {
        setContainerHeight(entry.contentRect.height);
      }
    });

    resizeObserver.observe(container);
    setContainerHeight(container.clientHeight);

    container.addEventListener("scroll", handleScroll, { passive: true });

    return () => {
      resizeObserver.disconnect();
      container.removeEventListener("scroll", handleScroll);
      if (scrollingTimeoutRef.current) {
        clearTimeout(scrollingTimeoutRef.current);
      }
    };
  }, [handleScroll]);

  const scrollToIndex = useCallback((index: number, align: "start" | "center" | "end" = "start") => {
    const container = containerRef.current;
    if (!container || index < 0 || index >= itemCount) return;

    const pos = itemPositions[index];
    let targetOffset: number;

    switch (align) {
      case "center":
        targetOffset = pos.start - (containerHeight - pos.size) / 2;
        break;
      case "end":
        targetOffset = pos.start - containerHeight + pos.size;
        break;
      default:
        targetOffset = pos.start;
    }

    container.scrollTop = Math.max(0, Math.min(targetOffset, totalHeight - containerHeight));
  }, [itemPositions, containerHeight, totalHeight, itemCount]);

  const scrollToOffset = useCallback((offset: number) => {
    const container = containerRef.current;
    if (!container) return;
    container.scrollTop = Math.max(0, Math.min(offset, totalHeight - containerHeight));
  }, [totalHeight, containerHeight]);

  return {
    virtualItems,
    totalHeight,
    containerRef,
    scrollToIndex,
    scrollToOffset,
    isScrolling
  };
};
', [NameStr, Overscan, Debounce, ItemHeight]).

% ============================================================================
% VIRTUAL LIST GENERATION
% ============================================================================

%% generate_virtual_list(+Name, -Component)
%  Generate a VirtualList component.
generate_virtual_list(Name, Component) :-
    (virtual_config(Name, Config) -> true ; virtual_config(default, Config)),
    (member(item_height(ItemHeight), Config) -> true ; ItemHeight = 40),
    (member(overscan(Overscan), Config) -> true ; Overscan = 5),
    atom_string(Name, NameStr),
    format(atom(Component), 'import React, { useMemo } from "react";
import { useVirtualScroll~w } from "./useVirtualScroll";

interface VirtualListProps<T> {
  items: T[];
  itemHeight?: number | ((item: T, index: number) => number);
  renderItem: (item: T, index: number, style: React.CSSProperties) => React.ReactNode;
  overscan?: number;
  className?: string;
  onEndReached?: () => void;
  endReachedThreshold?: number;
}

export function VirtualList~w<T>({
  items,
  itemHeight = ~w,
  renderItem,
  overscan = ~w,
  className = "",
  onEndReached,
  endReachedThreshold = 5
}: VirtualListProps<T>): JSX.Element {
  const getItemHeight = useMemo(() => {
    if (typeof itemHeight === "number") {
      return () => itemHeight;
    }
    return (index: number) => itemHeight(items[index], index);
  }, [itemHeight, items]);

  const {
    virtualItems,
    totalHeight,
    containerRef,
    isScrolling
  } = useVirtualScroll~w({
    itemCount: items.length,
    itemHeight: getItemHeight,
    overscan,
    onRangeChange: (start, end) => {
      if (onEndReached && end >= items.length - endReachedThreshold) {
        onEndReached();
      }
    }
  });

  return (
    <div
      ref={containerRef}
      className={`virtual-list ${className}`}
      style={{ overflow: "auto", position: "relative" }}
    >
      <div
        className="virtual-list__content"
        style={{
          height: totalHeight,
          width: "100%",
          position: "relative"
        }}
      >
        {virtualItems.map(({ index, start, size }) => {
          const style: React.CSSProperties = {
            position: "absolute",
            top: 0,
            left: 0,
            width: "100%",
            height: size,
            transform: `translateY(${start}px)`
          };

          return (
            <div key={index} style={style} className="virtual-list__item">
              {renderItem(items[index], index, style)}
            </div>
          );
        })}
      </div>

      {isScrolling && (
        <div className="virtual-list__scrolling-indicator" aria-hidden="true" />
      )}
    </div>
  );
}
', [NameStr, NameStr, ItemHeight, Overscan, NameStr]).

% ============================================================================
% VIRTUAL TABLE GENERATION
% ============================================================================

%% generate_virtual_table(+Name, -Component)
%  Generate a VirtualTable component.
generate_virtual_table(Name, Component) :-
    (virtual_config(Name, Config) -> true ; virtual_config(large_table, Config)),
    (member(item_height(ItemHeight), Config) -> true ; ItemHeight = 48),
    (member(header_height(HeaderHeight), Config) -> true ; HeaderHeight = 56),
    (member(overscan(Overscan), Config) -> true ; Overscan = 3),
    atom_string(Name, NameStr),
    format(atom(Component), 'import React, { useMemo, useRef } from "react";
import { useVirtualScroll~w } from "./useVirtualScroll";

interface Column<T> {
  key: keyof T | string;
  header: string;
  width?: number | string;
  render?: (value: unknown, row: T, index: number) => React.ReactNode;
  sortable?: boolean;
  align?: "left" | "center" | "right";
}

interface VirtualTableProps<T> {
  data: T[];
  columns: Column<T>[];
  rowHeight?: number;
  headerHeight?: number;
  overscan?: number;
  className?: string;
  onRowClick?: (row: T, index: number) => void;
  selectedIndex?: number;
  stickyHeader?: boolean;
  sortColumn?: string;
  sortDirection?: "asc" | "desc";
  onSort?: (column: string) => void;
}

export function VirtualTable~w<T extends Record<string, unknown>>({
  data,
  columns,
  rowHeight = ~w,
  headerHeight = ~w,
  overscan = ~w,
  className = "",
  onRowClick,
  selectedIndex,
  stickyHeader = true,
  sortColumn,
  sortDirection,
  onSort
}: VirtualTableProps<T>): JSX.Element {
  const headerRef = useRef<HTMLDivElement>(null);

  const {
    virtualItems,
    totalHeight,
    containerRef,
    scrollToIndex
  } = useVirtualScroll~w({
    itemCount: data.length,
    itemHeight: rowHeight,
    overscan
  });

  const getCellValue = (row: T, key: string): unknown => {
    return key.split(".").reduce((obj: unknown, k) => {
      if (obj && typeof obj === "object") {
        return (obj as Record<string, unknown>)[k];
      }
      return undefined;
    }, row);
  };

  const columnWidths = useMemo(() => {
    return columns.map(col => col.width || "1fr").join(" ");
  }, [columns]);

  return (
    <div className={`virtual-table ${className}`}>
      {/* Header */}
      <div
        ref={headerRef}
        className={`virtual-table__header ${stickyHeader ? "sticky" : ""}`}
        style={{
          display: "grid",
          gridTemplateColumns: columnWidths,
          height: headerHeight,
          position: stickyHeader ? "sticky" : "relative",
          top: 0,
          zIndex: 10
        }}
      >
        {columns.map((col) => (
          <div
            key={String(col.key)}
            className={`virtual-table__header-cell ${col.sortable ? "sortable" : ""}`}
            style={{ textAlign: col.align || "left" }}
            onClick={() => col.sortable && onSort?.(String(col.key))}
            role={col.sortable ? "button" : undefined}
            tabIndex={col.sortable ? 0 : undefined}
          >
            {col.header}
            {sortColumn === col.key && (
              <span className="sort-indicator">
                {sortDirection === "asc" ? " ▲" : " ▼"}
              </span>
            )}
          </div>
        ))}
      </div>

      {/* Body */}
      <div
        ref={containerRef}
        className="virtual-table__body"
        style={{
          overflow: "auto",
          height: `calc(100% - ${headerHeight}px)`
        }}
      >
        <div
          style={{
            height: totalHeight,
            position: "relative"
          }}
        >
          {virtualItems.map(({ index, start, size }) => {
            const row = data[index];
            const isSelected = selectedIndex === index;

            return (
              <div
                key={index}
                className={`virtual-table__row ${isSelected ? "selected" : ""}`}
                style={{
                  display: "grid",
                  gridTemplateColumns: columnWidths,
                  height: size,
                  position: "absolute",
                  top: start,
                  left: 0,
                  right: 0
                }}
                onClick={() => onRowClick?.(row, index)}
                role="row"
                aria-rowindex={index + 1}
                aria-selected={isSelected}
              >
                {columns.map((col) => {
                  const value = getCellValue(row, String(col.key));
                  return (
                    <div
                      key={String(col.key)}
                      className="virtual-table__cell"
                      style={{ textAlign: col.align || "left" }}
                      role="gridcell"
                    >
                      {col.render ? col.render(value, row, index) : String(value ?? "")}
                    </div>
                  );
                })}
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
', [NameStr, NameStr, ItemHeight, HeaderHeight, Overscan, NameStr]).

% ============================================================================
% VIRTUAL GRID GENERATION
% ============================================================================

%% generate_virtual_grid(+Name, -Component)
%  Generate a VirtualGrid component.
generate_virtual_grid(Name, Component) :-
    (virtual_config(Name, Config) -> true ; virtual_config(card_grid, Config)),
    (member(item_height(ItemHeight), Config) -> true ; ItemHeight = 200),
    (member(item_width(ItemWidth), Config) -> true ; ItemWidth = 250),
    (member(gap(Gap), Config) -> true ; Gap = 16),
    (member(overscan(Overscan), Config) -> true ; Overscan = 2),
    atom_string(Name, NameStr),
    format(atom(Component), 'import React, { useMemo, useRef, useState, useEffect } from "react";

interface VirtualGridProps<T> {
  items: T[];
  renderItem: (item: T, index: number) => React.ReactNode;
  itemWidth?: number;
  itemHeight?: number;
  gap?: number;
  overscan?: number;
  className?: string;
}

export function VirtualGrid~w<T>({
  items,
  renderItem,
  itemWidth = ~w,
  itemHeight = ~w,
  gap = ~w,
  overscan = ~w,
  className = ""
}: VirtualGridProps<T>): JSX.Element {
  const containerRef = useRef<HTMLDivElement>(null);
  const [containerWidth, setContainerWidth] = useState(0);
  const [containerHeight, setContainerHeight] = useState(0);
  const [scrollTop, setScrollTop] = useState(0);

  // Calculate grid dimensions
  const columnsCount = useMemo(() => {
    if (containerWidth === 0) return 1;
    return Math.max(1, Math.floor((containerWidth + gap) / (itemWidth + gap)));
  }, [containerWidth, itemWidth, gap]);

  const rowsCount = useMemo(() => {
    return Math.ceil(items.length / columnsCount);
  }, [items.length, columnsCount]);

  const rowHeight = itemHeight + gap;
  const totalHeight = rowsCount * rowHeight - gap;

  // Calculate visible range
  const visibleRange = useMemo(() => {
    const startRow = Math.max(0, Math.floor(scrollTop / rowHeight) - overscan);
    const endRow = Math.min(
      rowsCount,
      Math.ceil((scrollTop + containerHeight) / rowHeight) + overscan
    );

    const startIndex = startRow * columnsCount;
    const endIndex = Math.min(items.length, endRow * columnsCount);

    return { startIndex, endIndex, startRow };
  }, [scrollTop, containerHeight, rowHeight, columnsCount, rowsCount, overscan, items.length]);

  // Handle scroll
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const handleScroll = () => {
      setScrollTop(container.scrollTop);
    };

    const resizeObserver = new ResizeObserver((entries) => {
      for (const entry of entries) {
        setContainerWidth(entry.contentRect.width);
        setContainerHeight(entry.contentRect.height);
      }
    });

    container.addEventListener("scroll", handleScroll, { passive: true });
    resizeObserver.observe(container);

    return () => {
      container.removeEventListener("scroll", handleScroll);
      resizeObserver.disconnect();
    };
  }, []);

  // Generate visible items
  const visibleItems = useMemo(() => {
    const result: { item: T; index: number; style: React.CSSProperties }[] = [];

    for (let i = visibleRange.startIndex; i < visibleRange.endIndex; i++) {
      const row = Math.floor(i / columnsCount);
      const col = i % columnsCount;

      result.push({
        item: items[i],
        index: i,
        style: {
          position: "absolute",
          left: col * (itemWidth + gap),
          top: row * rowHeight,
          width: itemWidth,
          height: itemHeight
        }
      });
    }

    return result;
  }, [visibleRange, items, columnsCount, itemWidth, itemHeight, gap, rowHeight]);

  return (
    <div
      ref={containerRef}
      className={`virtual-grid ${className}`}
      style={{ overflow: "auto", position: "relative" }}
    >
      <div
        className="virtual-grid__content"
        style={{
          height: totalHeight,
          position: "relative"
        }}
      >
        {visibleItems.map(({ item, index, style }) => (
          <div
            key={index}
            className="virtual-grid__item"
            style={style}
          >
            {renderItem(item, index)}
          </div>
        ))}
      </div>
    </div>
  );
}
', [NameStr, ItemWidth, ItemHeight, Gap, Overscan]).

% ============================================================================
% CSS GENERATION
% ============================================================================

%% generate_virtual_scroll_css(-CSS)
%  Generate CSS for virtual scroll components.
generate_virtual_scroll_css(CSS) :-
    format(atom(CSS), '/* Virtual Scroll Components */

.virtual-list {
  width: 100%;
  height: 100%;
  contain: strict;
}

.virtual-list__content {
  contain: layout style;
}

.virtual-list__item {
  contain: layout style;
  will-change: transform;
}

.virtual-list__scrolling-indicator {
  position: fixed;
  top: 0;
  right: 0;
  width: 4px;
  height: 100%;
  background: var(--color-primary, #3b82f6);
  opacity: 0.3;
  pointer-events: none;
  animation: fade-out 150ms ease forwards;
}

@keyframes fade-out {
  to { opacity: 0; }
}

/* Virtual Table */
.virtual-table {
  width: 100%;
  height: 100%;
  display: flex;
  flex-direction: column;
  border: 1px solid var(--color-border, #e2e8f0);
  border-radius: var(--border-radius-md, 8px);
  overflow: hidden;
}

.virtual-table__header {
  background: var(--color-surface, #f8fafc);
  border-bottom: 2px solid var(--color-border, #e2e8f0);
  font-weight: 600;
}

.virtual-table__header.sticky {
  position: sticky;
  top: 0;
  z-index: 10;
}

.virtual-table__header-cell {
  padding: var(--spacing-sm, 8px) var(--spacing-md, 16px);
  display: flex;
  align-items: center;
  gap: var(--spacing-xs, 4px);
  border-right: 1px solid var(--color-border, #e2e8f0);
}

.virtual-table__header-cell:last-child {
  border-right: none;
}

.virtual-table__header-cell.sortable {
  cursor: pointer;
  user-select: none;
}

.virtual-table__header-cell.sortable:hover {
  background: var(--color-background, #f1f5f9);
}

.virtual-table__body {
  flex: 1;
  contain: strict;
}

.virtual-table__row {
  border-bottom: 1px solid var(--color-border, #e2e8f0);
  transition: background 150ms ease;
  contain: layout style;
}

.virtual-table__row:hover {
  background: var(--color-background, #f8fafc);
}

.virtual-table__row.selected {
  background: var(--color-primary-bg, #eff6ff);
}

.virtual-table__cell {
  padding: var(--spacing-sm, 8px) var(--spacing-md, 16px);
  display: flex;
  align-items: center;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  border-right: 1px solid var(--color-border, #e2e8f0);
}

.virtual-table__cell:last-child {
  border-right: none;
}

.sort-indicator {
  font-size: 0.75em;
  color: var(--color-text-secondary, #64748b);
}

/* Virtual Grid */
.virtual-grid {
  width: 100%;
  height: 100%;
  contain: strict;
}

.virtual-grid__content {
  contain: layout style;
}

.virtual-grid__item {
  contain: layout style;
  will-change: transform;
}

/* Performance optimizations */
.virtual-list,
.virtual-table,
.virtual-grid {
  -webkit-overflow-scrolling: touch;
  overscroll-behavior: contain;
}

/* Reduced motion */
@media (prefers-reduced-motion: reduce) {
  .virtual-list__item,
  .virtual-table__row,
  .virtual-grid__item {
    will-change: auto;
  }

  .virtual-list__scrolling-indicator {
    animation: none;
  }
}
', []).

% ============================================================================
% UTILITY PREDICATES
% ============================================================================

%% get_item_height(+Name, -Height)
%  Get the default item height for a configuration.
get_item_height(Name, Height) :-
    virtual_config(Name, Config),
    member(item_height(Height), Config), !.
get_item_height(_, 40).

% ============================================================================
% MANAGEMENT PREDICATES
% ============================================================================

%% declare_virtual_config(+Name, +Options)
%  Declare a virtual scroll configuration.
declare_virtual_config(Name, Options) :-
    retractall(virtual_config(Name, _)),
    assertz(virtual_config(Name, Options)).

%% clear_virtual_configs/0
%  Clear all virtual scroll configurations.
clear_virtual_configs :-
    retractall(virtual_config(_, _)).

% ============================================================================
% TESTING
% ============================================================================

test_virtual_scroll_generator :-
    writeln('Testing virtual scroll generator...'),

    % Test config existence
    (virtual_config(default, _) ->
        writeln('  [PASS] default config exists') ;
        writeln('  [FAIL] default config')),

    (virtual_config(large_table, _) ->
        writeln('  [PASS] large_table config exists') ;
        writeln('  [FAIL] large_table config')),

    % Test hook generation
    (generate_virtual_scroll_hook(default, Hook), atom_length(Hook, HL), HL > 1000 ->
        writeln('  [PASS] generate_virtual_scroll_hook produces code') ;
        writeln('  [FAIL] generate_virtual_scroll_hook')),

    % Test list generation
    (generate_virtual_list(default, List), atom_length(List, LL), LL > 500 ->
        writeln('  [PASS] generate_virtual_list produces code') ;
        writeln('  [FAIL] generate_virtual_list')),

    % Test table generation
    (generate_virtual_table(large_table, Table), atom_length(Table, TL), TL > 1000 ->
        writeln('  [PASS] generate_virtual_table produces code') ;
        writeln('  [FAIL] generate_virtual_table')),

    % Test grid generation
    (generate_virtual_grid(card_grid, Grid), atom_length(Grid, GL), GL > 500 ->
        writeln('  [PASS] generate_virtual_grid produces code') ;
        writeln('  [FAIL] generate_virtual_grid')),

    % Test CSS generation
    (generate_virtual_scroll_css(CSS), atom_length(CSS, CL), CL > 500 ->
        writeln('  [PASS] generate_virtual_scroll_css produces code') ;
        writeln('  [FAIL] generate_virtual_scroll_css')),

    writeln('Virtual scroll generator tests complete.').
