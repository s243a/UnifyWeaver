% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% Data Binding Generator - Declarative Prolog-to-React Bindings
%
% This module provides declarative specifications for binding Prolog
% facts and rules to React component props with reactive updates.
%
% Usage:
%   % Define a data source
%   data_source(sales_data, [
%       predicate(sales_record/4),
%       fields([date, product, quantity, amount])
%   ]).
%
%   % Bind to a chart component
%   binding(sales_chart, sales_data, [
%       x_axis(date),
%       y_axis(amount),
%       group_by(product)
%   ]).
%
%   % Generate React binding code
%   ?- generate_binding_hook(sales_chart, Hook).

:- module(data_binding_generator, [
    % Data source specifications
    data_source/2,                  % data_source(+Name, +Options)
    computed_source/2,              % computed_source(+Name, +Options)

    % Binding specifications
    binding/3,                      % binding(+Component, +Source, +Mapping)
    two_way_binding/3,              % two_way_binding(+Component, +Source, +Mapping)

    % Generation predicates
    generate_binding_hook/2,        % generate_binding_hook(+Component, -Hook)
    generate_data_provider/2,       % generate_data_provider(+Source, -Provider)
    generate_websocket_sync/2,      % generate_websocket_sync(+Source, -SyncCode)
    generate_mutation_handler/2,    % generate_mutation_handler(+Source, -Handler)
    generate_computed_hook/2,       % generate_computed_hook(+Source, -Hook)
    generate_binding_types/2,       % generate_binding_types(+Source, -Types)
    generate_binding_context/2,     % generate_binding_context(+Source, -Context)

    % Query generation
    generate_fetch_query/2,         % generate_fetch_query(+Source, -Query)
    generate_subscribe_query/2,     % generate_subscribe_query(+Source, -Query)
    generate_update_mutation/2,     % generate_update_mutation(+Source, -Mutation)

    % Utility predicates
    get_source_fields/2,            % get_source_fields(+Source, -Fields)
    get_binding_mapping/2,          % get_binding_mapping(+Component, -Mapping)
    is_two_way/1,                   % is_two_way(+Component)
    infer_field_type/2,             % infer_field_type(+Field, -Type)

    % Management
    declare_data_source/2,          % declare_data_source(+Name, +Options)
    declare_binding/3,              % declare_binding(+Component, +Source, +Mapping)
    clear_bindings/0,               % clear_bindings

    % Testing
    test_data_binding_generator/0
]).

:- use_module(library(lists)).

% ============================================================================
% DYNAMIC PREDICATES
% ============================================================================

:- dynamic data_source/2.
:- dynamic computed_source/2.
:- dynamic binding/3.
:- dynamic two_way_binding/3.

:- discontiguous data_source/2.
:- discontiguous binding/3.

% ============================================================================
% DEFAULT DATA SOURCES
% ============================================================================

data_source(default, [
    predicate(data_record/2),
    fields([id, value]),
    primary_key(id),
    refresh_interval(0),
    cache(true)
]).

data_source(time_series, [
    predicate(time_point/3),
    fields([timestamp, series, value]),
    primary_key(timestamp),
    sort_by(timestamp, asc),
    refresh_interval(5000),
    cache(true)
]).

data_source(graph_data, [
    predicate(graph_node/2),
    edge_predicate(graph_edge/3),
    node_fields([id, label, properties]),
    edge_fields([source, target, properties]),
    primary_key(id),
    cache(true)
]).

data_source(hierarchical, [
    predicate(tree_node/3),
    fields([id, parent_id, label]),
    primary_key(id),
    parent_field(parent_id),
    cache(true)
]).

% ============================================================================
% COMPUTED SOURCES (Derived from Prolog rules)
% ============================================================================

computed_source(aggregated, [
    base_source(time_series),
    computation(aggregate),
    group_by([series]),
    aggregations([
        sum(value, total),
        avg(value, average),
        count(_, record_count)
    ]),
    cache(true),
    cache_ttl(60000)
]).

computed_source(filtered, [
    base_source(time_series),
    computation(filter),
    filter_predicate(value > 0),
    cache(false)
]).

computed_source(joined, [
    sources([products, sales]),
    computation(join),
    join_on(product_id),
    join_type(inner),
    cache(true)
]).

% ============================================================================
% DEFAULT BINDINGS
% ============================================================================

binding(line_chart, time_series, [
    x_axis(timestamp),
    y_axis(value),
    series(series),
    transform(none)
]).

binding(bar_chart, aggregated, [
    x_axis(series),
    y_axis(total),
    color(series),
    transform(none)
]).

binding(scatter_plot, time_series, [
    x_axis(timestamp),
    y_axis(value),
    size(constant(5)),
    color(series),
    transform(none)
]).

binding(network_graph, graph_data, [
    nodes(graph_node),
    edges(graph_edge),
    node_id(id),
    node_label(label),
    edge_source(source),
    edge_target(target),
    transform(none)
]).

binding(treemap, hierarchical, [
    id(id),
    parent(parent_id),
    label(label),
    value(size),
    transform(none)
]).

% Two-way binding for editable components
two_way_binding(data_table, time_series, [
    columns([timestamp, series, value]),
    editable([value]),
    on_edit(update_record),
    validation([
        field(value, number),
        field(value, min(0))
    ])
]).

two_way_binding(form_inputs, default, [
    fields([id, value]),
    editable([value]),
    on_change(update_field),
    debounce(300)
]).

% ============================================================================
% BINDING HOOK GENERATION
% ============================================================================

%% generate_binding_hook(+Component, -Hook)
%  Generate a React hook for data binding.
generate_binding_hook(Component, Hook) :-
    (binding(Component, Source, Mapping) -> true ;
     two_way_binding(Component, Source, Mapping)),
    (data_source(Source, SourceOpts) -> true ;
     computed_source(Source, SourceOpts) -> true ;
     data_source(default, SourceOpts)),
    get_source_fields(Source, Fields),
    generate_type_name(Source, TypeName),
    generate_hook_name(Component, HookName),
    (member(refresh_interval(Interval), SourceOpts), Interval > 0 ->
        generate_polling_hook(Component, Source, Mapping, Fields, TypeName, HookName, Hook)
    ;
        generate_static_hook(Component, Source, Mapping, Fields, TypeName, HookName, Hook)
    ).

generate_static_hook(_Component, Source, Mapping, Fields, TypeName, HookName, Hook) :-
    generate_field_types(Fields, FieldTypes),
    generate_mapping_transform(Mapping, Transform),
    atom_string(Source, SourceStr),
    format(atom(Hook), 'import { useState, useEffect, useCallback } from "react";

interface ~w {
~w}

interface ~wResult {
  data: ~w[];
  loading: boolean;
  error: Error | null;
  refetch: () => Promise<void>;
}

export const ~w = (): ~wResult => {
  const [data, setData] = useState<~w[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  const fetchData = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch("/api/data/~w");
      if (!response.ok) throw new Error("Failed to fetch data");
      const rawData = await response.json();
      const transformedData = rawData.map((item: ~w) => (~w));
      setData(transformedData);
    } catch (err) {
      setError(err instanceof Error ? err : new Error("Unknown error"));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  return { data, loading, error, refetch: fetchData };
};
', [TypeName, FieldTypes, HookName, TypeName, HookName, HookName, TypeName,
    SourceStr, TypeName, Transform]).

generate_polling_hook(_Component, Source, Mapping, Fields, TypeName, HookName, Hook) :-
    data_source(Source, SourceOpts),
    member(refresh_interval(Interval), SourceOpts),
    generate_field_types(Fields, FieldTypes),
    generate_mapping_transform(Mapping, Transform),
    atom_string(Source, SourceStr),
    format(atom(Hook), 'import { useState, useEffect, useCallback, useRef } from "react";

interface ~w {
~w}

interface ~wResult {
  data: ~w[];
  loading: boolean;
  error: Error | null;
  refetch: () => Promise<void>;
  isPolling: boolean;
  startPolling: () => void;
  stopPolling: () => void;
}

export const ~w = (): ~wResult => {
  const [data, setData] = useState<~w[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);
  const [isPolling, setIsPolling] = useState(true);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);

  const fetchData = useCallback(async () => {
    try {
      const response = await fetch("/api/data/~w");
      if (!response.ok) throw new Error("Failed to fetch data");
      const rawData = await response.json();
      const transformedData = rawData.map((item: ~w) => (~w));
      setData(transformedData);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err : new Error("Unknown error"));
    } finally {
      setLoading(false);
    }
  }, []);

  const startPolling = useCallback(() => {
    setIsPolling(true);
  }, []);

  const stopPolling = useCallback(() => {
    setIsPolling(false);
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
  }, []);

  useEffect(() => {
    fetchData();
    if (isPolling) {
      intervalRef.current = setInterval(fetchData, ~w);
    }
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [fetchData, isPolling]);

  return { data, loading, error, refetch: fetchData, isPolling, startPolling, stopPolling };
};
', [TypeName, FieldTypes, HookName, TypeName, HookName, HookName, TypeName,
    SourceStr, TypeName, Transform, Interval]).

% ============================================================================
% DATA PROVIDER GENERATION
% ============================================================================

%% generate_data_provider(+Source, -Provider)
%  Generate a React context provider for data.
generate_data_provider(Source, Provider) :-
    (data_source(Source, _) -> true ;
     computed_source(Source, _) -> true ;
     data_source(default, _)),
    get_source_fields(Source, Fields),
    generate_type_name(Source, TypeName),
    generate_provider_name(Source, ProviderName),
    generate_context_name(Source, ContextName),
    generate_field_types(Fields, FieldTypes),
    format(atom(Provider), 'import React, { createContext, useContext, useState, useEffect, useCallback, ReactNode } from "react";

interface ~w {
~w}

interface ~wContextType {
  data: ~w[];
  loading: boolean;
  error: Error | null;
  refetch: () => Promise<void>;
  updateRecord: (id: string, updates: Partial<~w>) => Promise<void>;
  deleteRecord: (id: string) => Promise<void>;
  addRecord: (record: Omit<~w, "id">) => Promise<void>;
}

const ~w = createContext<~wContextType | undefined>(undefined);

export const ~w: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [data, setData] = useState<~w[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  const fetchData = useCallback(async () => {
    setLoading(true);
    try {
      const response = await fetch("/api/data/~w");
      const result = await response.json();
      setData(result);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err : new Error("Unknown error"));
    } finally {
      setLoading(false);
    }
  }, []);

  const updateRecord = useCallback(async (id: string, updates: Partial<~w>) => {
    const response = await fetch(`/api/data/~w/${id}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(updates)
    });
    if (response.ok) await fetchData();
  }, [fetchData]);

  const deleteRecord = useCallback(async (id: string) => {
    const response = await fetch(`/api/data/~w/${id}`, { method: "DELETE" });
    if (response.ok) await fetchData();
  }, [fetchData]);

  const addRecord = useCallback(async (record: Omit<~w, "id">) => {
    const response = await fetch("/api/data/~w", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(record)
    });
    if (response.ok) await fetchData();
  }, [fetchData]);

  useEffect(() => { fetchData(); }, [fetchData]);

  return (
    <~w.Provider value={{ data, loading, error, refetch: fetchData, updateRecord, deleteRecord, addRecord }}>
      {children}
    </~w.Provider>
  );
};

export const use~w = () => {
  const context = useContext(~w);
  if (!context) throw new Error("use~w must be used within ~w");
  return context;
};
', [TypeName, FieldTypes, ContextName, TypeName, TypeName, TypeName,
    ContextName, ContextName, ProviderName, TypeName, Source,
    TypeName, Source, Source, TypeName, Source, ContextName, ContextName,
    TypeName, ContextName, TypeName, ProviderName]).

% ============================================================================
% WEBSOCKET SYNC GENERATION
% ============================================================================

%% generate_websocket_sync(+Source, -SyncCode)
%  Generate WebSocket synchronization code for real-time updates.
generate_websocket_sync(Source, SyncCode) :-
    (data_source(Source, _SourceOpts) -> true ;
     computed_source(Source, _SourceOpts) -> true ;
     data_source(default, _SourceOpts)),
    get_source_fields(Source, Fields),
    generate_type_name(Source, TypeName),
    generate_field_types(Fields, FieldTypes),
    atom_string(Source, SourceStr),
    format(atom(SyncCode), 'import { useState, useEffect, useCallback, useRef } from "react";

interface ~w {
~w}

type SyncEvent =
  | { type: "insert"; record: ~w }
  | { type: "update"; id: string; changes: Partial<~w> }
  | { type: "delete"; id: string }
  | { type: "sync"; data: ~w[] };

interface WebSocketSyncResult {
  data: ~w[];
  connected: boolean;
  error: Error | null;
  send: (event: SyncEvent) => void;
  reconnect: () => void;
}

export const use~wSync = (wsUrl?: string): WebSocketSyncResult => {
  const [data, setData] = useState<~w[]>([]);
  const [connected, setConnected] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  const connect = useCallback(() => {
    const url = wsUrl || `ws://${window.location.host}/ws/~w`;
    const ws = new WebSocket(url);

    ws.onopen = () => {
      setConnected(true);
      setError(null);
      ws.send(JSON.stringify({ type: "subscribe", source: "~w" }));
    };

    ws.onmessage = (event) => {
      const message: SyncEvent = JSON.parse(event.data);
      switch (message.type) {
        case "sync":
          setData(message.data);
          break;
        case "insert":
          setData(prev => [...prev, message.record]);
          break;
        case "update":
          setData(prev => prev.map(item =>
            (item as any).id === message.id ? { ...item, ...message.changes } : item
          ));
          break;
        case "delete":
          setData(prev => prev.filter(item => (item as any).id !== message.id));
          break;
      }
    };

    ws.onerror = () => {
      setError(new Error("WebSocket connection error"));
    };

    ws.onclose = () => {
      setConnected(false);
      // Auto-reconnect after 3 seconds
      reconnectTimeoutRef.current = setTimeout(connect, 3000);
    };

    wsRef.current = ws;
  }, [wsUrl]);

  const send = useCallback((event: SyncEvent) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(event));
    }
  }, []);

  const reconnect = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.close();
    }
    connect();
  }, [connect]);

  useEffect(() => {
    connect();
    return () => {
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [connect]);

  return { data, connected, error, send, reconnect };
};
', [TypeName, FieldTypes, TypeName, TypeName, TypeName, TypeName,
    TypeName, TypeName, SourceStr, SourceStr]).

% ============================================================================
% MUTATION HANDLER GENERATION
% ============================================================================

%% generate_mutation_handler(+Source, -Handler)
%  Generate mutation handlers for two-way bindings.
generate_mutation_handler(Source, Handler) :-
    (data_source(Source, SourceOpts) -> true ;
     data_source(default, SourceOpts)),
    get_source_fields(Source, Fields),
    generate_type_name(Source, TypeName),
    generate_field_types(Fields, FieldTypes),
    (member(primary_key(PK), SourceOpts) -> true ; PK = id),
    atom_string(Source, SourceStr),
    atom_string(PK, PKStr),
    format(atom(Handler), 'import { useState, useCallback } from "react";

interface ~w {
~w}

interface MutationState {
  loading: boolean;
  error: Error | null;
}

interface MutationHandlers {
  create: (data: Omit<~w, "~w">) => Promise<~w | null>;
  update: (id: string, data: Partial<~w>) => Promise<~w | null>;
  remove: (id: string) => Promise<boolean>;
  state: MutationState;
}

export const use~wMutations = (onSuccess?: () => void): MutationHandlers => {
  const [state, setState] = useState<MutationState>({ loading: false, error: null });

  const create = useCallback(async (data: Omit<~w, "~w">): Promise<~w | null> => {
    setState({ loading: true, error: null });
    try {
      const response = await fetch("/api/data/~w", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data)
      });
      if (!response.ok) throw new Error("Failed to create record");
      const result = await response.json();
      onSuccess?.();
      setState({ loading: false, error: null });
      return result;
    } catch (err) {
      const error = err instanceof Error ? err : new Error("Unknown error");
      setState({ loading: false, error });
      return null;
    }
  }, [onSuccess]);

  const update = useCallback(async (id: string, data: Partial<~w>): Promise<~w | null> => {
    setState({ loading: true, error: null });
    try {
      const response = await fetch(`/api/data/~w/${id}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data)
      });
      if (!response.ok) throw new Error("Failed to update record");
      const result = await response.json();
      onSuccess?.();
      setState({ loading: false, error: null });
      return result;
    } catch (err) {
      const error = err instanceof Error ? err : new Error("Unknown error");
      setState({ loading: false, error });
      return null;
    }
  }, [onSuccess]);

  const remove = useCallback(async (id: string): Promise<boolean> => {
    setState({ loading: true, error: null });
    try {
      const response = await fetch(`/api/data/~w/${id}`, { method: "DELETE" });
      if (!response.ok) throw new Error("Failed to delete record");
      onSuccess?.();
      setState({ loading: false, error: null });
      return true;
    } catch (err) {
      const error = err instanceof Error ? err : new Error("Unknown error");
      setState({ loading: false, error });
      return false;
    }
  }, [onSuccess]);

  return { create, update, remove, state };
};
', [TypeName, FieldTypes, TypeName, PKStr, TypeName, TypeName, TypeName,
    TypeName, TypeName, PKStr, TypeName, SourceStr, TypeName, TypeName,
    SourceStr, SourceStr]).

% ============================================================================
% COMPUTED HOOK GENERATION
% ============================================================================

%% generate_computed_hook(+Source, -Hook)
%  Generate hooks for computed/derived data sources.
generate_computed_hook(Source, Hook) :-
    computed_source(Source, SourceOpts),
    member(base_source(BaseSource), SourceOpts),
    member(computation(CompType), SourceOpts),
    get_source_fields(BaseSource, BaseFields),
    generate_type_name(Source, TypeName),
    generate_type_name(BaseSource, BaseTypeName),
    generate_hook_name(Source, HookName),
    generate_computation_code(CompType, SourceOpts, CompCode),
    generate_field_types(BaseFields, FieldTypes),
    format(atom(Hook), 'import { useMemo } from "react";

interface ~w {
~w}

interface ~w {
  // Computed fields based on aggregation/transformation
  [key: string]: unknown;
}

export const ~w = (sourceData: ~w[]): ~w[] => {
  return useMemo(() => {
    if (!sourceData.length) return [];
    ~w
  }, [sourceData]);
};
', [BaseTypeName, FieldTypes, TypeName, HookName, BaseTypeName, TypeName, CompCode]).

generate_computation_code(aggregate, SourceOpts, Code) :-
    member(group_by(GroupFields), SourceOpts),
    member(aggregations(Aggs), SourceOpts),
    generate_group_by_code(GroupFields, GroupCode),
    generate_aggregation_code(Aggs, AggCode),
    format(atom(Code), '
    const grouped = sourceData.reduce((acc, item) => {
      const key = ~w;
      if (!acc[key]) acc[key] = [];
      acc[key].push(item);
      return acc;
    }, {} as Record<string, ~w[]>);

    return Object.entries(grouped).map(([key, items]) => ({
      _groupKey: key,
      ~w
    }));', [GroupCode, 'BaseTypeName', AggCode]).

generate_computation_code(filter, SourceOpts, Code) :-
    member(filter_predicate(Pred), SourceOpts),
    generate_filter_predicate(Pred, FilterCode),
    format(atom(Code), 'return sourceData.filter(item => ~w);', [FilterCode]).

generate_computation_code(_, _, 'return sourceData;').

generate_group_by_code([Field], Code) :-
    atom_string(Field, FieldStr),
    format(atom(Code), 'String(item.~w)', [FieldStr]).
generate_group_by_code(Fields, Code) :-
    is_list(Fields),
    findall(P, (member(F, Fields), atom_string(F, FS), format(atom(P), 'item.~w', [FS])), Parts),
    atomic_list_concat(Parts, ' + "_" + ', Code).

generate_aggregation_code(Aggs, Code) :-
    findall(Line, (
        member(Agg, Aggs),
        generate_single_agg(Agg, Line)
    ), Lines),
    atomic_list_concat(Lines, ',\n      ', Code).

generate_single_agg(sum(Field, Alias), Line) :-
    atom_string(Field, FieldStr),
    atom_string(Alias, AliasStr),
    format(atom(Line), '~w: items.reduce((sum, i) => sum + (i.~w || 0), 0)', [AliasStr, FieldStr]).
generate_single_agg(avg(Field, Alias), Line) :-
    atom_string(Field, FieldStr),
    atom_string(Alias, AliasStr),
    format(atom(Line), '~w: items.reduce((sum, i) => sum + (i.~w || 0), 0) / items.length', [AliasStr, FieldStr]).
generate_single_agg(count(_, Alias), Line) :-
    atom_string(Alias, AliasStr),
    format(atom(Line), '~w: items.length', [AliasStr]).
generate_single_agg(min(Field, Alias), Line) :-
    atom_string(Field, FieldStr),
    atom_string(Alias, AliasStr),
    format(atom(Line), '~w: Math.min(...items.map(i => i.~w || 0))', [AliasStr, FieldStr]).
generate_single_agg(max(Field, Alias), Line) :-
    atom_string(Field, FieldStr),
    atom_string(Alias, AliasStr),
    format(atom(Line), '~w: Math.max(...items.map(i => i.~w || 0))', [AliasStr, FieldStr]).

generate_filter_predicate(Field > Value, Code) :-
    atom_string(Field, FieldStr),
    format(atom(Code), 'item.~w > ~w', [FieldStr, Value]).
generate_filter_predicate(Field < Value, Code) :-
    atom_string(Field, FieldStr),
    format(atom(Code), 'item.~w < ~w', [FieldStr, Value]).
generate_filter_predicate(Field = Value, Code) :-
    atom_string(Field, FieldStr),
    format(atom(Code), 'item.~w === ~w', [FieldStr, Value]).
generate_filter_predicate(_, 'true').

% ============================================================================
% TYPE GENERATION
% ============================================================================

%% generate_binding_types(+Source, -Types)
%  Generate TypeScript interface for the data source.
generate_binding_types(Source, Types) :-
    get_source_fields(Source, Fields),
    generate_type_name(Source, TypeName),
    generate_field_types(Fields, FieldTypes),
    format(atom(Types), 'export interface ~w {
~w}
', [TypeName, FieldTypes]).

generate_type_name(Source, TypeName) :-
    atom_string(Source, SourceStr),
    string_chars(SourceStr, [First|Rest]),
    char_type(First, alpha),
    upcase_atom(First, Upper),
    atom_chars(Upper, [UpperChar]),
    atom_chars(RestAtom, Rest),
    format(atom(TypeName), '~w~wData', [UpperChar, RestAtom]).

generate_hook_name(Component, HookName) :-
    atom_string(Component, CompStr),
    string_chars(CompStr, [First|Rest]),
    upcase_atom(First, Upper),
    atom_chars(Upper, [UpperChar]),
    atom_chars(RestAtom, Rest),
    format(atom(HookName), 'use~w~wData', [UpperChar, RestAtom]).

generate_provider_name(Source, ProviderName) :-
    generate_type_name(Source, TypeName),
    format(atom(ProviderName), '~wProvider', [TypeName]).

generate_context_name(Source, ContextName) :-
    generate_type_name(Source, TypeName),
    format(atom(ContextName), '~wContext', [TypeName]).

generate_field_types(Fields, FieldTypes) :-
    findall(Line, (
        member(Field, Fields),
        infer_field_type(Field, Type),
        atom_string(Field, FieldStr),
        format(atom(Line), '  ~w: ~w;', [FieldStr, Type])
    ), Lines),
    atomic_list_concat(Lines, '\n', FieldTypes).

infer_field_type(id, 'string') :- !.
infer_field_type(Field, 'string') :-
    atom_string(Field, Str),
    (sub_string(Str, _, _, _, "id") ; sub_string(Str, _, _, _, "name") ;
     sub_string(Str, _, _, _, "label") ; sub_string(Str, _, _, _, "title")), !.
infer_field_type(Field, 'number') :-
    atom_string(Field, Str),
    (sub_string(Str, _, _, _, "count") ; sub_string(Str, _, _, _, "amount") ;
     sub_string(Str, _, _, _, "value") ; sub_string(Str, _, _, _, "quantity") ;
     sub_string(Str, _, _, _, "size") ; sub_string(Str, _, _, _, "price")), !.
infer_field_type(Field, 'Date | string') :-
    atom_string(Field, Str),
    (sub_string(Str, _, _, _, "date") ; sub_string(Str, _, _, _, "time") ;
     sub_string(Str, _, _, _, "timestamp")), !.
infer_field_type(Field, 'boolean') :-
    atom_string(Field, Str),
    (sub_string(Str, _, _, _, "is_") ; sub_string(Str, _, _, _, "has_") ;
     sub_string(Str, _, _, _, "enabled") ; sub_string(Str, _, _, _, "active")), !.
infer_field_type(properties, 'Record<string, unknown>') :- !.
infer_field_type(_, 'unknown').

generate_mapping_transform(Mapping, Transform) :-
    findall(P, (
        member(M, Mapping),
        M =.. [Key, Field],
        Key \= transform,
        atom_string(Key, KeyStr),
        atom_string(Field, FieldStr),
        format(atom(P), '~w: item.~w', [KeyStr, FieldStr])
    ), Parts),
    (Parts = [] ->
        Transform = 'item'
    ;
        atomic_list_concat(Parts, ', ', PartsStr),
        format(atom(Transform), '{ ~w }', [PartsStr])
    ).

% ============================================================================
% BINDING CONTEXT GENERATION
% ============================================================================

%% generate_binding_context(+Source, -Context)
%  Generate React context for sharing bound data.
generate_binding_context(Source, Context) :-
    generate_type_name(Source, TypeName),
    generate_context_name(Source, ContextName),
    format(atom(Context), 'import { createContext, useContext } from "react";

interface ~wType {
  data: ~w[];
  loading: boolean;
  error: Error | null;
}

export const ~w = createContext<~wType>({
  data: [],
  loading: true,
  error: null
});

export const use~wContext = () => useContext(~w);
', [ContextName, TypeName, ContextName, ContextName, TypeName, ContextName]).

% ============================================================================
% QUERY GENERATION
% ============================================================================

%% generate_fetch_query(+Source, -Query)
%  Generate fetch query for the data source.
generate_fetch_query(Source, Query) :-
    (data_source(Source, SourceOpts) -> true ;
     data_source(default, SourceOpts)),
    atom_string(Source, SourceStr),
    (member(sort_by(SortField, SortDir), SourceOpts) ->
        format(atom(Query), 'fetch("/api/data/~w?sort=~w&order=~w")', [SourceStr, SortField, SortDir])
    ;
        format(atom(Query), 'fetch("/api/data/~w")', [SourceStr])
    ).

%% generate_subscribe_query(+Source, -Query)
%  Generate WebSocket subscription query.
generate_subscribe_query(Source, Query) :-
    atom_string(Source, SourceStr),
    format(atom(Query), 'ws.send(JSON.stringify({ type: "subscribe", source: "~w" }))', [SourceStr]).

%% generate_update_mutation(+Source, -Mutation)
%  Generate update mutation for two-way binding.
generate_update_mutation(Source, Mutation) :-
    atom_string(Source, SourceStr),
    format(atom(Mutation), 'async (id: string, data: Partial<DataType>) => {
  const response = await fetch(`/api/data/~w/${id}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data)
  });
  return response.json();
}', [SourceStr]).

% ============================================================================
% UTILITY PREDICATES
% ============================================================================

%% get_source_fields(+Source, -Fields)
%  Get field list for a data source.
get_source_fields(Source, Fields) :-
    data_source(Source, SourceOpts),
    member(fields(Fields), SourceOpts), !.
get_source_fields(Source, Fields) :-
    computed_source(Source, SourceOpts),
    member(base_source(BaseSource), SourceOpts),
    get_source_fields(BaseSource, Fields), !.
get_source_fields(_, [id, value]).

%% get_binding_mapping(+Component, -Mapping)
%  Get binding mapping for a component.
get_binding_mapping(Component, Mapping) :-
    binding(Component, _, Mapping), !.
get_binding_mapping(Component, Mapping) :-
    two_way_binding(Component, _, Mapping), !.
get_binding_mapping(_, []).

%% is_two_way(+Component)
%  Check if component has two-way binding.
is_two_way(Component) :-
    two_way_binding(Component, _, _).

% ============================================================================
% MANAGEMENT PREDICATES
% ============================================================================

%% declare_data_source(+Name, +Options)
%  Declare a new data source.
declare_data_source(Name, Options) :-
    retractall(data_source(Name, _)),
    assertz(data_source(Name, Options)).

%% declare_binding(+Component, +Source, +Mapping)
%  Declare a new binding.
declare_binding(Component, Source, Mapping) :-
    retractall(binding(Component, _, _)),
    assertz(binding(Component, Source, Mapping)).

%% clear_bindings/0
%  Clear all dynamic bindings.
clear_bindings :-
    retractall(binding(_, _, _)),
    retractall(two_way_binding(_, _, _)),
    retractall(data_source(_, _)),
    retractall(computed_source(_, _)).

% ============================================================================
% TESTING
% ============================================================================

test_data_binding_generator :-
    writeln('Testing data binding generator...'),

    % Test data source queries
    (data_source(time_series, _) -> writeln('  [PASS] time_series source exists') ; writeln('  [FAIL] time_series source missing')),
    (get_source_fields(time_series, Fields), length(Fields, 3) -> writeln('  [PASS] time_series has 3 fields') ; writeln('  [FAIL] time_series fields')),

    % Test bindings
    (binding(line_chart, time_series, _) -> writeln('  [PASS] line_chart binding exists') ; writeln('  [FAIL] line_chart binding missing')),
    (is_two_way(data_table) -> writeln('  [PASS] data_table is two-way') ; writeln('  [FAIL] data_table two-way')),

    % Test code generation
    (generate_binding_hook(line_chart, Hook), atom_length(Hook, L), L > 100 ->
        writeln('  [PASS] generate_binding_hook produces code') ;
        writeln('  [FAIL] generate_binding_hook')),
    (generate_data_provider(time_series, Provider), atom_length(Provider, PL), PL > 100 ->
        writeln('  [PASS] generate_data_provider produces code') ;
        writeln('  [FAIL] generate_data_provider')),
    (generate_websocket_sync(time_series, Sync), atom_length(Sync, SL), SL > 100 ->
        writeln('  [PASS] generate_websocket_sync produces code') ;
        writeln('  [FAIL] generate_websocket_sync')),
    (generate_mutation_handler(time_series, Handler), atom_length(Handler, HL), HL > 100 ->
        writeln('  [PASS] generate_mutation_handler produces code') ;
        writeln('  [FAIL] generate_mutation_handler')),
    (generate_binding_types(time_series, Types), atom_length(Types, TL), TL > 20 ->
        writeln('  [PASS] generate_binding_types produces code') ;
        writeln('  [FAIL] generate_binding_types')),

    % Test type inference
    (infer_field_type(user_id, 'string') -> writeln('  [PASS] infer user_id as string') ; writeln('  [FAIL] infer user_id')),
    (infer_field_type(amount, 'number') -> writeln('  [PASS] infer amount as number') ; writeln('  [FAIL] infer amount')),
    (infer_field_type(timestamp, 'Date | string') -> writeln('  [PASS] infer timestamp as Date') ; writeln('  [FAIL] infer timestamp')),

    writeln('Data binding generator tests complete.').
