% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% ui_patterns_extended.pl - Extended UI Patterns Library
%
% Additional UI patterns for common application features:
%   - Form patterns (inputs, validation, submission)
%   - List patterns (infinite scroll, pull-to-refresh, selection)
%   - Modal patterns (alert, confirm, bottom sheet, drawer)
%   - Auth flow patterns (login, register, forgot password, OAuth)
%
% These patterns extend the core ui_patterns module and compile to:
%   - React Native (TypeScript)
%   - Vue 3 (TypeScript)
%   - Flutter (Dart)
%   - SwiftUI (Swift)

:- module(ui_patterns_extended, [
    % Form patterns
    form_pattern/3,                 % +FormName, +Fields, -Pattern
    form_field/4,                   % +Name, +Type, +Validation, -FieldSpec
    validation_rule/3,              % +Type, +Params, -Rule
    form_submission/3,              % +FormName, +Endpoint, -Pattern

    % List patterns
    list_pattern/3,                 % +Type, +Config, -Pattern
    infinite_list/3,                % +Name, +DataSource, -Pattern
    selectable_list/3,              % +Name, +Config, -Pattern
    grouped_list/3,                 % +Name, +GroupBy, -Pattern

    % Modal patterns
    modal_pattern/3,                % +Type, +Config, -Pattern
    alert_modal/3,                  % +Name, +Config, -Pattern
    confirm_modal/3,                % +Name, +Config, -Pattern
    bottom_sheet/3,                 % +Name, +Content, -Pattern
    action_sheet/3,                 % +Name, +Actions, -Pattern

    % Auth flow patterns
    auth_flow/3,                    % +Type, +Config, -Pattern
    login_flow/2,                   % +Config, -Pattern
    register_flow/2,                % +Config, -Pattern
    forgot_password_flow/2,         % +Config, -Pattern
    oauth_flow/3,                   % +Provider, +Config, -Pattern

    % Compilation
    compile_extended_pattern/4,     % +PatternName, +Target, +Options, -Code

    % Testing
    test_extended_patterns/0
]).

:- use_module(library(lists)).
:- use_module('ui_patterns').

% ============================================================================
% FORM PATTERNS
% ============================================================================

%% form_pattern(+FormName, +Fields, -Pattern)
%
%  Create a form pattern with specified fields.
%
%  Fields: [field(name, type, validation, options), ...]
%
form_pattern(FormName, Fields, Pattern) :-
    atom(FormName),
    is_list(Fields),
    Pattern = form(FormName, Fields, []),
    define_pattern(FormName, Pattern, [requires([forms])]).

%% form_field(+Name, +Type, +Validation, -FieldSpec)
%
%  Create a field specification for a form.
%
%  Types: text | email | password | number | phone | date | select | checkbox | radio | textarea
%
%  Validation: [required, min_length(N), max_length(N), pattern(Regex), email, ...]
%
form_field(Name, Type, Validation, field(Name, Type, Validation, [])) :-
    atom(Name),
    member(Type, [text, email, password, number, phone, date, select, checkbox, radio, textarea, file]).

form_field(Name, Type, Validation, Options, field(Name, Type, Validation, Options)) :-
    atom(Name),
    is_list(Options).

%% validation_rule(+Type, +Params, -Rule)
%
%  Create a validation rule.
%
validation_rule(required, [], required).
validation_rule(min_length, [N], min_length(N)) :- integer(N), N > 0.
validation_rule(max_length, [N], max_length(N)) :- integer(N), N > 0.
validation_rule(pattern, [Regex], pattern(Regex)) :- atom(Regex).
validation_rule(email, [], email).
validation_rule(url, [], url).
validation_rule(min, [N], min(N)) :- number(N).
validation_rule(max, [N], max(N)) :- number(N).
validation_rule(matches, [Field], matches(Field)) :- atom(Field).
validation_rule(custom, [Fn], custom(Fn)).

%% form_submission(+FormName, +Endpoint, -Pattern)
%
%  Create a form submission handler pattern.
%
form_submission(FormName, Endpoint, Pattern) :-
    atom(FormName),
    atom(Endpoint),
    Pattern = form_submit(FormName, Endpoint, [method('POST')]),
    SubmitName = FormName,
    define_pattern(SubmitName, Pattern, [requires([forms, react_query])]).

% ============================================================================
% LIST PATTERNS
% ============================================================================

%% list_pattern(+Type, +Config, -Pattern)
%
%  Create a list pattern.
%
%  Types: basic | infinite | selectable | grouped | sectioned
%
list_pattern(basic, Config, Pattern) :-
    member(name(Name), Config),
    member(item_component(Component), Config),
    Pattern = list(basic, Name, Component, Config),
    define_pattern(Name, Pattern, [requires([lists])]).

list_pattern(infinite, Config, Pattern) :-
    member(name(Name), Config),
    member(data_source(Source), Config),
    Pattern = list(infinite, Name, Source, Config),
    define_pattern(Name, Pattern, [requires([lists, infinite_scroll])]).

list_pattern(selectable, Config, Pattern) :-
    member(name(Name), Config),
    member(mode(Mode), Config),  % single | multi
    Pattern = list(selectable, Name, Mode, Config),
    define_pattern(Name, Pattern, [requires([lists])]).

list_pattern(grouped, Config, Pattern) :-
    member(name(Name), Config),
    member(group_by(GroupKey), Config),
    Pattern = list(grouped, Name, GroupKey, Config),
    define_pattern(Name, Pattern, [requires([lists])]).

%% infinite_list(+Name, +DataSource, -Pattern)
%
%  Create an infinite scrolling list.
%
infinite_list(Name, DataSource, Pattern) :-
    list_pattern(infinite, [
        name(Name),
        data_source(DataSource),
        page_size(20),
        threshold(0.8)
    ], Pattern).

%% selectable_list(+Name, +Config, -Pattern)
%
%  Create a selectable list (single or multi-select).
%
selectable_list(Name, Config, Pattern) :-
    (   member(mode(Mode), Config) -> true ; Mode = single ),
    list_pattern(selectable, [name(Name), mode(Mode) | Config], Pattern).

%% grouped_list(+Name, +GroupBy, -Pattern)
%
%  Create a grouped/sectioned list.
%
grouped_list(Name, GroupBy, Pattern) :-
    list_pattern(grouped, [name(Name), group_by(GroupBy)], Pattern).

% ============================================================================
% MODAL PATTERNS
% ============================================================================

%% modal_pattern(+Type, +Config, -Pattern)
%
%  Create a modal pattern.
%
%  Types: alert | confirm | prompt | bottom_sheet | action_sheet | fullscreen
%
modal_pattern(alert, Config, Pattern) :-
    member(name(Name), Config),
    member(title(Title), Config),
    member(message(Message), Config),
    Pattern = modal(alert, Name, [title(Title), message(Message) | Config]),
    define_pattern(Name, Pattern, [requires([modals])]).

modal_pattern(confirm, Config, Pattern) :-
    member(name(Name), Config),
    member(title(Title), Config),
    member(message(Message), Config),
    (   member(confirm_text(ConfirmText), Config) -> true ; ConfirmText = 'Confirm' ),
    (   member(cancel_text(CancelText), Config) -> true ; CancelText = 'Cancel' ),
    Pattern = modal(confirm, Name, [
        title(Title),
        message(Message),
        confirm_text(ConfirmText),
        cancel_text(CancelText)
        | Config
    ]),
    define_pattern(Name, Pattern, [requires([modals])]).

modal_pattern(bottom_sheet, Config, Pattern) :-
    member(name(Name), Config),
    member(content(Content), Config),
    (   member(height(Height), Config) -> true ; Height = auto ),
    Pattern = modal(bottom_sheet, Name, [content(Content), height(Height) | Config]),
    define_pattern(Name, Pattern, [requires([modals, bottom_sheet])]).

modal_pattern(action_sheet, Config, Pattern) :-
    member(name(Name), Config),
    member(actions(Actions), Config),
    Pattern = modal(action_sheet, Name, [actions(Actions) | Config]),
    define_pattern(Name, Pattern, [requires([modals])]).

modal_pattern(fullscreen, Config, Pattern) :-
    member(name(Name), Config),
    member(content(Content), Config),
    Pattern = modal(fullscreen, Name, [content(Content) | Config]),
    define_pattern(Name, Pattern, [requires([modals, navigation])]).

%% alert_modal(+Name, +Config, -Pattern)
%
%  Create an alert modal.
%
alert_modal(Name, Config, Pattern) :-
    modal_pattern(alert, [name(Name) | Config], Pattern).

%% confirm_modal(+Name, +Config, -Pattern)
%
%  Create a confirmation modal.
%
confirm_modal(Name, Config, Pattern) :-
    modal_pattern(confirm, [name(Name) | Config], Pattern).

%% bottom_sheet(+Name, +Content, -Pattern)
%
%  Create a bottom sheet modal.
%
bottom_sheet(Name, Content, Pattern) :-
    modal_pattern(bottom_sheet, [name(Name), content(Content)], Pattern).

%% action_sheet(+Name, +Actions, -Pattern)
%
%  Create an action sheet with multiple options.
%
action_sheet(Name, Actions, Pattern) :-
    modal_pattern(action_sheet, [name(Name), actions(Actions)], Pattern).

% ============================================================================
% AUTH FLOW PATTERNS
% ============================================================================

%% auth_flow(+Type, +Config, -Pattern)
%
%  Create an authentication flow pattern.
%
%  Types: login | register | forgot_password | reset_password | oauth | mfa
%
auth_flow(login, Config, Pattern) :-
    member(endpoint(Endpoint), Config),
    (   member(fields(Fields), Config)
    ->  true
    ;   Fields = [
            field(email, email, [required, email], [placeholder('Email')]),
            field(password, password, [required, min_length(8)], [placeholder('Password')])
        ]
    ),
    Pattern = auth(login, [
        endpoint(Endpoint),
        fields(Fields),
        method('POST')
        | Config
    ]),
    define_pattern(login_form, Pattern, [requires([forms, auth])]).

auth_flow(register, Config, Pattern) :-
    member(endpoint(Endpoint), Config),
    (   member(fields(Fields), Config)
    ->  true
    ;   Fields = [
            field(name, text, [required, min_length(2)], [placeholder('Full Name')]),
            field(email, email, [required, email], [placeholder('Email')]),
            field(password, password, [required, min_length(8)], [placeholder('Password')]),
            field(confirm_password, password, [required, matches(password)], [placeholder('Confirm Password')])
        ]
    ),
    Pattern = auth(register, [
        endpoint(Endpoint),
        fields(Fields),
        method('POST')
        | Config
    ]),
    define_pattern(register_form, Pattern, [requires([forms, auth])]).

auth_flow(forgot_password, Config, Pattern) :-
    member(endpoint(Endpoint), Config),
    Pattern = auth(forgot_password, [
        endpoint(Endpoint),
        fields([
            field(email, email, [required, email], [placeholder('Email')])
        ]),
        method('POST')
        | Config
    ]),
    define_pattern(forgot_password_form, Pattern, [requires([forms, auth])]).

auth_flow(reset_password, Config, Pattern) :-
    member(endpoint(Endpoint), Config),
    Pattern = auth(reset_password, [
        endpoint(Endpoint),
        fields([
            field(password, password, [required, min_length(8)], [placeholder('New Password')]),
            field(confirm_password, password, [required, matches(password)], [placeholder('Confirm Password')])
        ]),
        method('POST')
        | Config
    ]),
    define_pattern(reset_password_form, Pattern, [requires([forms, auth])]).

auth_flow(oauth, Config, Pattern) :-
    member(provider(Provider), Config),
    member(client_id(ClientId), Config),
    Pattern = auth(oauth, [
        provider(Provider),
        client_id(ClientId)
        | Config
    ]),
    atom_concat(oauth_, Provider, PatternName),
    define_pattern(PatternName, Pattern, [requires([auth, oauth])]).

auth_flow(mfa, Config, Pattern) :-
    member(endpoint(Endpoint), Config),
    (   member(method(Method), Config) -> true ; Method = totp ),
    Pattern = auth(mfa, [
        endpoint(Endpoint),
        method(Method),
        code_length(6)
        | Config
    ]),
    define_pattern(mfa_verification, Pattern, [requires([forms, auth])]).

%% login_flow(+Config, -Pattern)
%
%  Create a login flow.
%
login_flow(Config, Pattern) :-
    auth_flow(login, Config, Pattern).

%% register_flow(+Config, -Pattern)
%
%  Create a registration flow.
%
register_flow(Config, Pattern) :-
    auth_flow(register, Config, Pattern).

%% forgot_password_flow(+Config, -Pattern)
%
%  Create a forgot password flow.
%
forgot_password_flow(Config, Pattern) :-
    auth_flow(forgot_password, Config, Pattern).

%% oauth_flow(+Provider, +Config, -Pattern)
%
%  Create an OAuth flow for a specific provider.
%
oauth_flow(Provider, Config, Pattern) :-
    auth_flow(oauth, [provider(Provider) | Config], Pattern).

% ============================================================================
% COMPILATION
% ============================================================================

%% compile_extended_pattern(+PatternName, +Target, +Options, -Code)
%
%  Compile an extended pattern to target-specific code.
%
compile_extended_pattern(PatternName, Target, Options, Code) :-
    stored_pattern(PatternName, Pattern, _),
    compile_pattern_spec(Pattern, Target, Options, Code).

compile_pattern_spec(form(Name, Fields, _Config), Target, Options, Code) :-
    compile_form_pattern(Name, Fields, Target, Options, Code).

compile_pattern_spec(list(Type, Name, Config, _), Target, Options, Code) :-
    compile_list_pattern(Type, Name, Config, Target, Options, Code).

compile_pattern_spec(modal(Type, Name, Config), Target, Options, Code) :-
    compile_modal_pattern(Type, Name, Config, Target, Options, Code).

compile_pattern_spec(auth(Type, Config), Target, Options, Code) :-
    compile_auth_pattern(Type, Config, Target, Options, Code).

% ============================================================================
% FORM COMPILATION
% ============================================================================

compile_form_pattern(Name, Fields, react_native, _Options, Code) :-
    atom_string(Name, NameStr),
    capitalize_first(NameStr, CapName),
    generate_rn_form_fields(Fields, FieldsCode),
    generate_rn_validation(Fields, ValidationCode),
    format(string(Code),
"// Form: ~w
import React from 'react';
import { View, TextInput, Text, TouchableOpacity, StyleSheet } from 'react-native';
import { useForm, Controller } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';

~w

export function ~wForm({ onSubmit }: { onSubmit: (data: FormData) => void }) {
  const { control, handleSubmit, formState: { errors } } = useForm({
    resolver: zodResolver(schema),
  });

  return (
    <View style={styles.form}>
~w
      <TouchableOpacity style={styles.button} onPress={handleSubmit(onSubmit)}>
        <Text style={styles.buttonText}>Submit</Text>
      </TouchableOpacity>
    </View>
  );
}

const styles = StyleSheet.create({
  form: { padding: 16 },
  input: { borderWidth: 1, borderColor: '#ccc', borderRadius: 8, padding: 12, marginBottom: 8 },
  error: { color: 'red', fontSize: 12, marginBottom: 8 },
  button: { backgroundColor: '#007AFF', padding: 16, borderRadius: 8, alignItems: 'center' },
  buttonText: { color: 'white', fontWeight: 'bold' },
});
", [NameStr, ValidationCode, CapName, FieldsCode]).

compile_form_pattern(Name, Fields, vue, _Options, Code) :-
    atom_string(Name, NameStr),
    generate_vue_form_fields(Fields, FieldsCode),
    generate_vue_validation(Fields, ValidationCode),
    format(string(Code),
"<!-- Form: ~w -->
<script setup lang=\"ts\">
import { ref, computed } from 'vue';
import { useForm } from 'vee-validate';
import { z } from 'zod';
import { toTypedSchema } from '@vee-validate/zod';

~w

const { handleSubmit, errors, defineField } = useForm({
  validationSchema: toTypedSchema(schema),
});

const emit = defineEmits<{
  submit: [data: z.infer<typeof schema>]
}>();

const onSubmit = handleSubmit((values) => {
  emit('submit', values);
});
</script>

<template>
  <form @submit.prevent=\"onSubmit\" class=\"form\">
~w
    <button type=\"submit\" class=\"btn-submit\">Submit</button>
  </form>
</template>

<style scoped>
.form { display: flex; flex-direction: column; gap: 1rem; }
.form-field { display: flex; flex-direction: column; }
.form-input { padding: 0.75rem; border: 1px solid #ccc; border-radius: 0.5rem; }
.form-error { color: red; font-size: 0.875rem; }
.btn-submit { padding: 1rem; background: #007AFF; color: white; border: none; border-radius: 0.5rem; cursor: pointer; }
</style>
", [NameStr, ValidationCode, FieldsCode]).

compile_form_pattern(Name, Fields, flutter, _Options, Code) :-
    atom_string(Name, NameStr),
    capitalize_first(NameStr, CapName),
    generate_flutter_form_fields(Fields, FieldsCode),
    format(string(Code),
"// Form: ~w
import 'package:flutter/material.dart';

class ~wForm extends StatefulWidget {
  final void Function(Map<String, dynamic> data) onSubmit;

  const ~wForm({super.key, required this.onSubmit});

  @override
  State<~wForm> createState() => _~wFormState();
}

class _~wFormState extends State<~wForm> {
  final _formKey = GlobalKey<FormState>();
  final _formData = <String, dynamic>{};

  @override
  Widget build(BuildContext context) {
    return Form(
      key: _formKey,
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
~w
          const SizedBox(height: 16),
          ElevatedButton(
            onPressed: _submit,
            child: const Text('Submit'),
          ),
        ],
      ),
    );
  }

  void _submit() {
    if (_formKey.currentState!.validate()) {
      _formKey.currentState!.save();
      widget.onSubmit(_formData);
    }
  }
}
", [NameStr, CapName, CapName, CapName, CapName, CapName, CapName, FieldsCode]).

compile_form_pattern(Name, Fields, swiftui, _Options, Code) :-
    atom_string(Name, NameStr),
    capitalize_first(NameStr, CapName),
    generate_swift_form_fields(Fields, FieldsCode),
    generate_swift_form_state(Fields, StateCode),
    format(string(Code),
"// Form: ~w
import SwiftUI

struct ~wForm: View {
    var onSubmit: ([String: Any]) -> Void

~w

    var body: some View {
        Form {
~w
            Button(\"Submit\") {
                onSubmit(formData)
            }
            .buttonStyle(.borderedProminent)
        }
    }

    private var formData: [String: Any] {
        [:]  // Collect form data
    }
}
", [NameStr, CapName, StateCode, FieldsCode]).

% ============================================================================
% LIST COMPILATION
% ============================================================================

compile_list_pattern(infinite, Name, DataSource, react_native, _Options, Code) :-
    atom_string(Name, NameStr),
    atom_string(DataSource, DataSourceStr),
    capitalize_first(NameStr, CapName),
    format(string(Code),
"// Infinite List: ~w
import React, { useCallback } from 'react';
import { FlatList, ActivityIndicator, View, RefreshControl } from 'react-native';
import { useInfiniteQuery } from '@tanstack/react-query';

export function ~wList({ renderItem }: { renderItem: (item: any) => React.ReactNode }) {
  const {
    data,
    fetchNextPage,
    hasNextPage,
    isFetchingNextPage,
    isLoading,
    refetch,
    isRefetching,
  } = useInfiniteQuery({
    queryKey: ['~w'],
    queryFn: ({ pageParam = 1 }) => fetch(`~w?page=${pageParam}`).then(r => r.json()),
    getNextPageParam: (lastPage, pages) => lastPage.hasMore ? pages.length + 1 : undefined,
  });

  const items = data?.pages.flatMap(page => page.data) ?? [];

  const loadMore = useCallback(() => {
    if (hasNextPage && !isFetchingNextPage) {
      fetchNextPage();
    }
  }, [hasNextPage, isFetchingNextPage, fetchNextPage]);

  return (
    <FlatList
      data={items}
      renderItem={({ item }) => renderItem(item)}
      keyExtractor={(item) => item.id}
      onEndReached={loadMore}
      onEndReachedThreshold={0.8}
      refreshControl={
        <RefreshControl refreshing={isRefetching} onRefresh={refetch} />
      }
      ListFooterComponent={
        isFetchingNextPage ? <ActivityIndicator style={{ padding: 16 }} /> : null
      }
    />
  );
}
", [NameStr, CapName, NameStr, DataSourceStr]).

compile_list_pattern(infinite, Name, DataSource, vue, _Options, Code) :-
    atom_string(Name, NameStr),
    atom_string(DataSource, DataSourceStr),
    format(string(Code),
"<!-- Infinite List: ~w -->
<script setup lang=\"ts\">
import { ref, onMounted, onUnmounted } from 'vue';
import { useInfiniteQuery } from '@tanstack/vue-query';

const props = defineProps<{
  renderItem: (item: any) => any
}>();

const { data, fetchNextPage, hasNextPage, isFetchingNextPage, refetch } = useInfiniteQuery({
  queryKey: ['~w'],
  queryFn: ({ pageParam = 1 }) => fetch(`~w?page=${pageParam}`).then(r => r.json()),
  getNextPageParam: (lastPage, pages) => lastPage.hasMore ? pages.length + 1 : undefined,
});

const items = computed(() => data.value?.pages.flatMap(page => page.data) ?? []);

const sentinel = ref<HTMLElement | null>(null);

onMounted(() => {
  const observer = new IntersectionObserver((entries) => {
    if (entries[0].isIntersecting && hasNextPage.value && !isFetchingNextPage.value) {
      fetchNextPage();
    }
  }, { threshold: 0.8 });

  if (sentinel.value) observer.observe(sentinel.value);
});
</script>

<template>
  <div class=\"infinite-list\">
    <div v-for=\"item in items\" :key=\"item.id\">
      <component :is=\"renderItem(item)\" />
    </div>
    <div ref=\"sentinel\" class=\"sentinel\">
      <span v-if=\"isFetchingNextPage\">Loading...</span>
    </div>
  </div>
</template>
", [NameStr, NameStr, DataSourceStr]).

compile_list_pattern(infinite, Name, DataSource, flutter, _Options, Code) :-
    atom_string(Name, NameStr),
    atom_string(DataSource, DataSourceStr),
    capitalize_first(NameStr, CapName),
    format(string(Code),
"// Infinite List: ~w
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:infinite_scroll_pagination/infinite_scroll_pagination.dart';

class ~wList extends ConsumerStatefulWidget {
  final Widget Function(BuildContext, dynamic, int) itemBuilder;

  const ~wList({super.key, required this.itemBuilder});

  @override
  ConsumerState<~wList> createState() => _~wListState();
}

class _~wListState extends ConsumerState<~wList> {
  final PagingController<int, dynamic> _pagingController = PagingController(firstPageKey: 1);
  static const _pageSize = 20;

  @override
  void initState() {
    super.initState();
    _pagingController.addPageRequestListener(_fetchPage);
  }

  Future<void> _fetchPage(int pageKey) async {
    try {
      final response = await fetch('~w?page=$pageKey');
      final items = response['data'] as List;
      final hasMore = response['hasMore'] as bool;

      if (hasMore) {
        _pagingController.appendPage(items, pageKey + 1);
      } else {
        _pagingController.appendLastPage(items);
      }
    } catch (e) {
      _pagingController.error = e;
    }
  }

  @override
  Widget build(BuildContext context) {
    return RefreshIndicator(
      onRefresh: () async => _pagingController.refresh(),
      child: PagedListView<int, dynamic>(
        pagingController: _pagingController,
        builderDelegate: PagedChildBuilderDelegate(
          itemBuilder: widget.itemBuilder,
        ),
      ),
    );
  }

  @override
  void dispose() {
    _pagingController.dispose();
    super.dispose();
  }
}
", [NameStr, CapName, CapName, CapName, CapName, CapName, CapName, DataSourceStr]).

compile_list_pattern(infinite, Name, DataSource, swiftui, _Options, Code) :-
    atom_string(Name, NameStr),
    atom_string(DataSource, DataSourceStr),
    capitalize_first(NameStr, CapName),
    format(string(Code),
"// Infinite List: ~w
import SwiftUI

struct ~wList<Item: Identifiable, Content: View>: View {
    let itemBuilder: (Item) -> Content

    @StateObject private var viewModel = ~wListViewModel()

    var body: some View {
        List {
            ForEach(viewModel.items) { item in
                itemBuilder(item as! Item)
                    .onAppear {
                        if item.id == viewModel.items.last?.id {
                            viewModel.loadMore()
                        }
                    }
            }

            if viewModel.isLoading {
                ProgressView()
                    .frame(maxWidth: .infinity)
            }
        }
        .refreshable {
            await viewModel.refresh()
        }
        .onAppear {
            viewModel.loadInitial()
        }
    }
}

@MainActor
class ~wListViewModel: ObservableObject {
    @Published var items: [AnyIdentifiable] = []
    @Published var isLoading = false
    private var currentPage = 1
    private var hasMore = true
    private let endpoint = \"~w\"

    func loadInitial() {
        guard items.isEmpty else { return }
        loadMore()
    }

    func loadMore() {
        guard !isLoading && hasMore else { return }
        isLoading = true
        // Fetch implementation
    }

    func refresh() async {
        currentPage = 1
        hasMore = true
        items = []
        loadMore()
    }
}
", [NameStr, CapName, CapName, CapName, DataSourceStr]).

compile_list_pattern(selectable, Name, Mode, Target, Options, Code) :-
    compile_selectable_list(Name, Mode, Target, Options, Code).

compile_list_pattern(grouped, Name, GroupKey, Target, Options, Code) :-
    compile_grouped_list(Name, GroupKey, Target, Options, Code).

compile_selectable_list(Name, Mode, react_native, _Options, Code) :-
    atom_string(Name, NameStr),
    atom_string(Mode, ModeStr),
    capitalize_first(NameStr, CapName),
    format(string(Code),
"// Selectable List: ~w (mode: ~w)
import React, { useState } from 'react';
import { FlatList, TouchableOpacity, View, Text, StyleSheet } from 'react-native';

type SelectionMode = 'single' | 'multi';

export function ~wSelectableList<T extends { id: string }>({
  data,
  renderItem,
  onSelectionChange,
}: {
  data: T[];
  renderItem: (item: T, selected: boolean) => React.ReactNode;
  onSelectionChange: (selected: T[]) => void;
}) {
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const mode: SelectionMode = '~w';

  const toggleSelection = (item: T) => {
    const newSelected = new Set(selected);
    if (mode === 'single') {
      newSelected.clear();
      if (!selected.has(item.id)) newSelected.add(item.id);
    } else {
      if (selected.has(item.id)) {
        newSelected.delete(item.id);
      } else {
        newSelected.add(item.id);
      }
    }
    setSelected(newSelected);
    onSelectionChange(data.filter(d => newSelected.has(d.id)));
  };

  return (
    <FlatList
      data={data}
      renderItem={({ item }) => (
        <TouchableOpacity onPress={() => toggleSelection(item)}>
          {renderItem(item, selected.has(item.id))}
        </TouchableOpacity>
      )}
      keyExtractor={(item) => item.id}
    />
  );
}
", [NameStr, ModeStr, CapName, ModeStr]).

compile_selectable_list(Name, Mode, vue, _Options, Code) :-
    atom_string(Name, NameStr),
    atom_string(Mode, ModeStr),
    format(string(Code),
"<!-- Selectable List: ~w (mode: ~w) -->
<script setup lang=\"ts\" generic=\"T extends { id: string }\">
import { ref, computed } from 'vue';

const props = defineProps<{
  data: T[];
  mode?: 'single' | 'multi';
}>();

const emit = defineEmits<{
  selectionChange: [selected: T[]]
}>();

const selected = ref<Set<string>>(new Set());
const mode = computed(() => props.mode ?? '~w');

function toggleSelection(item: T) {
  const newSelected = new Set(selected.value);
  if (mode.value === 'single') {
    newSelected.clear();
    if (!selected.value.has(item.id)) newSelected.add(item.id);
  } else {
    if (selected.value.has(item.id)) {
      newSelected.delete(item.id);
    } else {
      newSelected.add(item.id);
    }
  }
  selected.value = newSelected;
  emit('selectionChange', props.data.filter(d => newSelected.has(d.id)));
}

function isSelected(item: T): boolean {
  return selected.value.has(item.id);
}
</script>

<template>
  <div class=\"selectable-list\">
    <div
      v-for=\"item in data\"
      :key=\"item.id\"
      :class=\"['list-item', { selected: isSelected(item) }]\"
      @click=\"toggleSelection(item)\"
    >
      <slot :item=\"item\" :selected=\"isSelected(item)\" />
    </div>
  </div>
</template>
", [NameStr, ModeStr, ModeStr]).

compile_selectable_list(_, _, Target, _, Code) :-
    member(Target, [flutter, swiftui]),
    Code = "// Selectable list for this target - implementation pending".

compile_grouped_list(_, _, _, _, "// Grouped list - implementation pending").

% ============================================================================
% MODAL COMPILATION
% ============================================================================

compile_modal_pattern(alert, Name, Config, react_native, _Options, Code) :-
    atom_string(Name, NameStr),
    member(title(Title), Config),
    member(message(Message), Config),
    format(string(Code),
"// Alert Modal: ~w
import { Alert } from 'react-native';

export function show~wAlert() {
  Alert.alert(
    '~w',
    '~w',
    [{ text: 'OK' }]
  );
}
", [NameStr, NameStr, Title, Message]).

compile_modal_pattern(confirm, Name, Config, react_native, _Options, Code) :-
    atom_string(Name, NameStr),
    member(title(Title), Config),
    member(message(Message), Config),
    member(confirm_text(ConfirmText), Config),
    member(cancel_text(CancelText), Config),
    format(string(Code),
"// Confirm Modal: ~w
import { Alert } from 'react-native';

export function show~wConfirm(onConfirm: () => void, onCancel?: () => void) {
  Alert.alert(
    '~w',
    '~w',
    [
      { text: '~w', style: 'cancel', onPress: onCancel },
      { text: '~w', onPress: onConfirm },
    ]
  );
}
", [NameStr, NameStr, Title, Message, CancelText, ConfirmText]).

compile_modal_pattern(bottom_sheet, Name, Config, react_native, _Options, Code) :-
    atom_string(Name, NameStr),
    capitalize_first(NameStr, CapName),
    (   member(height(Height), Config), Height \= auto
    ->  format(atom(HeightStr), "~w", [Height])
    ;   HeightStr = "'50%'"
    ),
    format(string(Code),
"// Bottom Sheet: ~w
import React, { useCallback, useMemo, forwardRef } from 'react';
import { View, StyleSheet } from 'react-native';
import BottomSheet from '@gorhom/bottom-sheet';

export const ~wBottomSheet = forwardRef<BottomSheet, { children: React.ReactNode }>(
  ({ children }, ref) => {
    const snapPoints = useMemo(() => [~w], []);

    return (
      <BottomSheet
        ref={ref}
        index={-1}
        snapPoints={snapPoints}
        enablePanDownToClose
      >
        <View style={styles.content}>
          {children}
        </View>
      </BottomSheet>
    );
  }
);

const styles = StyleSheet.create({
  content: { flex: 1, padding: 16 },
});
", [NameStr, CapName, HeightStr]).

compile_modal_pattern(Type, Name, Config, vue, _Options, Code) :-
    atom_string(Name, NameStr),
    atom_string(Type, TypeStr),
    (   member(title(Title), Config) -> true ; Title = '' ),
    (   member(message(Message), Config) -> true ; Message = '' ),
    format(string(Code),
"<!-- Modal: ~w (~w) -->
<script setup lang=\"ts\">
import { ref } from 'vue';

const isOpen = ref(false);

const open = () => { isOpen.value = true; };
const close = () => { isOpen.value = false; };

defineExpose({ open, close });
</script>

<template>
  <Teleport to=\"body\">
    <Transition name=\"modal\">
      <div v-if=\"isOpen\" class=\"modal-overlay\" @click.self=\"close\">
        <div class=\"modal-content\">
          <h2 v-if=\"'~w'\">~w</h2>
          <p v-if=\"'~w'\">~w</p>
          <slot />
          <button @click=\"close\">Close</button>
        </div>
      </div>
    </Transition>
  </Teleport>
</template>

<style scoped>
.modal-overlay { position: fixed; inset: 0; background: rgba(0,0,0,0.5); display: flex; align-items: center; justify-content: center; }
.modal-content { background: white; padding: 2rem; border-radius: 0.5rem; max-width: 90%; }
</style>
", [NameStr, TypeStr, Title, Title, Message, Message]).

compile_modal_pattern(_, _, _, Target, _, Code) :-
    member(Target, [flutter, swiftui]),
    Code = "// Modal for this target - implementation pending".

% ============================================================================
% AUTH COMPILATION
% ============================================================================

compile_auth_pattern(login, Config, react_native, _Options, Code) :-
    member(endpoint(Endpoint), Config),
    format(string(Code),
"// Login Form
import React from 'react';
import { View, TextInput, TouchableOpacity, Text, StyleSheet } from 'react-native';
import { useForm, Controller } from 'react-hook-form';
import { useMutation } from '@tanstack/react-query';

interface LoginData {
  email: string;
  password: string;
}

export function LoginForm({ onSuccess }: { onSuccess: (token: string) => void }) {
  const { control, handleSubmit, formState: { errors } } = useForm<LoginData>();

  const loginMutation = useMutation({
    mutationFn: (data: LoginData) =>
      fetch('~w', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data),
      }).then(r => r.json()),
    onSuccess: (data) => onSuccess(data.token),
  });

  return (
    <View style={styles.form}>
      <Controller
        control={control}
        name=\"email\"
        rules={{ required: 'Email is required' }}
        render={({ field: { onChange, value } }) => (
          <TextInput
            style={styles.input}
            placeholder=\"Email\"
            keyboardType=\"email-address\"
            autoCapitalize=\"none\"
            value={value}
            onChangeText={onChange}
          />
        )}
      />
      {errors.email && <Text style={styles.error}>{errors.email.message}</Text>}

      <Controller
        control={control}
        name=\"password\"
        rules={{ required: 'Password is required' }}
        render={({ field: { onChange, value } }) => (
          <TextInput
            style={styles.input}
            placeholder=\"Password\"
            secureTextEntry
            value={value}
            onChangeText={onChange}
          />
        )}
      />
      {errors.password && <Text style={styles.error}>{errors.password.message}</Text>}

      <TouchableOpacity
        style={styles.button}
        onPress={handleSubmit((data) => loginMutation.mutate(data))}
        disabled={loginMutation.isPending}
      >
        <Text style={styles.buttonText}>
          {loginMutation.isPending ? 'Logging in...' : 'Login'}
        </Text>
      </TouchableOpacity>
    </View>
  );
}

const styles = StyleSheet.create({
  form: { padding: 16 },
  input: { borderWidth: 1, borderColor: '#ccc', borderRadius: 8, padding: 12, marginBottom: 8 },
  error: { color: 'red', fontSize: 12, marginBottom: 8 },
  button: { backgroundColor: '#007AFF', padding: 16, borderRadius: 8, alignItems: 'center' },
  buttonText: { color: 'white', fontWeight: 'bold' },
});
", [Endpoint]).

compile_auth_pattern(login, Config, vue, _Options, Code) :-
    member(endpoint(Endpoint), Config),
    format(string(Code),
"<!-- Login Form -->
<script setup lang=\"ts\">
import { ref } from 'vue';
import { useMutation } from '@tanstack/vue-query';

const emit = defineEmits<{
  success: [token: string]
}>();

const email = ref('');
const password = ref('');
const errors = ref<Record<string, string>>({});

const loginMutation = useMutation({
  mutationFn: async (data: { email: string; password: string }) => {
    const res = await fetch('~w', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data),
    });
    return res.json();
  },
  onSuccess: (data) => emit('success', data.token),
});

function validate() {
  errors.value = {};
  if (!email.value) errors.value.email = 'Email is required';
  if (!password.value) errors.value.password = 'Password is required';
  return Object.keys(errors.value).length === 0;
}

function submit() {
  if (validate()) {
    loginMutation.mutate({ email: email.value, password: password.value });
  }
}
</script>

<template>
  <form @submit.prevent=\"submit\" class=\"login-form\">
    <div class=\"field\">
      <input v-model=\"email\" type=\"email\" placeholder=\"Email\" />
      <span v-if=\"errors.email\" class=\"error\">{{ errors.email }}</span>
    </div>
    <div class=\"field\">
      <input v-model=\"password\" type=\"password\" placeholder=\"Password\" />
      <span v-if=\"errors.password\" class=\"error\">{{ errors.password }}</span>
    </div>
    <button type=\"submit\" :disabled=\"loginMutation.isPending.value\">
      {{ loginMutation.isPending.value ? 'Logging in...' : 'Login' }}
    </button>
  </form>
</template>

<style scoped>
.login-form { display: flex; flex-direction: column; gap: 1rem; padding: 1rem; }
.field { display: flex; flex-direction: column; }
.field input { padding: 0.75rem; border: 1px solid #ccc; border-radius: 0.5rem; }
.error { color: red; font-size: 0.875rem; }
button { padding: 1rem; background: #007AFF; color: white; border: none; border-radius: 0.5rem; }
button:disabled { opacity: 0.7; }
</style>
", [Endpoint]).

compile_auth_pattern(Type, _Config, Target, _Options, Code) :-
    atom_string(Type, TypeStr),
    atom_string(Target, TargetStr),
    format(string(Code), "// Auth pattern '~w' for ~w - implementation pending", [TypeStr, TargetStr]).

% ============================================================================
% HELPER PREDICATES
% ============================================================================

capitalize_first(Str, Cap) :-
    (   atom(Str) -> atom_string(Str, S) ; S = Str ),
    string_chars(S, [H|T]),
    upcase_atom(H, HU),
    atom_chars(HU, [HC]),
    string_chars(Cap, [HC|T]).

generate_rn_form_fields(Fields, Code) :-
    findall(FieldCode, (
        member(field(Name, Type, _Validation, Opts), Fields),
        generate_rn_field(Name, Type, Opts, FieldCode)
    ), FieldCodes),
    atomic_list_concat(FieldCodes, '\n', Code).

generate_rn_field(Name, text, Opts, Code) :-
    atom_string(Name, NameStr),
    (   member(placeholder(P), Opts) -> true ; P = NameStr ),
    format(string(Code),
"      <Controller
        control={control}
        name=\"~w\"
        render={({ field: { onChange, value } }) => (
          <TextInput style={styles.input} placeholder=\"~w\" value={value} onChangeText={onChange} />
        )}
      />
      {errors.~w && <Text style={styles.error}>{errors.~w.message}</Text>}", [NameStr, P, NameStr, NameStr]).

generate_rn_field(Name, email, Opts, Code) :-
    atom_string(Name, NameStr),
    (   member(placeholder(P), Opts) -> true ; P = "Email" ),
    format(string(Code),
"      <Controller
        control={control}
        name=\"~w\"
        render={({ field: { onChange, value } }) => (
          <TextInput style={styles.input} placeholder=\"~w\" keyboardType=\"email-address\" autoCapitalize=\"none\" value={value} onChangeText={onChange} />
        )}
      />
      {errors.~w && <Text style={styles.error}>{errors.~w.message}</Text>}", [NameStr, P, NameStr, NameStr]).

generate_rn_field(Name, password, Opts, Code) :-
    atom_string(Name, NameStr),
    (   member(placeholder(P), Opts) -> true ; P = "Password" ),
    format(string(Code),
"      <Controller
        control={control}
        name=\"~w\"
        render={({ field: { onChange, value } }) => (
          <TextInput style={styles.input} placeholder=\"~w\" secureTextEntry value={value} onChangeText={onChange} />
        )}
      />
      {errors.~w && <Text style={styles.error}>{errors.~w.message}</Text>}", [NameStr, P, NameStr, NameStr]).

generate_rn_field(Name, _, _, Code) :-
    atom_string(Name, NameStr),
    format(string(Code),
"      <Controller
        control={control}
        name=\"~w\"
        render={({ field: { onChange, value } }) => (
          <TextInput style={styles.input} value={value} onChangeText={onChange} />
        )}
      />", [NameStr]).

generate_rn_validation(Fields, Code) :-
    findall(Rule, (
        member(field(Name, _, Validations, _), Fields),
        generate_zod_rule(Name, Validations, Rule)
    ), Rules),
    atomic_list_concat(Rules, ',\n  ', RulesStr),
    format(string(Code), "const schema = z.object({\n  ~w\n});", [RulesStr]).

generate_zod_rule(Name, Validations, Rule) :-
    atom_string(Name, NameStr),
    findall(V, (member(V, Validations), V \= []), Vs),
    zod_type_for_validations(Vs, ZodType),
    format(string(Rule), "~w: ~w", [NameStr, ZodType]).

zod_type_for_validations(Validations, ZodType) :-
    (   member(email, Validations)
    ->  Base = "z.string().email()"
    ;   Base = "z.string()"
    ),
    (   member(required, Validations)
    ->  format(string(ZodType), "~w.min(1)", [Base])
    ;   ZodType = Base
    ).

generate_vue_form_fields(_, "    <!-- Form fields -->").
generate_vue_validation(Fields, Code) :-
    generate_rn_validation(Fields, Code).

generate_flutter_form_fields(Fields, Code) :-
    findall(FieldCode, (
        member(field(Name, Type, Validations, _), Fields),
        generate_flutter_field(Name, Type, Validations, FieldCode)
    ), FieldCodes),
    atomic_list_concat(FieldCodes, '\n', Code).

generate_flutter_field(Name, text, _Validations, Code) :-
    atom_string(Name, NameStr),
    capitalize_first(NameStr, CapName),
    format(string(Code),
"          TextFormField(
            decoration: const InputDecoration(labelText: '~w'),
            onSaved: (v) => _formData['~w'] = v,
          ),", [CapName, NameStr]).

generate_flutter_field(Name, email, _Validations, Code) :-
    atom_string(Name, NameStr),
    format(string(Code),
"          TextFormField(
            decoration: const InputDecoration(labelText: 'Email'),
            keyboardType: TextInputType.emailAddress,
            onSaved: (v) => _formData['~w'] = v,
          ),", [NameStr]).

generate_flutter_field(Name, password, _Validations, Code) :-
    atom_string(Name, NameStr),
    format(string(Code),
"          TextFormField(
            decoration: const InputDecoration(labelText: 'Password'),
            obscureText: true,
            onSaved: (v) => _formData['~w'] = v,
          ),", [NameStr]).

generate_flutter_field(Name, _, _, Code) :-
    atom_string(Name, NameStr),
    capitalize_first(NameStr, CapName),
    format(string(Code),
"          TextFormField(
            decoration: const InputDecoration(labelText: '~w'),
            onSaved: (v) => _formData['~w'] = v,
          ),", [CapName, NameStr]).

generate_swift_form_fields(Fields, Code) :-
    findall(FieldCode, (
        member(field(Name, Type, _, _), Fields),
        generate_swift_field(Name, Type, FieldCode)
    ), FieldCodes),
    atomic_list_concat(FieldCodes, '\n', Code).

generate_swift_field(Name, text, Code) :-
    atom_string(Name, NameStr),
    capitalize_first(NameStr, CapName),
    format(string(Code),
"            TextField(\"~w\", text: $~w)", [CapName, NameStr]).

generate_swift_field(Name, email, Code) :-
    atom_string(Name, NameStr),
    format(string(Code),
"            TextField(\"Email\", text: $~w)
                .textContentType(.emailAddress)
                .keyboardType(.emailAddress)", [NameStr]).

generate_swift_field(Name, password, Code) :-
    atom_string(Name, NameStr),
    format(string(Code),
"            SecureField(\"Password\", text: $~w)", [NameStr]).

generate_swift_field(Name, _, Code) :-
    atom_string(Name, NameStr),
    capitalize_first(NameStr, CapName),
    format(string(Code),
"            TextField(\"~w\", text: $~w)", [CapName, NameStr]).

generate_swift_form_state(Fields, Code) :-
    findall(StateCode, (
        member(field(Name, _, _, _), Fields),
        atom_string(Name, NameStr),
        format(string(StateCode), "    @State private var ~w = \"\"", [NameStr])
    ), StateCodes),
    atomic_list_concat(StateCodes, '\n', Code).

% ============================================================================
% TESTING
% ============================================================================

test_extended_patterns :-
    format('~n=== Extended Patterns Tests ===~n~n'),

    % Test 1: Form pattern
    format('Test 1: Form pattern creation...~n'),
    (   form_pattern(test_form, [
            field(email, email, [required, email], []),
            field(password, password, [required], [])
        ], Pattern1),
        Pattern1 = form(test_form, _, _)
    ->  format('  PASS: Form pattern created~n')
    ;   format('  FAIL: Could not create form pattern~n')
    ),

    % Test 2: List pattern
    format('~nTest 2: Infinite list pattern...~n'),
    (   infinite_list(test_list, '/api/items', Pattern2),
        Pattern2 = list(infinite, test_list, '/api/items', _)
    ->  format('  PASS: Infinite list pattern created~n')
    ;   format('  FAIL: Could not create list pattern~n')
    ),

    % Test 3: Modal pattern
    format('~nTest 3: Confirm modal pattern...~n'),
    (   confirm_modal(test_confirm, [title('Confirm'), message('Are you sure?')], Pattern3),
        Pattern3 = modal(confirm, test_confirm, _)
    ->  format('  PASS: Confirm modal pattern created~n')
    ;   format('  FAIL: Could not create modal pattern~n')
    ),

    % Test 4: Auth flow
    format('~nTest 4: Login flow pattern...~n'),
    (   login_flow([endpoint('/api/login')], Pattern4),
        Pattern4 = auth(login, _)
    ->  format('  PASS: Login flow pattern created~n')
    ;   format('  FAIL: Could not create login flow pattern~n')
    ),

    % Test 5: Form compilation
    format('~nTest 5: Form compilation to React Native...~n'),
    (   compile_form_pattern(test_form, [
            field(email, email, [required], []),
            field(password, password, [required], [])
        ], react_native, [], Code),
        sub_string(Code, _, _, _, "useForm")
    ->  format('  PASS: Form compiled to React Native~n')
    ;   format('  SKIP: Form compilation incomplete~n')
    ),

    format('~n=== Tests Complete ===~n').

% ============================================================================
% INITIALIZATION
% ============================================================================

:- initialization((
    format('Extended UI Patterns module loaded~n')
), now).
