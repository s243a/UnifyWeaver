# Multi-Target Example: Task Manager App

Demonstrates compiling the same app specification to four different frontend frameworks.

## Overview

The Task Manager app is defined once using UnifyWeaver's UI patterns and compiles to:

| Target | Language | Navigation | State | Data Fetching | Persistence |
|--------|----------|------------|-------|---------------|-------------|
| React Native | TypeScript | React Navigation | Zustand | React Query | AsyncStorage |
| Vue 3 | TypeScript | Vue Router | Pinia | Vue Query | localStorage |
| Flutter | Dart | GoRouter | Riverpod | FutureProvider | SharedPreferences |
| SwiftUI | Swift | NavigationStack | ObservableObject | async/await | @AppStorage |

## App Features

- **Tab Navigation**: Tasks and Settings screens
- **Task Store**: Global state with actions (add, update, delete, setLoading)
- **Data Fetching**: REST API queries for tasks
- **Persistence**: User preferences storage

## Usage

### Interactive

```prolog
% Load the module
?- [task_manager_app].

% Compile to React Native
?- compile_task_manager_app(react_native, Code), writeln(Code).

% Compile to Vue 3
?- compile_task_manager_app(vue, Code), writeln(Code).

% Compile to Flutter
?- compile_task_manager_app(flutter, Code), writeln(Code).

% Compile to SwiftUI
?- compile_task_manager_app(swiftui, Code), writeln(Code).

% Compile to all targets at once
?- compile_all_targets(Results).
```

### Full Stack (Frontend + Backend)

```prolog
% Generate React Native frontend + Express backend
?- compile_full_stack(Frontend, Backend).
```

## Pattern Definitions

### Navigation
```prolog
navigation(tab, [
    screen(tasks, 'TasksScreen', [title('Tasks'), icon('list')]),
    screen(settings, 'SettingsScreen', [title('Settings'), icon('cog')])
], [])
```

### State Management
```prolog
state(global, [
    store(taskStore),
    slices([
        slice(tasks, [
            field(items, 'Task[]'),
            field(loading, boolean),
            field(error, 'string | null')
        ], [
            action(setTasks, "..."),
            action(addTask, "..."),
            action(updateTask, "..."),
            action(deleteTask, "...")
        ])
    ])
], [])
```

### Data Fetching
```prolog
data(query, [
    name(fetchTasks),
    endpoint('/api/tasks'),
    stale_time(60000),
    retry(3)
])
```

### Persistence
```prolog
persistence(local, [
    key(userPrefs),
    schema('{ theme: "light" | "dark", notifications: boolean }')
])
```

## Generated Code Samples

### React Native
```typescript
import React from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <NavigationContainer>
        <AppNavigation />
      </NavigationContainer>
    </QueryClientProvider>
  );
}
```

### Vue 3
```vue
<script setup lang="ts">
import { createApp } from 'vue';
import { createPinia } from 'pinia';
import { createRouter, createWebHistory } from 'vue-router';
// ...
</script>
```

### Flutter
```dart
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';

void main() {
  runApp(
    ProviderScope(
      child: MaterialApp.router(
        routerConfig: appRouter,
      ),
    ),
  );
}
```

### SwiftUI
```swift
import SwiftUI

@main
struct TaskManagerApp: App {
    @StateObject private var taskStore = TaskStore()

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(taskStore)
        }
    }
}
```

## Testing

```bash
# Run all 34 tests
swipl -g "run_tests" -t halt test_task_manager_app.pl

# Run inline tests
swipl -g "test_task_manager_app" -t halt task_manager_app.pl
```

## Files

| File | Description |
|------|-------------|
| `task_manager_app.pl` | App specification and compilation |
| `test_task_manager_app.pl` | 34 plunit tests |
| `README.md` | This documentation |
