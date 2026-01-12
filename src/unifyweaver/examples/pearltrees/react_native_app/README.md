# PT_Explorer - React Native Example App

A complete React Native app generated from composable UI patterns. This example
can be used to explore data exported from Pearltrees, or similar hierarchical
tree-structured data from other sources.

Demonstrates:
- Tab navigation
- React Query data fetching
- Zustand global state
- AsyncStorage persistence
- Mindmap visualization

## App Structure

```
src/
  navigation/
    AppNavigator.tsx      - Tab navigation (Home, Search, Favorites, Profile)
  screens/
    HomeScreen.tsx        - Tree list view
    SearchScreen.tsx      - Search trees
    FavoritesScreen.tsx   - Saved favorites
    ProfileScreen.tsx     - User settings
    TreeDetailScreen.tsx  - Mindmap visualization
  hooks/
    useTrees.ts           - Fetch trees (useQuery)
    useTreeDetail.ts      - Fetch tree detail (useQuery)
    useSearch.ts          - Search query (useQuery)
    useFavorites.ts       - Toggle favorite (useMutation)
  store/
    useAppStore.ts        - Zustand global state (theme, filters)
  storage/
    useUserPrefs.ts       - User preferences (AsyncStorage)
    useFavoritesCache.ts  - Offline favorites cache
  components/
    TreeCard.tsx          - Tree list item
    MindMapView.tsx       - Mindmap visualization
  api/
    client.ts             - Generated API client
```

## Patterns Used

| Pattern Type | Pattern Name | Generated Component |
|--------------|--------------|---------------------|
| Navigation | `app_navigation` | Tab navigator with 4 screens |
| Query | `fetch_trees` | useQuery hook for `/api/trees` |
| Query | `fetch_tree_detail` | useQuery hook for `/api/trees/:id` |
| Query | `search_trees` | useQuery hook for `/api/search` |
| Mutation | `toggle_favorite` | useMutation hook for `/api/favorites` |
| Infinite | `load_tree_feed` | useInfiniteQuery for paginated feed |
| Global State | `appStore` | Zustand store (theme, filters) |
| Persistence | `user_prefs` | AsyncStorage hook for preferences |
| Persistence | `favorites_cache` | AsyncStorage hook for offline cache |

## Usage

### Generate All Code

```prolog
?- use_module('generate_app').
?- generate_all_app_code.
```

### Generate Individual Components

```prolog
%% Generate navigation
?- generate_app_component(navigation, Code).

%% Generate data hook
?- generate_app_component(fetch_trees, Code).

%% Generate store
?- generate_app_component(store, Code).

%% Generate persistence hook
?- generate_app_component(user_prefs, Code).
```

### Generate Backend API

```prolog
?- generate_backend_api(ExpressCode).
```

### View App Structure

```prolog
?- show_app_structure.
```

## Generated Code Examples

### Navigation (Tab Navigator)

```tsx
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';

const Tab = createBottomTabNavigator();

export const AppNavigator: React.FC = () => (
  <NavigationContainer>
    <Tab.Navigator>
      <Tab.Screen name="home" component={HomeScreen} />
      <Tab.Screen name="search" component={SearchScreen} />
      <Tab.Screen name="favorites" component={FavoritesScreen} />
      <Tab.Screen name="profile" component={ProfileScreen} />
    </Tab.Navigator>
  </NavigationContainer>
);
```

### Query Hook (React Query)

```tsx
import { useQuery } from '@tanstack/react-query';

export const useFetch_trees = () => {
  return useQuery({
    queryKey: ['fetch_trees'],
    queryFn: async () => {
      const response = await fetch('/api/trees');
      return response.json();
    },
    staleTime: 300000,
  });
};
```

### Global State (Zustand)

```tsx
import { create } from 'zustand';

interface AppStoreState {
  theme: 'light' | 'dark';
  toggleTheme: () => void;
  sortBy: 'date' | 'name' | 'size';
  setSortBy: (sortBy: string) => void;
}

export const useAppStore = create<AppStoreState>((set) => ({
  theme: 'light',
  toggleTheme: () => set((state) => ({
    theme: state.theme === 'light' ? 'dark' : 'light'
  })),
  sortBy: 'date',
  setSortBy: (sortBy) => set({ sortBy }),
}));
```

### Backend (Express)

```tsx
import express from 'express';

const pearltreesRouter = express.Router();

router.get('/api/trees', async (req, res) => {
  const data = await fetchData(req.query);
  res.json({ success: true, data });
});

router.post('/api/favorites', async (req, res) => {
  const result = await mutateData(req.body);
  res.json({ success: true, data: result });
});

export default pearltreesRouter;
```

## Dependencies

### Frontend (React Native)

```json
{
  "dependencies": {
    "react": "^18.2.0",
    "react-native": "^0.73.0",
    "@react-navigation/native": "^6.1.0",
    "@react-navigation/bottom-tabs": "^6.5.0",
    "@tanstack/react-query": "^5.0.0",
    "zustand": "^4.4.0",
    "@react-native-async-storage/async-storage": "^1.21.0",
    "react-native-svg": "^14.0.0",
    "react-native-gesture-handler": "^2.14.0",
    "react-native-reanimated": "^3.6.0"
  }
}
```

### Backend (Express)

```json
{
  "dependencies": {
    "express": "^4.18.0",
    "cors": "^2.8.5"
  }
}
```

## Testing

```bash
# Run generator tests
swipl -g "generate_app:test_generate_app" -t halt generate_app.pl
```

## Integration with Glue

The app uses `pattern_glue.pl` to generate both frontend and backend:

```prolog
%% Generate full stack (frontend + backend)
?- generate_full_stack([fetch_trees, toggle_favorite],
                       [frontend_target(react_native), backend_target(express)],
                       FrontendCode, BackendCode).
```

## Extending the App

Add new patterns to `define_app_patterns/0`:

```prolog
%% Add a new query
query_pattern(fetch_comments, '/api/comments', [], _),

%% Add a new mutation
mutation_pattern(add_comment, '/api/comments', [method('POST')], _),

%% Generate the new hook
generate_app_component(fetch_comments, Code).
```
