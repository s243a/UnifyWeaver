# React Native Examples

Generated React Native components for Pearltrees visualization.

## Components

### PearltreesMindMap.tsx

Interactive mind map visualization component using:
- `react-native-svg` for vector graphics
- `react-native-gesture-handler` for pan/pinch gestures
- `react-native-reanimated` for smooth animations

Features:
- Pan and pinch to zoom
- Double-tap to reset view
- Light and dark themes
- Node press events
- Auto-layout with radial positioning

## Installation

### Expo Project

```bash
expo install react-native-svg react-native-gesture-handler react-native-reanimated
```

### Bare React Native

```bash
npm install react-native-svg react-native-gesture-handler react-native-reanimated
```

Then follow the platform-specific setup guides:
- [react-native-gesture-handler setup](https://docs.swmansion.com/react-native-gesture-handler/docs/installation)
- [react-native-reanimated setup](https://docs.swmansion.com/react-native-reanimated/docs/fundamentals/getting-started)

## Usage

```tsx
import { PearltreesMindMap } from './PearltreesMindMap';

// With default data
<PearltreesMindMap />

// With custom data and theme
<PearltreesMindMap
  nodes={myNodes}
  edges={myEdges}
  width={Dimensions.get('window').width}
  height={400}
  theme="dark"
  onNodePress={(node) => console.log('Pressed:', node)}
/>
```

## Generating Components from Prolog

```prolog
?- use_module('targets/react_native_target').
?- generate_rn_mindmap_component(Nodes, Edges, [theme(dark)], Code).
```

## Node Types

| Type | Description | Visual Style |
|------|-------------|--------------|
| root | Central topic | Larger ellipse, blue |
| hub | Category node | Medium ellipse, green |
| branch | Sub-category | Medium ellipse, orange |
| leaf | End node | Smaller ellipse, red |

## Customization

Pass options to `generate_rn_mindmap_component/4`:
- `component_name(Name)` - Custom component name
- `width(N)` - Default width
- `height(N)` - Default height
- `theme(light|dark)` - Color scheme
