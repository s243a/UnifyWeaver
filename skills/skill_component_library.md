# Skill: Component Library

Pre-built UI components that generate code for Vue, React Native, Flutter, and SwiftUI.

## When to Use

- User asks "how do I create a modal/card/toast?"
- User wants pre-built UI patterns
- User needs platform-specific component code
- User wants consistent component APIs across frameworks

## Quick Start

```prolog
:- use_module('src/unifyweaver/components/component_library').

% Create a toast notification
toast('Item saved!', [type(success), duration(3000)], Spec),
generate_component(Spec, react_native, Code).
```

## Component Categories

### Modal/Dialog Components

#### modal/3

```prolog
modal(Type, Options, Spec).
% Types: alert, confirm, custom
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `title(T)` | string | `''` | Modal title |
| `message(M)` | string | `''` | Modal message |
| `onClose(F)` | callback | `null` | Close handler |
| `dismissable(B)` | boolean | `true` | Can dismiss by tapping outside |

```prolog
modal(confirm, [
    title('Delete Item?'),
    message('This cannot be undone.'),
    dismissable(false)
], Spec).
```

#### alert_dialog/3

```prolog
alert_dialog(Title, Options, Spec).
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `message(M)` | string | `''` | Alert message |
| `confirmText(T)` | string | `'OK'` | Button text |
| `onConfirm(F)` | callback | `null` | Confirm handler |

#### bottom_sheet/3

```prolog
bottom_sheet(Content, Options, Spec).
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `height(H)` | size | `auto` | Sheet height |
| `dismissable(B)` | boolean | `true` | Can swipe to dismiss |
| `snapPoints(L)` | list | `[]` | Snap positions |

#### action_sheet/3

```prolog
action_sheet(Actions, Options, Spec).
% Actions: list of action atoms or action(name, handler) terms
```

### Feedback Components

#### toast/3

```prolog
toast(Message, Options, Spec).
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `type(T)` | atom | `info` | `info`, `success`, `warning`, `error` |
| `duration(D)` | integer | `3000` | Display time (ms) |
| `position(P)` | atom | `bottom` | `top`, `bottom`, `center` |
| `action(A)` | term | `null` | Action button spec |

```prolog
toast('Changes saved', [type(success), duration(2000)], Spec).
toast('Network error', [type(error), position(top)], Spec).
```

#### snackbar/3

```prolog
snackbar(Message, Options, Spec).
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `action(A)` | callback | `null` | Action handler |
| `actionText(T)` | string | `''` | Action button text |
| `duration(D)` | integer | `4000` | Display time (ms) |

#### banner/3

```prolog
banner(Message, Options, Spec).
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `type(T)` | atom | `info` | Banner type |
| `dismissable(B)` | boolean | `true` | Can dismiss |
| `icon(I)` | atom | `null` | Icon name |
| `actions(L)` | list | `[]` | Action buttons |

### Content Components

#### card/3

```prolog
card(Content, Options, Spec).
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `title(T)` | string | `null` | Card title |
| `subtitle(S)` | string | `null` | Card subtitle |
| `image(I)` | url | `null` | Header image |
| `footer(F)` | term | `null` | Footer content |
| `elevated(E)` | boolean | `true` | Show shadow |
| `onPress(H)` | callback | `null` | Tap handler |

```prolog
card(body_content, [
    title('Article Title'),
    subtitle('By Author'),
    image('https://example.com/image.jpg'),
    elevated(true)
], Spec).
```

#### list_item/3

```prolog
list_item(Content, Options, Spec).
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `leading(L)` | term | `null` | Leading widget (icon/avatar) |
| `trailing(T)` | term | `null` | Trailing widget |
| `subtitle(S)` | string | `null` | Secondary text |
| `onPress(H)` | callback | `null` | Tap handler |
| `divider(D)` | boolean | `true` | Show divider |

#### avatar/3

```prolog
avatar(Source, Options, Spec).
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `size(S)` | atom/int | `medium` | `small`, `medium`, `large`, or px |
| `fallback(F)` | term | `null` | Fallback content |
| `badge(B)` | term | `null` | Badge overlay |
| `shape(S)` | atom | `circle` | `circle`, `square`, `rounded` |

**Sizes:**
- `small`: 32px
- `medium`: 48px
- `large`: 64px

#### badge/3

```prolog
badge(Content, Options, Spec).
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `variant(V)` | atom | `default` | Badge style |
| `color(C)` | atom | `primary` | Color name |
| `size(S)` | atom | `medium` | Badge size |
| `dot(D)` | boolean | `false` | Dot-only mode |

#### chip/3

```prolog
chip(Label, Options, Spec).
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `variant(V)` | atom | `filled` | `filled`, `outlined` |
| `color(C)` | atom | `default` | Chip color |
| `icon(I)` | atom | `null` | Leading icon |
| `onDelete(D)` | callback | `null` | Delete handler |
| `selected(S)` | boolean | `false` | Selected state |

### Layout Components

#### divider/2

```prolog
divider(Options, Spec).
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `orientation(O)` | atom | `horizontal` | `horizontal`, `vertical` |
| `thickness(T)` | number | `1` | Line thickness |
| `color(C)` | color | `null` | Line color |
| `inset(I)` | boolean | `false` | Inset from edges |

#### spacer/2

```prolog
spacer(Options, Spec).
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `size(S)` | atom | `medium` | Spacing amount |
| `flex(F)` | boolean | `false` | Flexible spacer |

#### skeleton/3

```prolog
skeleton(Type, Options, Spec).
% Types: text, circle, rect
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `width(W)` | size | `'100%'` | Skeleton width |
| `height(H)` | size | `auto` | Skeleton height |
| `animated(A)` | boolean | `true` | Pulse animation |
| `borderRadius(R)` | number | `4` | Corner radius |

### Progress Components

#### progress_bar/3

```prolog
progress_bar(Value, Options, Spec).
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `max(M)` | number | `100` | Maximum value |
| `color(C)` | atom | `primary` | Bar color |
| `showLabel(L)` | boolean | `false` | Show percentage |
| `animated(A)` | boolean | `true` | Animate changes |

#### progress_circle/3

```prolog
progress_circle(Value, Options, Spec).
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `max(M)` | number | `100` | Maximum value |
| `size(S)` | number | `48` | Circle diameter |
| `strokeWidth(W)` | number | `4` | Ring thickness |
| `color(C)` | atom | `primary` | Ring color |
| `showValue(V)` | boolean | `false` | Show value in center |

#### spinner/2

```prolog
spinner(Options, Spec).
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `size(S)` | atom | `medium` | `small`, `medium`, `large` |
| `color(C)` | atom | `primary` | Spinner color |

### Input Components

#### search_bar/2

```prolog
search_bar(Options, Spec).
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `placeholder(P)` | string | `'Search...'` | Placeholder text |
| `onSearch(S)` | callback | `null` | Search handler |
| `onClear(C)` | callback | `null` | Clear handler |
| `showCancel(S)` | boolean | `false` | Show cancel button |
| `autoFocus(A)` | boolean | `false` | Auto-focus on mount |

#### rating/3

```prolog
rating(Value, Options, Spec).
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `max(M)` | number | `5` | Maximum stars |
| `allowHalf(H)` | boolean | `false` | Allow half stars |
| `readOnly(R)` | boolean | `false` | Read-only mode |
| `size(S)` | atom | `medium` | Star size |
| `onChange(C)` | callback | `null` | Change handler |

#### stepper/3

```prolog
stepper(Value, Options, Spec).
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `min(M)` | number | `0` | Minimum value |
| `max(X)` | number | `99` | Maximum value |
| `step(S)` | number | `1` | Step increment |
| `onChange(C)` | callback | `null` | Change handler |

#### slider_input/3

```prolog
slider_input(Value, Options, Spec).
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `min(M)` | number | `0` | Minimum value |
| `max(X)` | number | `100` | Maximum value |
| `step(S)` | number | `1` | Step increment |
| `showValue(V)` | boolean | `false` | Show current value |
| `onChange(C)` | callback | `null` | Change handler |

## Code Generation

### Generate for Specific Target

```prolog
generate_component(Spec, Target, Code).
% Targets: react_native, vue, flutter, swiftui
```

### Target-Specific Generators

```prolog
generate_react_native_component(Spec, Code).
generate_vue_component(Spec, Code).
generate_flutter_component(Spec, Code).
generate_swiftui_component(Spec, Code).
```

### Example Output

**React Native (toast):**
```jsx
<Toast
  message="Item saved!"
  type="success"
  visible={visible}
  onHide={onHide}
/>
```

**Vue (card):**
```vue
<template>
  <div class="card" :class="{ 'card--elevated': elevated }">
    <img v-if="image" :src="image" class="card__image" />
    <div class="card__content">
      <h3 v-if="title" class="card__title">Card Title</h3>
      <slot />
    </div>
  </div>
</template>
```

**Flutter (avatar):**
```dart
CircleAvatar(
  radius: 24,
  backgroundImage: NetworkImage('https://example.com/avatar.jpg'),
)
```

**SwiftUI (badge):**
```swift
Text("5")
    .font(.caption2)
    .padding(.horizontal, 8)
    .padding(.vertical, 4)
    .background(Color.primary)
    .foregroundColor(.white)
    .clipShape(Capsule())
```

## Component Registry

Register and retrieve custom components:

```prolog
% Register a custom component
register_component(my_button, button_spec([
    variant(primary),
    size(medium)
])).

% Get registered component
get_component(my_button, Spec).

% List all registered components
list_components(Names).
```

## Related

**Parent Skill:**
- `skill_gui_design.md` - GUI design sub-master

**Sibling Skills:**
- `skill_layout_system.md` - Layout primitives
- `skill_responsive_design.md` - Breakpoints
- `skill_theming.md` - Colors, typography

**Code:**
- `src/unifyweaver/components/component_library.pl` - Component definitions
- `src/unifyweaver/core/component_registry.pl` - Registry system
