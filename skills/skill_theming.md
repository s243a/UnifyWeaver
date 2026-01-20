# Skill: Theming

Design tokens (colors, typography, spacing) and theme variants (light/dark mode) that compile across platforms.

## When to Use

- User asks "how do I define app colors/fonts?"
- User wants dark mode support
- User needs consistent styling across platforms
- User asks about design tokens or CSS variables

## Quick Start

```prolog
:- use_module('src/unifyweaver/theming/theming').

% Define a theme
define_theme(my_app, [
    colors([primary-'#007AFF', background-'#FFFFFF']),
    typography([family-'Inter', sizeBase-16]),
    spacing([sm-8, md-16, lg-24])
]).

% Generate for Vue (CSS variables)
generate_theme_code(my_app, vue, CSS).
```

## Theme Definition

### Define Theme

```prolog
define_theme(Name, Components).
```

**Components:**
- `colors(List)` - Color palette
- `typography(List)` - Font settings
- `spacing(List)` - Spacing scale
- `borders(List)` - Border styles
- `shadows(List)` - Shadow definitions

### Example Theme

```prolog
define_theme(modern_app, [
    colors([
        % Brand colors
        primary-'#007AFF',
        primaryLight-'#4DA2FF',
        primaryDark-'#0055CC',

        % Semantic colors
        success-'#34C759',
        warning-'#FF9500',
        error-'#FF3B30',
        info-'#5856D6',

        % Neutral colors
        background-'#FFFFFF',
        surface-'#F2F2F7',
        text-'#000000',
        textSecondary-'#8E8E93',
        border-'#C6C6C8'
    ]),
    typography([
        family-'Inter',
        familyMono-'SF Mono',
        sizeXs-12,
        sizeSm-14,
        sizeBase-16,
        sizeLg-18,
        sizeXl-20,
        size2xl-24,
        size3xl-30,
        weightNormal-400,
        weightMedium-500,
        weightBold-700,
        lineHeightTight-1.25,
        lineHeightNormal-1.5,
        lineHeightRelaxed-1.75
    ]),
    spacing([
        xs-4,
        sm-8,
        md-16,
        lg-24,
        xl-32,
        xxl-48
    ]),
    borders([
        radiusSm-4,
        radiusMd-8,
        radiusLg-16,
        radiusFull-9999,
        widthThin-1,
        widthMedium-2
    ]),
    shadows([
        sm-'0 1px 2px rgba(0,0,0,0.05)',
        md-'0 4px 6px rgba(0,0,0,0.1)',
        lg-'0 10px 15px rgba(0,0,0,0.1)',
        xl-'0 20px 25px rgba(0,0,0,0.15)'
    ])
]).
```

## Theme Variants

### Define Dark Mode

```prolog
define_variant(ThemeName, dark, Overrides).
```

```prolog
define_variant(modern_app, dark, [
    colors([
        background-'#1C1C1E',
        surface-'#2C2C2E',
        text-'#FFFFFF',
        textSecondary-'#8E8E93',
        border-'#38383A'
    ])
]).
```

### Get Merged Variant

```prolog
get_variant(modern_app, dark, DarkTheme).
% Returns base theme with dark overrides applied
```

### Apply Variant

```prolog
apply_variant(BaseTheme, dark, Result).
% Merges variant overrides into theme
```

## Token Resolution

### Resolve Single Token

```prolog
resolve_token(token(Category, Name), Theme, Value).
```

```prolog
?- get_theme(modern_app, Theme),
   resolve_token(token(colors, primary), Theme, Value).
Value = '#007AFF'
```

### Resolve All Tokens in Spec

```prolog
resolve_all_tokens(Spec, Theme, Resolved).
```

Recursively resolves all `token(cat, name)` terms in a specification.

## Code Generation

### Generate Theme Code

```prolog
generate_theme_code(ThemeName, Target, Code).
% Targets: react_native, vue, flutter, swiftui
```

### React Native Output

JavaScript object with all tokens:

```typescript
export const theme = {
  colors: {
    primary: '#007AFF',
    background: '#FFFFFF',
    // ...
  },
  typography: {
    family: 'Inter',
    sizeBase: 16,
    // ...
  },
  spacing: {
    sm: 8,
    md: 16,
    // ...
  },
};
```

### Vue Output

CSS custom properties:

```css
:root {
  --color-primary: #007AFF;
  --color-background: #FFFFFF;
  --font-family: Inter;
  --font-sizeBase: 16px;
  --spacing-sm: 8px;
  --spacing-md: 16px;
}
```

### Flutter Output

Dart class with static constants:

```dart
class AppTheme {
  static const Color primary = Color(0xFF007AFF);
  static const Color background = Color(0xFFFFFFFF);
  static const String fontFamily = 'Inter';
  static const double fontSizeBase = 16;
}
```

### SwiftUI Output

Swift struct with nested types:

```swift
import SwiftUI

struct Theme {
  struct Colors {
    static let primary = Color(hex: "#007AFF")
    static let background = Color(hex: "#FFFFFF")
  }

  struct Fonts {
    static let family = "Inter"
    static let sizeBase: CGFloat = 16
  }
}
```

## Theme Management

### List All Themes

```prolog
list_themes(Names).
```

### Get Theme Definition

```prolog
get_theme(Name, Definition).
```

### Clear All Themes

```prolog
clear_themes.
```

## Color Utilities

### Named Colors

```prolog
color(primary, '#007AFF').
color(success, '#34C759').
```

### Color Variants

```prolog
color_variant(primary, light, '#4DA2FF').
color_variant(primary, dark, '#0055CC').
```

### Opacity

```prolog
opacity('#007AFF', 0.5).  % 50% opacity
```

## Typography Utilities

### Font Family

```prolog
font_family(body, 'Inter').
font_family(mono, 'SF Mono').
```

### Font Size

```prolog
font_size(xs, 12).
font_size(base, 16).
font_size(xl, 24).
```

### Font Weight

```prolog
font_weight(normal, 400).
font_weight(bold, 700).
```

### Line Height

```prolog
line_height(tight, 1.25).
line_height(normal, 1.5).
```

## Spacing Utilities

### Space Scale

```prolog
space(xs, 4).
space(sm, 8).
space(md, 16).
```

### Padding/Margin

Can be single value or list [top, right, bottom, left]:

```prolog
padding(card, 16).
padding(header, [16, 24, 16, 24]).
margin(section, 32).
```

## Common Patterns

### Theme with System Color Scheme

```prolog
define_theme(adaptive, [
    colors([
        background-'var(--system-background)',
        text-'var(--system-text)'
    ])
]).
```

### Semantic Color Palette

```prolog
define_theme(semantic, [
    colors([
        % Actions
        actionPrimary-'#007AFF',
        actionSecondary-'#5856D6',
        actionDestructive-'#FF3B30',

        % States
        stateSuccess-'#34C759',
        stateWarning-'#FF9500',
        stateError-'#FF3B30',
        stateDisabled-'#C6C6C8',

        % Surfaces
        surfaceBackground-'#FFFFFF',
        surfaceCard-'#F2F2F7',
        surfaceOverlay-'rgba(0,0,0,0.5)'
    ])
]).
```

### Typography Scale (Modular)

```prolog
% 1.25 scale factor
define_theme(modular_type, [
    typography([
        sizeXs-10,    % 16 / 1.25^2
        sizeSm-12.8,  % 16 / 1.25
        sizeBase-16,  % base
        sizeLg-20,    % 16 * 1.25
        sizeXl-25,    % 16 * 1.25^2
        size2xl-31.25 % 16 * 1.25^3
    ])
]).
```

## Related

**Parent Skill:**
- `skill_gui_design.md` - GUI design sub-master

**Sibling Skills:**
- `skill_component_library.md` - Pre-built components
- `skill_layout_system.md` - Layout primitives
- `skill_responsive_design.md` - Breakpoints

**Code:**
- `src/unifyweaver/theming/theming.pl` - Theme system
- `src/unifyweaver/glue/theme_generator.pl` - Theme generation
