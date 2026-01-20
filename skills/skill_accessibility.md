# Skill: Accessibility

Cross-platform accessibility patterns including ARIA attributes, keyboard navigation, focus management, and screen reader support.

## When to Use

- User asks "how do I add accessibility to my app?"
- User needs ARIA attributes for web components
- User wants keyboard navigation support
- User needs screen reader compatibility
- User asks about focus traps or skip links

## Quick Start

```prolog
:- use_module('src/unifyweaver/a11y/accessibility').
:- use_module('src/unifyweaver/glue/accessibility_generator').

% Define a11y attributes for a button
A11y = a11y([label('Submit'), role(button), hint('Submit form')]),
generate_a11y_attrs(A11y, vue, Code).

% Define ARIA spec for a component
aria_spec(my_chart, [
    role(img),
    label("Sales chart visualization"),
    describedby(chart_description)
]).

% Generate ARIA props
generate_aria_props(my_chart, Props).
```

## Cross-Platform A11y Patterns

The `accessibility.pl` module provides patterns that compile across targets.

### A11y Term Constructors

```prolog
a11y(Attrs).              % Accessibility attribute container
a11y_label(Text, Term).   % Create label attribute
a11y_role(Role, Term).    % Create role attribute
a11y_hint(Text, Term).    % Create hint attribute
a11y_required(Bool, Term). % Create required attribute
a11y_checked(Bool, Term).  % Create checked state
a11y_disabled(Bool, Term). % Create disabled state
a11y_hidden(Bool, Term).   % Hide from screen readers
a11y_live(Mode, Term).     % Live region: polite, assertive, off
```

### Supported Roles

```prolog
supported_roles([
    button, link, heading, image, text, checkbox, radio,
    slider, switch, tab, tablist, menu, menuitem, alert,
    dialog, textfield, searchfield, progressbar, list, listitem,
    none
]).
```

### Role Mapping Per Target

| Abstract Role | React Native | Vue | Flutter | SwiftUI |
|--------------|--------------|-----|---------|---------|
| button | button | button | button | button |
| heading | header | heading | header | header |
| image | image | img | image | image |
| slider | adjustable | slider | slider | slider |
| checkbox | checkbox | checkbox | checkbox | checkbox |
| searchfield | search | searchbox | textField | searchField |
| dialog | none | dialog | dialog | dialog |

## Code Generation by Target

### React Native

```prolog
generate_a11y_attrs(a11y([label('Submit'), role(button)]), react_native, Code).
```

**Output:**
```jsx
accessibilityLabel="Submit"
accessibilityRole="button"
```

**Additional props:**
- `accessibilityHint` - Description of action
- `accessibilityState` - `{ disabled: true, checked: true/false }`
- `accessibilityElementsHidden` - Hide from screen readers
- `accessibilityLiveRegion` - Live region mode

### Vue (ARIA)

```prolog
generate_a11y_attrs(a11y([label('Submit'), role(button)]), vue, Code).
```

**Output:**
```html
aria-label="Submit"
role="button"
```

**Additional attributes:**
- `aria-describedby` - Reference to description element
- `aria-disabled` - Disabled state
- `aria-checked` - Checked state
- `aria-hidden` - Hide from assistive tech
- `aria-required` - Required field
- `aria-live` - Live region mode

### Flutter (Semantics)

```prolog
generate_a11y_attrs(a11y([label('Submit'), role(button)]), flutter, Code).
```

**Output:**
```dart
Semantics(
  label: 'Submit',
  button: true,
  child:
```

**Flutter roles map to boolean flags:**
- `button: true`, `link: true`, `header: true`
- `image: true`, `textField: true`, `slider: true`
- `enabled: false` for disabled state
- `excludeSemantics: true` for hidden

### SwiftUI

```prolog
generate_a11y_attrs(a11y([label('Submit'), role(button)]), swiftui, Code).
```

**Output:**
```swift
.accessibilityLabel("Submit")
.accessibilityAddTraits(.isButton)
```

**SwiftUI traits:**
- `.isButton`, `.isLink`, `.isHeader`
- `.isImage`, `.isStaticText`, `.isSearchField`
- `.isTab`, `.isSelected`, `.isDisabled`

## ARIA Spec Generation

The `accessibility_generator.pl` module provides more detailed ARIA specifications.

### Define ARIA Specs

```prolog
aria_spec(Component, Attributes).
```

**Pre-defined specs:**

| Component | Role | Purpose |
|-----------|------|---------|
| line_chart | img | Chart visualization |
| bar_chart | img | Chart visualization |
| data_table | grid | Interactive table |
| control_panel | form | Form controls |
| slider_control | slider | Adjustable value |
| sidebar | navigation | Navigation area |
| main_content | main | Main content |

### ARIA Attributes

| Attribute | Description |
|-----------|-------------|
| `role(Role)` | ARIA role |
| `label(Text)` | Accessible label |
| `describedby(Id)` | Description element ID |
| `labelledby(Id)` | Label element ID |
| `controls(Id)` | Controlled element ID |
| `expanded(Bool)` | Expansion state |
| `selected(Bool)` | Selection state |
| `hidden(Bool)` | Hide from AT |
| `disabled(Bool)` | Disabled state |
| `readonly(Bool)` | Read-only state |
| `valuemin/valuemax/valuenow` | Slider values |
| `live(Mode)` | Live region mode |
| `atomic(Bool)` | Announce entire region |
| `level(N)` | Heading level |
| `posinset/setsize` | Position in set |
| `sort(Order)` | Sort direction |

### Generate Props Object

```prolog
generate_aria_props(line_chart, Props).
```

**Output:**
```javascript
{
  role: "img",
  "aria-label": "Line chart visualization",
  "aria-describedby": "chart_description"
}
```

## Keyboard Navigation

### Define Keyboard Handlers

```prolog
keyboard_nav(Component, Handlers).
```

**Pre-defined handlers:**

| Component | Keys | Actions |
|-----------|------|---------|
| data_table | Arrows, Home, End, Enter, Escape, Tab | Cell navigation |
| interactive_chart | Arrows, Enter, Escape, Space | Point selection |
| slider | Arrows, PageUp/Down, Home, End | Value adjustment |
| tablist | Arrows, Home, End, Enter, Space | Tab selection |
| modal | Escape, Tab | Close, focus trap |

### Generate Keyboard Handler

```prolog
generate_keyboard_handler(data_table, Handler).
```

**Output:**
```typescript
const handleKeyDown = (event: React.KeyboardEvent) => {
    switch (event.key) {
      case "ArrowUp":
        moveFocus("up");
        event.preventDefault();
        break;
      case "ArrowDown":
        moveFocus("down");
        event.preventDefault();
        break;
      // ... more cases
    }
  };
```

## Focus Management

### Focus Traps

For modals and popups that trap focus within a container.

```prolog
focus_trap(Component, Options).
```

**Options:**
- `container_selector(Selector)` - CSS selector for container
- `initial_focus(Selector)` - Element to focus on open
- `return_focus(Bool)` - Return focus on close
- `escape_closes(Bool)` - Escape key closes
- `outside_click_closes(Bool)` - Click outside closes

```prolog
focus_trap(modal_dialog, [
    container_selector('.modal'),
    initial_focus('.modal-close'),
    return_focus(true)
]).

generate_focus_trap_jsx(modal_dialog, JSX).
```

**Output:**
```jsx
<FocusTrap
  containerSelector=".modal"
  initialFocus=".modal-close"
  returnFocusOnDeactivate={true}
  escapeDeactivates={false}
>
  {children}
</FocusTrap>
```

### Focus Trap Hook

```prolog
generate_focus_trap_hook(Hook).
```

Generates a `useFocusTrap` React hook with:
- Previous focus restoration
- Tab key cycling between first/last focusable elements
- Escape key handling
- Configurable initial focus

## Skip Links

For keyboard users to skip navigation.

```prolog
skip_link(Name, Target).
```

**Pre-defined skip links:**

| Name | Target |
|------|--------|
| main | #main-content |
| nav | #navigation |
| chart | #chart-container |
| controls | #control-panel |

```prolog
generate_skip_links_jsx([main, nav, chart], JSX).
```

**Output:**
```jsx
<div className={styles.skipLinks}>
  <a href="#main-content" className={styles.skipLink}>Skip to Main</a>
  <a href="#navigation" className={styles.skipLink}>Skip to Nav</a>
  <a href="#chart-container" className={styles.skipLink}>Skip to Chart</a>
</div>
```

## Live Regions

For screen reader announcements.

```prolog
live_region(Name, Options).
```

**Options:**
- `aria_live(polite|assertive)` - Announcement priority
- `aria_atomic(Bool)` - Announce entire region
- `role(status|alert)` - Region role
- `aria_busy(Bool)` - Loading state

**Pre-defined regions:**

| Name | Live Mode | Role |
|------|-----------|------|
| chart_updates | polite | status |
| error_messages | assertive | alert |
| loading_status | polite | status |

```prolog
generate_live_region_jsx(error_messages, JSX).
```

**Output:**
```jsx
<div
  id="error_messages"
  role="alert"
  aria-live="assertive"
  aria-atomic={true}
  className={styles.srOnly}
>
  {errorMessagesMessage}
</div>
```

## Accessibility CSS

```prolog
generate_accessibility_css(_, CSS).
```

Generates:
- `.srOnly` - Screen reader only (visually hidden)
- `.focusVisible` - Focus ring styles
- `.focusWithin` - Parent focus indication
- `.touchTarget` - Minimum 44x44px touch targets
- `.interactive` - Cursor and selection styles
- `@media (prefers-reduced-motion)` - Animation disable
- `@media (prefers-contrast: high)` - High contrast mode

## Validation

### Validate A11y Spec

```prolog
validate_a11y(a11y(Attrs), Errors).
```

Checks for:
- Missing label
- Invalid role
- Empty label/hint

### Check Coverage

```prolog
check_a11y_coverage(Patterns, Report).
% Report = coverage(total(N), covered(M), percentage(P))
```

## Common Patterns

### Accessible Button

```prolog
Button = button(submit, [
    text('Submit Form'),
    a11y([
        label('Submit Form'),
        role(button),
        hint('Submits the current form data')
    ])
]).
```

### Accessible Chart

```prolog
aria_spec(sales_chart, [
    role(img),
    label("Monthly sales chart showing trends"),
    describedby(sales_description)
]).

keyboard_nav(sales_chart, [
    key('ArrowLeft', 'selectPreviousMonth()'),
    key('ArrowRight', 'selectNextMonth()'),
    key('Enter', 'showDetails()'),
    key('Escape', 'clearSelection()')
]).
```

### Accessible Modal

```prolog
focus_trap(confirmation_modal, [
    container_selector('#confirmation-modal'),
    initial_focus('#confirm-button'),
    return_focus(true),
    escape_closes(true)
]).

keyboard_nav(confirmation_modal, [
    key('Escape', 'closeModal()'),
    key('Enter', 'confirmAction()'),
    key('Tab', 'trapFocus()')
]).
```

### Accessible Form

```prolog
aria_spec(login_form, [
    role(form),
    label("Login form")
]).

aria_spec(username_field, [
    role(textfield),
    label("Username"),
    required(true)
]).

aria_spec(password_field, [
    role(textfield),
    label("Password"),
    required(true)
]).
```

## Related

**Parent Skill:**
- `skill_gui_tools.md` - GUI master skill

**Sibling Skills:**
- `skill_frontend_security.md` - Navigation guards, CORS

**Code:**
- `src/unifyweaver/a11y/accessibility.pl` - Cross-platform a11y
- `src/unifyweaver/glue/accessibility_generator.pl` - ARIA generation
