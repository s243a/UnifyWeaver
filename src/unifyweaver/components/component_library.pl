% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025 John William Creighton (s243a)
%
% component_library.pl - Expanded UI component patterns
%
% Pre-built component patterns that compile across all targets:
% modals, toasts, cards, avatars, badges, chips, dividers, etc.
%
% Usage:
%   use_module('src/unifyweaver/components/component_library').
%   modal(confirm, [title('Delete?'), ...], Spec),
%   generate_component(Spec, react_native, Code).

:- module(component_library, [
    % Modal/Dialog components
    modal/3,
    alert_dialog/3,
    bottom_sheet/3,
    action_sheet/3,

    % Feedback components
    toast/3,
    snackbar/3,
    banner/3,

    % Content components
    card/3,
    list_item/3,
    avatar/3,
    badge/3,
    chip/3,
    tag/3,

    % Layout components
    divider/2,
    spacer/2,
    skeleton/3,

    % Progress components
    progress_bar/3,
    progress_circle/3,
    spinner/2,

    % Input components
    search_bar/2,
    rating/3,
    stepper/3,
    slider_input/3,

    % Component generation
    generate_component/3,
    generate_react_native_component/2,
    generate_vue_component/2,
    generate_flutter_component/2,
    generate_swiftui_component/2,

    % Component registry
    register_component/2,
    get_component/2,
    list_components/1,

    % Testing
    test_component_library/0
]).

:- use_module(library(lists)).

% ============================================================================
% Dynamic Component Registry
% ============================================================================

:- dynamic registered_component/2.

%! register_component(+Name, +Spec) is det
register_component(Name, Spec) :-
    retractall(registered_component(Name, _)),
    assertz(registered_component(Name, Spec)).

%! get_component(+Name, -Spec) is semidet
get_component(Name, Spec) :-
    registered_component(Name, Spec).

%! list_components(-Names) is det
list_components(Names) :-
    findall(Name, registered_component(Name, _), Names).

% ============================================================================
% Modal/Dialog Components
% ============================================================================

%! modal(+Type, +Options, -Spec) is det
%  Create a modal dialog specification.
%  Types: alert, confirm, custom
modal(Type, Options, Spec) :-
    get_option(title, Options, Title, ''),
    get_option(message, Options, Message, ''),
    get_option(onClose, Options, OnClose, null),
    get_option(dismissable, Options, Dismissable, true),
    Spec = modal_spec(Type, [
        title(Title),
        message(Message),
        onClose(OnClose),
        dismissable(Dismissable),
        options(Options)
    ]).

%! alert_dialog(+Title, +Options, -Spec) is det
%  Create an alert dialog.
alert_dialog(Title, Options, Spec) :-
    get_option(message, Options, Message, ''),
    get_option(confirmText, Options, ConfirmText, 'OK'),
    get_option(onConfirm, Options, OnConfirm, null),
    Spec = alert_dialog_spec([
        title(Title),
        message(Message),
        confirmText(ConfirmText),
        onConfirm(OnConfirm)
    ]).

%! bottom_sheet(+Content, +Options, -Spec) is det
%  Create a bottom sheet.
bottom_sheet(Content, Options, Spec) :-
    get_option(height, Options, Height, auto),
    get_option(dismissable, Options, Dismissable, true),
    get_option(snapPoints, Options, SnapPoints, []),
    Spec = bottom_sheet_spec([
        content(Content),
        height(Height),
        dismissable(Dismissable),
        snapPoints(SnapPoints)
    ]).

%! action_sheet(+Actions, +Options, -Spec) is det
%  Create an action sheet with multiple options.
action_sheet(Actions, Options, Spec) :-
    get_option(title, Options, Title, ''),
    get_option(cancelText, Options, CancelText, 'Cancel'),
    Spec = action_sheet_spec([
        title(Title),
        actions(Actions),
        cancelText(CancelText)
    ]).

% ============================================================================
% Feedback Components
% ============================================================================

%! toast(+Message, +Options, -Spec) is det
%  Create a toast notification.
toast(Message, Options, Spec) :-
    get_option(type, Options, Type, info),
    get_option(duration, Options, Duration, 3000),
    get_option(position, Options, Position, bottom),
    get_option(action, Options, Action, null),
    Spec = toast_spec([
        message(Message),
        type(Type),
        duration(Duration),
        position(Position),
        action(Action)
    ]).

%! snackbar(+Message, +Options, -Spec) is det
%  Create a snackbar notification.
snackbar(Message, Options, Spec) :-
    get_option(action, Options, Action, null),
    get_option(actionText, Options, ActionText, ''),
    get_option(duration, Options, Duration, 4000),
    Spec = snackbar_spec([
        message(Message),
        action(Action),
        actionText(ActionText),
        duration(Duration)
    ]).

%! banner(+Message, +Options, -Spec) is det
%  Create a banner notification.
banner(Message, Options, Spec) :-
    get_option(type, Options, Type, info),
    get_option(dismissable, Options, Dismissable, true),
    get_option(icon, Options, Icon, null),
    get_option(actions, Options, Actions, []),
    Spec = banner_spec([
        message(Message),
        type(Type),
        dismissable(Dismissable),
        icon(Icon),
        actions(Actions)
    ]).

% ============================================================================
% Content Components
% ============================================================================

%! card(+Content, +Options, -Spec) is det
%  Create a card component.
card(Content, Options, Spec) :-
    get_option(title, Options, Title, null),
    get_option(subtitle, Options, Subtitle, null),
    get_option(image, Options, Image, null),
    get_option(footer, Options, Footer, null),
    get_option(elevated, Options, Elevated, true),
    get_option(onPress, Options, OnPress, null),
    Spec = card_spec([
        content(Content),
        title(Title),
        subtitle(Subtitle),
        image(Image),
        footer(Footer),
        elevated(Elevated),
        onPress(OnPress)
    ]).

%! list_item(+Content, +Options, -Spec) is det
%  Create a list item component.
list_item(Content, Options, Spec) :-
    get_option(leading, Options, Leading, null),
    get_option(trailing, Options, Trailing, null),
    get_option(subtitle, Options, Subtitle, null),
    get_option(onPress, Options, OnPress, null),
    get_option(divider, Options, Divider, true),
    Spec = list_item_spec([
        content(Content),
        leading(Leading),
        trailing(Trailing),
        subtitle(Subtitle),
        onPress(OnPress),
        divider(Divider)
    ]).

%! avatar(+Source, +Options, -Spec) is det
%  Create an avatar component.
avatar(Source, Options, Spec) :-
    get_option(size, Options, Size, medium),
    get_option(fallback, Options, Fallback, null),
    get_option(badge, Options, Badge, null),
    get_option(shape, Options, Shape, circle),
    Spec = avatar_spec([
        source(Source),
        size(Size),
        fallback(Fallback),
        badge(Badge),
        shape(Shape)
    ]).

%! badge(+Content, +Options, -Spec) is det
%  Create a badge component.
badge(Content, Options, Spec) :-
    get_option(variant, Options, Variant, default),
    get_option(color, Options, Color, primary),
    get_option(size, Options, Size, medium),
    get_option(dot, Options, Dot, false),
    Spec = badge_spec([
        content(Content),
        variant(Variant),
        color(Color),
        size(Size),
        dot(Dot)
    ]).

%! chip(+Label, +Options, -Spec) is det
%  Create a chip/tag component.
chip(Label, Options, Spec) :-
    get_option(variant, Options, Variant, filled),
    get_option(color, Options, Color, default),
    get_option(icon, Options, Icon, null),
    get_option(onDelete, Options, OnDelete, null),
    get_option(selected, Options, Selected, false),
    Spec = chip_spec([
        label(Label),
        variant(Variant),
        color(Color),
        icon(Icon),
        onDelete(OnDelete),
        selected(Selected)
    ]).

%! tag(+Label, +Options, -Spec) is det
%  Alias for chip.
tag(Label, Options, Spec) :-
    chip(Label, Options, Spec).

% ============================================================================
% Layout Components
% ============================================================================

%! divider(+Options, -Spec) is det
%  Create a divider component.
divider(Options, Spec) :-
    get_option(orientation, Options, Orientation, horizontal),
    get_option(thickness, Options, Thickness, 1),
    get_option(color, Options, Color, null),
    get_option(inset, Options, Inset, false),
    Spec = divider_spec([
        orientation(Orientation),
        thickness(Thickness),
        color(Color),
        inset(Inset)
    ]).

%! spacer(+Options, -Spec) is det
%  Create a spacer component.
spacer(Options, Spec) :-
    get_option(size, Options, Size, medium),
    get_option(flex, Options, Flex, false),
    Spec = spacer_spec([
        size(Size),
        flex(Flex)
    ]).

%! skeleton(+Type, +Options, -Spec) is det
%  Create a skeleton loading placeholder.
skeleton(Type, Options, Spec) :-
    get_option(width, Options, Width, '100%'),
    get_option(height, Options, Height, auto),
    get_option(animated, Options, Animated, true),
    get_option(borderRadius, Options, BorderRadius, 4),
    Spec = skeleton_spec([
        type(Type),
        width(Width),
        height(Height),
        animated(Animated),
        borderRadius(BorderRadius)
    ]).

% ============================================================================
% Progress Components
% ============================================================================

%! progress_bar(+Value, +Options, -Spec) is det
%  Create a progress bar.
progress_bar(Value, Options, Spec) :-
    get_option(max, Options, Max, 100),
    get_option(color, Options, Color, primary),
    get_option(showLabel, Options, ShowLabel, false),
    get_option(animated, Options, Animated, true),
    Spec = progress_bar_spec([
        value(Value),
        max(Max),
        color(Color),
        showLabel(ShowLabel),
        animated(Animated)
    ]).

%! progress_circle(+Value, +Options, -Spec) is det
%  Create a circular progress indicator.
progress_circle(Value, Options, Spec) :-
    get_option(max, Options, Max, 100),
    get_option(size, Options, Size, 48),
    get_option(strokeWidth, Options, StrokeWidth, 4),
    get_option(color, Options, Color, primary),
    get_option(showValue, Options, ShowValue, false),
    Spec = progress_circle_spec([
        value(Value),
        max(Max),
        size(Size),
        strokeWidth(StrokeWidth),
        color(Color),
        showValue(ShowValue)
    ]).

%! spinner(+Options, -Spec) is det
%  Create a loading spinner.
spinner(Options, Spec) :-
    get_option(size, Options, Size, medium),
    get_option(color, Options, Color, primary),
    Spec = spinner_spec([
        size(Size),
        color(Color)
    ]).

% ============================================================================
% Input Components
% ============================================================================

%! search_bar(+Options, -Spec) is det
%  Create a search bar.
search_bar(Options, Spec) :-
    get_option(placeholder, Options, Placeholder, 'Search...'),
    get_option(onSearch, Options, OnSearch, null),
    get_option(onClear, Options, OnClear, null),
    get_option(showCancel, Options, ShowCancel, false),
    get_option(autoFocus, Options, AutoFocus, false),
    Spec = search_bar_spec([
        placeholder(Placeholder),
        onSearch(OnSearch),
        onClear(OnClear),
        showCancel(ShowCancel),
        autoFocus(AutoFocus)
    ]).

%! rating(+Value, +Options, -Spec) is det
%  Create a rating component.
rating(Value, Options, Spec) :-
    get_option(max, Options, Max, 5),
    get_option(allowHalf, Options, AllowHalf, false),
    get_option(readOnly, Options, ReadOnly, false),
    get_option(size, Options, Size, medium),
    get_option(onChange, Options, OnChange, null),
    Spec = rating_spec([
        value(Value),
        max(Max),
        allowHalf(AllowHalf),
        readOnly(ReadOnly),
        size(Size),
        onChange(OnChange)
    ]).

%! stepper(+Value, +Options, -Spec) is det
%  Create a stepper/counter component.
stepper(Value, Options, Spec) :-
    get_option(min, Options, Min, 0),
    get_option(max, Options, Max, 99),
    get_option(step, Options, Step, 1),
    get_option(onChange, Options, OnChange, null),
    Spec = stepper_spec([
        value(Value),
        min(Min),
        max(Max),
        step(Step),
        onChange(OnChange)
    ]).

%! slider_input(+Value, +Options, -Spec) is det
%  Create a slider input component.
slider_input(Value, Options, Spec) :-
    get_option(min, Options, Min, 0),
    get_option(max, Options, Max, 100),
    get_option(step, Options, Step, 1),
    get_option(showValue, Options, ShowValue, false),
    get_option(onChange, Options, OnChange, null),
    Spec = slider_input_spec([
        value(Value),
        min(Min),
        max(Max),
        step(Step),
        showValue(ShowValue),
        onChange(OnChange)
    ]).

% ============================================================================
% Helper Predicates
% ============================================================================

get_option(Key, Options, Value, _Default) :-
    member(KeyVal, Options),
    KeyVal =.. [Key, Value],
    !.
get_option(_Key, _Options, Default, Default).

% ============================================================================
% Code Generation - Main Entry Point
% ============================================================================

%! generate_component(+Spec, +Target, -Code) is det
generate_component(Spec, Target, Code) :-
    (   Target = react_native
    ->  generate_react_native_component(Spec, Code)
    ;   Target = vue
    ->  generate_vue_component(Spec, Code)
    ;   Target = flutter
    ->  generate_flutter_component(Spec, Code)
    ;   Target = swiftui
    ->  generate_swiftui_component(Spec, Code)
    ;   Code = ""
    ).

% ============================================================================
% React Native Component Generation
% ============================================================================

%! generate_react_native_component(+Spec, -Code) is det
generate_react_native_component(modal_spec(Type, Props), Code) :-
    member(title(Title), Props),
    member(message(Message), Props),
    format(atom(Code), '<Modal visible={visible} animationType="slide">\n  <View style={styles.modalContainer}>\n    <Text style={styles.title}>~w</Text>\n    <Text style={styles.message}>~w</Text>\n    <TouchableOpacity onPress={onClose}>\n      <Text>Close</Text>\n    </TouchableOpacity>\n  </View>\n</Modal>', [Title, Message]).

generate_react_native_component(toast_spec(Props), Code) :-
    member(message(Message), Props),
    member(type(Type), Props),
    format(atom(Code), '<Toast\n  message="~w"\n  type="~w"\n  visible={visible}\n  onHide={onHide}\n/>', [Message, Type]).

generate_react_native_component(card_spec(Props), Code) :-
    member(title(Title), Props),
    member(content(Content), Props),
    format(atom(Code), '<View style={styles.card}>\n  ~w\n  <Text style={styles.cardTitle}>~w</Text>\n  <View style={styles.cardContent}>~w</View>\n</View>',
           ['{image && <Image source={image} style={styles.cardImage} />}', Title, Content]).

generate_react_native_component(avatar_spec(Props), Code) :-
    member(source(Source), Props),
    member(size(Size), Props),
    size_to_rn(Size, SizeVal),
    format(atom(Code), '<Image\n  source={{ uri: \'~w\' }}\n  style={{ width: ~w, height: ~w, borderRadius: ~w }}\n/>', [Source, SizeVal, SizeVal, SizeVal]).

generate_react_native_component(badge_spec(Props), Code) :-
    member(content(Content), Props),
    member(color(Color), Props),
    format(atom(Code), '<View style={[styles.badge, { backgroundColor: colors.~w }]}>\n  <Text style={styles.badgeText}>~w</Text>\n</View>', [Color, Content]).

generate_react_native_component(progress_bar_spec(Props), Code) :-
    member(value(Value), Props),
    member(max(Max), Props),
    format(atom(Code), '<View style={styles.progressContainer}>\n  <View style={[styles.progressBar, { width: \'~w%\' }]} />\n</View>', [Value]).

generate_react_native_component(spinner_spec(Props), Code) :-
    member(size(Size), Props),
    member(color(Color), Props),
    format(atom(Code), '<ActivityIndicator size="~w" color={colors.~w} />', [Size, Color]).

generate_react_native_component(search_bar_spec(Props), Code) :-
    member(placeholder(Placeholder), Props),
    format(atom(Code), '<TextInput\n  placeholder="~w"\n  style={styles.searchBar}\n  onChangeText={onSearch}\n  value={searchText}\n/>', [Placeholder]).

generate_react_native_component(_, '<View />').

size_to_rn(small, 32).
size_to_rn(medium, 48).
size_to_rn(large, 64).
size_to_rn(Size, Size) :- number(Size).

% ============================================================================
% Vue Component Generation
% ============================================================================

%! generate_vue_component(+Spec, -Code) is det
generate_vue_component(modal_spec(_, Props), Code) :-
    member(title(Title), Props),
    member(message(Message), Props),
    format(atom(Code), '<template>\n  <Teleport to="body">\n    <div v-if="visible" class="modal-overlay">\n      <div class="modal">\n        <h2>~w</h2>\n        <p>~w</p>\n        <button @click="close">Close</button>\n      </div>\n    </div>\n  </Teleport>\n</template>', [Title, Message]).

generate_vue_component(toast_spec(Props), Code) :-
    member(message(Message), Props),
    member(type(Type), Props),
    format(atom(Code), '<template>\n  <Transition name="toast">\n    <div v-if="visible" :class="[\'toast\', \'toast--~w\']">\n      {{ \'~w\' }}\n    </div>\n  </Transition>\n</template>', [Type, Message]).

generate_vue_component(card_spec(Props), Code) :-
    member(title(Title), Props),
    format(atom(Code), '<template>\n  <div class="card" :class="{ \'card--elevated\': elevated }">\n    <img v-if="image" :src="image" class="card__image" />\n    <div class="card__content">\n      <h3 v-if="title" class="card__title">~w</h3>\n      <slot />\n    </div>\n  </div>\n</template>', [Title]).

generate_vue_component(avatar_spec(Props), Code) :-
    member(source(Source), Props),
    member(size(Size), Props),
    format(atom(Code), '<template>\n  <img\n    :src="\'~w\'"\n    class="avatar avatar--~w"\n    :alt="alt"\n  />\n</template>', [Source, Size]).

generate_vue_component(badge_spec(Props), Code) :-
    member(content(Content), Props),
    member(color(Color), Props),
    format(atom(Code), '<template>\n  <span class="badge badge--~w">~w</span>\n</template>', [Color, Content]).

generate_vue_component(spinner_spec(Props), Code) :-
    member(size(Size), Props),
    format(atom(Code), '<template>\n  <div class="spinner spinner--~w" />\n</template>', [Size]).

generate_vue_component(search_bar_spec(Props), Code) :-
    member(placeholder(Placeholder), Props),
    format(atom(Code), '<template>\n  <div class="search-bar">\n    <input\n      v-model="searchText"\n      type="search"\n      placeholder="~w"\n      @input="onSearch"\n    />\n  </div>\n</template>', [Placeholder]).

generate_vue_component(_, '<template><div /></template>').

% ============================================================================
% Flutter Component Generation
% ============================================================================

%! generate_flutter_component(+Spec, -Code) is det
generate_flutter_component(modal_spec(_, Props), Code) :-
    member(title(Title), Props),
    member(message(Message), Props),
    format(atom(Code), 'showDialog(\n  context: context,\n  builder: (context) => AlertDialog(\n    title: Text(\'~w\'),\n    content: Text(\'~w\'),\n    actions: [\n      TextButton(onPressed: () => Navigator.pop(context), child: Text(\'Close\')),\n    ],\n  ),\n);', [Title, Message]).

generate_flutter_component(toast_spec(Props), Code) :-
    member(message(Message), Props),
    format(atom(Code), 'ScaffoldMessenger.of(context).showSnackBar(\n  SnackBar(content: Text(\'~w\')),\n);', [Message]).

generate_flutter_component(card_spec(Props), Code) :-
    member(title(Title), Props),
    format(atom(Code), 'Card(\n  elevation: elevated ? 4 : 0,\n  child: Column(\n    children: [\n      if (image != null) Image.network(image),\n      Padding(\n        padding: EdgeInsets.all(16),\n        child: Column(\n          crossAxisAlignment: CrossAxisAlignment.start,\n          children: [\n            Text(\'~w\', style: Theme.of(context).textTheme.titleLarge),\n            child,\n          ],\n        ),\n      ),\n    ],\n  ),\n)', [Title]).

generate_flutter_component(avatar_spec(Props), Code) :-
    member(source(Source), Props),
    member(size(Size), Props),
    size_to_flutter(Size, Radius),
    format(atom(Code), 'CircleAvatar(\n  radius: ~w,\n  backgroundImage: NetworkImage(\'~w\'),\n)', [Radius, Source]).

generate_flutter_component(badge_spec(Props), Code) :-
    member(content(Content), Props),
    format(atom(Code), 'Badge(\n  label: Text(\'~w\'),\n  child: child,\n)', [Content]).

generate_flutter_component(spinner_spec(Props), Code) :-
    member(size(Size), Props),
    size_to_flutter_spinner(Size, SizeVal),
    format(atom(Code), 'SizedBox(\n  width: ~w,\n  height: ~w,\n  child: CircularProgressIndicator(),\n)', [SizeVal, SizeVal]).

generate_flutter_component(progress_bar_spec(Props), Code) :-
    member(value(Value), Props),
    member(max(Max), Props),
    Progress is Value / Max,
    format(atom(Code), 'LinearProgressIndicator(value: ~w)', [Progress]).

generate_flutter_component(search_bar_spec(Props), Code) :-
    member(placeholder(Placeholder), Props),
    format(atom(Code), 'SearchBar(\n  hintText: \'~w\',\n  onChanged: onSearch,\n)', [Placeholder]).

generate_flutter_component(_, 'Container()').

size_to_flutter(small, 16).
size_to_flutter(medium, 24).
size_to_flutter(large, 32).
size_to_flutter(Size, Size) :- number(Size).

size_to_flutter_spinner(small, 20).
size_to_flutter_spinner(medium, 36).
size_to_flutter_spinner(large, 48).

% ============================================================================
% SwiftUI Component Generation
% ============================================================================

%! generate_swiftui_component(+Spec, -Code) is det
generate_swiftui_component(modal_spec(_, Props), Code) :-
    member(title(Title), Props),
    member(message(Message), Props),
    format(atom(Code), '.alert(\"~w\", isPresented: $showAlert) {\n    Button(\"OK\", role: .cancel) { }\n} message: {\n    Text(\"~w\")\n}', [Title, Message]).

generate_swiftui_component(toast_spec(Props), Code) :-
    member(message(Message), Props),
    format(atom(Code), 'Toast(message: \"~w\", isShowing: $showToast)', [Message]).

generate_swiftui_component(card_spec(Props), Code) :-
    member(title(Title), Props),
    format(atom(Code), 'VStack(alignment: .leading) {\n    if let image = image {\n        AsyncImage(url: URL(string: image))\n    }\n    Text(\"~w\")\n        .font(.headline)\n    content\n}\n.padding()\n.background(Color(.systemBackground))\n.cornerRadius(12)\n.shadow(radius: elevated ? 4 : 0)', [Title]).

generate_swiftui_component(avatar_spec(Props), Code) :-
    member(source(Source), Props),
    member(size(Size), Props),
    size_to_swift(Size, SizeVal),
    format(atom(Code), 'AsyncImage(url: URL(string: \"~w\")) { image in\n    image.resizable()\n} placeholder: {\n    Circle().fill(Color.gray)\n}\n.frame(width: ~w, height: ~w)\n.clipShape(Circle())', [Source, SizeVal, SizeVal]).

generate_swiftui_component(badge_spec(Props), Code) :-
    member(content(Content), Props),
    member(color(Color), Props),
    format(atom(Code), 'Text(\"~w\")\n    .font(.caption2)\n    .padding(.horizontal, 8)\n    .padding(.vertical, 4)\n    .background(Color.~w)\n    .foregroundColor(.white)\n    .clipShape(Capsule())', [Content, Color]).

generate_swiftui_component(spinner_spec(_), Code) :-
    Code = 'ProgressView()'.

generate_swiftui_component(progress_bar_spec(Props), Code) :-
    member(value(Value), Props),
    member(max(Max), Props),
    format(atom(Code), 'ProgressView(value: ~w, total: ~w)', [Value, Max]).

generate_swiftui_component(search_bar_spec(Props), Code) :-
    member(placeholder(Placeholder), Props),
    format(atom(Code), 'TextField(\"~w\", text: $searchText)\n    .textFieldStyle(.roundedBorder)', [Placeholder]).

generate_swiftui_component(_, 'EmptyView()').

size_to_swift(small, 32).
size_to_swift(medium, 48).
size_to_swift(large, 64).
size_to_swift(Size, Size) :- number(Size).

% ============================================================================
% Testing
% ============================================================================

%! test_component_library is det
test_component_library :-
    format('Running component_library tests...~n'),

    % Test 1: Modal creation
    modal(alert, [title('Test'), message('Hello')], ModalSpec),
    ModalSpec = modal_spec(alert, _),
    format('  Test 1 passed: modal creation~n'),

    % Test 2: Toast creation
    toast('Message', [type(success), duration(2000)], ToastSpec),
    ToastSpec = toast_spec(Props2),
    member(type(success), Props2),
    format('  Test 2 passed: toast creation~n'),

    % Test 3: Card creation
    card(content, [title('Card Title'), elevated(true)], CardSpec),
    CardSpec = card_spec(Props3),
    member(title('Card Title'), Props3),
    format('  Test 3 passed: card creation~n'),

    % Test 4: Avatar creation
    avatar('https://example.com/avatar.jpg', [size(large)], AvatarSpec),
    AvatarSpec = avatar_spec(Props4),
    member(size(large), Props4),
    format('  Test 4 passed: avatar creation~n'),

    % Test 5: Badge creation
    badge('5', [color(error)], BadgeSpec),
    BadgeSpec = badge_spec(Props5),
    member(color(error), Props5),
    format('  Test 5 passed: badge creation~n'),

    % Test 6: React Native generation
    generate_component(ToastSpec, react_native, RNCode),
    sub_string(RNCode, _, _, _, "Toast"),
    format('  Test 6 passed: React Native generation~n'),

    % Test 7: Vue generation
    generate_component(CardSpec, vue, VueCode),
    sub_string(VueCode, _, _, _, "card"),
    format('  Test 7 passed: Vue generation~n'),

    % Test 8: Flutter generation
    generate_component(AvatarSpec, flutter, FlutterCode),
    sub_string(FlutterCode, _, _, _, "CircleAvatar"),
    format('  Test 8 passed: Flutter generation~n'),

    % Test 9: SwiftUI generation
    generate_component(BadgeSpec, swiftui, SwiftCode),
    sub_string(SwiftCode, _, _, _, "Text"),
    format('  Test 9 passed: SwiftUI generation~n'),

    % Test 10: Progress bar
    progress_bar(75, [max(100), showLabel(true)], ProgressSpec),
    ProgressSpec = progress_bar_spec(Props10),
    member(value(75), Props10),
    format('  Test 10 passed: progress bar~n'),

    format('All 10 component_library tests passed!~n'),
    !.

:- initialization(test_component_library, main).
