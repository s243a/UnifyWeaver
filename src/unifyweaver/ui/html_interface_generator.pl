% SPDX-License-Identifier: MIT OR Apache-2.0
% Copyright (c) 2025-2026 John William Creighton (s243a)
%
% html_interface_generator.pl - Generate HTML/Vue interfaces from UI specs
%
% Compiles declarative UI specifications to complete HTML documents
% with embedded Vue.js for interactivity.
%
% Usage:
%   use_module('html_interface_generator').
%   use_module('http_cli_ui').
%
%   http_cli_interface(Spec),
%   http_cli_theme(Theme),
%   generate_html_interface(Spec, Theme, HTML).

:- module(html_interface_generator, [
    % Main generation
    generate_html_interface/3,      % +UISpec, +Theme, -HTML
    generate_html_interface/4,      % +UISpec, +Theme, +Options, -HTML

    % Section generators
    generate_html_head/3,           % +Title, +Theme, -Head
    generate_css_from_theme/2,      % +Theme, -CSS
    generate_vue_template/2,        % +UISpec, -Template
    generate_vue_app/2,             % +Options, -Script

    % Testing
    test_html_interface_generator/0
]).

:- use_module(library(lists)).
:- catch(use_module('vue_generator'), _, true).
:- catch(use_module('ui_patterns'), _, true).

% ============================================================================
% MAIN GENERATION
% ============================================================================

%! generate_html_interface(+UISpec, +Theme, -HTML) is det
%  Generate complete HTML interface from UI spec and theme.
generate_html_interface(UISpec, Theme, HTML) :-
    generate_html_interface(UISpec, Theme, [], HTML).

%! generate_html_interface(+UISpec, +Theme, +Options, -HTML) is det
%  Generate HTML interface with options.
generate_html_interface(UISpec, Theme, Options, HTML) :-
    % Extract page info
    get_page_title(UISpec, Title),

    % Generate sections
    generate_html_head(Title, Theme, Head),
    generate_vue_template_from_spec(UISpec, Template),
    generate_vue_app(Options, VueApp),

    % Combine into complete HTML
    format(atom(HTML), '<!DOCTYPE html>
<html lang="en">
~w
<body>
  <div id="app" class="container">
~w
  </div>
  <script>
~w
  </script>
</body>
</html>', [Head, Template, VueApp]).

get_page_title(page(_, Opts, _), Title) :-
    member(title(Title), Opts), !.
get_page_title(_, "UnifyWeaver Interface").

% ============================================================================
% HTML HEAD GENERATION
% ============================================================================

%! generate_html_head(+Title, +Theme, -Head) is det
generate_html_head(Title, Theme, Head) :-
    generate_css_from_theme(Theme, CSS),
    format(atom(Head), '<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>~w</title>
  <script src="https://unpkg.com/vue@3/dist/vue.global.prod.js"></script>
  <style>
~w
  </style>
</head>', [Title, CSS]).

% ============================================================================
% CSS GENERATION FROM THEME
% ============================================================================

%! generate_css_from_theme(+Theme, -CSS) is det
generate_css_from_theme(theme(_, Def), CSS) :-
    % Extract theme values
    get_theme_colors(Def, Colors),
    get_theme_typography(Def, Typography),
    get_theme_spacing(Def, Spacing),
    get_theme_borders(Def, Borders),

    % Generate CSS
    generate_base_css(Colors, Typography, BaseCSS),
    generate_component_css(Colors, Spacing, Borders, ComponentCSS),

    atomic_list_concat([BaseCSS, '\n', ComponentCSS], CSS).

generate_css_from_theme(_, CSS) :-
    % Fallback: use default HTTP CLI theme CSS
    generate_default_css(CSS).

get_theme_colors(Def, Colors) :-
    member(colors(Colors), Def), !.
get_theme_colors(_, []).

get_theme_typography(Def, Typography) :-
    member(typography(Typography), Def), !.
get_theme_typography(_, []).

get_theme_spacing(Def, Spacing) :-
    member(spacing(Spacing), Def), !.
get_theme_spacing(_, []).

get_theme_borders(Def, Borders) :-
    member(borders(Borders), Def), !.
get_theme_borders(_, []).

%! generate_base_css(+Colors, +Typography, -CSS) is det
generate_base_css(Colors, Typography, CSS) :-
    get_color(background, Colors, BgColor, '#1a1a2e'),
    get_color(text, Colors, TextColor, '#eee'),
    get_color(primary, Colors, PrimaryColor, '#e94560'),
    get_typography(font_family, Typography, FontFamily, '-apple-system, BlinkMacSystemFont, Segoe UI, Roboto, monospace'),

    format(atom(CSS), '    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: ~w;
      background: ~w;
      color: ~w;
      min-height: 100vh;
      padding: 20px;
    }
    .container { max-width: 1200px; margin: 0 auto; }
    h1 { color: ~w; margin-bottom: 20px; }
    h2 { color: ~w; margin-bottom: 15px; }
    a { color: ~w; text-decoration: none; }
    a:hover { text-decoration: underline; }
', [FontFamily, BgColor, TextColor, PrimaryColor, PrimaryColor, PrimaryColor]).

%! generate_component_css(+Colors, +Spacing, +Borders, -CSS) is det
generate_component_css(Colors, _Spacing, Borders, CSS) :-
    get_color(surface, Colors, SurfaceColor, '#16213e'),
    get_color(secondary, Colors, SecondaryColor, '#0f3460'),
    get_color(primary, Colors, PrimaryColor, '#e94560'),
    get_color(muted, Colors, MutedColor, '#94a3b8'),
    get_color(success, Colors, SuccessColor, '#4ade80'),
    get_color(warning, Colors, WarningColor, '#fbbf24'),
    get_color(error, Colors, ErrorColor, '#ff6b6b'),
    get_border(radius, Borders, BorderRadius, 5),

    format(atom(CSS), '    .tabs {
      display: flex;
      flex-wrap: wrap;
      gap: 5px;
      margin-bottom: 20px;
    }
    .tab {
      padding: 10px 20px;
      background: ~w;
      border: none;
      color: ~w;
      cursor: pointer;
      border-radius: ~wpx ~wpx 0 0;
    }
    .tab.active { background: ~w; color: ~w; }
    .tab:hover { background: ~w; }
    .panel {
      background: ~w;
      padding: 20px;
      border-radius: 0 ~wpx ~wpx ~wpx;
    }
    .form-group { margin-bottom: 15px; }
    label { display: block; margin-bottom: 5px; color: ~w; }
    input, textarea, select {
      width: 100%;
      padding: 10px;
      background: ~w;
      border: 1px solid #1a1a2e;
      color: #eee;
      border-radius: ~wpx;
      font-family: monospace;
    }
    input:focus, textarea:focus, select:focus {
      outline: none;
      border-color: ~w;
    }
    button {
      padding: 10px 20px;
      background: ~w;
      border: none;
      color: #fff;
      cursor: pointer;
      border-radius: ~wpx;
      font-weight: bold;
    }
    button:hover { background: #ff6b6b; }
    button:disabled { background: #444; cursor: not-allowed; }
    .results {
      margin-top: 20px;
      background: ~w;
      padding: 15px;
      border-radius: ~wpx;
      max-height: 500px;
      overflow: auto;
    }
    .results pre {
      white-space: pre-wrap;
      word-wrap: break-word;
      font-size: 13px;
      line-height: 1.5;
    }
    .error { color: ~w; }
    .success { color: ~w; }
    .warning { color: ~w; }
    .count { color: ~w; font-size: 14px; margin-bottom: 10px; }
    .login-container {
      max-width: 400px;
      margin: 50px auto;
      padding: 30px;
      background: ~w;
      border-radius: 10px;
    }
    .login-container h2 {
      color: ~w;
      margin-bottom: 20px;
      text-align: center;
    }
    .user-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      background: ~w;
      padding: 10px 15px;
      border-radius: ~wpx;
      margin-bottom: 15px;
      flex-wrap: wrap;
      gap: 10px;
    }
    .user-info {
      display: flex;
      align-items: center;
      gap: 10px;
    }
    .user-email { color: ~w; font-weight: bold; }
    .user-roles {
      display: flex;
      gap: 5px;
    }
    .role-badge {
      padding: 2px 8px;
      background: #1a1a2e;
      border-radius: 3px;
      font-size: 11px;
      color: ~w;
    }
    .role-badge.shell { background: ~w; color: #fff; }
    .role-badge.admin { background: #3b82f6; color: #fff; }
    .working-dir-bar {
      background: ~w;
      padding: 10px 15px;
      border-radius: ~wpx;
      margin-bottom: 15px;
      display: flex;
      align-items: center;
      gap: 10px;
      flex-wrap: wrap;
    }
    code {
      background: #1a1a2e;
      padding: 4px 8px;
      border-radius: 3px;
      font-family: monospace;
    }
    .file-entry {
      padding: 10px;
      background: ~w;
      margin: 3px 0;
      border-radius: ~wpx;
      cursor: pointer;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .file-entry:hover { background: #1a3a5e; }
    .terminal {
      background: #000;
      padding: 15px;
      border-radius: ~wpx;
      font-family: monospace;
      min-height: 300px;
      max-height: 500px;
      overflow-y: auto;
    }
    .terminal pre {
      color: #0f0;
      margin: 0;
      white-space: pre-wrap;
    }
    .shell-input {
      flex: 1;
      background: transparent;
      border: none;
      color: #0f0;
      font-family: monospace;
    }
    .shell-input:focus { outline: none; }
    .prompt { color: ~w; font-family: monospace; }
', [SurfaceColor, MutedColor, BorderRadius, BorderRadius, SecondaryColor, PrimaryColor,
    SecondaryColor, SurfaceColor, BorderRadius, BorderRadius, BorderRadius, MutedColor,
    SecondaryColor, BorderRadius, PrimaryColor, PrimaryColor, BorderRadius,
    SecondaryColor, BorderRadius, ErrorColor, SuccessColor, WarningColor, MutedColor,
    SurfaceColor, PrimaryColor, SecondaryColor, BorderRadius, SuccessColor, MutedColor,
    PrimaryColor, SecondaryColor, BorderRadius, SecondaryColor, BorderRadius, BorderRadius,
    SuccessColor]).

get_color(Name, Colors, Value, _Default) :-
    Term =.. [Name, Value],
    member(Term, Colors), !.
get_color(_, _, Default, Default).

get_typography(Name, Typography, Value, _Default) :-
    Term =.. [Name, Value],
    member(Term, Typography), !.
get_typography(_, _, Default, Default).

get_border(Name, Borders, Value, _Default) :-
    Term =.. [Name, Value],
    member(Term, Borders), !.
get_border(_, _, Default, Default).

%! generate_default_css(-CSS) is det
generate_default_css(CSS) :-
    generate_base_css([], [], BaseCSS),
    generate_component_css([], [], [], ComponentCSS),
    atomic_list_concat([BaseCSS, '\n', ComponentCSS], CSS).

% ============================================================================
% VUE TEMPLATE GENERATION
% ============================================================================

%! generate_vue_template_from_spec(+UISpec, -Template) is det
generate_vue_template_from_spec(page(_, _, Regions), Template) :- !,
    findall(RegionCode, (
        member(region(_, _, Content), Regions),
        generate_region_content(Content, RegionCode)
    ), RegionCodes),
    atomic_list_concat(RegionCodes, '\n', Template).

generate_vue_template_from_spec(UISpec, Template) :-
    % Try using vue_generator if available
    (   current_predicate(vue_generator:generate_vue_template/2)
    ->  vue_generator:generate_vue_template(UISpec, Template)
    ;   format(atom(Template), '<!-- UI Spec: ~w -->', [UISpec])
    ).

generate_region_content(Content, Code) :-
    is_list(Content), !,
    maplist(generate_content_item, Content, Codes),
    atomic_list_concat(Codes, '\n', Code).
generate_region_content(Content, Code) :-
    generate_content_item(Content, Code), !.

generate_content_item(when(Condition, Content), Code) :- !,
    condition_to_vue(Condition, VueCondition),
    generate_region_content(Content, InnerCode),
    format(atom(Code), '    <template v-if="~w">\n~w\n    </template>', [VueCondition, InnerCode]).

generate_content_item(use_spec(SpecPred), Code) :- !,
    % Call the spec predicate to get the actual spec
    % Build qualified call for http_cli_ui module
    QualifiedCall =.. [SpecPred, Spec],
    (   catch(http_cli_ui:QualifiedCall, _, fail)
    ->  generate_content_item(Spec, Code)
    ;   catch(call(SpecPred, Spec), _, fail)
    ->  generate_content_item(Spec, Code)
    ;   format(atom(Code), '<!-- Spec not found: ~w -->', [SpecPred])
    ).

generate_content_item(component(Type, Opts), Code) :- !,
    (   current_predicate(vue_generator:generate_vue_template/2)
    ->  vue_generator:generate_vue_template(component(Type, Opts), Code)
    ;   format(atom(Code), '<!-- Component: ~w -->', [Type])
    ).

generate_content_item(container(Type, Opts, Content), Code) :- !,
    (   current_predicate(vue_generator:generate_vue_template/2)
    ->  vue_generator:generate_vue_template(container(Type, Opts, Content), Code)
    ;   format(atom(Code), '<!-- Container: ~w -->', [Type])
    ).

generate_content_item(layout(Type, Opts, Children), Code) :- !,
    (   current_predicate(vue_generator:generate_vue_template/2)
    ->  vue_generator:generate_vue_template(layout(Type, Opts, Children), Code)
    ;   format(atom(Code), '<!-- Layout: ~w -->', [Type])
    ).

generate_content_item(panel_switch(Variable, Cases), Code) :- !,
    % Generate Vue v-if/v-else-if chain for panel switch
    generate_panel_switch_cases(Variable, Cases, 0, CaseCodes),
    atomic_list_concat(CaseCodes, '\n', Code).

generate_content_item(case(_, _), Code) :- !,
    % Cases are handled by panel_switch - shouldn't be standalone
    Code = ''.

generate_content_item(foreach(Collection, Var, Content), Code) :- !,
    % Generate Vue v-for loop
    collection_to_vue(Collection, VueCollection),
    generate_region_content(Content, InnerCode),
    format(atom(Code), '    <template v-for="~w in ~w" :key="~w">\n~w\n    </template>',
           [Var, VueCollection, Var, InnerCode]).

generate_content_item(Item, Code) :-
    % Fallback - don't serialize the entire term, just indicate what type it is
    functor(Item, Functor, _),
    format(atom(Code), '<!-- Unhandled item type: ~w -->', [Functor]).

%! generate_panel_switch_cases(+Variable, +Cases, +Index, -Codes) is det
%  Generate Vue conditional chain for panel switch cases.
generate_panel_switch_cases(_, [], _, []).
generate_panel_switch_cases(Variable, [case(Value, Content)|Rest], Index, [Code|RestCodes]) :-
    % Determine v-if vs v-else-if
    (   Index =:= 0
    ->  Directive = 'v-if'
    ;   Directive = 'v-else-if'
    ),
    % Generate the content
    generate_region_content(Content, InnerCode),
    % Create the conditional template
    format(atom(Code), '    <template ~w="~w === \'~w\'">\n~w\n    </template>',
           [Directive, Variable, Value, InnerCode]),
    NextIndex is Index + 1,
    generate_panel_switch_cases(Variable, Rest, NextIndex, RestCodes).

collection_to_vue(var(Name), CamelName) :- !,
    snake_to_camel(Name, CamelName).
collection_to_vue(Name, CamelName) :-
    snake_to_camel(Name, CamelName).

% Snake_case to camelCase conversion
snake_to_camel(Snake, Camel) :-
    atom_codes(Snake, Codes),
    snake_to_camel_codes(Codes, CamelCodes),
    atom_codes(Camel, CamelCodes).

snake_to_camel_codes([], []).
snake_to_camel_codes([0'_,C|Rest], [Upper|CamelRest]) :- !,
    (C >= 0'a, C =< 0'z -> Upper is C - 32 ; Upper = C),
    snake_to_camel_codes(Rest, CamelRest).
snake_to_camel_codes([C|Rest], [C|CamelRest]) :-
    snake_to_camel_codes(Rest, CamelRest).

condition_to_vue(and(A, B), VueCondition) :- !,
    condition_to_vue(A, VA),
    condition_to_vue(B, VB),
    format(atom(VueCondition), '(~w) && (~w)', [VA, VB]).
condition_to_vue(or(A, B), VueCondition) :- !,
    condition_to_vue(A, VA),
    condition_to_vue(B, VB),
    format(atom(VueCondition), '(~w) || (~w)', [VA, VB]).
condition_to_vue(not(A), VueCondition) :- !,
    condition_to_vue(A, VA),
    format(atom(VueCondition), '!~w', [VA]).
condition_to_vue(var(Name), CamelName) :- !,
    snake_to_camel(Name, CamelName).
condition_to_vue(Atom, CamelAtom) :- atom(Atom), !,
    snake_to_camel(Atom, CamelAtom).

% ============================================================================
% VUE APP GENERATION
% ============================================================================

%! generate_vue_app(+Options, -Script) is det
%  Generate the Vue.js application code.
generate_vue_app(_Options, Script) :-
    % Build the script in parts to avoid quoting issues
    generate_vue_app_parts(Parts),
    atomic_list_concat(Parts, Script).

generate_vue_app_parts([
'const { createApp, ref, reactive, computed, onMounted, watch } = Vue;

const app = createApp({
  setup() {
    // Auth state
    const authRequired = ref(false);
    const user = ref(null);
    const loginEmail = ref("");
    const loginPassword = ref("");
    const loginError = ref("");
    const loading = ref(false);
    const token = ref(localStorage.getItem("token") || "");

    // UI state
    const tab = ref("browse");
    const workingDir = ref(".");
    const results = ref("");
    const resultCount = ref(0);

    // Tab-specific state
    const browse = reactive({
      path: ".",
      entries: [],
      selected: null,
      parent: null
    });

    const grep = reactive({ pattern: "", path: ".", options: ""});
    const find = reactive({ pattern: "", path: ".", options: ""});
    const cat = reactive({ path: ""});
    const exec = reactive({ commandLine: ""});
    const feedback = reactive({ type: "info", message: ""});
    const shell = reactive({
      connected: false,
      output: "",
      input: "",
      ws: null
    });
    const shellTextMode = ref(true);  // true = text mode, false = capture mode

    // API helpers
    const apiCall = async (endpoint, method = "GET", body = null) => {
      const headers = { "Content-Type": "application/json" };
      if (token.value) headers["Authorization"] = `Bearer ${token.value}`;
      const opts = { method, headers };
      if (body) opts.body = JSON.stringify(body);
      const res = await fetch(endpoint, opts);
      return res.json();
    };

    // Auth methods
    const checkAuth = async () => {
      const res = await apiCall("/auth/status");
      authRequired.value = res.data?.authRequired || false;
      if (token.value) {
        const me = await apiCall("/auth/me");
        if (me.success) user.value = me.data;
      }
    };

    const doLogin = async () => {
      loading.value = true;
      loginError.value = "";
      try {
        const res = await apiCall("/auth/login", "POST", {
          email: loginEmail.value,
          password: loginPassword.value
        });
        if (res.success) {
          token.value = res.data.token;
          user.value = res.data.user;
          localStorage.setItem("token", res.data.token);
        } else {
          loginError.value = res.error || "Login failed";
        }
      } catch (e) {
        loginError.value = e.message;
      }
      loading.value = false;
    };

    const doLogout = () => {
      token.value = "";
      user.value = null;
      localStorage.removeItem("token");
    };

    // Browse methods
    const loadBrowse = async (path = browse.path) => {
      loading.value = true;
      const res = await apiCall("/browse", "POST", { path, workingDir: workingDir.value });
      if (res.success) {
        browse.path = res.data.path;
        browse.entries = res.data.entries || [];
        browse.parent = res.data.parent;
        browse.selected = null;
      } else {
        results.value = res.error;
      }
      loading.value = false;
    };

    const navigateTo = (path) => loadBrowse(path);
    const selectFile = (path) => { browse.selected = path; };

    // Search methods
    const doGrep = async () => {
      loading.value = true;
      const res = await apiCall("/grep", "POST", {
        pattern: grep.pattern,
        path: grep.path || workingDir.value,
        options: grep.options
      });
      results.value = res.success ? res.data.output : res.error;
      resultCount.value = res.data?.count || 0;
      loading.value = false;
    };

    const doFind = async () => {
      loading.value = true;
      const res = await apiCall("/find", "POST", {
        pattern: find.pattern,
        path: find.path || workingDir.value,
        options: find.options
      });
      results.value = res.success ? res.data.output : res.error;
      resultCount.value = res.data?.count || 0;
      loading.value = false;
    };

    const doCat = async () => {
      loading.value = true;
      const res = await apiCall("/cat", "POST", {
        path: cat.path,
        workingDir: workingDir.value
      });
      results.value = res.success ? res.data.content : res.error;
      loading.value = false;
    };

    const doExec = async () => {
      loading.value = true;
      const res = await apiCall("/exec", "POST", {
        command: exec.commandLine,
        workingDir: workingDir.value
      });
      results.value = res.success ? res.data.stdout : res.error;
      loading.value = false;
    };

    const doFeedback = async () => {
      loading.value = true;
      const res = await apiCall("/feedback", "POST", {
        type: feedback.type,
        message: feedback.message
      });
      if (res.success) {
        feedback.message = "";
        results.value = "Feedback submitted. Thank you!";
      } else {
        results.value = res.error;
      }
      loading.value = false;
    };

    // Shell methods
    const connectShell = () => {
      const protocol = location.protocol === "https:" ? "wss:" : "ws:";
      const wsUrl = `${protocol}//${location.host}/shell?token=${token.value}`;
      shell.ws = new WebSocket(wsUrl);
      shell.ws.onopen = () => { shell.connected = true; shell.output = "Connected to shell.\\\\n"; };
      shell.ws.onmessage = (e) => {
        try {
          const msg = JSON.parse(e.data);
          if (msg.type === "output" || msg.type === "error") {
            shell.output += msg.data;
          } else if (msg.type === "prompt") {
            shell.output += "$ ";
          }
        } catch {
          shell.output += e.data;
        }
        scrollShell();
      };
      shell.ws.onclose = () => { shell.connected = false; shell.output += "\\\\nDisconnected.\\\\n"; };
      shell.ws.onerror = () => { shell.connected = false; };
    };

    const sendShellCommand = () => {
      if (shell.ws && shell.connected && shell.input) {
        shell.ws.send(JSON.stringify({ type: "input", data: shell.input + "\\\\n" }));
        shell.input = "";
      }
    };

    const clearShell = () => { shell.output = ""; };

    const disconnectShell = () => {
      if (shell.ws) {
        shell.ws.close();
        shell.ws = null;
      }
    };

    const toggleShellMode = () => {
      shellTextMode.value = !shellTextMode.value;
      if (!shellTextMode.value) {
        // Switching to capture mode - focus the hidden input
        setTimeout(() => focusCaptureInput(), 100);
      }
    };

    const focusCaptureInput = () => {
      const input = document.getElementById("shell_capture_input");
      if (input) input.focus();
    };

    const handleCaptureInput = (e) => {
      if (!shell.ws || !shell.connected) return;
      // Handle backspace on mobile (inputType is deleteContentBackward)
      if (e.inputType === "deleteContentBackward") {
        shell.ws.send(JSON.stringify({ type: "input", data: String.fromCharCode(127) }));
        return;
      }
      const value = e.target.value;
      if (value) {
        // Send each character individually
        for (const char of value) {
          shell.ws.send(JSON.stringify({ type: "input", data: char }));
        }
        e.target.value = "";
      }
    };

    const handleCaptureKeydown = (e) => {
      if (!shell.ws || !shell.connected) return;
      const sendInput = (data) => shell.ws.send(JSON.stringify({ type: "input", data }));
      // Handle special keys
      if (e.key === "Enter") {
        e.preventDefault();
        sendInput(String.fromCharCode(13));
      } else if (e.key === "Backspace") {
        e.preventDefault();
        sendInput(String.fromCharCode(127));
      } else if (e.key === "Tab") {
        e.preventDefault();
        sendInput(String.fromCharCode(9));
      } else if (e.ctrlKey && e.key === "c") {
        e.preventDefault();
        sendInput(String.fromCharCode(3));
      } else if (e.ctrlKey && e.key === "d") {
        e.preventDefault();
        sendInput(String.fromCharCode(4));
      }
    };

    const scrollShell = () => {
      const el = document.getElementById("shell_output");
      if (el) el.scrollTop = el.scrollHeight;
    };

    // Utilities
    const formatSize = (bytes) => {
      if (!bytes) return "";
      if (bytes < 1024) return bytes + " B";
      if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
      return (bytes / 1024 / 1024).toFixed(1) + " MB";
    };

    // Lifecycle
    onMounted(() => {
      checkAuth();
      loadBrowse();
    });

    watch(tab, (newTab) => {
      if (newTab === "shell" && !shell.connected && user.value?.roles?.includes("shell")) {
        connectShell();
      }
    });

    return {
      authRequired, user, loginEmail, loginPassword, loginError, loading, token,
      tab, workingDir, results, resultCount,
      browse, grep, find, cat, exec, feedback, shell, shellTextMode,
      doLogin, doLogout, loadBrowse, navigateTo, selectFile,
      doGrep, doFind, doCat, doExec, doFeedback,
      connectShell, disconnectShell, sendShellCommand, clearShell,
      toggleShellMode, focusCaptureInput, handleCaptureInput, handleCaptureKeydown,
      formatSize
    };
  }
});

app.mount("#app");'
]).

% ============================================================================
% TESTING
% ============================================================================

test_html_interface_generator :-
    format('~n=== HTML Interface Generator Tests ===~n~n'),

    % Test 1: CSS generation
    format('Test 1: CSS generation from theme...~n'),
    Theme = theme(test, [
        colors([primary('#ff0000'), background('#000'), text('#fff')]),
        borders([radius(10)])
    ]),
    generate_css_from_theme(Theme, CSS),
    (   sub_atom(CSS, _, _, _, 'background')
    ->  format('  PASS: CSS generated~n')
    ;   format('  FAIL: CSS missing expected content~n')
    ),

    % Test 2: Simple HTML generation
    format('~nTest 2: Simple component generation...~n'),
    SimpleSpec = layout(stack, [spacing(16)], [
        component(heading, [level(1), content("Test")]),
        component(button, [label("Click")])
    ]),
    generate_vue_template_from_spec(SimpleSpec, SimpleTemplate),
    format('  Generated: ~w~n', [SimpleTemplate]),
    (   SimpleTemplate \= ''
    ->  format('  PASS: Template generated~n')
    ;   format('  FAIL: Empty template~n')
    ),

    % Test 3: Vue app generation
    format('~nTest 3: Vue app generation...~n'),
    generate_vue_app([], VueApp),
    (   sub_atom(VueApp, _, _, _, 'createApp')
    ->  format('  PASS: Vue app contains createApp~n')
    ;   format('  FAIL: Vue app missing createApp~n')
    ),

    % Test 4: Full HTML generation
    format('~nTest 4: Full HTML document generation...~n'),
    generate_html_interface(SimpleSpec, Theme, HTML),
    (   sub_atom(HTML, _, _, _, '<!DOCTYPE html>'),
        sub_atom(HTML, _, _, _, '<script>')
    ->  format('  PASS: Complete HTML document generated~n')
    ;   format('  FAIL: HTML document incomplete~n')
    ),

    format('~n=== Tests Complete ===~n').

% ============================================================================
% MODULE INITIALIZATION
% ============================================================================

:- initialization((
    format('HTML Interface Generator module loaded~n')
), now).
