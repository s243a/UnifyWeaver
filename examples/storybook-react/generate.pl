#!/usr/bin/env swipl

% Generate React components for Storybook from UnifyWeaver UI specifications

:- use_module('../../src/unifyweaver/ui/react_generator', [generate_react_template/2]).
:- use_module('../../src/unifyweaver/ui/ui_primitives', []).
:- use_module('../../src/unifyweaver/ui/http_cli_ui', [browse_panel_spec/1]).

% ============================================================================
% COMPONENT GENERATION
% ============================================================================

%! generate_browse_panel(-Code) is det
%  Generate the file browser panel component.
generate_browse_panel(Code) :-
    browse_panel_spec(UISpec),
    generate_react_template(UISpec, Template),
    format(atom(Code),
'import React, { useState } from "react";

// Generated from: browse_panel_spec/1 in http_cli_ui.pl

interface FileEntry {
  name: string;
  type: "file" | "directory";
  size: number;
}

interface BrowsePanelProps {
  initialPath?: string;
  onFileSelect?: (file: FileEntry) => void;
  onNavigate?: (path: string) => void;
}

export const BrowsePanel: React.FC<BrowsePanelProps> = ({
  initialPath = ".",
  onFileSelect,
  onNavigate,
}) => {
  const [browse, setBrowse] = useState({
    path: initialPath,
    entries: [
      { name: "example.txt", type: "file" as const, size: 1024 },
      { name: "folder", type: "directory" as const, size: 0 },
    ],
    selected: null as string | null,
    parent: false,
  });
  const [loading, setLoading] = useState(false);

  const formatSize = (bytes: number): string => {
    if (bytes === 0) return "0 B";
    const sizes = ["B", "KB", "MB", "GB"];
    const i = Math.floor(Math.log(bytes) / Math.log(1024));
    return `${(bytes / Math.pow(1024, i)).toFixed(1)} ${sizes[i]}`;
  };

  const navigateUp = () => {
    const parts = browse.path.split("/");
    if (parts.length > 1) {
      parts.pop();
      const newPath = parts.join("/") || ".";
      setBrowse(prev => ({ ...prev, path: newPath, parent: newPath !== "." }));
      onNavigate?.(newPath);
    }
  };

  const handleEntryClick = (entry: FileEntry) => {
    if (entry.type === "directory") {
      const newPath = browse.path === "." ? entry.name : `${browse.path}/${entry.name}`;
      setBrowse(prev => ({ ...prev, path: newPath, parent: true, selected: null }));
      onNavigate?.(newPath);
    } else {
      setBrowse(prev => ({ ...prev, selected: entry.name }));
      onFileSelect?.(entry);
    }
  };

  return (
~w  );
};
', [Template]).

%! generate_button_showcase(-Code) is det
%  Generate a showcase of button variants.
generate_button_showcase(Code) :-
    Code =
'import React from "react";

// Generated from: ui_primitives button variants

interface ButtonProps {
  variant?: "primary" | "secondary" | "danger" | "ghost";
  disabled?: boolean;
  loading?: boolean;
  children: React.ReactNode;
  onClick?: () => void;
}

export const Button: React.FC<ButtonProps> = ({
  variant = "primary",
  disabled = false,
  loading = false,
  children,
  onClick,
}) => {
  const baseStyles: React.CSSProperties = {
    padding: "0.5rem 1rem",
    borderRadius: "var(--border-radius-md, 6px)",
    border: "none",
    cursor: disabled ? "not-allowed" : "pointer",
    fontWeight: 500,
    fontSize: "0.875rem",
    display: "inline-flex",
    alignItems: "center",
    gap: "0.5rem",
    transition: "all 0.15s ease",
    opacity: disabled ? 0.5 : 1,
  };

  const variantStyles: Record<string, React.CSSProperties> = {
    primary: {
      background: "var(--color-primary, #3b82f6)",
      color: "white",
    },
    secondary: {
      background: "transparent",
      color: "var(--color-text-primary, #1f2937)",
      border: "1px solid var(--color-border, #e5e7eb)",
    },
    danger: {
      background: "var(--color-danger, #ef4444)",
      color: "white",
    },
    ghost: {
      background: "transparent",
      color: "var(--color-text-primary, #1f2937)",
    },
  };

  return (
    <button
      style={{ ...baseStyles, ...variantStyles[variant] }}
      disabled={disabled || loading}
      onClick={onClick}
    >
      {loading && (
        <svg
          width="16"
          height="16"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          style={{ animation: "spin 1s linear infinite" }}
        >
          <circle cx="12" cy="12" r="10" strokeOpacity="0.25" />
          <path d="M12 2a10 10 0 0 1 10 10" />
        </svg>
      )}
      {children}
    </button>
  );
};

export const ButtonShowcase: React.FC = () => (
  <div style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>
    <div style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap" }}>
      <Button variant="primary">Primary</Button>
      <Button variant="secondary">Secondary</Button>
      <Button variant="danger">Danger</Button>
      <Button variant="ghost">Ghost</Button>
    </div>
    <div style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap" }}>
      <Button variant="primary" disabled>Disabled</Button>
      <Button variant="primary" loading>Loading</Button>
    </div>
    <style>{`
      @keyframes spin {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
      }
    `}</style>
  </div>
);
'.

%! generate_input_showcase(-Code) is det
%  Generate a showcase of input components.
generate_input_showcase(Code) :-
    Code =
'import React, { useState } from "react";

// Generated from: ui_primitives input variants

interface TextInputProps {
  label?: string;
  placeholder?: string;
  type?: "text" | "password" | "email";
  value?: string;
  onChange?: (value: string) => void;
  error?: string;
}

export const TextInput: React.FC<TextInputProps> = ({
  label,
  placeholder,
  type = "text",
  value,
  onChange,
  error,
}) => {
  const inputStyles: React.CSSProperties = {
    width: "100%",
    padding: "0.5rem 0.75rem",
    fontSize: "0.875rem",
    border: `1px solid ${error ? "var(--color-danger, #ef4444)" : "var(--color-border, #e5e7eb)"}`,
    borderRadius: "var(--border-radius-md, 6px)",
    background: "var(--color-surface, white)",
    color: "var(--color-text-primary, #1f2937)",
    outline: "none",
    transition: "border-color 0.15s ease, box-shadow 0.15s ease",
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "0.25rem" }}>
      {label && (
        <label style={{ fontSize: "0.875rem", fontWeight: 500, color: "var(--color-text-secondary, #6b7280)" }}>
          {label}
        </label>
      )}
      <input
        type={type}
        placeholder={placeholder}
        value={value}
        onChange={(e) => onChange?.(e.target.value)}
        style={inputStyles}
      />
      {error && (
        <span style={{ fontSize: "0.75rem", color: "var(--color-danger, #ef4444)" }}>
          {error}
        </span>
      )}
    </div>
  );
};

export const InputShowcase: React.FC = () => {
  const [values, setValues] = useState({ name: "", email: "", password: "" });

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "1rem", maxWidth: "300px" }}>
      <TextInput
        label="Name"
        placeholder="Enter your name"
        value={values.name}
        onChange={(v) => setValues(prev => ({ ...prev, name: v }))}
      />
      <TextInput
        label="Email"
        type="email"
        placeholder="you@example.com"
        value={values.email}
        onChange={(v) => setValues(prev => ({ ...prev, email: v }))}
      />
      <TextInput
        label="Password"
        type="password"
        placeholder="••••••••"
        value={values.password}
        onChange={(v) => setValues(prev => ({ ...prev, password: v }))}
      />
      <TextInput
        label="With Error"
        placeholder="Invalid input"
        error="This field is required"
      />
    </div>
  );
};
'.

% ============================================================================
% MAIN
% ============================================================================

write_component(Path, Code) :-
    format('Writing: ~w~n', [Path]),
    open(Path, write, Stream),
    write(Stream, Code),
    close(Stream).

main :-
    % Generate components
    generate_button_showcase(ButtonCode),
    generate_input_showcase(InputCode),

    % Write to files
    write_component('src/components/generated/Button.tsx', ButtonCode),
    write_component('src/components/generated/TextInput.tsx', InputCode),

    format('~nGenerated components successfully!~n'),
    format('Run "npm run storybook" to view them.~n'),
    halt(0).

:-
    % Ensure output directory exists
    (exists_directory('src/components/generated') -> true ; make_directory('src/components/generated')),
    main.
