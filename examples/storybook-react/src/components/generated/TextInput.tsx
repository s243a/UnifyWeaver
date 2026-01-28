import React, { useState } from "react";

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
