import React from "react";

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
