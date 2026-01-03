import React from "react";

export interface ResponsiveLayoutProps {
  columns?: {
    mobile: number;
    tablet: number;
    desktop: number;
  };
  gap?: string;
  children?: React.ReactNode;
}

/**
 * ResponsiveLayout - Generated from layout_generator.pl
 *
 * Demonstrates responsive grid layouts with breakpoint-based column configuration.
 * Uses CSS custom properties for theme integration.
 */
export const ResponsiveLayout: React.FC<ResponsiveLayoutProps> = ({
  columns = { mobile: 1, tablet: 2, desktop: 4 },
  gap = "1rem",
  children,
}) => {
  return (
    <div
      className="responsive-layout"
      style={{
        display: "grid",
        gap,
        gridTemplateColumns: `repeat(var(--layout-columns, ${columns.desktop}), 1fr)`,
      }}
    >
      <style>{`
        .responsive-layout {
          --layout-columns: ${columns.desktop};
        }
        @media (max-width: 1024px) {
          .responsive-layout {
            --layout-columns: ${columns.tablet};
          }
        }
        @media (max-width: 640px) {
          .responsive-layout {
            --layout-columns: ${columns.mobile};
          }
        }
      `}</style>
      {children}
    </div>
  );
};

export interface ResponsiveCardProps {
  title: string;
  value: string | number;
  trend?: number;
  color?: string;
}

/**
 * ResponsiveCard - A card that adapts to container size
 */
export const ResponsiveCard: React.FC<ResponsiveCardProps> = ({
  title,
  value,
  trend,
  color = "var(--color-primary)",
}) => {
  return (
    <div
      style={{
        padding: "1.5rem",
        background: "var(--color-surface)",
        borderRadius: "var(--border-radius-lg)",
        boxShadow: "var(--shadow-sm)",
        border: "1px solid var(--color-border)",
        transition: "all var(--transition-normal)",
      }}
    >
      <h4
        style={{
          margin: "0 0 0.5rem 0",
          fontSize: "0.875rem",
          fontWeight: 500,
          color: "var(--color-text-secondary)",
        }}
      >
        {title}
      </h4>
      <div
        style={{
          fontSize: "1.75rem",
          fontWeight: 700,
          color: "var(--color-text-primary)",
        }}
      >
        {value}
      </div>
      {trend !== undefined && (
        <div
          style={{
            marginTop: "0.5rem",
            fontSize: "0.75rem",
            color: trend >= 0 ? "#22c55e" : "#ef4444",
            display: "flex",
            alignItems: "center",
            gap: "0.25rem",
          }}
        >
          <span>{trend >= 0 ? "↑" : "↓"}</span>
          <span>{Math.abs(trend)}%</span>
        </div>
      )}
    </div>
  );
};

export interface BreakpointIndicatorProps {
  showLabel?: boolean;
}

/**
 * BreakpointIndicator - Shows the current responsive breakpoint
 */
export const BreakpointIndicator: React.FC<BreakpointIndicatorProps> = ({
  showLabel = true,
}) => {
  const [breakpoint, setBreakpoint] = React.useState("desktop");

  React.useEffect(() => {
    const updateBreakpoint = () => {
      const width = window.innerWidth;
      if (width <= 640) {
        setBreakpoint("mobile");
      } else if (width <= 1024) {
        setBreakpoint("tablet");
      } else {
        setBreakpoint("desktop");
      }
    };

    updateBreakpoint();
    window.addEventListener("resize", updateBreakpoint);
    return () => window.removeEventListener("resize", updateBreakpoint);
  }, []);

  const colors: Record<string, string> = {
    mobile: "#ef4444",
    tablet: "#f59e0b",
    desktop: "#22c55e",
  };

  const icons: Record<string, string> = {
    mobile: "📱",
    tablet: "📱",
    desktop: "🖥️",
  };

  return (
    <div
      style={{
        display: "inline-flex",
        alignItems: "center",
        gap: "0.5rem",
        padding: "0.5rem 1rem",
        background: colors[breakpoint] + "20",
        color: colors[breakpoint],
        borderRadius: "var(--border-radius-full)",
        fontSize: "0.875rem",
        fontWeight: 600,
        border: `2px solid ${colors[breakpoint]}`,
      }}
    >
      <span>{icons[breakpoint]}</span>
      {showLabel && <span>{breakpoint.toUpperCase()}</span>}
    </div>
  );
};

export interface ContainerQueryDemoProps {
  children?: React.ReactNode;
}

/**
 * ContainerQueryDemo - Demonstrates container queries for component-based responsiveness
 */
export const ContainerQueryDemo: React.FC<ContainerQueryDemoProps> = ({
  children,
}) => {
  return (
    <div
      className="container-query-demo"
      style={{
        containerType: "inline-size",
        containerName: "card-container",
        resize: "horizontal",
        overflow: "auto",
        minWidth: "200px",
        maxWidth: "100%",
        padding: "1rem",
        border: "2px dashed var(--color-border)",
        borderRadius: "var(--border-radius-md)",
      }}
    >
      <style>{`
        @container card-container (min-width: 400px) {
          .adaptive-content {
            flex-direction: row !important;
          }
          .adaptive-content .content-text {
            text-align: left !important;
          }
        }
        @container card-container (max-width: 399px) {
          .adaptive-content {
            flex-direction: column !important;
          }
          .adaptive-content .content-text {
            text-align: center !important;
          }
        }
      `}</style>
      <div
        className="adaptive-content"
        style={{
          display: "flex",
          alignItems: "center",
          gap: "1rem",
          padding: "1rem",
          background: "var(--color-surface)",
          borderRadius: "var(--border-radius-md)",
          transition: "all var(--transition-normal)",
        }}
      >
        <div
          style={{
            width: "60px",
            height: "60px",
            background: "var(--color-primary)",
            borderRadius: "var(--border-radius-md)",
            flexShrink: 0,
          }}
        />
        <div className="content-text" style={{ flex: 1 }}>
          <h4 style={{ margin: "0 0 0.25rem 0", color: "var(--color-text-primary)" }}>
            Adaptive Card
          </h4>
          <p
            style={{
              margin: 0,
              fontSize: "0.875rem",
              color: "var(--color-text-secondary)",
            }}
          >
            Resize the container to see layout adapt using container queries.
          </p>
        </div>
      </div>
      <div
        style={{
          marginTop: "0.5rem",
          textAlign: "center",
          fontSize: "0.75rem",
          color: "var(--color-text-secondary)",
        }}
      >
        ↔️ Drag edge to resize
      </div>
    </div>
  );
};
