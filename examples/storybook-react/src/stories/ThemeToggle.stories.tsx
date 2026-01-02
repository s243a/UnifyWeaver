import type { Meta, StoryObj } from "@storybook/react";
import { ThemeToggle, ThemeProvider } from "../components/ThemeToggle";

const meta: Meta<typeof ThemeToggle> = {
  title: "Theme/ThemeToggle",
  component: ThemeToggle,
  tags: ["autodocs"],
  decorators: [
    (Story) => (
      <ThemeProvider>
        <div
          style={{
            padding: "2rem",
            background: "var(--color-surface)",
            borderRadius: "var(--border-radius-md)",
            transition: "background var(--transition-normal)",
          }}
        >
          <Story />
        </div>
      </ThemeProvider>
    ),
  ],
  parameters: {
    docs: {
      description: {
        component: `
The ThemeToggle component provides light/dark theme switching functionality.

**Generated from:** \`generate_theme_toggle/2\` in \`theme_generator.pl\`

\`\`\`prolog
?- generate_theme_toggle([light, dark], Toggle).
\`\`\`
        `,
      },
    },
  },
};

export default meta;
type Story = StoryObj<typeof ThemeToggle>;

export const Default: Story = {
  args: {},
};

export const WithMultipleThemes: Story = {
  args: {
    themes: ["light", "dark"],
  },
  parameters: {
    docs: {
      description: {
        story: "With only two themes, displays as a toggle button.",
      },
    },
  },
};

export const InCard: Story = {
  render: () => (
    <ThemeProvider>
      <div
        style={{
          padding: "1.5rem",
          background: "var(--color-surface)",
          borderRadius: "var(--border-radius-md)",
          boxShadow: "var(--shadow-md)",
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          gap: "1rem",
          transition: "all var(--transition-normal)",
        }}
      >
        <div>
          <h3
            style={{
              margin: "0 0 0.5rem 0",
              color: "var(--color-text-primary)",
            }}
          >
            Theme Settings
          </h3>
          <p
            style={{
              margin: 0,
              fontSize: "0.875rem",
              color: "var(--color-text-secondary)",
            }}
          >
            Toggle between light and dark mode
          </p>
        </div>
        <ThemeToggle />
      </div>
    </ThemeProvider>
  ),
  parameters: {
    docs: {
      description: {
        story: "Example usage in a settings card.",
      },
    },
  },
};
