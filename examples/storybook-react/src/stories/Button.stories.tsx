import type { Meta, StoryObj } from "@storybook/react";
import { Button, ButtonShowcase } from "../components/generated/Button";

const meta: Meta<typeof Button> = {
  title: "Generated/Button",
  component: Button,
  tags: ["autodocs"],
  parameters: {
    docs: {
      description: {
        component: "Button component generated from UnifyWeaver ui_primitives.",
      },
    },
  },
  argTypes: {
    variant: {
      control: "select",
      options: ["primary", "secondary", "danger", "ghost"],
      description: "Visual style variant",
    },
    disabled: {
      control: "boolean",
      description: "Disable the button",
    },
    loading: {
      control: "boolean",
      description: "Show loading spinner",
    },
  },
};

export default meta;
type Story = StoryObj<typeof Button>;

export const Primary: Story = {
  args: {
    variant: "primary",
    children: "Primary Button",
  },
};

export const Secondary: Story = {
  args: {
    variant: "secondary",
    children: "Secondary Button",
  },
};

export const Danger: Story = {
  args: {
    variant: "danger",
    children: "Delete",
  },
};

export const Ghost: Story = {
  args: {
    variant: "ghost",
    children: "Ghost Button",
  },
};

export const Disabled: Story = {
  args: {
    variant: "primary",
    disabled: true,
    children: "Disabled",
  },
};

export const Loading: Story = {
  args: {
    variant: "primary",
    loading: true,
    children: "Saving...",
  },
};

export const Showcase: Story = {
  render: () => <ButtonShowcase />,
  parameters: {
    docs: {
      description: {
        story: "All button variants displayed together.",
      },
    },
  },
};
