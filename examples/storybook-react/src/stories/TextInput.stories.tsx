import type { Meta, StoryObj } from "@storybook/react";
import { TextInput, InputShowcase } from "../components/generated/TextInput";

const meta: Meta<typeof TextInput> = {
  title: "Generated/TextInput",
  component: TextInput,
  tags: ["autodocs"],
  parameters: {
    docs: {
      description: {
        component: "Text input component generated from UnifyWeaver ui_primitives.",
      },
    },
  },
  argTypes: {
    type: {
      control: "select",
      options: ["text", "password", "email"],
      description: "Input type",
    },
    label: {
      control: "text",
      description: "Label text",
    },
    placeholder: {
      control: "text",
      description: "Placeholder text",
    },
    error: {
      control: "text",
      description: "Error message",
    },
  },
};

export default meta;
type Story = StoryObj<typeof TextInput>;

export const Default: Story = {
  args: {
    label: "Username",
    placeholder: "Enter username",
  },
};

export const WithValue: Story = {
  args: {
    label: "Email",
    type: "email",
    placeholder: "you@example.com",
    value: "user@example.com",
  },
};

export const Password: Story = {
  args: {
    label: "Password",
    type: "password",
    placeholder: "Enter password",
  },
};

export const WithError: Story = {
  args: {
    label: "Email",
    type: "email",
    placeholder: "you@example.com",
    error: "Please enter a valid email address",
  },
};

export const NoLabel: Story = {
  args: {
    placeholder: "Search...",
  },
};

export const Showcase: Story = {
  render: () => <InputShowcase />,
  parameters: {
    docs: {
      description: {
        story: "Interactive form with multiple input types.",
      },
    },
  },
};
