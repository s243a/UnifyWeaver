import subprocess
import sys
import shutil

class LLMProvider:
    def __init__(self, model="gemini-2.0-flash-exp"):
        self.model = model
        self.cli_path = shutil.which("gemini")

    def ask(self, prompt, context=None):
        if not self.cli_path:
            return "Error: gemini CLI not found in PATH"

        full_prompt = prompt
        if context:
            full_prompt += "\n\nContext:\n" + str(context)
            
        cmd = [self.cli_path, "-p", full_prompt, "--model", self.model]
        
        try:
            # Run CLI (capture output)
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                return f"Error running gemini: {result.stderr}"
            return result.stdout.strip()
        except Exception as e:
            return f"Exception calling LLM: {e}"