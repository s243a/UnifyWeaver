use std::process::Command;
use serde_json::Value;

pub struct LLMProvider {
    model: String,
}

impl LLMProvider {
    pub fn new(model: &str) -> Self {
        Self {
            model: model.to_string(),
        }
    }

    pub fn ask(&self, prompt: &str, context: &Vec<Value>) -> Result<String, Box<dyn std::error::Error>> {
        let context_str = serde_json::to_string_pretty(context)?;
        let full_prompt = format!("{}\n\nContext:\n{}", prompt, context_str);

        // Call gemini CLI
        let output = Command::new("gemini")
            .arg("-p")
            .arg(&full_prompt)
            .arg("--model")
            .arg(&self.model)
            .output()?;

        if output.status.success() {
            Ok(String::from_utf8_lossy(&output.stdout).to_string())
        } else {
            let err = String::from_utf8_lossy(&output.stderr);
            Err(format!("LLM Error: {}", err).into())
        }
    }
}

