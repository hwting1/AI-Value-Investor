from dataclasses import dataclass


@dataclass
class AgentConfig:
    research_model: str = "google_genai:gemini-3-flash-preview"
    assistant_model: str = "google_genai:gemini-3.1-flash-lite-preview"

agent_config = AgentConfig()
