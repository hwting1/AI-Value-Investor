from dataclasses import dataclass


@dataclass
class AgentConfig:
    research_model: str = "google_genai:gemma-4-31b-it"
    assistant_model: str = "google_genai:gemma-4-31b-it"

agent_config = AgentConfig()
