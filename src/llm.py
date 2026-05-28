from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

from config import agent_config
from ingest import load_config

load_dotenv()
load_config()

base_llm = init_chat_model(agent_config.research_model, temperature=0)
assistant_llm = init_chat_model(agent_config.assistant_model, temperature=0)
