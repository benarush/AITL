import os
from typing import Optional

ENV_AITL_ENDPOINT = "AGENT_IN_THE_LOOP_ENDPOINT"
ENV_AITL_API_KEY = "AGENT_IN_THE_LOOP_API_KEY"

DEFAULT_ENDPOINT = "http://localhost:6006"


def get_env_endpoint() -> str:
    return os.getenv(ENV_AITL_ENDPOINT, DEFAULT_ENDPOINT)


def get_env_api_key() -> Optional[str]:
    return os.getenv(ENV_AITL_API_KEY)
