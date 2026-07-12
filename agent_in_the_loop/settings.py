import os
from typing import Optional

ENV_AITL_API_KEY = "AGENT_IN_THE_LOOP_API_KEY"

DEFAULT_ENDPOINT = "https://api.trellar.io/"

def get_env_api_key() -> Optional[str]:
    return os.getenv(ENV_AITL_API_KEY)
