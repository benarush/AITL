import os
from typing import Optional

ENV_TRELLAR_API_KEY = "TRELLAR_API_KEY"

DEFAULT_ENDPOINT = "https://api.trellar.io/"
# DEFAULT_ENDPOINT = "http://localhost:8001"

def get_env_api_key() -> Optional[str]:
    return os.getenv(ENV_TRELLAR_API_KEY)
