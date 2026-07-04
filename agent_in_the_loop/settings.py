import os
from typing import Optional

ENV_AITL_API_KEY = "AGENT_IN_THE_LOOP_API_KEY"

# Fixed backend domain. Not configurable via env var or caller-supplied
# argument so clients can't point the SDK at an arbitrary/untrusted host.
AITL_ENDPOINT = "https://trellar.io"


def get_env_api_key() -> Optional[str]:
    return os.getenv(ENV_AITL_API_KEY)
