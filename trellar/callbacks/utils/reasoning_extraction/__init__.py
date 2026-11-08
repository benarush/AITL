"""Provider-aware extraction of a model's reasoning / thinking settings."""
from __future__ import annotations

# Import the provider modules before the dispatcher so ``from . import extract_*`` inside
# ``extract_reasoning`` resolves to the modules (the function below has a different name).
from . import extract_anthropic, extract_gemini, extract_generic, extract_openai  # noqa: F401
from .extract_reasoning import extract_reasoning

__all__ = ["extract_reasoning"]
