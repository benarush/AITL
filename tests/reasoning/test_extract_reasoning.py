"""Unit tests for ``extract_reasoning`` and its provider modules."""
from __future__ import annotations

import pytest

from trellar.callbacks.utils import extract_reasoning
from trellar.callbacks.utils.reasoning_extraction import (
    extract_anthropic,
    extract_gemini,
    extract_generic,
    extract_openai,
)
from trellar.callbacks.utils.reasoning_extraction._helpers import flatten_sources, pick_keys


class TestProviders:
    def test_anthropic_thinking_returned_as_is(self, anthropic_config, anthropic_thinking):
        result = extract_reasoning([anthropic_config], "ChatAnthropic", "claude-sonnet-4-5")

        assert result == {"thinking": anthropic_thinking}
        assert result["thinking"] is anthropic_thinking  # same object, untouched

    def test_gemini_flat_options(self, gemini_config):
        result = extract_reasoning([gemini_config], ["langchain", "ChatGoogleGenerativeAI"], "gemini-2.5-flash")

        assert result == {"thinking_budget": 1024, "include_thoughts": True}

    def test_gemini_nested_thinking_config_from_strands_params(self, strands_gemini_config):
        result = extract_reasoning([strands_gemini_config], "strands.models.gemini GeminiModel", "gemini-2.5-flash")

        assert result == {"thinking_config": {"thinking_budget": 512}}

    def test_openai_reasoning_effort(self, openai_config):
        result = extract_reasoning([openai_config], "ChatOpenAI", "o3-mini")

        assert result == {"reasoning_effort": "high"}

    def test_openai_nested_reasoning_object(self):
        reasoning = {"effort": "high", "summary": "auto"}

        assert extract_reasoning([{"reasoning": reasoning}], "ChatOpenAI", "gpt-5") == {"reasoning": reasoning}

    def test_claude_on_bedrock_found_in_additional_request_fields(self, bedrock_claude_config, anthropic_thinking):
        result = extract_reasoning([bedrock_claude_config], "strands.models.bedrock BedrockModel")

        assert result == {"thinking": anthropic_thinking}

    def test_model_name_alone_is_enough_to_pick_the_provider(self, anthropic_thinking):
        assert extract_reasoning([{"thinking": anthropic_thinking}], None, "claude-opus-4") == {
            "thinking": anthropic_thinking
        }

    def test_provider_keys_of_other_providers_are_not_picked(self):
        # An Anthropic model ignores OpenAI/Gemini-style keys.
        config = {"thinking": {"type": "enabled"}, "reasoning_effort": "high", "thinking_budget": 5}

        assert extract_reasoning([config], "ChatAnthropic", "claude-sonnet-4-5") == {"thinking": {"type": "enabled"}}


class TestMissingAndEmpty:
    @pytest.mark.parametrize("model", ["claude-sonnet-4-5", "gemini-2.5-flash", "gpt-4o", "mystery-model"])
    def test_no_reasoning_keys_returns_none(self, model):
        assert extract_reasoning([{"model": model, "temperature": 0}], None, model) is None

    def test_none_value_counts_as_not_set(self):
        assert extract_reasoning([{"thinking": None, "reasoning_effort": None}], None, "claude") is None

    def test_falsy_but_real_values_are_kept(self):
        # include_thoughts=False and a 0 budget are meaningful settings, not "missing".
        result = extract_reasoning([{"include_thoughts": False, "thinking_budget": 0}], None, "gemini-2.5")

        assert result == {"include_thoughts": False, "thinking_budget": 0}

    @pytest.mark.parametrize("sources", [None, [], [None], [None, "text", 5, ["a"]]])
    def test_bad_or_empty_sources_return_none(self, sources):
        assert extract_reasoning(sources, "ChatAnthropic", "claude") is None


class TestSourcePrecedence:
    def test_later_source_overrides_earlier(self):
        constructor = {"reasoning_effort": "low"}
        per_call = {"reasoning_effort": "high"}

        assert extract_reasoning([constructor, per_call], "ChatOpenAI", "gpt-5") == {"reasoning_effort": "high"}

    def test_sources_are_combined(self):
        result = extract_reasoning(
            [{"thinking_budget": 100}, {"include_thoughts": True}], "ChatGoogleGenerativeAI", "gemini-2.5"
        )

        assert result == {"thinking_budget": 100, "include_thoughts": True}

    def test_model_kwargs_and_extra_body_are_searched(self):
        assert extract_reasoning([{"model_kwargs": {"reasoning_effort": "low"}}], "ChatOpenAI", "gpt-5") == {
            "reasoning_effort": "low"
        }
        assert extract_reasoning([{"extra_body": {"reasoning": {"effort": "low"}}}], "ChatOpenAI", "gpt-5") == {
            "reasoning": {"effort": "low"}
        }


class TestSafety:
    @pytest.mark.parametrize(
        "config_fixture, hint, model",
        [
            ("anthropic_config", "ChatAnthropic", "claude-sonnet-4-5"),
            ("gemini_config", "ChatGoogleGenerativeAI", "gemini-2.5-flash"),
            ("openai_config", "ChatOpenAI", "o3-mini"),
            ("strands_gemini_config", "GeminiModel", "gemini-2.5-flash"),
            ("bedrock_claude_config", "BedrockModel", "anthropic.claude"),
        ],
    )
    def test_secrets_never_leak(self, request, secret_fields, config_fixture, hint, model):
        config = request.getfixturevalue(config_fixture)

        result = extract_reasoning([config], hint, model)

        assert result  # something was extracted...
        assert not set(result) & set(secret_fields)  # ...but never a secret key
        assert "sk-super-secret" not in repr(result)
        assert "top-secret" not in repr(result)

    def test_unknown_provider_uses_generic_fallback_and_still_skips_secrets(self, secret_fields):
        config = {"think": True, "temperature": 1, **secret_fields}

        assert extract_reasoning([config], "SomeCustomChatModel", "mystery") == {"think": True}

    def test_extraction_error_returns_none(self, monkeypatch):
        def boom(_sources):
            raise RuntimeError("provider bug")

        monkeypatch.setattr(extract_anthropic, "extract", boom)

        assert extract_reasoning([{"thinking": {}}], "ChatAnthropic", "claude") is None

    def test_hostile_mapping_returns_none(self):
        class Hostile(dict):
            def get(self, *_args, **_kwargs):
                raise RuntimeError("no access")

        assert extract_reasoning([Hostile(thinking={})], "ChatAnthropic", "claude") is None


class TestGenericFallbackAndMerging:
    def test_generic_fallback_reads_any_known_key(self):
        config = {"reasoning_effort": "low", "think": True, "reasoningConfig": {"type": "enabled"}}

        assert extract_reasoning([config], "ChatOllama", "llama3") == config

    def test_several_matching_providers_are_merged(self):
        # e.g. an OpenAI-compatible endpoint serving Claude.
        config = {"thinking": {"type": "enabled"}, "reasoning_effort": "low"}

        assert extract_reasoning([config], "ChatOpenAI", "claude-via-gateway") == config

    @pytest.mark.parametrize(
        "hint, module",
        [
            ("anthropic", extract_anthropic),
            ("claude-3", extract_anthropic),
            ("gemini-2.5", extract_gemini),
            ("vertexai", extract_gemini),
            ("openai", extract_openai),
            ("gpt-5", extract_openai),
            ("o1-mini", extract_openai),
            ("o4", extract_openai),
        ],
    )
    def test_matches(self, hint, module):
        assert module.matches(hint) is True

    @pytest.mark.parametrize("hint", ["", "llama3", "mistral-large", "falcon", "bedrock"])
    def test_matches_rejects_unrelated(self, hint):
        assert not any(m.matches(hint) for m in (extract_anthropic, extract_gemini, extract_openai))
        assert extract_generic.matches(hint) is False


class TestHelpers:
    def test_flatten_sources_keeps_order_and_drops_non_mappings(self):
        outer = {"params": {"a": 1}, "x": 1}
        flat = flatten_sources([None, outer, "nope"])

        assert flat == [outer, {"a": 1}]

    def test_pick_keys_only_returns_allowlisted(self):
        assert pick_keys([{"a": 1, "secret": "x"}], ("a", "b")) == {"a": 1}
        assert pick_keys([{"secret": "x"}], ("a",)) is None
