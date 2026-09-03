"""Unit tests for the module-level helper functions in
trellar/callbacks/langchain_callback.py -- previously 0% covered.
"""
from __future__ import annotations

import json

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from trellar.callbacks.langchain_callback import (
    _compact_json,
    _content_to_str,
    _extract_llm_input,
    _extract_model_name,
    _serialize_message,
)


# ---------------------------------------------------------------------------
# _content_to_str
# ---------------------------------------------------------------------------

class TestContentToStr:
    def test_plain_string_passthrough(self):
        assert _content_to_str("hello") == "hello"

    def test_empty_string_passthrough(self):
        assert _content_to_str("") == ""

    def test_multimodal_list_of_dicts_is_json_encoded(self):
        content = [{"type": "text", "text": "hi"}, {"type": "image_url", "image_url": "http://x"}]
        result = _content_to_str(content)
        assert json.loads(result) == content

    def test_dict_content_is_json_encoded(self):
        assert json.loads(_content_to_str({"a": 1})) == {"a": 1}

    def test_non_serializable_object_uses_default_str_encoding(self):
        class Weird:
            def __str__(self):
                return "weird-repr"

        # json.dumps(..., default=str) handles arbitrary objects without raising.
        assert _content_to_str(Weird()) == '"weird-repr"'

    def test_circular_structure_falls_back_to_str(self):
        """json.dumps raises ValueError for a self-referencing container even
        with default=str (cycle detection happens before default() is ever
        called on a leaf value); the except branch falls back to str()."""
        circular: list = []
        circular.append(circular)
        assert _content_to_str(circular) == str(circular)


# ---------------------------------------------------------------------------
# _extract_llm_input
# ---------------------------------------------------------------------------

class TestExtractLlmInput:
    def test_empty_messages_returns_both_none(self):
        assert _extract_llm_input([]) == {"system": None, "human": None}

    def test_system_and_human_from_base_messages(self):
        messages = [SystemMessage(content="sys"), HumanMessage(content="hi")]
        assert _extract_llm_input(messages) == {"system": "sys", "human": "hi"}

    def test_first_system_wins_across_multiple_system_messages(self):
        messages = [
            SystemMessage(content="first"),
            SystemMessage(content="second"),
            HumanMessage(content="hi"),
        ]
        assert _extract_llm_input(messages)["system"] == "first"

    def test_last_human_wins_across_multi_turn_history(self):
        messages = [
            SystemMessage(content="sys"),
            HumanMessage(content="turn 1"),
            AIMessage(content="reply 1"),
            HumanMessage(content="turn 2"),
        ]
        assert _extract_llm_input(messages)["human"] == "turn 2"

    def test_plain_dict_messages_with_role_key(self):
        messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]
        assert _extract_llm_input(messages) == {"system": "sys", "human": "hi"}

    def test_plain_dict_messages_with_type_key(self):
        messages = [{"type": "system", "content": "sys"}, {"type": "human", "content": "hi"}]
        assert _extract_llm_input(messages) == {"system": "sys", "human": "hi"}

    def test_user_role_is_synonym_for_human(self):
        messages = [{"role": "user", "content": "hi there"}]
        assert _extract_llm_input(messages)["human"] == "hi there"

    def test_mixed_base_message_and_dict(self):
        messages = [SystemMessage(content="sys"), {"role": "user", "content": "hi"}]
        assert _extract_llm_input(messages) == {"system": "sys", "human": "hi"}

    def test_unrecognized_entries_are_skipped(self):
        messages = ["a plain string", 42, HumanMessage(content="hi")]
        assert _extract_llm_input(messages)["human"] == "hi"

    def test_ai_and_tool_messages_are_ignored(self):
        messages = [
            HumanMessage(content="hi"),
            AIMessage(content="reply"),
            ToolMessage(content="result", tool_call_id="1"),
        ]
        result = _extract_llm_input(messages)
        assert result["human"] == "hi"
        assert result["system"] is None

    def test_dict_message_with_missing_content_defaults_to_empty_string(self):
        messages = [{"role": "user"}]
        assert _extract_llm_input(messages)["human"] == ""


# ---------------------------------------------------------------------------
# _compact_json
# ---------------------------------------------------------------------------

class TestCompactJson:
    def test_dict_is_json_encoded(self):
        assert json.loads(_compact_json({"a": 1})) == {"a": 1}

    def test_list_is_json_encoded(self):
        assert json.loads(_compact_json([1, 2, 3])) == [1, 2, 3]

    def test_non_serializable_object_uses_default_str_fallback(self):
        class Custom:
            def __str__(self):
                return "custom-val"

        assert _compact_json(Custom()) == '"custom-val"'

    def test_circular_reference_falls_back_to_repr(self):
        circular: dict = {}
        circular["self"] = circular
        assert _compact_json(circular) == repr(circular)


# ---------------------------------------------------------------------------
# _extract_model_name
# ---------------------------------------------------------------------------

class TestExtractModelName:
    def test_model_name_kwarg_takes_priority(self):
        serialized = {
            "name": "ChatOpenAI",
            "kwargs": {"model_name": "gpt-4o", "model": "should-not-be-used"},
        }
        assert _extract_model_name(serialized) == "gpt-4o"

    def test_falls_back_to_model_kwarg(self):
        serialized = {"name": "ChatOpenAI", "kwargs": {"model": "gpt-4o-mini"}}
        assert _extract_model_name(serialized) == "gpt-4o-mini"

    def test_falls_back_to_class_name_when_kwargs_empty(self):
        serialized = {"name": "ChatGoogleGenerativeAI", "kwargs": {}}
        assert _extract_model_name(serialized) == "ChatGoogleGenerativeAI"

    def test_falls_back_to_class_name_when_kwargs_missing_entirely(self):
        serialized = {"name": "ChatAnthropic"}
        assert _extract_model_name(serialized) == "ChatAnthropic"

    def test_returns_none_when_nothing_available(self):
        assert _extract_model_name({}) is None


# ---------------------------------------------------------------------------
# _serialize_message (module-level, dict-returning)
# ---------------------------------------------------------------------------
# NOTE: this function is currently unused anywhere in the codebase --
# `_AgentGuardCallback` uses its own static methods (`_serialize_message_obj`
# / `_serialize_messages`) instead, which produce labeled strings rather than
# dicts. Covered here so a future reintroduction doesn't silently regress.

class TestSerializeMessageDeadCode:
    def test_non_message_object_falls_back_to_raw(self):
        assert _serialize_message("just a string") == {"raw": "just a string"}

    def test_basic_message_extracts_role_and_content(self):
        result = _serialize_message(HumanMessage(content="hi"))
        assert result["role"] == "human"
        assert result["content"] == "hi"
        assert "additional_kwargs" not in result
        assert "tool_calls" not in result

    def test_additional_kwargs_included_when_present(self):
        msg = AIMessage(content="hi", additional_kwargs={"function_call": {"name": "f"}})
        result = _serialize_message(msg)
        assert result["additional_kwargs"] == {"function_call": {"name": "f"}}

    def test_tool_calls_included_when_present(self):
        msg = AIMessage(content="", tool_calls=[{"name": "f", "args": {}, "id": "1"}])
        result = _serialize_message(msg)
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["name"] == "f"
        assert result["tool_calls"][0]["id"] == "1"
