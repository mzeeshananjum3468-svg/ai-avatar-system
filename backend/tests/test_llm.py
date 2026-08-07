import pytest

from app.services.llm import _cacheable_system, _merge_system_prompts, DEFAULT_SYSTEM_PROMPT


def test_merge_system_prompts_appends_avatar_prompt():
    avatar_prompt = "You are a pirate."
    merged = _merge_system_prompts(avatar_prompt)

    assert merged.startswith(DEFAULT_SYSTEM_PROMPT.strip())
    assert merged.endswith(avatar_prompt)
    assert "You are a pirate." in merged


def test_cacheable_system_uses_default_and_cache_control():
    avatar_prompt = "You are a pirate."
    blocks = _cacheable_system(avatar_prompt)

    assert isinstance(blocks, list)
    assert len(blocks) == 1
    assert blocks[0]["type"] == "text"
    assert blocks[0]["text"].startswith(DEFAULT_SYSTEM_PROMPT.strip())
    assert blocks[0]["text"].endswith(avatar_prompt)
    assert blocks[0]["cache_control"] == {"type": "ephemeral"}
