# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Unit tests for Memory Systems Lab (Day 17 - Track 03).

Tests 1 & 2: Pure local logic – no LLM needed (fast).
Tests 3 & 4: Integration tests that call the *real* LLM via the .env config.
"""

import time
from pathlib import Path

import pytest

from config import load_config
from memory_store import CompactMemoryManager, UserProfileStore, estimate_tokens


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def safe_print(label: str, text: str) -> None:
    """Print text safely on Windows cp1252 terminals."""
    safe = text.encode("ascii", errors="replace").decode("ascii")
    print(f"{label}: {safe}")


def real_config(tmp_path: Path):
    """Load the real config from .env, but redirect state/data to tmp_path."""
    cfg = load_config()
    from dataclasses import replace
    cfg = replace(
        cfg,
        state_dir=tmp_path / "state",
        data_dir=tmp_path / "data",
        compact_threshold_tokens=150,   # low enough to trigger after ~3-4 turns
        compact_keep_messages=2,
    )
    (cfg.state_dir / "profiles").mkdir(parents=True, exist_ok=True)
    return cfg


# ---------------------------------------------------------------------------
# Test 1: User.md CRUD – pure local, no LLM
# ---------------------------------------------------------------------------

def test_user_markdown_read_write_edit(tmp_path: Path) -> None:
    """Verify User.md can be created, updated, and edited in-place."""
    store = UserProfileStore(tmp_path)
    user_id = "test_user"

    # Read default (file should not exist yet → returns template)
    text = store.read_text(user_id)
    assert "# User Profile" in text

    # Write
    content = "# User Profile\n- Tên: Alice\n- Nơi ở: Paris"
    store.write_text(user_id, content)
    assert store.read_text(user_id) == content
    assert store.file_size(user_id) > 0

    # Edit a single fact in-place
    changed = store.edit_text(user_id, "Nơi ở: Paris", "Nơi ở: London")
    assert changed is True
    assert "Nơi ở: London" in store.read_text(user_id)


# ---------------------------------------------------------------------------
# Test 2: CompactMemoryManager trigger – pure local, no LLM
# ---------------------------------------------------------------------------

def test_compact_trigger(tmp_path: Path) -> None:
    """Verify that compaction fires once total tokens exceed the threshold."""
    manager = CompactMemoryManager(threshold_tokens=10, keep_messages=2)
    thread_id = "test_thread"

    manager.append(thread_id, "user", "Hello first message")
    ctx = manager.context(thread_id)
    assert len(ctx["messages"]) == 1
    assert ctx["compactions"] == 0

    manager.append(thread_id, "assistant", "Hi there!")
    ctx = manager.context(thread_id)
    assert len(ctx["messages"]) == 2
    assert ctx["compactions"] == 0

    manager.append(thread_id, "user", "What is your name?")
    ctx = manager.context(thread_id)
    assert len(ctx["messages"]) == 3
    assert ctx["compactions"] == 0

    # 4th message pushes total tokens > threshold AND len > keep_messages
    manager.append(thread_id, "assistant", "I am a chatbot.")
    ctx = manager.context(thread_id)
    assert len(ctx["messages"]) == 2        # only keep_messages=2 remain
    assert ctx["compactions"] == 1
    assert "Tóm tắt hội thoại cũ" in str(ctx["summary"])


# ---------------------------------------------------------------------------
# Test 3: Cross-session recall – calls REAL LLM
# ---------------------------------------------------------------------------

def test_cross_session_recall(tmp_path: Path) -> None:
    """Advanced agent remembers a fact across sessions; baseline does not."""
    from agent_advanced import AdvancedAgent
    from agent_baseline import BaselineAgent

    cfg = real_config(tmp_path)

    baseline = BaselineAgent(cfg)
    advanced = AdvancedAgent(cfg)

    user_id = "user_recall"
    thread_1 = "thread_1"
    thread_2 = "thread_2"

    # --- Session 1: tell the agents a fact ---
    fact_msg = "My name is Alice and I live in Paris."
    print(f"\n[Session 1] Send: {fact_msg}")

    baseline.reply(user_id, thread_1, fact_msg)
    advanced.reply(user_id, thread_1, fact_msg)

    # --- Session 2 (new thread): ask about the fact ---
    question = "Ban co the nhac lai ten va noi o cua minh khong?"
    print(f"[Session 2] Question: {question}")

    ans_baseline = baseline.reply(user_id, thread_2, question)["response"]
    ans_advanced = advanced.reply(user_id, thread_2, question)["response"]

    safe_print("[Baseline response]", ans_baseline)
    safe_print("[Advanced response]", ans_advanced)

    # Baseline has no cross-session memory -> should NOT recall both Alice AND Paris
    assert "Alice" not in ans_baseline or "Paris" not in ans_baseline, (
        f"Baseline should NOT recall cross-session info.\nResponse: {ans_baseline}"
    )

    # Advanced persists User.md -> MUST recall both name and location
    assert "Alice" in ans_advanced, (
        f"Advanced must recall name 'Alice' from User.md.\nResponse: {ans_advanced}"
    )
    assert "Paris" in ans_advanced, (
        f"Advanced must recall location 'Paris' from User.md.\nResponse: {ans_advanced}"
    )


# ---------------------------------------------------------------------------
# Test 4: Compact reduces prompt load – calls REAL LLM
# ---------------------------------------------------------------------------

def test_compact_reduces_prompt_load_on_long_thread(tmp_path: Path) -> None:
    """Advanced agent's prompt token load stays lower than baseline on long threads."""
    from agent_advanced import AdvancedAgent
    from agent_baseline import BaselineAgent

    cfg = real_config(tmp_path)

    baseline = BaselineAgent(cfg)
    advanced = AdvancedAgent(cfg)

    user_id = "user_compact"
    thread_id = "long_thread"

    turns = [
        "News 1: Artemis III is scheduled to launch in 2027 with a crew of 4 people heading to lunar orbit.",
        "News 2: NASA X-59 aircraft reached Mach 1.1 and reduced sonic boom noise by 40 percent.",
        "News 3: El Nino 2026 is forecast to raise global temperatures by 0.3 degrees C according to WMO.",
        "News 4: BC Hydro launched new Power Smart program, saving electricity equivalent to 80,000 households.",
        "News 5: The UN warns sea levels could rise by 1 meter by 2100 without emissions reductions.",
    ]

    print("\n[Sending 5 turns to trigger compaction...]")
    for i, turn in enumerate(turns):
        print(f"  Turn {i+1}: {turn[:60]}...")
        baseline.reply(user_id, thread_id, turn)
        advanced.reply(user_id, thread_id, turn)

    # Advanced must have compacted at least once
    compact_count = advanced.compaction_count(thread_id)
    print(f"\n[Compaction count] Advanced: {compact_count}, Baseline: {baseline.compaction_count(thread_id)}")
    assert compact_count > 0, "Advanced agent must perform at least 1 compaction"
    assert baseline.compaction_count(thread_id) == 0, "Baseline must not compact"

    # Measure prompt load on the next turn
    next_turn = "Please briefly summarize the news you have heard."
    rep_baseline = baseline.reply(user_id, thread_id, next_turn)
    rep_advanced = advanced.reply(user_id, thread_id, next_turn)

    pt_baseline = rep_baseline["prompt_tokens"]
    pt_advanced = rep_advanced["prompt_tokens"]

    print(f"[Prompt tokens] Baseline: {pt_baseline}, Advanced: {pt_advanced}")

    assert pt_advanced < pt_baseline, (
        f"Advanced ({pt_advanced} tokens) must be less than Baseline ({pt_baseline} tokens)."
    )
