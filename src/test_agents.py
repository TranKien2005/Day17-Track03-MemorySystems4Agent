from __future__ import annotations

from pathlib import Path

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import LabConfig, load_config
from model_provider import ProviderConfig


def make_config(tmp_path: Path) -> LabConfig:
    """Build an isolated config for tests."""
    dummy_model = ProviderConfig(
        provider="ollama",
        model_name="dummy",
        temperature=0.0
    )
    return LabConfig(
        base_dir=tmp_path,
        data_dir=tmp_path / "data",
        state_dir=tmp_path / "state",
        compact_threshold_tokens=20,  # small threshold so compaction happens quickly
        compact_keep_messages=2,
        model=dummy_model,
        judge_model=dummy_model
    )


def test_user_markdown_read_write_edit(tmp_path: Path) -> None:
    """Verify `User.md` can be created, updated, and edited."""
    from memory_store import UserProfileStore

    store = UserProfileStore(tmp_path)
    user_id = "test_user"

    # Read default
    text = store.read_text(user_id)
    assert "# User Profile" in text

    # Write
    content = "# User Profile\n- Tên: Alice\n- Nơi ở: Paris"
    store.write_text(user_id, content)
    assert store.read_text(user_id) == content
    assert store.file_size(user_id) > 0

    # Edit
    changed = store.edit_text(user_id, "Nơi ở: Paris", "Nơi ở: London")
    assert changed is True
    assert "Nơi ở: London" in store.read_text(user_id)


def test_compact_trigger(tmp_path: Path) -> None:
    """Verify long threads trigger compaction."""
    from memory_store import CompactMemoryManager

    # Threshold 10 tokens, keep 2 messages
    manager = CompactMemoryManager(threshold_tokens=10, keep_messages=2)
    thread_id = "test_thread"

    # Message 1 (approx 20 chars / 5 tokens)
    manager.append(thread_id, "user", "Hello first message")
    ctx = manager.context(thread_id)
    assert len(ctx["messages"]) == 1
    assert ctx["compactions"] == 0

    # Message 2 (approx 20 chars / 5 tokens)
    manager.append(thread_id, "assistant", "Hi there!")
    ctx = manager.context(thread_id)
    assert len(ctx["messages"]) == 2
    assert ctx["compactions"] == 0

    # Message 3 (approx 20 chars / 5 tokens) -> total 10 tokens
    manager.append(thread_id, "user", "What is your name?")
    ctx = manager.context(thread_id)
    assert len(ctx["messages"]) == 3
    assert ctx["compactions"] == 0

    # Message 4 (approx 20 chars / 5 tokens) -> total 15 tokens (> threshold)
    manager.append(thread_id, "assistant", "I am a chatbot.")
    ctx = manager.context(thread_id)
    assert len(ctx["messages"]) == 2
    assert ctx["compactions"] == 1
    assert "Tóm tắt hội thoại cũ" in str(ctx["summary"])


def test_cross_session_recall(tmp_path: Path) -> None:
    """Verify advanced remembers across sessions and baseline does not."""
    config = make_config(tmp_path)

    baseline = BaselineAgent(config, force_offline=True)
    advanced = AdvancedAgent(config, force_offline=True)

    user_id = "user_recall"
    thread_1 = "thread_1"
    thread_2 = "thread_2"

    # Tell them a fact in thread_1
    msg = "Mình tên là Alice và sống ở Paris"
    baseline.reply(user_id, thread_1, msg)
    advanced.reply(user_id, thread_1, msg)

    # Ask in thread_2 (fresh session)
    question = "Nhắc lại tên và nơi ở của mình"

    ans_baseline = baseline.reply(user_id, thread_2, question)["response"]
    ans_advanced = advanced.reply(user_id, thread_2, question)["response"]

    # Baseline forgets across threads
    assert "Alice" not in ans_baseline
    assert "Paris" not in ans_baseline

    # Advanced remembers across threads using User.md
    assert "Alice" in ans_advanced
    assert "Paris" in ans_advanced


def test_compact_reduces_prompt_load_on_long_thread(tmp_path: Path) -> None:
    """Compare prompt load of baseline vs advanced on a long thread."""
    config = make_config(tmp_path)

    baseline = BaselineAgent(config, force_offline=True)
    advanced = AdvancedAgent(config, force_offline=True)

    user_id = "user_compact"
    thread_id = "long_thread"

    # Send a series of turns to trigger compaction on advanced
    turns = [
        "Tin tức một về Artemis III phóng năm 2027. Đây là một tin tức rất dài để kiểm chứng xem liệu prompt load của baseline có tăng nhanh hơn advanced hay không khi hội thoại kéo dài.",
        "Tin tức hai về X-59 đạt vận tốc Mach 1.1. Chiếc máy bay này lần đầu tiên bay siêu thanh ở độ cao cực lớn nhằm mục tiêu giảm tiếng ồn siêu thanh đối với người dân bên dưới mặt đất.",
        "Tin tức ba về El Nino quay trở lại 2026. Tổ chức khí tượng thế giới cảnh báo nhiệt độ toàn cầu sẽ tăng mạnh và các quốc gia cần lên kịch bản ứng phó khẩn cấp ngay từ mùa hè này.",
        "Tin tức tư về British Columbia tiết kiệm điện. BC hydro công bố chương trình Power Smart thế hệ mới nhằm cắt giảm lượng điện tiêu thụ tương đương hàng trăm ngàn căn hộ.",
        "Đây là tin tức thứ năm để chắc chắn compaction xảy ra. Chúng ta đưa thêm nhiều chi tiết kỹ thuật về hệ thống mạng, hạ tầng dữ liệu và logs để tối ưu hóa chi phí token context."
    ]

    for turn in turns:
        baseline.reply(user_id, thread_id, turn)
        advanced.reply(user_id, thread_id, turn)

    # Check compaction count
    assert advanced.compaction_count(thread_id) > 0
    assert baseline.compaction_count(thread_id) == 0

    # Compare prompt load on next turn
    next_turn = "Hãy trả lời câu hỏi của mình."
    rep_baseline = baseline.reply(user_id, thread_id, next_turn)
    rep_advanced = advanced.reply(user_id, thread_id, next_turn)

    assert rep_advanced["prompt_tokens"] < rep_baseline["prompt_tokens"]
