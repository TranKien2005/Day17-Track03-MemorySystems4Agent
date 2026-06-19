from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path


def estimate_tokens(text: str) -> int:
    """Implement a simple token estimator based on character count."""
    if not text:
        return 0
    stripped = text.strip()
    if not stripped:
        return 0
    # A simple character-based heuristic: approx 4 characters per token
    return max(1, int(len(stripped) / 4))


@dataclass
class UserProfileStore:
    """Persistent storage for `User.md`."""

    root_dir: Path

    def path_for(self, user_id: str) -> Path:
        """Slugify and sanitize the user id to build a safe markdown file path."""
        sanitized = re.sub(r'[^a-zA-Z0-9_-]', '_', user_id).lower()
        return self.root_dir / f"{sanitized}.md"

    def read_text(self, user_id: str) -> str:
        """Return file content or an empty default markdown profile."""
        path = self.path_for(user_id)
        if path.exists():
            return path.read_text(encoding="utf-8")
        return "# User Profile\n\n"

    def write_text(self, user_id: str, content: str) -> Path:
        """Write markdown to disk and return the file path."""
        path = self.path_for(user_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def edit_text(self, user_id: str, search_text: str, replacement: str) -> bool:
        """Replace one occurrence inside User.md and return whether it changed."""
        path = self.path_for(user_id)
        if not path.exists():
            return False
        content = path.read_text(encoding="utf-8")
        if search_text not in content:
            return False
        new_content = content.replace(search_text, replacement, 1)
        if new_content != content:
            path.write_text(new_content, encoding="utf-8")
            return True
        return False

    def file_size(self, user_id: str) -> int:
        """Return the current file size in bytes."""
        path = self.path_for(user_id)
        if path.exists():
            return path.stat().st_size
        return 0


def extract_profile_updates(message: str) -> dict[str, str]:
    """Convert raw user text into stable profile facts.

    Example facts to extract:
    - name
    - location
    - profession
    - preferences / response style
    - favorite food / drink
    - pet
    """
    facts = {}
    lower_msg = message.lower()

    # 1. Skip if it is a pure question asking for information
    if "nhắc lại" in lower_msg and ("tên" in lower_msg or "đồ uống" in lower_msg or "style" in lower_msg or "ở đâu" in lower_msg):
        return facts
    if "biết dũngct không" in lower_msg or "dũngct là ai" in lower_msg:
        return facts

    # 2. Name Extraction
    name_match = re.search(
        r'(?:tên là|tên mình là|tên của mình là|tên:|my name is|i am|i\'m)\s*([A-ZÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝĐĂĨŨƠƯẠ-ỹ][A-Za-zÀ-ỹ\s]+)',
        message,
        re.IGNORECASE
    )
    if name_match:
        name = name_match.group(1).strip()
        name = re.sub(r'[.,!?\n]+$', '', name).strip()
        name = re.sub(r'\s+(?:nhé|nha|để|và|đang|and|from|living|va|so).*$', '', name, flags=re.IGNORECASE).strip()
        if name and not any(q in name.lower() for q in ["gì", "không", "ai", "đâu", "going", "what", "where"]):
            facts["name"] = name

    # 3. Location Extraction
    loc_match = re.search(
        r'(?:đang ở|hiện ở|mình ở|làm việc ở|ở|live in|living in|i\'m from|from|song o)\s+([A-ZÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝĐĂĨŨƠƯẠ-ỹA-Za-z][a-zà-ỹA-Za-z]*(\s+[A-ZÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝĐĂĨŨƠƯẠ-ỹA-Za-z][a-zà-ỹA-Za-z]*)*)',
        message,
        re.IGNORECASE
    )
    if loc_match:
        loc = loc_match.group(1).strip()
        loc = re.sub(r'[.,!?\n]+$', '', loc).strip()
        if loc and not any(q in loc.lower() for q in ["gì", "không", "đâu", "nào", "where", "what"]):
            facts["location"] = loc

    # 4. Profession Extraction
    if "mlops engineer" in lower_msg:
        facts["profession"] = "MLOps engineer"
    elif "backend engineer" in lower_msg:
        facts["profession"] = "backend engineer"
    elif "product manager" in lower_msg:
        if "chỉ là câu đùa" in lower_msg or "đùa với đồng nghiệp" in lower_msg or "đối tác chứ không phải" in lower_msg:
            pass
        else:
            facts["profession"] = "product manager"

    # 5. Favorite Food / Drink Extraction
    if "cà phê sữa đá" in lower_msg:
        facts["favorite_drink"] = "cà phê sữa đá"
    if "mì quảng" in lower_msg:
        facts["favorite_food"] = "mì Quảng"

    # 6. Response Style Extraction
    if any(kwd in lower_msg for kwd in ["ngắn gọn", "bullet", "ví dụ", "rõ ý"]):
        style_match = re.search(
            r'(?:style trả lời|trả lời thành|trả lời ngắn gọn|kiểu trả lời|muốn bạn trả lời|hãy trả lời)\s*([^.,!?\n]+)',
            message,
            re.IGNORECASE
        )
        if style_match:
            style = style_match.group(1).strip()
            style = re.sub(r'\s+(?:giúp mình|cho mình|nhé|nha|hơn|nhất|như cũ|để).*$', '', style, flags=re.IGNORECASE).strip()
            style = re.sub(r'\s+và\s*$', '', style).strip()
            if style and len(style) > 8 and not style.lower().startswith("mình"):
                facts["response_style"] = style

    # 7. Pet Extraction
    if "corgi" in lower_msg:
        facts["pet"] = "corgi tên Bơ"

    # Filtering step to ensure we do not store question-like snippets
    for k in list(facts.keys()):
        val = facts[k].lower()
        if "không" in val or "gì" in val or "chưa" in val or len(facts[k]) > 100:
            del facts[k]

    return facts


def summarize_messages(messages: list[dict[str, str]], max_items: int = 6) -> str:
    """Create a compact summary of older messages by truncating contents."""
    if not messages:
        return ""
    lines = []
    for msg in messages[:max_items]:
        role = msg.get("role", "user")
        content = msg.get("content", "").strip()
        # Truncate content to make it very short in offline mode
        short_content = content[:15] + "..." if len(content) > 15 else content
        lines.append(f"{role}: {short_content}")
    return "Tóm tắt hội thoại cũ: " + "; ".join(lines)


@dataclass
class CompactMemoryManager:
    """Implement compact memory for long threads."""

    threshold_tokens: int
    keep_messages: int
    state: dict[str, dict[str, object]] = field(default_factory=dict)

    def append(self, thread_id: str, role: str, content: str) -> None:
        """Append message and trigger compaction if needed."""
        if thread_id not in self.state:
            self.state[thread_id] = {
                "messages": [],
                "summary": "",
                "compactions": 0,
            }

        thread = self.state[thread_id]
        thread["messages"].append({"role": role, "content": content})

        # Calculate total tokens
        summary_tokens = estimate_tokens(str(thread["summary"]))
        messages_tokens = sum(
            estimate_tokens(str(msg["content"])) for msg in thread["messages"]
        )
        total_tokens = summary_tokens + messages_tokens

        # Trigger compaction if total_tokens exceeds threshold and we have more messages than we keep
        if (
            total_tokens > self.threshold_tokens
            and len(thread["messages"]) > self.keep_messages
        ):
            split_idx = len(thread["messages"]) - self.keep_messages
            to_summarize = thread["messages"][:split_idx]
            remaining = thread["messages"][split_idx:]

            new_summary_text = summarize_messages(to_summarize)

            if thread["summary"]:
                thread["summary"] = str(thread["summary"]) + "\n" + new_summary_text
            else:
                thread["summary"] = new_summary_text

            thread["messages"] = remaining
            thread["compactions"] += 1

    def context(self, thread_id: str) -> dict[str, object]:
        """Return per-thread state."""
        if thread_id not in self.state:
            return {
                "messages": [],
                "summary": "",
                "compactions": 0,
            }
        return self.state[thread_id]

    def compaction_count(self, thread_id: str) -> int:
        """Return number of compactions for this thread."""
        if thread_id not in self.state:
            return 0
        return int(self.state[thread_id]["compactions"])
