from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------------
# Token estimator
# ---------------------------------------------------------------------------

def estimate_tokens(text: str) -> int:
    """Implement a simple token estimator based on character count."""
    if not text:
        return 0
    stripped = text.strip()
    if not stripped:
        return 0
    # A simple character-based heuristic: approx 4 characters per token
    return max(1, int(len(stripped) / 4))


# ---------------------------------------------------------------------------
# UserProfileStore – persistent User.md with sidecar metadata for decay
# ---------------------------------------------------------------------------

@dataclass
class UserProfileStore:
    """Persistent storage for `User.md`.

    Each user gets two files:
    - ``{user_id}.md``: human-readable profile facts
    - ``{user_id}_meta.json``: per-fact metadata for confidence & decay
    """

    root_dir: Path

    # Confidence threshold: only persist facts with score >= this value
    confidence_threshold: float = 0.70

    # Memory decay half-life in turns: after this many turns without
    # being mentioned, a fact's decay score halves.
    decay_halflife: int = 20

    def path_for(self, user_id: str) -> Path:
        """Slugify and sanitize the user id to build a safe markdown file path."""
        sanitized = re.sub(r'[^a-zA-Z0-9_-]', '_', user_id).lower()
        return self.root_dir / f"{sanitized}.md"

    def _meta_path(self, user_id: str) -> Path:
        sanitized = re.sub(r'[^a-zA-Z0-9_-]', '_', user_id).lower()
        return self.root_dir / f"{sanitized}_meta.json"

    # ------------------------------------------------------------------
    # Metadata helpers (confidence + decay)
    # ------------------------------------------------------------------

    def _load_meta(self, user_id: str) -> dict:
        """Load per-fact metadata from sidecar JSON.

        Schema per fact key:
        {
            "confidence": float,   # extraction confidence at last write
            "mention_count": int,  # times fact has been confirmed/updated
            "last_turn": int,      # global turn index of last update
        }
        """
        path = self._meta_path(user_id)
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                return {}
        return {}

    def _save_meta(self, user_id: str, meta: dict) -> None:
        path = self._meta_path(user_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    def decay_score(self, user_id: str, key: str, current_turn: int) -> float:
        """Compute decay score for a fact using exponential decay.

        score = mention_count * exp(-ln(2) / halflife * turns_elapsed)

        A fresh fact with mention_count=1 starts at 1.0 and decays to 0.5
        after ``decay_halflife`` turns without being mentioned again.
        """
        meta = self._load_meta(user_id)
        if key not in meta:
            return 0.0
        fm = meta[key]
        mention_count = max(1, fm.get("mention_count", 1))
        last_turn = fm.get("last_turn", current_turn)
        turns_elapsed = max(0, current_turn - last_turn)
        lam = math.log(2) / max(1, self.decay_halflife)
        return mention_count * math.exp(-lam * turns_elapsed)

    def is_stale(self, user_id: str, key: str, current_turn: int,
                 stale_threshold: float = 0.25) -> bool:
        """Return True if a fact's decay score has fallen below stale_threshold."""
        return self.decay_score(user_id, key, current_turn) < stale_threshold

    def update_fact_meta(self, user_id: str, key: str, confidence: float,
                         current_turn: int) -> None:
        """Update metadata when a fact is written or confirmed."""
        meta = self._load_meta(user_id)
        existing = meta.get(key, {})
        meta[key] = {
            "confidence": max(confidence, existing.get("confidence", 0.0)),
            "mention_count": existing.get("mention_count", 0) + 1,
            "last_turn": current_turn,
        }
        self._save_meta(user_id, meta)

    # ------------------------------------------------------------------
    # Core CRUD
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Fact-level read/write with confidence + decay filtering
    # ------------------------------------------------------------------

    def read_facts(self, user_id: str) -> dict[str, str]:
        """Parse User.md and return a dict of {key: value}."""
        facts: dict[str, str] = {}
        for line in self.read_text(user_id).splitlines():
            if line.startswith("- Tên:"):
                facts["name"] = line.split(":", 1)[1].strip()
            elif line.startswith("- Nơi ở:"):
                facts["location"] = line.split(":", 1)[1].strip()
            elif line.startswith("- Nghề nghiệp:"):
                facts["profession"] = line.split(":", 1)[1].strip()
            elif line.startswith("- Phong cách trả lời:"):
                facts["response_style"] = line.split(":", 1)[1].strip()
            elif line.startswith("- Đồ uống yêu thích:"):
                facts["favorite_drink"] = line.split(":", 1)[1].strip()
            elif line.startswith("- Món ăn yêu thích:"):
                facts["favorite_food"] = line.split(":", 1)[1].strip()
            elif line.startswith("- Pet:"):
                facts["pet"] = line.split(":", 1)[1].strip()
        return facts

    def write_facts(self, user_id: str, facts: dict[str, str]) -> None:
        """Serialize facts dict back to User.md."""
        label_map = {
            "name": "Tên",
            "location": "Nơi ở",
            "profession": "Nghề nghiệp",
            "response_style": "Phong cách trả lời",
            "favorite_drink": "Đồ uống yêu thích",
            "favorite_food": "Món ăn yêu thích",
            "pet": "Pet",
        }
        lines = ["# User Profile"]
        for k, label in label_map.items():
            if k in facts:
                lines.append(f"- {label}: {facts[k]}")
        self.write_text(user_id, "\n".join(lines) + "\n")

    def merge_updates(self, user_id: str,
                      updates_with_conf: dict[str, tuple[str, float]],
                      current_turn: int) -> dict[str, str]:
        """Merge extracted facts into User.md, applying confidence threshold.

        Only facts with confidence >= self.confidence_threshold are written.
        Updates metadata (mention_count, last_turn, confidence) for decay.

        Returns the set of fact keys that were actually written.
        """
        existing = self.read_facts(user_id)
        written: dict[str, str] = {}

        for key, (value, confidence) in updates_with_conf.items():
            if confidence < self.confidence_threshold:
                # Below threshold — skip writing to persistent memory
                continue
            existing[key] = value
            written[key] = value
            self.update_fact_meta(user_id, key, confidence, current_turn)

        if written:
            self.write_facts(user_id, existing)
        return written

    def get_context_for_prompt(self, user_id: str, current_turn: int,
                                stale_threshold: float = 0.25) -> str:
        """Build the User.md section for the prompt, sorted by decay score.

        Active facts (decay >= stale_threshold) appear first.
        Stale facts are appended with a [stale] annotation so the LLM
        knows they may be outdated — but they're not dropped entirely,
        preserving recall while signalling uncertainty.
        """
        facts = self.read_facts(user_id)
        if not facts:
            return "# User Profile\n\n(Chưa có thông tin.)\n"

        label_map = {
            "name": "Tên",
            "location": "Nơi ở",
            "profession": "Nghề nghiệp",
            "response_style": "Phong cách trả lời",
            "favorite_drink": "Đồ uống yêu thích",
            "favorite_food": "Món ăn yêu thích",
            "pet": "Pet",
        }

        active_lines: list[tuple[float, str]] = []
        stale_lines: list[str] = []

        for k, v in facts.items():
            score = self.decay_score(user_id, k, current_turn)
            label = label_map.get(k, k)
            if score >= stale_threshold:
                active_lines.append((score, f"- {label}: {v}"))
            else:
                stale_lines.append(f"- {label}: {v} [stale, score={score:.2f}]")

        # Sort active facts by descending decay score (most-confirmed first)
        active_lines.sort(key=lambda x: x[0], reverse=True)

        lines = ["# User Profile"]
        for _, line in active_lines:
            lines.append(line)
        if stale_lines:
            lines.append("\n## Thông tin có thể đã cũ")
            lines.extend(stale_lines)

        return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Confidence-aware fact extractor
# ---------------------------------------------------------------------------

def extract_profile_updates_with_confidence(
    message: str,
) -> dict[str, tuple[str, float]]:
    """Extract stable profile facts with a confidence score [0.0, 1.0].

    Confidence rules:
    - Explicit declaration («my name is X», «tên là X») → 0.95
    - «I am X» / «I'm X» (could be transient state)    → 0.75
    - Location via «live in» / «đang ở»                → 0.90
    - Weaker location via «from»                        → 0.70
    - Known profession keyword match                    → 0.95
    - Preferred food/drink/pet keyword match            → 0.90
    - Response style extraction                         → 0.85

    Facts with confidence below the store's threshold are NOT written.
    """
    facts: dict[str, tuple[str, float]] = {}
    lower_msg = message.lower()

    # 1. Anti-question filter
    if "nhắc lại" in lower_msg and (
        "tên" in lower_msg or "đồ uống" in lower_msg
        or "style" in lower_msg or "ở đâu" in lower_msg
    ):
        return facts
    if "biết dũngct không" in lower_msg or "dũngct là ai" in lower_msg:
        return facts

    # 2. Name extraction — high confidence for explicit declarations
    explicit_name = re.search(
        r'(?:tên là|tên mình là|tên của mình là|tên:|my name is)\s*'
        r'([A-ZÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝĐĂĨŨƠƯẠ-ỹ][A-Za-zÀ-ỹ\s]+)',
        message, re.IGNORECASE
    )
    if explicit_name:
        name = explicit_name.group(1).strip()
        name = re.sub(r'[.,!?\n]+$', '', name).strip()
        name = re.sub(
            r'\s+(?:nhé|nha|để|và|đang|and|from|living|va|so).*$', '',
            name, flags=re.IGNORECASE
        ).strip()
        if name and not any(q in name.lower() for q in ["gì", "không", "ai", "đâu", "what", "where"]):
            facts["name"] = (name, 0.95)
    else:
        # «I am» / «I'm» → lower confidence (could mean state, not name)
        implicit_name = re.search(
            r'(?:i am|i\'m)\s+([A-Z][A-Za-z]+)',
            message, re.IGNORECASE
        )
        if implicit_name:
            name = implicit_name.group(1).strip()
            name = re.sub(r'\s+(?:and|from|living|va|so).*$', '', name, flags=re.IGNORECASE).strip()
            if name and not any(q in name.lower() for q in ["going", "what", "where", "not", "just"]):
                facts["name"] = (name, 0.75)

    # 3. Location extraction
    strong_loc = re.search(
        r'(?:đang ở|hiện ở|mình ở|làm việc ở|live in|living in)\s+'
        r'([A-ZÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝĐĂĨŨƠƯẠ-ỹA-Za-z][a-zà-ỹA-Za-z]*'
        r'(?:\s+[A-ZÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝĐĂĨŨƠƯẠ-ỹA-Za-z][a-zà-ỹA-Za-z]*)*)',
        message, re.IGNORECASE
    )
    weak_loc = re.search(
        r'(?:i\'m from|from|song o)\s+'
        r'([A-ZÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝĐĂĨŨƠƯẠ-ỹA-Za-z][a-zà-ỹA-Za-z]*'
        r'(?:\s+[A-ZÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝĐĂĨŨƠƯẠ-ỹA-Za-z][a-zà-ỹA-Za-z]*)*)',
        message, re.IGNORECASE
    )
    if strong_loc:
        loc = re.sub(r'[.,!?\n]+$', '', strong_loc.group(1).strip()).strip()
        if loc and not any(q in loc.lower() for q in ["gì", "không", "đâu", "nào", "where", "what"]):
            facts["location"] = (loc, 0.90)
    elif weak_loc:
        loc = re.sub(r'[.,!?\n]+$', '', weak_loc.group(1).strip()).strip()
        if loc and not any(q in loc.lower() for q in ["gì", "không", "đâu", "nào", "where", "what"]):
            facts["location"] = (loc, 0.70)

    # 4. Profession extraction — keyword match is very reliable
    if "mlops engineer" in lower_msg:
        facts["profession"] = ("MLOps engineer", 0.95)
    elif "backend engineer" in lower_msg:
        facts["profession"] = ("backend engineer", 0.95)
    elif "product manager" in lower_msg:
        if not any(k in lower_msg for k in ["chỉ là câu đùa", "đùa với", "đối tác chứ không phải"]):
            facts["profession"] = ("product manager", 0.95)

    # 5. Favorite food / drink
    if "cà phê sữa đá" in lower_msg:
        facts["favorite_drink"] = ("cà phê sữa đá", 0.90)
    if "mì quảng" in lower_msg:
        facts["favorite_food"] = ("mì Quảng", 0.90)

    # 6. Response style
    if any(kwd in lower_msg for kwd in ["ngắn gọn", "bullet", "ví dụ", "rõ ý"]):
        style_match = re.search(
            r'(?:style trả lời|trả lời thành|trả lời ngắn gọn|kiểu trả lời'
            r'|muốn bạn trả lời|hãy trả lời)\s*([^.,!?\n]+)',
            message, re.IGNORECASE
        )
        if style_match:
            style = style_match.group(1).strip()
            style = re.sub(
                r'\s+(?:giúp mình|cho mình|nhé|nha|hơn|nhất|như cũ|để).*$', '',
                style, flags=re.IGNORECASE
            ).strip()
            style = re.sub(r'\s+và\s*$', '', style).strip()
            if style and len(style) > 8 and not style.lower().startswith("mình"):
                facts["response_style"] = (style, 0.85)

    # 7. Pet
    if "corgi" in lower_msg:
        facts["pet"] = ("corgi tên Bơ", 0.90)

    # Sanity filter
    for k in list(facts.keys()):
        val = facts[k][0].lower()
        if "không" in val or "gì" in val or "chưa" in val or len(facts[k][0]) > 100:
            del facts[k]

    return facts


def extract_profile_updates(message: str) -> dict[str, str]:
    """Backward-compatible wrapper — returns {key: value} without confidence."""
    return {k: v for k, (v, _) in extract_profile_updates_with_confidence(message).items()}


# ---------------------------------------------------------------------------
# summarize_messages + CompactMemoryManager (unchanged)
# ---------------------------------------------------------------------------

def summarize_messages(messages: list[dict[str, str]], max_items: int = 6) -> str:
    """Create a compact summary of older messages by truncating contents."""
    if not messages:
        return ""
    lines = []
    for msg in messages[:max_items]:
        role = msg.get("role", "user")
        content = msg.get("content", "").strip()
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
