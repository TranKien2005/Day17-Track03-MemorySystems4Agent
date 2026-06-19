from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from config import LabConfig, load_config
from memory_store import (
    CompactMemoryManager,
    UserProfileStore,
    estimate_tokens,
    extract_profile_updates,
    extract_profile_updates_with_confidence,
)
from model_provider import build_chat_model


@dataclass
class AgentContext:
    user_id: str
    memory_path: str


class AdvancedAgent:
    """Advanced Agent (Agent B).

    Required memory layers:
    1. within-session memory
    2. persistent `User.md`
    3. compact memory for long threads
    """

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.profile_store = UserProfileStore(
            root_dir=self.config.state_dir / "profiles",
            confidence_threshold=getattr(self.config, "confidence_threshold", 0.70),
            decay_halflife=getattr(self.config, "decay_halflife", 20),
        )
        self.compact_memory = CompactMemoryManager(
            threshold_tokens=self.config.compact_threshold_tokens,
            keep_messages=self.config.compact_keep_messages,
        )
        self.thread_tokens: dict[str, int] = {}
        self.thread_prompt_tokens: dict[str, int] = {}
        self.session_histories = {}
        # Global turn counter per user (used for memory decay)
        self.user_turn_counter: dict[str, int] = {}

        try:
            self._maybe_build_langchain_agent()
        except Exception:
            self.langchain_agent = None

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Route between offline mode and live mode."""
        # Increment global turn counter for this user (drives memory decay)
        self.user_turn_counter[user_id] = self.user_turn_counter.get(user_id, 0) + 1
        current_turn = self.user_turn_counter[user_id]
        if self.langchain_agent and not self.force_offline:
            try:
                # 1. Extract facts with confidence scores and merge into User.md
                updates_with_conf = extract_profile_updates_with_confidence(message)
                if updates_with_conf:
                    self.profile_store.merge_updates(user_id, updates_with_conf, current_turn)

                # 2. Append user message to compact memory
                self.compact_memory.append(thread_id, "user", message)

                # 3. Estimate prompt context tokens
                prompt_tokens = self._estimate_prompt_context_tokens(
                    user_id, thread_id, current_turn
                )

                # 4. Invoke the live agent with decay-sorted profile
                profile_md = self.profile_store.get_context_for_prompt(user_id, current_turn)
                ctx = self.compact_memory.context(thread_id)
                summary_text = str(ctx["summary"])

                response = self.langchain_agent.invoke(
                    {
                        "input": message,
                        "profile": profile_md,
                        "summary": summary_text
                    },
                    config={"configurable": {"session_id": thread_id}}
                )
                response_text = response.content

                # 5. Append assistant reply to compact memory
                self.compact_memory.append(thread_id, "assistant", response_text)

                response_tokens = estimate_tokens(response_text)

                if thread_id not in self.thread_tokens:
                    self.thread_tokens[thread_id] = 0
                if thread_id not in self.thread_prompt_tokens:
                    self.thread_prompt_tokens[thread_id] = 0

                self.thread_tokens[thread_id] += response_tokens
                self.thread_prompt_tokens[thread_id] += prompt_tokens

                return {
                    "response": response_text,
                    "tokens": response_tokens,
                    "prompt_tokens": prompt_tokens
                }
            except Exception:
                return self._reply_offline(user_id, thread_id, message, current_turn)
        else:
            return self._reply_offline(user_id, thread_id, message, current_turn)

    def token_usage(self, thread_id: str) -> int:
        return self.thread_tokens.get(thread_id, 0)

    def prompt_token_usage(self, thread_id: str) -> int:
        return self.thread_prompt_tokens.get(thread_id, 0)

    def memory_file_size(self, user_id: str) -> int:
        return self.profile_store.file_size(user_id)

    def compaction_count(self, thread_id: str) -> int:
        return self.compact_memory.compaction_count(thread_id)

    def _reply_offline(self, user_id: str, thread_id: str, message: str,
                       current_turn: int = 1) -> dict[str, Any]:
        """Implement the deterministic advanced path."""
        # 1. Extract facts with confidence + apply threshold before writing
        updates_with_conf = extract_profile_updates_with_confidence(message)
        written = self.profile_store.merge_updates(user_id, updates_with_conf, current_turn)

        # 2. Old boilerplate removed — merge_updates() handles persistence
        _ = written  # suppress unused warning

        # 3. Append the message into compact memory
        self.compact_memory.append(thread_id, "user", message)

        # 4. Estimate prompt-context load using decay-sorted profile
        prompt_tokens = self._estimate_prompt_context_tokens(user_id, thread_id, current_turn)

        # 5. Generate a response that can answer long-term recall questions
        response = self._offline_response(user_id, thread_id, message, current_turn)

        # 6. Append the assistant reply and update token counters
        self.compact_memory.append(thread_id, "assistant", response)

        response_tokens = estimate_tokens(response)

        if thread_id not in self.thread_tokens:
            self.thread_tokens[thread_id] = 0
        if thread_id not in self.thread_prompt_tokens:
            self.thread_prompt_tokens[thread_id] = 0

        self.thread_tokens[thread_id] += response_tokens
        self.thread_prompt_tokens[thread_id] += prompt_tokens

        return {
            "response": response,
            "tokens": response_tokens,
            "prompt_tokens": prompt_tokens
        }

    def _estimate_prompt_context_tokens(self, user_id: str, thread_id: str,
                                        current_turn: int = 1) -> int:
        """Estimate the context carried into one turn (uses decay-sorted profile)."""
        user_md = self.profile_store.get_context_for_prompt(user_id, current_turn)
        user_tokens = estimate_tokens(user_md)

        ctx = self.compact_memory.context(thread_id)
        summary_tokens = estimate_tokens(str(ctx["summary"]))

        messages = ctx["messages"]
        messages_tokens = 0
        # Exclude the latest user message which is the current input
        if len(messages) > 1:
            messages_tokens = sum(estimate_tokens(str(msg["content"])) for msg in messages[:-1])

        return user_tokens + summary_tokens + messages_tokens

    def _offline_response(self, user_id: str, thread_id: str, message: str,
                          current_turn: int = 1) -> str:
        """Return a deterministic answer using persisted memory (decay-sorted)."""
        facts = self.profile_store.read_facts(user_id)

        response_parts = []
        if facts:
            response_parts.append("Thông tin về bạn mà mình nhớ:")
            for k, v in facts.items():
                if k == "name": response_parts.append(f"- Tên: {v}")
                elif k == "location": response_parts.append(f"- Nơi ở hiện tại: {v}")
                elif k == "profession": response_parts.append(f"- Nghề nghiệp: {v}")
                elif k == "response_style": response_parts.append(f"- Phong cách trả lời: {v}")
                elif k == "favorite_drink": response_parts.append(f"- Đồ uống yêu thích: {v}")
                elif k == "favorite_food": response_parts.append(f"- Món ăn yêu thích: {v}")
                elif k == "pet": response_parts.append(f"- Pet: {v}")
        else:
            response_parts.append("Mình chưa có thông tin nào về bạn trong hồ sơ.")

        return "\n".join(response_parts)

    def _maybe_build_langchain_agent(self):
        """Wire a live agent with tools and compact middleware."""
        try:
            from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
            from langchain_core.runnables.history import RunnableWithMessageHistory
            from langchain_community.chat_message_histories import ChatMessageHistory

            chat_model = build_chat_model(self.config.model)

            prompt = ChatPromptTemplate.from_messages([
                ("system", "Bạn là một AI assistant nâng cao (Advanced Agent). Bạn có khả năng nhớ thông tin dài hạn của người dùng. "
                           "Dưới đây là thông tin hiện tại trong hồ sơ của người dùng (User Profile):\n{profile}\n\n"
                           "Dưới đây là tóm tắt lịch sử hội thoại cũ (nếu có):\n{summary}\n\n"
                           "Hãy trả lời bằng tiếng Việt, bám sát phong cách trả lời yêu thích của người dùng nếu có."),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{input}")
            ])

            self.session_histories = {}

            def get_session_history(session_id: str):
                if session_id not in self.session_histories:
                    self.session_histories[session_id] = ChatMessageHistory()
                return self.session_histories[session_id]

            chain = prompt | chat_model

            self.langchain_agent = RunnableWithMessageHistory(
                chain,
                get_session_history,
                input_messages_key="input",
                history_messages_key="history"
            )
        except Exception:
            self.langchain_agent = None
