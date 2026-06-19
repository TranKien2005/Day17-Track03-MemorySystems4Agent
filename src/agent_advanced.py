from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from config import LabConfig, load_config
from memory_store import CompactMemoryManager, UserProfileStore, estimate_tokens, extract_profile_updates
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
        self.profile_store = UserProfileStore(self.config.state_dir / "profiles")
        self.compact_memory = CompactMemoryManager(
            threshold_tokens=self.config.compact_threshold_tokens,
            keep_messages=self.config.compact_keep_messages,
        )
        self.thread_tokens: dict[str, int] = {}
        self.thread_prompt_tokens: dict[str, int] = {}
        self.session_histories = {}

        try:
            self._maybe_build_langchain_agent()
        except Exception:
            self.langchain_agent = None

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Route between offline mode and live mode."""
        if self.langchain_agent and not self.force_offline:
            try:
                # 1. Proactively extract stable profile facts and update User.md
                updates = extract_profile_updates(message)
                if updates:
                    profile_text = self.profile_store.read_text(user_id)
                    lines = profile_text.splitlines()
                    facts_in_file = {}
                    for line in lines:
                        if line.startswith("- Tên:"):
                            facts_in_file["name"] = line.split(":", 1)[1].strip()
                        elif line.startswith("- Nơi ở:"):
                            facts_in_file["location"] = line.split(":", 1)[1].strip()
                        elif line.startswith("- Nghề nghiệp:"):
                            facts_in_file["profession"] = line.split(":", 1)[1].strip()
                        elif line.startswith("- Phong cách trả lời:"):
                            facts_in_file["response_style"] = line.split(":", 1)[1].strip()
                        elif line.startswith("- Đồ uống yêu thích:"):
                            facts_in_file["favorite_drink"] = line.split(":", 1)[1].strip()
                        elif line.startswith("- Món ăn yêu thích:"):
                            facts_in_file["favorite_food"] = line.split(":", 1)[1].strip()
                        elif line.startswith("- Pet:"):
                            facts_in_file["pet"] = line.split(":", 1)[1].strip()
                    facts_in_file.update(updates)

                    new_lines = ["# User Profile"]
                    for k, v in facts_in_file.items():
                        if k == "name": new_lines.append(f"- Tên: {v}")
                        elif k == "location": new_lines.append(f"- Nơi ở: {v}")
                        elif k == "profession": new_lines.append(f"- Nghề nghiệp: {v}")
                        elif k == "response_style": new_lines.append(f"- Phong cách trả lời: {v}")
                        elif k == "favorite_drink": new_lines.append(f"- Đồ uống yêu thích: {v}")
                        elif k == "favorite_food": new_lines.append(f"- Món ăn yêu thích: {v}")
                        elif k == "pet": new_lines.append(f"- Pet: {v}")
                    self.profile_store.write_text(user_id, "\n".join(new_lines) + "\n")

                # 2. Append user message to compact memory
                self.compact_memory.append(thread_id, "user", message)

                # 3. Estimate prompt context tokens
                prompt_tokens = self._estimate_prompt_context_tokens(user_id, thread_id)

                # 4. Invoke the live agent
                profile_md = self.profile_store.read_text(user_id)
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
                return self._reply_offline(user_id, thread_id, message)
        else:
            return self._reply_offline(user_id, thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        return self.thread_tokens.get(thread_id, 0)

    def prompt_token_usage(self, thread_id: str) -> int:
        return self.thread_prompt_tokens.get(thread_id, 0)

    def memory_file_size(self, user_id: str) -> int:
        return self.profile_store.file_size(user_id)

    def compaction_count(self, thread_id: str) -> int:
        return self.compact_memory.compaction_count(thread_id)

    def _reply_offline(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Implement the deterministic advanced path."""
        # 1. Extract stable profile facts from the incoming message
        updates = extract_profile_updates(message)

        # 2. Persist those facts into User.md
        if updates:
            profile_text = self.profile_store.read_text(user_id)
            lines = profile_text.splitlines()
            facts_in_file = {}
            for line in lines:
                if line.startswith("- Tên:"):
                    facts_in_file["name"] = line.split(":", 1)[1].strip()
                elif line.startswith("- Nơi ở:"):
                    facts_in_file["location"] = line.split(":", 1)[1].strip()
                elif line.startswith("- Nghề nghiệp:"):
                    facts_in_file["profession"] = line.split(":", 1)[1].strip()
                elif line.startswith("- Phong cách trả lời:"):
                    facts_in_file["response_style"] = line.split(":", 1)[1].strip()
                elif line.startswith("- Đồ uống yêu thích:"):
                    facts_in_file["favorite_drink"] = line.split(":", 1)[1].strip()
                elif line.startswith("- Món ăn yêu thích:"):
                    facts_in_file["favorite_food"] = line.split(":", 1)[1].strip()
                elif line.startswith("- Pet:"):
                    facts_in_file["pet"] = line.split(":", 1)[1].strip()
            facts_in_file.update(updates)

            new_lines = ["# User Profile"]
            for k, v in facts_in_file.items():
                if k == "name": new_lines.append(f"- Tên: {v}")
                elif k == "location": new_lines.append(f"- Nơi ở: {v}")
                elif k == "profession": new_lines.append(f"- Nghề nghiệp: {v}")
                elif k == "response_style": new_lines.append(f"- Phong cách trả lời: {v}")
                elif k == "favorite_drink": new_lines.append(f"- Đồ uống yêu thích: {v}")
                elif k == "favorite_food": new_lines.append(f"- Món ăn yêu thích: {v}")
                elif k == "pet": new_lines.append(f"- Pet: {v}")
            self.profile_store.write_text(user_id, "\n".join(new_lines) + "\n")

        # 3. Append the message into compact memory
        self.compact_memory.append(thread_id, "user", message)

        # 4. Estimate prompt-context load from User.md + summary + recent messages
        prompt_tokens = self._estimate_prompt_context_tokens(user_id, thread_id)

        # 5. Generate a response that can answer long-term recall questions
        response = self._offline_response(user_id, thread_id, message)

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

    def _estimate_prompt_context_tokens(self, user_id: str, thread_id: str) -> int:
        """Estimate the context carried into one turn."""
        user_md = self.profile_store.read_text(user_id)
        user_tokens = estimate_tokens(user_md)

        ctx = self.compact_memory.context(thread_id)
        summary_tokens = estimate_tokens(str(ctx["summary"]))

        messages = ctx["messages"]
        messages_tokens = 0
        # Exclude the latest user message which is the current input
        if len(messages) > 1:
            messages_tokens = sum(estimate_tokens(str(msg["content"])) for msg in messages[:-1])

        return user_tokens + summary_tokens + messages_tokens

    def _offline_response(self, user_id: str, thread_id: str, message: str) -> str:
        """Return a deterministic answer using persisted memory."""
        profile_text = self.profile_store.read_text(user_id)
        lines = profile_text.splitlines()
        facts = {}
        for line in lines:
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
