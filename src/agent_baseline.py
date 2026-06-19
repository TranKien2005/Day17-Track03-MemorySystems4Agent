from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from config import LabConfig, load_config
from memory_store import estimate_tokens
from model_provider import build_chat_model


@dataclass
class SessionState:
    messages: list[dict[str, str]] = field(default_factory=list)
    token_usage: int = 0
    prompt_tokens_processed: int = 0


class BaselineAgent:
    """Baseline Agent (Agent A).

    Requirements:
    - Within-session memory only
    - No persistent `User.md`
    - Should forget long-term facts across new threads
    """

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.sessions: dict[str, SessionState] = {}
        self.session_histories = {}

        try:
            self._maybe_build_langchain_agent()
        except Exception:
            self.langchain_agent = None

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Return the agent response and token accounting."""
        if self.langchain_agent and not self.force_offline:
            try:
                response = self.langchain_agent.invoke(
                    {"input": message},
                    config={"configurable": {"session_id": thread_id}}
                )
                response_text = response.content

                if thread_id not in self.sessions:
                    self.sessions[thread_id] = SessionState()
                session = self.sessions[thread_id]

                # Estimate prompt tokens processed in this turn
                history_msgs = self.session_histories.get(thread_id)
                prompt_context = ""
                if history_msgs:
                    prompt_context = "".join(getattr(msg, "content", str(msg)) for msg in history_msgs.messages)
                
                prompt_tokens = estimate_tokens(prompt_context)
                session.prompt_tokens_processed += prompt_tokens

                response_tokens = estimate_tokens(response_text)
                session.token_usage += response_tokens

                return {
                    "response": response_text,
                    "tokens": response_tokens,
                    "prompt_tokens": prompt_tokens
                }
            except Exception:
                return self._reply_offline(thread_id, message)
        else:
            return self._reply_offline(thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        """Return cumulative agent token count for one thread."""
        if thread_id not in self.sessions:
            return 0
        return self.sessions[thread_id].token_usage

    def prompt_token_usage(self, thread_id: str) -> int:
        """Estimate how much prompt context this baseline kept processing."""
        if thread_id not in self.sessions:
            return 0
        return self.sessions[thread_id].prompt_tokens_processed

    def compaction_count(self, thread_id: str) -> int:
        # Baseline has no compact memory.
        return 0

    def _reply_offline(self, thread_id: str, message: str) -> dict[str, Any]:
        """Implement a simple offline behavior."""
        if thread_id not in self.sessions:
            self.sessions[thread_id] = SessionState()
        session = self.sessions[thread_id]

        # Estimate prompt tokens processed
        prompt_context = "".join(msg["content"] for msg in session.messages)
        prompt_tokens = estimate_tokens(prompt_context)
        session.prompt_tokens_processed += prompt_tokens

        # Store the new user message
        session.messages.append({"role": "user", "content": message})

        # Generate a short deterministic reply
        # Baseline does not remember facts across different thread ids.
        # If it is in the same thread, we can scan previous messages.
        lower_msg = message.lower()
        response = "Xin chào! Mình là chatbot baseline."

        if "tên" in lower_msg or "style" in lower_msg or "uống" in lower_msg or "ăn" in lower_msg or "ở đâu" in lower_msg:
            found_facts = []
            # Scan previous messages in this session
            for msg in session.messages[:-1]:  # exclude latest user message
                content = msg["content"]
                if "tên là" in content.lower():
                    found_facts.append(content)
                elif "thích" in content.lower():
                    found_facts.append(content)
                elif "ở" in content.lower():
                    found_facts.append(content)
            if found_facts:
                response = "Dựa trên hội thoại hiện tại: " + "; ".join(found_facts)
            else:
                response = "Mình không nhớ thông tin này vì đây là thread mới và mình không có bộ nhớ dài hạn."

        # Store response
        session.messages.append({"role": "assistant", "content": response})

        # Update tokens
        response_tokens = estimate_tokens(response)
        session.token_usage += response_tokens

        return {
            "response": response,
            "tokens": response_tokens,
            "prompt_tokens": prompt_tokens
        }

    def _maybe_build_langchain_agent(self):
        """Wire a basic LangChain agent with memory."""
        try:
            from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
            from langchain_core.runnables.history import RunnableWithMessageHistory
            from langchain_community.chat_message_histories import ChatMessageHistory

            chat_model = build_chat_model(self.config.model)

            prompt = ChatPromptTemplate.from_messages([
                ("system", "Bạn là một AI assistant cơ bản (Baseline Agent). Bạn chỉ có thể nhớ các tin nhắn trong cùng một thread hội thoại. Trả lời bằng tiếng Việt."),
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
