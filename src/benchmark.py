from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config


@dataclass
class BenchmarkRow:
    agent_name: str
    agent_tokens_only: int
    prompt_tokens_processed: int
    recall_score: float
    response_quality: float
    memory_growth_bytes: int
    compactions: int


def load_conversations(path: Path) -> list[dict[str, Any]]:
    """Read JSON conversations from disk."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def recall_points(answer: str, expected: list[str]) -> float:
    """Return 0 / 0.5 / 1 depending on how many expected facts appear (case-insensitive)."""
    if not expected:
        return 1.0
    ans_lower = answer.lower()
    matches = 0
    for fact in expected:
        if fact.lower() in ans_lower:
            matches += 1
    return float(matches) / len(expected)


def heuristic_quality(answer: str, expected: list[str]) -> float:
    """Add a lightweight quality score for offline mode."""
    if not answer:
        return 0.0
    recall = recall_points(answer, expected)
    score = recall * 0.8
    # Length constraint
    if 20 < len(answer) < 600:
        score += 0.15
    # Structured answer constraint
    if any(line.strip().startswith(("-", "*", "1.", "2.", "3.")) for line in answer.splitlines()):
        score += 0.05
    return min(1.0, max(0.0, score))


def run_agent_benchmark(agent_name: str, agent, conversations: list[dict[str, Any]], config) -> BenchmarkRow:
    """Evaluate one agent over many conversations."""
    total_agent_tokens = 0
    total_prompt_tokens = 0
    recall_scores = []
    quality_scores = []
    total_compactions = 0

    user_ids = set(conv["user_id"] for conv in conversations)
    initial_mem_sizes = {}
    for uid in user_ids:
        initial_mem_sizes[uid] = agent.memory_file_size(uid) if hasattr(agent, "memory_file_size") else 0

    for conv in conversations:
        conv_id = conv["id"]
        user_id = conv["user_id"]
        turns = conv["turns"]

        # 1. Feed all turns to the agent.
        for turn in turns:
            reply_dict = agent.reply(user_id, conv_id, turn)
            total_agent_tokens += reply_dict.get("tokens", 0)
            total_prompt_tokens += reply_dict.get("prompt_tokens", 0)

        total_compactions += agent.compaction_count(conv_id)

        # 2. Ask recall questions in a fresh thread.
        for i, q_item in enumerate(conv["recall_questions"]):
            question = q_item["question"]
            expected = q_item["expected_contains"]
            recall_thread_id = f"{conv_id}-recall-{i}"

            reply_dict = agent.reply(user_id, recall_thread_id, question)
            ans = reply_dict["response"]

            total_agent_tokens += reply_dict.get("tokens", 0)
            total_prompt_tokens += reply_dict.get("prompt_tokens", 0)

            # 3. Compute recall and quality scores.
            rec_score = recall_points(ans, expected)
            qual_score = heuristic_quality(ans, expected)

            recall_scores.append(rec_score)
            quality_scores.append(qual_score)

    # 4. Record memory file growth.
    final_mem_growth = 0
    for uid in user_ids:
        final_size = agent.memory_file_size(uid) if hasattr(agent, "memory_file_size") else 0
        final_mem_growth += max(0, final_size - initial_mem_sizes[uid])

    avg_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 1.0
    avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 1.0

    return BenchmarkRow(
        agent_name=agent_name,
        agent_tokens_only=total_agent_tokens,
        prompt_tokens_processed=total_prompt_tokens,
        recall_score=avg_recall,
        response_quality=avg_quality,
        memory_growth_bytes=final_mem_growth,
        compactions=total_compactions,
    )


def format_rows(rows: list[BenchmarkRow]) -> str:
    """Print a markdown table output."""
    from tabulate import tabulate

    headers = [
        "Agent Name",
        "Agent tokens only",
        "Prompt tokens processed",
        "Cross-session recall",
        "Response quality",
        "Memory growth (bytes)",
        "Compactions"
    ]

    table_data = []
    for r in rows:
        table_data.append([
            r.agent_name,
            f"{r.agent_tokens_only:,}",
            f"{r.prompt_tokens_processed:,}",
            f"{r.recall_score * 100:.1f}%",
            f"{r.response_quality * 100:.1f}%",
            f"{r.memory_growth_bytes:,}",
            r.compactions
        ])

    return tabulate(table_data, headers=headers, tablefmt="github")


def main() -> None:
    """Run both benchmark suites."""
    config = load_config(Path(__file__).resolve().parent.parent)

    # Clean up state profiles directory before benchmarking
    profiles_dir = config.state_dir / "profiles"
    if profiles_dir.exists():
        import shutil
        shutil.rmtree(profiles_dir)
    profiles_dir.mkdir(parents=True, exist_ok=True)

    # Load both datasets from root/data
    standard_data_path = config.data_dir / "conversations.json"
    stress_data_path = config.data_dir / "advanced_long_context.json"

    standard_convs = load_conversations(standard_data_path)
    stress_convs = load_conversations(stress_data_path)

    print("=== RUNNING STANDARD BENCHMARK ===")
    baseline_agent_std = BaselineAgent(config, force_offline=True)
    advanced_agent_std = AdvancedAgent(config, force_offline=True)

    row_baseline_std = run_agent_benchmark("Baseline Agent (Offline)", baseline_agent_std, standard_convs, config)
    row_advanced_std = run_agent_benchmark("Advanced Agent (Offline)", advanced_agent_std, standard_convs, config)

    print(format_rows([row_baseline_std, row_advanced_std]))
    print("\n")

    # Clean profiles again before stress test to start fresh
    if profiles_dir.exists():
        import shutil
        shutil.rmtree(profiles_dir)
    profiles_dir.mkdir(parents=True, exist_ok=True)

    print("=== RUNNING LONG-CONTEXT STRESS BENCHMARK ===")
    baseline_agent_stress = BaselineAgent(config, force_offline=True)
    advanced_agent_stress = AdvancedAgent(config, force_offline=True)

    row_baseline_stress = run_agent_benchmark("Baseline Agent (Offline)", baseline_agent_stress, stress_convs, config)
    row_advanced_stress = run_agent_benchmark("Advanced Agent (Offline)", advanced_agent_stress, stress_convs, config)

    print(format_rows([row_baseline_stress, row_advanced_stress]))
    print()


if __name__ == "__main__":
    main()
