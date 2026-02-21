import json
import os
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage

HISTORY_ENABLED_DEFAULT = True
HISTORY_FILE_NAME = "message_history.json"
CONTEXT_TOKEN_BUDGET = int(os.getenv("CONTEXT_TOKEN_BUDGET", "4096"))


def estimate_tokens(text: str) -> int:
    """
    Estimate token count for a text string. Prefer a tokenizer if available (tiktoken),
    otherwise fallback to a chars/4 heuristic.
    """
    if not text:
        return 0

    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return max(1, int(len(text) / 4))


def cap_messages(messages, max_tokens: int):
    """
    Trim messages so the estimated token usage does not exceed max_tokens.
    Keeps the most recent messages and always includes the final message.
    """
    if not messages:
        return []

    tokens = [estimate_tokens(getattr(message, "content", "")) for message in messages]

    total = 0
    kept = []
    for message, token_count in reversed(list(zip(messages, tokens))):
        if total + token_count > max_tokens and kept:
            break
        kept.append(message)
        total += token_count

    kept.reverse()

    if messages and kept and kept[-1] is not messages[-1]:
        last_message = messages[-1]
        last_message_tokens = tokens[-1]

        if last_message_tokens >= max_tokens:
            return [last_message]

        total = last_message_tokens
        rebuilt = [last_message]
        for message, token_count in reversed(list(zip(messages[:-1], tokens[:-1]))):
            if total + token_count > max_tokens:
                break
            rebuilt.insert(0, message)
            total += token_count
        return rebuilt

    return kept


def load_history(history_path: Path):
    """
    Load persisted chat history from disk and return LangChain message objects.
    """
    if not history_path.exists():
        return []

    try:
        with history_path.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except Exception:
        return []

    messages = []
    for item in data:
        role = item.get("role")
        content = item.get("content", "")
        if role == "human":
            messages.append(HumanMessage(content=content))
        elif role == "ai":
            messages.append(AIMessage(content=content))

    return messages


def save_history(history_path: Path, messages):
    """
    Persist chat history to disk in JSON format with UTC ISO timestamps.
    """
    payload = []
    for message in messages:
        timestamp = datetime.now(timezone.utc).isoformat()
        if isinstance(message, HumanMessage):
            payload.append({"role": "human", "content": message.content, "timestamp": timestamp})
        elif isinstance(message, AIMessage):
            payload.append({"role": "ai", "content": message.content, "timestamp": timestamp})

    with history_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def reset_history(history_path: Path):
    """
    Delete persisted history file if it exists.
    """
    if history_path.exists():
        history_path.unlink()
