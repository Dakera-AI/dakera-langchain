"""Session-scoped memory with LangChain and Dakera.

Demonstrates starting a session, storing memories within it, and
ending with a summary. Sessions group related memories together.

Usage:
    export DAKERA_API_URL="http://localhost:3300"
    python sessions.py
"""

import os

from langchain_dakera import DakeraMemory
from langchain_dakera.sessions import DakeraSessionManager

api_url = os.environ.get("DAKERA_API_URL", "http://localhost:3300")
api_key = os.environ.get("DAKERA_API_KEY", "")

sessions = DakeraSessionManager(
    api_url=api_url,
    api_key=api_key,
    agent_id="langchain-session-demo",
)

session_id = sessions.start(metadata={"topic": "onboarding", "user": "alice"})
print(f"Started session: {session_id}")

memory = DakeraMemory(
    api_url=api_url,
    api_key=api_key,
    agent_id="langchain-session-demo",
    session_id=session_id,
    importance=0.8,
)

memory.save_context(
    {"input": "How do I reset my password?"},
    {"output": "Go to Settings > Security > Reset Password."},
)
memory.save_context(
    {"input": "Where do I find my API keys?"},
    {"output": "Under Settings > Developer > API Keys."},
)

session_memories = sessions.memories(session_id)
print(f"\nMemories in session: {len(session_memories)}")
for m in session_memories:
    print(f"  - [{m['importance']:.1f}] {m['content'][:60]}")

sessions.end(summary="Helped user with password reset and API key location")
print("\nSession ended.")

all_sessions = sessions.list()
print(f"\nTotal sessions: {len(all_sessions)}")
