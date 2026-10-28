"""Session-scoped memory with LangChain and Dakera.

Starts a session, stores conversation turns in it, keeps it alive, reads it
back and ends it with a summary. Sessions group the memories of one
conversation; the server ends a session after 4 h without activity by default.

Run a Dakera server first (see the README), then:

    pip install langchain-dakera
    export DAKERA_API_URL=http://localhost:3000
    export DAKERA_API_KEY=dk-mykey   # the server's root key; leave unset if auth is off
    python examples/sessions.py
"""

import os
import uuid

from langchain_dakera import DakeraMemory, DakeraSessionManager

api_url = os.environ.get("DAKERA_API_URL", "http://localhost:3000")
api_key = os.environ.get("DAKERA_API_KEY", "")
agent_id = f"langchain-sessions-{uuid.uuid4().hex[:8]}"

sessions = DakeraSessionManager(api_url=api_url, api_key=api_key, agent_id=agent_id)

# idle_timeout_secs (server v0.12.2+): end this session after 30 min without activity.
session_id = sessions.start(metadata={"topic": "onboarding"}, idle_timeout_secs=1800)
print(f"Started session {session_id}")

memory = DakeraMemory(
    api_url=api_url,
    api_key=api_key,
    agent_id=agent_id,
    session_id=session_id,  # every stored turn belongs to the session
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

print(f"Touched the session: {sessions.touch()}")  # keeps an idle session open

print("\n--- Memories in the session ---")
for m in sessions.memories(session_id):
    print(f"  [{m['importance']:.1f}] {m['content'].splitlines()[0]}")

sessions.end(summary="Helped with password reset and API key location.")
info = sessions.get(session_id)
print(f"\nEnded: {info['summary']!r} ({info['memory_count']} memories)")

# As a context manager the session starts on entry and ends on exit.
with DakeraSessionManager(api_url=api_url, api_key=api_key, agent_id=agent_id) as follow_up:
    print(f"\nFollow-up session {follow_up.active_session_id} is active")

print(f"Sessions of the agent: {len(sessions.list_sessions())}")
