# Changelog

## [0.4.0] - 2026-10-27

Dakera server **v0.12.3** support. Requires the `dakera` Python SDK **0.14.1** or later; works
with Dakera server v0.12.0 – v0.12.3 (the v0.12.3 additions are ignored by older servers).

### Added
- `DakeraMemory.store(..., event_time=)`: when the memory's event happened (Unix seconds, an
  RFC 3339 string or a timezone-aware `datetime`); the server keeps it as `_dakera_event_time` and
  uses it for recency and `since` / `until`.
- `DakeraMemory.recall(..., rerank=)` and a `score` (the server's `rank_score`) on each result.
- `DakeraMemory.consolidate(memory_ids=None, *, target_type=None)` merges the given memories.
- `DakeraSessionManager.start(..., idle_timeout_secs=)` and `touch()`: the server ends a session
  after 4 h without activity by default. `get()` and `list_sessions()` return `ended_reason`;
  `get()` also returns the session `summary`.
- `DakeraAgentTools.import_memories(..., format=)` / `export_memories(format=)` (`jsonl`, `csv`,
  `mem0`, `zep`).
- `tests/test_integration.py`: live tests against a server, skipped unless `DAKERA_API_URL` is set.

### Changed
- Dependency: `dakera>=0.14.1` (was `>=0.13.1`).
- `DakeraMemory` recalls (`load_memory_variables`, `recall()`, `batch_recall()`) return at most
  `recall_k` / `top_k` ranked memories. A v0.12.3 server appends a session's neighbouring memories
  to the results by default, so a recall could return more than `top_k` and put unranked context
  into the prompt history; recalls now ask for `neighborhood_mode="separate"` and keep the ranked
  results only (a v0.12.2 server's inline context is cut at `top_k`).
- `DakeraSessionManager.memories()` leaves out the sentence sub-memories the server derives from
  each stored memory (tagged `dakera-curated`), which `GET /v1/sessions/{id}/memories` lists next
  to them; `include_derived=True` keeps them.

### Fixed
- `DakeraAgentTools.import_memories()` failed against every v0.12 server (the SDK posted JSON);
  with dakera 0.14.1 it uploads the file the server reads. `jsonl` takes the records
  `export_memories()` returns.

### CI
- Integration tests and examples run against `ghcr.io/dakera-ai/dakera:0.12.3`. That image ships
  no `curl`, so the service container has no in-container health check; the job polls `/health`
  from the runner instead.

## [0.3.0] - 2026-10-01

Dakera server **v0.12.0** support. Requires the `dakera` Python SDK **0.13.1** or later;
works with Dakera server v0.12.0 and v0.11.108.

### Changed
- Dependency: `dakera>=0.13.1` (was `>=0.8.6`).
- `DakeraVectorStore.ahybrid_search()` runs the hybrid search in a worker thread. It
  previously called `AsyncDakeraClient.hybrid_search()` without the query vector that
  method requires, and always raised `TypeError`.
- `DakeraKnowledgeGraph.link(source_id, target_id, *, label=None)` sends the agent id the server
  requires (every link was a 422 before) and returns the server's `{from_id, to_id, edge_type}`.
  The `edge_type` argument is gone: the server records every explicit link as `linked_by`.
- `DakeraKnowledgeGraph.summarize(memory_ids, *, target_type=None)` needs at least two ids and has
  no `dry_run`: the server has no dry run and always stores the summary (dakera 0.13.1 raises on
  `dry_run=True` rather than silently writing).
- `DakeraKnowledgeGraph.build(memory_id, depth=None)`: the seed `memory_id` is required by the server.
- `DakeraMemory.recall()` and `DakeraMemory.search()` take `tags` (memories carrying at least one
  of them); dakera 0.13.1 forwards them to the server.

### Removed
- `DakeraEntityExtractor.list_providers()`. It called `GET /v1/extract/providers`, which no
  Dakera server serves; dakera 0.13.0 removed `list_extract_providers()`. Use the new
  `DakeraEntityExtractor.get_extractor(namespace)` (`GET /v1/namespaces/{ns}/extractor`)
  to read a namespace's configured provider.

### Fixed
- `DakeraSessionManager.list_sessions()` unwraps the `{"sessions": [...], "total": n}`
  answer of `GET /v1/sessions`; it iterated over the envelope's keys and raised
  `AttributeError`.
- Examples `advanced_memory.py`, `batch_operations.py`, `hybrid_search.py` and
  `sessions.py` used methods and arguments the package does not have; they now use the
  real API and run against a v0.12.0 server.
- CI: integration and example jobs run against `ghcr.io/dakera-ai/dakera:0.12.0`; the
  security audit upgrades `setuptools` too (the Python 3.10 image ships a vulnerable one).

### Tests
- `tests/test_sdk_surface.py` checks the SDK calls with `create_autospec(DakeraClient)`, so
  a method or argument the SDK does not have fails the test.

## [0.1.1] - 2026-05-13

### Fixed
- **Mutable aliasing bug in vectorstore** (`DakeraVectorStore.add_texts`): default mutable argument replaced with `None` sentinel
- **Improved error handling**: `assert` statements converted to `RuntimeError` for clearer debugging in production
- **README**: corrected `recall_k` parameter name in code examples
- **Removed unused import** in `memory.py`

### Changed
- Bumped GitHub Actions: `actions/checkout` v4 → v6, `actions/setup-python` v5 → v6

### Added
- Community health files: `CONTRIBUTING.md`, `SECURITY.md`, issue templates, PR template

## [0.1.0] - 2026-05-13

### Added
- Initial release — LangChain integration for Dakera AI memory platform
- `DakeraMemory` class for conversational memory with `save_context` / `load_memory_variables`
- `DakeraVectorStore` class for vector similarity search over Dakera memory
- PyPI publish via OIDC Trusted Publisher
