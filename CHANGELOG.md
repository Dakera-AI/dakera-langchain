# Changelog

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
