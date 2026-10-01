# Changelog

## [0.3.0] - 2026-10-01

Dakera server **v0.12.0** support. Requires the `dakera` Python SDK **0.13.0** or later;
works with Dakera server v0.12.0 and v0.11.108.

### Changed
- Dependency: `dakera>=0.13.0` (was `>=0.8.6`).
- `DakeraKnowledgeGraph.summarize()` now takes the `memory_ids` to summarize (plus
  `target_type` and `dry_run`). `POST /v1/knowledge/summarize` requires them, so the old
  no-argument call always failed with a 422. An empty list raises `ValueError`.
- `DakeraVectorStore.ahybrid_search()` runs the hybrid search in a worker thread. It
  previously called `AsyncDakeraClient.hybrid_search()` without the query vector that
  method requires, and always raised `TypeError`.

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
