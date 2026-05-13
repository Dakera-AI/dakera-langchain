# Changelog

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
