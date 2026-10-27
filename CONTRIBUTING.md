# Contributing to dakera-langchain

Thank you for your interest in contributing to the Dakera LangChain integration! This guide covers everything you need to get started.

## Reporting Bugs

Use the [Bug Report](https://github.com/Dakera-AI/dakera-langchain/issues/new?template=bug_report.md) template. Please include:
- A clear description of the bug and steps to reproduce
- Your Python, LangChain, `dakera` and `langchain-dakera` versions, and the Dakera server version (`curl http://localhost:3000/health`)
- Which component you are using (`DakeraMemory`, `DakeraVectorStore`, `DakeraSessionManager`, ...)
- Relevant error messages or tracebacks

## Suggesting Features

Use the [Feature Request](https://github.com/Dakera-AI/dakera-langchain/issues/new?template=feature_request.md) template. Describe the problem you are solving, your proposed solution, and any alternatives you have considered.

## Security Vulnerabilities

**Do not open public issues for security vulnerabilities.** See [SECURITY.md](.github/SECURITY.md) for responsible disclosure instructions — report via [GitHub Security Advisories](https://github.com/Dakera-AI/dakera-langchain/security/advisories/new).

## Pull Request Process

1. Fork the repository and create a branch from `main`:
   ```bash
   git checkout -b fix/your-fix-name
   ```
2. Make your changes, add tests, and run the checks below
3. Add an entry to [CHANGELOG.md](CHANGELOG.md)
4. Open a pull request against `main` with a clear description

## Development Setup

**Prerequisites:** Python 3.10+; Docker for the live tests.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

The checks CI runs (`.github/workflows/ci.yml`):

```bash
ruff check .                              # lint
ruff format --check .                     # formatting (`ruff format .` fixes it)
mypy src/ --ignore-missing-imports        # type check
pytest tests/ -v                          # unit tests; live tests are skipped without a server
pip-audit                                 # known vulnerabilities in the environment
```

## Code Style

- **Formatter / linter**: `ruff format` and `ruff check` (line length 100)
- **Types**: every public API is typed and passes `mypy`
- **Docstrings**: every public class and method has one; keep them accurate when behaviour changes
- **Tests**: new behaviour needs tests. Unit tests mock the SDK with
  `create_autospec(DakeraClient, instance=True)` (see `tests/test_sdk_surface.py`), so a call
  that does not match the real `dakera` SDK signature fails the test.

## Testing Against a Live Dakera Server

`tests/test_integration.py` runs against a real server and is skipped unless `DAKERA_API_URL` is
set. CI runs it and every example against `ghcr.io/dakera-ai/dakera:0.12.3`
(`.github/workflows/integration.yml`). Locally:

```bash
docker run -d --name dakera -p 3000:3000 \
  -e DAKERA_ROOT_API_KEY=test-key \
  ghcr.io/dakera-ai/dakera:0.12.3
curl http://localhost:3000/health   # wait for "healthy"

export DAKERA_API_URL=http://localhost:3000
export DAKERA_API_KEY=test-key
pytest tests/ -v                    # unit + live tests
for f in examples/*.py; do python "$f" || break; done
```

The examples read the same two variables. Live tests use a fresh agent id or namespace per test,
so they can run repeatedly against one server.

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).
