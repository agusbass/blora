# Contributing to BloRA

BloRA is proprietary (see [LICENSE](LICENSE)). Contributions are accepted from
authorized collaborators of PT Raya Laksana Teknologi (ralafs.com).

## Development setup

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
make dev-install      # editable install + dev tools + pre-commit hooks
```

## Before you push

```bash
make check            # ruff lint + ruff format check + mypy + pytest
```

CI runs the same on Python 3.10/3.11/3.12 and must be green to merge.

## Standards

- **Style:** `ruff format` (line length 100). No manual formatting debates.
- **Typing:** public functions in `src/blora/` carry type hints; `mypy src/`
  must pass.
- **Tests:** every behavioural change ships a test. Numerical claims
  (train≡eval, merged≡unmerged, batched≡per-adapter) must be asserted, not
  described. Tests are CPU-only and ASCII-only (Windows-safe).
- **No weights / secrets in git.** Checkpoints, eval dumps, and tokens are
  git-ignored — keep it that way.
- **Commits:** small and focused; imperative subject ("Add batched BCMS FFT").

## Branching

Work on a feature branch, open a PR against `main`. Do not commit directly to
`main`.
