# Security Policy

BloRA is proprietary software of PT Raya Laksana Teknologi (ralafs.com).

## Reporting a Vulnerability

Please report security issues **privately** — do not open a public issue.

- Email: **demo@ralafs.com**
- Include: affected version/commit, reproduction steps, and impact.
- We aim to acknowledge within 3 business days.

## Supported Versions

| Version | Supported |
|---------|-----------|
| 1.x     | ✅        |
| < 1.0   | ❌        |

## Handling Secrets

- Never commit credentials. HuggingFace tokens, API keys, and `*.key` files
  are git-ignored; CI runs `detect-private-key` via pre-commit.
- Model weights and evaluation dumps are git-ignored and must not be pushed.
- If a token is exposed (e.g. in logs or chat), rotate it immediately at the
  provider.
