# Security policy

## Reporting a vulnerability

Please report security problems privately. Do not open a public issue.

1. Go to the repository's **Security** tab: https://github.com/aditya-forge/RiskSynth-AI/security
2. Choose **Report a vulnerability** and describe the problem, the affected file or page, and how to reproduce it.

You should get a first reply within 7 days. Please give the issue time to be fixed before you disclose it publicly.

## Scope

In scope:

- the `risksynth` package in `src/`;
- the Streamlit workbench in `app/` and the live deployment at https://risksynth.streamlit.app/;
- the Dockerfile and the GitHub Actions workflow.

Out of scope:

- findings that need physical access or a compromised machine;
- denial of service by sending very large generation requests;
- issues in third-party dependencies that are already reported upstream. Report those to the dependency.

## How the project handles data

- The workbench generates synthetic profiles only. It does not accept uploads, ask for credentials or store personal data.
- The inputs are INE public-use microdata, anonymised by INE before publication.
- Generated runs are written to the instance's local disk under `data/synthetic/` and are lost when the instance restarts.
- Secrets are never committed. `.env` and `.streamlit/secrets.toml` are listed in `.gitignore`, and `.env.example` holds placeholders only.

## Supported versions

Only the latest commit on `main` receives fixes.

## Licensing

This repository is proprietary. Reporting a vulnerability does not grant any right to copy or reuse the code. See [LICENSE](LICENSE).
