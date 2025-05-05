# GitHub Actions Workflows

This directory contains GitHub Actions workflows for the Financial Security System project.

## Workflows

### Run Python Tests (`python-test.yml`)

This workflow runs tests when Python files are changed.

- **Triggers**:
  - Pull requests that modify Python files.
  - Pushes to the `main` branch that modify Python files.
  - Manual.
- **Actions**:
  - Sets up Python environment
  - Syncs virtual environments using `sync_venvs.sh`
  - Runs tests using `run_all_tests.sh`

## Local Development vs. CI/CD

- In local development, pre-commit hooks include running tests when Python files change
- In GitHub Actions, the test hook is skipped in pre-commit and run separately in dedicated workflows
