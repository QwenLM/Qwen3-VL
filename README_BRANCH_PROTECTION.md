# Branch Protection Setup Guide

This repository uses branch protection rules to ensure code quality and prevent direct commits to the main branch.

## Branch Protection Rules

The main branch has the following protection rules:

- ✅ **Required status checks**: CI/lint and CI/test must pass
- ✅ **Required pull request reviews**: At least 1 approval required
- ✅ **Dismiss stale reviews**: Stale PR reviews are dismissed when new commits are pushed
- ✅ **Required conversation resolution**: All PR conversations must be resolved before merging
- ✅ **Enforce admins**: Even admins must follow these rules
- ❌ **Force pushes**: Not allowed
- ❌ **Deletions**: Not allowed

## Setting Up Branch Protection

You can set up branch protection using one of the following methods:

### Method 1: Using GitHub CLI (Recommended)

**For Linux/Mac:**
```bash
chmod +x scripts/setup_branch_protection.sh
./scripts/setup_branch_protection.sh
```

**For Windows (PowerShell):**
```powershell
.\scripts\setup_branch_protection.ps1
```

**Prerequisites:**
- Install GitHub CLI: https://cli.github.com/
- Authenticate: `gh auth login`

### Method 2: Using Python Script

```bash
export GITHUB_TOKEN=your_github_token
python scripts/setup_branch_protection.py
```

**Prerequisites:**
- Python 3.6+
- `requests` library: `pip install requests`
- GitHub Personal Access Token with `repo` scope
- Create token at: https://github.com/settings/tokens

### Method 3: Manual Setup via GitHub Web UI

1. Go to: `https://github.com/adikothuri3/Qwen3-VL/settings/branches`
2. Click "Add rule" or edit the existing rule for `main`
3. Configure the following:
   - ✅ Require a pull request before merging
     - Require approvals: 1
     - Dismiss stale pull request approvals when new commits are pushed
   - ✅ Require status checks to pass before merging
     - Require branches to be up to date before merging
     - Status checks: `CI/lint`, `CI/test`
   - ✅ Require conversation resolution before merging
   - ✅ Include administrators
   - ❌ Allow force pushes
   - ❌ Allow deletions

## CI/CD Workflows

The repository includes the following CI workflows:

- **`.github/workflows/ci.yml`**: Runs linting, type-checking, and syntax validation
- **`.github/workflows/pre-commit.yml`**: Runs pre-commit hooks on PRs

## Pre-commit Hooks (Local Development)

To run checks locally before committing:

```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

This will run:
- Trailing whitespace checks
- File formatting (Ruff)
- Type checking (MyPy)
- Linting (Pylint)
- Large file detection
- And more...

## Workflow

1. Create a feature branch: `git checkout -b feature/my-feature`
2. Make your changes
3. Run pre-commit hooks: `pre-commit run --all-files`
4. Commit and push: `git push origin feature/my-feature`
5. Create a Pull Request targeting `main`
6. Wait for CI checks to pass
7. Get at least 1 code review approval
8. Resolve any conversations
9. Merge the PR

## Troubleshooting

### CI Checks Failing

If CI checks fail:
1. Check the Actions tab: https://github.com/adikothuri3/Qwen3-VL/actions
2. Review the error messages
3. Fix the issues locally
4. Push new commits to your PR branch

### Can't Push to Main

If you see an error when trying to push directly to main:
- This is expected! Branch protection is working
- Create a feature branch and open a PR instead

### Permission Errors

If you get permission errors when running the setup scripts:
- Make sure your GitHub token has `repo` scope
- For GitHub CLI, ensure you're authenticated: `gh auth status`
