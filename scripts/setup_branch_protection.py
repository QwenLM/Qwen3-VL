#!/usr/bin/env python3
"""
Set up branch protection rules for the main branch via the GitHub API.

Requires a GitHub Personal Access Token with 'repo' scope.
  Create one at: https://github.com/settings/tokens

Usage:
  $env:GITHUB_TOKEN='ghp_...'   # PowerShell
  python scripts/setup_branch_protection.py
"""

import json
import os
import sys

import requests

REPO = "adikothuri3/Qwen3-VL"
BRANCH = "main"
API = "https://api.github.com"

# These must match the job `name:` fields in .github/workflows/ci.yml exactly.
REQUIRED_CHECKS = ["Lint", "Type Check"]


def get_token() -> str:
    token = os.environ.get("GITHUB_TOKEN", "").strip()
    if token:
        return token

    print("GITHUB_TOKEN not set.")
    print("Create a token at https://github.com/settings/tokens (scope: repo)")
    print()
    token = input("Paste token here (not saved): ").strip()
    if not token:
        sys.exit(1)
    return token


def apply_protection(token: str) -> dict:
    url = f"{API}/repos/{REPO}/branches/{BRANCH}/protection"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
    }
    body = {
        "required_status_checks": {
            "strict": True,
            "contexts": REQUIRED_CHECKS,
        },
        "enforce_admins": True,
        "required_pull_request_reviews": {
            "required_approving_review_count": 1,
            "dismiss_stale_reviews": True,
        },
        "restrictions": None,
        "allow_force_pushes": False,
        "allow_deletions": False,
        "required_conversation_resolution": True,
    }

    resp = requests.put(url, headers=headers, json=body)
    if not resp.ok:
        print(f"Error {resp.status_code}: {resp.text}")
        sys.exit(1)
    return resp.json()


def main() -> None:
    token = get_token()
    print(f"Applying branch protection to {REPO} ({BRANCH})...")
    result = apply_protection(token)
    print("Done. Current rules:")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
