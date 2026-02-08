#!/usr/bin/env python3
"""
Script to set up branch protection rules for the main branch using GitHub API.
Requires a GitHub Personal Access Token with repo permissions.
"""

import os
import sys
import json
import requests
from typing import Dict, Any

REPO = "adikothuri3/Qwen3-VL"
BRANCH = "main"
GITHUB_API = "https://api.github.com"


def get_github_token() -> str:
    """Get GitHub token from environment or prompt user."""
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        print("=" * 60)
        print("GitHub Personal Access Token Required")
        print("=" * 60)
        print("\nTo set up branch protection, you need a GitHub token.")
        print("\n1. Create a token at: https://github.com/settings/tokens")
        print("   - Click 'Generate new token' -> 'Generate new token (classic)'")
        print("   - Give it a name like 'Branch Protection Setup'")
        print("   - Select scope: 'repo' (Full control of private repositories)")
        print("   - Click 'Generate token' and copy it")
        print("\n2. Set it as an environment variable:")
        print("   Windows PowerShell: $env:GITHUB_TOKEN='your_token_here'")
        print("   Windows CMD: set GITHUB_TOKEN=your_token_here")
        print("   Linux/Mac: export GITHUB_TOKEN=your_token_here")
        print("\n3. Run this script again")
        print("\nOr enter your token now (will not be saved):")
        token = input("GitHub Token: ").strip()
        if not token:
            sys.exit(1)
    return token


def setup_branch_protection(token: str) -> Dict[str, Any]:
    """Set up branch protection rules."""
    url = f"{GITHUB_API}/repos/{REPO}/branches/{BRANCH}/protection"
    
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
        "Content-Type": "application/json"
    }
    
    protection_config = {
        "required_status_checks": {
            "strict": True,
            "contexts": ["CI/lint", "CI/test"]
        },
        "enforce_admins": True,
        "required_pull_request_reviews": {
            "required_approving_review_count": 1,
            "dismiss_stale_reviews": True,
            "require_code_owner_reviews": False,
            "require_last_push_approval": False
        },
        "restrictions": None,
        "allow_force_pushes": False,
        "allow_deletions": False,
        "block_creations": False,
        "required_conversation_resolution": True,
        "lock_branch": False,
        "allow_fork_syncing": False
    }
    
    print(f"Setting up branch protection for {BRANCH} branch in {REPO}...")
    
    try:
        response = requests.put(url, headers=headers, json=protection_config)
        response.raise_for_status()
        print("✓ Branch protection rules configured successfully!")
        return response.json()
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 403:
            print("Error: Insufficient permissions. Make sure your token has 'repo' scope.")
        elif e.response.status_code == 404:
            print(f"Error: Repository or branch not found. Check that {REPO} and {BRANCH} exist.")
        else:
            print(f"Error: {e.response.status_code} - {e.response.text}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def get_current_protection(token: str) -> Dict[str, Any]:
    """Get current branch protection rules."""
    url = f"{GITHUB_API}/repos/{REPO}/branches/{BRANCH}/protection"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json"
    }
    
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            return response.json()
        return {}
    except Exception as e:
        print(f"Warning: Could not fetch current protection rules: {e}")
        return {}


def main():
    token = get_github_token()
    
    print(f"Repository: {REPO}")
    print(f"Branch: {BRANCH}")
    print()
    
    # Get current protection
    current = get_current_protection(token)
    if current:
        print("Current protection rules:")
        print(json.dumps(current, indent=2))
        print()
    
    # Set up protection
    result = setup_branch_protection(token)
    
    print("\nNew protection rules:")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
