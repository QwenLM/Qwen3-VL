#!/bin/bash

# Script to set up branch protection rules for the main branch
# Requires GitHub CLI (gh) to be installed and authenticated

set -e

REPO="adikothuri3/Qwen3-VL"
BRANCH="main"

echo "Setting up branch protection for $BRANCH branch in $REPO..."

# Check if gh CLI is installed
if ! command -v gh &> /dev/null; then
    echo "Error: GitHub CLI (gh) is not installed."
    echo "Install it from: https://cli.github.com/"
    exit 1
fi

# Check if authenticated
if ! gh auth status &> /dev/null; then
    echo "Error: Not authenticated with GitHub CLI."
    echo "Run: gh auth login"
    exit 1
fi

# Set branch protection rules
echo "Configuring branch protection rules..."

gh api repos/$REPO/branches/$BRANCH/protection \
  --method PUT \
  --field required_status_checks='{"strict":true,"contexts":["CI/lint","CI/test"]}' \
  --field enforce_admins=true \
  --field required_pull_request_reviews='{"required_approving_review_count":1,"dismiss_stale_reviews":true,"require_code_owner_reviews":false,"require_last_push_approval":false}' \
  --field restrictions=null \
  --field allow_force_pushes=false \
  --field allow_deletions=false \
  --field block_creations=false \
  --field required_conversation_resolution=true \
  --field lock_branch=false \
  --field allow_fork_syncing=false

echo "Branch protection rules configured successfully!"
echo ""
echo "Current protection rules:"
gh api repos/$REPO/branches/$BRANCH/protection | jq '.'
