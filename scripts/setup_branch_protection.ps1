# PowerShell script to set up branch protection rules for the main branch
# Requires GitHub CLI (gh) to be installed and authenticated

$REPO = "adikothuri3/Qwen3-VL"
$BRANCH = "main"

Write-Host "Setting up branch protection for $BRANCH branch in $REPO..." -ForegroundColor Cyan

# Check if gh CLI is installed
try {
    $null = gh --version
} catch {
    Write-Host "Error: GitHub CLI (gh) is not installed." -ForegroundColor Red
    Write-Host "Install it from: https://cli.github.com/" -ForegroundColor Yellow
    exit 1
}

# Check if authenticated
try {
    gh auth status 2>&1 | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "Not authenticated"
    }
} catch {
    Write-Host "Error: Not authenticated with GitHub CLI." -ForegroundColor Red
    Write-Host "Run: gh auth login" -ForegroundColor Yellow
    exit 1
}

# Set branch protection rules
Write-Host "Configuring branch protection rules..." -ForegroundColor Cyan

$protectionConfig = @{
    required_status_checks = @{
        strict = $true
        contexts = @("CI/lint", "CI/test")
    }
    enforce_admins = $true
    required_pull_request_reviews = @{
        required_approving_review_count = 1
        dismiss_stale_reviews = $true
        require_code_owner_reviews = $false
        require_last_push_approval = $false
    }
    restrictions = $null
    allow_force_pushes = $false
    allow_deletions = $false
    block_creations = $false
    required_conversation_resolution = $true
    lock_branch = $false
    allow_fork_syncing = $false
} | ConvertTo-Json -Depth 10

$protectionConfig | gh api repos/$REPO/branches/$BRANCH/protection --method PUT --input -

Write-Host "Branch protection rules configured successfully!" -ForegroundColor Green
Write-Host ""
Write-Host "Current protection rules:" -ForegroundColor Cyan
gh api repos/$REPO/branches/$BRANCH/protection | ConvertFrom-Json | ConvertTo-Json -Depth 10
