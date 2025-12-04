# Quick Setup Script for MAF Agent Examples
# Run this script to set up your environment

Write-Host "=" -ForegroundColor Cyan -NoNewline
Write-Host ("=" * 59) -ForegroundColor Cyan
Write-Host "Microsoft Agent Framework - Quick Setup" -ForegroundColor Cyan
Write-Host "=" -ForegroundColor Cyan -NoNewline
Write-Host ("=" * 59) -ForegroundColor Cyan
Write-Host ""

# Step 1: Install packages
Write-Host "[1/4] Installing required packages..." -ForegroundColor Yellow
try {
    uv pip install agent-framework --prerelease=allow
    uv pip install python-dotenv rich
    Write-Host "✓ Packages installed successfully" -ForegroundColor Green
} catch {
    Write-Host "✗ Failed to install packages. Please run manually:" -ForegroundColor Red
    Write-Host "  uv pip install agent-framework --prerelease=allow" -ForegroundColor White
    Write-Host "  uv pip install python-dotenv rich" -ForegroundColor White
    exit 1
}

Write-Host ""

# Step 2: Check for .env file
Write-Host "[2/4] Checking environment configuration..." -ForegroundColor Yellow
if (Test-Path ".env") {
    Write-Host "✓ .env file found" -ForegroundColor Green
} else {
    if (Test-Path ".env.sample") {
        Copy-Item ".env.sample" ".env"
        Write-Host "✓ Created .env from .env.sample" -ForegroundColor Green
        Write-Host "⚠ Please edit .env file with your Azure OpenAI details" -ForegroundColor Yellow
    } else {
        Write-Host "✗ .env.sample not found" -ForegroundColor Red
        exit 1
    }
}

Write-Host ""

# Step 3: Check Azure CLI login
Write-Host "[3/4] Checking Azure CLI authentication..." -ForegroundColor Yellow
try {
    $account = az account show 2>$null | ConvertFrom-Json
    if ($account) {
        Write-Host "✓ Logged in as: $($account.user.name)" -ForegroundColor Green
    } else {
        throw "Not logged in"
    }
} catch {
    Write-Host "✗ Not logged in to Azure CLI" -ForegroundColor Red
    Write-Host "  Run: az login" -ForegroundColor White
}

Write-Host ""

# Step 4: Summary
Write-Host "[4/4] Setup Summary" -ForegroundColor Yellow
Write-Host "✓ Dependencies installed" -ForegroundColor Green
Write-Host "✓ Environment file ready" -ForegroundColor Green
Write-Host ""
Write-Host "Next Steps:" -ForegroundColor Cyan
Write-Host "1. Edit .env file with your Azure OpenAI credentials" -ForegroundColor White
Write-Host "2. Ensure you're logged in: az login" -ForegroundColor White
Write-Host "3. Run example scripts:" -ForegroundColor White
Write-Host "   python 1_azure_openai_chat_agent.py" -ForegroundColor Gray
Write-Host "   python 2_magentic_orchestration.py" -ForegroundColor Gray
Write-Host "   python 3_agent_as_tool.py" -ForegroundColor Gray
Write-Host ""
Write-Host "=" -ForegroundColor Cyan -NoNewline
Write-Host ("=" * 59) -ForegroundColor Cyan
Write-Host "Setup Complete!" -ForegroundColor Green -NoNewline
Write-Host " Ready to run examples." -ForegroundColor White
Write-Host "=" -ForegroundColor Cyan -NoNewline
Write-Host ("=" * 59) -ForegroundColor Cyan
