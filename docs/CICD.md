# CI/CD Pipeline Documentation

**Continuous Integration and Deployment for Trading Bot**

This document explains the automated testing, building, and deployment pipeline.

---

## Overview

**Current CI/CD Stack:**
- **Platform:** GitHub Actions
- **Docker Registry:** GitHub Container Registry (ghcr.io)
- **Testing:** pytest, ruff, black, mypy
- **Security:** Bandit, TruffleHog, Trivy
- **Deployment:** Docker images

**Workflows:**
1. `.github/workflows/ci.yml` - Code testing and linting
2. `.github/workflows/docker-build.yml` - Docker build and push
3. `.github/workflows/trading_bot.yml` - (Existing, needs review)

---

## Workflow: CI (Code Quality)

**File:** `.github/workflows/ci.yml`

**Triggers:**
- Push to `main` or `develop` branches
- Pull requests to `main` or `develop`

**Jobs:**

### 1. Test (Matrix: Ubuntu + macOS, Python 3.10 + 3.11)

**Steps:**
1. Checkout code
2. Set up Python
3. Install UV (fast Python package installer)
4. Install TA-Lib (platform-specific)
   - Linux: Build from source
   - macOS: Install via Homebrew
5. Install Python dependencies
6. Lint with ruff
7. Format check with black
8. Type check with mypy (continue-on-error)
9. Run tests with pytest + coverage
10. Upload coverage to Codecov

**Environment Variables (Secrets Required):**
- `ALPACA_API_KEY_TEST` - Paper trading API key for tests
- `ALPACA_SECRET_KEY_TEST` - Paper trading secret key

**Coverage:**
- Targets: strategies/, brokers/, engine/, utils/
- Report: XML format
- Upload: Codecov.io

### 2. Security

**Steps:**
1. Bandit security linter
   - Scans for common security issues
   - Output: JSON report
   - Continue on error (warnings only)

2. TruffleHog secret scanner
   - Checks for accidentally committed secrets
   - Scans entire git history
   - Fails build if secrets found

**Why This Matters:**
- Prevents committing API keys to git
- Catches security vulnerabilities early
- Compliance requirement for financial software

### 3. Dependency Check

**Steps:**
1. Install Safety
2. Check for known vulnerabilities in dependencies
3. Continue on error (warnings only)

**Why Continue on Error:**
- Don't block development on low-severity CVEs
- Team can review and decide on patching
- Production images scanned separately (Trivy)

### 4. Build (Main Branch Only)

**Steps:**
1. Build Python package
2. Check package with twine
3. Upload dist/ artifacts

**When:**
- Only on push to `main`
- After tests pass
- Creates distributable package

---

## Workflow: Docker Build

**File:** `.github/workflows/docker-build.yml`

**Triggers:**
- Push to `main` branch
- Tags matching `v*` (e.g., v1.0.0)
- Pull requests to `main`

**Jobs:**

### 1. Build and Push

**Steps:**
1. Checkout code
2. Set up Docker Buildx (multi-platform builds)
3. Login to GitHub Container Registry
4. Extract metadata (tags, labels)
5. Build and push Docker image
   - Multi-platform: linux/amd64, linux/arm64
   - Cache layers for faster rebuilds
   - Tag with version, branch, SHA
6. Run Trivy vulnerability scanner
7. Upload security report to GitHub Security tab

**Image Tags Created:**
- `main` - Latest from main branch
- `latest` - Alias for main
- `pr-123` - PR number
- `sha-abc123` - Git commit SHA
- `v1.0.0` - Semver tags

**Platforms:**
- linux/amd64 - Intel/AMD servers, most cloud VMs
- linux/arm64 - AWS Graviton, Apple Silicon

**Security Scanning:**
- Trivy scans for:
  - OS vulnerabilities
  - Python package vulnerabilities
  - Misconfigurations
- Results uploaded to GitHub Security tab
- Fails build on HIGH/CRITICAL vulnerabilities

### 2. Test Docker (Pull Requests Only)

**Steps:**
1. Build Docker image locally
2. Test that image runs
3. Test API connectivity
4. Validate docker-compose config

**Why:**
- Ensures Dockerfile changes don't break builds
- Catches issues before merging to main
- Validates docker-compose syntax

---

## Secrets Configuration

**Required Secrets (GitHub Repository Settings):**

### Testing Secrets
```
ALPACA_API_KEY_TEST
ALPACA_SECRET_KEY_TEST
```

**Purpose:**
- Used in CI tests
- Should be paper trading keys
- Never use live trading keys in CI

**How to Add:**
1. Go to GitHub repo → Settings → Secrets and variables → Actions
2. Click "New repository secret"
3. Name: `ALPACA_API_KEY_TEST`
4. Value: Your paper trading API key
5. Repeat for `ALPACA_SECRET_KEY_TEST`

### Production Secrets (For Deployment)

**Not currently used, but would be:**
```
ALPACA_API_KEY_LIVE
ALPACA_SECRET_KEY_LIVE
DOCKER_USERNAME
DOCKER_PASSWORD
```

**Security Best Practices:**
- Never log secrets
- Use separate keys for CI/prod
- Rotate keys periodically
- Use minimal permissions (read-only for tests)

---

## Build Status Badges

Add to README.md:

```markdown
[![CI](https://github.com/yourusername/trading-bot/actions/workflows/ci.yml/badge.svg)](https://github.com/yourusername/trading-bot/actions/workflows/ci.yml)
[![Docker Build](https://github.com/yourusername/trading-bot/actions/workflows/docker-build.yml/badge.svg)](https://github.com/yourusername/trading-bot/actions/workflows/docker-build.yml)
[![codecov](https://codecov.io/gh/yourusername/trading-bot/branch/main/graph/badge.svg)](https://codecov.io/gh/yourusername/trading-bot)
```

---

## Deployment Workflow (Recommended)

**Current State:** Manual deployment
**Recommended:** Automated deployment after validation

### Deployment Strategy

**Option A: Continuous Deployment (Risky for Trading)**
```yaml
# .github/workflows/deploy.yml
on:
  push:
    branches: [ main ]

jobs:
  deploy:
    needs: [test, build-and-push]
    runs-on: ubuntu-latest
    steps:
      - name: Deploy to production
        # SSH into server, pull latest image, restart
```

**❌ NOT RECOMMENDED for trading bot:**
- Auto-deploying untested code to production = financial risk
- Need manual validation before deployment

**Option B: Manual Deployment (Recommended)**
```yaml
# .github/workflows/deploy.yml
on:
  workflow_dispatch:  # Manual trigger only
    inputs:
      environment:
        description: 'Environment to deploy'
        required: true
        type: choice
        options:
          - staging
          - production

jobs:
  deploy:
    # Deploy only when manually triggered
    # After paper trading validation
```

**✅ RECOMMENDED:**
- CI builds and tests automatically
- Docker images pushed automatically
- Deployment triggered manually
- Requires code review + testing

---

## Pull Request Workflow

**Typical PR Process:**

1. **Developer Creates PR**
   - Push branch to GitHub
   - Open pull request to `main`

2. **CI Runs Automatically**
   - Linting (ruff, black)
   - Type checking (mypy)
   - Tests (pytest)
   - Security scans (Bandit, TruffleHog)
   - Docker build test

3. **Review Checks**
   - All CI jobs must pass (green checkmarks)
   - Code review required
   - No merge conflicts

4. **Merge to Main**
   - After approval + passing CI
   - Triggers:
     - Build job (create package)
     - Docker build and push
     - Image tagged with commit SHA

5. **Manual Deployment (After Validation)**
   - Pull latest Docker image
   - Deploy to staging/production
   - Monitor for issues

---

## Testing Locally Before Push

**Pre-Push Checklist:**

```bash
# 1. Run linters
pip install ruff black mypy
ruff check .
black --check .
mypy strategies/ brokers/ engine/ utils/

# 2. Run tests
pytest tests/ -v

# 3. Test Docker build
docker build -t trading-bot-test .

# 4. Test Docker run
docker run --rm \
  -e ALPACA_API_KEY="${ALPACA_API_KEY}" \
  -e ALPACA_SECRET_KEY="${ALPACA_SECRET_KEY}" \
  -e PAPER=True \
  trading-bot-test \
  python -c "print('Docker test passed')"

# 5. Test docker-compose
docker-compose config
docker-compose build
```

**If all pass locally → Safe to push**

---

## Monitoring CI/CD

### GitHub Actions Tab

**View Workflow Runs:**
1. Go to GitHub repo
2. Click "Actions" tab
3. See all workflow runs (past and current)

**Status Indicators:**
- ✅ Green checkmark - Passed
- ❌ Red X - Failed
- 🟡 Yellow dot - In progress
- ⭕ Gray circle - Skipped/canceled

### Notifications

**Configure Email Alerts:**
1. GitHub → Settings → Notifications
2. Enable "Actions" notifications
3. Get email on workflow failures

**Slack Integration (Optional):**
```yaml
# Add to workflow
- name: Notify Slack
  if: failure()
  uses: 8398a7/action-slack@v3
  with:
    status: ${{ job.status }}
    webhook_url: ${{ secrets.SLACK_WEBHOOK }}
```

---

## Troubleshooting CI/CD

### Tests Fail in CI but Pass Locally

**Common Causes:**
1. Environment differences
   - Python version mismatch
   - Missing system dependencies
   - Timezone differences

2. Secrets not configured
   - Check GitHub repository secrets
   - Verify secret names match workflow

3. File not committed
   - `git status` - uncommitted files
   - `.gitignore` - accidentally ignored

**Solution:**
```bash
# Reproduce CI environment locally
docker run -it --rm python:3.10-slim /bin/bash
# Inside container: run same commands as CI
```

### Docker Build Fails in CI

**Common Causes:**
1. TA-Lib download fails
   - SourceForge sometimes slow
   - Add retry logic

2. Platform mismatch
   - Building for wrong architecture
   - Use Buildx for multi-platform

3. Cache corruption
   - Clear GitHub Actions cache
   - Rebuild with --no-cache

**Solution:**
```bash
# Test build locally
docker build --no-cache -t trading-bot-test .

# Test multi-platform build
docker buildx build --platform linux/amd64,linux/arm64 -t trading-bot-test .
```

### Security Scan Fails

**Trivy finds vulnerabilities:**
1. Check severity (LOW/MEDIUM/HIGH/CRITICAL)
2. For CRITICAL:
   - Update base image (Python version)
   - Update vulnerable packages
3. For MEDIUM/LOW:
   - Review and document
   - Fix if possible, otherwise accept risk

**Bandit finds security issues:**
1. Review code flagged
2. Fix if valid concern
3. Suppress false positives:
```python
# nosec B101
some_code_here()
```

**TruffleHog finds secrets:**
1. **CRITICAL:** Remove secret from git history
2. Rotate compromised secret immediately
3. Add pattern to .gitignore
4. Use environment variables instead

---

## Performance Optimization

### Faster CI Runs

**Current Duration:** ~5-10 minutes
**Optimization Strategies:**

1. **Cache Dependencies:**
```yaml
- name: Cache pip packages
  uses: actions/cache@v3
  with:
    path: ~/.cache/pip
    key: ${{ runner.os }}-pip-${{ hashFiles('requirements.txt') }}
```

2. **Parallel Jobs:**
```yaml
strategy:
  matrix:
    os: [ubuntu-latest, macos-latest]
    python-version: ["3.10", "3.11"]
  max-parallel: 4  # Run 4 jobs at once
```

3. **Skip Redundant Builds:**
```yaml
on:
  push:
    paths-ignore:
      - '**.md'  # Don't run CI on docs-only changes
      - 'docs/**'
```

### Faster Docker Builds

1. **Layer Caching:**
   - Already using GitHub Actions cache
   - `cache-from: type=gha`

2. **Multi-Stage Builds:**
   - Already implemented (builder + runtime)

3. **Minimize Layers:**
   - Combine RUN commands
   - Delete cache in same layer

---

## Best Practices

### 1. Never Commit Secrets

**Bad:**
```python
API_KEY = "PK123ABC..."  # Committed to git
```

**Good:**
```python
API_KEY = os.getenv("ALPACA_API_KEY")  # From environment
```

**Check:**
```bash
# Scan for accidentally committed secrets
docker run --rm -v $(pwd):/src trufflesecurity/trufflehog:latest filesystem /src
```

### 2. Pin Dependencies

**Bad:**
```txt
alpaca-trade-api
pandas
numpy
```

**Good:**
```txt
alpaca-trade-api==3.0.2
pandas==2.0.3
numpy<2.0.0
```

**Why:**
- Reproducible builds
- Avoid breaking changes
- Security (know what versions are tested)

### 3. Test Before Merge

**Process:**
1. Create feature branch
2. Develop and test locally
3. Push to GitHub → CI runs
4. Fix any CI failures
5. Request code review
6. Merge only after:
   - ✅ All CI checks pass
   - ✅ Code review approved
   - ✅ No merge conflicts

### 4. Semantic Versioning

**Tag Releases:**
```bash
# After successful validation
git tag -a v1.0.0 -m "First validated release"
git push origin v1.0.0
```

**Docker Image Tags:**
- `ghcr.io/user/trading-bot:v1.0.0` - Specific version
- `ghcr.io/user/trading-bot:v1.0` - Minor version
- `ghcr.io/user/trading-bot:v1` - Major version
- `ghcr.io/user/trading-bot:latest` - Latest release

### 5. Rollback Strategy

**If Deployment Fails:**
```bash
# Option 1: Rollback to previous Docker image
docker pull ghcr.io/user/trading-bot:v0.9.0
docker-compose up -d

# Option 2: Revert git commit
git revert HEAD
git push origin main
# CI rebuilds previous version

# Option 3: Redeploy last known good tag
docker pull ghcr.io/user/trading-bot:v0.9.0
```

---

## Future Enhancements

**Planned CI/CD Improvements:**

1. **Automated Performance Testing:**
   - Backtest on every PR
   - Compare Sharpe ratio to baseline
   - Fail if performance degrades >10%

2. **Canary Deployments:**
   - Deploy to 1 bot instance first
   - Monitor for 24 hours
   - Roll out to rest if successful

3. **Integration Tests:**
   - Test full trading loop
   - Mock Alpaca API responses
   - Verify order execution logic

4. **Nightly Builds:**
   - Run backtests overnight
   - Generate performance reports
   - Email if anomalies detected

5. **Multi-Environment:**
   - Dev → Staging → Production pipeline
   - Separate credentials per environment
   - Progressive rollout

---

## Metrics & Monitoring

**What to Track:**

### Build Metrics
- Build duration (target: <5 minutes)
- Success rate (target: >95%)
- Flaky tests (tests that fail intermittently)

### Docker Metrics
- Image size (target: <500MB)
- Build time (target: <3 minutes)
- Vulnerability count (target: 0 CRITICAL)

### Deployment Metrics
- Deployment frequency
- Time to production (commit → deployed)
- Rollback frequency
- MTTR (Mean Time To Recovery)

**Tools:**
- GitHub Actions metrics (built-in)
- Docker Hub/GHCR analytics
- Custom dashboards (Grafana)

---

## Quick Reference

### Trigger CI Manually

```bash
# From GitHub UI
Actions tab → Select workflow → Run workflow

# Using GitHub CLI
gh workflow run ci.yml
```

### View Workflow Logs

```bash
# Using GitHub CLI
gh run list --workflow=ci.yml
gh run view <run-id> --log
```

### Cancel Running Workflow

```bash
# From GitHub UI
Actions tab → Running workflow → Cancel

# Using GitHub CLI
gh run cancel <run-id>
```

### Download Artifacts

```bash
# Using GitHub CLI
gh run download <run-id>
```

---

## Support

**If CI/CD issues arise:**

1. Check workflow file syntax:
   ```bash
   # Validate YAML
   yamllint .github/workflows/*.yml
   ```

2. Check secrets are configured:
   - GitHub repo → Settings → Secrets

3. Review workflow logs:
   - Actions tab → Failed run → View logs

4. Test locally:
   - Run same commands CI runs
   - Use `act` to run GitHub Actions locally

5. Search GitHub Actions docs:
   - https://docs.github.com/en/actions

---

**Updated:** 2025-11-10
**Status:** CI/CD fully configured and documented
**Next:** Test Docker build, monitor first workflow runs
