# Docker Deployment Guide

**Trading Bot - Containerized Deployment**

This guide covers running the trading bot in Docker containers for consistent, reproducible deployments.

---

## Why Docker?

**Benefits:**
- ✅ **Consistent Environment** - Same Python version, dependencies everywhere
- ✅ **Easy Deployment** - Single command to start/stop bot
- ✅ **Isolation** - Bot runs independently, no conflicts with system packages
- ✅ **Portability** - Run on any machine (Mac, Linux, cloud servers)
- ✅ **Reproducibility** - Exact same environment for testing and production
- ✅ **Resource Limits** - Control CPU/memory usage
- ✅ **Health Monitoring** - Automatic health checks and restarts

**When to Use Docker:**
- Deploying to cloud servers (AWS, GCP, DigitalOcean)
- Running multiple bots with different configurations
- CI/CD deployments
- Production environments
- Team collaboration (consistent environment for all developers)

**When NOT to Use Docker:**
- Local development (direct Python is faster)
- Initial testing and debugging (easier without container overhead)

---

## Quick Start

### Prerequisites

```bash
# Install Docker
# macOS: https://docs.docker.com/desktop/mac/install/
# Linux: https://docs.docker.com/engine/install/
# Windows: https://docs.docker.com/desktop/windows/install/

# Verify installation
docker --version
docker-compose --version
```

### Basic Usage

**1. Build the Docker image:**
```bash
docker-compose build trading-bot-paper
```

**2. Start paper trading bot:**
```bash
docker-compose up -d trading-bot-paper
```

**3. View logs:**
```bash
docker-compose logs -f trading-bot-paper
```

**4. Stop the bot:**
```bash
docker-compose down
```

---

## Docker Files Overview

### Dockerfile

**Multi-stage build for optimized image size:**

**Stage 1 (Builder):**
- Installs TA-Lib from source (requires compilation)
- Installs all Python dependencies
- ~800MB image size

**Stage 2 (Runtime):**
- Copies only necessary files from builder
- Minimal dependencies (only runtime libs)
- ~400MB final image size
- Non-root user for security

**Key Features:**
- Python 3.10 slim base image
- TA-Lib pre-compiled
- Non-root user (trader:1000)
- Health checks every 60 seconds
- Optimized layer caching

### docker-compose.yml

**Services Defined:**

1. **trading-bot-paper** (active by default)
   - Paper trading mode
   - Auto-restart enabled
   - Persists logs and data
   - Health monitoring

2. **trading-bot-live** (commented out)
   - Live trading mode
   - Requires separate API keys
   - Only enable after 60+ days paper trading validation

3. **backtest** (on-demand)
   - Run backtests in container
   - Profile: `tools`
   - Usage: `docker-compose --profile tools up backtest`

4. **dashboard** (future enhancement, commented out)
   - Real-time monitoring dashboard
   - Port 8050

5. **prometheus + grafana** (future enhancement, commented out)
   - Metrics collection and visualization
   - Profile: `monitoring`

---

## Configuration

### Environment Variables

Create `.env` file in project root:

```bash
# Alpaca API Credentials (Paper Trading)
ALPACA_API_KEY=your_paper_api_key
ALPACA_SECRET_KEY=your_paper_secret_key
PAPER=True

# Optional: Logging
LOG_LEVEL=INFO

# Optional: Grafana (if using monitoring)
# GRAFANA_PASSWORD=secure_password
```

**Security Note:** Never commit `.env` to git. It's already in `.gitignore`.

### Volume Mounts

**Persistent Data:**
- `./logs:/app/logs` - Trading logs (survives container restarts)
- `./data:/app/data` - SQLite database, cached data
- `./config.py:/app/config.py:ro` - Configuration (read-only)

**Why Read-Only Config:**
- Prevents container from modifying config
- Forces config changes to be intentional (rebuild container)
- Better security

---

## Usage Examples

### Paper Trading (Default)

```bash
# Start paper trading bot
docker-compose up -d trading-bot-paper

# View logs in real-time
docker-compose logs -f trading-bot-paper

# Check bot status
docker-compose ps

# Stop bot
docker-compose stop trading-bot-paper

# Restart bot (after config change)
docker-compose restart trading-bot-paper

# Remove container (keeps volumes)
docker-compose down
```

### Run Backtest in Docker

```bash
# Run backtest (on-demand)
docker-compose --profile tools run --rm backtest

# Or build and run directly
docker build -t trading-bot .
docker run --rm \
  -e ALPACA_API_KEY="${ALPACA_API_KEY}" \
  -e ALPACA_SECRET_KEY="${ALPACA_SECRET_KEY}" \
  -e PAPER=True \
  -v $(pwd)/logs:/app/logs \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/results:/app/results \
  trading-bot \
  python simple_backtest.py
```

### Custom Strategy

```bash
# Override default command
docker-compose run --rm trading-bot-paper \
  python main.py live --strategy MeanReversionStrategy --force
```

### Interactive Shell (Debugging)

```bash
# Access container shell
docker-compose exec trading-bot-paper /bin/bash

# Or run one-off commands
docker-compose exec trading-bot-paper python -c "from brokers.alpaca_broker import AlpacaBroker; print('Connected!')"
```

---

## Monitoring

### Health Checks

Docker automatically monitors bot health:

```bash
# Check container health
docker-compose ps

# Healthy output:
# NAME                  STATUS
# trading-bot-paper     Up (healthy)

# Unhealthy output:
# trading-bot-paper     Up (unhealthy)
```

**Health check criteria:**
- Checks if `bot.pid` file exists
- Runs every 60 seconds
- 3 retries before marking unhealthy
- Auto-restart on unhealthy (if configured)

### View Logs

```bash
# All logs
docker-compose logs trading-bot-paper

# Last 100 lines
docker-compose logs --tail=100 trading-bot-paper

# Follow logs (real-time)
docker-compose logs -f trading-bot-paper

# Search logs for errors
docker-compose logs trading-bot-paper | grep ERROR

# Search for trades
docker-compose logs trading-bot-paper | grep -E "(ENTRY|EXIT)"
```

### Resource Usage

```bash
# View resource usage
docker stats trading-bot-paper

# Output:
# CONTAINER          CPU %     MEM USAGE / LIMIT     MEM %
# trading-bot-paper  5.2%      120MB / 2GB          6.0%
```

### Exec into Container

```bash
# Access running container
docker-compose exec trading-bot-paper /bin/bash

# Inside container:
trader@container:/app$ ps aux
trader@container:/app$ tail -f logs/paper_trading.log
trader@container:/app$ python -c "import sys; print(sys.version)"
```

---

## Production Deployment

### Cloud Deployment (AWS EC2 Example)

```bash
# 1. SSH into server
ssh user@your-server.com

# 2. Clone repository
git clone https://github.com/yourusername/trading-bot.git
cd trading-bot

# 3. Create .env file
nano .env
# Add your API keys, save and exit

# 4. Start bot
docker-compose up -d trading-bot-paper

# 5. Monitor
docker-compose logs -f trading-bot-paper
```

### Resource Limits

Add to `docker-compose.yml` under service:

```yaml
trading-bot-paper:
  # ... existing config ...
  deploy:
    resources:
      limits:
        cpus: '1.0'      # Max 1 CPU core
        memory: 2G       # Max 2GB RAM
      reservations:
        cpus: '0.5'      # Reserve 0.5 CPU
        memory: 512M     # Reserve 512MB RAM
```

### Auto-Restart Policies

```yaml
trading-bot-paper:
  restart: unless-stopped  # Default (recommended)
  # restart: always        # Always restart (even after manual stop)
  # restart: on-failure    # Only restart on crash
  # restart: "no"          # Never restart
```

---

## CI/CD Integration

### GitHub Actions

The `.github/workflows/docker-build.yml` workflow:

**On Push to Main:**
1. Builds Docker image
2. Runs security scans (Trivy)
3. Pushes to GitHub Container Registry
4. Tags with version and commit SHA

**On Pull Request:**
1. Builds image
2. Tests that it runs
3. Validates docker-compose config

### Using GitHub Container Registry

```bash
# Login to GitHub Container Registry
echo $GITHUB_TOKEN | docker login ghcr.io -u USERNAME --password-stdin

# Pull latest image
docker pull ghcr.io/yourusername/trading-bot:latest

# Run pulled image
docker run -d \
  --name trading-bot-paper \
  --env-file .env \
  -v $(pwd)/logs:/app/logs \
  -v $(pwd)/data:/app/data \
  ghcr.io/yourusername/trading-bot:latest
```

---

## Troubleshooting

### Container Won't Start

```bash
# Check logs
docker-compose logs trading-bot-paper

# Check if port is in use
docker-compose ps

# Rebuild image (force)
docker-compose build --no-cache trading-bot-paper
docker-compose up -d trading-bot-paper
```

### Bot Crashes Immediately

```bash
# View exit code
docker-compose ps

# Check last 100 log lines
docker-compose logs --tail=100 trading-bot-paper

# Common issues:
# 1. Missing .env file
# 2. Invalid API keys
# 3. Missing volumes
```

### Health Check Failing

```bash
# Check health status
docker-compose ps

# View health check logs
docker inspect trading-bot-paper | jq '.[0].State.Health'

# Common causes:
# 1. Bot crashed but container is running
# 2. bot.pid file not created
# 3. Health check command incorrect
```

### Logs Not Persisting

```bash
# Check volume mounts
docker-compose config | grep volumes

# Verify permissions
ls -la logs/

# Create logs directory if missing
mkdir -p logs data results
```

### Build Fails on TA-Lib

```bash
# Try building with --no-cache
docker-compose build --no-cache trading-bot-paper

# If still fails, check TA-Lib download URL in Dockerfile
# Sometimes SourceForge is slow/down
```

---

## Advanced Usage

### Multiple Bots (Different Strategies)

```yaml
# docker-compose.override.yml
version: '3.8'

services:
  momentum-bot:
    extends:
      service: trading-bot-paper
    container_name: momentum-bot
    command: ["python", "main.py", "live", "--strategy", "MomentumStrategy", "--force"]
    volumes:
      - ./logs/momentum:/app/logs

  meanrev-bot:
    extends:
      service: trading-bot-paper
    container_name: meanrev-bot
    command: ["python", "main.py", "live", "--strategy", "MeanReversionStrategy", "--force"]
    volumes:
      - ./logs/meanrev:/app/logs
```

```bash
# Start both
docker-compose -f docker-compose.yml -f docker-compose.override.yml up -d
```

### Development vs Production

```bash
# Development (mount code for live reload)
docker run -d \
  --name trading-bot-dev \
  --env-file .env \
  -v $(pwd):/app \
  trading-bot \
  python main.py live --strategy MomentumStrategy --force

# Production (code baked into image)
docker-compose up -d trading-bot-paper
```

### Scheduled Backtests (Cron)

```bash
# Add to crontab
0 0 * * * cd /path/to/trading-bot && docker-compose --profile tools run --rm backtest >> /var/log/backtest.log 2>&1
```

---

## Best Practices

### Security

**1. Never commit credentials:**
```bash
# .env should be in .gitignore
git check-ignore .env  # Should return .env
```

**2. Use non-root user:**
- Dockerfile already creates `trader` user
- Container runs as UID 1000

**3. Read-only config:**
```yaml
volumes:
  - ./config.py:/app/config.py:ro  # :ro = read-only
```

**4. Scan for vulnerabilities:**
```bash
# Trivy scan (included in CI/CD)
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
  aquasec/trivy image trading-bot:latest
```

### Performance

**1. Multi-stage builds:**
- Builder stage for compilation
- Runtime stage for execution
- Reduces final image size by 50%

**2. Layer caching:**
- Copy requirements.txt before code
- Pip install cached if requirements unchanged
- Faster rebuilds during development

**3. Resource limits:**
- Prevent bot from consuming all server resources
- Especially important in cloud environments

### Monitoring

**1. Structured logging:**
```python
# Future enhancement: JSON logs
import json
import logging

logger.info(json.dumps({
    "event": "trade_executed",
    "symbol": "AAPL",
    "qty": 10,
    "price": 150.00
}))
```

**2. Health checks:**
- Already configured in docker-compose
- Can extend to check API connectivity

**3. Metrics export:**
- Future: Prometheus metrics endpoint
- Track: trades/hour, P/L, API latency

---

## Migration from Local to Docker

### Current Setup (Local Python)

```bash
# Stop local bot
kill $(cat bot.pid)

# Backup logs and data
cp -r logs logs.backup
cp -r data data.backup
```

### Switch to Docker

```bash
# Build and start
docker-compose up -d trading-bot-paper

# Verify logs are persisting
docker-compose logs -f trading-bot-paper

# Check data directory
ls -la data/
```

### Rollback if Needed

```bash
# Stop Docker bot
docker-compose down

# Restore backups
cp -r logs.backup/* logs/
cp -r data.backup/* data/

# Restart local bot
nohup python3 main.py live --strategy MomentumStrategy --force > paper_trading.log 2>&1 &
echo $! > bot.pid
```

---

## Future Enhancements

**Planned Docker Features:**

1. **Monitoring Stack:**
   - Prometheus for metrics
   - Grafana for dashboards
   - Already scaffolded in docker-compose.yml

2. **Multi-Bot Orchestration:**
   - Kubernetes deployment
   - Auto-scaling based on market hours
   - Rolling updates

3. **Backup Automation:**
   - Automated data backups to S3
   - Log rotation and archival

4. **Security Hardening:**
   - Secret management (HashiCorp Vault)
   - mTLS between containers
   - Network policies

---

## Quick Reference

### Common Commands

```bash
# Build
docker-compose build

# Start (background)
docker-compose up -d

# Start (foreground, see logs)
docker-compose up

# Stop
docker-compose stop

# Stop and remove
docker-compose down

# View logs
docker-compose logs -f

# Restart
docker-compose restart

# Check status
docker-compose ps

# Access shell
docker-compose exec trading-bot-paper /bin/bash

# Run backtest
docker-compose --profile tools run --rm backtest

# View resource usage
docker stats
```

### File Locations (Inside Container)

```
/app/                    # Working directory
/app/main.py             # Entry point
/app/strategies/         # Strategy code
/app/brokers/            # Broker integration
/app/logs/               # Log files (mounted)
/app/data/               # SQLite DB (mounted)
/app/config.py           # Configuration (mounted, read-only)
/app/bot.pid             # Process ID (for health checks)
```

---

## Support

**If Docker issues arise:**
1. Check logs: `docker-compose logs -f trading-bot-paper`
2. Check container status: `docker-compose ps`
3. Verify .env file exists and has correct API keys
4. Try rebuilding: `docker-compose build --no-cache`
5. Check Docker daemon: `docker info`

**Common solutions:**
- Missing .env → Create from .env.example
- Port conflicts → Change ports in docker-compose.yml
- Permission errors → Check volume mount ownership
- Build failures → Clear Docker cache, rebuild

---

**Updated:** 2025-11-10
**Status:** Ready for use (Docker implementation complete)
**Next:** Test locally, then deploy to production
