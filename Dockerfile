# Multi-stage Docker build for trading bot
# Optimized for production deployment with minimal image size

# Stage 1: Builder
FROM python:3.10-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install TA-Lib from source
# TARGETARCH is set automatically by Docker BuildKit (amd64, arm64, etc.)
# TA-Lib's 2006-era config.guess fails under QEMU emulation for ARM64,
# so we pass --build explicitly when cross-compiling.
ARG TARGETARCH
RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    tar -xzf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib/ && \
    if [ "$TARGETARCH" = "arm64" ]; then \
      ./configure --prefix=/usr --build=aarch64-unknown-linux-gnu; \
    else \
      ./configure --prefix=/usr; \
    fi && \
    make && \
    make install && \
    cd .. && \
    rm -rf ta-lib ta-lib-0.4.0-src.tar.gz

# Set working directory
WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy dependency files
COPY pyproject.toml uv.lock* ./

# Install Python dependencies with uv
RUN uv sync --frozen --no-dev --no-install-project

# Stage 2: Runtime
FROM python:3.10-slim

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy TA-Lib from builder
COPY --from=builder /usr/lib/libta_lib.* /usr/lib/
COPY --from=builder /usr/include/ta-lib/ /usr/include/ta-lib/

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Create non-root user for security
RUN useradd -m -u 1000 trader && \
    mkdir -p /app/logs /app/data && \
    chown -R trader:trader /app

# Set working directory
WORKDIR /app

# Copy application code and dependency files
COPY --chown=trader:trader . .

# Switch to non-root user
USER trader

# Health check
HEALTHCHECK --interval=60s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')" || exit 1

# Default environment variables
ENV PYTHONUNBUFFERED=1 \
    PAPER=True \
    LOG_LEVEL=INFO \
    VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH"

# Expose metrics port (if using Prometheus)
EXPOSE 8000

# Default command: run paper trading
CMD ["python", "start.py"]
