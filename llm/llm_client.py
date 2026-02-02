"""
LLM Client Abstraction Layer

Provides unified interface for OpenAI GPT-4 and Anthropic Claude with:
- Automatic fallback on failure
- Token counting and cost tracking
- Rate limiting and retry logic
- Response caching
"""

import asyncio
import hashlib
import json
import logging
import os
import sqlite3
import time
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type

from llm.llm_types import (
    LLMClientConfig,
    LLMProvider,
    LLMResponse,
    calculate_cost,
)

logger = logging.getLogger(__name__)


class RateLimiter:
    """
    Token bucket rate limiter for LLM API calls.

    Provides smooth rate limiting that allows bursts while
    maintaining average rate compliance.
    """

    def __init__(
        self,
        tokens_per_minute: int = 50000,
        requests_per_minute: int = 10,
        burst_multiplier: float = 1.5,
    ):
        """
        Initialize rate limiter.

        Args:
            tokens_per_minute: Maximum tokens per minute
            requests_per_minute: Maximum requests per minute
            burst_multiplier: Allow bursts up to this multiple
        """
        # Token bucket for tokens
        self._token_rate = tokens_per_minute / 60.0  # per second
        self._token_capacity = int(tokens_per_minute * burst_multiplier)
        self._tokens = self._token_capacity
        self._token_last_update = time.time()

        # Token bucket for requests
        self._request_rate = requests_per_minute / 60.0
        self._request_capacity = int(requests_per_minute * burst_multiplier)
        self._requests = self._request_capacity
        self._request_last_update = time.time()

        self._lock = asyncio.Lock()

    async def acquire(self, tokens: int = 1) -> float:
        """
        Acquire tokens for a request.

        Args:
            tokens: Number of tokens to acquire

        Returns:
            Wait time in seconds (0 if acquired immediately)
        """
        async with self._lock:
            now = time.time()
            wait_time = 0.0

            # Refill token buckets
            token_elapsed = now - self._token_last_update
            self._tokens = min(
                self._token_capacity,
                self._tokens + token_elapsed * self._token_rate
            )
            self._token_last_update = now

            request_elapsed = now - self._request_last_update
            self._requests = min(
                self._request_capacity,
                self._requests + request_elapsed * self._request_rate
            )
            self._request_last_update = now

            # Check if we need to wait for tokens
            if self._tokens < tokens:
                wait_time = max(wait_time, (tokens - self._tokens) / self._token_rate)

            # Check if we need to wait for request capacity
            if self._requests < 1:
                wait_time = max(wait_time, (1 - self._requests) / self._request_rate)

            if wait_time > 0:
                await asyncio.sleep(wait_time)
                # Refill after waiting
                self._tokens = min(
                    self._token_capacity,
                    self._tokens + wait_time * self._token_rate
                )
                self._requests = min(
                    self._request_capacity,
                    self._requests + wait_time * self._request_rate
                )

            # Consume tokens
            self._tokens -= tokens
            self._requests -= 1

            return wait_time

    def get_status(self) -> Dict[str, Any]:
        """Get current rate limiter status."""
        return {
            "tokens_available": int(self._tokens),
            "token_capacity": self._token_capacity,
            "requests_available": int(self._requests),
            "request_capacity": self._request_capacity,
        }


class CostTracker:
    """
    Tracks LLM API costs with daily/weekly/monthly caps.

    Persists to SQLite for long-term tracking and analysis.
    """

    def __init__(
        self,
        db_path: str = ".cache/llm_costs.db",
        daily_cap: float = 50.0,
        weekly_cap: float = 250.0,
        monthly_cap: float = 750.0,
    ):
        """
        Initialize cost tracker.

        Args:
            db_path: Path to SQLite database
            daily_cap: Maximum daily spend in USD
            weekly_cap: Maximum weekly spend in USD
            monthly_cap: Maximum monthly spend in USD
        """
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        self._daily_cap = daily_cap
        self._weekly_cap = weekly_cap
        self._monthly_cap = monthly_cap

        self._init_db()

        # In-memory cache for current period
        self._today_cost = 0.0
        self._week_cost = 0.0
        self._month_cost = 0.0
        self._load_current_costs()

    def _init_db(self):
        """Initialize SQLite database."""
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS llm_costs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    model TEXT NOT NULL,
                    input_tokens INTEGER NOT NULL,
                    output_tokens INTEGER NOT NULL,
                    cost_usd REAL NOT NULL,
                    source_type TEXT,
                    symbol TEXT
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_costs_timestamp
                ON llm_costs(timestamp)
            """)
            conn.commit()

    def _load_current_costs(self):
        """Load costs for current periods from database."""
        now = datetime.now()
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        week_start = today_start - timedelta(days=today_start.weekday())
        month_start = today_start.replace(day=1)

        with sqlite3.connect(self._db_path) as conn:
            # Today's cost
            result = conn.execute(
                "SELECT COALESCE(SUM(cost_usd), 0) FROM llm_costs WHERE timestamp >= ?",
                (today_start.isoformat(),)
            ).fetchone()
            self._today_cost = result[0] if result else 0.0

            # Week's cost
            result = conn.execute(
                "SELECT COALESCE(SUM(cost_usd), 0) FROM llm_costs WHERE timestamp >= ?",
                (week_start.isoformat(),)
            ).fetchone()
            self._week_cost = result[0] if result else 0.0

            # Month's cost
            result = conn.execute(
                "SELECT COALESCE(SUM(cost_usd), 0) FROM llm_costs WHERE timestamp >= ?",
                (month_start.isoformat(),)
            ).fetchone()
            self._month_cost = result[0] if result else 0.0

    def record(
        self,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float,
        source_type: Optional[str] = None,
        symbol: Optional[str] = None,
    ) -> None:
        """Record a cost entry."""
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                """
                INSERT INTO llm_costs
                (timestamp, provider, model, input_tokens, output_tokens, cost_usd, source_type, symbol)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    datetime.now().isoformat(),
                    provider,
                    model,
                    input_tokens,
                    output_tokens,
                    cost_usd,
                    source_type,
                    symbol,
                )
            )
            conn.commit()

        # Update in-memory cache
        self._today_cost += cost_usd
        self._week_cost += cost_usd
        self._month_cost += cost_usd

    def check_cap(self, period: str = "daily") -> bool:
        """
        Check if we're under the cap for a period.

        Args:
            period: "daily", "weekly", or "monthly"

        Returns:
            True if under cap, False if at or over cap
        """
        if period == "daily":
            return self._today_cost < self._daily_cap
        elif period == "weekly":
            return self._week_cost < self._weekly_cap
        elif period == "monthly":
            return self._month_cost < self._monthly_cap
        else:
            return True

    def get_remaining(self, period: str = "daily") -> float:
        """Get remaining budget for a period."""
        if period == "daily":
            return max(0.0, self._daily_cap - self._today_cost)
        elif period == "weekly":
            return max(0.0, self._weekly_cap - self._week_cost)
        elif period == "monthly":
            return max(0.0, self._monthly_cap - self._month_cost)
        else:
            return float("inf")

    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics."""
        return {
            "daily": {
                "used": round(self._today_cost, 4),
                "cap": self._daily_cap,
                "remaining": round(self.get_remaining("daily"), 4),
                "pct_used": round(self._today_cost / self._daily_cap * 100, 1),
            },
            "weekly": {
                "used": round(self._week_cost, 4),
                "cap": self._weekly_cap,
                "remaining": round(self.get_remaining("weekly"), 4),
                "pct_used": round(self._week_cost / self._weekly_cap * 100, 1),
            },
            "monthly": {
                "used": round(self._month_cost, 4),
                "cap": self._monthly_cap,
                "remaining": round(self.get_remaining("monthly"), 4),
                "pct_used": round(self._month_cost / self._monthly_cap * 100, 1),
            },
        }


class ResponseCache:
    """
    LRU cache for LLM responses with SQLite persistence.
    """

    def __init__(
        self,
        db_path: str = ".cache/llm_responses.db",
        memory_size: int = 100,
        default_ttl_seconds: int = 3600,
    ):
        """
        Initialize cache.

        Args:
            db_path: Path to SQLite database
            memory_size: Number of items to cache in memory
            default_ttl_seconds: Default TTL for cache entries
        """
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        self._memory_cache: Dict[str, tuple] = {}  # key -> (value, expiry)
        self._memory_size = memory_size
        self._default_ttl = default_ttl_seconds

        self._init_db()

    def _init_db(self):
        """Initialize SQLite database."""
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS response_cache (
                    cache_key TEXT PRIMARY KEY,
                    response TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    provider TEXT,
                    model TEXT
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_cache_expires
                ON response_cache(expires_at)
            """)
            conn.commit()

    def _generate_key(self, prompt: str, system: str, model: str) -> str:
        """Generate cache key from prompt, system message, and model."""
        data = json.dumps({"prompt": prompt, "system": system, "model": model}, sort_keys=True)
        return hashlib.sha256(data.encode()).hexdigest()

    def get(self, prompt: str, system: str, model: str) -> Optional[str]:
        """Get cached response if available and not expired."""
        key = self._generate_key(prompt, system, model)
        now = datetime.now()

        # Check memory cache first
        if key in self._memory_cache:
            value, expiry = self._memory_cache[key]
            if expiry > now:
                return value
            else:
                del self._memory_cache[key]

        # Check SQLite cache
        with sqlite3.connect(self._db_path) as conn:
            result = conn.execute(
                "SELECT response FROM response_cache WHERE cache_key = ? AND expires_at > ?",
                (key, now.isoformat())
            ).fetchone()

            if result:
                # Add to memory cache
                self._memory_cache[key] = (result[0], now + timedelta(seconds=self._default_ttl))
                self._trim_memory_cache()
                return result[0]

        return None

    def set(
        self,
        prompt: str,
        system: str,
        model: str,
        response: str,
        ttl_seconds: Optional[int] = None,
    ) -> str:
        """Store response in cache."""
        key = self._generate_key(prompt, system, model)
        now = datetime.now()
        ttl = ttl_seconds or self._default_ttl
        expiry = now + timedelta(seconds=ttl)

        # Store in memory
        self._memory_cache[key] = (response, expiry)
        self._trim_memory_cache()

        # Store in SQLite
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO response_cache
                (cache_key, response, created_at, expires_at, provider, model)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (key, response, now.isoformat(), expiry.isoformat(), "", model)
            )
            conn.commit()

        return key

    def _trim_memory_cache(self):
        """Trim memory cache to size limit (LRU)."""
        if len(self._memory_cache) > self._memory_size:
            # Remove oldest entries
            sorted_items = sorted(
                self._memory_cache.items(),
                key=lambda x: x[1][1]  # Sort by expiry
            )
            for key, _ in sorted_items[: len(self._memory_cache) - self._memory_size]:
                del self._memory_cache[key]

    def cleanup_expired(self) -> int:
        """Remove expired entries from SQLite."""
        with sqlite3.connect(self._db_path) as conn:
            cursor = conn.execute(
                "DELETE FROM response_cache WHERE expires_at < ?",
                (datetime.now().isoformat(),)
            )
            conn.commit()
            return cursor.rowcount


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""

    def __init__(self, config: LLMClientConfig):
        """Initialize client with config."""
        self.config = config
        self._initialized = False

    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the client (load API keys, etc.)."""
        pass

    @abstractmethod
    async def analyze(
        self,
        prompt: str,
        system: Optional[str] = None,
        json_schema: Optional[Dict] = None,
        max_tokens: int = 1000,
    ) -> LLMResponse:
        """
        Send prompt to LLM and get response.

        Args:
            prompt: User prompt
            system: System message
            json_schema: Expected JSON schema for validation
            max_tokens: Maximum tokens in response

        Returns:
            LLMResponse with content and metadata
        """
        pass

    @abstractmethod
    def get_token_count(self, text: str) -> int:
        """Estimate token count for text."""
        pass

    @property
    @abstractmethod
    def provider(self) -> LLMProvider:
        """Get provider type."""
        pass

    @property
    @abstractmethod
    def model(self) -> str:
        """Get model name."""
        pass


class OpenAIClient(BaseLLMClient):
    """OpenAI GPT-4 client."""

    def __init__(self, config: LLMClientConfig):
        super().__init__(config)
        self._client = None
        self._model = config.openai_model

    @property
    def provider(self) -> LLMProvider:
        return LLMProvider.OPENAI

    @property
    def model(self) -> str:
        return self._model

    async def initialize(self) -> bool:
        """Initialize OpenAI client."""
        try:
            import openai

            api_key = self.config.openai_api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                logger.warning("No OpenAI API key found")
                return False

            self._client = openai.AsyncOpenAI(api_key=api_key)
            self._initialized = True
            logger.info(f"OpenAI client initialized with model {self._model}")
            return True

        except ImportError:
            logger.warning("openai package not installed")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            return False

    async def analyze(
        self,
        prompt: str,
        system: Optional[str] = None,
        json_schema: Optional[Dict] = None,
        max_tokens: int = 1000,
    ) -> LLMResponse:
        """Send prompt to GPT-4."""
        if not self._initialized or not self._client:
            raise RuntimeError("OpenAI client not initialized")

        start_time = time.time()

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        # Use JSON mode if schema provided
        response_format = None
        if json_schema:
            response_format = {"type": "json_object"}

        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=max_tokens,
                response_format=response_format,
                timeout=self.config.timeout_seconds,
            )

            content = response.choices[0].message.content
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens

            latency_ms = (time.time() - start_time) * 1000
            cost = calculate_cost(self._model, input_tokens, output_tokens)

            return LLMResponse(
                content=content,
                provider=LLMProvider.OPENAI,
                model=self._model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
                cost_usd=cost,
                latency_ms=latency_ms,
                request_id=response.id,
            )

        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise

    def get_token_count(self, text: str) -> int:
        """Estimate token count (rough approximation)."""
        # GPT-4 uses cl100k_base tokenizer
        # Rough estimate: ~4 chars per token for English
        return len(text) // 4


class AnthropicClient(BaseLLMClient):
    """Anthropic Claude client."""

    def __init__(self, config: LLMClientConfig):
        super().__init__(config)
        self._client = None
        self._model = config.anthropic_model

    @property
    def provider(self) -> LLMProvider:
        return LLMProvider.ANTHROPIC

    @property
    def model(self) -> str:
        return self._model

    async def initialize(self) -> bool:
        """Initialize Anthropic client."""
        try:
            import anthropic

            api_key = self.config.anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                logger.warning("No Anthropic API key found")
                return False

            self._client = anthropic.AsyncAnthropic(api_key=api_key)
            self._initialized = True
            logger.info(f"Anthropic client initialized with model {self._model}")
            return True

        except ImportError:
            logger.warning("anthropic package not installed")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize Anthropic client: {e}")
            return False

    async def analyze(
        self,
        prompt: str,
        system: Optional[str] = None,
        json_schema: Optional[Dict] = None,
        max_tokens: int = 1000,
    ) -> LLMResponse:
        """Send prompt to Claude."""
        if not self._initialized or not self._client:
            raise RuntimeError("Anthropic client not initialized")

        start_time = time.time()

        # If JSON schema provided, add instruction to prompt
        if json_schema:
            prompt = f"{prompt}\n\nRespond ONLY with valid JSON matching this schema. No additional text."

        try:
            kwargs = {
                "model": self._model,
                "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": prompt}],
            }

            if system:
                kwargs["system"] = system

            response = await self._client.messages.create(**kwargs)

            content = response.content[0].text
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens

            latency_ms = (time.time() - start_time) * 1000
            cost = calculate_cost(self._model, input_tokens, output_tokens)

            return LLMResponse(
                content=content,
                provider=LLMProvider.ANTHROPIC,
                model=self._model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
                cost_usd=cost,
                latency_ms=latency_ms,
                request_id=response.id,
            )

        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            raise

    def get_token_count(self, text: str) -> int:
        """Estimate token count (rough approximation)."""
        # Claude uses similar tokenization to GPT
        return len(text) // 4


class LLMClientWithFallback:
    """
    LLM client with automatic fallback between providers.

    Features:
    - Primary/secondary provider pattern
    - Automatic fallback on 5XX errors, timeouts, rate limits
    - Cost tracking with daily caps
    - Token counting for budgeting
    - Response caching
    """

    def __init__(
        self,
        config: Optional[LLMClientConfig] = None,
        primary_provider: Optional[LLMProvider] = None,
    ):
        """
        Initialize client with fallback.

        Args:
            config: Client configuration
            primary_provider: Override primary provider from config
        """
        self.config = config or LLMClientConfig()

        if primary_provider:
            self.config.primary_provider = primary_provider

        # Initialize clients
        self._openai = OpenAIClient(self.config)
        self._anthropic = AnthropicClient(self.config)

        # Set primary/fallback
        if self.config.primary_provider == LLMProvider.ANTHROPIC:
            self._primary = self._anthropic
            self._fallback = self._openai
        else:
            self._primary = self._openai
            self._fallback = self._anthropic

        # Rate limiter and cost tracker
        self._rate_limiter = RateLimiter(
            tokens_per_minute=self.config.max_tokens_per_minute,
            requests_per_minute=self.config.max_requests_per_minute,
        )
        self._cost_tracker = CostTracker(
            daily_cap=self.config.daily_cost_cap_usd,
            weekly_cap=self.config.weekly_cost_cap_usd,
            monthly_cap=self.config.monthly_cost_cap_usd,
        )

        # Response cache
        self._cache = ResponseCache(
            default_ttl_seconds=self.config.cache_ttl_seconds,
        )

        self._initialized = False

    async def initialize(self) -> bool:
        """Initialize both clients."""
        primary_ok = await self._primary.initialize()
        fallback_ok = await self._fallback.initialize()

        if not primary_ok and not fallback_ok:
            logger.error("Failed to initialize any LLM client")
            return False

        if not primary_ok:
            logger.warning(
                f"Primary provider {self._primary.provider.value} failed, "
                f"using {self._fallback.provider.value} only"
            )
            self._primary, self._fallback = self._fallback, self._primary

        self._initialized = True
        logger.info(
            f"LLM client initialized (primary: {self._primary.provider.value}, "
            f"fallback: {self._fallback.provider.value if fallback_ok else 'none'})"
        )
        return True

    async def analyze(
        self,
        prompt: str,
        system: Optional[str] = None,
        json_schema: Optional[Dict] = None,
        max_tokens: int = 1000,
        cache_ttl_seconds: Optional[int] = None,
        source_type: Optional[str] = None,
        symbol: Optional[str] = None,
    ) -> LLMResponse:
        """
        Analyze prompt with automatic fallback.

        Args:
            prompt: User prompt
            system: System message
            json_schema: Expected JSON schema
            max_tokens: Maximum response tokens
            cache_ttl_seconds: Cache TTL (None = use default)
            source_type: Analysis type for cost tracking
            symbol: Stock symbol for cost tracking

        Returns:
            LLMResponse with content and metadata
        """
        if not self._initialized:
            raise RuntimeError("Client not initialized")

        # Check cost cap
        if not self._cost_tracker.check_cap("daily"):
            raise RuntimeError(
                f"Daily cost cap reached: ${self._cost_tracker._today_cost:.2f} / "
                f"${self._cost_tracker._daily_cap:.2f}"
            )

        # Check cache first
        if self.config.cache_enabled:
            cached = self._cache.get(prompt, system or "", self._primary.model)
            if cached:
                logger.debug("Cache hit for prompt")
                return LLMResponse(
                    content=cached,
                    provider=self._primary.provider,
                    model=self._primary.model,
                    input_tokens=0,
                    output_tokens=0,
                    total_tokens=0,
                    cost_usd=0.0,
                    latency_ms=0.0,
                    cached=True,
                )

        # Estimate tokens for rate limiting
        estimated_tokens = self._primary.get_token_count(prompt)
        if system:
            estimated_tokens += self._primary.get_token_count(system)

        # Acquire rate limit
        await self._rate_limiter.acquire(estimated_tokens)

        # Try primary provider
        last_error = None
        for client in [self._primary, self._fallback]:
            if not client._initialized:
                continue

            try:
                response = await client.analyze(
                    prompt=prompt,
                    system=system,
                    json_schema=json_schema,
                    max_tokens=max_tokens,
                )

                # Record cost
                self._cost_tracker.record(
                    provider=client.provider.value,
                    model=client.model,
                    input_tokens=response.input_tokens,
                    output_tokens=response.output_tokens,
                    cost_usd=response.cost_usd,
                    source_type=source_type,
                    symbol=symbol,
                )

                # Cache response
                if self.config.cache_enabled:
                    self._cache.set(
                        prompt=prompt,
                        system=system or "",
                        model=client.model,
                        response=response.content,
                        ttl_seconds=cache_ttl_seconds,
                    )

                return response

            except Exception as e:
                last_error = e
                logger.warning(f"{client.provider.value} failed: {e}, trying fallback")
                continue

        # Both failed
        raise RuntimeError(f"All LLM providers failed. Last error: {last_error}")

    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics."""
        return {
            "cost": self._cost_tracker.get_usage_stats(),
            "rate_limiter": self._rate_limiter.get_status(),
        }

    @property
    def cost_tracker(self) -> CostTracker:
        """Get cost tracker for external access."""
        return self._cost_tracker


def create_llm_client(
    openai_api_key: Optional[str] = None,
    anthropic_api_key: Optional[str] = None,
    primary_provider: str = "anthropic",
    daily_cost_cap: float = 50.0,
) -> LLMClientWithFallback:
    """
    Factory function to create LLM client.

    Args:
        openai_api_key: OpenAI API key (or from env)
        anthropic_api_key: Anthropic API key (or from env)
        primary_provider: "openai" or "anthropic"
        daily_cost_cap: Maximum daily spend in USD

    Returns:
        Configured LLMClientWithFallback
    """
    config = LLMClientConfig(
        openai_api_key=openai_api_key,
        anthropic_api_key=anthropic_api_key,
        primary_provider=LLMProvider(primary_provider),
        daily_cost_cap_usd=daily_cost_cap,
    )
    return LLMClientWithFallback(config)
