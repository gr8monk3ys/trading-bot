"""
News Theme Analysis Prompts

Structured prompts for extracting trading signals and themes from news articles.
"""

NEWS_SYSTEM_PROMPT = """You are an expert news analyst specializing in financial markets.
Your job is to extract actionable trading signals and themes from news articles.

Focus on:
1. Primary narrative/theme identification
2. Catalyst identification (product launches, M&A, earnings, etc.)
3. Time sensitivity of the news
4. Market impact assessment
5. Sentiment analysis
6. Cross-company implications
7. Sector-wide impact

Be objective and focus on material news that could affect stock prices."""

NEWS_THEME_ANALYSIS_PROMPT = """Analyze these news articles for {symbol}.

COMPANY: {symbol}
TIME PERIOD: Last {hours} hours
ARTICLE COUNT: {article_count}

ARTICLES:
{articles}

Provide a structured analysis with:
1. Overall sentiment signal (-1 = very bearish to +1 = very bullish)
2. Your confidence in this assessment (0 to 1)
3. Primary theme identified
4. Secondary themes
5. Catalysts identified
6. Time sensitivity of the news
7. Estimated market impact
8. Key insights and reasoning

Respond in JSON format:
{{
    "sentiment_score": <float -1 to +1>,
    "confidence": <float 0 to 1>,
    "primary_theme": "<theme>",
    "secondary_themes": ["<theme 1>", "<theme 2>", ...],
    "catalysts_identified": ["<catalyst 1>", "<catalyst 2>", ...],
    "time_sensitivity": "<immediate|short_term|long_term>",
    "market_impact_estimate": "<high|medium|low>",
    "key_insights": ["<insight 1>", "<insight 2>", ...],
    "risks": ["<risk 1>", "<risk 2>", ...],
    "opportunities": ["<opportunity 1>", "<opportunity 2>", ...],
    "reasoning": "<2-3 sentences explaining your analysis>"
}}

Respond ONLY with valid JSON. No additional text."""

NEWS_THEME_ANALYSIS_SCHEMA = {
    "type": "object",
    "required": [
        "sentiment_score",
        "confidence",
        "primary_theme",
        "time_sensitivity",
        "reasoning",
    ],
    "properties": {
        "sentiment_score": {
            "type": "number",
            "minimum": -1,
            "maximum": 1,
            "description": "Sentiment: -1 (very bearish) to +1 (very bullish)",
        },
        "confidence": {
            "type": "number",
            "minimum": 0,
            "maximum": 1,
            "description": "Confidence in assessment: 0 to 1",
        },
        "primary_theme": {
            "type": "string",
            "description": "Primary theme of the news (e.g., product_launch, earnings, acquisition)",
        },
        "secondary_themes": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Secondary themes identified",
        },
        "catalysts_identified": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Specific catalysts that could move the stock",
        },
        "time_sensitivity": {
            "type": "string",
            "enum": ["immediate", "short_term", "long_term"],
            "description": "How time-sensitive is this news",
        },
        "market_impact_estimate": {
            "type": "string",
            "enum": ["high", "medium", "low"],
            "description": "Estimated market impact",
        },
        "key_insights": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Key insights from the news",
        },
        "risks": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Risks identified",
        },
        "opportunities": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Opportunities identified",
        },
        "reasoning": {
            "type": "string",
            "description": "2-3 sentences explaining the analysis",
        },
    },
}

# Common news themes
NEWS_THEMES = [
    "product_launch",
    "earnings_preview",
    "earnings_reaction",
    "acquisition",
    "divestiture",
    "partnership",
    "regulatory",
    "litigation",
    "management_change",
    "guidance_update",
    "analyst_action",
    "insider_activity",
    "competitive",
    "macro_impact",
    "sector_rotation",
    "technical_breakout",
    "dividend",
    "buyback",
    "restructuring",
    "labor",
]


def format_news_theme_prompt(
    symbol: str,
    articles: list,
    hours: int = 24,
    max_articles: int = 10,
    max_chars_per_article: int = 2000,
) -> str:
    """
    Format news theme analysis prompt.

    Args:
        symbol: Stock symbol
        articles: List of article dicts with 'headline', 'summary', 'source', 'created_at'
        hours: Time period in hours
        max_articles: Maximum articles to include
        max_chars_per_article: Maximum characters per article

    Returns:
        Formatted prompt string
    """
    # Format articles
    formatted_articles = []
    for i, article in enumerate(articles[:max_articles], 1):
        headline = article.get("headline", "")
        summary = article.get("summary", "")[:max_chars_per_article]
        source = article.get("source", "Unknown")
        date = article.get("created_at", "")

        formatted_articles.append(
            f"[Article {i}]\n"
            f"Source: {source}\n"
            f"Date: {date}\n"
            f"Headline: {headline}\n"
            f"Summary: {summary}\n"
        )

    articles_text = "\n---\n".join(formatted_articles)

    return NEWS_THEME_ANALYSIS_PROMPT.format(
        symbol=symbol,
        hours=hours,
        article_count=len(articles),
        articles=articles_text,
    )
