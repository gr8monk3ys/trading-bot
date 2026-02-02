"""
Earnings Call Analysis Prompts

Structured prompts for extracting trading signals from earnings call transcripts.
"""

EARNINGS_SYSTEM_PROMPT = """You are an expert equity analyst specializing in earnings call analysis.
Your job is to extract actionable trading signals from earnings call transcripts.

Focus on:
1. Management tone and confidence level
2. Guidance changes (raised, lowered, maintained)
3. Key metrics and their context
4. Analyst Q&A sentiment
5. Forward-looking statements and their implications
6. Competitive positioning changes
7. Risk factors mentioned

Be objective and data-driven. Avoid speculation."""

EARNINGS_ANALYSIS_PROMPT = """Analyze this earnings call transcript for {symbol}.

COMPANY: {symbol}
QUARTER: {quarter}
DATE: {date}

TRANSCRIPT:
{transcript}

Provide a structured analysis with:
1. Overall sentiment score (-1 = very bearish to +1 = very bullish)
2. Your confidence in this assessment (0 to 1)
3. Management tone classification
4. Guidance change classification
5. Key insights (3-5 bullet points)
6. Risks mentioned
7. Opportunities identified
8. Reasoning for your assessment

Respond in JSON format:
{{
    "sentiment_score": <float -1 to +1>,
    "confidence": <float 0 to 1>,
    "management_tone": "<confident|cautious|defensive|optimistic>",
    "guidance_change": "<raised|lowered|maintained|no_guidance>",
    "key_insights": ["<insight 1>", "<insight 2>", ...],
    "risks": ["<risk 1>", "<risk 2>", ...],
    "opportunities": ["<opportunity 1>", "<opportunity 2>", ...],
    "key_metrics": {{
        "<metric_name>": "<context and value>",
        ...
    }},
    "analyst_sentiment": "<bullish|bearish|mixed>",
    "surprise_factor": <float -1 to +1>,
    "reasoning": "<2-3 sentences explaining your analysis>"
}}

Respond ONLY with valid JSON. No additional text."""

EARNINGS_ANALYSIS_SCHEMA = {
    "type": "object",
    "required": [
        "sentiment_score",
        "confidence",
        "management_tone",
        "guidance_change",
        "key_insights",
        "reasoning",
    ],
    "properties": {
        "sentiment_score": {
            "type": "number",
            "minimum": -1,
            "maximum": 1,
            "description": "Overall sentiment: -1 (very bearish) to +1 (very bullish)",
        },
        "confidence": {
            "type": "number",
            "minimum": 0,
            "maximum": 1,
            "description": "Confidence in assessment: 0 to 1",
        },
        "management_tone": {
            "type": "string",
            "enum": ["confident", "cautious", "defensive", "optimistic"],
            "description": "Classification of management's tone",
        },
        "guidance_change": {
            "type": "string",
            "enum": ["raised", "lowered", "maintained", "no_guidance"],
            "description": "How guidance changed compared to prior quarter",
        },
        "key_insights": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
            "maxItems": 5,
            "description": "3-5 key insights from the call",
        },
        "risks": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Risks mentioned or identified",
        },
        "opportunities": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Opportunities mentioned or identified",
        },
        "key_metrics": {
            "type": "object",
            "additionalProperties": {"type": "string"},
            "description": "Key metrics and their context",
        },
        "analyst_sentiment": {
            "type": "string",
            "enum": ["bullish", "bearish", "mixed"],
            "description": "Overall sentiment from analyst Q&A",
        },
        "surprise_factor": {
            "type": "number",
            "minimum": -1,
            "maximum": 1,
            "description": "How surprising was the content: -1 (negatively) to +1 (positively)",
        },
        "reasoning": {
            "type": "string",
            "description": "2-3 sentences explaining the analysis",
        },
    },
}


def format_earnings_prompt(
    symbol: str,
    quarter: str,
    date: str,
    transcript: str,
    max_chars: int = 30000,
) -> str:
    """
    Format earnings analysis prompt with transcript.

    Args:
        symbol: Stock symbol
        quarter: Fiscal quarter (e.g., "Q4 2024")
        date: Earnings date
        transcript: Full transcript text
        max_chars: Maximum characters to include (for token limits)

    Returns:
        Formatted prompt string
    """
    # Truncate transcript if necessary
    if len(transcript) > max_chars:
        transcript = transcript[:max_chars] + "\n\n[TRANSCRIPT TRUNCATED FOR LENGTH]"

    return EARNINGS_ANALYSIS_PROMPT.format(
        symbol=symbol,
        quarter=quarter,
        date=date,
        transcript=transcript,
    )
