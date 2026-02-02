"""
Fed Speech Analysis Prompts

Structured prompts for extracting macro trading signals from Federal Reserve communications.
"""

FED_SYSTEM_PROMPT = """You are an expert macro analyst specializing in Federal Reserve communications.
Your job is to extract actionable signals about monetary policy and market implications from Fed speeches.

Focus on:
1. Interest rate trajectory signals (hawkish/dovish)
2. Inflation commentary and concerns
3. Labor market assessment
4. Economic growth outlook
5. Financial stability concerns
6. Forward guidance language
7. Deviation from consensus/prior statements

Be objective and focus on policy implications for financial markets."""

FED_SPEECH_ANALYSIS_PROMPT = """Analyze this Federal Reserve speech for market implications.

SPEAKER: {speaker}
TITLE: {title}
DATE: {date}
EVENT TYPE: {event_type}
DAYS TO/FROM FOMC: {fomc_days}

SPEECH TEXT:
{speech_text}

Provide a structured analysis with:
1. Overall market sentiment signal (-1 = very hawkish/bearish for equities to +1 = very dovish/bullish)
2. Your confidence in this assessment (0 to 1)
3. Rate expectations classification
4. Key themes identified
5. Policy signals
6. Market implications by asset class
7. Deviation from consensus/prior communications
8. Reasoning for your assessment

Respond in JSON format:
{{
    "sentiment_score": <float -1 to +1>,
    "confidence": <float 0 to 1>,
    "rate_expectations": "<hawkish|dovish|neutral>",
    "key_themes": ["<theme 1>", "<theme 2>", ...],
    "policy_signals": ["<signal 1>", "<signal 2>", ...],
    "market_implications": {{
        "equities": "<bullish|bearish|neutral>",
        "bonds": "<bullish|bearish|neutral>",
        "dollar": "<bullish|bearish|neutral>"
    }},
    "key_insights": ["<insight 1>", "<insight 2>", ...],
    "risks": ["<risk 1>", "<risk 2>", ...],
    "opportunities": ["<opportunity 1>", "<opportunity 2>", ...],
    "deviation_from_consensus": <float -1 to +1>,
    "reasoning": "<2-3 sentences explaining your analysis>"
}}

Respond ONLY with valid JSON. No additional text."""

FED_SPEECH_ANALYSIS_SCHEMA = {
    "type": "object",
    "required": [
        "sentiment_score",
        "confidence",
        "rate_expectations",
        "key_themes",
        "reasoning",
    ],
    "properties": {
        "sentiment_score": {
            "type": "number",
            "minimum": -1,
            "maximum": 1,
            "description": "Market sentiment: -1 (hawkish/bearish) to +1 (dovish/bullish)",
        },
        "confidence": {
            "type": "number",
            "minimum": 0,
            "maximum": 1,
            "description": "Confidence in assessment: 0 to 1",
        },
        "rate_expectations": {
            "type": "string",
            "enum": ["hawkish", "dovish", "neutral"],
            "description": "Rate path expectations from the speech",
        },
        "key_themes": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
            "maxItems": 5,
            "description": "Key themes discussed",
        },
        "policy_signals": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Specific policy signals identified",
        },
        "market_implications": {
            "type": "object",
            "properties": {
                "equities": {
                    "type": "string",
                    "enum": ["bullish", "bearish", "neutral"],
                },
                "bonds": {
                    "type": "string",
                    "enum": ["bullish", "bearish", "neutral"],
                },
                "dollar": {
                    "type": "string",
                    "enum": ["bullish", "bearish", "neutral"],
                },
            },
            "description": "Market implications by asset class",
        },
        "key_insights": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Key insights from the speech",
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
        "deviation_from_consensus": {
            "type": "number",
            "minimum": -1,
            "maximum": 1,
            "description": "How much this deviates from consensus: -1 (more hawkish) to +1 (more dovish)",
        },
        "reasoning": {
            "type": "string",
            "description": "2-3 sentences explaining the analysis",
        },
    },
}


def format_fed_speech_prompt(
    speaker: str,
    title: str,
    date: str,
    event_type: str,
    fomc_days: int,
    speech_text: str,
    max_chars: int = 25000,
) -> str:
    """
    Format Fed speech analysis prompt.

    Args:
        speaker: Fed official name
        title: Speech title
        date: Speech date
        event_type: Type of event (speech, testimony, press conference)
        fomc_days: Days to/from nearest FOMC meeting
        speech_text: Full speech text
        max_chars: Maximum characters to include

    Returns:
        Formatted prompt string
    """
    if len(speech_text) > max_chars:
        speech_text = speech_text[:max_chars] + "\n\n[SPEECH TRUNCATED FOR LENGTH]"

    return FED_SPEECH_ANALYSIS_PROMPT.format(
        speaker=speaker,
        title=title,
        date=date,
        event_type=event_type,
        fomc_days=fomc_days,
        speech_text=speech_text,
    )
