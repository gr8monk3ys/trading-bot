"""
SEC Filing Analysis Prompts

Structured prompts for extracting trading signals from SEC filings (10-K, 10-Q, 8-K).
"""

SEC_SYSTEM_PROMPT = """You are an expert securities analyst specializing in SEC filing analysis.
Your job is to extract actionable trading signals from SEC filings.

Focus on:
1. New or changed risk factors
2. Material changes in business
3. Litigation and legal matters
4. Related party transactions
5. Going concern language
6. Management discussion and analysis (MD&A) tone
7. Significant accounting changes
8. Competitive positioning

Be thorough and flag any red flags or material changes."""

SEC_FILING_ANALYSIS_PROMPT = """Analyze this SEC filing for {symbol}.

COMPANY: {symbol}
FILING TYPE: {filing_type}
FILING DATE: {filing_date}

{section_header}
{filing_content}

Provide a structured analysis with:
1. Overall sentiment signal (-1 = very concerning to +1 = very positive)
2. Your confidence in this assessment (0 to 1)
3. Material changes identified
4. New risk factors
5. Removed or reduced risk factors
6. Litigation mentions
7. Related party concerns
8. Going concern risk assessment
9. Key insights and reasoning

Respond in JSON format:
{{
    "sentiment_score": <float -1 to +1>,
    "confidence": <float 0 to 1>,
    "material_changes": ["<change 1>", "<change 2>", ...],
    "new_risk_factors": ["<risk 1>", "<risk 2>", ...],
    "removed_risk_factors": ["<removed risk 1>", ...],
    "litigation_mentions": ["<litigation 1>", ...],
    "related_party_concerns": ["<concern 1>", ...],
    "going_concern_risk": <boolean>,
    "key_insights": ["<insight 1>", "<insight 2>", ...],
    "risks": ["<risk 1>", "<risk 2>", ...],
    "opportunities": ["<opportunity 1>", "<opportunity 2>", ...],
    "reasoning": "<2-3 sentences explaining your analysis>"
}}

Respond ONLY with valid JSON. No additional text."""

SEC_FILING_ANALYSIS_SCHEMA = {
    "type": "object",
    "required": [
        "sentiment_score",
        "confidence",
        "material_changes",
        "going_concern_risk",
        "reasoning",
    ],
    "properties": {
        "sentiment_score": {
            "type": "number",
            "minimum": -1,
            "maximum": 1,
            "description": "Sentiment: -1 (very concerning) to +1 (very positive)",
        },
        "confidence": {
            "type": "number",
            "minimum": 0,
            "maximum": 1,
            "description": "Confidence in assessment: 0 to 1",
        },
        "material_changes": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Material changes identified in the filing",
        },
        "new_risk_factors": {
            "type": "array",
            "items": {"type": "string"},
            "description": "New risk factors added since prior filing",
        },
        "removed_risk_factors": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Risk factors removed or reduced since prior filing",
        },
        "litigation_mentions": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Litigation and legal matters mentioned",
        },
        "related_party_concerns": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Related party transaction concerns",
        },
        "going_concern_risk": {
            "type": "boolean",
            "description": "Whether going concern language is present",
        },
        "key_insights": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Key insights from the filing",
        },
        "risks": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Overall risks identified",
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


SEC_COMPARISON_PROMPT = """Compare these two SEC filings and identify changes.

COMPANY: {symbol}
FILING TYPE: {filing_type}

CURRENT FILING ({current_date}):
{current_content}

PREVIOUS FILING ({previous_date}):
{previous_content}

Identify:
1. New risk factors added
2. Risk factors removed
3. Significant language changes in MD&A
4. Material changes in business
5. New litigation or legal matters
6. Changes in going concern status

Respond in JSON format with sentiment_score, new_risk_factors, removed_risk_factors,
material_changes, and detailed reasoning."""


def format_sec_filing_prompt(
    symbol: str,
    filing_type: str,
    filing_date: str,
    filing_content: str,
    section: str = "Risk Factors",
    max_chars: int = 25000,
) -> str:
    """
    Format SEC filing analysis prompt.

    Args:
        symbol: Stock symbol
        filing_type: 10-K, 10-Q, 8-K, etc.
        filing_date: Filing date
        filing_content: Filing content (specific section)
        section: Section being analyzed
        max_chars: Maximum characters to include

    Returns:
        Formatted prompt string
    """
    if len(filing_content) > max_chars:
        filing_content = filing_content[:max_chars] + "\n\n[FILING TRUNCATED FOR LENGTH]"

    section_header = f"SECTION: {section}" if section else ""

    return SEC_FILING_ANALYSIS_PROMPT.format(
        symbol=symbol,
        filing_type=filing_type,
        filing_date=filing_date,
        section_header=section_header,
        filing_content=filing_content,
    )
