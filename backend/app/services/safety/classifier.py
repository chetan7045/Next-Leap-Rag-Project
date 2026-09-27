"""Deterministic pre-LLM query classification.

Running this before retrieval keeps advice/performance/PII requests away from the
LLM entirely: no cost, no latency, and no chance of a pressured answer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.models.enums import QueryType
from app.services.safety.pii import PII_NOTICE, detect_pii

ADVICE_REFUSAL = (
    "I can share factual details about HDFC Mutual Fund schemes — expense ratio, exit load, "
    "minimum SIP, benchmark, riskometer and lock-in — but I can't recommend a fund or advise "
    "you on whether to invest. Which fact would you like to know?"
)

PERFORMANCE_REFUSAL = (
    "This assistant shares factual scheme information rather than return predictions or "
    "comparisons between funds. For published historical performance, please refer to the "
    "official scheme documents, factsheets, or AMFI."
)

OUT_OF_SCOPE_REFUSAL = (
    "I only answer factual questions about HDFC Mutual Fund schemes from my indexed sources. "
    "Ask me about fees, SIP amounts, exit load, benchmark, riskometer, or lock-in periods."
)

# --- Advice: recommending, suitability, or personal allocation -------------------
_ADVICE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bshould\s+(?:i|we)\b", re.I),
    re.compile(r"\b(?:can|would)\s+you\s+recommend\b", re.I),
    re.compile(r"\b(?:which|what)\s+(?:fund|scheme|mutual\s+fund)\s+(?:should|do)\b", re.I),
    re.compile(r"\b(?:is|are)\s+(?:it|this|that)\s+(?:a\s+)?(?:good|bad|safe|worth)\b", re.I),
    re.compile(r"\b(?:good|safe|worth|better)\s+(?:fund|scheme|investment|option)\b", re.I),
    re.compile(r"\b(?:best|top|ideal|right)\s+(?:fund|scheme|mutual\s+fund)\b", re.I),
    re.compile(r"\b(?:suitable|fit)\s+for\s+me\b", re.I),
    # "Is HDFC Small Cap Fund good for me?" — the subject is a fund name, not
    # "it"/"this", so a judgement word followed by "for me" is the reliable signal.
    re.compile(
        r"\b(?:is|are|was|were|does|do)\b[^.?!]{0,50}?\b(?:good|bad|safe|risky|worth|better|"
        r"best|poor|great)\b\s+(?:for|to)\s+(?:me|my\s+\w+|someone|a\s+\w+\s+my\s+age)\b",
        re.I,
    ),
    re.compile(r"\b(?:good|safe|worth|suitable)\b[^.?!]{0,20}\bfor\s+me\b", re.I),
    re.compile(r"\bis\s+(?:it|this|that|the\s+\w+\s+fund)\s+(?:good|bad|safe|risky|worth)\b", re.I),
    re.compile(r"\b(?:allocate|allocation|portfolio|sip\s+amount\s+should)\b", re.I),
    re.compile(r"\b(?:buy|sell|switch|redeem|invest\s+in)\s+(?:the\s+|this\s+|a\s+)?"
               r"(?:hdfc|fund|scheme|elss|large\s*cap|small\s*cap|flexi|balanced)", re.I),
    re.compile(r"\b(?:start|begin)\s+(?:a\s+)?(?:sip|invest)", re.I),
    re.compile(r"\bmy\s+(?:risk|age|salary|income|portfolio|holdings|net\s+worth|goal)", re.I),
    re.compile(r"\bhow\s+much\s+(?:should|must)\s+i\b", re.I),
    re.compile(r"\b(?:tell|advise)\s+me\s+(?:whether|if|what|how)\b.*\binvest", re.I),
    re.compile(r"\bworth\s+investing\b", re.I),
)

# --- Performance: future returns, comparison, projections -----------------------
_PERFORMANCE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b(?:will|would|can)\s+(?:give|generate|earn|return|deliver|yield)", re.I),
    re.compile(r"\b(?:highest|best|better|more)\s+(?:returns?|performance|yield)", re.I),
    re.compile(r"\bexpected\s+(?:returns?|performance|yield|growth)", re.I),
    re.compile(r"\bprojected?\s+(?:returns?|performance|nav|growth|value)", re.I),
    re.compile(r"\b(?:future|coming)\s+(?:returns?|performance|nav|price|growth)", re.I),
    re.compile(r"\bhow\s+(?:much|many)\s+will\s+(?:i|it|it'?s)\b", re.I),
    re.compile(r"\bwhich\s+(?:one|fund|scheme)?\s*(?:will|should)\s+(?:be|perform|do)\s+better", re.I),
    re.compile(r"\bcompare\s+(?:the\s+)?(?:performance|returns?)\b", re.I),
    re.compile(r"\b(?:cagr|xirr|irr|calculator|projection|simulate|forecast)\b", re.I),
    re.compile(r"\b\d+\s*(?:years?|decades?)\s+from\s+now\b", re.I),
    re.compile(r"\bnav\s+(?:in|after)\s+\d{4}\b", re.I),
    # "What will the NAV be in 2030?" / "what will HDFC Small Cap's value be in 5 years"
    re.compile(
        r"\b(?:nav|navs|value|price|worth|growth)\b[^.?!]{0,40}?\b(?:in|after|by)\s+"
        r"(?:\d{4}|\d+\s*years?)",
        re.I,
    ),
    re.compile(r"\bwhat\s+will\b[^.?!]{0,50}?\b(?:nav|value|price|growth|return|yield|worth)\b", re.I),
    re.compile(r"\bhow\s+(?:much|high|low)\s+will\b", re.I),
    re.compile(r"\b(?:10|15|20)\s*year\s+(?:return|performance|cagr)", re.I),
)

# --- Clearly out of scope ------------------------------------------------------
_OFF_TOPIC_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b(?:tell\s+me\s+a\s+joke|joke\s+about|write\s+a\s+poem|sing\s+me)\b", re.I),
    re.compile(r"\b(?:weather|temperature|news\s+today|horoscope|recipe|cricket\s+score)\b", re.I),
    re.compile(r"\b(?:who\s+is\s+the\s+president|what\s+is\s+2\s*\+\s*2|capital\s+of\s+france)\b", re.I),
    re.compile(r"\b(?:write\s+code|debug\s+this|translate\s+to\s+python)\b", re.I),
    re.compile(r"\bstock\s+price\s+of\b", re.I),
    # Personal account data is not published scheme data.
    re.compile(r"\bhow\s+many\s+units\b", re.I),
    re.compile(r"\b(?:i|we)\s+(?:hold|held|own|owned|have\s+invested)\b", re.I),
    re.compile(r"\bmy\s+(?:units|holdings|investments?|balance|folio|portfolio\s+value)\b", re.I),
    re.compile(r"\bwhat\s+(?:did|do)\s+(?:i|we)\s+(?:hold|own|invest)\b", re.I),
    re.compile(r"\b(?:cricket|ipl|football|ipl|match\s+score|who\s+won)\b", re.I),
    # Personal contact details are never served, for any scheme.
    re.compile(
        r"\b(?:email|e-?mail\s+address|phone\s+number|contact\s+number|address|office\s+address)\b"
        r"[^.?!]{0,30}\b(?:of|for)\b",
        re.I,
    ),
    # "the fund manager's email address", "phone number for the AMC" — personal
    # contact details are never served, whatever the phrasing.
    re.compile(
        r"\b(?:e-?mail|phone|telephone|contact|postal|registered|office)\b[^.?!]{0,30}?"
        r"\b(?:address|number|details?|info|information)\b",
        re.I,
    ),
    re.compile(r"\bhow\s+do\s+i\s+contact\b", re.I),
)


@dataclass(frozen=True, slots=True)
class Classification:
    query_type: QueryType
    reason: str
    canned_answer: str | None = None

    @property
    def is_answerable(self) -> bool:
        return self.query_type is QueryType.FACTUAL


def _first_match(text: str, patterns: tuple[re.Pattern[str], ...]) -> str | None:
    for pattern in patterns:
        match = pattern.search(text)
        if match:
            return match.group(0).strip().lower()
    return None


def classify(query: str) -> Classification:
    """Classify a user question into one of the fixed query types.

    PII is checked first: it must win even when the sentence also looks like advice.
    """
    text = " ".join(query.split())
    if not text:
        return Classification(QueryType.OUT_OF_SCOPE, "empty", OUT_OF_SCOPE_REFUSAL)

    finding = detect_pii(text)
    if finding is not None:
        return Classification(
            QueryType.PII_RISK,
            f"pii:{','.join(finding.categories)}",
            PII_NOTICE,
        )

    advice_hit = _first_match(text, _ADVICE_PATTERNS)
    performance_hit = _first_match(text, _PERFORMANCE_PATTERNS)
    off_topic_hit = _first_match(text, _OFF_TOPIC_PATTERNS)

    # Performance is checked before advice: "which fund will give higher returns?"
    # is a prediction request, not a suitability request.
    if performance_hit:
        return Classification(QueryType.PERFORMANCE, f"match:{performance_hit}", PERFORMANCE_REFUSAL)
    if advice_hit:
        return Classification(QueryType.ADVICE, f"match:{advice_hit}", ADVICE_REFUSAL)
    if off_topic_hit:
        return Classification(QueryType.OUT_OF_SCOPE, f"match:{off_topic_hit}", OUT_OF_SCOPE_REFUSAL)

    return Classification(QueryType.FACTUAL, "default", None)
