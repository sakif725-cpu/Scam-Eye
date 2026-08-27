"""Explainable AI & Suspicious Indicator Extraction Module.

Identifies fraud patterns, linguistic red flags, and high-risk triggers to provide
transparent, human-understandable explanations for the AI's predictions.

Important:
This module does NOT replace the machine learning classifier. The trained ML/Transformer
model remains the primary decision engine. This module supplies interpretability,
extracted evidence, and explainable safety advice.
"""

from typing import Dict, List, Any
import re


# Comprehensive dictionary of fraud patterns, regexes, and severity levels
FRAUD_INDICATOR_RULES = [
    {
        "id": "OTP_OR_PIN_SOLICITATION",
        "category": "Authentication Hijacking",
        "severity": "CRITICAL",
        "weight": 0.35,
        "description": "Requests one-time password (OTP), MPIN, or UPI PIN sharing",
        "advice": "Banks and reputable institutions NEVER ask for your OTP, MPIN, or password over call or SMS.",
        "patterns": [
            r"\b(?:otp|one[\s-]?time[\s-]?password|mpin|upi[\s-]?pin|security[\s-]?pin|verification[\s-]?code)\b",
            r"\b(?:enter|share|tell|send|confirm)\s+(?:your\s+)?(?:otp|pin|password|code)\b",
            r"\b(?:apna\s+)?otp\s+(?:batayein|bhejein|share\s+karein)\b"
        ]
    },
    {
        "id": "KYC_ACCOUNT_SUSPENSION_THREAT",
        "category": "Coercive Account Threat",
        "severity": "HIGH",
        "weight": 0.30,
        "description": "Threatens immediate account blocking, KYC expiration, or penalty",
        "advice": "Legitimate banks do not suspend accounts via SMS links. Visit your official branch or app.",
        "patterns": [
            r"\b(?:account|pan|aadhaar|kyc|sim|card|yono)\b.*\b(?:blocked|suspended|deactivated|expire[d]?|terminated)\b",
            r"\b(?:temporarily\s+blocked|missing\s+pan\s+kyc|kyc\s+verification\s+pending|sim\s+card\s+will\s+be\s+blocked)\b",
            r"\b(?:khata|account)\s+(?:block|band|suspend)\s+(?:ho\s+gaya|hone\s+wala)\b",
            r"\b(?:avoid\s+penalty|avoid\s+suspension|restore\s+access)\b"
        ]
    },
    {
        "id": "LOTTERY_PRIZE_FAKE_REWARD",
        "category": "Advance-Fee / Prize Scam",
        "severity": "HIGH",
        "weight": 0.25,
        "description": "Claims unexpected lottery, bumper draw, lucky winner, or iPhone prize",
        "advice": "If you did not enter an official lottery, you cannot win one. Never pay 'customs' or 'processing' fees.",
        "patterns": [
            r"\b(?:won|winner|cash\s+prize|lucky\s+draw|bumper\s+draw|kbc|festive\s+bonus|lottery)\b",
            r"\b(?:won\s+a\s+brand\s+new|selected\s+for\s+a\s+prize|reward\s+points\s+worth|claim\s+your\s+free)\b",
            r"\b(?:inaam|kismat|lucky\s+contest|jackpot)\b",
            r"\b(?:pay\s+[\d,]+\s+(?:processing\s+fee|customs\s+duty))\b"
        ]
    },
    {
        "id": "UPI_REVERSE_PAYMENT_TRAP",
        "category": "Payment / Refund Trap",
        "severity": "CRITICAL",
        "weight": 0.35,
        "description": "Asks to enter UPI PIN or scan QR code to receive money",
        "advice": "Entering your UPI PIN ALWAYS debits money from your account. You NEVER need a PIN to receive money.",
        "patterns": [
            r"\b(?:enter|put|type)\s+(?:your\s+)?(?:upi\s+)?pin\s+to\s+(?:receive|accept|claim|get)\b",
            r"\b(?:scan\s+(?:this\s+)?qr\s+code\s+to\s+receive|claim\s+cashback)\b",
            r"\b(?:upi\s+pin\s+enter\s+karein\s+aur\s+paise\s+payein)\b",
            r"\b(?:payment\s+request\s+of|refund\s+is\s+approved|subsidy\s+is\s+waiting)\b"
        ]
    },
    {
        "id": "SUSPICIOUS_UNVERIFIED_URL",
        "category": "Phishing Link",
        "severity": "HIGH",
        "weight": 0.25,
        "description": "Contains shortened, deceptive, or non-standard domain URLs",
        "advice": "Never click unverified links from SMS. Always type the official domain manually into your browser.",
        "patterns": [
            r"https?://(?:bit\.ly|tinyurl\.com|t\.me|goo\.gl|is\.gd|cutt\.ly)/\S+",
            r"https?://\S+\.(?:xyz|top|info|pw|cc|tk|ml|ga|cf|gq|work|click|club|site|online|space|live|rest|top)/\S*",
            r"https?://[^\s/]*(?:sbi|hdfc|icici|axis|yono|paytm|kyc|support|bank|netflix|amazon|parivahan)[^\s/]*\.(?:top|xyz|info|cc|net|site|org|pw)"
        ]
    },
    {
        "id": "HIGH_PRESSURE_URGENCY_FEAR",
        "category": "Psychological Manipulation",
        "severity": "MEDIUM",
        "weight": 0.20,
        "description": "Imposes tight time pressure or threats of police, legal, or utility disconnection",
        "advice": "Scammers manufacture artificial panic to prevent rational thinking. Take a breath and independently verify.",
        "patterns": [
            r"\b(?:immediately|urgent|urgently|within\s+\d+\s+hours?|tonight\s+at|final\s+reminder)\b",
            r"\b(?:arrest\s+warrant|police\s+cyber\s+cell|court\s+summons|power\s+will\s+be\s+cut\s+off|disconnected\s+tonight)\b",
            r"\b(?:turant|aaj\s+hi|jaldi\s+karein|warna\s+police)\b"
        ]
    },
    {
        "id": "AUTHORITY_OR_BRAND_IMPERSONATION",
        "category": "Impersonation",
        "severity": "MEDIUM",
        "weight": 0.20,
        "description": "Claims representation from banks, telecom providers, government, or law enforcement",
        "advice": "Verify the sender's identity through official public channels, not the contact info provided in the message.",
        "patterns": [
            r"\b(?:sbi|hdfc|icici|axis|kotak|yono|trai|income\s+tax\s+department|customs\s+officer|cyber\s+crime\s+cell)\b",
            r"\b(?:electricity\s+officer|epf\s+provident\s+fund|pm\s+kisan|airtel\s+customer\s+care)\b"
        ]
    },
    {
        "id": "JOB_OR_CRYPTO_GET_RICH_SCAM",
        "category": "Investment / Task Fraud",
        "severity": "HIGH",
        "weight": 0.25,
        "description": "Offers unrealistic returns, part-time video liking tasks, or crypto bots",
        "advice": "No legitimate job pays thousands daily for liking videos, and no trading bot guarantees 30% monthly gains.",
        "patterns": [
            r"\b(?:work\s+from\s+home\s+job|earn\s+rs\s+[\d,]+\s+daily|liking\s+youtube\s+videos|part[\s-]?time\s+job)\b",
            r"\b(?:guaranteed\s+returns\s+of\s+\d+%|crypto\s+trading\s+bot|invest\s+rs\s+[\d,]+\s+and\s+get)\b",
            r"\b(?:telegram\s+@|t\.me/)\b"
        ]
    }
]


def explain_prediction(text: str) -> Dict[str, Any]:
    """Scans text against comprehensive fraud heuristic patterns and provides explanation.

    Args:
        text: Input message or call transcript text.

    Returns:
        Structured dictionary with detected indicators, highlighted matches,
        severity rating, and user-facing security advice.
    """
    if not isinstance(text, str) or not text.strip():
        return {
            "indicator_count": 0,
            "detected_indicators": [],
            "risk_boost": 0.0,
            "summary": "No text provided for analysis.",
            "highlighted_spans": []
        }

    detected_indicators = []
    total_boost = 0.0
    highlighted_spans = []

    for rule in FRAUD_INDICATOR_RULES:
        rule_matches = []
        for pat in rule["patterns"]:
            matches = list(re.finditer(pat, text, flags=re.IGNORECASE))
            for m in matches:
                matched_text = m.group(0)
                span = m.span()
                rule_matches.append(matched_text)
                highlighted_spans.append({
                    "span": span,
                    "text": matched_text,
                    "category": rule["category"],
                    "severity": rule["severity"]
                })

        if rule_matches:
            detected_indicators.append({
                "id": rule["id"],
                "category": rule["category"],
                "severity": rule["severity"],
                "description": rule["description"],
                "advice": rule["advice"],
                "weight": rule["weight"],
                "matches": list(set(rule_matches))
            })
            total_boost += rule["weight"]

    # Normalize total indicator boost (capped at 0.50)
    capped_boost = min(round(total_boost, 3), 0.50)

    # Generate user-friendly summary
    if not detected_indicators:
        summary = "No suspicious linguistic or structural fraud indicators detected in the text."
    else:
        severities = [ind["severity"] for ind in detected_indicators]
        if "CRITICAL" in severities:
            summary = f"Identified {len(detected_indicators)} suspicious pattern(s) including CRITICAL security red flags (such as credentials/OTP harvesting or UPI deception)."
        elif "HIGH" in severities:
            summary = f"Identified {len(detected_indicators)} suspicious pattern(s) with HIGH risk attributes (such as account suspension threats, phishing links, or prize claims)."
        else:
            summary = f"Identified {len(detected_indicators)} mild suspicious indicator(s) (such as urgency cues or impersonation keywords)."

    return {
        "indicator_count": len(detected_indicators),
        "detected_indicators": detected_indicators,
        "risk_boost": capped_boost,
        "summary": summary,
        "highlighted_spans": highlighted_spans
    }


if __name__ == "__main__":
    test_msg = "URGENT: Your SBI account is blocked. Update PAN KYC at http://bit.ly/sbi-kyc or share OTP to avoid penalty."
    exp = explain_prediction(test_msg)
    print("Explainability Report:")
    print(f"Summary: {exp['summary']}")
    print(f"Indicators Detected: {exp['indicator_count']}")
    for ind in exp["detected_indicators"]:
        print(f" - [{ind['severity']}] {ind['category']}: {ind['description']} (Matches: {ind['matches']})")

