"""
🕵️‍♂️ The Creepiness Index: Privacy Policy Hazard Scanner
Hybrid NLP pipeline — TensorFlow local scoring + Gemini semantic analysis
"""

import os
import json
import re
import math
import time
import streamlit as st

# ──────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Creepiness Index",
    page_icon="🕵️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ──────────────────────────────────────────────────────────────
# CSS — Cyberpunk / dark terminal aesthetic
# ──────────────────────────────────────────────────────────────
STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Exo+2:wght@300;600;900&display=swap');

/* ---------- root tokens ---------- */
:root {
    --bg:       #0a0e1a;
    --bg2:      #0f1526;
    --bg3:      #141c30;
    --border:   #1e2d50;
    --accent:   #00f5d4;
    --accent2:  #f700ff;
    --warn:     #ff6b35;
    --safe:     #39ff14;
    --text:     #c8d8ff;
    --dim:      #5a6a8a;
    --danger:   #ff2d55;
    --glow:     0 0 12px rgba(0,245,212,.35);
    --glow2:    0 0 16px rgba(247,0,255,.3);
}

/* ---------- global ---------- */
html, body, [data-testid="stAppViewContainer"] { background: var(--bg) !important; }
[data-testid="stHeader"] { background: transparent !important; }
* { font-family: 'Exo 2', sans-serif; color: var(--text); box-sizing: border-box; }
h1,h2,h3 { font-family: 'Exo 2', sans-serif; font-weight: 900; }
.mono { font-family: 'Share Tech Mono', monospace; }

/* ---------- hero ---------- */
.hero {
    text-align: center;
    padding: 3rem 1rem 1.5rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 2rem;
}
.hero-title {
    font-size: clamp(1.8rem, 4vw, 3.2rem);
    font-weight: 900;
    letter-spacing: .04em;
    background: linear-gradient(90deg, var(--accent), var(--accent2));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.2;
}
.hero-sub {
    color: var(--dim);
    font-size: .95rem;
    margin-top: .5rem;
    font-family: 'Share Tech Mono', monospace;
}

/* ---------- panels ---------- */
.panel {
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1.2rem;
}
.panel-title {
    font-size: .7rem;
    font-family: 'Share Tech Mono', monospace;
    color: var(--accent);
    letter-spacing: .15em;
    text-transform: uppercase;
    margin-bottom: .8rem;
}

/* ---------- score ring ---------- */
.score-wrap {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: .5rem;
    padding: 1rem 0;
}
.score-ring {
    width: 160px; height: 160px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-family: 'Share Tech Mono', monospace;
    font-size: 2.8rem;
    font-weight: bold;
    position: relative;
    transition: box-shadow .4s;
}
.score-label {
    font-size: .75rem;
    font-family: 'Share Tech Mono', monospace;
    letter-spacing: .12em;
    text-transform: uppercase;
    color: var(--dim);
}
.verdict-badge {
    font-family: 'Share Tech Mono', monospace;
    font-size: .85rem;
    padding: .3rem .9rem;
    border-radius: 3px;
    letter-spacing: .1em;
    text-transform: uppercase;
}

/* ---------- hazard bars ---------- */
.hazard-row { margin-bottom: .9rem; }
.hazard-header {
    display: flex; justify-content: space-between;
    font-size: .8rem;
    font-family: 'Share Tech Mono', monospace;
    margin-bottom: .3rem;
}
.bar-track {
    background: var(--bg3);
    border-radius: 2px;
    height: 8px;
    overflow: hidden;
    border: 1px solid var(--border);
}
.bar-fill {
    height: 100%;
    border-radius: 2px;
    transition: width 1s ease;
}

/* ---------- finding cards ---------- */
.finding-card {
    background: var(--bg3);
    border-left: 3px solid var(--accent2);
    border-radius: 0 4px 4px 0;
    padding: .75rem 1rem;
    margin-bottom: .7rem;
    font-size: .88rem;
    line-height: 1.6;
}
.finding-card.critical { border-left-color: var(--danger); }
.finding-card.medium { border-left-color: var(--warn); }
.finding-card.low { border-left-color: var(--dim); }

/* ---------- tag chips ---------- */
.chip-row { display: flex; flex-wrap: wrap; gap: .4rem; margin-top: .6rem; }
.chip {
    background: rgba(0,245,212,.08);
    border: 1px solid rgba(0,245,212,.25);
    border-radius: 2px;
    padding: .15rem .5rem;
    font-size: .72rem;
    font-family: 'Share Tech Mono', monospace;
    color: var(--accent);
}

/* ---------- key-value grid ---------- */
.kv-grid { display: grid; grid-template-columns: 1fr 1fr; gap: .5rem; }
.kv-item {
    background: var(--bg3);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: .6rem .8rem;
}
.kv-key {
    font-size: .68rem;
    color: var(--dim);
    font-family: 'Share Tech Mono', monospace;
    letter-spacing: .1em;
    text-transform: uppercase;
}
.kv-val {
    font-size: 1rem;
    font-weight: 600;
    color: var(--accent);
}

/* ---------- terminal log ---------- */
.terminal {
    background: #050810;
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 1rem 1.2rem;
    font-family: 'Share Tech Mono', monospace;
    font-size: .78rem;
    color: #4a9;
    line-height: 1.8;
    max-height: 200px;
    overflow-y: auto;
}
.terminal .t-dim { color: var(--dim); }
.terminal .t-accent { color: var(--accent); }
.terminal .t-warn { color: var(--warn); }

/* ---------- streamlit widget overrides ---------- */
[data-testid="stTextArea"] textarea {
    background: var(--bg2) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: .82rem !important;
}
[data-testid="stTextInput"] input {
    background: var(--bg2) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    font-family: 'Share Tech Mono', monospace !important;
}
button[kind="primary"] {
    background: var(--accent) !important;
    color: #000 !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 3px !important;
    font-family: 'Exo 2', sans-serif !important;
    letter-spacing: .06em !important;
}
button[kind="secondary"] {
    background: transparent !important;
    border: 1px solid var(--border) !important;
    color: var(--dim) !important;
}
[data-testid="stTabs"] [role="tab"] {
    font-family: 'Share Tech Mono', monospace;
    font-size: .8rem;
    letter-spacing: .08em;
}
[data-testid="stTabs"] [aria-selected="true"] { color: var(--accent) !important; }
</style>
"""
st.markdown(STYLE, unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
# LAZY IMPORTS  (heavy; load once)
# ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_tf():
    import tensorflow as tf
    return tf

@st.cache_resource(show_spinner=False)
def load_genai():
    import google.generativeai as genai
    return genai

# ──────────────────────────────────────────────────────────────
# TIER-1: TensorFlow local risk scoring
# ──────────────────────────────────────────────────────────────
# Weighted hazard vocabulary — grouped by category
HAZARD_VOCAB = {
    "data_retention": {
        "indefinitely": 3, "permanently": 3, "perpetual": 3,
        "as long as necessary": 2, "retained": 2, "backup": 1,
        "archive": 1, "deletion request": -1, "right to erasure": -2,
        "30 days": -1, "90 days": -1,
    },
    "location_tracking": {
        "precise location": 3, "gps": 3, "geolocation": 2,
        "location data": 2, "ip address": 2, "wifi signals": 2,
        "bluetooth": 1, "approximate location": 1,
        "location services": 1, "opt-out": -1,
    },
    "third_party": {
        "sell": 3, "sold": 3, "monetize": 3, "advertising partners": 3,
        "data brokers": 3, "share with third parties": 2,
        "third-party": 2, "affiliates": 2, "business partners": 2,
        "analytics providers": 1, "not sell": -3, "never sell": -3,
        "do not share": -2,
    },
    "user_profiling": {
        "behavioral profiling": 3, "inferred": 2, "predictive": 2,
        "cross-device": 2, "fingerprinting": 3, "tracking pixel": 3,
        "cookie": 1, "session": 1, "opt out": -1,
    },
    "data_collection": {
        "biometric": 3, "face recognition": 3, "voice data": 3,
        "health data": 3, "financial information": 2, "ssn": 3,
        "device identifiers": 2, "contacts": 2, "messages": 2,
        "collect": 1, "gather": 1, "process": 1,
    },
}

def tf_risk_score(text: str) -> dict:
    """
    TensorFlow-powered local scoring:
    — lowercases & tokenises the policy text
    — builds a tf.constant token vocabulary per category
    — uses tf.strings.reduce_join + regex match counts to
      compute a weighted hazard coefficient per dimension
    Returns per-category scores (0-100) + overall Creepiness Index (0-100).
    """
    tf = load_tf()
    text_lower = text.lower()

    # Tokenise — keep multi-word phrases intact for matching
    tokens_tensor = tf.constant([text_lower], dtype=tf.string)

    category_scores = {}
    raw_hits = {}

    for category, vocab in HAZARD_VOCAB.items():
        total_weight = 0.0
        hits = {}

        for phrase, weight in vocab.items():
            # Count occurrences via tf.strings.regex_replace trick
            count_tensor = tf.strings.regex_replace(
                tokens_tensor,
                re.escape(phrase),
                "§MATCH§"
            )
            match_count = tf.strings.reduce_join(count_tensor).numpy().decode().count("§MATCH§")

            if match_count > 0:
                total_weight += weight * math.log1p(match_count)
                hits[phrase] = match_count

        raw_hits[category] = hits

        # Normalise to 0-100; positive = risky, anchored at ±10 raw
        clamped = max(-10.0, min(10.0, total_weight))
        normalised = (clamped + 10.0) / 20.0 * 100.0
        category_scores[category] = round(normalised, 1)

    # Overall Creepiness Index: weighted mean favoring worst categories
    weights = [0.20, 0.18, 0.22, 0.20, 0.20]
    keys = list(category_scores.keys())
    overall = sum(category_scores[k] * w for k, w in zip(keys, weights))

    # Doc-length signal: very short policies are suspicious
    word_count = len(text_lower.split())
    if word_count < 200:
        overall = min(100, overall + 15)

    return {
        "category_scores": category_scores,
        "raw_hits": raw_hits,
        "overall": round(min(100, max(0, overall)), 1),
        "word_count": word_count,
        "token_dims": len(text_lower.split()),
    }

# ──────────────────────────────────────────────────────────────
# TIER-2: Gemini semantic analysis
# ──────────────────────────────────────────────────────────────
GEMINI_PROMPT = """
You are a privacy law expert and data-rights analyst specialising in exposing hidden risks in Privacy Policies and Terms of Service agreements.

You will receive:
1. RAW_POLICY: the full text of a privacy policy
2. TENSOR_METRICS: a JSON object with TensorFlow-computed statistical risk scores (0-100 per category, 100 = maximum risk)

Your task: produce a single valid JSON object (no markdown fences, no preamble) with EXACTLY this schema:

{{
  "summary": "<2-sentence plain-English TL;DR of the policy's most alarming aspects>",
  "threat_level": "<CRITICAL | HIGH | MEDIUM | LOW>",
  "findings": [
    {{
      "severity": "<critical | medium | low>",
      "category": "<Data Retention | Location Tracking | Third-Party Sharing | User Profiling | Data Collection | Other>",
      "headline": "<≤12-word alarming but factual headline>",
      "detail": "<1-2 sentences explaining the real-world consequence for the user>",
      "quoted_clause": "<exact short excerpt from the policy that triggered this, ≤40 words>"
    }}
  ],
  "plain_english": [
    "<bullet: what this policy ACTUALLY allows, phrased as 'They CAN...' or 'You CANNOT...'>"
  ],
  "redeeming_clauses": [
    "<any user-protective clause found, or empty list>"
  ],
  "data_points_exposed": ["<list of specific data types collected>"],
  "recommended_actions": ["<actionable advice for the user>"],
  "gemini_score_adjustment": <integer -15 to +15 to adjust the tensor score based on semantic context>
}}

Rules:
- findings: 3-7 items, ordered by severity
- plain_english: 4-8 bullets
- If the policy is benign, findings can be short and threat_level = LOW
- NEVER output anything outside the JSON object

TENSOR_METRICS:
{tensor_json}

RAW_POLICY (first 6000 chars):
{policy_text}
"""

def gemini_analysis(text: str, tensor_metrics: dict, api_key: str) -> dict:
    genai = load_genai()
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.0-flash")

    prompt = GEMINI_PROMPT.format(
        tensor_json=json.dumps(tensor_metrics, indent=2),
        policy_text=text[:6000],
    )

    response = model.generate_content(prompt)
    raw = response.text.strip()

    # Strip accidental markdown fences
    raw = re.sub(r"^```json\s*", "", raw)
    raw = re.sub(r"^```\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    return json.loads(raw)

# ──────────────────────────────────────────────────────────────
# HELPER RENDERERS
# ──────────────────────────────────────────────────────────────
def score_color(score: float) -> str:
    if score >= 75: return "#ff2d55"
    if score >= 50: return "#ff6b35"
    if score >= 25: return "#f0c420"
    return "#39ff14"

def verdict(score: float) -> tuple[str, str]:
    if score >= 80: return "CRITICALLY INVASIVE", "#ff2d55"
    if score >= 60: return "HIGH RISK", "#ff6b35"
    if score >= 40: return "MODERATE RISK", "#f0c420"
    if score >= 20: return "LOW RISK", "#39ff14"
    return "RELATIVELY SAFE", "#00f5d4"

CATEGORY_LABELS = {
    "data_retention":   "Data Retention",
    "location_tracking":"Location Tracking",
    "third_party":      "Third-Party Sharing",
    "user_profiling":   "User Profiling",
    "data_collection":  "Data Collection",
}

def render_hazard_bars(category_scores: dict):
    for cat, score in category_scores.items():
        label = CATEGORY_LABELS.get(cat, cat)
        color = score_color(score)
        st.markdown(f"""
        <div class="hazard-row">
          <div class="hazard-header">
            <span class="mono">{label}</span>
            <span class="mono" style="color:{color}">{score:.0f} / 100</span>
          </div>
          <div class="bar-track">
            <div class="bar-fill" style="width:{score}%;background:{color}"></div>
          </div>
        </div>
        """, unsafe_allow_html=True)

def render_score_ring(score: float):
    color = score_color(score)
    v_text, v_color = verdict(score)
    st.markdown(f"""
    <div class="score-wrap">
      <div class="score-ring" style="
          background: conic-gradient({color} {score*3.6:.0f}deg, #141c30 0deg);
          box-shadow: 0 0 24px {color}55;
      ">
        <div style="
            width:128px; height:128px; border-radius:50%;
            background:var(--bg2);
            display:flex; align-items:center; justify-content:center;
            flex-direction:column;
        ">
          <span style="color:{color}; font-family:'Share Tech Mono',monospace; font-size:2.4rem; font-weight:700">{score:.0f}</span>
          <span style="color:var(--dim); font-size:.6rem; font-family:'Share Tech Mono',monospace">/ 100</span>
        </div>
      </div>
      <div class="score-label">CREEPINESS INDEX</div>
      <div class="verdict-badge" style="background:{v_color}22; border:1px solid {v_color}; color:{v_color}">{v_text}</div>
    </div>
    """, unsafe_allow_html=True)

def render_findings(findings: list):
    sev_map = {"critical": "critical", "medium": "medium", "low": "low"}
    for f in findings:
        sev = sev_map.get(f.get("severity", "low"), "low")
        sev_colors = {"critical": "#ff2d55", "medium": "#ff6b35", "low": "#5a6a8a"}
        color = sev_colors[sev]
        st.markdown(f"""
        <div class="finding-card {sev}">
          <div style="display:flex; justify-content:space-between; margin-bottom:.3rem">
            <strong style="color:{color}; font-size:.8rem; text-transform:uppercase; letter-spacing:.08em">
              ⚠ {f.get('headline','—')}
            </strong>
            <span class="chip" style="background:{color}22; border-color:{color}55; color:{color}">
              {f.get('category','—')}
            </span>
          </div>
          <div style="color:var(--text); margin-bottom:.4rem">{f.get('detail','')}</div>
          <div class="mono" style="color:var(--dim); font-size:.75rem; border-left:2px solid var(--border); padding-left:.6rem; margin-top:.3rem">
            "{f.get('quoted_clause','')}"
          </div>
        </div>
        """, unsafe_allow_html=True)

def render_terminal_log(metrics: dict):
    st.markdown(f"""
    <div class="terminal">
      <span class="t-dim">[TF-INIT]</span> tensorflow loaded · local inference active<br>
      <span class="t-dim">[TOKENISE]</span> <span class="t-accent">{metrics['word_count']}</span> words · <span class="t-accent">{metrics['token_dims']}</span> token dims mapped<br>
      <span class="t-dim">[HAZARD-VOCAB]</span> {sum(len(v) for v in HAZARD_VOCAB.values())} weighted phrases · 5 category tensors<br>
      <span class="t-dim">[SCORES]</span>
      {'  '.join(f'<span class="t-accent">{CATEGORY_LABELS.get(k,k)[:4].upper()}={v:.0f}</span>'
                 for k,v in metrics['category_scores'].items())}<br>
      <span class="t-dim">[CREEPINESS-INDEX]</span> <span class="t-warn">{metrics['overall']:.1f} / 100</span> (pre-semantic adjustment)<br>
      <span class="t-dim">[PIPELINE]</span> tensor metrics → gemini-2.0-flash → JSON payload → render
    </div>
    """, unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
# HERO
# ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-title">🕵️‍♂️ The Creepiness Index</div>
  <div class="hero-sub">Privacy Policy Hazard Scanner · Hybrid TensorFlow + Gemini NLP Pipeline</div>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
# INPUT SECTION
# ──────────────────────────────────────────────────────────────
col_left, col_right = st.columns([3, 2], gap="large")

with col_left:
    st.markdown('<div class="panel-title">◈ INPUT POLICY TEXT</div>', unsafe_allow_html=True)

    input_tab, url_tab = st.tabs(["PASTE TEXT", "FETCH FROM URL"])

    policy_text = ""

    with input_tab:
        policy_text_area = st.text_area(
            "Paste your Privacy Policy or ToS here",
            height=280,
            placeholder="Paste the full text of a Privacy Policy or Terms of Service agreement…",
            label_visibility="collapsed",
        )
        policy_text = policy_text_area

    with url_tab:
        url_input = st.text_input(
            "Policy URL",
            placeholder="https://example.com/privacy",
            label_visibility="collapsed",
        )
        if url_input and st.button("FETCH PAGE TEXT", type="secondary"):
            try:
                import urllib.request
                from html.parser import HTMLParser

                class TextExtractor(HTMLParser):
                    def __init__(self):
                        super().__init__()
                        self.text_parts = []
                        self._skip = False

                    def handle_starttag(self, tag, attrs):
                        if tag in ("script", "style", "nav", "header", "footer"):
                            self._skip = True

                    def handle_endtag(self, tag):
                        if tag in ("script", "style", "nav", "header", "footer"):
                            self._skip = False

                    def handle_data(self, data):
                        if not self._skip:
                            stripped = data.strip()
                            if stripped:
                                self.text_parts.append(stripped)

                req = urllib.request.Request(
                    url_input,
                    headers={"User-Agent": "Mozilla/5.0 (compatible; CreepinessIndexBot/1.0)"}
                )
                with urllib.request.urlopen(req, timeout=10) as r:
                    html = r.read().decode("utf-8", errors="ignore")

                parser = TextExtractor()
                parser.feed(html)
                policy_text = " ".join(parser.text_parts)
                st.success(f"✓ Fetched {len(policy_text):,} characters")
            except Exception as e:
                st.error(f"Fetch failed: {e}")

with col_right:
    st.markdown('<div class="panel-title">◈ API CONFIGURATION</div>', unsafe_allow_html=True)
    api_key = st.text_input(
        "Google Gemini API Key",
        type="password",
        placeholder="AIza…  (get free key at aistudio.google.com)",
        label_visibility="collapsed",
        help="Free tier available at Google AI Studio. Required for semantic analysis.",
    )
    st.markdown("""
    <div class="panel" style="margin-top:.8rem">
      <div class="panel-title">PIPELINE ARCHITECTURE</div>
      <div style="font-size:.8rem; line-height:1.9; color:var(--dim)">
        <span style="color:var(--accent)">TIER 1 →</span> TensorFlow local token analysis<br>
        <span style="color:var(--accent2)">TIER 2 →</span> Gemini 2.0 Flash semantic parse<br>
        <span style="color:var(--text)">OUTPUT →</span> Composite Creepiness Index score<br>
        <span style="color:var(--dim)">COST →</span> $0.00 (free tier APIs only)
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Sample policies for quick demo
    st.markdown('<div class="panel-title" style="margin-top:.6rem">◈ QUICK DEMO</div>', unsafe_allow_html=True)
    demo = st.selectbox(
        "Load a sample policy",
        ["— select —", "Meta (Facebook)", "TikTok", "Signal", "Generic Scary Policy"],
        label_visibility="collapsed",
    )

DEMO_POLICIES = {
    "Meta (Facebook)": """
We collect the content, communications and other information you provide when you use our Products. We collect information about the people, Pages, accounts, hashtags and groups you are connected to. We collect information about how you use our Products. We collect device information including device identifiers, device signals, and data from device settings. We collect information from partners about your use of their websites and apps. We use the information we have to deliver our Products and to personalize features and content. We use the information we have to show you ads, offers and other sponsored content that we believe will be meaningful to you. We provide measurement, analytics, and other business services. We share information with third-party partners who use it to deliver personalized ads. We transfer, store or process your information in countries outside of where you live. We retain data as long as it is necessary to provide you with our services or until you delete your account, whichever comes first, though backup copies may persist indefinitely. We share your precise location data with advertising partners, affiliates and business partners. We may sell aggregated or de-identified information to data brokers. We use biometric data including facial recognition to improve your tagging experience. We collect voice data from your microphone with your permission.
    """,
    "TikTok": """
We collect information you provide when you create an account or use our Platform. We collect information when you create or share content, including biometric identifiers. We automatically collect certain device information, including device identifiers, IP address, location data, browsing history, and usage patterns. We collect and process information about your location, including your precise GPS location where you've granted permission. We may infer your location from your IP address, WiFi signals, Bluetooth, or other device signals. We use cookies, pixels, device fingerprinting, and other tracking technologies. We share your information with our parent company, affiliates, business partners, and third-party advertising and analytics providers. We may share your information with law enforcement. We retain your information for as long as your account is active or as needed to provide services, or as required by law. You cannot opt out of all data collection while using our services.
    """,
    "Signal": """
Signal is designed to never collect or store any sensitive information. We designed Signal to minimize data collection. Signal cannot decrypt or otherwise access the content of your messages or calls. Signal doesn't have access to your messages, calls, profile name, profile picture, or the people you message. Signal stores your phone number and the date you created an account and the date you last connected. We don't sell, rent, or monetize your personal data. You have the right to delete your account and data at any time. We use end-to-end encryption so that only you and the people you choose can see your messages. Signal is open source. We do not share your data with third parties for advertising purposes. We do not track your location.
    """,
    "Generic Scary Policy": """
By using our services, you consent to perpetual, irrevocable, worldwide collection of all data transmitted through our platform. We collect precise GPS location data, biometric identifiers including facial recognition and fingerprints, financial information, health data, and all device contacts and messages. We sell this information to data brokers, advertising partners, and business affiliates. We may retain your data indefinitely, including in backup systems that may never be deleted. Your data may be processed in countries without adequate privacy protections. We engage in cross-device tracking, behavioral profiling, and predictive modeling. We use tracking pixels, cookies, and fingerprinting technologies. You cannot opt out of data collection while using this service. We may share data with law enforcement without notice to you. We reserve the right to modify this policy at any time without notification. Third-party monetization partners have indefinite access to your behavioral profile.
    """,
}

if demo != "— select —":
    policy_text = DEMO_POLICIES[demo]
    st.info(f"Sample loaded: **{demo}** ({len(policy_text.split())} words)")

# ──────────────────────────────────────────────────────────────
# RUN ANALYSIS
# ──────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
run_col, _ = st.columns([2, 5])
with run_col:
    run_btn = st.button("⚡ SCAN POLICY", type="primary", use_container_width=True)

if run_btn:
    clean_text = (policy_text or "").strip()
    if not clean_text:
        st.error("Please paste or load a policy first.")
        st.stop()
    if len(clean_text.split()) < 30:
        st.error("Policy text is too short — please provide at least 30 words.")
        st.stop()
    if not api_key and demo == "— select —":
        st.warning("No Gemini API key provided — only Tier-1 TensorFlow analysis will run.")

    # ── TIER 1: TensorFlow ──
    with st.spinner("⚙ Tier-1 · Running TensorFlow tensor analysis…"):
        t0 = time.time()
        tf_metrics = tf_risk_score(clean_text)
        tf_time = time.time() - t0

    st.success(f"✓ TensorFlow analysis complete in {tf_time:.2f}s")

    gemini_result = None
    if api_key:
        with st.spinner("⚙ Tier-2 · Sending to Gemini 2.0 Flash…"):
            try:
                t1 = time.time()
                gemini_result = gemini_analysis(clean_text, tf_metrics, api_key)
                g_time = time.time() - t1
                st.success(f"✓ Gemini semantic analysis complete in {g_time:.2f}s")
            except json.JSONDecodeError as e:
                st.error(f"JSON parse error from Gemini: {e}")
            except Exception as e:
                st.error(f"Gemini API error: {e}")

    # ── Final score ──
    final_score = tf_metrics["overall"]
    if gemini_result:
        adj = gemini_result.get("gemini_score_adjustment", 0)
        final_score = min(100, max(0, final_score + adj))

    # ──────────────────────────────────────────────────────────
    # RESULTS DASHBOARD
    # ──────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="panel-title" style="font-size:.9rem">◈ SCAN RESULTS</div>', unsafe_allow_html=True)

    res_ring, res_bars, res_meta = st.columns([1.3, 2, 1.5], gap="large")

    with res_ring:
        render_score_ring(final_score)

    with res_bars:
        st.markdown('<div class="panel-title">CATEGORY HAZARD BREAKDOWN</div>', unsafe_allow_html=True)
        render_hazard_bars(tf_metrics["category_scores"])

    with res_meta:
        st.markdown('<div class="panel-title">DOCUMENT METRICS</div>', unsafe_allow_html=True)
        threat = gemini_result.get("threat_level", "—") if gemini_result else "—"
        tcolors = {"CRITICAL": "#ff2d55", "HIGH": "#ff6b35", "MEDIUM": "#f0c420",
                   "LOW": "#39ff14", "—": "#5a6a8a"}
        tc = tcolors.get(threat, "#5a6a8a")
        st.markdown(f"""
        <div class="kv-grid">
          <div class="kv-item">
            <div class="kv-key">Word Count</div>
            <div class="kv-val">{tf_metrics['word_count']:,}</div>
          </div>
          <div class="kv-item">
            <div class="kv-key">Threat Level</div>
            <div class="kv-val" style="color:{tc}">{threat}</div>
          </div>
          <div class="kv-item">
            <div class="kv-key">TF Score</div>
            <div class="kv-val">{tf_metrics['overall']:.1f}</div>
          </div>
          <div class="kv-item">
            <div class="kv-key">Final Score</div>
            <div class="kv-val" style="color:{score_color(final_score)}">{final_score:.1f}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        if gemini_result and gemini_result.get("data_points_exposed"):
            st.markdown('<div class="panel-title" style="margin-top:1rem">DATA POINTS EXPOSED</div>', unsafe_allow_html=True)
            chips = "".join(f'<span class="chip">{d}</span>' for d in gemini_result["data_points_exposed"][:10])
            st.markdown(f'<div class="chip-row">{chips}</div>', unsafe_allow_html=True)

    # ── TF Tensor log ──
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("▶ TIER-1 TENSOR PROCESSING LOG"):
        render_terminal_log(tf_metrics)
        raw_hits_flat = {CATEGORY_LABELS.get(k, k): v for k, v in tf_metrics["raw_hits"].items() if v}
        if raw_hits_flat:
            st.markdown('<div class="panel-title" style="margin-top:.8rem">MATCHED HAZARD PHRASES</div>', unsafe_allow_html=True)
            for cat, hits in raw_hits_flat.items():
                if hits:
                    chips = "".join(
                        f'<span class="chip" title="×{cnt}">{phrase} ×{cnt}</span>'
                        for phrase, cnt in hits.items()
                    )
                    st.markdown(f'<div style="margin-bottom:.4rem"><span style="color:var(--dim); font-size:.72rem; font-family:monospace">{cat}:</span><div class="chip-row">{chips}</div></div>', unsafe_allow_html=True)

    # ── Gemini results ──
    if gemini_result:
        st.markdown("---")

        # TL;DR
        if gemini_result.get("summary"):
            st.markdown(f"""
            <div class="panel">
              <div class="panel-title">◈ TL;DR — WHAT THIS POLICY ACTUALLY MEANS</div>
              <div style="font-size:1rem; line-height:1.7; color:var(--text)">{gemini_result['summary']}</div>
            </div>
            """, unsafe_allow_html=True)

        tab_findings, tab_plain, tab_actions = st.tabs([
            "⚠ FINDINGS", "📋 PLAIN ENGLISH", "🛡 RECOMMENDED ACTIONS"
        ])

        with tab_findings:
            if gemini_result.get("findings"):
                render_findings(gemini_result["findings"])
            else:
                st.info("No specific findings returned.")

        with tab_plain:
            if gemini_result.get("plain_english"):
                for bullet in gemini_result["plain_english"]:
                    is_bad = bullet.lower().startswith("they can") or "cannot" in bullet.lower()
                    icon = "🔴" if is_bad else "🟢"
                    st.markdown(f"""
                    <div class="finding-card {'critical' if is_bad else 'low'}" style="border-left-color:{'var(--danger)' if is_bad else 'var(--safe)'}">
                      {icon} {bullet}
                    </div>
                    """, unsafe_allow_html=True)

            if gemini_result.get("redeeming_clauses"):
                st.markdown('<div class="panel-title" style="margin-top:1rem">✅ REDEEMING CLAUSES FOUND</div>', unsafe_allow_html=True)
                for clause in gemini_result["redeeming_clauses"]:
                    st.markdown(f"""
                    <div class="finding-card low" style="border-left-color:var(--safe)">
                      ✅ {clause}
                    </div>
                    """, unsafe_allow_html=True)

        with tab_actions:
            if gemini_result.get("recommended_actions"):
                for i, action in enumerate(gemini_result["recommended_actions"], 1):
                    st.markdown(f"""
                    <div class="finding-card medium" style="border-left-color:var(--accent)">
                      <span class="mono" style="color:var(--accent)">ACTION {i:02d}</span><br>
                      {action}
                    </div>
                    """, unsafe_allow_html=True)

    # ── Raw JSON expander ──
    if gemini_result:
        with st.expander("▶ RAW GEMINI JSON PAYLOAD"):
            st.json(gemini_result)

# ──────────────────────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding:2rem 0 1rem; color:var(--dim); font-size:.72rem; font-family:'Share Tech Mono',monospace; border-top:1px solid var(--border); margin-top:3rem">
  CREEPINESS INDEX · HYBRID NLP PIPELINE · TENSORFLOW + GEMINI 2.0 FLASH · $0 INFRASTRUCTURE
</div>
""", unsafe_allow_html=True)
