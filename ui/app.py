"""
Smart Interview Analyzer — Streamlit UI
========================================
Run with: streamlit run ui/app.py

Tabs:
  🏠 Home           — project overview
  🎙 Live Analysis  — record from microphone and predict in real time
  📂 Upload & Analyze — analyse a saved .wav file
  📊 Session Report  — full feedback after a complete session
  🏋 Train Model    — trigger training pipeline from the UI
"""

import os
import sys
import time
import tempfile
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import librosa
import sounddevice as sd
import soundfile as sf

# Add project root to path so src.* imports work
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.inference.realtime_predictor import (
    EmotionPredictor, SAMPLE_RATE, WINDOW_SEC, EMOTION_CONTEXT
)

def record_audio(duration=WINDOW_SEC, sample_rate=SAMPLE_RATE):
    import sounddevice as sd
    frames = sd.rec(int(duration * sample_rate), samplerate=sample_rate,
                    channels=1, dtype="float32")
    sd.wait()
    return frames[:, 0]
from src.utils.feedback_generator import generate_feedback, format_report


# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Interview Analyzer",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;700&display=swap');

/* ── Global ── */
html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
}

/* ── Dark gradient background ── */
.stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    color: #e8e8f0;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: rgba(15, 12, 41, 0.95);
    border-right: 1px solid rgba(139, 92, 246, 0.3);
}

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(139,92,246,0.3);
    border-radius: 12px;
    padding: 12px 16px;
    backdrop-filter: blur(10px);
}
[data-testid="stMetricValue"] { font-size: 2rem !important; font-weight: 700; }
[data-testid="stMetricLabel"] { color: #a78bfa !important; font-size: 0.8rem; }

/* ── Hero header ── */
.hero {
    text-align: center;
    padding: 3rem 1rem 2rem;
}
.hero h1 {
    font-size: 3.5rem;
    font-weight: 700;
    background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.5rem;
}
.hero p { color: #94a3b8; font-size: 1.2rem; }

/* ── Emotion badge ── */
.emotion-badge {
    display: inline-block;
    padding: 0.5rem 1.5rem;
    border-radius: 2rem;
    font-size: 1.4rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    margin: 0.5rem 0;
}
.confident { background: linear-gradient(135deg,#059669,#10b981); color:#fff; }
.happy     { background: linear-gradient(135deg,#d97706,#f59e0b); color:#fff; }
.neutral   { background: linear-gradient(135deg,#4b5563,#6b7280); color:#fff; }
.nervous   { background: linear-gradient(135deg,#dc2626,#f87171); color:#fff; }
.stressed  { background: linear-gradient(135deg,#7c3aed,#a78bfa); color:#fff; }

/* ── Score ring container ── */
.score-container {
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 1rem;
}
.score-number {
    font-size: 4rem;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    background: linear-gradient(90deg, #a78bfa, #60a5fa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

/* ── Info / warning boxes ── */
.info-box {
    background: rgba(96,165,250,0.1);
    border: 1px solid rgba(96,165,250,0.3);
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin: 0.75rem 0;
}
.warn-box {
    background: rgba(251,191,36,0.1);
    border: 1px solid rgba(251,191,36,0.4);
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin: 0.75rem 0;
}
.success-box {
    background: rgba(52,211,153,0.1);
    border: 1px solid rgba(52,211,153,0.4);
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin: 0.75rem 0;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #7c3aed, #4f46e5);
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    padding: 0.6rem 2rem;
    transition: all 0.2s;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #8b5cf6, #6366f1);
    transform: translateY(-1px);
    box-shadow: 0 4px 20px rgba(139,92,246,0.4);
}

/* ── Tab styling ── */
.stTabs [role="tab"] {
    color: #94a3b8;
    font-weight: 500;
}
.stTabs [aria-selected="true"] {
    color: #a78bfa !important;
    border-bottom: 2px solid #a78bfa !important;
}

/* ── Progress bar ── */
.stProgress > div > div > div {
    background: linear-gradient(90deg, #7c3aed, #60a5fa);
}
</style>
""", unsafe_allow_html=True)


# ─── Session state initialisation ─────────────────────────────────────────────
defaults = {
    "predictor":       None,
    "model_loaded":    False,
    "session_results": [],
    "recording":       False,
    "last_result":     None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ─── Helper: load model (cached) ──────────────────────────────────────────────
@st.cache_resource
def load_predictor():
    return EmotionPredictor()


# ──────────────────────────────────────────────────────────────────────────────
# Visualisation helpers
# ──────────────────────────────────────────────────────────────────────────────

def emotion_color(emotion: str) -> str:
    colors = {
        "confident": "#10b981",
        "happy":     "#f59e0b",
        "neutral":   "#6b7280",
        "nervous":   "#f87171",
        "stressed":  "#a78bfa",
    }
    return colors.get(emotion, "#60a5fa")


def plot_emotion_gauge(score: float) -> go.Figure:
    """Speedometer gauge for final score."""
    color = "#10b981" if score >= 70 else "#f59e0b" if score >= 50 else "#f87171"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"suffix": "/100", "font": {"size": 36, "color": color,
                                            "family": "JetBrains Mono"}},
        gauge={
            "axis":      {"range": [0, 100], "tickcolor": "#94a3b8"},
            "bar":       {"color": color, "thickness": 0.3},
            "bgcolor":   "rgba(0,0,0,0)",
            "steps": [
                {"range": [0, 40],  "color": "rgba(248,113,113,0.15)"},
                {"range": [40, 70], "color": "rgba(251,191,36,0.15)"},
                {"range": [70, 100],"color": "rgba(52,211,153,0.15)"},
            ],
            "threshold": {
                "line": {"color": color, "width": 4},
                "thickness": 0.75,
                "value": score,
            },
        },
        title={"text": "Interview Score", "font": {"color": "#94a3b8", "size": 16}},
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e8e8f0",
        height=280,
        margin=dict(t=30, b=0, l=20, r=20),
    )
    return fig


def plot_emotion_distribution(dist: dict) -> go.Figure:
    """Horizontal bar chart of emotion distribution."""
    emotions = list(dist.keys())
    values   = list(dist.values())
    colors   = [emotion_color(e) for e in emotions]

    fig = go.Figure(go.Bar(
        x=values, y=emotions,
        orientation="h",
        marker_color=colors,
        marker_line_width=0,
        text=[f"{v}%" for v in values],
        textposition="outside",
        textfont={"color": "#e8e8f0"},
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e8e8f0",
        xaxis={"title": "% of session", "gridcolor": "rgba(255,255,255,0.1)",
               "range": [0, 110]},
        yaxis={"gridcolor": "rgba(0,0,0,0)"},
        height=260,
        margin=dict(t=10, b=30, l=10, r=60),
    )
    return fig


def plot_score_timeline(results: list[dict]) -> go.Figure:
    """Line chart of score over time."""
    times  = [r.get("timestamp", i * WINDOW_SEC) for i, r in enumerate(results)]
    scores = [r["score"] for r in results]
    colors = [emotion_color(r["emotion"]) for r in results]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=times, y=scores,
        mode="lines+markers",
        line=dict(color="#a78bfa", width=2),
        marker=dict(color=colors, size=10, line=dict(color="#fff", width=1)),
        name="Score",
    ))
    fig.add_hline(y=70, line_dash="dash", line_color="#10b981",
                  annotation_text="Good (70)", annotation_font_color="#10b981")
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e8e8f0",
        xaxis={"title": "Time (s)", "gridcolor": "rgba(255,255,255,0.05)"},
        yaxis={"title": "Score", "range": [0, 105],
               "gridcolor": "rgba(255,255,255,0.05)"},
        height=280,
        margin=dict(t=10, b=40, l=40, r=20),
    )
    return fig


def plot_radar(metrics: dict) -> go.Figure:
    """Radar chart of metric scores."""
    cats   = list(metrics.keys())
    vals   = list(metrics.values())
    cats  += [cats[0]]
    vals  += [vals[0]]

    fig = go.Figure(go.Scatterpolar(
        r=vals, theta=cats,
        fill="toself",
        fillcolor="rgba(139,92,246,0.2)",
        line=dict(color="#a78bfa", width=2),
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, range=[0, 100],
                            gridcolor="rgba(255,255,255,0.1)",
                            tickcolor="#94a3b8", tickfont={"color": "#94a3b8"}),
            angularaxis=dict(gridcolor="rgba(255,255,255,0.1)",
                             tickcolor="#94a3b8", tickfont={"color": "#e8e8f0"}),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#e8e8f0",
        height=300,
        margin=dict(t=20, b=20, l=40, r=40),
        showlegend=False,
    )
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 1rem 0;">
        <div style="font-size:3rem">🎙️</div>
        <div style="font-size:1.3rem; font-weight:700; color:#a78bfa">
            Interview Analyzer
        </div>
        <div style="color:#64748b; font-size:0.8rem">AI-Powered Voice Analysis</div>
    </div>
    <hr style="border-color:rgba(139,92,246,0.3)">
    """, unsafe_allow_html=True)

    # Model status
    model_path = os.path.join(ROOT, "models", "emotion_model.h5")
    if os.path.exists(model_path):
        st.markdown('<div class="success-box">✅ Model loaded & ready</div>',
                    unsafe_allow_html=True)
        if not st.session_state.model_loaded:
            try:
                st.session_state.predictor    = load_predictor()
                st.session_state.model_loaded = True
            except Exception as e:
                st.error(f"Load error: {e}")
    else:
        st.markdown('<div class="warn-box">⚠️ Model not trained yet<br>'
                    'Go to <b>Train Model</b> tab first</div>',
                    unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**Session Stats**")
    n = len(st.session_state.session_results)
    if n:
        avg = np.mean([r["score"] for r in st.session_state.session_results])
        st.metric("Segments Analysed", n)
        st.metric("Average Score", f"{avg:.1f}/100")
    else:
        st.info("No session data yet.")

    if st.button("🗑️ Clear Session"):
        st.session_state.session_results = []
        st.session_state.last_result = None
        st.rerun()

    st.markdown("---")
    st.markdown("""
    <div style="color:#64748b; font-size:0.75rem">
    <b>Emotions detected:</b><br>
    💪 Confident &nbsp;|&nbsp; 😊 Happy<br>
    😐 Neutral &nbsp;|&nbsp; 😰 Nervous<br>
    😤 Stressed
    </div>
    """, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# Tabs
# ──────────────────────────────────────────────────────────────────────────────

tab_home, tab_live, tab_upload, tab_report, tab_train = st.tabs([
    "🏠 Home", "🎙️ Live Analysis", "📂 Upload & Analyze",
    "📊 Session Report", "🏋️ Train Model"
])


# ═══════════════════════════════════════════════════════════════════════════════
# HOME TAB
# ═══════════════════════════════════════════════════════════════════════════════

with tab_home:
    st.markdown("""
    <div class="hero">
        <h1>🎙️ Smart Interview Analyzer</h1>
        <p>AI-powered voice analysis for mock interview coaching</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div style="text-align:center;background:rgba(255,255,255,0.04);
                    border-radius:12px;padding:1.5rem;border:1px solid rgba(139,92,246,0.3)">
            <div style="font-size:2.5rem">🧠</div>
            <div style="font-weight:600;color:#a78bfa;margin-top:0.5rem">Deep Learning</div>
            <div style="color:#94a3b8;font-size:0.85rem;margin-top:0.3rem">
                CNN + LSTM hybrid model trained on RAVDESS & CREMA-D
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div style="text-align:center;background:rgba(255,255,255,0.04);
                    border-radius:12px;padding:1.5rem;border:1px solid rgba(96,165,250,0.3)">
            <div style="font-size:2.5rem">⚡</div>
            <div style="font-weight:600;color:#60a5fa;margin-top:0.5rem">Real-Time</div>
            <div style="color:#94a3b8;font-size:0.85rem;margin-top:0.3rem">
                3-second rolling window analysis from your microphone
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div style="text-align:center;background:rgba(255,255,255,0.04);
                    border-radius:12px;padding:1.5rem;border:1px solid rgba(52,211,153,0.3)">
            <div style="font-size:2.5rem">📊</div>
            <div style="font-weight:600;color:#34d399;margin-top:0.5rem">Full Metrics</div>
            <div style="color:#94a3b8;font-size:0.85rem;margin-top:0.3rem">
                Confidence, pace, tone consistency, and clarity scored
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div style="text-align:center;background:rgba(255,255,255,0.04);
                    border-radius:12px;padding:1.5rem;border:1px solid rgba(251,191,36,0.3)">
            <div style="font-size:2.5rem">💡</div>
            <div style="font-weight:600;color:#f59e0b;margin-top:0.5rem">Coaching</div>
            <div style="color:#94a3b8;font-size:0.85rem;margin-top:0.3rem">
                Personalised improvement suggestions after every session
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🗺️ How to use this app")

    steps = [
        ("1️⃣ Download Datasets", "RAVDESS + CREMA-D (links in README)", "#a78bfa"),
        ("2️⃣ Preprocess", "Run `feature_extractor.py` or use the **Train Model** tab", "#60a5fa"),
        ("3️⃣ Train", "CNN-LSTM model trains in ~10 min on GPU", "#34d399"),
        ("4️⃣ Analyse", "Record live or upload a .wav file", "#f59e0b"),
        ("5️⃣ Review", "See score, emotion breakdown, and suggestions", "#f87171"),
    ]
    for title, desc, color in steps:
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:1rem;padding:0.75rem 0;
                    border-bottom:1px solid rgba(255,255,255,0.06)">
            <div style="font-size:1.5rem;min-width:2.5rem">{title.split()[0]}</div>
            <div>
                <div style="font-weight:600;color:{color}">{title[2:]}</div>
                <div style="color:#94a3b8;font-size:0.9rem">{desc}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🎭 Audio Feature Extraction Pipeline")
    st.markdown("""
    Each 3-second audio window produces a **222-dimensional feature vector**:

    | Feature | Dimensions | What it captures |
    |---------|-----------|-----------------|
    | MFCC mean + std | 40 × 2 = 80 | Timbre, spectral envelope |
    | Chroma mean | 12 | Pitch class distribution |
    | Mel Spectrogram mean | 128 | Perceptual frequency content |
    | Zero Crossing Rate | 1 | Noisiness / nervousness |
    | RMS Energy | 1 | Volume / confidence |
    | **Total** | **222** | |
    """)


# ═══════════════════════════════════════════════════════════════════════════════
# LIVE ANALYSIS TAB
# ═══════════════════════════════════════════════════════════════════════════════

with tab_live:
    st.markdown("## 🎙️ Live Microphone Analysis")

    if not st.session_state.model_loaded:
        st.warning("⚠️ Please train the model first (Train Model tab).")
    else:
        st.markdown("""
        <div class="info-box">
        🎤 Click <b>Record One Segment</b> to capture 3 seconds of your voice,
        or use <b>Start Session</b> to record a full mock interview.
        </div>
        """, unsafe_allow_html=True)

        col_ctrl1, col_ctrl2 = st.columns([1, 2])
        with col_ctrl1:
            session_duration = st.slider(
                "Session Duration (seconds)", 15, 120, 60, step=15)

        c1, c2 = st.columns(2)
        single_btn  = c1.button("🔴 Record One Segment (3s)")
        session_btn = c2.button(f"▶️ Start {session_duration}s Session")

        # ── Single segment ──
        if single_btn:
            with st.spinner("🎙️ Recording 3 seconds …"):
                audio = record_audio(duration=WINDOW_SEC)
            with st.spinner("🧠 Predicting emotion …"):
                result = st.session_state.predictor.predict_from_array(audio)
            st.session_state.last_result = result
            st.session_state.session_results.append(result)

        # ── Full session ──
        if session_btn:
            st.session_state.session_results = []
            n_windows = session_duration // WINDOW_SEC
            progress  = st.progress(0, text="Starting session …")
            live_col1, live_col2 = st.columns([1, 2])

            for i in range(n_windows):
                progress.progress((i + 1) / n_windows,
                                  text=f"Recording segment {i+1}/{n_windows} …")
                audio  = record_audio(duration=WINDOW_SEC)
                result = st.session_state.predictor.predict_from_array(audio)
                result["timestamp"] = i * WINDOW_SEC
                st.session_state.session_results.append(result)
                st.session_state.last_result = result

                with live_col1:
                    emo = result["emotion"]
                    st.markdown(f"""
                    <div style="text-align:center;padding:1rem">
                        <div style="font-size:4rem">{result['emotion_icon']}</div>
                        <div class="emotion-badge {emo}">{emo.upper()}</div>
                        <div style="color:#94a3b8;font-size:0.85rem;margin-top:0.5rem">
                            {result['emotion_meaning']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                with live_col2:
                    st.plotly_chart(
                        plot_score_timeline(st.session_state.session_results),
                        use_container_width=True, key=f"timeline_{i}")

            progress.progress(1.0, text="✅ Session complete!")
            st.success("Session recorded! Go to **Session Report** tab for full analysis.")

        # ── Show last result ──
        if st.session_state.last_result:
            r = st.session_state.last_result
            st.markdown("---")
            st.markdown("### Last Prediction")

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Emotion",         f"{r['emotion_icon']} {r['emotion'].capitalize()}")
            m2.metric("Score",           f"{r['score']:.1f}/100")
            m3.metric("Speaking Rate",   f"{r['metrics']['speaking_rate']:.1f} syl/s")
            m4.metric("Tone Consistency",f"{r['metrics']['tone_consistency']*100:.0f}%")

            # Probability bars
            st.markdown("**Emotion Probabilities**")
            for emo, prob in sorted(r["probabilities"].items(),
                                    key=lambda x: -x[1]):
                col_e, col_b = st.columns([1, 5])
                col_e.markdown(f"`{emo}`")
                col_b.progress(prob, text=f"{prob*100:.1f}%")


# ═══════════════════════════════════════════════════════════════════════════════
# UPLOAD & ANALYZE TAB
# ═══════════════════════════════════════════════════════════════════════════════

with tab_upload:
    st.markdown("## 📂 Upload & Analyze Audio File")

    if not st.session_state.model_loaded:
        st.warning("⚠️ Train the model first.")
    else:
        uploaded = st.file_uploader(
            "Upload a .wav file (interview recording)",
            type=["wav"],
            help="Mono or stereo WAV, any sample rate — we resample automatically.",
        )
        if uploaded:
            # Save to temp file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(uploaded.read())
                tmp_path = tmp.name

            # Load and display waveform
            y, sr = librosa.load(tmp_path, sr=SAMPLE_RATE)
            duration = len(y) / SAMPLE_RATE

            st.markdown(f"**File:** `{uploaded.name}` | "
                        f"**Duration:** {duration:.1f}s | "
                        f"**Sample Rate:** {SAMPLE_RATE} Hz")

            # Plot waveform
            times = np.linspace(0, duration, len(y))
            fig_wave = go.Figure(go.Scatter(
                x=times, y=y, mode="lines",
                line=dict(color="#a78bfa", width=0.8),
                name="Waveform",
            ))
            fig_wave.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#e8e8f0",
                xaxis=dict(title="Time (s)", gridcolor="rgba(255,255,255,0.05)"),
                yaxis=dict(title="Amplitude", gridcolor="rgba(255,255,255,0.05)"),
                height=180,
                margin=dict(t=10, b=30, l=40, r=10),
                showlegend=False,
            )
            st.plotly_chart(fig_wave, use_container_width=True, key="chart_1")

            if st.button("🧠 Analyse Full File"):
                # Segment into WINDOW_SEC chunks
                results = []
                n_chunks = int(duration // WINDOW_SEC)
                if n_chunks == 0:
                    n_chunks = 1

                prog = st.progress(0)
                for i in range(n_chunks):
                    start = int(i * WINDOW_SEC * SAMPLE_RATE)
                    end   = int(start + WINDOW_SEC * SAMPLE_RATE)
                    chunk = y[start:end]
                    if len(chunk) < WINDOW_SEC * SAMPLE_RATE * 0.5:
                        continue
                    result = st.session_state.predictor.predict_from_array(chunk)
                    result["timestamp"] = i * WINDOW_SEC
                    results.append(result)
                    prog.progress((i + 1) / n_chunks)

                st.session_state.session_results = results
                st.success(f"Analysed {len(results)} segments. "
                           "View full report in **Session Report** tab.")

                # Quick summary
                fb = generate_feedback(results)
                col_s1, col_s2 = st.columns(2)
                with col_s1:
                    st.plotly_chart(
                        plot_emotion_gauge(fb["final_score"]),
                        use_container_width=True, key="chart_2")
                with col_s2:
                    st.plotly_chart(
                        plot_emotion_distribution(fb["emotion_distribution"]),
                        use_container_width=True, key="chart_3")

            os.unlink(tmp_path)


# ═══════════════════════════════════════════════════════════════════════════════
# SESSION REPORT TAB
# ═══════════════════════════════════════════════════════════════════════════════

with tab_report:
    st.markdown("## 📊 Session Performance Report")

    if not st.session_state.session_results:
        st.info("No session data yet. Record or upload audio to generate a report.")
    else:
        fb = generate_feedback(st.session_state.session_results)

        # ── Score + Grade ──
        grade_color = {
            "A": "#10b981", "B": "#34d399",
            "C": "#f59e0b", "D": "#f87171", "F": "#ef4444",
        }.get(fb["grade"], "#a78bfa")

        col_g1, col_g2, col_g3 = st.columns([2, 1, 2])
        with col_g1:
            st.plotly_chart(plot_emotion_gauge(fb["final_score"]),
                            use_container_width=True, key="chart_4")
        with col_g2:
            st.markdown(f"""
            <div style="text-align:center;padding:2rem 0">
                <div style="font-size:5rem;font-weight:700;color:{grade_color}">
                    {fb['grade']}
                </div>
                <div style="color:#94a3b8">Grade</div>
                <br>
                <div style="font-size:1.4rem">
                    {EMOTION_CONTEXT.get(fb['dominant_emotion'],{}).get('icon','❓')}
                </div>
                <div style="color:#a78bfa;font-weight:600">
                    {fb['dominant_emotion'].capitalize()}
                </div>
                <div style="color:#64748b;font-size:0.8rem">dominant emotion</div>
            </div>
            """, unsafe_allow_html=True)
        with col_g3:
            st.plotly_chart(
                plot_radar(fb["score_breakdown"]),
                use_container_width=True, key="chart_5")

        st.markdown("---")

        # ── Metrics ──
        st.markdown("### 📈 Detailed Metrics")
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Speaking Rate",  f"{fb['avg_metrics']['speaking_rate']} syl/s",
                   help="Ideal: 3–4 syl/s")
        mc2.metric("Tone Consistency", f"{fb['avg_metrics']['tone_consistency']}%")
        mc3.metric("Clarity",          f"{fb['avg_metrics']['clarity']}%")
        mc4.metric("Pace Score",       f"{fb['avg_metrics']['pace_score']}%")

        # ── Timeline ──
        if len(st.session_state.session_results) > 1:
            st.markdown("### 📉 Score Timeline")
            st.plotly_chart(
                plot_score_timeline(st.session_state.session_results),
                use_container_width=True, key="chart_6")

        # ── Emotion distribution ──
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            st.markdown("### 🎭 Emotion Distribution")
            st.plotly_chart(
                plot_emotion_distribution(fb["emotion_distribution"]),
                use_container_width=True, key="chart_7")
        with col_d2:
            st.markdown("### 📝 Summary")
            st.markdown(f"""
            <div style="background:rgba(255,255,255,0.04);border-radius:12px;
                        padding:1.5rem;border:1px solid rgba(139,92,246,0.3);
                        color:#e8e8f0;line-height:1.7">
                {fb['summary']}
            </div>
            """, unsafe_allow_html=True)

        # ── Strengths & Improvements ──
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            st.markdown("### ✅ Strengths")
            for s in fb["strengths"]:
                st.markdown(f"""
                <div class="success-box">✔ {s}</div>
                """, unsafe_allow_html=True)
        with col_f2:
            st.markdown("### ⚠️ Areas to Improve")
            if fb["improvements"]:
                for imp in fb["improvements"]:
                    st.markdown(f'<div class="warn-box">⚡ {imp}</div>',
                                unsafe_allow_html=True)
            else:
                st.markdown('<div class="success-box">No major issues detected! 🎉</div>',
                            unsafe_allow_html=True)

        # ── Suggestions ──
        st.markdown("### 💡 Coaching Suggestions")
        for i, sug in enumerate(fb["suggestions"]):
            st.markdown(f"""
            <div class="info-box">
                <b>{i+1}.</b> {sug}
            </div>
            """, unsafe_allow_html=True)

        # ── Download text report ──
        report_text = format_report(fb)
        st.download_button(
            "📥 Download Report (.txt)",
            data=report_text,
            file_name="interview_report.txt",
            mime="text/plain",
        )


# ═══════════════════════════════════════════════════════════════════════════════
# TRAIN MODEL TAB
# ═══════════════════════════════════════════════════════════════════════════════

with tab_train:
    st.markdown("## 🏋️ Model Training")

    st.markdown("""
    <div class="info-box">
    <b>Steps to train the model:</b><br>
    1. Download <a href="https://zenodo.org/record/1188976" target="_blank">RAVDESS</a>
       and extract to <code>data/raw/RAVDESS/</code><br>
    2. Download <a href="https://github.com/CheyneyComputerScience/CREMA-D" target="_blank">CREMA-D</a>
       and extract <code>.wav</code> files to <code>data/raw/CREMA_D/</code><br>
    3. Click <b>Preprocess Datasets</b> → then <b>Train Model</b>
    </div>
    """, unsafe_allow_html=True)

    # Dataset status
    ravdess_ok = os.path.isdir(os.path.join(ROOT, "data/raw/RAVDESS")) and \
                 len([f for f in os.listdir(os.path.join(ROOT, "data/raw/RAVDESS"))
                      if f.endswith(".wav")]) > 0 \
                 if os.path.isdir(os.path.join(ROOT, "data/raw/RAVDESS")) else False

    cremad_ok  = os.path.isdir(os.path.join(ROOT, "data/raw/CREMA_D")) and \
                 len([f for f in os.listdir(os.path.join(ROOT, "data/raw/CREMA_D"))
                      if f.endswith(".wav")]) > 0 \
                 if os.path.isdir(os.path.join(ROOT, "data/raw/CREMA_D")) else False

    processed_ok = os.path.exists(os.path.join(ROOT, "data/processed/X.npy"))
    model_ok     = os.path.exists(os.path.join(ROOT, "models/emotion_model.h5"))

    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    col_s1.metric("RAVDESS",   "✅ Found" if ravdess_ok  else "❌ Missing")
    col_s2.metric("CREMA-D",   "✅ Found" if cremad_ok   else "❌ Missing")
    col_s3.metric("Processed", "✅ Ready" if processed_ok else "❌ Missing")
    col_s4.metric("Model",     "✅ Trained" if model_ok  else "❌ Missing")

    st.markdown("---")
    st.markdown("### Run from terminal (recommended)")
    st.code("""# Step 1 – Preprocess
python src/preprocessing/feature_extractor.py

# Step 2 – Train (uses GPU if available)
python src/training/train_model.py

# Step 3 – Launch UI
streamlit run ui/app.py
""", language="bash")

    st.markdown("### Or run directly from here")

    col_b1, col_b2 = st.columns(2)
    if col_b1.button("⚙️ Run Preprocessing"):
        with st.spinner("Extracting features — this may take several minutes …"):
            import subprocess
            res = subprocess.run(
                [sys.executable,
                 os.path.join(ROOT, "src/preprocessing/feature_extractor.py")],
                capture_output=True, text=True,
            )
        if res.returncode == 0:
            st.success("✅ Preprocessing complete!")
        else:
            st.error(f"Error:\n{res.stderr[-2000:]}")

    if col_b2.button("🚀 Train Model"):
        if not processed_ok:
            st.warning("Run preprocessing first.")
        else:
            with st.spinner("Training model — this may take 5–20 minutes …"):
                import subprocess
                res = subprocess.run(
                    [sys.executable,
                     os.path.join(ROOT, "src/training/train_model.py")],
                    capture_output=True, text=True,
                )
            if res.returncode == 0:
                st.success("✅ Model trained and saved to `models/emotion_model.h5`!")
                st.session_state.model_loaded = False  # force reload
                st.rerun()
            else:
                st.error(f"Training error:\n{res.stderr[-2000:]}")

    # Show training curves if available
    curves_path = os.path.join(ROOT, "models/training_curves.png")
    if os.path.exists(curves_path):
        st.markdown("### 📈 Training Curves")
        st.image(curves_path, use_column_width=True)

    st.markdown("---")
    st.markdown("### 🏗️ Model Architecture")
    st.markdown("""
    ```
    Input: (222, 1)          ← 222-dim feature vector reshaped
         │
    Conv1D(64, k=5) + BN + MaxPool + Dropout
         │
    Conv1D(128, k=3) + BN + MaxPool + Dropout
         │
    Conv1D(256, k=3) + BN + Dropout
         │
    LSTM(128)
         │
    Dense(128) → Dense(64) → Dense(5, softmax)
         │
    Output: 5 emotion probabilities
    ```
    """)