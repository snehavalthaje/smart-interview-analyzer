"""
Feedback Generator
===================
Converts raw prediction results into human-readable interview feedback
and improvement suggestions.

Imported by both the CLI predictor and the Streamlit UI.
"""

from __future__ import annotations
import numpy as np
from collections import Counter


# ──────────────────────────────────────────────────────────────────────────────
# Thresholds
# ──────────────────────────────────────────────────────────────────────────────

PACE_SLOW     = 2.0   # syllables/sec below this → too slow
PACE_FAST     = 5.0   # syllables/sec above this → too fast
TONE_LOW      = 0.40  # tone consistency below → monotone / erratic
CLARITY_LOW   = 0.45  # clarity below → mumbling / unclear


# ──────────────────────────────────────────────────────────────────────────────
# Per-emotion coaching tips
# ──────────────────────────────────────────────────────────────────────────────

EMOTION_TIPS: dict[str, list[str]] = {
    "confident": [
        "Great confidence! Maintain this energy throughout.",
        "Keep your tone steady — you're doing well.",
        "Consider adding pauses for emphasis to appear even more authoritative.",
    ],
    "happy": [
        "Your positivity is infectious — ideal for cultural-fit interviews.",
        "Balance enthusiasm with precision so technical answers stay clear.",
        "Smile through your words — it's already working!",
    ],
    "neutral": [
        "You're calm — now try injecting more enthusiasm.",
        "Vary your pitch slightly to keep the interviewer engaged.",
        "Confidence comes from conviction; emphasise key points.",
    ],
    "nervous": [
        "Take a slow, deep breath before answering.",
        "Pause briefly instead of rushing — silence is okay.",
        "Practice your answers aloud 5× before the real interview.",
        "Remember: interviewers want you to succeed.",
    ],
    "stressed": [
        "Identify what's triggering your stress and address it head-on.",
        "Lower your voice slightly — it signals control.",
        "Use the STAR method (Situation, Task, Action, Result) to stay structured.",
        "Hydrate and take short breaks between practice rounds.",
    ],
}


# ──────────────────────────────────────────────────────────────────────────────
# Main feedback function
# ──────────────────────────────────────────────────────────────────────────────

def generate_feedback(session_results: list[dict]) -> dict:
    """
    Aggregate a list of per-window prediction dicts into a full session report.

    Parameters
    ----------
    session_results : list of dicts returned by EmotionPredictor.predict_from_array()

    Returns
    -------
    dict with keys:
        final_score, grade, dominant_emotion, emotion_distribution,
        avg_metrics, strengths, improvements, suggestions, summary
    """
    if not session_results:
        return {"error": "No data to analyse."}

    scores       = [r["score"]   for r in session_results]
    emotions     = [r["emotion"] for r in session_results]

    # ── Aggregate metrics ──
    avg_pace  = np.mean([r["metrics"]["speaking_rate"]    for r in session_results])
    avg_tone  = np.mean([r["metrics"]["tone_consistency"] for r in session_results])
    avg_clar  = np.mean([r["metrics"]["clarity"]          for r in session_results])
    avg_pace_score = np.mean([r["metrics"]["pace_score"]  for r in session_results])

    final_score      = float(np.mean(scores))
    dominant_emotion = Counter(emotions).most_common(1)[0][0]
    emotion_dist     = {k: round(v / len(emotions) * 100, 1)
                        for k, v in Counter(emotions).items()}

    # ── Grade ──
    grade = (
        "A" if final_score >= 85 else
        "B" if final_score >= 70 else
        "C" if final_score >= 55 else
        "D" if final_score >= 40 else "F"
    )

    # ── Strengths ──
    strengths = []
    if dominant_emotion in ("confident", "happy"):
        strengths.append("Positive emotional presence throughout the session.")
    if avg_tone >= 0.65:
        strengths.append("Consistent and steady vocal tone.")
    if PACE_SLOW <= avg_pace <= PACE_FAST:
        strengths.append("Speaking pace is well-calibrated for interviews.")
    if avg_clar >= 0.60:
        strengths.append("Speech clarity is high — easy to understand.")
    if not strengths:
        strengths.append("You completed the full session — consistency is a strength!")

    # ── Improvement areas ──
    improvements = []
    if avg_pace < PACE_SLOW:
        improvements.append(
            f"Speaking too slowly ({avg_pace:.1f} syl/s). Try to speak at "
            f"3–4 syllables per second to sound more dynamic."
        )
    elif avg_pace > PACE_FAST:
        improvements.append(
            f"Speaking too fast ({avg_pace:.1f} syl/s). Slow down — "
            f"aim for 3–4 syllables/second so every word lands."
        )

    if avg_tone < TONE_LOW:
        improvements.append(
            "Tone is inconsistent — voice volume varies a lot. "
            "Work on breath support and maintaining steady energy."
        )

    if avg_clar < CLARITY_LOW:
        improvements.append(
            "Speech clarity needs work. Practise enunciation exercises "
            "and avoid mumbling at the end of sentences."
        )

    if dominant_emotion in ("nervous", "stressed"):
        improvements.append(
            f"Dominant emotion is '{dominant_emotion}'. "
            "Try mock interviews with a friend to build comfort."
        )

    # ── Specific suggestions from emotion tips ──
    suggestions = EMOTION_TIPS.get(dominant_emotion, [])

    # ── One-paragraph summary ──
    summary = (
        f"You scored {final_score:.1f}/100 (Grade {grade}). "
        f"Your dominant emotion was '{dominant_emotion}', "
        f"average speaking pace was {avg_pace:.1f} syllables/second, "
        f"tone consistency was {avg_tone * 100:.0f}%, "
        f"and speech clarity was {avg_clar * 100:.0f}%. "
        + (
            "Overall a strong performance — keep refining those details."
            if final_score >= 70
            else "With focused practice on the areas above, you'll see rapid improvement."
        )
    )

    return {
        "final_score":        round(final_score, 1),
        "grade":              grade,
        "dominant_emotion":   dominant_emotion,
        "emotion_distribution": emotion_dist,
        "avg_metrics": {
            "speaking_rate":    round(avg_pace, 2),
            "tone_consistency": round(avg_tone * 100, 1),
            "clarity":          round(avg_clar * 100, 1),
            "pace_score":       round(avg_pace_score * 100, 1),
        },
        "strengths":    strengths,
        "improvements": improvements,
        "suggestions":  suggestions,
        "summary":      summary,
        "score_breakdown": {
            "Confidence":     round(
                np.mean([r["probabilities"].get(r["emotion"], 0)
                         for r in session_results]) * 100, 1),
            "Tone":           round(avg_tone * 100, 1),
            "Speaking Pace":  round(avg_pace_score * 100, 1),
            "Clarity":        round(avg_clar * 100, 1),
        },
    }


def format_report(feedback: dict) -> str:
    """Return a plain-text formatted report string."""
    if "error" in feedback:
        return f"Error: {feedback['error']}"

    lines = [
        "=" * 55,
        f"  INTERVIEW PERFORMANCE REPORT",
        "=" * 55,
        f"  Final Score      : {feedback['final_score']}/100  (Grade {feedback['grade']})",
        f"  Dominant Emotion : {feedback['dominant_emotion'].upper()}",
        "",
        "  Score Breakdown:",
    ]
    for k, v in feedback["score_breakdown"].items():
        bar = "█" * int(v // 10) + "░" * (10 - int(v // 10))
        lines.append(f"    {k:<18} [{bar}] {v}%")

    lines += ["", "  Emotion Distribution:"]
    for emo, pct in sorted(feedback["emotion_distribution"].items(),
                           key=lambda x: -x[1]):
        lines.append(f"    {emo:<12} {pct}%")

    lines += ["", "  ✅ Strengths:"]
    for s in feedback["strengths"]:
        lines.append(f"    • {s}")

    if feedback["improvements"]:
        lines += ["", "  ⚠  Areas to Improve:"]
        for i in feedback["improvements"]:
            lines.append(f"    • {i}")

    lines += ["", "  💡 Suggestions:"]
    for s in feedback["suggestions"]:
        lines.append(f"    • {s}")

    lines += ["", "  Summary:", f"  {feedback['summary']}", "=" * 55]
    return "\n".join(lines)
