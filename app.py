import json
import os
import uuid
import threading

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from app.agent import app as adk_app
from app.agent import compile_local_fallback_explanation
from app.diagrams import (
    get_longitudinal_wave_diagram,
    get_shear_wave_diagram,
    get_through_transmission_diagram,
)
from app.generate_signal import METALS_DATABASE, generate_signal_data
from app.signal_utils import find_signal_peaks, load_signal

# Save the true original server key once when the Python process starts
ORIGINAL_SERVER_KEY = os.environ.get("GEMINI_API_KEY", "")

# Global execution lock to ensure thread-safe API Key environment variable usage
genai_lock = threading.Lock()

def run_agent_with_isolation(runner_obj, prompt_text, session_id):
    from app.agent import research_assistant_agent, principal_investigator_agent
    
    with genai_lock:
        # Clear cached client properties on ADK Gemini instances to force rebuild on new session key
        for agent in [research_assistant_agent, principal_investigator_agent]:
            model_inst = agent.model
            if "api_client" in model_inst.__dict__:
                del model_inst.__dict__["api_client"]
            if "_live_api_client" in model_inst.__dict__:
                del model_inst.__dict__["_live_api_client"]
                
        # Cache old key and determine target key for this session
        old_env_key = os.environ.get("GEMINI_API_KEY")
        target_key = None
        
        if st.session_state.get("api_key_activated") and st.session_state.get("visitor_key_value"):
            target_key = st.session_state["visitor_key_value"]
        elif ORIGINAL_SERVER_KEY:
            target_key = ORIGINAL_SERVER_KEY
            
        if target_key:
            os.environ["GEMINI_API_KEY"] = target_key
        else:
            if "GEMINI_API_KEY" in os.environ:
                del os.environ["GEMINI_API_KEY"]
                
        try:
            content = types.Content(parts=[types.Part.from_text(text=prompt_text)])
            events = list(
                runner_obj.run(
                    user_id="streamlit-user",
                    session_id=session_id,
                    new_message=content,
                )
            )
            return events
        finally:
            # Restore environment variables
            if old_env_key:
                os.environ["GEMINI_API_KEY"] = old_env_key
            else:
                if "GEMINI_API_KEY" in os.environ:
                    del os.environ["GEMINI_API_KEY"]
                    
            # Clear cache again so no keys leak in model singleton memory
            for agent in [research_assistant_agent, principal_investigator_agent]:
                model_inst = agent.model
                if "api_client" in model_inst.__dict__:
                    del model_inst.__dict__["api_client"]
                if "_live_api_client" in model_inst.__dict__:
                    del model_inst.__dict__["_live_api_client"]

# Load literature values
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
lit_path = os.path.join(BASE_DIR, "references", "literature_values.json")
with open(lit_path) as f:
    literature_db = json.load(f)

# Set page config
st.set_page_config(
    page_title="WaveTutor | Ultrasonic NDT Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Load text files dynamically
def load_text_content(filename: str) -> str:
    filepath = os.path.join(BASE_DIR, "references", filename)
    if os.path.exists(filepath):
        with open(filepath, encoding="utf-8") as f:
            return f.read()
    return f"Error: Content file '{filename}' not found."


# Modal popup dialog to enlarge diagrams
@st.dialog("🔍 Enlarge Diagram", width="large")
def show_large_diagram(svg_content: str):
    html_code = f"""
    <body style="margin: 0; background: transparent; display: flex; justify-content: center; align-items: center;">
        {svg_content.replace('height="150"', 'height="450"').replace('height="160"', 'height="450"')}
    </body>
    """
    components.html(html_code, height=480)


# Helper to render SVG diagrams inside a transparent, theme-responsive iframe
def render_svg(svg_content: str, key: str):
    html_code = f"""
    <body style="margin: 0; background: transparent; display: flex; justify-content: center; align-items: center; overflow: hidden;">
        <style>
            svg {{
                color: #475569;
            }}
            @media (prefers-color-scheme: dark) {{
                svg {{
                    color: #cbd5e1;
                }}
            }}
        </style>
        {svg_content}
    </body>
    """
    components.html(html_code, height=160)
    if st.button("🔍 Zoom Diagram", key=f"zoom_{key}"):
        show_large_diagram(svg_content)


# Tukey window helper function
def generate_tukey_burst(t, t_center, duration, alpha=0.5, f0=0.5):
    w = np.zeros_like(t)
    t_rel = t - (t_center - duration / 2)
    idx = (t_rel >= 0) & (t_rel <= duration)
    tr = t_rel[idx]
    w_val = np.zeros_like(tr)
    t_rise = alpha * duration / 2
    r_idx = tr < t_rise
    w_val[r_idx] = 0.5 * (1 + np.cos(np.pi * (tr[r_idx] - t_rise) / t_rise))
    c_idx = (tr >= t_rise) & (tr <= duration - t_rise)
    w_val[c_idx] = 1.0
    f_idx = tr > duration - t_rise
    w_val[f_idx] = 0.5 * (
        1 + np.cos(np.pi * (tr[f_idx] - (duration - t_rise)) / t_rise)
    )
    w[idx] = w_val
    return w * np.cos(2 * np.pi * f0 * (t - t_center)), w


# Dynamic Tukey Peak Matching plot
def plot_peak_matching_diagram():
    t = np.linspace(0, 24, 1000)
    excit, _ = generate_tukey_burst(t, 5.0, 8.0, alpha=0.5, f0=0.5)
    received, _ = generate_tukey_burst(t, 14.0, 8.0, alpha=0.5, f0=0.5)
    received *= 0.8

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5.5))
    for ax in (ax1, ax2):
        ax.set_facecolor("#0F172A")
        ax.grid(True, color="#334155", linestyle=":", linewidth=0.8)
        ax.tick_params(colors="#94A3B8")
        ax.xaxis.label.set_color("#94A3B8")
        ax.yaxis.label.set_color("#94A3B8")
        ax.title.set_color("#F1F5F9")
    fig.patch.set_facecolor("#1E293B")

    ax1.plot(t, excit, color="#38BDF8", linewidth=1.8, label="Excitation Signal")
    ax1.plot(t, np.zeros_like(t), color="#94A3B8", alpha=0.3, linestyle="--")
    ax1.plot(5.0, 1.0, "ro", markersize=8, label="Selected Peak")
    ax1.annotate(
        "Selected Peak\n(t = 5.0 µs)",
        xy=(5.0, 1.0),
        xytext=(7.5, 0.7),
        arrowprops={
            "facecolor": "#ef4444",
            "shrink": 0.08,
            "width": 1.5,
            "headwidth": 6,
        },
        color="#ef4444",
        fontweight="bold",
        fontsize=9,
    )
    ax1.set_title("Excitation Waveform", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Time (µs)")
    ax1.set_ylabel("Signal Amplitude (V)")
    ax1.legend(loc="lower right")

    ax2.plot(t, received, color="#F43F5E", linewidth=1.8, label="Received Signal")
    ax2.plot(t, np.zeros_like(t), color="#94A3B8", alpha=0.3, linestyle="--")
    ax2.plot(14.0, 0.8, "ro", markersize=8, label="Corresponding Peak")
    ax2.annotate(
        "Corresponding Peak\n(t = 14.0 µs)",
        xy=(14.0, 0.8),
        xytext=(16.5, 0.5),
        arrowprops={
            "facecolor": "#ef4444",
            "shrink": 0.08,
            "width": 1.5,
            "headwidth": 6,
        },
        color="#ef4444",
        fontweight="bold",
        fontsize=9,
    )
    ax2.set_title("Received Waveform", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Time (µs)")
    ax2.set_ylabel("Signal Amplitude (V)")
    ax2.legend(loc="lower right")

    ax1.axvline(5.0, color="#ef4444", linestyle=":", alpha=0.7)
    ax2.axvline(5.0, color="#ef4444", linestyle=":", alpha=0.7)
    ax1.axvline(14.0, color="#ef4444", linestyle=":", alpha=0.7)
    ax2.axvline(14.0, color="#ef4444", linestyle=":", alpha=0.7)

    ax2.annotate(
        "",
        xy=(5.0, -0.6),
        xytext=(14.0, -0.6),
        arrowprops={"arrowstyle": "<->", "color": "#f59e0b", "lw": 2},
    )
    ax2.text(
        9.5,
        -0.5,
        "Time-of-Flight (Δt = 9.0 µs)",
        color="#f59e0b",
        fontweight="bold",
        ha="center",
        fontsize=9,
    )

    plt.subplots_adjust(hspace=0.45)
    return fig


# Dynamic Tukey Cross-Correlation plot
def plot_signal_alignment_diagram():
    t = np.linspace(0, 24, 1000)
    excit, ex_env = generate_tukey_burst(t, 5.0, 8.0, alpha=0.5, f0=0.5)
    received, _ = generate_tukey_burst(t, 14.0, 8.0, alpha=0.5, f0=0.5)
    received *= 0.8

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax in (ax1, ax2):
        ax.set_facecolor("#0F172A")
        ax.grid(True, color="#334155", linestyle=":", linewidth=0.8)
        ax.tick_params(colors="#94A3B8")
        ax.xaxis.label.set_color("#94A3B8")
        ax.yaxis.label.set_color("#94A3B8")
        ax.title.set_color("#F1F5F9")
    fig.patch.set_facecolor("#1E293B")

    # Shifted (Raw Signals)
    ax1.plot(t, excit, color="#38BDF8", linewidth=1.5, label="Excitation Pulse")
    ax1.plot(t, received, color="#F43F5E", linewidth=1.5, label="Received Signal")

    # Arrow showing the direction the received signal is shifted (leftward) to become aligned
    ax1.annotate(
        "Shift left to align",
        xy=(5.0, 0.4),
        xytext=(14.0, 0.4),
        arrowprops={"arrowstyle": "->", "color": "#10B981", "lw": 2, "ls": "--"},
        color="#10B981",
        fontweight="bold",
        fontsize=9,
        ha="center",
    )

    ax1.set_title("Shifted (Raw Signals)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Time (µs)")
    ax1.set_ylabel("Signal Amplitude (V)")
    ax1.legend(loc="lower right")

    # Aligned Envelopes
    aligned_rec, aligned_rec_env = generate_tukey_burst(t, 5.0, 8.0, alpha=0.5, f0=0.5)
    aligned_rec *= 0.8
    aligned_rec_env *= 0.8

    ax2.plot(
        t,
        ex_env,
        color="#38BDF8",
        linestyle="--",
        linewidth=1.8,
        label="Excitation Envelope",
    )
    ax2.plot(
        t,
        aligned_rec_env,
        color="#F43F5E",
        linestyle="-.",
        linewidth=1.8,
        label="Aligned Rec Envelope",
    )
    ax2.plot(
        t,
        aligned_rec,
        color="#F43F5E",
        alpha=0.3,
        linewidth=1.2,
        label="Aligned Signal",
    )
    ax2.set_title(
        "Aligned Envelopes (Cross-Correlation)", fontsize=11, fontweight="bold"
    )
    ax2.set_xlabel("Time (µs)")
    ax2.set_ylabel("Signal Amplitude (V)")
    ax2.legend(loc="lower right")

    plt.tight_layout()
    return fig


def render_attempt_card(att: dict) -> str:
    err_type = att.get("error_type", "UNKNOWN")
    msg = att.get("message", "")
    attempt_num = att.get("attempt", 1)

    status_map = {
        "REJECT_CAT2": "Transducer Ringing / Initial Bang Noise Overlap",
        "REJECT_CAT1": "Wave Mode Overlap (Longitudinal wave leakage)",
        "REJECT_CYCLE": "Cycle Mismatch",
        "REJECT_BOUNDS": "Velocity Out of Typical Range",
        "PASS": "Success (Pass)",
    }
    status_display = status_map.get(err_type, err_type)

    if err_type == "PASS":
        badge_text = "VERIFIED"
        badge_bg = "#d1fae5"
        badge_fg = "#065f46"
        border_color = "#10b981"
        icon = "✅"
    else:
        badge_text = "REJECTED"
        badge_bg = "#fee2e2"
        badge_fg = "#991b1b"
        border_color = "#ef4444"
        icon = "❌"

    import textwrap

    return textwrap.dedent(f"""\
        <div style='background-color: #f8fafc; padding: 1rem; border-radius: 8px; margin-bottom: 0.8rem; border-left: 4px solid {border_color}; box-shadow: 0 1px 3px rgba(0,0,0,0.05);'>
            <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;'>
                <strong style='font-size: 1rem; color: #1e293b;'>{icon} Attempt {attempt_num}: {status_display}</strong>
                <span style='background-color: {badge_bg}; color: {badge_fg}; padding: 0.15rem 0.5rem; border-radius: 9999px; font-size: 0.75rem; font-weight: 600;'>{badge_text}</span>
            </div>
            <div style='color: #475569; font-size: 0.9rem; line-height: 1.4;'>{msg}</div>
        </div>""")


def parse_learning_mode_blocks(tutor_output: str) -> dict:
    blocks = tutor_output.split("\n\n=========================================\n\n")
    result = {
        "step1": blocks[0] if len(blocks) > 0 else "",
        "step2": blocks[1] if len(blocks) > 1 else "",
        "att3_selections": "",
        "att1_review": "",
        "att2_review": "",
        "att3_review": "",
        "elastic_review": "",
    }

    if len(blocks) > 2:
        block3 = blocks[2]

        tags = [
            "[ATTEMPT_1_REVIEW]",
            "[ATTEMPT_2_REVIEW]",
            "[ATTEMPT_3_REVIEW]",
            "[ELASTIC_PROPERTIES_REVIEW]",
        ]
        split_idx = len(block3)
        for tag in tags:
            idx = block3.find(tag)
            if idx != -1 and idx < split_idx:
                split_idx = idx

        result["att3_selections"] = block3[:split_idx].strip()

        def extract_tag(tag, next_tags, text):
            start = text.find(tag)
            if start == -1:
                return ""
            start += len(tag)
            end = len(text)
            for nt in next_tags:
                idx = text.find(nt)
                if idx != -1 and idx > start and idx < end:
                    end = idx
            return text[start:end].strip()

        result["att1_review"] = extract_tag("[ATTEMPT_1_REVIEW]", tags, block3)
        result["att2_review"] = extract_tag("[ATTEMPT_2_REVIEW]", tags, block3)
        result["att3_review"] = extract_tag("[ATTEMPT_3_REVIEW]", tags, block3)
        result["elastic_review"] = extract_tag(
            "[ELASTIC_PROPERTIES_REVIEW]", tags, block3
        )

    return result


def get_tutorial_content(key: str, placeholders: dict) -> str:
    path = os.path.join(BASE_DIR, "references", "tutorial_static_content.json")
    defaults = {
        "step0_welcome_card": "Welcome to the WaveTutor Guided Tutorial! Before we begin, let's review the parameters used for this experiment:\n* **Specimen Material:** {{material}}\n* **Specimen Thickness ($d$):** {{thickness}} mm\n* **Signal Frequency ($f_0$):** {{frequency}} kHz\n\n##### 🎯 Tutorial Objectives\nIn this guided tutorial, we will take you step-by-step through the solver's attempts to measure the ultrasonic sound speeds:\n* **Transducer Ringing & Startup Electrical Noise:** Analyze the impact of transducer startup ringing.\n* **Longitudinal Wave Leakage (Mode Overlap):** Inspect wave mode leakage on the shear channel.\n* **Correct Peak Alignment & Moduli Calculation:** Match corresponding peaks, calculate true wave velocities, and evaluate dynamic elastic properties.",
        "attempt3_pass_message": "#### 🟢 Status: Verification PASS\n\n**Well done! The calculated wave velocities and travel times are correct.**\n\n* **Why they are correct:** By using cross-correlation to align the signals, we matched the corresponding peaks (the same peak index of the wave package) on both waveforms. This avoids the early transducer startup ringing noise and the longitudinal wave mode leakage, yielding the true physical travel time of the sound wave.\n\n* **Next Steps:** We will now move on to a step-by-step revision of the results and discuss what these physical properties mean for the material.",
        "teacher_review_greeting": "Hello there! Great job persevering through the NDT measurement session for the {{material}} sample. It's a fantastic learning experience, and understanding these signals is a critical skill in ultrasonic testing. Let's break down your attempts and see what we can learn.",
    }
    msg = defaults.get(key, "")
    if os.path.exists(path):
        try:
            with open(path) as f:
                data = json.load(f)
                msg = data.get(key, msg)
        except Exception:
            pass
    for k, v in placeholders.items():
        msg = msg.replace(f"{{{{{k}}}}}", str(v))
    return msg


# Calculate and store tutor resolved peaks globally to display in all steps
def populate_tutorial_peaks(info: dict, thickness: float):
    ex_peaks = find_signal_peaks(info["excit_file"])
    rec_l_peaks = find_signal_peaks(info["longi_file"])
    rec_s_peaks = find_signal_peaks(info["shear_file"])

    actual_cL = info["actual_cL_m_s"]
    actual_cS = info["actual_cS_m_s"]
    true_tof_l = (thickness / 1000.0) / actual_cL * 1e6
    true_tof_s = (thickness / 1000.0) / actual_cS * 1e6

    ex_ref_l = ex_peaks[2] if len(ex_peaks) > 2 else 2.25
    ex_ref_s = ex_peaks[2] if len(ex_peaks) > 2 else 2.25

    def find_closest(val, peak_list):
        if not peak_list:
            return val
        return min(peak_list, key=lambda x: abs(x - val))

    # Attempt 1 (Transducer ringing)
    p1_l_ex = ex_ref_l
    p1_l_rec = rec_l_peaks[1] if len(rec_l_peaks) > 1 else 1.267
    p1_s_ex = ex_ref_s
    p1_s_rec = rec_s_peaks[1] if len(rec_s_peaks) > 1 else 1.267

    # Attempt 2 (Longitudinal leakage)
    p2_l_ex = ex_ref_l
    p2_l_rec = find_closest(p1_l_ex + true_tof_l, rec_l_peaks)
    p2_s_ex = ex_ref_s
    p2_s_rec = find_closest(p1_s_ex + true_tof_l, rec_s_peaks)

    # Attempt 3 (PASS)
    p3_l_ex = ex_ref_l
    p3_l_rec = find_closest(p3_l_ex + true_tof_l, rec_l_peaks)
    p3_s_ex = ex_ref_s
    p3_s_rec = find_closest(p3_s_ex + true_tof_s, rec_s_peaks)

    # Save to state
    st.session_state["tutor_resolved_l_ex_att1"] = p1_l_ex
    st.session_state["tutor_resolved_l_rec_att1"] = p1_l_rec
    st.session_state["tutor_resolved_s_ex_att1"] = p1_s_ex
    st.session_state["tutor_resolved_s_rec_att1"] = p1_s_rec

    st.session_state["tutor_resolved_l_ex_att2"] = p2_l_ex
    st.session_state["tutor_resolved_l_rec_att2"] = p2_l_rec
    st.session_state["tutor_resolved_s_ex_att2"] = p2_s_ex
    st.session_state["tutor_resolved_s_rec_att2"] = p2_s_rec

    st.session_state["tutor_resolved_l_ex"] = p3_l_ex
    st.session_state["tutor_resolved_l_rec"] = p3_l_rec
    st.session_state["tutor_resolved_s_ex"] = p3_s_ex
    st.session_state["tutor_resolved_s_rec"] = p3_s_rec


# Waveform plotter helper inside guided step card
def render_learning_step_plot(info: dict, step_num: int):
    if step_num in (1, 4):
        l_ex = st.session_state.get("tutor_resolved_l_ex_att1", 0.0)
        l_rec = st.session_state.get("tutor_resolved_l_rec_att1", 0.0)
        s_ex = st.session_state.get("tutor_resolved_s_ex_att1", 0.0)
        s_rec = st.session_state.get("tutor_resolved_s_rec_att1", 0.0)
        lbl = "Attempt 1"
    elif step_num in (2, 5):
        l_ex = st.session_state.get("tutor_resolved_l_ex_att2", 0.0)
        l_rec = st.session_state.get("tutor_resolved_l_rec_att2", 0.0)
        s_ex = st.session_state.get("tutor_resolved_s_ex_att2", 0.0)
        s_rec = st.session_state.get("tutor_resolved_s_rec_att2", 0.0)
        lbl = "Attempt 2"
    elif step_num in (3, 6):
        l_ex = st.session_state.get("tutor_resolved_l_ex", 0.0)
        l_rec = st.session_state.get("tutor_resolved_l_rec", 0.0)
        s_ex = st.session_state.get("tutor_resolved_s_ex", 0.0)
        s_rec = st.session_state.get("tutor_resolved_s_rec", 0.0)
        lbl = "Attempt 3"
    else:
        return

    t_ex, excit = load_signal(info["excit_file"])
    _, longi = load_signal(info["longi_file"])
    _, shear = load_signal(info["shear_file"])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
    for ax in (ax1, ax2):
        ax.set_facecolor("#0F172A")
        ax.grid(True, color="#334155", linestyle=":", linewidth=0.8)
        ax.tick_params(colors="#94A3B8")
        ax.xaxis.label.set_color("#94A3B8")
        ax.yaxis.label.set_color("#94A3B8")
        ax.title.set_color("#F1F5F9")
    fig.patch.set_facecolor("#1E293B")

    # Plot Longitudinal Channel
    ax1.plot(
        t_ex,
        excit,
        label="Excitation Pulse",
        color="#94A3B8",
        alpha=0.5,
        linewidth=1.2,
    )
    ax1.plot(
        t_ex,
        longi,
        label="Received Longitudinal Wave",
        color="#38BDF8",
        linewidth=1.5,
    )
    ax1.set_title("Longitudinal Channel", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Time (µs)")
    ax1.set_ylabel("Signal Amplitude (V)")
    if l_ex > 0:
        ax1.axvline(
            l_ex,
            color="#38BDF8",
            linestyle="--",
            linewidth=1.2,
            label=f"{lbl} L Ex Selection",
        )
    if l_rec > 0:
        ax1.axvline(
            l_rec,
            color="#38BDF8",
            linestyle="-",
            linewidth=1.5,
            label=f"{lbl} L Rec Selection",
        )
    ax1.legend(loc="lower right")

    # Plot Shear Channel
    ax2.plot(
        t_ex,
        excit,
        label="Excitation Pulse",
        color="#94A3B8",
        alpha=0.5,
        linewidth=1.2,
    )
    ax2.plot(
        t_ex,
        shear,
        label="Received Shear Wave",
        color="#F43F5E",
        linewidth=1.5,
    )
    ax2.set_title("Shear Channel", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Time (µs)")
    ax2.set_ylabel("Signal Amplitude (V)")
    if s_ex > 0:
        ax2.axvline(
            s_ex,
            color="#F43F5E",
            linestyle="--",
            linewidth=1.2,
            label=f"{lbl} S Ex Selection",
        )
    if s_rec > 0:
        ax2.axvline(
            s_rec,
            color="#F43F5E",
            linestyle="-",
            linewidth=1.5,
            label=f"{lbl} S Rec Selection",
        )
    ax2.legend(loc="lower right")

    plt.subplots_adjust(hspace=0.45)
    st.pyplot(fig)
    plt.close(fig)


# Function to inline waveforms plot inside "Signal Check" text block
def render_step_with_plot(step_text: str, step_num: int):
    if "Signal Check" in step_text:
        parts = step_text.split("Signal Check")
        st.markdown(parts[0] + "Signal Check")
        render_learning_step_plot(info, step_num)
        st.markdown(parts[1])
    else:
        st.markdown(step_text)
        render_learning_step_plot(info, step_num)


# Custom Styles
st.markdown(
    """
    <style>
    .main-title-card {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        color: white;
        padding: 2.5rem;
        border-radius: 16px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
    }
    .main-title {
        font-size: 3rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.025em;
    }
    .main-subtitle {
        font-size: 1.25rem;
        opacity: 0.9;
        margin-top: 0.5rem;
        font-weight: 400;
    }
    .metric-card {
        background-color: var(--secondary-background-color);
        padding: 1.25rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.02);
        text-align: center;
        border: 1px solid rgba(128, 128, 128, 0.15);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #3b82f6;
    }
    .metric-label {
        font-size: 0.95rem;
        opacity: 0.85;
    }
    .equation-box {
        background-color: rgba(59, 130, 246, 0.08);
        border: 1.5px solid #3b82f6;
        border-radius: 8px;
        padding: 1.25rem;
        margin: 1.25rem 0;
        text-align: center;
    }
    .footer-note {
        font-size: 0.85rem;
        opacity: 0.7;
        margin-top: 2rem;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# WaveTutor Banner
st.markdown(
    """
    <div class='main-title-card'>
        <div class='main-title'>🔊 WaveTutor</div>
        <div class='main-subtitle'>Interactive Learning Module for Ultrasonic Nondestructive Testing</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# State initialization
if "current_page" not in st.session_state:
    st.session_state["current_page"] = "🏠 Home / Dashboard"
if "session_service" not in st.session_state:
    st.session_state["session_service"] = InMemorySessionService()

# Learning / Tutor Mode persistent state
if "tutor_output" not in st.session_state:
    st.session_state["tutor_output"] = None
if "tutor_history" not in st.session_state:
    st.session_state["tutor_history"] = None
if "tutor_resolved_l_ex" not in st.session_state:
    st.session_state["tutor_resolved_l_ex"] = 0.0
if "tutor_resolved_l_rec" not in st.session_state:
    st.session_state["tutor_resolved_l_rec"] = 0.0
if "tutor_resolved_s_ex" not in st.session_state:
    st.session_state["tutor_resolved_s_ex"] = 0.0
if "tutor_resolved_s_rec" not in st.session_state:
    st.session_state["tutor_resolved_s_rec"] = 0.0
if "tutor_resolved_l_ex_att1" not in st.session_state:
    st.session_state["tutor_resolved_l_ex_att1"] = 0.0
if "tutor_resolved_l_rec_att1" not in st.session_state:
    st.session_state["tutor_resolved_l_rec_att1"] = 0.0
if "tutor_resolved_s_ex_att1" not in st.session_state:
    st.session_state["tutor_resolved_s_ex_att1"] = 0.0
if "tutor_resolved_s_rec_att1" not in st.session_state:
    st.session_state["tutor_resolved_s_rec_att1"] = 0.0
if "tutor_resolved_l_ex_att2" not in st.session_state:
    st.session_state["tutor_resolved_l_ex_att2"] = 0.0
if "tutor_resolved_l_rec_att2" not in st.session_state:
    st.session_state["tutor_resolved_l_rec_att2"] = 0.0
if "tutor_resolved_s_ex_att2" not in st.session_state:
    st.session_state["tutor_resolved_s_ex_att2"] = 0.0
if "tutor_resolved_s_rec_att2" not in st.session_state:
    st.session_state["tutor_resolved_s_rec_att2"] = 0.0
if "learning_step" not in st.session_state:
    st.session_state["learning_step"] = 0
if "last_learning_step" not in st.session_state:
    st.session_state["last_learning_step"] = 0
if "learning_material" not in st.session_state:
    st.session_state["learning_material"] = "Aluminum"
if "learning_thickness" not in st.session_state:
    st.session_state["learning_thickness"] = 5.0
if "learning_frequency" not in st.session_state:
    st.session_state["learning_frequency"] = 500.0
if "learning_signal_info" not in st.session_state:
    st.session_state["learning_signal_info"] = None

# Practice Mode persistent state
if "practice_material" not in st.session_state:
    st.session_state["practice_material"] = "Aluminum"
if "practice_thickness" not in st.session_state:
    st.session_state["practice_thickness"] = 5.0
if "practice_frequency" not in st.session_state:
    st.session_state["practice_frequency"] = 500.0
if "practice_signal_info" not in st.session_state:
    st.session_state["practice_signal_info"] = None
if "practice_output" not in st.session_state:
    st.session_state["practice_output"] = None
if "practice_history" not in st.session_state:
    st.session_state["practice_history"] = []
if "practice_last_status" not in st.session_state:
    st.session_state["practice_last_status"] = None
if "practice_last_inputs" not in st.session_state:
    st.session_state["practice_last_inputs"] = None
if "practice_inputs" not in st.session_state:
    st.session_state["practice_inputs"] = {
        "l_ex": 2.25,
        "l_rec": 5.45,
        "s_ex": 2.25,
        "s_rec": 8.55,
    }
if "practice_resolved_l_ex" not in st.session_state:
    st.session_state["practice_resolved_l_ex"] = 0.0
if "practice_resolved_l_rec" not in st.session_state:
    st.session_state["practice_resolved_l_rec"] = 0.0
if "practice_resolved_s_ex" not in st.session_state:
    st.session_state["practice_resolved_s_ex"] = 0.0
if "practice_resolved_s_rec" not in st.session_state:
    st.session_state["practice_resolved_s_rec"] = 0.0

# Initialize peak picker keys directly in session state
if "prac_l_ex_num" not in st.session_state:
    st.session_state["prac_l_ex_num"] = 2.25
if "prac_l_rec_num" not in st.session_state:
    st.session_state["prac_l_rec_num"] = 5.45
if "prac_s_ex_num" not in st.session_state:
    st.session_state["prac_s_ex_num"] = 2.25
if "prac_s_rec_num" not in st.session_state:
    st.session_state["prac_s_rec_num"] = 8.55

# Cache keys for dynamic signal generation checks
if "learning_generated_material" not in st.session_state:
    st.session_state["learning_generated_material"] = None
if "learning_generated_thickness" not in st.session_state:
    st.session_state["learning_generated_thickness"] = None
if "learning_generated_frequency" not in st.session_state:
    st.session_state["learning_generated_frequency"] = None
if "practice_generated_material" not in st.session_state:
    st.session_state["practice_generated_material"] = None
if "practice_generated_thickness" not in st.session_state:
    st.session_state["practice_generated_thickness"] = None
if "practice_generated_frequency" not in st.session_state:
    st.session_state["practice_generated_frequency"] = None

# Sidebar Navigation
st.sidebar.markdown(
    "<h2 style='text-align: center; font-weight: 800;'>🧭 Menu</h2>",
    unsafe_allow_html=True,
)
pages = [
    "🏠 Home / Dashboard",
    "🌊 Longitudinal vs. Shear Waves",
    "⏱️ Measuring Sound Speed",
    "🏗️ Material Properties",
    "🎓 Guided Tutorial",
    "🧪 Practice Sandbox",
]
selected_page = st.sidebar.radio(
    "Go to:", pages, index=pages.index(st.session_state["current_page"])
)
st.session_state["current_page"] = selected_page

if "api_key_activated" not in st.session_state:
    st.session_state["api_key_activated"] = False
if "original_server_key" not in st.session_state:
    st.session_state["original_server_key"] = os.environ.get("GEMINI_API_KEY", "")

st.sidebar.markdown("---")
st.sidebar.markdown(
    "<h4 style='font-weight: 700; margin-bottom: 2px;'>🔑 Gemini API Authentication</h4>",
    unsafe_allow_html=True,
)

key_input = st.sidebar.text_input(
    "Enter your API Key:",
    type="password",
    placeholder="AIzaSy...",
    value=st.session_state.get("visitor_key_value", ""),
    help="Optional: Enter a Gemini API Key from Google AI Studio. If empty, the app will use the server's default configuration (or seamlessly switch to the local physics fallback).",
)

col1, col2 = st.sidebar.columns(2)
with col1:
    if st.button("Activate Key", use_container_width=True, key="act_btn"):
        if key_input:
            st.session_state["visitor_key_value"] = key_input
            st.session_state["api_key_activated"] = True
            st.rerun()
with col2:
    if st.button("Clear Key", use_container_width=True, key="clr_btn"):
        st.session_state["visitor_key_value"] = ""
        st.session_state["api_key_activated"] = False
        st.rerun()

# Apply key status message in sidebar (the actual key is isolated dynamically inside run_agent_with_isolation)
if st.session_state["api_key_activated"] and st.session_state.get("visitor_key_value"):
    st.sidebar.success("🟢 API Key Activated!")
else:
    # If not visitor-activated, show if server has default key or if we fall back
    if ORIGINAL_SERVER_KEY:
        key_prefix = ORIGINAL_SERVER_KEY[:5] + "..." if len(ORIGINAL_SERVER_KEY) > 5 else ORIGINAL_SERVER_KEY
        st.sidebar.info(f"ℹ️ Running on server configuration ({key_prefix}).")
    else:
        st.sidebar.info("ℹ️ Running on local fallback.")


# =========================================================================
# Page 1: Home / Dashboard
# =========================================================================
if st.session_state["current_page"] == "🏠 Home / Dashboard":
    col_left, col_right = st.columns([1.5, 1.0], gap="large")

    with col_left:
        with st.container(border=True):
            st.markdown("### 📖 Introduction")
            st.markdown(load_text_content("homepage_introduction.txt"))

    with col_right:
        st.markdown(
            "<h3 style='margin-bottom: 1rem;'>📚 Study Topics</h3>",
            unsafe_allow_html=True,
        )

        with st.container(border=True):
            st.markdown("#### 🌊 Longitudinal vs. Shear Waves")
            st.write(
                "Understand compressional vs transverse particle oscillations and acoustic velocities."
            )
            if st.button("Open Wave Review", key="nav_waves", use_container_width=True):
                st.session_state["current_page"] = "🌊 Longitudinal vs. Shear Waves"
                st.rerun()

        st.write("")
        with st.container(border=True):
            st.markdown("#### ⏱️ Measuring Sound Speed")
            st.write(
                "Study through-transmission setups, manual peak picking, and cross-correlation alignment."
            )
            if st.button(
                "Open Measurement Review", key="nav_speed", use_container_width=True
            ):
                st.session_state["current_page"] = "⏱️ Measuring Sound Speed"
                st.rerun()

        st.write("")
        with st.container(border=True):
            st.markdown("#### 🏗️ Material Properties")
            st.write(
                "Explore how sound speed reveals Young's modulus, shear modulus, and density parameters."
            )
            if st.button(
                "Open Properties Review",
                key="nav_properties",
                use_container_width=True,
            ):
                st.session_state["current_page"] = "🏗️ Material Properties"
                st.rerun()

    st.markdown("---")
    st.markdown(
        "<h3 style='text-align: center; margin-bottom: 1.5rem;'>⚙️ Testing Simulation</h3>",
        unsafe_allow_html=True,
    )

    col_t1, col_t2 = st.columns(2, gap="medium")
    with col_t1:
        with st.container(border=True):
            st.markdown("### 🎓 Guided Tutorial")
            st.write(
                "Walk step-by-step through a simulated experiment. This interactive mode features prewritten educational templates "
                "coupled with live explanations dynamically compiled by our autonomous AI agents—the Research Assistant (RA) drafting lessons "
                "and the Principal Investigator (PI) auditing and validating selections."
            )
            if st.button(
                "Start Guided Tutorial ➔",
                key="start_learn_home",
                use_container_width=True,
            ):
                st.session_state["current_page"] = "🎓 Guided Tutorial"
                st.rerun()
    with col_t2:
        with st.container(border=True):
            st.markdown("### 🧪 Practice Sandbox")
            st.write(
                "Conduct independent TOF analysis. Input your selections and get reviewed by the AI agents."
            )
            if st.button(
                "Start Practice Sandbox ➔",
                key="start_practice_home",
                use_container_width=True,
            ):
                st.session_state["current_page"] = "🧪 Practice Sandbox"
                st.rerun()


# =========================================================================
# Page 2: Longitudinal vs. Shear Waves
# =========================================================================
elif st.session_state["current_page"] == "🌊 Longitudinal vs. Shear Waves":
    st.markdown("## 🌊 Longitudinal vs. Shear Sound Waves")
    st.markdown("---")

    # Load text content
    waves_text = load_text_content("longitudinal_vs_shear_text.txt")

    long_desc = waves_text.split("### Shear Waves")[0].strip()
    shear_desc = "### Shear Waves\n" + waves_text.split("### Shear Waves")[1].strip()

    # Row 1: Longitudinal
    col_l_text, col_l_diag = st.columns([1.2, 1.0], gap="large")
    with col_l_text:
        with st.container(border=True):
            st.markdown(long_desc)
    with col_l_diag:
        with st.container(border=True):
            st.markdown("#### Longitudinal Wave Diagram")
            render_svg(get_longitudinal_wave_diagram(), "longitudinal")

    # Row 2: Shear
    col_s_text, col_s_diag = st.columns([1.2, 1.0], gap="large")
    with col_s_text:
        with st.container(border=True):
            st.markdown(shear_desc)
    with col_s_diag:
        with st.container(border=True):
            st.markdown("#### Shear Wave Diagram")
            render_svg(get_shear_wave_diagram(), "shear")

    # Comparison summary card
    with st.container(border=True):
        st.markdown("#### 💡 Core Takeaway")
        st.markdown(
            "*   **Longitudinal Waves:** Particle displacement is **parallel** to the direction of wave travel. These are fastest and propagate in all media.\n"
            "*   **Shear Waves:** Particle displacement is **perpendicular** to the direction of wave travel. They typically propagate only in solids."
        )


# =========================================================================
# Page 3: Measuring Sound Speed
# =========================================================================
elif st.session_state["current_page"] == "⏱️ Measuring Sound Speed":
    st.markdown("## ⏱️ Measuring Sound Speed Using Excitation & Received Signals")
    st.markdown("---")

    meas_text = load_text_content("measuring_sound_speed_text.txt")
    blocks = meas_text.split("---")

    # Top Overview
    with st.container(border=True):
        st.markdown(blocks[0].strip())

    # Stacked teaching blocks
    # Block 1: Through-Transmission
    col_t_text, col_t_diag = st.columns([1.2, 1.0], gap="large")
    with col_t_text:
        with st.container(border=True):
            st.markdown(blocks[1].strip())
    with col_t_diag:
        with st.container(border=True):
            st.markdown("#### Through-Transmission Setup")
            render_svg(get_through_transmission_diagram(), "through")

    # Block 2: Time of Flight Peak Matching (Matplotlib Tukey plot)
    col_p_text, col_p_diag = st.columns([1.2, 1.0], gap="large")
    with col_p_text:
        with st.container(border=True):
            st.markdown(blocks[2].strip())
    with col_p_diag:
        with st.container(border=True):
            st.markdown("#### Time-of-Flight Peak Matching")
            st.pyplot(plot_peak_matching_diagram())

    # Block 3: Signal Alignment and Cross Correlation (Matplotlib Tukey plot)
    col_a_text, col_a_diag = st.columns([1.2, 1.0], gap="large")
    with col_a_text:
        with st.container(border=True):
            st.markdown(blocks[3].strip())
    with col_a_diag:
        with st.container(border=True):
            st.markdown("#### Signal Alignment & Cross-Correlation")
            st.pyplot(plot_signal_alignment_diagram())


# =========================================================================
# Page 4: Material Properties
# =========================================================================
elif st.session_state["current_page"] == "🏗️ Material Properties":
    st.markdown("## 🏗️ Material Properties from Sound Speed Measurements")
    st.markdown("---")

    col_left, col_right = st.columns(2, gap="large")

    with col_left:
        with st.container(border=True):
            st.markdown("### 🔬 Physical Concepts")
            st.markdown(load_text_content("material_properties_text.txt"))

    with col_right:
        with st.container(border=True):
            st.markdown("### 📐 Governing Physics Equations")
            st.markdown(load_text_content("material_property_equations.txt"))

    with st.container(border=True):
        st.markdown("#### ⚠️ Physical Assumptions and Boundary Conditions")
        st.markdown(
            "These dynamic equations are valid strictly under these physical boundary conditions:\n\n"
            "*   **Homogeneity:** The material properties must be uniform throughout the volume.\n"
            "*   **Isotropy:** The elastic stiffness must be identical in all directions (not true for single crystals or textured/rolled sheets).\n"
            "*   **Linear Elasticity:** Strains must remain extremely small so Hooke's law is obeyed.\n"
            "*   **Perfect Acoustic Coupling:** Transducers must have flat, uniform contact to avoid introducing spurious phase delays."
        )


# =========================================================================
# Page 5: Guided Tutorial
# =========================================================================
elif st.session_state["current_page"] == "🎓 Guided Tutorial":
    st.markdown("## 🎓 Guided Tutorial")
    st.markdown("---")

    # Dynamic intro
    st.markdown(load_text_content("learning_mode_text.txt"))

    # Parameter Selection Panel
    with st.container(border=True):
        st.markdown("### ⚙️ Specimen Parameters")
        col_p1, col_p2, col_p3 = st.columns([1.5, 1.5, 1.0])
        with col_p1:
            l_material = st.selectbox(
                "Select Specimen Metal:",
                options=[m.capitalize() for m in METALS_DATABASE.keys()],
                key="learning_material",
            )
        with col_p2:
            l_thickness = st.slider(
                "Specimen Thickness (d):",
                min_value=5.0,
                max_value=50.0,
                step=5.0,
                format="%g mm",
                key="learning_thickness",
            )
        with col_p3:
            l_frequency = st.slider(
                "Testing Frequency (kHz):",
                min_value=500.0,
                max_value=1000.0,
                step=50.0,
                format="%g kHz",
                key="learning_frequency",
            )

    # Dynamic signal generation on inputs change (Pre-render Waveforms)
    if (
        st.session_state["learning_signal_info"] is None
        or st.session_state.get("learning_generated_material") != l_material
        or st.session_state.get("learning_generated_thickness") != l_thickness
        or st.session_state.get("learning_generated_frequency") != l_frequency
    ):
        st.session_state["learning_generated_material"] = l_material
        st.session_state["learning_generated_thickness"] = l_thickness
        st.session_state["learning_generated_frequency"] = l_frequency

        info = generate_signal_data(
            material_name=l_material,
            thickness_mm=l_thickness,
            frequency_khz=l_frequency,
            output_folder="outputs",
        )
        st.session_state["learning_signal_info"] = info
        populate_tutorial_peaks(info, l_thickness)
        st.session_state["tutor_output"] = None
        st.session_state["tutor_history"] = None

    # Separate box showing physical signals before clicks (RAW ONLY)
    info = st.session_state["learning_signal_info"]
    with st.container(border=True):
        st.markdown("### 📈 Physical Waveforms")

        t_ex, excit = load_signal(info["excit_file"])
        _, longi = load_signal(info["longi_file"])
        _, shear = load_signal(info["shear_file"])

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
        for ax in (ax1, ax2):
            ax.set_facecolor("#0F172A")
            ax.grid(True, color="#334155", linestyle=":", linewidth=0.8)
            ax.tick_params(colors="#94A3B8")
            ax.xaxis.label.set_color("#94A3B8")
            ax.yaxis.label.set_color("#94A3B8")
            ax.title.set_color("#F1F5F9")
        fig.patch.set_facecolor("#1E293B")

        # Plot L
        ax1.plot(
            t_ex,
            excit,
            label="Excitation Pulse",
            color="#94A3B8",
            alpha=0.5,
            linewidth=1.2,
        )
        ax1.plot(
            t_ex,
            longi,
            label="Received Longitudinal Wave",
            color="#38BDF8",
            linewidth=1.5,
        )
        ax1.set_title("Longitudinal Channel", fontsize=11, fontweight="bold")
        ax1.set_xlabel("Time (µs)")
        ax1.set_ylabel("Signal Amplitude (V)")
        ax1.legend(loc="lower right")

        # Plot S
        ax2.plot(
            t_ex,
            excit,
            label="Excitation Pulse",
            color="#94A3B8",
            alpha=0.5,
            linewidth=1.2,
        )
        ax2.plot(
            t_ex,
            shear,
            label="Received Shear Wave",
            color="#F43F5E",
            linewidth=1.5,
        )
        ax2.set_title("Shear Channel", fontsize=11, fontweight="bold")
        ax2.set_xlabel("Time (µs)")
        ax2.set_ylabel("Signal Amplitude (V)")
        ax2.legend(loc="lower right")

        plt.subplots_adjust(hspace=0.45)
        st.pyplot(fig)
        plt.close(fig)

    # Action button
    tutor_trigger = st.button("🤖 Run Guided Tutorial", use_container_width=True)

    if tutor_trigger:
        with st.spinner(
            "Team of agents collaborating (Research Assistant drafting lesson -> Principal Investigator auditing & stamping)..."
        ):
            try:
                r = Runner(
                    app=adk_app,
                    session_service=st.session_state["session_service"],
                    auto_create_session=True,
                )
                prompt = f"Run teaching session for {l_material} with thickness {l_thickness}mm and frequency {l_frequency}kHz"
                run_session_id = f"session-{uuid.uuid4()}"
                events = run_agent_with_isolation(r, prompt, run_session_id)

                if events:
                    st.session_state["tutor_output"] = events[-1].output
                    try:
                        session = r.session_service.get_session_sync(
                            app_name="wave_tutor",
                            user_id="streamlit-user",
                            session_id=run_session_id,
                        )
                        if session:
                            st.session_state["tutor_history"] = session.state.get(
                                "error_history", []
                            )
                            # Repopulate from the runner outcome
                            populate_tutorial_peaks(info, l_thickness)
                    except Exception:
                        pass
                    st.session_state["learning_step"] = 0
                    st.rerun()
            except Exception:
                # Local fallback compile
                populate_tutorial_peaks(info, l_thickness)
                t_history = [
                    {
                        "attempt": 1,
                        "error_type": "REJECT_CAT2",
                        "message": "Selected startup noise.",
                    },
                    {
                        "attempt": 2,
                        "error_type": "REJECT_CAT1",
                        "message": "Selected longitudinal wave leakage.",
                    },
                    {
                        "attempt": 3,
                        "error_type": "PASS",
                        "message": "Longitudinal and shear acoustic wave velocities match corresponding features, and the calculated dynamic elastic moduli are within the typical physical range for this material.",
                    },
                ]

                # Moduli calculations
                actual_cL = info["actual_cL_m_s"]
                actual_cS = info["actual_cS_m_s"]
                rho = literature_db[l_material.lower()]["density_kg_m3"]
                G_calc = rho * (actual_cS**2) / 1e9
                nu_calc = (actual_cL**2 - 2 * actual_cS**2) / (
                    2 * (actual_cL**2 - actual_cS**2)
                )
                E_calc = 2 * G_calc * (1 + nu_calc)
                lit_E = literature_db[l_material.lower()]["youngs_modulus_gpa"]
                lit_G = literature_db[l_material.lower()]["shear_modulus_gpa"]
                lit_nu = literature_db[l_material.lower()]["poissons_ratio"]

                moduli_data = {
                    "calculated_youngs_modulus_gpa": E_calc,
                    "calculated_shear_modulus_gpa": G_calc,
                    "calculated_poissons_ratio": nu_calc,
                    "youngs_error_percent": abs(E_calc - lit_E) / lit_E * 100.0,
                    "shear_error_percent": abs(G_calc - lit_G) / lit_G * 100.0,
                    "poissons_error_percent": abs(nu_calc - lit_nu) / lit_nu * 100.0
                    if lit_nu > 0
                    else 0.0,
                }

                p3_l_ex = st.session_state["tutor_resolved_l_ex"]
                p3_l_rec = st.session_state["tutor_resolved_l_rec"]
                p3_s_ex = st.session_state["tutor_resolved_s_ex"]
                p3_s_rec = st.session_state["tutor_resolved_s_rec"]

                p1_l_ex = st.session_state["tutor_resolved_l_ex_att1"]
                p1_l_rec = st.session_state["tutor_resolved_l_rec_att1"]
                p1_s_ex = st.session_state["tutor_resolved_s_ex_att1"]
                p1_s_rec = st.session_state["tutor_resolved_s_rec_att1"]

                p2_l_ex = st.session_state["tutor_resolved_l_ex_att2"]
                p2_l_rec = st.session_state["tutor_resolved_l_rec_att2"]
                p2_s_ex = st.session_state["tutor_resolved_s_ex_att2"]
                p2_s_rec = st.session_state["tutor_resolved_s_rec_att2"]

                l_diff1 = p1_l_rec - p1_l_ex
                s_diff1 = p1_s_rec - p1_s_ex
                cL1 = (l_thickness / 1000.0) / (l_diff1 / 1e6) if l_diff1 > 0 else 0.0
                cS1 = (l_thickness / 1000.0) / (s_diff1 / 1e6) if s_diff1 > 0 else 0.0

                l_diff2 = p2_l_rec - p2_l_ex
                s_diff2 = p2_s_rec - p2_s_ex
                cL2 = (l_thickness / 1000.0) / (l_diff2 / 1e6) if l_diff2 > 0 else 0.0
                cS2 = (l_thickness / 1000.0) / (s_diff2 / 1e6) if s_diff2 > 0 else 0.0

                block1 = f"Attempt 1 Selections:\nLongitudinal: {p1_l_ex} -> {p1_l_rec} (Speed: {cL1:.1f} m/s)\nShear: {p1_s_ex} -> {p1_s_rec} (Speed: {cS1:.1f} m/s)"
                block2 = f"Attempt 2 Selections:\nLongitudinal: {p2_l_ex} -> {p2_l_rec} (Speed: {cL2:.1f} m/s)\nShear: {p2_s_ex} -> {p2_s_rec} (Speed: {cS2:.1f} m/s)"

                block3_content = compile_local_fallback_explanation(
                    material=l_material,
                    thickness=l_thickness,
                    freq=l_frequency,
                    history=t_history,
                    moduli=moduli_data,
                    cL=actual_cL,
                    cS=actual_cS,
                    practice_mode=False,
                    l_ex=p3_l_ex,
                    l_rec=p3_l_rec,
                    s_ex=p3_s_ex,
                    s_rec=p3_s_rec,
                )

                fallback_output = (
                    block1
                    + "\n\n=========================================\n\n"
                    + block2
                    + "\n\n=========================================\n\n"
                    + block3_content
                )
                st.session_state["tutor_output"] = fallback_output
                st.session_state["tutor_history"] = t_history
                st.session_state["learning_step"] = 0
                st.rerun()

    # Step-by-step workspace below waveforms and button
    if (
        st.session_state["tutor_output"] is not None
        and st.session_state["learning_signal_info"] is not None
    ):
        parsed = parse_learning_mode_blocks(st.session_state["tutor_output"])
        step = st.session_state["learning_step"]

        # Handle smooth auto-scroll to top of page when step changes
        if step != st.session_state["last_learning_step"]:
            st.session_state["last_learning_step"] = step
            st.html(
                "<script>window.parent.document.querySelector('.main').scrollTo({top: 0, behavior: 'smooth'});</script>"
            )

        st.markdown("### 📖 Step-by-Step Guidance")
        st.progress(step / 7.0)

        with st.container(border=True):
            if step == 0:
                welcome_md = get_tutorial_content(
                    "step0_welcome_card",
                    {
                        "material": st.session_state["learning_material"].capitalize(),
                        "thickness": f"{st.session_state['learning_thickness']:.1f}",
                        "frequency": f"{st.session_state['learning_frequency']:.0f}",
                    },
                )
                st.markdown(welcome_md)
            elif step == 1:
                render_step_with_plot(parsed["step1"], 1)
            elif step == 2:
                render_step_with_plot(parsed["step2"], 2)
            elif step == 3:
                content = parsed["att3_selections"]
                idx = content.find("#### ")
                if idx != -1:
                    selections_table = content[:idx].strip()
                    pass_msg = get_tutorial_content("attempt3_pass_message", {})
                    content = f"{selections_table}\n\n{pass_msg}"
                render_step_with_plot(content, 3)
            elif step == 4:
                st.markdown("### 🔍 Attempt 1 Review & Discussion")
                greeting = get_tutorial_content(
                    "teacher_review_greeting",
                    {"material": st.session_state["learning_material"].lower()},
                )
                review_text = parsed["att1_review"]
                import re

                match = re.search(r"\battempt\s+1\b", review_text, re.IGNORECASE)
                if match:
                    idx = match.start()
                    subtext = review_text[max(0, idx - 15) : idx]
                    prep_match = re.search(
                        r"\b(in|for|during|on|at)\s+$",
                        subtext.lstrip(),
                        re.IGNORECASE,
                    )
                    if prep_match:
                        idx = idx - (
                            len(subtext)
                            - subtext.lower().rfind(prep_match.group(1).lower())
                        )
                    review_text = review_text[idx:]
                st.markdown(f"*{greeting}*\n\n---\n\n{review_text}")
                render_learning_step_plot(info, 4)
            elif step == 5:
                st.markdown("### 🔍 Attempt 2 Review & Discussion")
                st.markdown(parsed["att2_review"])
                render_learning_step_plot(info, 5)
            elif step == 6:
                st.markdown("### 🎉 Attempt 3 Success Analysis")
                st.markdown(parsed["att3_review"])
                render_learning_step_plot(info, 6)
            elif step == 7:
                # No peak selection waveforms plot on elastic properties step
                st.markdown("### ⚙️ Specimen Elastic Properties & Moduli")
                st.markdown(parsed["elastic_review"])

        col_prev, col_middle, col_next = st.columns([1.2, 2.0, 1.2])
        with col_prev:
            if step > 0:
                if st.button("Previous", use_container_width=True):
                    st.session_state["learning_step"] -= 1
                    st.rerun()
        with col_middle:
            st.markdown(
                f"<div style='text-align: center; color: #64748b; font-weight: 600; padding-top: 0.5rem;'>"
                f"Step {step + 1} of 8<br><span style='font-size: 0.8rem; font-weight: normal;'>▲ Scroll up for full text</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
        with col_next:
            if step < 7:
                if st.button("Next", use_container_width=True):
                    st.session_state["learning_step"] += 1
                    st.rerun()
            else:
                if st.button("Restart Tutorial 🔄", use_container_width=True):
                    st.session_state["learning_step"] = 0
                    st.session_state["tutor_output"] = None
                    st.session_state["tutor_history"] = None
                    st.rerun()


# =========================================================================
# Page 6: Practice Sandbox
# =========================================================================
elif st.session_state["current_page"] == "🧪 Practice Sandbox":
    st.markdown("## 🧪 Practice Sandbox")
    st.markdown("---")

    # Introduction text loaded dynamically
    st.markdown(load_text_content("practice_mode_text.txt"))

    # Parameter Selection Panel
    with st.container(border=True):
        st.markdown("### ⚙️ Specimen Parameters")
        col_p1, col_p2, col_p3 = st.columns([1.5, 1.5, 1.0])
        with col_p1:
            p_material = st.selectbox(
                "Select Material Metal:",
                options=[m.capitalize() for m in METALS_DATABASE.keys()],
                key="practice_material",
            )
        with col_p2:
            p_thickness = st.slider(
                "Specimen Thickness (d):",
                min_value=5.0,
                max_value=50.0,
                step=5.0,
                format="%g mm",
                key="practice_thickness",
            )
        with col_p3:
            p_frequency = st.slider(
                "Testing Frequency (kHz):",
                min_value=500.0,
                max_value=1000.0,
                step=50.0,
                format="%g kHz",
                key="practice_frequency",
            )

    # Dynamic generation of waveforms when inputs change (Pre-render Waveforms)
    if (
        st.session_state["practice_signal_info"] is None
        or st.session_state.get("practice_generated_material") != p_material
        or st.session_state.get("practice_generated_thickness") != p_thickness
        or st.session_state.get("practice_generated_frequency") != p_frequency
    ):
        st.session_state["practice_generated_material"] = p_material
        st.session_state["practice_generated_thickness"] = p_thickness
        st.session_state["practice_generated_frequency"] = p_frequency

        info = generate_signal_data(
            material_name=p_material,
            thickness_mm=p_thickness,
            frequency_khz=p_frequency,
            output_folder="outputs",
        )
        st.session_state["practice_signal_info"] = info
        st.session_state["practice_output"] = None
        st.session_state["practice_history"] = []
        st.session_state["practice_resolved_l_ex"] = 0.0
        st.session_state["practice_resolved_l_rec"] = 0.0
        st.session_state["practice_resolved_s_ex"] = 0.0
        st.session_state["practice_resolved_s_rec"] = 0.0

        # Reset selection keys
        st.session_state["prac_l_ex_num"] = 2.25
        st.session_state["prac_l_rec_num"] = 5.45
        st.session_state["prac_s_ex_num"] = 2.25
        st.session_state["prac_s_rec_num"] = 8.55

    info = st.session_state["practice_signal_info"]

    # Waveform display box with selections and legends in the bottom right corner
    with st.container(border=True):
        st.markdown("### 📈 Physical Waveforms")

        t_ex, excit = load_signal(info["excit_file"])
        _, longi = load_signal(info["longi_file"])
        _, shear = load_signal(info["shear_file"])

        # Capture selection keys in real-time directly from widget state for instant graphing sync
        practice_l_ex = st.session_state.get("prac_l_ex_num", 2.25)
        practice_l_rec = st.session_state.get("prac_l_rec_num", 5.45)
        practice_s_ex = st.session_state.get("prac_s_ex_num", 2.25)
        practice_s_rec = st.session_state.get("prac_s_rec_num", 8.55)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
        for ax in (ax1, ax2):
            ax.set_facecolor("#0F172A")
            ax.grid(True, color="#334155", linestyle=":", linewidth=0.8)
            ax.tick_params(colors="#94A3B8")
            ax.xaxis.label.set_color("#94A3B8")
            ax.yaxis.label.set_color("#94A3B8")
            ax.title.set_color("#F1F5F9")
        fig.patch.set_facecolor("#1E293B")

        # Plot L
        ax1.plot(
            t_ex,
            excit,
            label="Excitation Pulse",
            color="#94A3B8",
            alpha=0.5,
            linewidth=1.2,
        )
        ax1.plot(
            t_ex,
            longi,
            label="Received Longitudinal Wave",
            color="#38BDF8",
            linewidth=1.5,
        )
        ax1.set_title("Longitudinal Channel", fontsize=11, fontweight="bold")
        ax1.set_xlabel("Time (µs)")
        ax1.set_ylabel("Signal Amplitude (V)")

        # Overlay Longitudinal vertical lines
        if practice_l_ex > 0:
            ax1.axvline(
                practice_l_ex,
                color="#10B981",
                linestyle="-",
                linewidth=1.8,
                label="Your L Ex Selection",
            )
        if practice_l_rec > 0:
            ax1.axvline(
                practice_l_rec,
                color="#10B981",
                linestyle="--",
                linewidth=1.8,
                label="Your L Rec Selection",
            )
        ax1.legend(loc="lower right")

        # Plot S
        ax2.plot(
            t_ex,
            excit,
            label="Excitation Pulse",
            color="#94A3B8",
            alpha=0.5,
            linewidth=1.2,
        )
        ax2.plot(
            t_ex,
            shear,
            label="Received Shear Wave",
            color="#F43F5E",
            linewidth=1.5,
        )
        ax2.set_title("Shear Channel", fontsize=11, fontweight="bold")
        ax2.set_xlabel("Time (µs)")
        ax2.set_ylabel("Signal Amplitude (V)")

        # Overlay Shear vertical lines
        if practice_s_ex > 0:
            ax2.axvline(
                practice_s_ex,
                color="#f59e0b",
                linestyle="-",
                linewidth=1.8,
                label="Your S Ex Selection",
            )
        if practice_s_rec > 0:
            ax2.axvline(
                practice_s_rec,
                color="#f59e0b",
                linestyle="--",
                linewidth=1.8,
                label="Your S Rec Selection",
            )
        ax2.legend(loc="lower right")

        plt.subplots_adjust(hspace=0.45)
        st.pyplot(fig)
        plt.close(fig)

    # Work area: Peak Picker & Time Delay displays (st.number_input with arrows)
    st.markdown("### ✍️ Peak Picker Controls")

    with st.container(border=True):
        col_ctrl1, col_ctrl2 = st.columns(2, gap="medium")

        with col_ctrl1:
            st.markdown("##### 🔵 Longitudinal Wave Selections")
            practice_l_ex = st.number_input(
                "Longitudinal Excitation Peak Selection (µs):",
                min_value=0.0,
                max_value=40.0,
                step=0.1,
                format="%.2f",
                key="prac_l_ex_num",
            )

            practice_l_rec = st.number_input(
                "Longitudinal Receiver Peak Selection (µs):",
                min_value=0.0,
                max_value=40.0,
                step=0.1,
                format="%.2f",
                key="prac_l_rec_num",
            )

            dt_l = practice_l_rec - practice_l_ex
            st.markdown(
                f"<div style='background-color: rgba(56, 189, 248, 0.1); border-left: 3px solid #38bdf8; padding: 0.5rem; margin-top: 0.5rem; border-radius: 4px; font-weight: 600; color: #0284c7;'>"
                f"Longitudinal Delay (Δt<sub>L</sub>): {dt_l:.2f} µs"
                f"</div>",
                unsafe_allow_html=True,
            )

        with col_ctrl2:
            st.markdown("##### 🔴 Shear Wave Selections")
            practice_s_ex = st.number_input(
                "Shear Excitation Peak Selection (µs):",
                min_value=0.0,
                max_value=40.0,
                step=0.1,
                format="%.2f",
                key="prac_s_ex_num",
            )

            practice_s_rec = st.number_input(
                "Shear Receiver Peak Selection (µs):",
                min_value=0.0,
                max_value=40.0,
                step=0.1,
                format="%.2f",
                key="prac_s_rec_num",
            )

            dt_s = practice_s_rec - practice_s_ex
            st.markdown(
                f"<div style='background-color: rgba(244, 63, 94, 0.1); border-left: 3px solid #f43f5e; padding: 0.5rem; margin-top: 0.5rem; border-radius: 4px; font-weight: 600; color: #be123c;'>"
                f"Shear Delay (Δt<sub>S</sub>): {dt_s:.2f} µs"
                f"</div>",
                unsafe_allow_html=True,
            )

        # Update real-time state for grading submit sync
        st.session_state["practice_inputs"] = {
            "l_ex": practice_l_ex,
            "l_rec": practice_l_rec,
            "s_ex": practice_s_ex,
            "s_rec": practice_s_rec,
        }

        # Button to submit
        p_submit = st.button("📝 Submit Practice Attempt", use_container_width=True)

    # Trigger grading
    if p_submit:
        with st.spinner("Grading..."):
            try:
                r = Runner(
                    app=adk_app,
                    session_service=st.session_state["session_service"],
                    auto_create_session=True,
                )
                prompt = f"practice material={p_material} thickness={p_thickness}mm frequency={p_frequency}kHz l_ex={practice_l_ex} l_rec={practice_l_rec} s_ex={practice_s_ex} s_rec={practice_s_rec}"
                content = types.Content(parts=[types.Part.from_text(text=prompt)])

                prev_history = st.session_state.get("practice_history", [])
                try:
                    r.session_service.delete_session_sync(
                        app_name="wave_tutor",
                        user_id="streamlit-user",
                        session_id="streamlit-practice-session",
                    )
                except Exception:
                    pass

                r.session_service.create_session_sync(
                    app_name="wave_tutor",
                    user_id="streamlit-user",
                    session_id="streamlit-practice-session",
                    state={
                        "error_history": prev_history,
                        "material": p_material,
                        "thickness_mm": p_thickness,
                        "frequency_khz": p_frequency,
                    },
                )

                events = run_agent_with_isolation(r, prompt, "streamlit-practice-session")

                if events:
                    st.session_state["practice_output"] = events[-1].output
                    try:
                        session = r.session_service.get_session_sync(
                            app_name="wave_tutor",
                            user_id="streamlit-user",
                            session_id="streamlit-practice-session",
                        )
                        if session:
                            st.session_state["practice_history"] = session.state.get(
                                "error_history", []
                            )
                            st.session_state["practice_last_status"] = (
                                session.state.get("session_status", "UNKNOWN")
                            )
                            st.session_state["practice_last_inputs"] = {
                                "l_ex": practice_l_ex,
                                "l_rec": practice_l_rec,
                                "s_ex": practice_s_ex,
                                "s_rec": practice_s_rec,
                            }
                            if session.state.get("session_status") == "PASS":
                                st.session_state["practice_resolved_l_ex"] = (
                                    session.state.get("selected_l_ex_peak_us", 0.0)
                                )
                                st.session_state["practice_resolved_l_rec"] = (
                                    session.state.get("selected_l_rec_peak_us", 0.0)
                                )
                                st.session_state["practice_resolved_s_ex"] = (
                                    session.state.get("selected_s_ex_peak_us", 0.0)
                                )
                                st.session_state["practice_resolved_s_rec"] = (
                                    session.state.get("selected_s_rec_peak_us", 0.0)
                                )
                            else:
                                st.session_state["practice_resolved_l_ex"] = 0.0
                                st.session_state["practice_resolved_l_rec"] = 0.0
                                st.session_state["practice_resolved_s_ex"] = 0.0
                                st.session_state["practice_resolved_s_rec"] = 0.0
                            st.rerun()
                    except Exception:
                        pass
            except Exception:
                # Local fallback
                ex_peaks = find_signal_peaks(info["excit_file"])
                rec_l_peaks = find_signal_peaks(info["longi_file"])
                rec_s_peaks = find_signal_peaks(info["shear_file"])

                thickness_m = p_thickness / 1000.0
                l_diff = practice_l_rec - practice_l_ex
                s_diff = practice_s_rec - practice_s_ex
                p_cL = thickness_m / (l_diff * 1e-6) if l_diff > 0 else 0.0
                p_cS = thickness_m / (s_diff * 1e-6) if s_diff > 0 else 0.0

                status = "PASS"
                error_message = ""
                expected_cL = METALS_DATABASE[p_material.lower()]["cL"]
                expected_cS = METALS_DATABASE[p_material.lower()]["cS"]

                if abs(practice_l_ex - 1.567) < 0.1 or abs(practice_s_ex - 1.567) < 0.1:
                    status = "REJECT_CAT2"
                    error_message = "Selected startup ringing noise."
                elif p_cS > (expected_cL * 0.9):
                    status = "REJECT_CAT1"
                    error_message = "Selected longitudinal leakage."
                elif (
                    abs(l_diff - info["longitudinal_delay_us"]) > 0.05
                    or abs(s_diff - info["shear_delay_us"]) > 0.05
                ):
                    status = "REJECT_CYCLE"
                    error_message = "Selected non-corresponding peaks."
                elif p_cL < 1000 or p_cL > 8000 or p_cS < 500 or p_cS > 5000:
                    status = "REJECT_BOUNDS"
                    error_message = "Velocities out of typical range."

                attempt_num = len(st.session_state.get("practice_history", [])) + 1
                prev_history = st.session_state.get("practice_history", [])
                success_msg = "Longitudinal and shear acoustic wave velocities match corresponding features, and the calculated dynamic elastic moduli are within the typical physical range for this material."
                prev_history.append(
                    {
                        "attempt": attempt_num,
                        "error_type": status,
                        "message": error_message if error_message else success_msg,
                    }
                )

                moduli_data = {}
                if status == "PASS" and p_cL > 0 and p_cS > 0:
                    rho = literature_db[p_material.lower()]["density_kg_m3"]
                    G_calc = rho * (p_cS**2) / 1e9
                    nu_calc = (p_cL**2 - 2 * p_cS**2) / (2 * (p_cL**2 - p_cS**2))
                    E_calc = 2 * G_calc * (1 + nu_calc)

                    lit_E = literature_db[p_material.lower()]["youngs_modulus_gpa"]
                    lit_G = literature_db[p_material.lower()]["shear_modulus_gpa"]
                    lit_nu = literature_db[p_material.lower()]["poissons_ratio"]

                    moduli_data = {
                        "calculated_youngs_modulus_gpa": E_calc,
                        "calculated_shear_modulus_gpa": G_calc,
                        "calculated_poissons_ratio": nu_calc,
                        "youngs_error_percent": abs(E_calc - lit_E) / lit_E * 100.0,
                        "shear_error_percent": abs(G_calc - lit_G) / lit_G * 100.0,
                        "poissons_error_percent": abs(nu_calc - lit_nu) / lit_nu * 100.0
                        if lit_nu > 0
                        else 0.0,
                    }

                fallback_output = compile_local_fallback_explanation(
                    material=p_material,
                    thickness=p_thickness,
                    freq=p_frequency,
                    history=prev_history,
                    moduli=moduli_data,
                    cL=p_cL,
                    cS=p_cS,
                    practice_mode=True,
                    l_ex=practice_l_ex,
                    l_rec=practice_l_rec,
                    s_ex=practice_s_ex,
                    s_rec=practice_s_rec,
                )

                st.session_state["practice_output"] = fallback_output
                st.session_state["practice_history"] = prev_history
                st.session_state["practice_last_status"] = status
                st.session_state["practice_last_inputs"] = {
                    "l_ex": practice_l_ex,
                    "l_rec": practice_l_rec,
                    "s_ex": practice_s_ex,
                    "s_rec": practice_s_rec,
                }
                if status == "PASS":
                    st.session_state["practice_resolved_l_ex"] = practice_l_ex
                    st.session_state["practice_resolved_l_rec"] = practice_l_rec
                    st.session_state["practice_resolved_s_ex"] = practice_s_ex
                    st.session_state["practice_resolved_s_rec"] = practice_s_rec
                else:
                    st.session_state["practice_resolved_l_ex"] = 0.0
                    st.session_state["practice_resolved_l_rec"] = 0.0
                    st.session_state["practice_resolved_s_ex"] = 0.0
                    st.session_state["practice_resolved_s_rec"] = 0.0
                st.rerun()

    # Output Feedback & Attempts history
    if (
        st.session_state["practice_history"]
        or st.session_state.get("practice_last_inputs") is not None
    ):
        st.markdown("### 📜 Practice History & Feedback")

        inputs_changed = False
        last_in = st.session_state.get("practice_last_inputs")
        if last_in:
            if (
                abs(practice_l_ex - last_in["l_ex"]) > 1e-4
                or abs(practice_l_rec - last_in["l_rec"]) > 1e-4
                or abs(practice_s_ex - last_in["s_ex"]) > 1e-4
                or abs(practice_s_rec - last_in["s_rec"]) > 1e-4
            ):
                inputs_changed = True

        if inputs_changed:
            st.info(
                "⚠️ **You have adjusted the peak times. Click 'Submit Practice Attempt' to grade your new selection.**"
            )

        last_status = st.session_state["practice_last_status"]
        if last_status == "PASS":
            st.success(
                "✅ **PASS**: Outstanding! Your peak selections and sound speed calculations are 100% physically accurate."
            )
        elif last_status == "REJECT_CAT2":
            st.error(
                "❌ **Incorrect due to Transducer Ringing**: Your speeds are impossibly fast! You selected a peak in the startup ringing region."
            )
        elif last_status == "REJECT_CAT1":
            st.error(
                "❌ **Incorrect due to Mode Overlap**: Your shear wave speed overlaps with longitudinal velocities. You selected the early longitudinal wave leakage."
            )
        elif last_status == "REJECT_CYCLE":
            st.error(
                "❌ **Incorrect due to Cycle Mismatch**: You matched non-corresponding peaks of the waveform cycles."
            )
        elif last_status == "REJECT_BOUNDS":
            st.error(
                "❌ **Incorrect due to Out of Bounds Velocity**: Calculated velocities do not match realistic physical properties of this metal."
            )

        if st.session_state["practice_output"]:
            st.markdown("#### 🎓 Professor Review & Analysis")
            st.markdown(st.session_state["practice_output"])

        st.markdown("##### 📜 All Attempts")
        for att in st.session_state["practice_history"]:
            st.html(render_attempt_card(att))


# Footer
st.markdown("---")
st.markdown(
    "<div class='footer-note'>WaveTutor Educational Module | Powered by ADK Multi-Agent Orchestration & Local Physics Validation</div>",
    unsafe_allow_html=True,
)
