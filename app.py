"""AI Fraud Call and Message Detector - Streamlit Web Application.

A complete hackathon-grade AI system for detecting fraudulent SMS, WhatsApp,
and voice call communications using Baseline ML, DistilBERT Transformers,
Faster-Whisper Speech-to-Text, and Multi-Factor Explainable Risk Scoring.
"""

from typing import Dict, Any, Optional
import io
import json
import logging
from pathlib import Path
import pandas as pd
import streamlit as st

from config import settings
from src.predictor import get_predictor, predict_message
from src.speech_to_text import get_transcriber, transcribe_audio_file
from src.evaluate_models import load_all_metrics, generate_model_comparison
from src.model_selector import get_current_selected_model, set_active_model

# Page Configuration
st.set_page_config(
    page_title="AI Fraud Call & Message Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Modern UI Polish
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .badge-critical {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 0.35rem 0.8rem;
        border-radius: 0.375rem;
        font-weight: 700;
        border: 1px solid #F87171;
    }
    .badge-high {
        background-color: #FFEDD5;
        color: #9A3412;
        padding: 0.35rem 0.8rem;
        border-radius: 0.375rem;
        font-weight: 700;
        border: 1px solid #FB923C;
    }
    .badge-medium {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 0.35rem 0.8rem;
        border-radius: 0.375rem;
        font-weight: 700;
        border: 1px solid #FCD34D;
    }
    .badge-low {
        background-color: #DCFCE7;
        color: #166534;
        padding: 0.35rem 0.8rem;
        border-radius: 0.375rem;
        font-weight: 700;
        border: 1px solid #4ADE80;
    }
    .indicator-card {
        background-color: #F8FAFC;
        border-left: 4px solid #EF4444;
        padding: 0.75rem 1rem;
        margin-bottom: 0.5rem;
        border-radius: 0 0.375rem 0.375rem 0;
    }
    .safe-card {
        background-color: #F0FDF4;
        border-left: 4px solid #22C55E;
        padding: 0.75rem 1rem;
        margin-bottom: 0.5rem;
        border-radius: 0 0.375rem 0.375rem 0;
    }
    .metric-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 0.5rem;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)


# Pre-configured Demo Scenarios
DEMO_SCENARIOS = {
    "-- Select a Quick Demo Scenario --": "",
    "🚨 Fake Bank KYC Deactivation Scam (SBI)": (
        "URGENT: Your SBI Bank account has been temporarily blocked due to missing PAN KYC. "
        "Click http://bit.ly/sbi-kyc-update to update immediately to avoid penalty and restore access."
    ),
    "🎁 Lottery / KBC Lucky Draw Scam": (
        "Dear customer, congratulations! You have won a cash prize of Rs 25,00,000 in KBC Lucky Draw! "
        "Send your bank account details and Rs 500 processing fee to claim via WhatsApp 9876543210."
    ),
    "⚡ Electricity Disconnection Scam (Fear Pressure)": (
        "Dear user, your electricity power will be disconnected tonight at 9:30 PM because previous month "
        "bill was not updated. Immediately call electricity officer at 9123456789."
    ),
    "💸 Reverse UPI Payment / PIN Trap": (
        "You have received a payment request of Rs 2,000 from PayTM merchant. Enter your 6-digit UPI PIN "
        "to receive money in your bank account."
    ),
    "⚠️ Hinglish Account Block Threat": (
        "Aapka bank khata block ho gaya hai. Turant pan card aur aadhaar link karein is link par "
        "http://bank-update-kyc.org nahi to account permanently band ho jayega."
    ),
    "📦 Genuine Amazon Delivery Update": (
        "Your Amazon order #402-8921892 has been shipped via BlueDart. Track package at "
        "https://www.amazon.in/orders. Delivery expected tomorrow by 8 PM."
    ),
    "🔐 Genuine Swiggy Login OTP": (
        "Your OTP for logging into Swiggy is 739218. Valid for 5 minutes. Do not share this OTP with anyone."
    ),
    "☕ Genuine Casual Message": (
        "Hi Rahul, are we still meeting for coffee at Starbucks at 4 PM today? Let me know if you are running late."
    )
}


def render_sidebar():
    """Renders the application sidebar with active model selection and system telemetry."""
    st.sidebar.image("https://img.icons8.com/fluency/96/shield.png", width=64)
    st.sidebar.title("System Control")

    st.sidebar.markdown("---")
    st.sidebar.subheader("🤖 Active AI Model")

    # Load available models
    all_metrics = load_all_metrics()
    selected_info = get_current_selected_model()
    current_key = selected_info.get("active_model_type", "baseline_logistic_regression")

    model_options = {
        "baseline_logistic_regression": "TF-IDF + Logistic Regression (Baseline)",
        "baseline_naive_bayes": "TF-IDF + Naive Bayes (Lightweight)",
        "transformer_distilbert": "DistilBERT (Fine-Tuned Transformer)"
    }

    # Filter to available models
    available_keys = [k for k in model_options.keys() if k in all_metrics or k.startswith("baseline")]
    if not available_keys:
        available_keys = ["baseline_logistic_regression"]

    default_idx = available_keys.index(current_key) if current_key in available_keys else 0

    chosen_model_key = st.sidebar.selectbox(
        "Select Active Classifier:",
        options=available_keys,
        index=default_idx,
        format_func=lambda k: model_options.get(k, k)
    )

    if chosen_model_key != current_key:
        set_active_model(chosen_model_key)
        get_predictor(force_reload=True)
        st.sidebar.success(f"Switched to {model_options.get(chosen_model_key)}")
        st.rerun()

    st.sidebar.markdown(f"**Hardware Device:** `{settings.DEVICE.upper()}`")
    st.sidebar.markdown(f"**Speech Engine:** `Faster-Whisper ({settings.WHISPER_DEFAULT_MODEL_SIZE})`")
    st.sidebar.markdown(f"**Classification Threshold:** `{settings.CLASSIFICATION_THRESHOLD:.2f}`")

    st.sidebar.markdown("---")
    st.sidebar.subheader("🔒 Privacy Guarantee")
    st.sidebar.caption(
        "All speech recognition, NLP processing, and model inference run **100% locally**. "
        "No paid cloud APIs or user data are transmitted to external servers."
    )

    st.sidebar.markdown("---")
    st.sidebar.caption("AI Fraud Call & Message Detector • Hackathon Edition")


def render_risk_badge(risk_level: str) -> str:
    """Returns HTML badge for risk level."""
    classes = {
        "CRITICAL": "badge-critical",
        "HIGH": "badge-high",
        "MEDIUM": "badge-medium",
        "LOW": "badge-low"
    }
    cls = classes.get(risk_level, "badge-low")
    return f"<span class='{cls}'>{risk_level} RISK</span>"


def display_analysis_results(result: Dict[str, Any]):
    """Renders formatted analysis results with risk engine scores and explainability cards."""
    st.markdown("### 📊 Detection & Risk Report")

    pred = result["prediction"]
    conf = result["confidence_percentage"]
    risk = result["risk_level"]
    score = result["risk_score"]
    model_name = result["active_model"]

    # Top Metric Columns
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if pred == "FRAUD":
            st.error(f"### 🚨 {pred}")
        else:
            st.success(f"### ✅ {pred}")
        st.caption("AI Classification")

    with col2:
        st.metric(label="AI Confidence", value=f"{conf:.1f}%")
        st.progress(min(max(conf / 100.0, 0.0), 1.0))

    with col3:
        st.markdown(f"<div style='padding-top:12px;'>{render_risk_badge(risk)}</div>", unsafe_allow_html=True)
        st.caption(f"Risk Score: {score * 100:.1f} / 100")

    with col4:
        st.metric(label="Active Model", value=model_name.split()[0])
        st.caption(model_name)

    st.markdown("---")

    # User Safety Action Box
    if pred == "FRAUD" or risk in ["HIGH", "CRITICAL"]:
        st.error(f"**Action Recommendation:** {result['recommended_action']}")
    elif risk == "MEDIUM":
        st.warning(f"**Action Recommendation:** {result['recommended_action']}")
    else:
        st.success(f"**Action Recommendation:** {result['recommended_action']}")

    # Explainability & Indicators Section
    exp = result.get("explanation", {})
    indicators = exp.get("detected_indicators", [])

    st.markdown("#### 🔍 Explainable Evidence & Red Flags")
    st.info(f"**Analysis Summary:** {exp.get('summary', 'No summary available.')}")

    if indicators:
        for idx, ind in enumerate(indicators, 1):
            severity_color = "🔴" if ind["severity"] in ["CRITICAL", "HIGH"] else "🟡"
            with st.container():
                st.markdown(f"""
                <div class="indicator-card">
                    <strong>{severity_color} [{ind['severity']}] {ind['category']}</strong>: {ind['description']}<br>
                    <small><b>Detected Phrases:</b> <code>{', '.join(ind['matches'])}</code></small><br>
                    <span style="color:#475569; font-size:0.9rem;">💡 <b>Safety Tip:</b> {ind['advice']}</span>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="safe-card">
            ✅ <b>Clean Communication:</b> No known fraud patterns, deceptive URLs, urgency threats, or credential solicitation detected.
        </div>
        """, unsafe_allow_html=True)

    # Technical Details Expander
    with st.expander("🛠️ Technical Inspection Details"):
        st.json({
            "Input Text": result["input_text"],
            "Cleaned / Tokenized Text": result["clean_text"],
            "Fraud Probability": result["fraud_probability"],
            "Synthesized Risk Score": result["risk_score"],
            "Risk Level": result["risk_level"],
            "Reasoning": result["reasoning"],
            "Indicator Count": result["indicator_count"]
        })


def tab_dashboard():
    """Renders the Home/Dashboard overview section."""
    st.markdown("<div class='main-header'>🛡️ AI Fraud Call & Message Detector</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='sub-header'>A unified multimodal AI system designed to detect phishing SMS, "
        "financial fraud messages, and scam voice calls in real time.</div>",
        unsafe_allow_html=True
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3 style="color:#2563EB; margin:0;">Unified Core</h3>
            <p style="color:#64748B; margin:0;">Text & Speech Pipeline</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3 style="color:#16A34A; margin:0;">100% Local</h3>
            <p style="color:#64748B; margin:0;">Privacy Preserving</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3 style="color:#D97706; margin:0;">Multi-Factor</h3>
            <p style="color:#64748B; margin:0;">AI + Risk Engine</p>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class="metric-card">
            <h3 style="color:#DC2626; margin:0;">Explainable AI</h3>
            <p style="color:#64748B; margin:0;">Transparent Red Flags</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    col_left, col_right = st.columns([3, 2])
    with col_left:
        st.markdown("### 📌 Problem Statement")
        st.write(
            "Telecommunication fraud—including bank impersonation SMS, fake KYC deactivations, "
            "UPI refund traps, lottery scams, and voice vishing calls—causes billions in financial losses annually. "
            "Most traditional spam filters rely only on simple keyword blacklists, missing emerging scam phrasing, "
            "Hinglish multilingual fraud, and deceptive voice calls."
        )

        st.markdown("### 💡 The Solution")
        st.write(
            "Our solution creates an **end-to-end protective shield**: "
            "incoming audio calls are transcribed locally via **Faster-Whisper**, converted into text transcripts, "
            "and evaluated by an **active AI classifier** (TF-IDF + Logistic Regression / Fine-Tuned DistilBERT). "
            "A **multi-factor risk engine** combines machine learning confidence with behavioral indicator analysis "
            "to output calibrated risk levels (`LOW`, `MEDIUM`, `HIGH`, `CRITICAL`) and transparent explanations."
        )

    with col_right:
        st.markdown("### ⚙️ System Status")
        selected = get_current_selected_model()
        st.success(f"**Active Model:** {selected.get('model_name', 'Baseline Logistic Regression')}")
        st.info(f"**Whisper ASR Engine:** Faster-Whisper `{settings.WHISPER_DEFAULT_MODEL_SIZE}` on `{settings.DEVICE.upper()}`")
        st.warning("**Multilingual Support:** English, Hindi & Hinglish Token Preservation")
        st.caption("Ready for real-time analysis in the tabs above ⬆️")


def tab_message_analysis():
    """Renders the text message analysis tab with interactive input and quick-load demo scenarios."""
    st.markdown("### 💬 Message Analysis & Fraud Detection")
    st.write("Analyze incoming SMS, WhatsApp messages, emails, or paste text to evaluate fraud probability.")

    # Demo Scenario Quick Loader
    selected_demo = st.selectbox(
        "💡 Quick-Load Hackathon Demo Scenario:",
        options=list(DEMO_SCENARIOS.keys()),
        index=0
    )

    demo_text = DEMO_SCENARIOS.get(selected_demo, "")

    # Input Text Area
    user_input = st.text_area(
        "Enter or paste the message to analyze:",
        value=demo_text,
        height=140,
        placeholder="e.g. URGENT: Your bank account is blocked. Update KYC at http://bit.ly/bank-kyc..."
    )

    col1, col2 = st.columns([1, 4])
    with col1:
        analyze_clicked = st.button("🔍 Analyze Message", type="primary", use_container_width=True)
    with col2:
        if st.button("🧹 Clear Input", use_container_width=False):
            st.rerun()

    if analyze_clicked or (user_input and selected_demo != "-- Select a Quick Demo Scenario --"):
        if not user_input.strip():
            st.warning("⚠️ Please enter or select a message before analyzing.")
            return

        with st.spinner("Analyzing message with AI model and risk engine..."):
            try:
                result = predict_message(user_input)
                display_analysis_results(result)
            except Exception as err:
                st.error(f"❌ Error during message analysis: {err}")
                st.caption("Ensure baseline model has been trained via `python3 src/train_baseline.py`.")


def tab_audio_analysis():
    """Renders the Call / Audio analysis tab with upload, transcription, and fraud evaluation."""
    st.markdown("### 📞 Call & Voice Audio Analysis")
    st.write(
        "Upload a call recording or voicemail (`.wav`, `.mp3`, `.m4a`, `.ogg`) to transcribe speech locally "
        "via **Faster-Whisper** and evaluate it using the fraud detection pipeline."
    )

    # Audio input method tabs
    upload_tab, mic_tab, sample_tab = st.tabs(["📁 Upload Audio File", "🎙️ Microphone Input", "🎧 Pre-Recorded Demo Audios"])

    audio_bytes_to_process = None
    source_description = ""

    with upload_tab:
        uploaded_file = st.file_uploader(
            "Upload Call Audio File (WAV, MP3, M4A, OGG):",
            type=["wav", "mp3", "m4a", "ogg", "flac"]
        )
        if uploaded_file is not None:
            audio_bytes_to_process = uploaded_file.read()
            source_description = f"Uploaded File: `{uploaded_file.name}`"
            st.audio(audio_bytes_to_process)

    with mic_tab:
        st.info("💡 You can record a message using your microphone below (supported in Streamlit).")
        try:
            mic_audio = st.audio_input("Record voice message:")
            if mic_audio is not None:
                audio_bytes_to_process = mic_audio.read()
                source_description = "Microphone Recording"
                st.audio(audio_bytes_to_process)
        except Exception:
            st.caption("Microphone input widget requires Streamlit >= 1.39. Use audio upload as primary reliable method.")

    with sample_tab:
        st.write("Select a simulated scam/genuine voice scenario to test transcription and detection:")
        sample_choice = st.selectbox(
            "Select Scenario Transcript for Audio Simulation:",
            [
                "-- Select a Voice Scenario --",
                "📞 Scam Call: Police Cyber Cell Arrest Threat & Fine",
                "📞 Scam Call: Bank Manager KYC Suspension & OTP Demand",
                "📞 Genuine Call: Doctor Clinic Appointment Confirmation",
                "📞 Genuine Call: Courier Delivery Rider Calling for Directions"
            ]
        )

        sample_transcripts = {
            "📞 Scam Call: Police Cyber Cell Arrest Threat & Fine": (
                "Hello, I am calling from Mumbai Police Cyber Crime Cell. Your IP address and Aadhaar number were "
                "found involved in illegal money laundering. An arrest warrant has been issued in your name. "
                "To cancel the warrant, you must immediately transfer 15000 rupees penalty via UPI to our settlement desk."
            ),
            "📞 Scam Call: Bank Manager KYC Suspension & OTP Demand": (
                "Hello sir, I am Senior Manager Sharma from SBI Head Office. Your ATM debit card and net banking "
                "are getting blocked today because your KYC document is incomplete. I am sending a 6-digit verification code "
                "to your phone right now. Please tell me the OTP immediately so I can unblock your account."
            ),
            "📞 Genuine Call: Doctor Clinic Appointment Confirmation": (
                "Hello, this is Apollo Clinic calling to confirm your appointment with Dr. Sharma tomorrow at 11:30 AM. "
                "Please arrive 10 minutes early and carry your previous medical prescription. Have a good day."
            ),
            "📞 Genuine Call: Courier Delivery Rider Calling for Directions": (
                "Hello sir, I am BlueDart delivery agent. I have reached near your apartment gate. "
                "Could you please confirm if your flat is on the third floor? I will deliver the package now."
            )
        }

        if sample_choice in sample_transcripts:
            simulated_text = sample_transcripts[sample_choice]
            st.text_area("Simulated Voice Call Content:", value=simulated_text, height=100, disabled=True)
            if st.button("🚀 Analyze Simulated Call Transcript", type="primary"):
                with st.spinner("Analyzing simulated call transcript..."):
                    result = predict_message(simulated_text)
                    display_analysis_results(result)

    # Process Uploaded/Recorded Audio
    if audio_bytes_to_process is not None:
        st.markdown("---")
        if st.button("🚀 Transcribe Audio & Detect Fraud", type="primary"):
            st.info(f"Processing {source_description}...")

            progress_bar = st.progress(0.2)
            status_text = st.empty()

            try:
                status_text.text("1/3 Loading Faster-Whisper Speech-to-Text engine...")
                transcriber = get_transcriber()
                progress_bar.progress(0.5)

                status_text.text("2/3 Transcribing audio content...")
                trans_result = transcriber.transcribe(audio_bytes_to_process)
                progress_bar.progress(0.8)

                status_text.text("3/3 Running NLP Fraud Detection & Risk Analysis...")
                transcript = trans_result.get("transcript", "").strip()

                if not transcript:
                    progress_bar.progress(1.0)
                    st.warning("⚠️ No speech could be transcribed from the audio file. Please check audio quality.")
                    return

                # Display transcript
                st.markdown("#### 📝 Extracted Speech Transcript")
                st.success(f'"{transcript}"')

                col_t1, col_t2, col_t3 = st.columns(3)
                with col_t1:
                    st.caption(f"Detected Language: `{trans_result.get('language', 'en').upper()}`")
                with col_t2:
                    st.caption(f"Language Confidence: `{trans_result.get('language_probability', 1.0)*100:.1f}%`")
                with col_t3:
                    st.caption(f"Audio Duration: `{trans_result.get('duration', 0.0):.1f}s`")

                # Pass transcript to AI detection engine
                result = predict_message(transcript)
                progress_bar.progress(1.0)
                status_text.empty()

                display_analysis_results(result)

            except Exception as err:
                progress_bar.empty()
                status_text.empty()
                st.error(f"❌ Error processing audio: {err}")
                st.caption("Ensure `faster-whisper` and audio dependencies are properly configured.")


def tab_model_performance():
    """Renders the Model Performance and Benchmark Comparison section."""
    st.markdown("### 📈 Model Performance & Benchmark Evaluation")
    st.write(
        "Comparison of baseline machine learning models vs. fine-tuned deep learning transformer models "
        "on the held-out test dataset."
    )

    all_metrics = load_all_metrics()
    if not all_metrics:
        st.warning("⚠️ No evaluated metrics found in `models/metrics.json`. Please train baseline models first.")
        if st.button("🚀 Train Baseline Models Now"):
            with st.spinner("Training baseline TF-IDF models..."):
                from src.train_baseline import train_and_evaluate_baseline
                train_and_evaluate_baseline()
                st.success("Baseline training completed!")
                st.rerun()
        return

    # Benchmark Comparison Table
    comparison = generate_model_comparison()
    models = comparison.get("models", [])

    if models:
        df_comp = pd.DataFrame(models)
        display_df = pd.DataFrame({
            "Model Name": df_comp["name"],
            "Accuracy": df_comp["accuracy"].apply(lambda v: f"{v*100:.2f}%"),
            "Precision": df_comp["precision"].apply(lambda v: f"{v*100:.2f}%"),
            "Recall (Fraud)": df_comp["recall"].apply(lambda v: f"{v*100:.2f}%"),
            "F1-Score": df_comp["f1_score"].apply(lambda v: f"{v*100:.2f}%"),
            "ROC-AUC": df_comp["roc_auc"].apply(lambda v: f"{v:.4f}"),
            "Test Samples": df_comp["test_samples"]
        })

        st.dataframe(display_df, use_container_width=True)
        st.caption(
            "💡 **Key Metric:** **Recall** is prioritized over raw accuracy because failing to catch a fraudulent "
            "communication (False Negative) can lead to direct financial loss, whereas a false alarm can be easily reviewed."
        )

    st.markdown("---")

    # Confusion Matrix Visualization
    st.markdown("#### 🧩 Confusion Matrix Breakdown")
    col_m1, col_m2 = st.columns(2)

    for idx, (m_key, m_data) in enumerate(all_metrics.items()):
        target_col = col_m1 if idx % 2 == 0 else col_m2
        with target_col:
            st.markdown(f"**{m_data.get('model_name', m_key)}**")
            cm = m_data.get("confusion_matrix", [[0, 0], [0, 0]])
            if len(cm) == 2 and len(cm[0]) == 2:
                tn, fp = cm[0][0], cm[0][1]
                fn, tp = cm[1][0], cm[1][1]

                cm_df = pd.DataFrame(
                    [[f"TN: {tn}", f"FP: {fp}"],
                     [f"FN: {fn}", f"TP: {tp}"]],
                    index=["Actual Genuine (0)", "Actual Fraud (1)"],
                    columns=["Pred Genuine (0)", "Pred Fraud (1)"]
                )
                st.table(cm_df)

    st.markdown("---")

    # Dataset Statistics
    st.markdown("#### 📊 Dataset & Split Statistics")
    if settings.PROCESSED_DATA_PATH.exists():
        from src.data_preprocessor import get_dataset_statistics
        df_clean = pd.read_csv(settings.PROCESSED_DATA_PATH)
        stats = get_dataset_statistics(df_clean)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Clean Samples", stats["total_samples"])
        c2.metric("Fraud Count", stats["fraud_count"], f"{stats['fraud_percentage']}%")
        c3.metric("Genuine Count", stats["genuine_count"], f"{stats['genuine_percentage']}%")
        c4.metric("Avg Words / Message", stats["avg_word_count"])


def tab_architecture():
    """Renders the System Architecture and Workflow explanation section."""
    st.markdown("### 🏗️ System Architecture & Workflow")
    st.write("An overview of the multimodal fraud detection pipeline, components, and privacy architecture.")

    st.markdown("""
    #### 🔄 Dual-Channel Ingestion & Processing Flow
    
    ```
    ┌─────────────────────────┐         ┌─────────────────────────┐
    │   Text Message Input    │         │  Call Audio File / Mic  │
    └────────────┬────────────┘         └────────────┬────────────┘
                 │                                   │
                 │                                   ▼
                 │                      ┌─────────────────────────┐
                 │                      │ Faster-Whisper ASR (v3) │
                 │                      └────────────┬────────────┘
                 │                                   │ (Transcript)
                 ▼                                   ▼
    ┌─────────────────────────────────────────────────────────────┐
    │ Safe Preprocessor (Preserves URLs, OTP, Phone, Hinglish)    │
    └──────────────────────────────┬──────────────────────────────┘
                                   │
                                   ▼
    ┌─────────────────────────────────────────────────────────────┐
    │ Active AI Classifier (TF-IDF+LR / Fine-Tuned DistilBERT)    │
    └──────────────────────────────┬──────────────────────────────┘
                                   │ Probability Score
                                   ▼
    ┌─────────────────────────────────────────────────────────────┐
    │ Explainable AI Engine (Pattern & Red Flag Extractor)        │
    └──────────────────────────────┬──────────────────────────────┘
                                   │ Indicators & Severities
                                   ▼
    ┌─────────────────────────────────────────────────────────────┐
    │ Multi-Factor Risk Engine (LOW / MEDIUM / HIGH / CRITICAL)   │
    └──────────────────────────────┬──────────────────────────────┘
                                   │
                                   ▼
    ┌─────────────────────────────────────────────────────────────┐
    │ Interactive Streamlit Dashboard (Evidence, Actions, Gauges) │
    └─────────────────────────────────────────────────────────────┘
    ```
    """)

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 🌟 Core Architectural Features")
        st.markdown("""
        - **Decoupled NLP Intelligence**: The core classifier is completely decoupled from the input medium, allowing text, voice audio, emails, and future telephony feeds to share one unified brain.
        - **Lightweight On-Device ASR**: Faster-Whisper runs on CTranslate2, delivering 4x faster transcription without cloud API costs or privacy vulnerabilities.
        - **Explainable by Design**: Rather than a black-box probability, the system supplies human-understandable red flags, matched phrases, and safety advice.
        - **Multi-Factor Risk Engine**: AI probability is fused with heuristic indicator severities to ensure critical threats (like reverse UPI scams) are never overlooked.
        """)

    with col2:
        st.markdown("#### 🚀 Future Production Roadmap")
        st.markdown("""
        - **FastAPI Microservice**: Core logic in `src/` is built without Streamlit dependencies, enabling a drop-in FastAPI REST/WebSocket server.
        - **Android Telephony Integration**: Real-time incoming call audio streaming via Android Accessibility/Telecom APIs.
        - **Multilingual DistilBERT**: Expanding to multi-lingual models (`xlm-roberta` or `indic-bert`) for deep regional language support.
        - **Federated On-Device Learning**: Privacy-preserving model updates on consumer mobile devices.
        """)


def main():
    """Main application entry point rendering top navigation tabs and content."""
    render_sidebar()

    # Top Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🏠 Dashboard",
        "💬 Message Analysis",
        "📞 Call / Audio Analysis",
        "📈 Model Performance",
        "🏗️ System Architecture"
    ])

    with tab1:
        tab_dashboard()
    with tab2:
        tab_message_analysis()
    with tab3:
        tab_audio_analysis()
    with tab4:
        tab_model_performance()
    with tab5:
        tab_architecture()


if __name__ == "__main__":
    main()

