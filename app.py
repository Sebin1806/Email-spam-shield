import streamlit as st
import joblib
import json
import os
import re
import pandas as pd
import numpy as np

# Set page configuration
st.set_page_config(
    page_title="Email Spam Shield AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern UI design
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    * {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 24px 30px;
        margin-bottom: 24px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
    }
    
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 6px;
    }
    
    .subtitle {
        color: #94a3b8;
        font-size: 1.05rem;
    }
    
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 18px 20px;
        text-align: center;
        backdrop-filter: blur(10px);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: rgba(99, 102, 241, 0.4);
    }
    
    .metric-val {
        font-size: 1.8rem;
        font-weight: 700;
        color: #f8fafc;
    }
    
    .metric-lbl {
        font-size: 0.85rem;
        font-weight: 600;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 4px;
    }
    
    .verdict-box {
        border-radius: 16px;
        padding: 24px 28px;
        margin: 20px 0;
        animation: fadeIn 0.4s ease-in-out;
    }
    
    .verdict-spam {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.15), rgba(185, 28, 28, 0.25));
        border: 1px solid rgba(239, 68, 68, 0.4);
    }
    
    .verdict-ham {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15), rgba(5, 150, 105, 0.25));
        border: 1px solid rgba(16, 185, 129, 0.4);
    }
    
    .verdict-title {
        font-size: 1.5rem;
        font-weight: 800;
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    
    .badge-pill {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    
    .badge-red {
        background: rgba(239, 68, 68, 0.25);
        color: #fca5a5;
        border: 1px solid rgba(239, 68, 68, 0.4);
    }
    
    .badge-green {
        background: rgba(16, 185, 129, 0.25);
        color: #6ee7b7;
        border: 1px solid rgba(16, 185, 129, 0.4);
    }
    
    .badge-blue {
        background: rgba(59, 130, 246, 0.25);
        color: #93c5fd;
        border: 1px solid rgba(59, 130, 246, 0.4);
    }
    
    .indicator-tag {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 6px 14px;
        border-radius: 8px;
        font-size: 0.88rem;
        margin: 4px;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(6px); }
        to { opacity: 1; transform: translateY(0); }
    }
</style>
""", unsafe_allow_html=True)

# Preprocessing logic identical to training pipeline
def preprocess_text(text):
    if not isinstance(text, str) or not text.strip():
        return ""
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', ' http_url ', text)
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', ' email_addr ', text)
    text = re.sub(r'[\$£€¥₹]', ' currency_sym ', text)
    text = re.sub(r'\b\d{10,}\b', ' phone_num ', text)
    text = re.sub(r'\b\d{3,6}\b', ' short_code ', text)
    text = re.sub(r'\b\d+\b', ' num_val ', text)
    text = re.sub(r'!+', ' excl_mark ', text)
    text = re.sub(r'\?+', ' quest_mark ', text)
    text = re.sub(r'[^a-zA-Z0-9_\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_indicators(raw_text):
    """Detect specific spam patterns for interpretability."""
    indicators = []
    if re.search(r'[\$£€¥₹]', raw_text) or re.search(r'\b(cash|prize|reward|winner|won|dollar|pounds|million|claim)\b', raw_text, re.IGNORECASE):
        indicators.append(("💰 Monetary / Prize Signals", "Contains currency signs or reward terms"))
    if re.search(r'https?://\S+|www\.\S+|\.com|\.ly|\.gl', raw_text, re.IGNORECASE):
        indicators.append(("🔗 External Links / URLs", "Contains embedded hyperlinks or shortened URLs"))
    if re.search(r'\b\d{10,}\b|\b\d{4,6}\b', raw_text):
        indicators.append(("📱 Phone / Shortcode", "Contains telephone digits or SMS shortcodes"))
    if re.search(r'!{2,}|\?{2,}', raw_text) or raw_text.count('!') >= 2:
        indicators.append(("⚡ Urgency Punctuation", "Repeated exclamation or question marks"))
    if re.search(r'\b(urgent|action required|verify|password|account suspended|bank|expire|immediately)\b', raw_text, re.IGNORECASE):
        indicators.append(("🚨 High-Urgency Call to Action", "Pressure or threat tactics detected"))
    if re.search(r'\b(free|100% free|guaranteed|congratulations|click here|subscribe|opt out|reply win)\b', raw_text, re.IGNORECASE):
        indicators.append(("🎁 Promotional / Incentive Bait", "Promotional trigger phrases detected"))
    return indicators

# Load Model & Vectorizer
@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load('spam_model.pkl')
        vectorizer = joblib.load('vectorizer.pkl')
        
        metrics = {}
        if os.path.exists('model_metrics.json'):
            with open('model_metrics.json', 'r', encoding='utf-8') as f:
                metrics = json.load(f)
        return model, vectorizer, metrics, None
    except Exception as e:
        return None, None, {}, str(e)

model, vectorizer, metrics, error = load_artifacts()

# Sidebar Configuration
with st.sidebar:
    st.markdown("### ⚙️ Engine Settings")
    
    threshold = st.slider(
        "Spam Decision Threshold",
        min_value=0.20,
        max_value=0.80,
        value=0.50,
        step=0.05,
        help="Adjust the probability cutoff for classifying messages as Spam."
    )
    
    st.markdown("---")
    st.markdown("### 📊 Model Status")
    if model is not None:
        st.markdown('<span class="badge-pill badge-green">● Active & Online</span>', unsafe_allow_html=True)
        st.write(f"**Architecture:** Calibrated Ensemble (LinearSVC + LogReg + ComplementNB + ExtraTrees)")
        st.write(f"**Feature Set:** Word (1-3) & Char (3-5) N-Grams")
        if metrics:
            st.write(f"**Accuracy:** `{metrics.get('accuracy', 0.9893) * 100:.2f}%`")
            st.write(f"**Spam Recall:** `{metrics.get('recall', 0.9297) * 100:.2f}%`")
    else:
        st.markdown('<span class="badge-pill badge-red">● Model Missing</span>', unsafe_allow_html=True)
        st.error(f"Error loading model: {error}")
        st.info("Run `python Email_spam.py` to train and build the model.")

    st.markdown("---")
    st.markdown("### 🛡️ About Email Spam Shield")
    st.caption("A high-precision machine learning system engineered to detect spam, phishing attacks, and promotional scams with ultra-low false-positive rates.")

# Main Header Banner
st.markdown("""
<div class="main-header">
    <div class="main-title">🛡️ Email Spam Shield AI</div>
    <div class="subtitle">Next-generation dual N-Gram text classification engine with calibrated ensemble inference.</div>
</div>
""", unsafe_allow_html=True)

if error:
    st.error(f"⚠️ Could not load the model files (`spam_model.pkl`, `vectorizer.pkl`). Please make sure `Email_spam.py` has been executed. Error: {error}")
    st.stop()

# Tab Navigation
tab_inspect, tab_batch, tab_analytics = st.tabs([
    "🔍 Single Message Inspector",
    "📂 Batch File Scanner",
    "📈 Model Health & Metrics"
])

# ==========================================
# TAB 1: SINGLE MESSAGE INSPECTION
# ==========================================
with tab_inspect:
    st.markdown("#### Test or Paste Any Email / SMS Message")
    
    # Preset sample selector
    sample_presets = {
        "Custom Message": "",
        "🚨 Spam Example 1 (Lottery Prize)": "CONGRATULATIONS! You have won a £1,000,000 cash prize in our international lottery! Claim your money now by calling +447912345678 or reply CLAIM immediately.",
        "🚨 Spam Example 2 (Account Phishing)": "URGENT: Your bank account access has been suspended due to suspicious activity. Click http://secure-bank-login-verify.com to re-verify your identity now or your account will be closed.",
        "🚨 Spam Example 3 (Promo / Free SMS)": "FREE ENTRY: Win a brand new Apple iPhone 15 Pro Max! Text WIN to 80488 right now. T&Cs apply. 150p/msg.",
        "💬 Ham Example 1 (Meeting Invite)": "Hi Alex, hope you are having a productive week. Can we schedule a quick 15-minute sync on Thursday at 2 PM to review the quarterly roadmap?",
        "💬 Ham Example 2 (Casual Chat)": "Hey mate! Are we still on for lunch today at Jurong Point? Let me know once you reach the station."
    }
    
    selected_sample = st.selectbox("💡 Quick Demo Presets:", list(sample_presets.keys()))
    
    # Text area
    default_text = sample_presets[selected_sample] if selected_sample != "Custom Message" else ""
    user_message = st.text_area(
        "Enter message text below:",
        value=default_text,
        height=140,
        placeholder="Paste an email body, subject, or SMS text here to inspect..."
    )
    
    col_btn, col_info = st.columns([1, 4])
    with col_btn:
        analyze_clicked = st.button("🛡️ Scan Message", type="primary", use_container_width=True)
    with col_info:
        if user_message:
            word_count = len(user_message.split())
            char_count = len(user_message)
            st.caption(f"📝 Length: **{char_count}** characters | **{word_count}** words")

    if analyze_clicked or (user_message and selected_sample != "Custom Message"):
        if not user_message.strip():
            st.warning("Please enter some text or select a preset example to analyze.")
        else:
            with st.spinner("Analyzing message vectors..."):
                cleaned = preprocess_text(user_message)
                vec_input = vectorizer.transform([cleaned])
                
                # Predict probabilities
                probs = model.predict_proba(vec_input)[0]
                ham_prob = float(probs[0])
                spam_prob = float(probs[1])
                is_spam = spam_prob >= threshold

            # Verdict Display
            if is_spam:
                st.markdown(f"""
                <div class="verdict-box verdict-spam">
                    <div class="verdict-title" style="color: #ef4444;">
                        🚨 SPAM / PHISHING DETECTED
                        <span class="badge-pill badge-red">Threat Confidence: {spam_prob*100:.1f}%</span>
                    </div>
                    <p style="color: #fca5a5; margin: 0; font-size: 0.95rem;">
                        This message exhibits strong characteristics commonly associated with spam, unsolicited marketing, or malicious phishing attempts.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="verdict-box verdict-ham">
                    <div class="verdict-title" style="color: #10b981;">
                        ✅ LEGITIMATE MESSAGE (HAM)
                        <span class="badge-pill badge-green">Safety Confidence: {ham_prob*100:.1f}%</span>
                    </div>
                    <p style="color: #6ee7b7; margin: 0; font-size: 0.95rem;">
                        This message matches patterns of authentic, normal communication with no critical threats detected.
                    </p>
                </div>
                """, unsafe_allow_html=True)

            # Confidence Gauges
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Spam Probability", f"{spam_prob * 100:.1f}%")
                st.progress(spam_prob)
            with c2:
                st.metric("Ham (Safe) Probability", f"{ham_prob * 100:.1f}%")
                st.progress(ham_prob)
            with c3:
                risk_tier = "🔴 Critical Spam" if spam_prob > 0.75 else "🟠 Suspicious" if spam_prob > 0.45 else "🟡 Low Risk" if spam_prob > 0.20 else "🟢 Clean Ham"
                st.metric("Risk Classification", risk_tier)
                st.caption(f"Decision Threshold: {threshold*100:.0f}%")

            # Threat Indicators Breakdown
            st.markdown("---")
            st.markdown("##### 🔬 Threat Intelligence Breakdown")
            indicators = extract_indicators(user_message)
            
            if indicators:
                cols = st.columns(len(indicators) if len(indicators) <= 3 else 3)
                for idx, (title, desc) in enumerate(indicators):
                    with cols[idx % len(cols)]:
                        st.markdown(f"""
                        <div class="metric-card" style="text-align: left; margin-bottom: 10px;">
                            <div style="font-weight: 700; color: #f87171; font-size: 0.95rem;">{title}</div>
                            <div style="font-size: 0.8rem; color: #94a3b8; margin-top: 4px;">{desc}</div>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.info("ℹ️ No prominent malicious trigger patterns detected (clean vocabulary structure).")

            with st.expander("🛠️ View Normalized NLP Pipeline Tokens"):
                st.code(cleaned, language="text")

# ==========================================
# TAB 2: BATCH SCANNER
# ==========================================
with tab_batch:
    st.markdown("#### 📂 Bulk Message & File Classifier")
    st.write("Scan entire datasets, customer tickets, or email logs simultaneously.")
    
    batch_mode = st.radio("Choose Input Method:", ["Upload CSV File", "Paste Multiple Messages"], horizontal=True)
    
    if batch_mode == "Upload CSV File":
        uploaded_file = st.file_uploader("Upload a CSV containing text messages", type=["csv"])
        if uploaded_file is not None:
            try:
                batch_df = pd.read_csv(uploaded_file)
                text_col = st.selectbox("Select the column containing the text to analyze:", batch_df.columns)
                
                if st.button("🚀 Process Batch CSV", type="primary"):
                    with st.spinner("Processing records..."):
                        texts = batch_df[text_col].fillna("").astype(str)
                        cleaned_texts = [preprocess_text(t) for t in texts]
                        vec_batch = vectorizer.transform(cleaned_texts)
                        probs = model.predict_proba(vec_batch)
                        
                        batch_df['Spam_Probability'] = np.round(probs[:, 1], 4)
                        batch_df['Ham_Probability'] = np.round(probs[:, 0], 4)
                        batch_df['Prediction'] = np.where(probs[:, 1] >= threshold, 'SPAM', 'HAM')
                        
                        spam_total = (batch_df['Prediction'] == 'SPAM').sum()
                        ham_total = (batch_df['Prediction'] == 'HAM').sum()
                        
                        # Summary metrics
                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("Total Processed", len(batch_df))
                        m2.metric("Spam Detected", spam_total, delta=f"{spam_total/len(batch_df)*100:.1f}%", delta_color="inverse")
                        m3.metric("Ham (Safe)", ham_total, delta=f"{ham_total/len(batch_df)*100:.1f}%")
                        m4.metric("Avg Spam Risk", f"{batch_df['Spam_Probability'].mean()*100:.1f}%")
                        
                        st.dataframe(batch_df, use_container_width=True)
                        
                        csv_data = batch_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Download Scanned CSV Results",
                            data=csv_data,
                            file_name="spam_shield_results.csv",
                            mime="text/csv"
                        )
            except Exception as e:
                st.error(f"Error parsing CSV file: {e}")

    else:
        multi_text = st.text_area(
            "Paste multiple messages (one per line):",
            height=180,
            placeholder="URGENT: Call 0800123456 to claim £500\nHey are you free for lunch tomorrow?\nFree entry to win tickets, reply YES"
        )
        if st.button("🚀 Process Multi-line Messages", type="primary"):
            lines = [l.strip() for l in multi_text.strip().split("\n") if l.strip()]
            if not lines:
                st.warning("Please paste at least one message.")
            else:
                with st.spinner("Processing lines..."):
                    cleaned_lines = [preprocess_text(l) for l in lines]
                    vec_lines = vectorizer.transform(cleaned_lines)
                    probs = model.predict_proba(vec_lines)
                    
                    res_df = pd.DataFrame({
                        "Message": lines,
                        "Prediction": np.where(probs[:, 1] >= threshold, "SPAM", "HAM"),
                        "Spam_Probability": [f"{p*100:.1f}%" for p in probs[:, 1]],
                        "Ham_Probability": [f"{p*100:.1f}%" for p in probs[:, 0]],
                    })
                    
                    st.dataframe(res_df, use_container_width=True)

# ==========================================
# TAB 3: MODEL HEALTH & METRICS
# ==========================================
with tab_analytics:
    st.markdown("#### 📈 Model Performance & Validation Analytics")
    st.caption("Evaluated on stratified held-out test data (20% split) from 5,157 validated email samples.")
    
    # Metrics Cards
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-val" style="color: #38bdf8;">98.93%</div>
            <div class="metric-lbl">Accuracy</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-val" style="color: #34d399;">98.35%</div>
            <div class="metric-lbl">Precision</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-val" style="color: #fbbf24;">92.97%</div>
            <div class="metric-lbl">Spam Recall</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-val" style="color: #a78bfa;">95.58%</div>
            <div class="metric-lbl">F1-Score</div>
        </div>
        """, unsafe_allow_html=True)
    with c5:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-val" style="color: #f472b6;">99.52%</div>
            <div class="metric-lbl">ROC-AUC</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    
    col_cm, col_arch = st.columns([1, 1])
    
    with col_cm:
        st.markdown("##### 🎯 Test Confusion Matrix (1,032 Samples)")
        cm_df = pd.DataFrame(
            [
                [902, 2],
                [9, 119]
            ],
            columns=["Predicted HAM", "Predicted SPAM"],
            index=["Actual HAM", "Actual SPAM"]
        )
        st.table(cm_df)
        st.markdown("""
        - **True Negatives (Correct Ham):** `902 / 904 (99.8%)`
        - **False Positives (Ham Flagged Spam):** `Only 2 (0.2%)`
        - **True Positives (Correct Spam):** `119 / 128 (93.0%)`
        - **False Negatives (Missed Spam):** `9 (7.0%)`
        """)

    with col_arch:
        st.markdown("##### 🏗️ System Pipeline & Architecture")
        st.markdown("""
        1. **Corpus Sanitization:** Removes corrupted/malformed metadata rows & deduplicates records.
        2. **Signal Preservation Engine:** Normalizes currency symbols (`$`, `£`, `€`, `₹`), phone numbers, shortcodes, URLs, and urgency punctuation marks.
        3. **Dual Feature Union:**
           - **Word TF-IDF:** Uni-grams, bi-grams, and tri-grams (sublinear TF scaling).
           - **Char TF-IDF:** Subword 3-to-5 character n-grams.
        4. **Soft-Voting Ensemble Classifier:**
           - Calibrated Linear Support Vector Classifier (LinearSVC)
           - Regularized Logistic Regression
           - Complement Naive Bayes (imbalance-robust)
           - Multinomial Naive Bayes
           - ExtraTrees Classifier
        """)
