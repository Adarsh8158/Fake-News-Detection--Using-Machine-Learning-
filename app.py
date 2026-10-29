"""
📰 Fake News Detector - Modern End-to-End Streamlit Application
Interactive Machine Learning Dashboard with Feature Explainability, In-Text Highlighting,
and Batch Processing.
"""

import os
import sys
import json
import re
import html
import requests
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from bs4 import BeautifulSoup
from PIL import Image

# Ensure src/ is in python path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from explain import predict_article, get_model
from preprocess import clean_text

# ---------------------------------------------------------
# Streamlit Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Fake News Detector | End-to-End ML",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Custom Modern UI Styling (CSS)
# ---------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .hero-container {
        padding: 1.5rem 0 1rem 0;
    }
    
    .hero-title {
        font-size: 2.6rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        background: linear-gradient(135deg, #1e3a8a 0%, #2563eb 50%, #06b6d4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
    }
    
    .hero-subtitle {
        color: #4b5563;
        font-size: 1.1rem;
        margin-bottom: 1rem;
        font-weight: 500;
    }
    
    .tag-pill {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 700;
        margin-right: 0.4rem;
        background-color: #eff6ff;
        color: #1d4ed8;
        border: 1px solid #bfdbfe;
    }
    
    /* Result Banners */
    .banner-fake {
        background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
        border: 2px solid #ef4444;
        border-radius: 12px;
        padding: 1.4rem;
        color: #991b1b;
        font-size: 1.4rem;
        font-weight: 800;
        margin: 1.2rem 0;
        box-shadow: 0 4px 12px rgba(239, 68, 68, 0.12);
    }
    
    .banner-real {
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
        border: 2px solid #22c55e;
        border-radius: 12px;
        padding: 1.4rem;
        color: #14532d;
        font-size: 1.4rem;
        font-weight: 800;
        margin: 1.2rem 0;
        box-shadow: 0 4px 12px rgba(34, 197, 94, 0.12);
    }
    
    .word-chip-fake {
        display: inline-block;
        background: #fee2e2;
        color: #991b1b;
        border: 1px solid #fca5a5;
        padding: 0.35rem 0.75rem;
        margin: 0.25rem;
        border-radius: 8px;
        font-weight: 600;
        font-size: 0.9rem;
    }
    
    .word-chip-real {
        display: inline-block;
        background: #dcfce7;
        color: #14532d;
        border: 1px solid #86efac;
        padding: 0.35rem 0.75rem;
        margin: 0.25rem;
        border-radius: 8px;
        font-weight: 600;
        font-size: 0.9rem;
    }
    
    .card-box {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.2rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
    }
</style>
""", unsafe_allow_html=True)


def create_gauge_chart(prob_real: float):
    """Generates an interactive Plotly Gauge speedometer."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob_real,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Credibility Index (0% Fake → 100% Real)", 'font': {'size': 18, 'color': "#1f2937"}},
        number={'suffix': "%", 'font': {'size': 26, 'color': "#111827"}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#4b5563"},
            'bar': {'color': "#2563eb", 'thickness': 0.25},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "#e5e7eb",
            'steps': [
                {'range': [0, 40], 'color': '#fecaca'},
                {'range': [40, 60], 'color': '#fef08a'},
                {'range': [60, 100], 'color': '#bbf7d0'}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': prob_real
            }
        }
    ))
    fig.update_layout(
        height=260,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font={'family': "Plus Jakarta Sans"}
    )
    return fig


def extract_text_from_url(url: str) -> str:
    """Scrapes headline and main paragraph text from a web article URL."""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        resp = requests.get(url, headers=headers, timeout=8)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        
        # Extract title
        title = ""
        if soup.find("h1"):
            title = soup.find("h1").get_text(strip=True)
        elif soup.title:
            title = soup.title.get_text(strip=True)
            
        # Extract paragraphs
        paragraphs = [p.get_text(strip=True) for p in soup.find_all("p")]
        body = " ".join([p for p in paragraphs if len(p.split()) > 5][:8])
        
        full_text = f"{title}. {body}" if title else body
        return full_text.strip()
    except Exception as e:
        return f"Error: Could not extract article content ({str(e)})"


# ---------------------------------------------------------
# Sidebar: Model Performance & Reports
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### 📊 Model Architecture & Stats")
    
    metrics_path = os.path.join(ROOT_DIR, "reports", "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        
        st.success(f"**Model:** {metrics.get('model_name', 'Logistic Regression + TF-IDF')}")
        
        c1, c2 = st.columns(2)
        c1.metric("Accuracy", f"{metrics.get('accuracy', 0.996) * 100:.1f}%")
        c2.metric("F1-Score", f"{metrics.get('f1_score', 0.996) * 100:.1f}%")
        
        c3, c4 = st.columns(2)
        c3.metric("Precision", f"{metrics.get('precision', 0.994) * 100:.1f}%")
        c4.metric("Recall", f"{metrics.get('recall', 0.998) * 100:.1f}%")

    cm_path = os.path.join(ROOT_DIR, "reports", "confusion_matrix.png")
    if os.path.exists(cm_path):
        st.markdown("---")
        st.markdown("#### 🧪 Confusion Matrix (Test Split)")
        st.image(Image.open(cm_path), caption="Evaluated on Verified News Split", use_container_width=True)

    st.markdown("---")
    st.markdown("""
    #### ⚙️ Pipeline Specifications:
    - **Dataset:** ISOT Benchmark (44,898 articles)
    - **Feature Extraction:** TF-IDF (N-grams 1-2)
    - **Max Features:** 50,000 vocabulary tokens
    - **Classifier:** L2 Regularized Logistic Regression
    - **Explainability:** Feature Log-Odds Attribution
    """)


# ---------------------------------------------------------
# Main Application Content
# ---------------------------------------------------------
st.markdown('<div class="hero-container">', unsafe_allow_html=True)
st.markdown('<div class="hero-title">📰 Fake News Detector</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-subtitle">Production-grade classical Machine Learning pipeline with real-time text classification, feature attribution, and in-text token highlighting.</div>',
    unsafe_allow_html=True
)

st.markdown("""
<span class="tag-pill">🚀 End-to-End ML</span>
<span class="tag-pill">⚡ Real-Time Inference</span>
<span class="tag-pill">🔍 XAI Word Attribution</span>
<span class="tag-pill">🎨 In-Text Highlighting</span>
<span class="tag-pill">📂 Batch Processing</span>
""", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

st.write("")

# ---------------------------------------------------------
# Navigation Tabs: Single Article vs Batch CSV
# ---------------------------------------------------------
mode_tab1, mode_tab2 = st.tabs(["✍️ Single Article Analysis", "📂 Batch CSV / File Evaluation"])

with mode_tab1:
    # Input Mode Selection
    input_type = st.radio(
        "Choose Input Method:",
        ["✍️ Paste News Text", "🌐 Scrape from Web URL"],
        horizontal=True
    )

    if "current_input" not in st.session_state:
        st.session_state.current_input = ""

    if input_type == "✍️ Paste News Text":
        # Preset Test Examples
        st.markdown("##### 💡 Try Instant Benchmark Samples:")
        b1, b2, b3, b4 = st.columns(4)

        if b1.button("🚨 Fake: Miracle Health Cure"):
            st.session_state.current_input = (
                "ALERT: Drinking Boiled Lemon Water With Baking Soda Completely Destroys All Cancer Cells Overnight! "
                "Doctors and Big Pharma are terrified of this suppressed breakthrough! Share this viral secret with your loved ones now!"
            )

        if b2.button("👽 Fake: Political Conspiracy"):
            st.session_state.current_input = (
                "BOMBSHELL: Declassified Whistleblower Memo Exposes Secret Underground Government Base Beneath Denver! "
                "Elite billionaires are hiding catastrophic planetary secrets from the American public. Mainstream media refuses to report this!"
            )

        if b3.button("🏛️ Real: Reuters World News"):
            st.session_state.current_input = (
                "WASHINGTON (Reuters) - The United States Senate on Tuesday voted to approve a bipartisan federal funding measure. "
                "Congressional leaders announced the agreement following extended negotiations between party representatives in the capital."
            )

        if b4.button("🔬 Real: Scientific Discovery"):
            st.session_state.current_input = (
                "PARIS (Reuters) - European space scientists confirmed the successful deployment of deep-space optical sensors. "
                "Researchers published their peer-reviewed findings in the Nature astronomy journal on Thursday morning."
            )

        user_article = st.text_area(
            "Enter or paste the news headline and content:",
            value=st.session_state.current_input,
            height=150,
            placeholder="Paste news headline or full article paragraphs here..."
        )
    else:
        url_input = st.text_input("Enter Web Article URL:", placeholder="https://www.reuters.com/world/...")
        fetch_btn = st.button("📥 Fetch Article Content")
        if fetch_btn and url_input:
            with st.spinner("Scraping and extracting article text..."):
                extracted = extract_text_from_url(url_input)
                if extracted.startswith("Error:"):
                    st.error(extracted)
                    user_article = ""
                else:
                    st.session_state.current_input = extracted
                    st.success("Successfully extracted article!")
        user_article = st.text_area(
            "Extracted Article Text:",
            value=st.session_state.current_input,
            height=150
        )

    col_run, _ = st.columns([1, 4])
    with col_run:
        run_btn = st.button("🔍 Analyze Credibility", type="primary", use_container_width=True)

    if run_btn:
        if not user_article or not user_article.strip():
            st.warning("⚠️ Please provide news text or select one of the samples above to analyze.")
        else:
            with st.spinner("Processing through ML Vectorizer and Classifier..."):
                res = predict_article(user_article)

            if "error" in res:
                st.error(res["error"])
            else:
                is_fake = res["is_fake"]
                label = res["label"]
                conf = res["confidence"]
                probs = res["probabilities"]
                explain = res["explainability"]
                style = res["stylometry"]
                html_highlight = res.get("highlighted_html", "")

                # 1. Verdict Banner
                if is_fake:
                    st.markdown(
                        f'<div class="banner-fake">🚨 CLASSIFIED AS FAKE NEWS &nbsp;—&nbsp; {conf:.1f}% Model Confidence</div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f'<div class="banner-real">🛡️ CLASSIFIED AS VERIFIED REAL NEWS &nbsp;—&nbsp; {conf:.1f}% Model Confidence</div>',
                        unsafe_allow_html=True
                    )

                # 2. Credibility Gauge & Probability Breakdown
                gauge_col, prob_col = st.columns([1.1, 1])
                with gauge_col:
                    st.plotly_chart(create_gauge_chart(probs["Real"]), use_container_width=True)
                with prob_col:
                    st.markdown("#### Probability Breakdown")
                    st.write(f"**Real News Probability:** {probs['Real']}%")
                    st.progress(probs['Real'] / 100.0)
                    st.write(f"**Fake News Probability:** {probs['Fake']}%")
                    st.progress(probs['Fake'] / 100.0)
                    
                    st.caption("Derived from the sigmoid activation of the logistic regression classification hyperplane.")

                st.write("")

                # 3. Interactive Multi-Tab Deep Dive
                tab_high, tab_xai, tab_style, tab_clean = st.tabs([
                    "🎨 In-Text Highlighting",
                    "🔍 Word Attribution (XAI)",
                    "📐 Stylometry & Clickbait Signals",
                    "🧹 Cleaned Model Input"
                ])

                with tab_high:
                    st.markdown("##### Visual In-Text Token Breakdown:")
                    st.caption("Tokens pushing toward <span style='color:#ef4444;font-weight:bold;'>Fake</span> are underlined in wavy red; tokens pushing toward <span style='color:#22c55e;font-weight:bold;'>Real</span> are underlined in green.", unsafe_allow_html=True)
                    st.markdown(html_highlight, unsafe_allow_html=True)

                with tab_xai:
                    st.markdown("##### Significant Vocabulary Coefficients:")
                    x_col1, x_col2 = st.columns(2)

                    with x_col1:
                        st.markdown("###### 🔴 Top Fake News Trigger Words:")
                        if explain["fake_cues"]:
                            for item in explain["fake_cues"]:
                                st.markdown(
                                    f'<span class="word-chip-fake">⚠️ {item["word"]} (+{item["weight"]})</span>',
                                    unsafe_allow_html=True
                                )
                        else:
                            st.info("No strong fake news trigger tokens detected.")

                    with x_col2:
                        st.markdown("###### 🟢 Top Real News Credibility Words:")
                        if explain["real_cues"]:
                            for item in explain["real_cues"]:
                                st.markdown(
                                    f'<span class="word-chip-real">✓ {item["word"]} (+{item["weight"]})</span>',
                                    unsafe_allow_html=True
                                )
                        else:
                            st.info("No strong formal news tokens detected.")

                with tab_style:
                    st.markdown("##### Stylistic and Psychological Formatting Analysis:")
                    sc1, sc2, sc3, sc4 = st.columns(4)
                    sc1.metric("Total Words", style["word_count"])
                    sc2.metric("ALL CAPS Words", style["uppercase_words"])
                    sc3.metric("Exclamation Marks (!)", style["exclamation_count"])
                    sc4.metric("Sensationalism Risk", f"{style['sensationalism_score']}%")
                    
                    if style["sensationalism_score"] > 40:
                        st.warning("⚠️ High sensationalism detected: excessive capitalization and exclamation marks are statistically associated with manipulative clickbait.")
                    else:
                        st.success("✅ Formatting is formal, measured, and consistent with standard journalistic style.")

                with tab_clean:
                    st.markdown("##### Preprocessed Representation")
                    st.caption("How the text appears after regex normalization, lowercasing, and punctuation stripping:")
                    st.code(res["cleaned_preview"], language="text")

                # Export Report Option
                st.markdown("---")
                report_data = {
                    "verdict": label,
                    "confidence_percentage": conf,
                    "probabilities": probs,
                    "stylometry": style,
                    "top_fake_indicators": explain["fake_cues"],
                    "top_real_indicators": explain["real_cues"]
                }
                st.download_button(
                    label="📥 Download JSON Analysis Report",
                    data=json.dumps(report_data, indent=4),
                    file_name="fake_news_analysis_report.json",
                    mime="application/json"
                )

with mode_tab2:
    st.markdown("### 📂 Batch News Classification")
    st.write("Upload a CSV file with news headlines or articles to classify them in bulk.")
    
    uploaded_file = st.file_uploader("Upload CSV File (must have a 'text' or 'title' column):", type=["csv"])
    
    if uploaded_file is not None:
        try:
            df_upload = pd.read_csv(uploaded_file)
            st.dataframe(df_upload.head(3), use_container_width=True)
            
            # Identify text column
            possible_cols = [c for c in df_upload.columns if c.lower() in ["text", "title", "content", "headline", "news"]]
            selected_col = st.selectbox("Select column containing news text:", df_upload.columns, index=df_upload.columns.get_loc(possible_cols[0]) if possible_cols else 0)
            
            if st.button("🚀 Process Batch Predictions", type="primary"):
                model = get_model()
                with st.spinner(f"Predicting {len(df_upload)} records..."):
                    cleaned_series = df_upload[selected_col].fillna("").astype(str).apply(clean_text)
                    preds = model.predict(cleaned_series)
                    probs = model.predict_proba(cleaned_series)
                    
                    df_upload["Prediction"] = ["FAKE" if p == 0 else "REAL" for p in preds]
                    df_upload["Confidence_%"] = [round(max(pr) * 100, 2) for pr in probs]
                    df_upload["Real_Prob_%"] = [round(pr[1] * 100, 2) for pr in probs]
                    df_upload["Fake_Prob_%"] = [round(pr[0] * 100, 2) for pr in probs]

                st.success("Batch processing complete!")
                
                # Distribution Summary
                b_c1, b_c2 = st.columns([1, 2])
                with b_c1:
                    st.metric("Total Processed", len(df_upload))
                    fake_count = (df_upload["Prediction"] == "FAKE").sum()
                    real_count = (df_upload["Prediction"] == "REAL").sum()
                    st.metric("Detected Fake", fake_count)
                    st.metric("Detected Real", real_count)
                
                with b_c2:
                    st.dataframe(df_upload[[selected_col, "Prediction", "Confidence_%"]].head(10), use_container_width=True)
                
                # Download Result
                csv_data = df_upload.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "📥 Download Processed CSV",
                    data=csv_data,
                    file_name="classified_news_results.csv",
                    mime="text/csv"
                )
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

# ---------------------------------------------------------
# Footer / Disclaimer
# ---------------------------------------------------------
st.markdown("---")
st.caption(
    "Disclaimer: This tool utilizes statistical Natural Language Processing and Machine Learning (TF-IDF + Logistic Regression) "
    "trained on the benchmark ISOT dataset. Classifications are probabilistic assessments of text structure and word distribution."
)
