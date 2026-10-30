"""
Fake News Intelligence System - Streamlit Dashboard
Modern, clean, and professional interface for news credibility analysis.
"""

import os
import sys
import json
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
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="News Credibility Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Clean, Professional CSS (Theme-Adaptive)
# ---------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Header Section */
    .app-header {
        padding-bottom: 1.5rem;
        margin-bottom: 1.5rem;
        border-bottom: 1px solid rgba(128, 128, 128, 0.15);
    }
    
    .app-title {
        font-size: 2.1rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        margin-bottom: 0.35rem;
    }
    
    .app-subtitle {
        color: #71717a;
        font-size: 1.0rem;
        font-weight: 400;
        margin-bottom: 0.8rem;
    }
    
    .meta-badge {
        display: inline-flex;
        align-items: center;
        padding: 0.25rem 0.65rem;
        border-radius: 6px;
        font-size: 0.78rem;
        font-weight: 600;
        background: rgba(128, 128, 128, 0.08);
        border: 1px solid rgba(128, 128, 128, 0.15);
        margin-right: 0.4rem;
        color: inherit;
    }

    /* Result Cards */
    .status-card-fake {
        border-left: 4px solid #ef4444;
        background: rgba(239, 68, 68, 0.06);
        border-radius: 8px;
        padding: 1.25rem;
        margin-bottom: 1rem;
    }
    
    .status-card-real {
        border-left: 4px solid #10b981;
        background: rgba(16, 185, 129, 0.06);
        border-radius: 8px;
        padding: 1.25rem;
        margin-bottom: 1rem;
    }

    .status-badge-fake {
        display: inline-block;
        color: #ef4444;
        font-weight: 700;
        font-size: 1.15rem;
        margin-bottom: 0.3rem;
    }
    
    .status-badge-real {
        display: inline-block;
        color: #10b981;
        font-weight: 700;
        font-size: 1.15rem;
        margin-bottom: 0.3rem;
    }

    .word-pill-fake {
        display: inline-block;
        background: rgba(239, 68, 68, 0.1);
        color: #ef4444;
        border: 1px solid rgba(239, 68, 68, 0.25);
        padding: 0.25rem 0.6rem;
        margin: 0.2rem;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
    }

    .word-pill-real {
        display: inline-block;
        background: rgba(16, 185, 129, 0.1);
        color: #10b981;
        border: 1px solid rgba(16, 185, 129, 0.25);
        padding: 0.25rem 0.6rem;
        margin: 0.2rem;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
    }

    .clean-box {
        background: rgba(128, 128, 128, 0.04);
        border: 1px solid rgba(128, 128, 128, 0.12);
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


def create_clean_gauge(prob_real: float):
    """
    Renders an elegant, minimalist Plotly gauge with clean transparency
    and adaptive typography for light and dark environments.
    """
    if prob_real >= 60.0:
        bar_color = "#10b981"  # Emerald
    elif prob_real >= 40.0:
        bar_color = "#f59e0b"  # Amber
    else:
        bar_color = "#ef4444"  # Rose / Red

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob_real,
        domain={'x': [0, 1], 'y': [0, 1]},
        number={
            'suffix': "%",
            'font': {'size': 36, 'family': "Inter"}
        },
        gauge={
            'axis': {
                'range': [0, 100],
                'tickwidth': 1,
                'tickcolor': "gray",
                'tickvals': [0, 25, 50, 75, 100],
                'tickfont': {'size': 11, 'family': "Inter"}
            },
            'bar': {'color': bar_color, 'thickness': 0.25},
            'bgcolor': "rgba(128, 128, 128, 0.08)",
            'borderwidth': 0,
            'steps': [
                {'range': [0, 40], 'color': 'rgba(239, 68, 68, 0.18)'},
                {'range': [40, 60], 'color': 'rgba(245, 158, 11, 0.18)'},
                {'range': [60, 100], 'color': 'rgba(16, 185, 129, 0.18)'}
            ]
        }
    ))

    fig.update_layout(
        height=210,
        margin=dict(l=20, r=20, t=25, b=15),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
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
        
        title = ""
        if soup.find("h1"):
            title = soup.find("h1").get_text(strip=True)
        elif soup.title:
            title = soup.title.get_text(strip=True)
            
        paragraphs = [p.get_text(strip=True) for p in soup.find_all("p")]
        body = " ".join([p for p in paragraphs if len(p.split()) > 5][:8])
        
        full_text = f"{title}. {body}" if title else body
        return full_text.strip()
    except Exception as e:
        return f"Error: Could not extract article content ({str(e)})"


# ---------------------------------------------------------
# Sidebar: System Specifications & Benchmarks
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### System Specifications")

    metrics_path = os.path.join(ROOT_DIR, "reports", "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)

        c1, c2 = st.columns(2)
        c1.metric("Accuracy", f"{metrics.get('accuracy', 0) * 100:.1f}%")
        c2.metric("F1-Score", f"{metrics.get('f1_score', 0) * 100:.1f}%")

        c3, c4 = st.columns(2)
        c3.metric("Precision", f"{metrics.get('precision', 0) * 100:.1f}%")
        c4.metric("Recall", f"{metrics.get('recall', 0) * 100:.1f}%")

    st.markdown("---")
    st.markdown("""
    **Model Configuration:**
    - **Architecture:** TF-IDF + Logistic Regression
    - **Vocabulary:** 50,000 N-gram Features (1, 2)
    - **Dataset:** ISOT Fake News dataset
    - **Loss Function:** Log-Loss with L2 Penalty
    """)

    cm_path = os.path.join(ROOT_DIR, "reports", "confusion_matrix.png")
    if os.path.exists(cm_path):
        with st.expander("View Confusion Matrix"):
            st.image(Image.open(cm_path), use_container_width=True)

    st.markdown("---")
    st.caption("Developed for natural language credibility analysis and verifiable feature attribution.")


# ---------------------------------------------------------
# Header Section
# ---------------------------------------------------------
_acc = None
try:
    with open(os.path.join(ROOT_DIR, "reports", "metrics.json"), "r", encoding="utf-8") as _f:
        _acc = json.load(_f).get("accuracy")
except Exception:
    pass
_acc_badge = f"Test Accuracy: {_acc * 100:.1f}%" if _acc is not None else "Test Accuracy: n/a"

st.markdown(f"""
<div class="app-header">
    <div class="app-title">🛡️ News Credibility Intelligence</div>
    <div class="app-subtitle">Machine learning system for news credibility verification, text explainability, and stylometric analysis.</div>
    <div>
        <span class="meta-badge">Model: Logistic Regression</span>
        <span class="meta-badge">{_acc_badge}</span>
        <span class="meta-badge">Benchmark: ISOT Dataset</span>
        <span class="meta-badge">XAI Enabled</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Main Tabs: Single Article vs Batch Analysis
# ---------------------------------------------------------
main_tab1, main_tab2 = st.tabs(["Article Analysis", "Batch Dataset Evaluation"])

with main_tab1:
    # Input Mode Selector
    input_mode = st.radio(
        "Source Input:",
        ["Direct Text Input", "Web URL Import"],
        horizontal=True,
        label_visibility="collapsed"
    )

    if "current_input" not in st.session_state:
        st.session_state.current_input = ""

    if input_mode == "Direct Text Input":
        # Clean Presets
        st.markdown("**Sample Benchmark Articles:**")
        p1, p2, p3, p4 = st.columns(4)
        
        if p1.button("🚨 Health Clickbait", use_container_width=True):
            st.session_state.current_input = (
                "ALERT: Drinking Boiled Lemon Water With Baking Soda Completely Destroys All Cancer Cells Overnight! "
                "Doctors and Big Pharma are terrified of this suppressed breakthrough! Share this viral secret with your loved ones now!"
            )
        if p2.button("🛸 Political Rumor", use_container_width=True):
            st.session_state.current_input = (
                "BOMBSHELL: Declassified Whistleblower Memo Exposes Secret Underground Government Base Beneath Denver! "
                "Elite billionaires are hiding catastrophic planetary secrets from the American public. Mainstream media refuses to report this!"
            )
        if p3.button("🏛️ Reuters Wire", use_container_width=True):
            st.session_state.current_input = (
                "WASHINGTON (Reuters) - The United States Senate on Tuesday voted to approve a bipartisan federal funding measure. "
                "Congressional leaders announced the agreement following extended negotiations between party representatives in the capital."
            )
        if p4.button("🔬 Science Journal", use_container_width=True):
            st.session_state.current_input = (
                "PARIS (Reuters) - European space scientists confirmed the successful deployment of deep-space optical sensors. "
                "Researchers published their peer-reviewed findings in the Nature astronomy journal on Thursday morning."
            )

        news_input = st.text_area(
            "Article Text / Headline:",
            value=st.session_state.current_input,
            height=140,
            placeholder="Paste news headline or full article text here..."
        )
    else:
        url_input = st.text_input("Web Article URL:", placeholder="https://www.reuters.com/world/...")
        fetch_col, _ = st.columns([1, 4])
        with fetch_col:
            fetch_btn = st.button("Fetch Content", use_container_width=True)
            
        if fetch_btn and url_input:
            with st.spinner("Fetching article content..."):
                extracted = extract_text_from_url(url_input)
                if extracted.startswith("Error:"):
                    st.error(extracted)
                    news_input = ""
                else:
                    st.session_state.current_input = extracted
                    st.success("Article successfully fetched.")
        news_input = st.text_area(
            "Fetched Article Text:",
            value=st.session_state.current_input,
            height=140
        )

    action_col, _ = st.columns([1, 3])
    with action_col:
        analyze_btn = st.button("Analyze Article", type="primary", use_container_width=True)

    # ---------------------------------------------------------
    # Analysis Results Display
    # ---------------------------------------------------------
    if analyze_btn:
        if not news_input or not news_input.strip():
            st.warning("Please provide an article text or select a preset sample.")
        else:
            with st.spinner("Evaluating article patterns..."):
                res = predict_article(news_input)

            if "error" in res:
                st.error(res["error"])
            else:
                is_fake = res["is_fake"]
                conf = res["confidence"]
                probs = res["probabilities"]
                explain = res["explainability"]
                style = res["stylometry"]
                html_highlight = res.get("highlighted_html", "")

                st.write("")

                # 1. Primary Verdict Banner
                if is_fake:
                    st.markdown(f"""
                    <div class="status-card-fake">
                        <div class="status-badge-fake">● Flagged as Manipulated / Fake Content</div>
                        <div style="font-size: 1.05rem; opacity: 0.95;">
                            The classifier detected rhetorical patterns, sensationalist styling, and vocabulary strongly consistent with fabricated news stories.
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="status-card-real">
                        <div class="status-badge-real">● Verified as Authentic Reporting</div>
                        <div style="font-size: 1.05rem; opacity: 0.95;">
                            The text exhibits measured syntax, neutral attribution, and professional journalistic vocabulary consistent with established reporting.
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                # 2. Metric Overview Row (Gauge + Probabilities)
                c_left, c_right = st.columns([1, 1.2])

                with c_left:
                    st.markdown("**Credibility Meter (Real %)**")
                    st.plotly_chart(create_clean_gauge(probs["Real"]), use_container_width=True)

                with c_right:
                    st.markdown("**Probability Breakdown**")
                    st.write(f"Authentic News Likelihood: **{probs['Real']}%**")
                    st.progress(probs["Real"] / 100.0)

                    st.write(f"Fabricated Content Likelihood: **{probs['Fake']}%**")
                    st.progress(probs["Fake"] / 100.0)

                    st.caption(f"Decision confidence: **{conf:.1f}%** based on TF-IDF feature attribution.")

                st.write("")

                # 3. Clean Detail Tabs
                tab1, tab2, tab3, tab4 = st.tabs([
                    "Text Inspector",
                    "Feature Attribution (XAI)",
                    "Stylometric Signals",
                    "Raw Data & Export"
                ])

                with tab1:
                    st.markdown("**In-Text Token Highlighting**")
                    st.caption("Tokens swaying the model toward <span style='color:#ef4444;font-weight:600;'>Fake</span> are highlighted in red; tokens indicating <span style='color:#10b981;font-weight:600;'>Authentic</span> news are highlighted in green.", unsafe_allow_html=True)
                    st.markdown(html_highlight, unsafe_allow_html=True)

                with tab2:
                    st.markdown("**Key Vocabulary Drivers**")
                    st.caption("Mathematical impact coefficients ($w_i \\cdot x_i$) calculated from the model's decision hyperplane.")
                    col_sig1, col_sig2 = st.columns(2)

                    with col_sig1:
                        st.markdown("**🔴 Suspicious Indicators:**")
                        if explain["fake_cues"]:
                            for item in explain["fake_cues"]:
                                st.markdown(
                                    f'<span class="word-pill-fake">{item["word"]} · +{item["weight"]}</span>',
                                    unsafe_allow_html=True
                                )
                        else:
                            st.info("No significant fake indicators found.")

                    with col_sig2:
                        st.markdown("**🟢 Credibility Indicators:**")
                        if explain["real_cues"]:
                            for item in explain["real_cues"]:
                                st.markdown(
                                    f'<span class="word-pill-real">{item["word"]} · +{item["weight"]}</span>',
                                    unsafe_allow_html=True
                                )
                        else:
                            st.info("No formal news indicators found.")

                with tab3:
                    st.markdown("**Stylistic Structure Analysis**")
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Word Count", style["word_count"])
                    m2.metric("ALL CAPS Words", style["uppercase_words"])
                    m3.metric("Exclamation Marks (!)", style["exclamation_count"])
                    m4.metric("Sensationalism Score", f"{style['sensationalism_score']}%")

                    if style["sensationalism_score"] > 40:
                        st.warning("Notice: Elevated sensationalism detected in punctuation or capitalization patterns.")
                    else:
                        st.success("Notice: Text structure aligns with neutral, standard journalism conventions.")

                with tab4:
                    st.markdown("**Preprocessed Representation:**")
                    st.code(res["cleaned_preview"], language="text")

                    report_obj = {
                        "verdict": res["label"],
                        "confidence": conf,
                        "probabilities": probs,
                        "stylometrics": style,
                        "fake_tokens": explain["fake_cues"],
                        "real_tokens": explain["real_cues"]
                    }

                    st.download_button(
                        "Download JSON Report",
                        data=json.dumps(report_obj, indent=2),
                        file_name="credibility_analysis_report.json",
                        mime="application/json"
                    )

with main_tab2:
    st.markdown("### Batch Article Processing")
    st.write("Upload a CSV callset to perform mass classification across article headlines or texts.")

    uploaded_csv = st.file_uploader("Select CSV file:", type=["csv"])

    if uploaded_csv is not None:
        try:
            batch_df = pd.read_csv(uploaded_csv)
            st.dataframe(batch_df.head(3), use_container_width=True)

            target_candidates = [c for c in batch_df.columns if c.lower() in ["text", "title", "content", "headline", "news"]]
            default_index = batch_df.columns.get_loc(target_candidates[0]) if target_candidates else 0
            text_column = st.selectbox("Select text column:", batch_df.columns, index=default_index)

            if st.button("Run Batch Classification", type="primary"):
                clf_pipeline = get_model()
                with st.spinner(f"Classifying {len(batch_df)} rows..."):
                    cleaned_texts = batch_df[text_column].fillna("").astype(str).apply(clean_text)
                    predictions = clf_pipeline.predict(cleaned_texts)
                    probabilities = clf_pipeline.predict_proba(cleaned_texts)

                    batch_df["Verdict"] = ["FAKE" if p == 0 else "REAL" for p in predictions]
                    batch_df["Confidence_%"] = [round(max(prob) * 100, 2) for prob in probabilities]
                    batch_df["Real_Prob_%"] = [round(prob[1] * 100, 2) for prob in probabilities]
                    batch_df["Fake_Prob_%"] = [round(prob[0] * 100, 2) for prob in probabilities]

                st.success("Classification complete.")

                res_c1, res_c2 = st.columns([1, 2])
                with res_c1:
                    st.metric("Total Analyzed", len(batch_df))
                    st.metric("Detected Fake", (batch_df["Verdict"] == "FAKE").sum())
                    st.metric("Detected Real", (batch_df["Verdict"] == "REAL").sum())

                with res_c2:
                    st.dataframe(batch_df[[text_column, "Verdict", "Confidence_%"]].head(8), use_container_width=True)

                out_csv = batch_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "Download Processed CSV",
                    data=out_csv,
                    file_name="batch_credibility_results.csv",
                    mime="text/csv"
                )
        except Exception as err:
            st.error(f"Error parsing file: {str(err)}")
