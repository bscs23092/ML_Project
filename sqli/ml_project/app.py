from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="SQLi Detector",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');

    :root {
        --bg:#0d0f14; --surface:#161a22; --border:#252b38;
        --accent:#00e5a0; --danger:#ff4f6d;
        --text:#e2e8f0; --text-dim:#8b95a8;
    }

    html, body, [data-testid="stAppViewContainer"] { background: var(--bg)!important; }
    [data-testid="stHeader"] { background: transparent!important; }
    section.main > div { padding-top: 1.5rem!important; }
    * { font-family: 'DM Sans', sans-serif; color: var(--text); }

    #MainMenu, footer { visibility:hidden; }

    .stTextArea textarea {
        background: var(--surface)!important;
        border:1px solid var(--border)!important;
        border-radius:8px!important;
        color:var(--text)!important;
        font-family:'JetBrains Mono', monospace!important;
    }

    .stButton > button[kind="primary"] {
        background:var(--accent)!important;
        color:#000!important;
        border-radius:8px!important;
        font-weight:600!important;
    }

    .stButton > button[kind="secondary"] {
        background:transparent!important;
        border:1px solid var(--border)!important;
        color:var(--text-dim)!important;
        border-radius:8px!important;
    }

    .result-card {
        background: var(--surface);
        border:1px solid var(--border);
        border-radius:10px;
        padding:14px;
        margin-bottom:10px;
        display:flex;
        align-items:center;
        gap:12px;
        font-size:12.5px;
    }

    .result-card.malicious { border-left:3px solid var(--danger); }
    .result-card.normal { border-left:3px solid var(--accent); }

    .badge {
        padding:4px 10px;
        border-radius:20px;
        font-size:11px;
        font-weight:700;
    }

    .badge-malicious { background:rgba(255,79,109,0.15); color:var(--danger); }
    .badge-normal { background:rgba(0,229,160,0.12); color:var(--accent); }

    .prob-bar-bg {
        height:6px;
        background:var(--border);
        border-radius:3px;
        overflow:hidden;
        flex:1;
    }

    .prob-bar-fill {
        height:100%;
        border-radius:3px;
    }

    .query-text {
        color:var(--text-dim);
        white-space:nowrap;
        overflow:hidden;
        text-overflow:ellipsis;
        max-width:400px;
    }

    .single-result {
        border-radius:12px;
        padding:24px;
        text-align:center;
        border:1px solid var(--border);
    }

    .single-result.malicious { background:rgba(255,79,109,0.07); }
    .single-result.normal { background:rgba(0,229,160,0.06); }

    .single-verdict {
        font-size:1.8rem;
        font-weight:700;
    }

    .single-verdict.malicious { color:var(--danger); }
    .single-verdict.normal { color:var(--accent); }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### Configuration")
    model_path = st.text_input("Model path", value="artifacts/sqli_logreg_model.json")
    threshold_override = st.slider("Threshold", 0.0, 1.0, 0.5, 0.01)

@st.cache_resource
def load_pipeline(path):
    try:
        from sql_injection_lr.pipeline import SQLInjectionPipeline
        return SQLInjectionPipeline.load(path), None
    except Exception as e:
        return None, str(e)

pipeline, load_error = load_pipeline(model_path)

st.markdown("<h1>SQL Injection Detector</h1>", unsafe_allow_html=True)

if load_error:
    st.error(f"Could not load model: {load_error}")
    st.stop()

if "queue" not in st.session_state:
    st.session_state.queue = []
if "results" not in st.session_state:
    st.session_state.results = []

tab1, tab2 = st.tabs(["Single Query", "Batch"])

with tab1:
    query = st.text_area("SQL Query", height=120)

    if st.button("Classify"):
        if query.strip():
            prob = pipeline.predict_proba([query])[0]
            pred = int(prob >= threshold_override)

            label = "Malicious" if pred else "Normal"
            cls = "malicious" if pred else "normal"
            color = "#ff4f6d" if pred else "#00e5a0"

            st.markdown(
                f"""
                <div class="single-result {cls}">
                    <div class="single-verdict {cls}">{label}</div>
                    <div style="color:{color};margin-top:10px">
                        Probability: {prob:.4f}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.warning("Enter a query")

with tab2:
    batch_input = st.text_area("Add queries (one per line)", height=120)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Add"):
            lines = [l.strip() for l in batch_input.splitlines() if l.strip()]
            st.session_state.queue.extend(lines)
            st.session_state.results = []

    with col2:
        if st.button("Clear"):
            st.session_state.queue = []
            st.session_state.results = []

    if st.session_state.queue:
        st.write(f"{len(st.session_state.queue)} queries queued")

        if st.button("Run classification"):
            probs = pipeline.predict_proba(st.session_state.queue)
            st.session_state.results = [
                {"query": q, "prob": p, "pred": int(p >= threshold_override)}
                for q, p in zip(st.session_state.queue, probs)
            ]

    if st.session_state.results:
        results = st.session_state.results

        total = len(results)
        mal = sum(r["pred"] for r in results)
        norm = total - mal

        c1, c2, c3 = st.columns(3)
        c1.metric("Total", total)
        c2.metric("Malicious", mal)
        c3.metric("Normal", norm)

        for r in results:
            is_mal = r["pred"] == 1
            cls = "malicious" if is_mal else "normal"
            badge_cls = "badge-malicious" if is_mal else "badge-normal"
            label = "MALICIOUS" if is_mal else "NORMAL"
            color = "#ff4f6d" if is_mal else "#00e5a0"

            st.markdown(
                f"""
                <div class="result-card {cls}">
                    <span class="badge {badge_cls}">{label}</span>
                    <div class="prob-bar-bg">
                        <div class="prob-bar-fill" style="width:{r['prob']*100}%;background:{color}"></div>
                    </div>
                    <span style="color:{color}">{r['prob']:.4f}</span>
                    <span class="query-text">{r['query']}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        csv = "query,probability,prediction\n"
        for r in results:
            csv += f'"{r["query"]}",{r["prob"]},{"Malicious" if r["pred"] else "Normal"}\n'

        st.download_button(
            "Export CSV",
            data=csv,
            file_name="results.csv",
            mime="text/csv",
        )