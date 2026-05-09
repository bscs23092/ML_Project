from __future__ import annotations

from html import escape
import json
from pathlib import Path
import sys

import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
XSS_DIR = BASE_DIR / "xss"
SQLI_DIR = BASE_DIR / "sqli" / "ml_project"


st.set_page_config(
    page_title="Web Attack Detector",
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


def ensure_import_path(path: Path) -> None:
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


@st.cache_resource
def load_xss_pipeline(path: str | Path):
    try:
        import torch

        ensure_import_path(XSS_DIR)
        from classifier import XSSClassifier
        from preprocessing import CharTokenizer

        path = Path(path)
        if not path.is_absolute():
            path = BASE_DIR / path

        model_path = path / "best_model.pt" if path.is_dir() else path
        tok_path = path / "tokenizer.json"

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        if not tok_path.exists():
            raise FileNotFoundError(f"Tokenizer not found: {tok_path}")

        hparams_path = path / "hparams.json"
        hparams = {"embed_dim": 128, "num_filters": 64, "dropout": 0.5}
        if hparams_path.exists():
            with open(hparams_path, encoding="utf-8") as f:
                hparams.update(json.load(f))

        tok = CharTokenizer.load(str(tok_path))
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

        model = XSSClassifier(tok.vocab_size, hparams["embed_dim"], hparams["num_filters"], hparams["dropout"])
        state = torch.load(str(model_path), map_location=device)
        model.load_state_dict(state)
        model.to(device)
        model.eval()

        class Pipeline:
            def __init__(self, model, tok, device):
                self.model = model
                self.tok = tok
                self.device = device

            def predict_proba(self, texts: list[str]) -> list[float]:
                x = torch.stack(self.tok.transform(texts)).to(self.device)
                with torch.no_grad():
                    probs = torch.sigmoid(self.model(x)).cpu().numpy()
                return [float(p) for p in probs]

        return Pipeline(model, tok, device), None
    except Exception as e:
        return None, str(e)


@st.cache_resource
def load_sqli_pipeline(path: str | Path):
    try:
        ensure_import_path(SQLI_DIR)
        from sql_injection_lr.pipeline import SQLInjectionPipeline

        path = Path(path)
        if not path.is_absolute():
            path = BASE_DIR / path

        return SQLInjectionPipeline.load(path), None
    except Exception as e:
        return None, str(e)


def render_single_result(probability: float, threshold: float) -> None:
    prediction = int(probability >= threshold)
    label = "Malicious" if prediction else "Normal"
    cls = "malicious" if prediction else "normal"
    color = "#ff4f6d" if prediction else "#00e5a0"

    st.markdown(
        f"""
        <div class="single-result {cls}">
            <div class="single-verdict {cls}">{label}</div>
            <div style="color:{color};margin-top:10px">
                Probability: {probability:.4f}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_batch_results(results: list[dict], text_key: str, csv_header: str, file_name: str) -> None:
    total = len(results)
    malicious = sum(r["pred"] for r in results)
    normal = total - malicious

    c1, c2, c3 = st.columns(3)
    c1.metric("Total", total)
    c2.metric("Malicious", malicious)
    c3.metric("Normal", normal)

    for result in results:
        is_malicious = result["pred"] == 1
        cls = "malicious" if is_malicious else "normal"
        badge_cls = "badge-malicious" if is_malicious else "badge-normal"
        label = "MALICIOUS" if is_malicious else "NORMAL"
        color = "#ff4f6d" if is_malicious else "#00e5a0"
        display_text = escape(result[text_key])

        st.markdown(
            f"""
            <div class="result-card {cls}">
                <span class="badge {badge_cls}">{label}</span>
                <div class="prob-bar-bg">
                    <div class="prob-bar-fill" style="width:{result['prob']*100}%;background:{color}"></div>
                </div>
                <span style="color:{color}">{result['prob']:.4f}</span>
                <span class="query-text">{display_text}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    csv = f"{csv_header},probability,prediction\n"
    for result in results:
        escaped_value = result[text_key].replace('"', '""')
        prediction = "Malicious" if result["pred"] else "Normal"
        csv += f'"{escaped_value}",{result["prob"]},{prediction}\n'

    st.download_button("Export CSV", data=csv, file_name=file_name, mime="text/csv")


with st.sidebar:
    st.markdown("### Configuration")
    detector = st.selectbox("Detector", ["XSS", "SQL Injection"])

    if detector == "XSS":
        model_path = st.text_input("Model directory", value="xss/saved_models")
    else:
        model_path = st.text_input("Model path", value="sqli/ml_project/artifacts/sqli_logreg_model.json")

    threshold_override = st.slider("Threshold", 0.0, 1.0, 0.5, 0.01)


if detector == "XSS":
    pipeline, load_error = load_xss_pipeline(model_path)
    title = "XSS Detector"
    single_tab_label = "Single Input"
    batch_tab_label = "Batch"
    single_input_label = "Input string (HTML / payload)"
    batch_input_label = "Add inputs (one per line)"
    empty_warning = "Enter an input string"
    queued_label = "items queued"
    text_key = "input"
    csv_header = "input"
    csv_file_name = "xss_results.csv"
else:
    pipeline, load_error = load_sqli_pipeline(model_path)
    title = "SQL Injection Detector"
    single_tab_label = "Single Query"
    batch_tab_label = "Batch"
    single_input_label = "SQL Query"
    batch_input_label = "Add queries (one per line)"
    empty_warning = "Enter a query"
    queued_label = "queries queued"
    text_key = "query"
    csv_header = "query"
    csv_file_name = "sqli_results.csv"


st.markdown(f"<h1>{title}</h1>", unsafe_allow_html=True)

if load_error:
    st.error(f"Could not load model: {load_error}")
    st.stop()


queue_key = f"{detector.lower().replace(' ', '_')}_queue"
results_key = f"{detector.lower().replace(' ', '_')}_results"

if queue_key not in st.session_state:
    st.session_state[queue_key] = []
if results_key not in st.session_state:
    st.session_state[results_key] = []


tab1, tab2 = st.tabs([single_tab_label, batch_tab_label])

with tab1:
    query = st.text_area(single_input_label, height=120)

    if st.button("Classify"):
        if query.strip():
            probability = pipeline.predict_proba([query])[0]
            render_single_result(probability, threshold_override)
        else:
            st.warning(empty_warning)

with tab2:
    batch_input = st.text_area(batch_input_label, height=120)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Add"):
            lines = [line.strip() for line in batch_input.splitlines() if line.strip()]
            st.session_state[queue_key].extend(lines)
            st.session_state[results_key] = []

    with col2:
        if st.button("Clear"):
            st.session_state[queue_key] = []
            st.session_state[results_key] = []

    if st.session_state[queue_key]:
        st.write(f"{len(st.session_state[queue_key])} {queued_label}")

        if st.button("Run classification"):
            probabilities = pipeline.predict_proba(st.session_state[queue_key])
            st.session_state[results_key] = [
                {text_key: item, "prob": probability, "pred": int(probability >= threshold_override)}
                for item, probability in zip(st.session_state[queue_key], probabilities)
            ]

    if st.session_state[results_key]:
        render_batch_results(st.session_state[results_key], text_key, csv_header, csv_file_name)
