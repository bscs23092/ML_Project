from __future__ import annotations

from html import escape
import json
from pathlib import Path
import re
import sys
from urllib.parse import parse_qs, urlparse

import joblib
import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
XSS_DIR = BASE_DIR / "xss"
SQLI_DIR = BASE_DIR / "sqli" / "ml_project"
CSRF_DEFAULT_MODEL = BASE_DIR / "csrf" / "artifacts" / "csrf_boosting_pipeline.joblib"
CSRF_KEYWORDS = [
    ("create", "create"),
    ("add", "add"),
    ("set", "set"),
    ("delete", "delete"),
    ("update", "update"),
    ("remove", "remove"),
    ("friend", "friend"),
    ("setting", "setting"),
    ("password", "password"),
    ("token", "token"),
    ("change", "change"),
    ("action", "action"),
    ("pay", "pay"),
    ("login", "login"),
    ("logout", "logout"),
    ("post", "post"),
    ("comment", "comment"),
    ("follow", "follow"),
    ("subscribe", "subscribe"),
    ("signIn", "signin"),
    ("view", "view"),
]


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
        white-space:normal;
        overflow-wrap:anywhere;
        line-height:1.45;
        flex:1;
    }

    .result-main {
        width:100%;
        display:flex;
        align-items:center;
        gap:12px;
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


@st.cache_resource
def load_csrf_artifact(path: str | Path):
    try:
        path = Path(path)
        if not path.is_absolute():
            path = BASE_DIR / path
        artifact = joblib.load(path)
        required = {"pipeline", "threshold", "feature_columns"}
        missing = required - set(artifact)
        if missing:
            raise ValueError(f"Model artifact is missing: {sorted(missing)}")
        return artifact, None
    except Exception as e:
        return None, str(e)


def normalize_for_keyword(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def has_keyword(text: str, keyword: str) -> int:
    normalized_text = normalize_for_keyword(text)
    normalized_keyword = normalize_for_keyword(keyword)
    return int(normalized_keyword in normalized_text)


def parse_params(raw_params: str) -> dict[str, list[str]]:
    raw_params = raw_params.strip()
    if not raw_params:
        return {}

    try:
        parsed_json = json.loads(raw_params)
        if isinstance(parsed_json, dict):
            parsed: dict[str, list[str]] = {}
            for key, value in parsed_json.items():
                if isinstance(value, list):
                    parsed[str(key)] = [str(item) for item in value]
                else:
                    parsed[str(key)] = [str(value)]
            return parsed
    except json.JSONDecodeError:
        pass

    query_like = raw_params[1:] if raw_params.startswith("?") else raw_params
    parsed_query = parse_qs(query_like, keep_blank_values=True)
    if parsed_query:
        return {str(key): [str(item) for item in values] for key, values in parsed_query.items()}

    parsed_lines: dict[str, list[str]] = {}
    for line in raw_params.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        parsed_lines.setdefault(key.strip(), []).append(value.strip())
    return parsed_lines


def count_bool_values(params: dict[str, list[str]]) -> int:
    bool_values = {"true", "false", "yes", "no", "on", "off"}
    return sum(str(value).strip().lower() in bool_values for values in params.values() for value in values)


def count_id_values(params: dict[str, list[str]]) -> int:
    count = 0
    for key, values in params.items():
        key_has_id = bool(re.search(r"(^|[_-])id($|[_-])|id$", key.lower()))
        value_has_id = any(re.fullmatch(r"\d{2,}", str(value).strip()) for value in values)
        count += int(key_has_id or value_has_id)
    return count


def count_blob_values(params: dict[str, list[str]]) -> int:
    blob_like = 0
    for values in params.values():
        for value in values:
            value = str(value).strip()
            compact = re.sub(r"\s+", "", value)
            if len(compact) >= 40 or value.startswith(("{", "[")):
                blob_like += 1
    return blob_like


def build_csrf_features(method: str, path_or_url: str, raw_params: str, feature_columns: list[str]) -> dict[str, float]:
    parsed_url = urlparse(path_or_url.strip())
    path = parsed_url.path or path_or_url.strip()
    query_params = parse_qs(parsed_url.query, keep_blank_values=True)
    body_params = parse_params(raw_params)
    params = {str(key): [str(item) for item in values] for key, values in query_params.items()}
    for key, values in body_params.items():
        params.setdefault(str(key), []).extend([str(value) for value in values])

    param_text = " ".join([*params.keys(), *[value for values in params.values() for value in values]])
    request_length = len(method) + len(path_or_url) + len(raw_params)

    features = {column: 0 for column in feature_columns}
    features.update(
        {
            "numOfParams": len(params),
            "numOfBools": count_bool_values(params),
            "numOfIds": count_id_values(params),
            "numOfBlobs": count_blob_values(params),
            "reqLen": request_length,
            "isPUT": int(method.upper() == "PUT"),
            "isDELETE": int(method.upper() == "DELETE"),
            "isPOST": int(method.upper() == "POST"),
            "isGET": int(method.upper() == "GET"),
            "isOPTIONS": int(method.upper() == "OPTIONS"),
        }
    )

    for feature_prefix, keyword in CSRF_KEYWORDS:
        path_column = f"{feature_prefix}InPath"
        params_column = f"{feature_prefix}InParams"
        if path_column in features:
            features[path_column] = has_keyword(path, keyword)
        if params_column in features:
            features[params_column] = has_keyword(param_text, keyword)

    return {column: features[column] for column in feature_columns}


def csrf_predict_proba(artifact: dict, rows: list[dict]) -> list[float]:
    frame = pd.DataFrame(rows, columns=artifact["feature_columns"])
    return [float(value) for value in artifact["pipeline"].predict_proba(frame)[:, 1]]


def csrf_method_from_features(row: pd.Series) -> str:
    for method, column in [
        ("PUT", "isPUT"),
        ("DELETE", "isDELETE"),
        ("POST", "isPOST"),
        ("GET", "isGET"),
        ("OPTIONS", "isOPTIONS"),
    ]:
        if int(row.get(column, 0)) == 1:
            return method
    return "POST"


def build_csrf_query_from_feature_row(row: pd.Series) -> str:
    method = csrf_method_from_features(row)
    path_parts = [
        keyword
        for feature_prefix, keyword in CSRF_KEYWORDS
        if int(row.get(f"{feature_prefix}InPath", 0)) == 1
    ]
    param_keywords = [
        keyword
        for feature_prefix, keyword in CSRF_KEYWORDS
        if int(row.get(f"{feature_prefix}InParams", 0)) == 1
    ]

    if path_parts:
        path = "/" + "/".join(path_parts)
    elif method in {"POST", "PUT", "DELETE"}:
        path = "/state/change"
    else:
        path = "/view"

    params: list[tuple[str, str]] = []
    for idx, keyword in enumerate(param_keywords, start=1):
        params.append((keyword, f"{keyword}_{idx}"))

    for idx in range(int(row.get("numOfIds", 0))):
        params.append((f"id{idx + 1}", str(1000 + idx)))
    for idx in range(int(row.get("numOfBools", 0))):
        params.append((f"enabled{idx + 1}", "true"))
    for idx in range(int(row.get("numOfBlobs", 0))):
        params.append((f"payload{idx + 1}", "{\"data\":\"...\"}"))

    target_param_count = max(int(row.get("numOfParams", 0)), len(params))
    while len(params) < target_param_count:
        params.append((f"param{len(params) + 1}", f"value{len(params) + 1}"))

    query_string = "&".join(f"{key}={value}" for key, value in params)
    if method == "GET" and query_string:
        return f"{method} {path}?{query_string}"
    if query_string:
        return f"{method} {path} BODY {query_string}"
    return f"{method} {path}"


def read_batch_table(uploaded_file) -> pd.DataFrame:
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(uploaded_file)
    if suffix == ".xlsx":
        return pd.read_excel(uploaded_file)
    raise ValueError("Upload a .csv or .xlsx file.")


def extract_batch_values(frame: pd.DataFrame, preferred_column: str) -> list[str]:
    if frame.empty:
        return []

    candidates = [preferred_column, "query", "input", "payload", "text", "request"]
    selected = next((column for column in candidates if column in frame.columns), frame.columns[0])
    return [str(value).strip() for value in frame[selected].dropna().tolist() if str(value).strip()]


def render_single_result(
    probability: float,
    threshold: float,
    positive_label: str = "Malicious",
    negative_label: str = "Normal",
) -> None:
    prediction = int(probability >= threshold)
    label = positive_label if prediction else negative_label
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


def render_batch_results(
    results: list[dict],
    text_key: str,
    csv_header: str,
    file_name: str,
    positive_label: str = "Malicious",
    negative_label: str = "Normal",
    download_key: str | None = None,
) -> None:
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
        label = positive_label.upper() if is_malicious else negative_label.upper()
        color = "#ff4f6d" if is_malicious else "#00e5a0"
        display_text = escape(result[text_key])

        st.markdown(
            f"""
            <div class="result-card {cls}">
                <div class="result-main">
                    <span class="badge {badge_cls}">{label}</span>
                    <div class="prob-bar-bg">
                        <div class="prob-bar-fill" style="width:{result['prob']*100}%;background:{color}"></div>
                    </div>
                    <span style="color:{color}">{result['prob']:.4f}</span>
                    <span class="query-text">{display_text}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    csv = f"{csv_header},probability,prediction\n"
    for result in results:
        escaped_value = result[text_key].replace('"', '""')
        prediction = positive_label if result["pred"] else negative_label
        csv += f'"{escaped_value}",{result["prob"]},{prediction}\n'

    st.download_button("Export CSV", data=csv, file_name=file_name, mime="text/csv", key=download_key)


def render_uploaded_text_batch(
    pipeline,
    threshold: float,
    text_key: str,
    csv_header: str,
    file_name: str,
    positive_label: str = "Malicious",
    negative_label: str = "Normal",
) -> None:
    uploaded = st.file_uploader("Batch file", type=["csv", "xlsx"], key=f"{text_key}_batch_upload")
    if uploaded is None:
        return

    try:
        frame = read_batch_table(uploaded)
        values = extract_batch_values(frame, text_key)
        if not values:
            st.warning(f"No batch values found. Include a '{text_key}' column or put values in the first column.")
            return

        probabilities = pipeline.predict_proba(values)
        results = [
            {text_key: item, "prob": probability, "pred": int(probability >= threshold)}
            for item, probability in zip(values, probabilities)
        ]
        render_batch_results(
            results,
            text_key,
            csv_header,
            file_name,
            positive_label,
            negative_label,
            download_key=f"{text_key}_upload_download",
        )
    except Exception as e:
        st.error(f"Could not score batch file: {e}")


def render_csrf_batch(artifact: dict, threshold: float) -> None:
    uploaded = st.file_uploader("Feature file", type=["csv", "xlsx"])
    if uploaded is None:
        return

    try:
        frame = read_batch_table(uploaded)
        frame = frame.drop(columns=[column for column in ("reqId", "flag") if column in frame.columns])
        missing = [column for column in artifact["feature_columns"] if column not in frame.columns]
        if missing:
            st.error(f"Missing feature columns: {missing}")
            return

        feature_frame = frame[artifact["feature_columns"]]
        probabilities = artifact["pipeline"].predict_proba(feature_frame)[:, 1]
        output = frame.copy()
        output["probability"] = probabilities
        output["prediction"] = ["CSRF Relevant" if prob >= threshold else "Not CSRF Relevant" for prob in probabilities]
        display_column = next((column for column in ["request", "query", "path", "url"] if column in frame.columns), None)

        c1, c2, c3 = st.columns(3)
        c1.metric("Total", len(output))
        c2.metric("CSRF Relevant", int((probabilities >= threshold).sum()))
        c3.metric("Not CSRF Relevant", int((probabilities < threshold).sum()))

        for idx, (_, row) in enumerate(output.iterrows(), start=1):
            probability = float(row["probability"])
            is_csrf = probability >= threshold
            label = "CSRF Relevant" if is_csrf else "Not CSRF Relevant"
            cls = "malicious" if is_csrf else "normal"
            badge_cls = "badge-malicious" if is_csrf else "badge-normal"
            color = "#ff4f6d" if is_csrf else "#00e5a0"
            display_text = row[display_column] if display_column else build_csrf_query_from_feature_row(row)

            st.markdown(
                f"""
                <div class="result-card {cls}">
                    <div class="result-main">
                        <span class="badge {badge_cls}">{label.upper()}</span>
                        <div class="prob-bar-bg">
                            <div class="prob-bar-fill" style="width:{probability*100}%;background:{color}"></div>
                        </div>
                        <span style="color:{color}">{probability:.4f}</span>
                        <span class="query-text">{escape(str(display_text))}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            with st.expander(f"Feature row {idx}"):
                st.dataframe(pd.DataFrame([row[artifact["feature_columns"]].to_dict()]), use_container_width=True)

        st.download_button(
            "Export CSV",
            data=output.to_csv(index=False),
            file_name="csrf_results.csv",
            mime="text/csv",
            key="csrf_upload_download",
        )
    except Exception as e:
        st.error(f"Could not score batch file: {e}")


with st.sidebar:
    st.markdown("### Configuration")
    detector = st.selectbox("Detector", ["XSS", "SQL Injection", "CSRF"])

    if detector == "XSS":
        model_path = st.text_input("Model directory", value="xss/saved_models")
    elif detector == "SQL Injection":
        model_path = st.text_input("Model path", value="sqli/ml_project/artifacts/sqli_logreg_model.json")
    else:
        model_path = st.text_input("Model path", value="csrf/artifacts/csrf_boosting_pipeline.joblib")


if detector == "XSS":
    pipeline, load_error = load_xss_pipeline(model_path)
    decision_threshold = 0.5
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
elif detector == "SQL Injection":
    pipeline, load_error = load_sqli_pipeline(model_path)
    decision_threshold = float(getattr(pipeline, "threshold", 0.5))
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
else:
    artifact, load_error = load_csrf_artifact(model_path)
    decision_threshold = float(artifact["threshold"]) if artifact else 0.5
    title = "CSRF Detector"


st.markdown(f"<h1>{title}</h1>", unsafe_allow_html=True)

if load_error:
    st.error(f"Could not load model: {load_error}")
    st.stop()


if detector == "CSRF":
    tab1, tab2 = st.tabs(["Single Request", "Batch CSV"])

    with tab1:
        col1, col2 = st.columns([1, 3])
        with col1:
            method = st.selectbox("Method", ["POST", "GET", "PUT", "DELETE", "OPTIONS"])
        with col2:
            path_or_url = st.text_input("Path or URL", value="/account/password/change")

        raw_params = st.text_area(
            "Parameters",
            value="userId=42&password=new-password&token=&action=change",
            height=120,
        )

        if st.button("Classify"):
            features = build_csrf_features(method, path_or_url, raw_params, artifact["feature_columns"])
            probability = csrf_predict_proba(artifact, [features])[0]
            render_single_result(
                probability,
                decision_threshold,
                positive_label="CSRF Relevant",
                negative_label="Not CSRF Relevant",
            )
            with st.expander("Feature row"):
                st.dataframe(pd.DataFrame([features]), use_container_width=True)

    with tab2:
        render_csrf_batch(artifact, decision_threshold)

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
            render_single_result(probability, decision_threshold)
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
        with st.expander("Items queued for testing"):
            for idx, item in enumerate(st.session_state[queue_key], start=1):
                st.markdown(f"**{idx}.** `{item}`")

        if st.button("Run classification"):
            probabilities = pipeline.predict_proba(st.session_state[queue_key])
            st.session_state[results_key] = [
                {text_key: item, "prob": probability, "pred": int(probability >= decision_threshold)}
                for item, probability in zip(st.session_state[queue_key], probabilities)
            ]

    if st.session_state[results_key]:
        render_batch_results(
            st.session_state[results_key],
            text_key,
            csv_header,
            csv_file_name,
            download_key=f"{text_key}_manual_download",
        )

    st.divider()
    render_uploaded_text_batch(pipeline, decision_threshold, text_key, csv_header, csv_file_name)
