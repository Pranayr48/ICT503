#importing libraries
from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

from model_utils import make_features
from network_monitor import get_capture

ROOT = Path(__file__).parent
DATA_PATH = ROOT / "dataset" / "cybersecurity.csv"
MODEL_PATH = ROOT / "artifacts" / "cybersecurity_model.joblib"

st.set_page_config(
    page_title="Sentinel Grid | Threat Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Space+Grotesk:wght@400;500;600;700&display=swap');
    :root { --ink:#eaf3f5; --muted:#8ca5ab; --line:rgba(159,198,205,.16); --cyan:#67e8f9; --mint:#80edc1; --coral:#ff7d73; }
    .stApp { background:radial-gradient(circle at 80% 0%,rgba(18,74,75,.28),transparent 30%),linear-gradient(135deg,#071113 0%,#09191d 50%,#0c1719 100%); color:var(--ink); font-family:'Space Grotesk',sans-serif; }
    [data-testid='stSidebar'] { background:#081416; border-right:1px solid var(--line); }
    [data-testid='stSidebar'] * { font-family:'Space Grotesk',sans-serif; }
    h1,h2,h3 { letter-spacing:0 !important; color:var(--ink); }
    h1 { font-size:2.45rem !important; margin-bottom:.15rem !important; }
    h2 { font-size:1.1rem !important; }
    p,label,.stCaption { color:var(--muted); }
    .eyebrow { color:var(--cyan); font-family:'DM Mono',monospace; font-size:.73rem; letter-spacing:.12em; text-transform:uppercase; margin-bottom:.4rem; }
    .subtitle { color:var(--muted); font-size:1rem; margin-bottom:1.8rem; }
    .status-pill { display:inline-flex; align-items:center; gap:.45rem; border:1px solid rgba(128,237,193,.35); border-radius:999px; color:var(--mint); padding:.42rem .75rem; font-family:'DM Mono',monospace; font-size:.72rem; background:rgba(128,237,193,.07); }
    .status-dot { width:7px; height:7px; border-radius:50%; background:var(--mint); box-shadow:0 0 12px var(--mint); }
    .metric-card { background:linear-gradient(145deg,rgba(18,39,42,.92),rgba(10,25,28,.9)); border:1px solid var(--line); padding:1.05rem 1.15rem; min-height:105px; }
    .metric-label { color:var(--muted); font-family:'DM Mono',monospace; font-size:.68rem; letter-spacing:.1em; text-transform:uppercase; }
    .metric-value { color:var(--ink); font-size:1.9rem; font-weight:700; margin-top:.32rem; }
    .metric-value.alert { color:var(--coral); }
    .section-head { display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid var(--line); padding-bottom:.65rem; margin:1.6rem 0 .9rem; }
    .section-head span { color:var(--muted); font-family:'DM Mono',monospace; font-size:.72rem; }
    .alert-row { border-left:3px solid var(--coral); background:rgba(255,125,115,.07); border-top:1px solid rgba(255,125,115,.14); border-right:1px solid rgba(255,125,115,.14); border-bottom:1px solid rgba(255,125,115,.14); padding:.8rem .95rem; margin:.5rem 0; }
    .alert-row strong { color:#ffd0cb; }
    .alert-row small { color:var(--muted); font-family:'DM Mono',monospace; }
    .briefing { border:1px solid rgba(103,232,249,.22); background:linear-gradient(120deg,rgba(14,48,52,.82),rgba(9,27,30,.75)); padding:1.35rem 1.5rem; margin:1.8rem 0 1rem; }
    .briefing-kicker { color:var(--cyan); font-family:'DM Mono',monospace; font-size:.7rem; letter-spacing:.13em; text-transform:uppercase; }
    .briefing-title { color:var(--ink); font-size:1.55rem; font-weight:600; margin:.45rem 0 .45rem; }
    .briefing-copy { color:#b7cdd0; max-width:790px; line-height:1.55; margin:0; }
    .evidence { border-left:3px solid var(--coral); background:rgba(255,125,115,.08); padding:1rem 1.15rem; margin:.85rem 0 1.2rem; color:#ffd0cb; font-size:1.03rem; line-height:1.45; }
    .evidence small { display:block; color:var(--muted); font-family:'DM Mono',monospace; font-size:.68rem; margin-top:.45rem; }
    .result-grid { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:.75rem; margin:.8rem 0 1.2rem; }
    .result-card { border:1px solid var(--line); background:rgba(13,31,34,.78); padding:1rem; min-height:145px; }
    .result-number { color:var(--cyan); font-family:'DM Mono',monospace; font-size:.72rem; }
    .result-name { color:var(--ink); font-size:1.05rem; font-weight:600; text-transform:capitalize; margin:.55rem 0 .85rem; }
    .result-count { color:var(--coral); font-size:1.45rem; font-weight:700; }
    .result-meta { color:var(--muted); font-family:'DM Mono',monospace; font-size:.66rem; line-height:1.6; }
    .result-action { color:#b7cdd0; font-size:.78rem; margin-top:.65rem; }
    @media (max-width: 900px) { .result-grid { grid-template-columns:1fr; } }
    div[data-testid='stButton'] > button { border:1px solid rgba(103,232,249,.45); background:rgba(103,232,249,.1); color:var(--cyan); border-radius:3px; font-family:'DM Mono',monospace; }
    div[data-testid='stButton'] > button:hover { border-color:var(--cyan); color:#fff; background:rgba(103,232,249,.18); }
    .block-container { padding-top:2.8rem; padding-bottom:3rem; max-width:1500px; }
    [data-testid='stDataFrame'] { border:1px solid var(--line); }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    artifact = joblib.load(MODEL_PATH)
    return artifact["model"] if isinstance(artifact, dict) else artifact


@st.cache_data
def load_events():
    frame = pd.read_csv(DATA_PATH)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    return frame.sort_values("timestamp").reset_index(drop=True)


def predict(frame: pd.DataFrame) -> pd.DataFrame:
    model = load_model()
    if model is None:
        return frame.assign(prediction=0, risk_score=0.0)
    probabilities = model.predict_proba(make_features(frame))[:, 1]
    result = frame.copy()
    result["risk_score"] = probabilities
    result["prediction"] = (probabilities >= 0.5).astype(int)
    return result


def build_threat_map(frame: pd.DataFrame, confidence: float):
    """Create a 3D traffic map where model alerts are visually isolated."""
    plot_frame = frame.copy()
    plot_frame["sent_volume"] = (plot_frame["bytes_sent"].fillna(0) + 1).map(
        lambda value: __import__("math").log10(value)
    )
    plot_frame["received_volume"] = (plot_frame["bytes_received"].fillna(0) + 1).map(
        lambda value: __import__("math").log10(value)
    )
    plot_frame["destination_port"] = pd.to_numeric(
        plot_frame["dst_port"], errors="coerce"
    ).fillna(0)
    plot_frame["status"] = plot_frame["risk_score"].ge(confidence).map(
        {True: "ATTACK", False: "CLEAR"}
    )
    plot_frame["risk_label"] = plot_frame["risk_score"].map(
        lambda value: f"{value:.1%} risk"
    )
    plot_frame["observed"] = plot_frame["timestamp"].dt.strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    plot_frame["traffic"] = (
        plot_frame["src_ip"].astype(str)
        + " → "
        + plot_frame["dst_ip"].astype(str)
    )
    figure = px.scatter_3d(
        plot_frame,
        x="sent_volume",
        y="received_volume",
        z="destination_port",
        color="status",
        color_discrete_map={"CLEAR": "#67e8f9", "ATTACK": "#ff6259"},
        symbol="status",
        size="risk_score",
        size_max=15,
        hover_name="traffic",
        hover_data={
            "observed": True,
            "protocol": True,
            "risk_label": True,
            "src_ip": True,
            "dst_ip": True,
            "sent_volume": False,
            "received_volume": False,
            "destination_port": False,
            "status": False,
            "risk_score": False,
        },
        labels={
            "sent_volume": "Sent bytes · log10",
            "received_volume": "Received bytes · log10",
            "destination_port": "Destination port",
            "risk_label": "Model risk",
        },
        template="plotly_dark",
    )
    figure.update_layout(
        height=590,
        margin={"l": 0, "r": 0, "t": 12, "b": 0},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend={"title": "MODEL LABEL", "orientation": "h", "y": 1.04},
        scene={
            "bgcolor": "rgba(0,0,0,0)",
            "xaxis": {"gridcolor": "rgba(159,198,205,.16)"},
            "yaxis": {"gridcolor": "rgba(159,198,205,.16)"},
            "zaxis": {"gridcolor": "rgba(159,198,205,.16)"},
        },
    )
    return figure


def render_results_briefing(frame: pd.DataFrame):
    """Render a numbered, evidence-led summary of observed attack results."""
    attacks = frame[
        frame["attack_type"].notna()
        & ~frame["attack_type"].isin(["benign", "unlabeled", "unknown"])
    ].copy()
    if attacks.empty:
        st.info("No labeled attack categories are present in the current results window.")
        return

    ranked = (
        attacks.groupby("attack_type", as_index=False)
        .agg(events=("attack_type", "size"), mean_risk=("risk_score", "mean"))
        .sort_values(["events", "mean_risk"], ascending=False)
        .head(6)
    )
    top = ranked.iloc[0]
    attack_share = len(attacks) / len(frame)
    st.markdown(
        f"""
        <div class='briefing'>
            <div class='briefing-kicker'>DETECTION RESULTS / EXECUTIVE BRIEFING</div>
            <div class='briefing-title'>The current window contains {len(attacks):,} labeled attack events across {ranked.shape[0]} priority patterns.</div>
            <p class='briefing-copy'>Results are ranked by observed event volume and paired with the model's average risk score. Use the 3D map above to move from this summary into individual source and destination events.</p>
        </div>
        <div class='evidence'>
            <strong>Primary signal: {top['attack_type'].replace('-', ' ').title()}</strong> accounts for {int(top['events']):,} events ({top['events'] / len(attacks):.1%} of labeled attacks) with an average model risk of {top['mean_risk']:.1%}.
            <small>ATTACK SHARE OF CURRENT WINDOW · {attack_share:.1%}</small>
        </div>
        """,
        unsafe_allow_html=True,
    )
    cards = []
    for index, row in enumerate(ranked.itertuples(index=False), start=1):
        action = {
            "brute-force": "Review authentication bursts",
            "port-scan": "Inspect exposed services",
            "sql-injection": "Validate query boundaries",
            "xss": "Review input sanitization",
            "credential-stuffing": "Protect reused credentials",
            "ddos": "Check traffic saturation",
        }.get(row.attack_type, "Open investigation queue")
        cards.append(
            f"<div class='result-card'><div class='result-number'>0{index} / RESULT</div><div class='result-name'>{row.attack_type.replace('-', ' ')}</div><div class='result-count'>{row.events:,} <span class='result-meta'>events</span></div><div class='result-meta'>AVG MODEL RISK · {row.mean_risk:.1%}</div><div class='result-action'>{action} →</div></div>"
        )
    st.markdown("<div class='result-grid'>" + "".join(cards) + "</div>", unsafe_allow_html=True)


def render_manual_analysis(confidence: float, reference_data: pd.DataFrame):
    st.markdown(
        "<div class='briefing'><div class='briefing-kicker'>MANUAL ANALYSIS / MODEL CHECK</div><div class='briefing-title'>Test a network event before it reaches the queue.</div><p class='briefing-copy'>Enter one event using the same fields as the training data. The fitted preprocessing pipeline will return a risk score and an operational label.</p></div>",
        unsafe_allow_html=True,
    )
    with st.form("manual_event_form", clear_on_submit=False):
        first, second, third = st.columns(3)
        with first:
            timestamp = st.text_input("Timestamp", "2025-10-01 12:00:00")
            src_ip = st.text_input("Source IP", "188.176.27.165")
            dst_ip = st.text_input("Destination IP", "253.240.113.218")
            protocol = st.selectbox("Protocol", ["TCP", "UDP", "ICMP"])
        with second:
            src_port = st.number_input("Source port", min_value=0, max_value=65535, value=56377)
            dst_port = st.number_input("Destination port", min_value=0, max_value=65535, value=443)
            bytes_sent = st.number_input("Bytes sent", min_value=0, value=8029)
            bytes_received = st.number_input("Bytes received", min_value=0, value=17204)
        with third:
            user_agent = st.text_input("User agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64)")
            url = st.text_input("URL", "https://webmail.corp/login?id=385071")
            is_internal_traffic = st.checkbox("Internal traffic", value=False)
        submitted = st.form_submit_button("Run model analysis", use_container_width=True)

    if submitted:
        manual_row = pd.DataFrame(
            [{
                "timestamp": timestamp,
                "src_ip": src_ip,
                "dst_ip": dst_ip,
                "src_port": src_port,
                "dst_port": dst_port,
                "protocol": protocol,
                "bytes_sent": bytes_sent,
                "bytes_received": bytes_received,
                "user_agent": user_agent,
                "url": url,
                "is_internal_traffic": is_internal_traffic,
            }]
        )
        result = predict(manual_row)
        risk_score = float(result.iloc[0]["risk_score"])
        label = "ATTACK" if risk_score >= confidence else "NORMAL"
        if label == "ATTACK":
            st.error(f"ATTACK DETECTED · model risk {risk_score:.1%}")
            st.warning("Recommended response: isolate the source, preserve the event, and open an investigation.")
        else:
            st.success(f"NORMAL ACTIVITY · model risk {risk_score:.1%}")
        result_view = manual_row.copy()
        result_view.insert(0, "model_label", label)
        result_view.insert(1, "risk_score", f"{risk_score:.1%}")
        st.dataframe(result_view, use_container_width=True, hide_index=True)

    st.markdown(
        "<div class='section-head'><h2>Reference dataset</h2><span>SELECT · COPY · PASTE INTO THE FORM</span></div>",
        unsafe_allow_html=True,
    )
    st.caption("The table below contains the original event values used by the model. Click a cell to copy its value, or download the full reference CSV.")
    st.download_button(
        "Download reference CSV",
        reference_data.to_csv(index=False).encode("utf-8"),
        "cybersecurity-reference.csv",
        "text/csv",
    )
    st.dataframe(reference_data.head(100), use_container_width=True, hide_index=True, height=430)


model = load_model()
events = load_events()
capture = get_capture()
if model is None:
    st.error("Model artifact is missing. Run `python train_model.py` first.")
    st.stop()

if "feed_cursor" not in st.session_state:
    st.session_state.feed_cursor = min(25, len(events))
if "processed" not in st.session_state:
    st.session_state.processed = predict(events.iloc[: st.session_state.feed_cursor])

with st.sidebar:
    st.markdown("<div class='eyebrow'>SENTINEL GRID / CONTROL</div>", unsafe_allow_html=True)
    st.markdown("### Monitoring console")
    st.caption("Model-backed network activity watch")
    st.divider()
    st.markdown("**Feed source**")
    monitor_mode = st.radio(
        "Monitoring source",
        ["Live network capture", "Dataset replay"],
        index=1,
        label_visibility="collapsed",
    )
    batch_size = st.slider("Packets per scan", 1, 25, 8)
    confidence = st.slider("Alert confidence", 0.50, 0.99, 0.70, 0.01)
    if monitor_mode == "Live network capture":
        st.caption("Captures IP traffic from this computer using Scapy/Npcap.")
        if not capture.available:
            st.warning("Install Scapy and Npcap to enable packet capture.")
        if capture.running:
            st.success("LIVE CAPTURE RUNNING")
            if st.button("■  Stop live capture", use_container_width=True):
                capture.stop()
                st.rerun()
        else:
            if st.button("▶  Start live capture", use_container_width=True):
                try:
                    capture.start()
                    st.rerun()
                except RuntimeError as error:
                    st.error(f"Capture could not start: {error}")
        if st.button("↻  Refresh live packets", use_container_width=True):
            st.rerun()
    else:
        st.caption("Replays the supplied CSV for demonstration purposes.")
        if st.button("▶  Scan next activity", use_container_width=True):
            start = st.session_state.feed_cursor
            end = min(start + batch_size, len(events))
            if start < end:
                new_events = predict(events.iloc[start:end])
                st.session_state.processed = pd.concat(
                    [st.session_state.processed, new_events], ignore_index=True
                )
                st.session_state.feed_cursor = end
                new_alerts = new_events[new_events["risk_score"] >= confidence]
                if len(new_alerts):
                    st.toast(f"{len(new_alerts)} threat alert(s) detected", icon="🚨")
                else:
                    st.toast("Scan complete. No high-confidence threats.", icon="✅")
            else:
                st.toast("End of available activity feed.", icon="ℹ️")
    if monitor_mode == "Dataset replay":
        if st.button("↺  Reset activity feed", use_container_width=True):
            st.session_state.feed_cursor = min(25, len(events))
            st.session_state.processed = predict(events.iloc[: st.session_state.feed_cursor])
            st.rerun()
    if monitor_mode == "Live network capture":
        live_records = capture.recent()
        if live_records:
            st.session_state.processed = predict(pd.DataFrame(live_records))
    st.divider()
    st.markdown("**Model status**")
    st.caption("ExtraTreesClassifier · 400 estimators")
    st.caption(f"{len(events):,} records available in dataset")

if monitor_mode == "Live network capture" and not capture.recent():
    st.markdown("<div class='eyebrow'>LIVE NETWORK CAPTURE / WAITING</div>", unsafe_allow_html=True)
    st.title("Live capture is ready when packets arrive.")
    if not capture.available:
        st.error("Scapy is not installed. Install the project requirements, then restart Streamlit.")
    elif not capture.running:
        st.info("Click Start live capture in the sidebar, then generate network activity and click Refresh live packets.")
    else:
        st.info("Capture is running. Generate network activity, then click Refresh live packets.")
    st.stop()

processed = st.session_state.processed.copy()
processed["is_alert"] = processed["risk_score"] >= confidence
alerts = processed[processed["is_alert"]].sort_values("risk_score", ascending=False)
attack_count = int(processed["prediction"].sum())
alert_rate = attack_count / len(processed) if len(processed) else 0
latest = processed.iloc[-1]

st.markdown("<div class='eyebrow'>THREAT OPERATIONS CENTER / LIVE FEED</div>", unsafe_allow_html=True)
st.title("See the signal before it becomes an incident.")
mode_label = "LIVE NETWORK CAPTURE" if monitor_mode == "Live network capture" else "DATASET REPLAY"
st.markdown(f"<div class='subtitle'>A calm, model-backed view of the network surface, tuned for immediate response.</div>", unsafe_allow_html=True)
st.markdown(f"<div class='status-pill'><span class='status-dot'></span> MODEL ONLINE · {mode_label}</div>", unsafe_allow_html=True)

monitor_tab, manual_tab = st.tabs(["Live Monitor", "Manual Analysis"])

with manual_tab:
    render_manual_analysis(confidence, events)

metric_cols = st.columns(4)
metrics = [
    ("Events observed", f"{len(processed):,}", ""),
    ("Threats flagged", f"{attack_count:,}", "alert" if attack_count else ""),
    ("Threat rate", f"{alert_rate:.1%}", "alert" if alert_rate > .1 else ""),
    ("Latest risk score", f"{latest['risk_score']:.1%}", "alert" if latest["is_alert"] else ""),
]
for column, (label, value, tone) in zip(metric_cols, metrics):
    column.markdown(
        f"<div class='metric-card'><div class='metric-label'>{label}</div><div class='metric-value {tone}'>{value}</div></div>",
        unsafe_allow_html=True,
    )

with monitor_tab:
    left, right = st.columns([1.5, 1], gap="large")
    with left:
        st.markdown("<div class='section-head'><h2>Risk activity</h2><span>LAST 50 EVENTS</span></div>", unsafe_allow_html=True)
        chart = processed[["timestamp", "risk_score"]].tail(50).set_index("timestamp")
        st.line_chart(chart, y="risk_score", color="#67e8f9", height=260)
    with right:
        st.markdown("<div class='section-head'><h2>Response queue</h2><span>PRIORITIZED</span></div>", unsafe_allow_html=True)
        if alerts.empty:
            st.success("No high-confidence alerts in the current window.")
        else:
            for _, alert in alerts.head(4).iterrows():
                st.markdown(
                    f"<div class='alert-row'><strong>Threat signal · {alert['risk_score']:.1%}</strong><br><small>{alert['src_ip']} → {alert['dst_ip']} · port {int(alert['dst_port'])} · {alert['protocol']}</small></div>",
                    unsafe_allow_html=True,
                )

    st.markdown(
        "<div class='section-head'><h2>Threat coordinate map</h2><span>ROTATE · ZOOM · HOVER TO PINPOINT</span></div>",
        unsafe_allow_html=True,
    )
    st.caption(
        "Attack labels are model decisions at the selected confidence threshold. Red points are the events to investigate first."
    )
    st.plotly_chart(build_threat_map(processed, confidence), use_container_width=True)

    render_results_briefing(processed)

    st.markdown("<div class='section-head'><h2>Activity ledger</h2><span>NEWEST FIRST · SCAN WINDOW</span></div>", unsafe_allow_html=True)
    view = processed.sort_values("timestamp", ascending=False).head(16).copy()
    view["timestamp"] = view["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    view["status"] = view["is_alert"].map({True: "ALERT", False: "CLEAR"})
    view["risk_score"] = view["risk_score"].map(lambda value: f"{value:.1%}")
    st.dataframe(
        view[["timestamp", "src_ip", "dst_ip", "protocol", "dst_port", "status", "risk_score"]],
        use_container_width=True,
        hide_index=True,
        column_config={
            "timestamp": "Observed",
            "src_ip": "Source",
            "dst_ip": "Destination",
            "dst_port": "Port",
            "risk_score": "Risk",
        },
    )
    st.caption("This dashboard demonstrates model-assisted detection over the supplied CSV feed. Connect the scan action to your live event stream for production response workflows.")
