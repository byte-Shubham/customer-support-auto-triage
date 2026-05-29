# frontend/app.py
# Run: streamlit run frontend/app.py
# Install: pip install streamlit requests pandas plotly

import io
import time
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────
BASE_URL = "https://customer-support-auto-triage-production.up.railway.app"
API_KEY  = "capstone-2026"
HEADERS  = {"Content-Type": "application/json", "X-API-Key": API_KEY}

CATEGORY_COLORS = {
    "Technical Issue":    "#2563eb",
    "Billing Inquiry":    "#d97706",
    "Account Management": "#7c3aed",
    "Bug Report":         "#dc2626",
    "Feature Request":    "#059669",
}
SENTIMENT_COLORS = {"Negative": "#dc2626", "Neutral": "#d97706", "Positive": "#059669"}
PRIORITY_COLORS  = {"High": "#dc2626", "Medium": "#d97706", "Low": "#059669"}

# ─────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TriageIQ — Support Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Force sidebar open always
st.markdown("""
<style>
    section[data-testid="stSidebar"] {
        width: 260px !important;
        transform: none !important;
        visibility: visible !important;
    }
    button[data-testid="collapsedControl"] { display: none !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────
# CSS — Professional SaaS look (clean, white, subtle)
# ─────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

* { font-family: 'DM Sans', sans-serif; }

/* ── App background ── */
.stApp { background: #f8fafc; color: #0f172a; }
.stApp > header { background: transparent; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid #e2e8f0;
    padding-top: 0;
}
section[data-testid="stSidebar"] * { color: #334155 !important; }

/* ── Hide streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

/* ── Typography ── */
h1, h2, h3 { font-weight: 700; color: #0f172a; letter-spacing: -0.02em; }

/* ── Cards ── */
.kpi-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 20px 24px;
    transition: box-shadow 0.2s;
}
.kpi-card:hover { box-shadow: 0 4px 16px rgba(0,0,0,0.06); }
.kpi-value {
    font-size: 2rem;
    font-weight: 700;
    color: #0f172a;
    letter-spacing: -0.03em;
    line-height: 1;
    margin-bottom: 4px;
}
.kpi-label {
    font-size: 0.8rem;
    font-weight: 500;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

/* ── Result panel ── */
.result-panel {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    padding: 28px;
    margin: 16px 0;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}

/* ── Tag/Badge ── */
.tag {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 4px 12px;
    border-radius: 6px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.01em;
}

/* ── Reply box ── */
.reply-panel {
    background: #f0f9ff;
    border: 1px solid #bae6fd;
    border-left: 4px solid #0ea5e9;
    border-radius: 10px;
    padding: 18px 20px;
    font-size: 0.9rem;
    line-height: 1.7;
    color: #0c4a6e;
    margin-top: 12px;
}

/* ── Low confidence warning ── */
.flag-banner {
    background: #fffbeb;
    border: 1px solid #fde68a;
    border-left: 4px solid #f59e0b;
    border-radius: 10px;
    padding: 12px 18px;
    font-size: 0.875rem;
    color: #78350f;
    font-weight: 500;
    margin: 12px 0;
}

/* ── Similar ticket item ── */
.similar-row {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 10px 14px;
    margin: 6px 0;
    font-size: 0.85rem;
    color: #334155;
}

/* ── Section divider ── */
.section-label {
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #94a3b8;
    margin: 20px 0 10px;
}

/* ── Confidence bar ── */
.conf-track {
    background: #e2e8f0;
    border-radius: 99px;
    height: 6px;
    margin-top: 8px;
    overflow: hidden;
}
.conf-fill {
    height: 6px;
    border-radius: 99px;
    transition: width 0.6s ease;
}

/* ── Streamlit input styling ── */
.stTextInput input, .stTextArea textarea {
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 8px !important;
    color: #0f172a !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.9rem !important;
    padding: 10px 14px !important;
}
.stTextInput input:focus, .stTextArea textarea:focus {
    border-color: #2563eb !important;
    box-shadow: 0 0 0 3px rgba(37,99,235,0.1) !important;
}

/* ── Button ── */
.stButton > button {
    background: #0f172a !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 10px 24px !important;
    font-weight: 600 !important;
    font-size: 0.875rem !important;
    letter-spacing: 0.01em !important;
    transition: background 0.15s !important;
}
.stButton > button:hover {
    background: #1e293b !important;
}

/* ── Radio nav ── */
div[data-testid="stRadio"] label {
    font-size: 0.875rem !important;
    font-weight: 500 !important;
    padding: 8px 12px !important;
    border-radius: 8px !important;
    color: #475569 !important;
    cursor: pointer !important;
}
div[data-testid="stRadio"] label:hover {
    background: #f1f5f9 !important;
    color: #0f172a !important;
}

/* ── Chart background ── */
.js-plotly-plot { border-radius: 12px; }

/* ── Review queue item ── */
.queue-item {
    background: #ffffff;
    border: 1px solid #fde68a;
    border-left: 4px solid #f59e0b;
    border-radius: 10px;
    padding: 16px 20px;
    margin: 10px 0;
}
.queue-item-title { font-weight: 600; color: #0f172a; font-size: 0.95rem; }
.queue-item-desc { color: #64748b; font-size: 0.875rem; margin-top: 4px; line-height: 1.5; }

/* ── Status dot ── */
.status-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    display: inline-block;
    margin-right: 6px;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────
def api_get(endpoint):
    try:
        r = requests.get(f"{BASE_URL}{endpoint}", timeout=20)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"API error: {e}")
        return None

def api_post(endpoint, payload):
    try:
        r = requests.post(f"{BASE_URL}{endpoint}", json=payload, headers=HEADERS, timeout=40)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"API error: {e}")
        return None

def check_health():
    try:
        r = requests.get(f"{BASE_URL}/health", timeout=6)
        return r.status_code == 200
    except:
        return False

def tag_html(text, color):
    return (f'<span class="tag" style="background:{color}14;'
            f'color:{color};border:1px solid {color}30">{text}</span>')

def conf_bar(pct, color):
    return (f'<div class="conf-track">'
            f'<div class="conf-fill" style="width:{pct}%;background:{color}"></div>'
            f'</div>')

def plotly_theme():
    return dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans", color="#334155", size=12),
        margin=dict(t=16, b=16, l=8, r=8),
    )


# ─────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────
with st.sidebar:
    # Logo area
    st.markdown("""
    <div style="padding:24px 16px 16px;">
        <div style="font-size:1.3rem;font-weight:800;color:#0f172a;letter-spacing:-0.03em">
            ⚡ TriageIQ
        </div>
        <div style="font-size:0.75rem;color:#94a3b8;margin-top:2px;font-weight:500">
            Support Intelligence Platform
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div style="height:1px;background:#e2e8f0;margin:0 16px 16px"></div>',
                unsafe_allow_html=True)

    page = st.radio(
        "nav",
        ["Classify Ticket", "Analytics", "Review Queue", "Batch Upload"],
        label_visibility="collapsed",
    )

    st.markdown('<div style="height:1px;background:#e2e8f0;margin:16px 16px"></div>',
                unsafe_allow_html=True)

    # Health status
    healthy = check_health()
    dot_color = "#059669" if healthy else "#dc2626"
    status_text = "API Online" if healthy else "API Offline"
    st.markdown(f"""
    <div style="padding:0 16px;">
        <span class="status-dot" style="background:{dot_color}"></span>
        <span style="font-size:0.8rem;font-weight:500;color:#475569">{status_text}</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="padding:8px 16px;">
        <span style="font-size:0.75rem;color:#94a3b8">Model: all-MiniLM-L6-v2</span><br>
        <span style="font-size:0.75rem;color:#94a3b8">LLM: Groq · llama-3.1-8b</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div style="position:fixed;bottom:20px;left:0;width:280px;padding:0 20px">'
                '<span style="font-size:0.72rem;color:#cbd5e1">Capstone Project · v2.0</span>'
                '</div>', unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════
# PAGE 1 — Classify Ticket
# ═════════════════════════════════════════════════════════════════════════
if page == "Classify Ticket":
    st.markdown("""
    <div style="margin-bottom:28px">
        <h1 style="font-size:1.6rem;margin-bottom:4px">Classify a Support Ticket</h1>
        <p style="color:#64748b;font-size:0.9rem;margin:0">
            Paste any customer support ticket to get instant category, priority,
            sentiment analysis, and a suggested agent reply.
        </p>
    </div>
    """, unsafe_allow_html=True)

    left, right = st.columns([3, 2], gap="large")

    with left:
        subject = st.text_input("Subject line", placeholder="e.g. Login failed after update")
        description = st.text_area(
            "Ticket description",
            placeholder="e.g. I can't log in since the last app update. It shows a 500 error on the login screen and then the app crashes.",
            height=130,
        )
        col_btn, col_tip = st.columns([1, 2])
        with col_btn:
            classify = st.button("Classify →", use_container_width=True)
        with col_tip:
            st.markdown('<p style="font-size:0.78rem;color:#94a3b8;padding-top:10px">'
                        'Auth: X-API-Key header added automatically</p>',
                        unsafe_allow_html=True)

    with right:
        st.markdown('<div class="section-label">How it works</div>', unsafe_allow_html=True)
        steps = [
            ("01", "Sentence-Transformer encodes your ticket into semantic embeddings"),
            ("02", "3 classifiers predict Category, Priority and Sentiment"),
            ("03", "TF-IDF cosine similarity finds the 3 most similar past tickets"),
            ("04", "Groq LLM drafts a suggested reply for your support agent"),
        ]
        for num, desc in steps:
            st.markdown(f"""
            <div style="display:flex;gap:12px;margin-bottom:12px;align-items:flex-start">
                <span style="font-family:'DM Mono',monospace;font-size:0.7rem;
                             font-weight:500;color:#2563eb;background:#eff6ff;
                             border:1px solid #bfdbfe;border-radius:5px;
                             padding:2px 7px;flex-shrink:0;margin-top:1px">{num}</span>
                <span style="font-size:0.83rem;color:#475569;line-height:1.5">{desc}</span>
            </div>
            """, unsafe_allow_html=True)

    if classify:
        if not subject or not description:
            st.warning("Please fill in both fields.")
        else:
            with st.spinner("Analysing ticket..."):
                result = api_post("/predict", {"Subject": subject, "Description": description})

            if result:
                st.markdown("---")
                cat   = result["predicted_category"]
                pri   = result["predicted_priority"]
                sen   = result["predicted_sentiment"]
                conf  = result["confidence"]
                conf_pct = round(conf * 100, 1)

                cat_c = CATEGORY_COLORS.get(cat, "#334155")
                pri_c = PRIORITY_COLORS.get(pri, "#334155")
                sen_c = SENTIMENT_COLORS.get(sen, "#334155")
                conf_c = "#059669" if conf_pct >= 75 else "#d97706" if conf_pct >= 60 else "#dc2626"

                # KPI row
                k1, k2, k3, k4 = st.columns(4)
                for col, label, value, color in [
                    (k1, "CATEGORY",   cat,             cat_c),
                    (k2, "PRIORITY",   pri,             pri_c),
                    (k3, "SENTIMENT",  sen,             sen_c),
                    (k4, "CONFIDENCE", f"{conf_pct}%",  conf_c),
                ]:
                    with col:
                        st.markdown(f"""
                        <div class="kpi-card">
                            <div class="kpi-label">{label}</div>
                            <div class="kpi-value" style="color:{color};font-size:1.15rem;margin-top:6px">
                                {value}
                            </div>
                            {conf_bar(conf_pct, conf_c) if label == "CONFIDENCE" else ""}
                        </div>""", unsafe_allow_html=True)

                if result.get("requires_review"):
                    st.markdown("""
                    <div class="flag-banner">
                        ⚠ Confidence below 60% — this ticket has been added to the
                        Review Queue for human verification.
                    </div>""", unsafe_allow_html=True)

                st.caption(f"Classified in {result['latency_ms']:.0f} ms")

                # Reply + Similar
                r1, r2 = st.columns([3, 2], gap="large")

                with r1:
                    st.markdown('<div class="section-label">Suggested Agent Reply</div>',
                                unsafe_allow_html=True)
                    if result.get("suggested_reply"):
                        st.markdown(f'<div class="reply-panel">{result["suggested_reply"]}</div>',
                                    unsafe_allow_html=True)
                        st.code(result["suggested_reply"], language=None)
                    else:
                        st.info("LLM reply unavailable — check GROQ_API_KEY.")

                with r2:
                    st.markdown('<div class="section-label">Similar Past Tickets</div>',
                                unsafe_allow_html=True)
                    for t in result.get("similar_tickets", []):
                        cc = CATEGORY_COLORS.get(t["category"], "#334155")
                        st.markdown(f"""
                        <div class="similar-row">
                            <div style="font-weight:600;color:#0f172a;
                                        margin-bottom:4px">{t["subject"]}</div>
                            <div style="display:flex;justify-content:space-between;
                                        align-items:center">
                                {tag_html(t["category"], cc)}
                                <span style="font-family:'DM Mono',monospace;
                                             font-size:0.72rem;color:#94a3b8">
                                    {round(t["score"]*100,0):.0f}% match
                                </span>
                            </div>
                        </div>""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════
# PAGE 2 — Analytics
# ═════════════════════════════════════════════════════════════════════════
elif page == "Analytics":
    top, refresh_col = st.columns([5, 1])
    with top:
        st.markdown("""
        <div style="margin-bottom:24px">
            <h1 style="font-size:1.6rem;margin-bottom:4px">Analytics</h1>
            <p style="color:#64748b;font-size:0.9rem;margin:0">
                Live statistics from all tickets processed by the API.
            </p>
        </div>""", unsafe_allow_html=True)
    with refresh_col:
        if st.button("Refresh"):
            st.rerun()

    data = api_get("/analytics")

    if data:
        total   = data["total_tickets"]
        avg_c   = round(data["avg_confidence"] * 100, 1)
        flagged = data["flagged_for_review"]
        avg_l   = round(data["avg_latency_ms"])

        k1, k2, k3, k4 = st.columns(4)
        for col, label, value, color in [
            (k1, "TOTAL TICKETS",      total,     "#2563eb"),
            (k2, "AVG CONFIDENCE",     f"{avg_c}%", "#059669" if avg_c >= 75 else "#d97706"),
            (k3, "FLAGGED FOR REVIEW", flagged,   "#d97706"),
            (k4, "AVG LATENCY",        f"{avg_l}ms", "#334155"),
        ]:
            with col:
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-label">{label}</div>
                    <div class="kpi-value" style="color:{color}">{value}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

        if total == 0:
            st.info("No tickets yet — classify some tickets to see analytics.")
        else:
            c1, c2 = st.columns(2, gap="large")

            with c1:
                st.markdown('<div class="section-label">Category Distribution</div>',
                            unsafe_allow_html=True)
                cat_df = pd.DataFrame(list(data["category_dist"].items()),
                                      columns=["Category", "Count"])
                fig = px.pie(cat_df, values="Count", names="Category",
                             color="Category",
                             color_discrete_map=CATEGORY_COLORS,
                             hole=0.5)
                fig.update_traces(textposition="outside",
                                  textfont=dict(family="DM Sans", size=11))
                fig.update_layout(**plotly_theme(),
                                  legend=dict(bgcolor="rgba(0,0,0,0)",
                                              font=dict(size=11)))
                st.plotly_chart(fig, use_container_width=True)

            with c2:
                st.markdown('<div class="section-label">Sentiment Breakdown</div>',
                            unsafe_allow_html=True)
                sen_df = pd.DataFrame(list(data["sentiment_dist"].items()),
                                      columns=["Sentiment", "Count"])
                fig2 = px.bar(sen_df, x="Sentiment", y="Count",
                              color="Sentiment",
                              color_discrete_map=SENTIMENT_COLORS,
                              text="Count")
                fig2.update_traces(textposition="outside",
                                   marker_line_width=0,
                                   textfont=dict(family="DM Sans"))
                fig2.update_layout(**plotly_theme(),
                                   showlegend=False,
                                   xaxis=dict(showgrid=False),
                                   yaxis=dict(showgrid=True,
                                              gridcolor="#f1f5f9"))
                st.plotly_chart(fig2, use_container_width=True)

            c3, c4 = st.columns(2, gap="large")

            with c3:
                st.markdown('<div class="section-label">Priority Distribution</div>',
                            unsafe_allow_html=True)
                pri_df = pd.DataFrame(list(data["priority_dist"].items()),
                                      columns=["Priority", "Count"])
                fig3 = px.bar(pri_df, x="Priority", y="Count",
                              color="Priority",
                              color_discrete_map=PRIORITY_COLORS,
                              text="Count",
                              category_orders={"Priority": ["High", "Medium", "Low"]})
                fig3.update_traces(textposition="outside",
                                   marker_line_width=0,
                                   textfont=dict(family="DM Sans"))
                fig3.update_layout(**plotly_theme(),
                                   showlegend=False,
                                   xaxis=dict(showgrid=False),
                                   yaxis=dict(showgrid=True, gridcolor="#f1f5f9"))
                st.plotly_chart(fig3, use_container_width=True)

            with c4:
                st.markdown('<div class="section-label">Tickets Over Time</div>',
                            unsafe_allow_html=True)
                if data["tickets_over_time"]:
                    t_df = pd.DataFrame(data["tickets_over_time"])
                    fig4 = go.Figure()
                    fig4.add_trace(go.Scatter(
                        x=t_df["date"], y=t_df["count"],
                        mode="lines+markers",
                        line=dict(color="#2563eb", width=2.5, shape="spline"),
                        marker=dict(color="#2563eb", size=6),
                        fill="tozeroy",
                        fillcolor="rgba(37,99,235,0.06)",
                    ))
                    fig4.update_layout(**plotly_theme(),
                                       xaxis=dict(showgrid=False),
                                       yaxis=dict(showgrid=True, gridcolor="#f1f5f9"))
                    st.plotly_chart(fig4, use_container_width=True)
                else:
                    st.info("Not enough data yet.")


# ═════════════════════════════════════════════════════════════════════════
# PAGE 3 — Review Queue
# ═════════════════════════════════════════════════════════════════════════
elif page == "Review Queue":
    st.markdown("""
    <div style="margin-bottom:24px">
        <h1 style="font-size:1.6rem;margin-bottom:4px">Review Queue</h1>
        <p style="color:#64748b;font-size:0.9rem;margin:0">
            Tickets where model confidence was below 60% — pending human verification.
        </p>
    </div>""", unsafe_allow_html=True)

    if st.button("Refresh Queue"):
        st.rerun()

    data = api_get("/review")

    if data:
        count = data.get("count", 0)
        queue = data.get("queue", [])

        if count == 0:
            st.markdown("""
            <div style="background:#f0fdf4;border:1px solid #bbf7d0;border-radius:10px;
                        padding:20px 24px;color:#166534;font-weight:500">
                ✓ Review queue is empty — all recent predictions were high confidence.
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="flag-banner">
                {count} ticket{'s' if count > 1 else ''} pending human review
            </div>""", unsafe_allow_html=True)

            for ticket in queue:
                cat_c = CATEGORY_COLORS.get(ticket["pred_category"], "#334155")
                pri_c = PRIORITY_COLORS.get(ticket["pred_priority"], "#334155")
                sen_c = SENTIMENT_COLORS.get(ticket["pred_sentiment"], "#334155")
                conf_pct = round(ticket["confidence"] * 100, 1)
                conf_c = "#d97706" if conf_pct >= 50 else "#dc2626"

                with st.expander(
                    f"#{ticket['id']}  ·  {ticket['subject']}  ·  {conf_pct}% confidence"
                ):
                    d1, d2 = st.columns([4, 1])
                    with d1:
                        st.markdown(f"""
                        <p style="color:#475569;font-size:0.875rem;
                                  line-height:1.6;margin-bottom:14px">
                            {ticket['description']}
                        </p>
                        <div style="display:flex;gap:6px;flex-wrap:wrap;margin-bottom:8px">
                            {tag_html(ticket['pred_category'], cat_c)}
                            {tag_html(ticket['pred_priority'], pri_c)}
                            {tag_html(ticket['pred_sentiment'], sen_c)}
                        </div>
                        <span style="font-size:0.75rem;color:#94a3b8">
                            {ticket['timestamp'][:10]}
                        </span>
                        """, unsafe_allow_html=True)
                    with d2:
                        st.markdown(f"""
                        <div class="kpi-card" style="text-align:center">
                            <div class="kpi-label">CONFIDENCE</div>
                            <div class="kpi-value" style="color:{conf_c}">{conf_pct}%</div>
                            {conf_bar(conf_pct, conf_c)}
                        </div>""", unsafe_allow_html=True)
                        if st.button("Mark reviewed", key=f"rev_{ticket['id']}"):
                            res = api_post("/review/mark-reviewed",
                                          {"ticket_id": ticket["id"]})
                            if res:
                                st.success("Done")
                                time.sleep(0.8)
                                st.rerun()


# ═════════════════════════════════════════════════════════════════════════
# PAGE 4 — Batch Upload
# ═════════════════════════════════════════════════════════════════════════
elif page == "Batch Upload":
    st.markdown("""
    <div style="margin-bottom:24px">
        <h1 style="font-size:1.6rem;margin-bottom:4px">Batch Classification</h1>
        <p style="color:#64748b;font-size:0.9rem;margin:0">
            Upload a CSV with Subject and Description columns to classify
            multiple tickets at once.
        </p>
    </div>""", unsafe_allow_html=True)

    sample = pd.DataFrame([
        {"Subject": "App crashes on startup",
         "Description": "The app crashes immediately after opening on Android 13 since the last update."},
        {"Subject": "Refund not received",
         "Description": "I cancelled my order 10 days ago but still haven't received my refund of Rs 499."},
        {"Subject": "Add dark mode",
         "Description": "Please add a dark mode option to reduce eye strain during night usage."},
        {"Subject": "Wrong amount charged",
         "Description": "I was charged Rs 899 but my plan costs Rs 499. Please refund the difference."},
        {"Subject": "Cannot reset password",
         "Description": "The password reset email never arrives. I've checked spam too. Need urgent access."},
    ])

    dl, info = st.columns([1, 3])
    with dl:
        st.download_button(
            "Download sample CSV",
            data=sample.to_csv(index=False),
            file_name="sample_tickets.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with info:
        st.markdown('<p style="font-size:0.82rem;color:#94a3b8;padding-top:10px">'
                    'CSV must have Subject and Description columns. '
                    'Download the sample to see the format.</p>',
                    unsafe_allow_html=True)

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload CSV file", type=["csv"],
                                 label_visibility="collapsed")

    if uploaded:
        df = pd.read_csv(uploaded)
        if "Subject" not in df.columns or "Description" not in df.columns:
            st.error("CSV must have 'Subject' and 'Description' columns.")
        else:
            st.markdown(f"""
            <div style="background:#eff6ff;border:1px solid #bfdbfe;border-radius:8px;
                        padding:10px 16px;font-size:0.875rem;color:#1e40af;margin-bottom:12px">
                {len(df)} tickets loaded
            </div>""", unsafe_allow_html=True)
            st.dataframe(df.head(), use_container_width=True, hide_index=True)

            if st.button(f"Classify all {len(df)} tickets →", use_container_width=False):
                tickets = df[["Subject","Description"]].fillna("").to_dict(orient="records")
                with st.spinner(f"Classifying {len(tickets)} tickets..."):
                    result = api_post("/predict_batch", {"tickets": tickets})

                if result:
                    preds = result["predictions"]
                    out = df.copy()
                    out["Category"]       = [p["predicted_category"]  for p in preds]
                    out["Priority"]       = [p["predicted_priority"]  for p in preds]
                    out["Sentiment"]      = [p["predicted_sentiment"] for p in preds]
                    out["Confidence (%)"] = [round(p["confidence"]*100,1) for p in preds]
                    out["Needs Review"]   = [p["requires_review"] for p in preds]

                    st.markdown("---")
                    s1, s2, s3 = st.columns(3)
                    flagged = sum(1 for p in preds if p["requires_review"])
                    for col, label, val, color in [
                        (s1, "CLASSIFIED",       result["count"],        "#2563eb"),
                        (s2, "AVG LATENCY",      f"{result['avg_latency_ms']}ms", "#334155"),
                        (s3, "FLAGGED FOR REVIEW", flagged,              "#d97706"),
                    ]:
                        with col:
                            st.markdown(f"""
                            <div class="kpi-card">
                                <div class="kpi-label">{label}</div>
                                <div class="kpi-value" style="color:{color}">{val}</div>
                            </div>""", unsafe_allow_html=True)

                    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
                    st.dataframe(out, use_container_width=True, hide_index=True)

                    st.download_button(
                        "Download results CSV",
                        data=out.to_csv(index=False),
                        file_name="classified_tickets.csv",
                        mime="text/csv",
                        use_container_width=False,
                    )
