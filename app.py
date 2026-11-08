"""
app.py — QueryMind AI: Streamlit UI entry point.

Premium dark-themed conversational database management interface.
Run with: streamlit run app.py
"""

import os
import time
import traceback

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# Local modules
import database as db
import ai_engine as ai
import rag as rag_module
import utils

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="QueryMind AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS — dark glassmorphism theme + animations
# ─────────────────────────────────────────────────────────────────────────────

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Global ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0a0a0f;
    color: #e0e0e0;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0a0a0f; }
::-webkit-scrollbar-thumb { background: #3b3b5c; border-radius: 3px; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d0d1a 0%, #111128 100%);
    border-right: 1px solid rgba(99,102,241,0.2);
}
[data-testid="stSidebar"] .block-container { padding-top: 1rem; }

/* ── Hero header ── */
.hero-container {
    background: linear-gradient(135deg, #0f0f23 0%, #1a1a3e 40%, #0d1a2e 100%);
    border: 1px solid rgba(99,102,241,0.3);
    border-radius: 20px;
    padding: 2.5rem 2rem 2rem 2rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.hero-container::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle at 30% 50%, rgba(99,102,241,0.08) 0%, transparent 60%),
                radial-gradient(circle at 80% 20%, rgba(168,85,247,0.06) 0%, transparent 50%);
    animation: heroGlow 8s ease-in-out infinite alternate;
    pointer-events: none;
}
@keyframes heroGlow {
    0%  { transform: translate(0, 0) scale(1); }
    100%{ transform: translate(2%, 2%) scale(1.05); }
}
.hero-title {
    font-size: 2.4rem;
    font-weight: 800;
    background: linear-gradient(135deg, #818cf8, #c084fc, #38bdf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 0.3rem 0;
    line-height: 1.1;
}
.hero-subtitle {
    font-size: 1.05rem;
    color: #94a3b8;
    font-weight: 300;
    margin: 0 0 0.5rem 0;
    letter-spacing: 0.5px;
}
.hero-tagline {
    font-size: 0.85rem;
    color: rgba(129,140,248,0.7);
    font-weight: 400;
}

/* ── Floating particles ── */
.particle { position: absolute; border-radius: 50%; animation: float linear infinite; }
@keyframes float {
    0%   { transform: translateY(100%) translateX(0); opacity: 0; }
    10%  { opacity: 1; }
    90%  { opacity: 1; }
    100% { transform: translateY(-100px) translateX(20px); opacity: 0; }
}

/* ── Cards / panels ── */
.glass-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 1rem;
    backdrop-filter: blur(10px);
    transition: border-color 0.3s ease, transform 0.2s ease;
}
.glass-card:hover {
    border-color: rgba(99,102,241,0.4);
    transform: translateY(-1px);
}

/* ── Chat bubbles ── */
.chat-user {
    background: linear-gradient(135deg, rgba(99,102,241,0.25), rgba(99,102,241,0.1));
    border: 1px solid rgba(99,102,241,0.35);
    border-radius: 16px 16px 4px 16px;
    padding: 0.9rem 1.1rem;
    margin: 0.5rem 0 0.5rem 15%;
    animation: slideInRight 0.3s ease;
}
.chat-ai {
    background: linear-gradient(135deg, rgba(16,185,129,0.15), rgba(16,185,129,0.05));
    border: 1px solid rgba(16,185,129,0.25);
    border-radius: 16px 16px 16px 4px;
    padding: 0.9rem 1.1rem;
    margin: 0.5rem 15% 0.5rem 0;
    animation: slideInLeft 0.3s ease;
}
@keyframes slideInRight { from { opacity:0; transform:translateX(20px); } to { opacity:1; transform:translateX(0); } }
@keyframes slideInLeft  { from { opacity:0; transform:translateX(-20px); } to { opacity:1; transform:translateX(0); } }

.chat-label {
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-bottom: 0.4rem;
    opacity: 0.6;
}
.chat-content { font-size: 0.95rem; line-height: 1.6; }

/* ── SQL preview box ── */
.sql-box {
    background: #0d1117;
    border: 1px solid rgba(99,102,241,0.4);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.88rem;
    color: #79c0ff;
    white-space: pre-wrap;
    word-break: break-all;
    line-height: 1.6;
    margin: 0.5rem 0;
}

/* ── Status badges ── */
.badge {
    display: inline-block;
    padding: 0.2rem 0.65rem;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.5px;
}
.badge-success { background: rgba(16,185,129,0.2); color: #34d399; border: 1px solid rgba(16,185,129,0.3); }
.badge-error   { background: rgba(239,68,68,0.2);  color: #f87171; border: 1px solid rgba(239,68,68,0.3); }
.badge-warning { background: rgba(245,158,11,0.2); color: #fbbf24; border: 1px solid rgba(245,158,11,0.3); }
.badge-info    { background: rgba(99,102,241,0.2); color: #818cf8; border: 1px solid rgba(99,102,241,0.3); }

/* ── Suggested prompts ── */
.prompt-chip {
    display: inline-block;
    background: rgba(99,102,241,0.1);
    border: 1px solid rgba(99,102,241,0.3);
    border-radius: 20px;
    padding: 0.4rem 0.9rem;
    font-size: 0.82rem;
    color: #818cf8;
    margin: 0.2rem;
    cursor: pointer;
    transition: all 0.2s ease;
}
.prompt-chip:hover {
    background: rgba(99,102,241,0.25);
    border-color: rgba(99,102,241,0.6);
    transform: translateY(-1px);
}

/* ── Section headers ── */
.section-header {
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    color: rgba(129,140,248,0.6);
    margin: 1rem 0 0.5rem 0;
    padding-bottom: 0.3rem;
    border-bottom: 1px solid rgba(129,140,248,0.15);
}

/* ── Streamlit overrides ── */
.stButton > button {
    border-radius: 10px !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover { transform: translateY(-1px) !important; }
div[data-testid="stMetricValue"] { font-size: 1.5rem !important; color: #818cf8 !important; }
.stDataFrame { border-radius: 12px; overflow: hidden; }
.stTextArea textarea {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(99,102,241,0.3) !important;
    border-radius: 12px !important;
    color: #e0e0e0 !important;
    font-family: 'Inter', sans-serif !important;
}
.stTextArea textarea:focus { border-color: rgba(99,102,241,0.7) !important; }

/* ── Divider ── */
.fancy-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(99,102,241,0.4), transparent);
    margin: 1rem 0;
}

/* ── Sidebar brand ── */
.sidebar-brand {
    text-align: center;
    padding: 1rem 0;
    border-bottom: 1px solid rgba(99,102,241,0.2);
    margin-bottom: 1rem;
}
.brand-icon { font-size: 2.5rem; }
.brand-name {
    font-size: 1.3rem;
    font-weight: 800;
    background: linear-gradient(135deg, #818cf8, #c084fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.brand-tagline { font-size: 0.72rem; color: #64748b; }

/* ── Spinner override ── */
.stSpinner > div { border-top-color: #818cf8 !important; }

/* ── Confirmation warning box ── */
.confirm-box {
    background: rgba(245,158,11,0.1);
    border: 1px solid rgba(245,158,11,0.4);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin: 0.5rem 0;
}

/* ── Metrics row ── */
.metric-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px;
    padding: 0.9rem;
    text-align: center;
}
.metric-value {
    font-size: 1.6rem;
    font-weight: 700;
    color: #818cf8;
}
.metric-label {
    font-size: 0.72rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.8px;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Session state initialisation
# ─────────────────────────────────────────────────────────────────────────────

def init_session():
    defaults = {
        "memory": ai.ConversationMemory(max_turns=6),
        "chat_history": [],       # list of {role, content, sql?, df?}
        "query_history": [],      # list of SQL strings
        "pending_sql": None,      # SQL awaiting confirmation
        "pending_question": None, # original user question for pending SQL
        "last_df": None,          # last query result
        "db_ok": False,
        "api_ok": False,
        "api_msg": "",
        "rag": None,
        "tables": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()

# ─────────────────────────────────────────────────────────────────────────────
# Startup: init DB + API + RAG
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def startup():
    """Run once: initialise database, check API, init RAG."""
    try:
        db.init_db()
    except Exception as e:
        return False, False, f"DB error: {e}", None

    api_ok, api_msg = ai.check_api_connection()

    try:
        rag_inst = rag_module.get_rag()
        full_schema = db.get_full_schema()
        if full_schema and api_ok:
            rag_inst.index_schema(full_schema)
    except Exception:
        rag_inst = None

    return True, api_ok, api_msg, rag_inst


db_ok, api_ok, api_msg, rag_inst = startup()
st.session_state.db_ok  = db_ok
st.session_state.api_ok = api_ok
st.session_state.api_msg = api_msg
st.session_state.rag    = rag_inst

# Refresh table list
try:
    st.session_state.tables = db.get_tables()
except Exception:
    st.session_state.tables = []

# ─────────────────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────────────────

def add_chat(role: str, content: str, sql: str = None, df=None):
    st.session_state.chat_history.append(
        {"role": role, "content": content, "sql": sql, "df": df}
    )


def get_schema_context(question: str) -> str:
    """Get schema context: try RAG first, fall back to full schema text."""
    rag = st.session_state.rag
    if rag and st.session_state.api_ok:
        try:
            ctx = rag.get_relevant_schema(question, top_k=3)
            if ctx:
                return ctx
        except Exception:
            pass
    return db.get_full_schema_text()


def handle_query_pipeline(user_question: str):
    """Full pipeline: generate SQL → validate → check safety → store pending."""
    tables = db.get_tables()
    if not tables:
        add_chat("ai", "⚠️ No tables found. Please upload a dataset first.")
        return

    schema_ctx = get_schema_context(user_question)

    with st.spinner("🧠 Generating SQL..."):
        try:
            raw_sql = ai.generate_sql(user_question, schema_ctx, st.session_state.memory)
        except Exception as e:
            add_chat("ai", f"❌ AI error: {e}")
            return

    if raw_sql == "CANNOT_GENERATE":
        add_chat("ai", "😕 I couldn't generate a SQL query for that question. Please rephrase or check the schema.")
        return

    # Clean any stray markdown
    sql = utils.clean_sql(raw_sql)

    # Validate
    valid, err = utils.validate_sql(sql)
    if not valid:
        add_chat("ai", f"❌ Invalid SQL generated:\n`{err}`\n\nTry rephrasing your question.")
        return

    # Check for blocked DDL
    if utils.is_destructive(sql):
        add_chat("ai", "🚫 This operation is blocked for safety (DROP, ALTER, TRUNCATE, CREATE are not allowed).")
        return

    # Check if confirmation needed (UPDATE / DELETE / INSERT)
    if utils.requires_confirmation(sql):
        st.session_state.pending_sql      = sql
        st.session_state.pending_question = user_question
        add_chat("ai",
                 "⚠️ This query will modify data. Please review the SQL below and confirm.",
                 sql=sql)
        return

    # SELECT — execute immediately
    execute_and_display(sql, user_question)


def execute_and_display(sql: str, question: str):
    """Execute a SQL query and render results in chat."""
    with st.spinner("⚡ Executing query..."):
        try:
            result_df = db.execute_query(sql)
        except Exception as e:
            add_chat("ai", f"❌ Execution error: {e}", sql=sql)
            return

    # Update conversation memory
    st.session_state.memory.add(question, sql)

    # Store for context and export
    st.session_state.last_df = result_df

    # Add to query history
    if sql not in st.session_state.query_history:
        st.session_state.query_history.insert(0, sql)
    st.session_state.query_history = st.session_state.query_history[:20]

    n = len(result_df)
    summary = f"✅ Query returned **{n}** row{'s' if n != 1 else ''}."
    add_chat("ai", summary, sql=sql, df=result_df)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    # Brand
    st.markdown("""
    <div class="sidebar-brand">
        <div class="brand-icon">🧠</div>
        <div class="brand-name">QueryMind AI</div>
        <div class="brand-tagline">Natural Language · SQL · Insights</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Status badges ──
    st.markdown('<p class="section-header">System Status</p>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        if st.session_state.db_ok:
            st.markdown('<span class="badge badge-success">🗄️ DB Connected</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="badge badge-error">🗄️ DB Error</span>', unsafe_allow_html=True)
    with col2:
        if st.session_state.api_ok:
            st.markdown('<span class="badge badge-success">🤖 AI Ready</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="badge badge-error">🤖 AI Offline</span>', unsafe_allow_html=True)

    if not st.session_state.api_ok:
        st.warning(f"Gemini: {st.session_state.api_msg}", icon="⚠️")

    # ── Upload dataset ──
    st.markdown('<p class="section-header">Upload Dataset</p>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "CSV or Excel",
        type=["csv", "xlsx", "xls"],
        label_visibility="collapsed",
    )

    if uploaded_file:
        with st.spinner("Processing dataset…"):
            try:
                fname = uploaded_file.name
                if fname.endswith(".csv"):
                    tname, preview_df = db.upload_csv(uploaded_file, fname)
                else:
                    tname, preview_df = db.upload_excel(uploaded_file, fname)

                st.session_state.tables = db.get_tables()

                # Re-index schema in RAG
                if st.session_state.rag and st.session_state.api_ok:
                    try:
                        st.session_state.rag.refresh(db.get_full_schema())
                    except Exception:
                        pass

                st.success(f"✅ Uploaded as table: `{tname}`")
                with st.expander("📋 Preview (5 rows)", expanded=False):
                    st.dataframe(preview_df.head(5), use_container_width=True)
            except Exception as e:
                st.error(f"Upload failed: {e}")

    # ── Tables ──
    st.markdown('<p class="section-header">Database Tables</p>', unsafe_allow_html=True)
    tables = st.session_state.tables
    if tables:
        for t in tables:
            with st.expander(f"📊 {t}", expanded=False):
                try:
                    schema = db.get_schema(t)
                    for col in schema["columns"]:
                        pk = " 🔑" if col["primary_key"] else ""
                        st.markdown(f"`{col['name']}` — *{col['type']}*{pk}")
                    row_count = db.get_row_count(t)
                    st.markdown(f"<small>{row_count} rows</small>", unsafe_allow_html=True)
                except Exception:
                    st.markdown("*Could not load schema*")
    else:
        st.markdown("<small style='color:#64748b'>No tables yet — upload a dataset.</small>", unsafe_allow_html=True)

    # ── Query history ──
    if st.session_state.query_history:
        st.markdown('<p class="section-header">Query History</p>', unsafe_allow_html=True)
        for i, q in enumerate(st.session_state.query_history[:8]):
            short = q[:55] + "…" if len(q) > 55 else q
            if st.button(short, key=f"hist_{i}", use_container_width=True):
                # Re-run historical query
                execute_and_display(q, q)
                st.rerun()

    # ── Settings ──
    st.markdown('<p class="section-header">Settings</p>', unsafe_allow_html=True)
    max_rows = st.slider("Max result rows", 10, 1000, 100, 10)
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.memory.clear()
        st.session_state.pending_sql = None
        st.session_state.last_df = None
        st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# MAIN AREA
# ─────────────────────────────────────────────────────────────────────────────

# ── Hero header ──
st.markdown("""
<div class="hero-container">
    <div class="hero-title">🧠 QueryMind AI</div>
    <div class="hero-subtitle">Your data. Your language. Your insights.</div>
    <div class="hero-tagline">
        Upload any dataset → ask questions in plain English → get instant SQL, results &amp; charts.
    </div>
</div>
""", unsafe_allow_html=True)

# ── Suggested prompts (shown when no chat history) ──
SUGGESTED_PROMPTS = [
    "Show all employees earning more than 50,000",
    "Find the top 5 highest-paid employees",
    "Calculate average salary department-wise",
    "Find duplicate records",
    "Show monthly sales trends",
    "Sort records by joining date",
    "Count records per category",
    "Show records where salary is NULL",
]

if not st.session_state.chat_history:
    st.markdown('<p class="section-header">✨ Suggested Questions</p>', unsafe_allow_html=True)
    cols = st.columns(4)
    for i, prompt in enumerate(SUGGESTED_PROMPTS):
        with cols[i % 4]:
            if st.button(prompt, key=f"sug_{i}", use_container_width=True):
                add_chat("user", prompt)
                handle_query_pipeline(prompt)
                st.rerun()

# ── Chat history display ──
chat_container = st.container()
with chat_container:
    for msg in st.session_state.chat_history:
        role    = msg["role"]
        content = msg["content"]
        sql     = msg.get("sql")
        df      = msg.get("df")

        if role == "user":
            st.markdown(f"""
            <div class="chat-user">
                <div class="chat-label">👤 You</div>
                <div class="chat-content">{content}</div>
            </div>
            """, unsafe_allow_html=True)

        else:
            st.markdown(f"""
            <div class="chat-ai">
                <div class="chat-label">🧠 QueryMind</div>
                <div class="chat-content">{content}</div>
            </div>
            """, unsafe_allow_html=True)

            # SQL preview
            if sql:
                with st.expander("🔍 View SQL", expanded=True):
                    st.markdown(f'<div class="sql-box">{sql}</div>', unsafe_allow_html=True)

            # Results table
            if df is not None and not df.empty:
                st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)

                # Metrics row
                m1, m2, m3 = st.columns(3)
                m1.metric("Rows returned", len(df))
                m2.metric("Columns", len(df.columns))
                m3.metric("Max rows cap", max_rows)

                # Interactive table
                st.dataframe(df.head(max_rows), use_container_width=True)

                # Export buttons
                col_csv, col_xlsx, col_stats = st.columns(3)
                with col_csv:
                    st.download_button(
                        "📥 Export CSV",
                        data=utils.df_to_csv_bytes(df),
                        file_name="querymind_results.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )
                with col_xlsx:
                    st.download_button(
                        "📥 Export Excel",
                        data=utils.df_to_excel_bytes(df),
                        file_name="querymind_results.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True,
                    )
                with col_stats:
                    if st.button("📊 Show Stats", key=f"stats_{id(df)}", use_container_width=True):
                        stats_df = utils.compute_stats(df)
                        st.dataframe(stats_df, use_container_width=True)

                # Auto chart suggestion
                chart_type = utils.suggest_chart(df)
                if chart_type:
                    st.markdown(f'<p class="section-header">📈 Auto-generated Chart ({chart_type.upper()})</p>', unsafe_allow_html=True)

                    num_cols = df.select_dtypes(include="number").columns.tolist()
                    cat_cols = df.select_dtypes(exclude="number").columns.tolist()

                    ctrl1, ctrl2 = st.columns(2)
                    with ctrl1:
                        x_options = df.columns.tolist()
                        x_default = cat_cols[0] if cat_cols else x_options[0]
                        x_col = st.selectbox("X axis", x_options,
                                             index=x_options.index(x_default),
                                             key=f"x_{id(df)}")
                    with ctrl2:
                        y_options = num_cols if num_cols else df.columns.tolist()
                        y_col = st.selectbox("Y axis", y_options, key=f"y_{id(df)}")

                    # Chart type picker
                    chart_choice = st.radio(
                        "Chart type",
                        ["bar", "line", "pie", "scatter"],
                        index=["bar","line","pie","scatter"].index(chart_type),
                        horizontal=True,
                        key=f"ct_{id(df)}",
                    )
                    fig = utils.render_chart(df.head(max_rows), chart_choice, x_col, y_col)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)

# ── Confirmation box for UPDATE / DELETE / INSERT ──
if st.session_state.pending_sql:
    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="confirm-box">
        <strong style="color:#fbbf24">⚠️ Confirmation Required</strong><br>
        This query will modify your data. Review it carefully before executing.
    </div>
    """, unsafe_allow_html=True)
    st.markdown(f'<div class="sql-box">{st.session_state.pending_sql}</div>', unsafe_allow_html=True)

    col_yes, col_no = st.columns(2)
    with col_yes:
        if st.button("✅ Confirm & Execute", type="primary", use_container_width=True):
            sql = st.session_state.pending_sql
            q   = st.session_state.pending_question
            st.session_state.pending_sql = None
            st.session_state.pending_question = None
            execute_and_display(sql, q)
            st.rerun()
    with col_no:
        if st.button("❌ Cancel", use_container_width=True):
            st.session_state.pending_sql = None
            st.session_state.pending_question = None
            add_chat("ai", "🚫 Query cancelled.")
            st.rerun()

# ── Main input box (pinned to bottom) ──
st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)
st.markdown('<p class="section-header">💬 Ask anything about your data</p>', unsafe_allow_html=True)

with st.form("query_form", clear_on_submit=True):
    user_input = st.text_area(
        "Your question",
        placeholder="e.g. Show all employees earning more than 50,000  •  Find duplicate records  •  Sort by joining date",
        height=90,
        label_visibility="collapsed",
    )
    col_submit, col_clear = st.columns([3, 1])
    with col_submit:
        submitted = st.form_submit_button(
            "🚀 Generate & Run",
            type="primary",
            use_container_width=True,
        )
    with col_clear:
        clear = st.form_submit_button("🔄 Clear", use_container_width=True)

if submitted and user_input.strip():
    add_chat("user", user_input.strip())
    handle_query_pipeline(user_input.strip())
    st.rerun()

if clear:
    st.session_state.chat_history = []
    st.session_state.memory.clear()
    st.session_state.pending_sql = None
    st.session_state.last_df = None
    st.rerun()

# ── Footer ──
st.markdown("""
<div style="text-align:center; margin-top:2rem; padding:1rem; color:rgba(100,116,139,0.6); font-size:0.75rem; border-top:1px solid rgba(255,255,255,0.05);">
    QueryMind AI • Powered by Gemini 2.0 Flash + ChromaDB + SQLite<br>
    <span style="color:rgba(129,140,248,0.4)">Your data never leaves your machine.</span>
</div>
""", unsafe_allow_html=True)
