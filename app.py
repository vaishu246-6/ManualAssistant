"""

Run:
    streamlit run app.py

Requires:
    ollama serve        (in a separate terminal)
    ollama pull mistral
"""

import os
import json
import uuid
import sqlite3
import logging
from pathlib import Path
from datetime import datetime

import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Manual Assistant",
    page_icon="📘",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
html, body, [data-testid="stAppViewContainer"] {
    background: #F7FAFF !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
}
[data-testid="stSidebar"] {
    background: #EEF5FF !important;
    border-right: 1px solid #D9E8FF;
    min-width: 280px !important;
    max-width: 280px !important;
}
[data-testid="stSidebar"] > div { padding: 1rem 0.75rem; }
#MainMenu {
    display: none;
}

footer {
    display: none;
}

/* Don't hide the Streamlit header */
header {
    background: transparent !important;
}
[data-testid="stDecoration"] { display: none; }
.stDeployButton { display: none; }
.main .block-container { padding: 2rem 2.5rem !important; max-width: 960px; }
/* Buttons */
.stButton > button {
    width: 100%;
    background: #FFFFFF;
    color: #2F4B6E;
    border: 1.5px solid #D6E8FF;
    border-radius: 10px;
    font-size: 14px;
    font-weight: 500;
    padding: 0.5rem 1rem;
    cursor: pointer;
    transition: all 0.2s ease;
    text-align: left;
}

.stButton > button:hover {
    background: #EAF4FF;
    border-color: #8BB8FF;
    color: #1E3A5F;
}

/* Text Input */
.stTextInput > div > input {
    background: #FFFFFF !important;
    border: 1.5px solid #D6E8FF !important;
    border-radius: 12px !important;
    font-size: 14px !important;
    padding: 0.75rem 1rem !important;
    color: #243B53 !important;
}

.stTextInput > div > input:focus {
    border-color: #7DAFFF !important;
    box-shadow: 0 0 0 3px rgba(125, 175, 255, 0.18) !important;
}
[data-testid="stFileUploader"] {
    border: 1.5px dashed #D1D5DB !important;
    border-radius: 10px !important; background: #FFFFFF !important;
}
.welcome-card {
    background: #FFFFFF; border: 1px solid #E5E7EB;
    border-radius: 14px; padding: 2rem 2.25rem; margin: 1rem 0;
}
.msg-user { display:flex; justify-content:flex-end; margin-bottom:12px; }
.msg-user-bubble {
    background: #6366F1; color: #FFFFFF;
    border-radius: 18px 18px 4px 18px;
    padding: 0.65rem 1rem; max-width: 78%;
    font-size: 14px; line-height: 1.55;
}
.msg-assistant { display:flex; gap:10px; margin-bottom:12px; }
.msg-avatar {
    width:32px; height:32px; border-radius:50%;
    background:#EEF2FF; display:flex; align-items:center;
    justify-content:center; font-size:15px; flex-shrink:0; margin-top:2px;
}
.msg-assistant-bubble {
    background: #FFFFFF; border: 1px solid #E5E7EB;
    border-radius: 4px 18px 18px 18px;
    padding: 0.75rem 1rem; max-width: 82%;
    font-size: 14px; line-height: 1.6; color: #1F2937;
}
.source-pill {
    background: #F3F4F6; border: 1px solid #E5E7EB;
    border-radius: 20px; padding: 2px 8px;
    font-size: 11px; color: #6B7280; display:inline-block; margin:2px;
}
.status-ok {
    background:#F0FDF4; border:1px solid #BBF7D0;
    border-radius:10px; padding:10px 16px;
    font-size:13px; color:#166534; margin:8px 0;
}
.status-warn {
    background:#FEF3C7; border:1px solid #FDE68A;
    border-radius:10px; padding:10px 16px;
    font-size:13px; color:#92400E; margin:8px 0;
}
hr { border:none; border-top:1px solid #E5E7EB; margin:1rem 0; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
OUTPUT_DIR  = Path("output")
INDEX_FILE  = OUTPUT_DIR / "faiss.index"
CHUNKS_FILE = OUTPUT_DIR / "chunks.json"
DB_FILE     = OUTPUT_DIR / "equipai.db"
DATA_DIR    = Path("data")

CAPABILITIES = [
    ("📋", "Product Specifications"),
    ("🔧", "Installation Procedures"),
    ("🛠", "Maintenance Instructions"),
    ("⚠️", "Troubleshooting"),
    ("📦", "Parts & Components"),
    ("⚡", "Electrical Connections"),
    ("📖", "Manual Summaries"),
]

# ── Session state ─────────────────────────────────────────────────────────────
def init_state():
    defaults = {
        "session_id":     str(uuid.uuid4()),
        "messages":       [],
        "manual_ready":   False,
        "manual_name":    "",
        "active_session": None,
        "active_tab":     "chat",
        "pending_query":  None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ── DB helpers ────────────────────────────────────────────────────────────────
def get_db():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_FILE))
    conn.row_factory = sqlite3.Row
    conn.execute("""CREATE TABLE IF NOT EXISTS chat_sessions
                    (id TEXT PRIMARY KEY, title TEXT, created_at TEXT)""")
    conn.execute("""CREATE TABLE IF NOT EXISTS chat_messages
                    (id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT,
                     role TEXT, content TEXT, sources TEXT, created_at TEXT)""")
    conn.commit()
    return conn

def save_message(sid, role, content, sources=None):
    conn = get_db()
    conn.execute(
        "INSERT INTO chat_messages (session_id,role,content,sources,created_at) VALUES(?,?,?,?,?)",
        (sid, role, content, json.dumps(sources or []), datetime.now().isoformat())
    )
    conn.commit(); conn.close()

def save_session(sid, title):
    conn = get_db()
    conn.execute("INSERT OR REPLACE INTO chat_sessions VALUES(?,?,?)",
                 (sid, title, datetime.now().isoformat()))
    conn.commit(); conn.close()

def load_sessions():
    try:
        conn = get_db()
        rows = conn.execute(
            "SELECT id,title,created_at FROM chat_sessions ORDER BY created_at DESC LIMIT 30"
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception:
        return []

def load_session_messages(sid):
    try:
        conn = get_db()
        rows = conn.execute(
            "SELECT role,content,sources FROM chat_messages WHERE session_id=? ORDER BY id",
            (sid,)
        ).fetchall()
        conn.close()
        return [{"role":r["role"],"content":r["content"],
                 "sources":json.loads(r["sources"] or "[]")} for r in rows]
    except Exception:
        return []

def delete_session(sid):
    conn = get_db()
    conn.execute("DELETE FROM chat_sessions WHERE id=?", (sid,))
    conn.execute("DELETE FROM chat_messages WHERE session_id=?", (sid,))
    conn.commit(); conn.close()

# ── RAG (Phase 3) ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_retriever():
    """Load retriever once and cache."""
    try:
        from retriever import retrieve, build_prompt
        return retrieve, build_prompt
    except Exception as e:
        return None, None

def index_exists():
    return INDEX_FILE.exists() and CHUNKS_FILE.exists()

# ── PDF ingestion ─────────────────────────────────────────────────────────────
def ingest_pdf(uploaded_file) -> bool:
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        from extractor import extract_pdf, build_chunks
        from embedder  import embed_chunks, build_faiss_index, save_chunks_metadata
        from database  import init_db, insert_document, insert_chunks_bulk, update_document_status

        DATA_DIR.mkdir(exist_ok=True)
        pdf_path = DATA_DIR / uploaded_file.name
        pdf_path.write_bytes(uploaded_file.read())

        pages  = extract_pdf(pdf_path)
        chunks = build_chunks(pages)
        init_db()
        doc_id = insert_document(uploaded_file.name, str(pdf_path))
        rows = [{
            "doc_id": doc_id, "chunk_index": c.chunk_index, "page": c.page,
            "section": c.section, "content": c.content,
            "image_refs": json.dumps(c.image_refs),
            "is_table": int(c.is_table), "word_count": c.word_count,
        } for c in chunks]
        insert_chunks_bulk(rows)
        save_chunks_metadata(chunks)
        update_document_status(doc_id, "done", len(pages), len(chunks))
        embeddings = embed_chunks(chunks)
        build_faiss_index(embeddings)
        get_retriever.clear()
        return True
    except Exception as e:
        st.error(f"Processing error: {e}")
        return False

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="font-size:17px;font-weight:700;color:#111827;
                display:flex;align-items:center;gap:8px;
                padding:0.25rem 0.5rem 1rem;
                border-bottom:1px solid #E5E7EB;margin-bottom:0.75rem">
        📘 Manual Assistant
    </div>""", unsafe_allow_html=True)

    if st.button("＋  New Chat"):
        st.session_state.session_id  = str(uuid.uuid4())
        st.session_state.messages    = []
        st.session_state.active_session = None
        st.rerun()

    st.markdown("""<div style="font-size:12px;font-weight:600;color:#6B7280;
                               text-transform:uppercase;letter-spacing:.05em;
                               padding:.75rem .5rem .5rem">
                    Chat History</div>""", unsafe_allow_html=True)

    sessions = load_sessions()
    if not sessions:
        st.markdown('<p style="font-size:12px;color:#9CA3AF;padding:.5rem">No history yet.</p>',
                    unsafe_allow_html=True)
    for sess in sessions:
        c1, c2 = st.columns([5, 1])
        with c1:
            label = sess["title"][:30] + ("…" if len(sess["title"]) > 30 else "")
            if st.button(label, key=f"s_{sess['id']}", use_container_width=True):
                st.session_state.session_id     = sess["id"]
                st.session_state.messages       = load_session_messages(sess["id"])
                st.session_state.active_session = sess["id"]
                st.rerun()
        with c2:
            if st.button("🗑", key=f"d_{sess['id']}"):
                delete_session(sess["id"])
                if st.session_state.active_session == sess["id"]:
                    st.session_state.messages = []
                    st.session_state.session_id = str(uuid.uuid4())
                    st.session_state.active_session = None
                st.rerun()

    # Ollama status in sidebar footer
    st.markdown("<hr>", unsafe_allow_html=True)
    try:
        from llm import is_ollama_running, get_available_models
        if is_ollama_running():
            models = get_available_models()
            model_list = ", ".join(m.split(":")[0] for m in models[:3]) or "none"
            st.markdown(f"""<div class="status-ok">🟢 Ollama running<br>
                <span style="font-size:11px">{model_list}</span></div>""",
                        unsafe_allow_html=True)
        else:
            st.markdown("""<div class="status-warn">🔴 Ollama offline<br>
                <span style="font-size:11px">Run: ollama serve</span></div>""",
                        unsafe_allow_html=True)
    except Exception:
        pass

# ══════════════════════════════════════════════════════════════════════════════
# MAIN AREA
# ══════════════════════════════════════════════════════════════════════════════

# Header
st.markdown("""
<div style="display:flex;align-items:center;gap:12px;margin-bottom:1.25rem">
    <span style="font-size:36px">🤖</span>
    <span style="font-size:32px;font-weight:800;color:#111827;letter-spacing:-.5px">
        AI-Powered User Manual Assistant
    </span>
</div>""", unsafe_allow_html=True)

# Upload
st.markdown('<div style="font-size:13px;font-weight:600;color:#374151;margin-bottom:.4rem">'
            'Upload Manual</div>', unsafe_allow_html=True)

uploaded = st.file_uploader("Upload Manual", type=["pdf","docx"],
                             label_visibility="collapsed")

if uploaded:
    if st.session_state.get("last_uploaded") != uploaded.name:
        st.session_state.last_uploaded = uploaded.name
        with st.status("Processing manual…", expanded=True) as status:
            st.write("Extracting text, tables and images…")
            ok = ingest_pdf(uploaded)
            if ok:
                st.session_state.manual_ready = True
                st.session_state.manual_name  = uploaded.name
                status.update(label=f"✓ {uploaded.name} ready", state="complete")
            else:
                status.update(label="Processing failed", state="error")

elif index_exists():
    st.session_state.manual_ready = True
    if not st.session_state.manual_name:
        st.session_state.manual_name = "Manual"
    st.markdown(f'<div class="status-ok">✓ &nbsp;<strong>'
                f'{st.session_state.manual_name}</strong> loaded and ready</div>',
                unsafe_allow_html=True)

# ── Chat ──────────────────────────────────────────────────────────────────────
tab_chat = st.container()

# ══════════════════════════════════════════════════════════════════════════════
# CHAT
# ══════════════════════════════════════════════════════════════════════════════
with tab_chat:

    # Welcome card
    if not st.session_state.messages:
        caps = "".join(
            f'<li style="font-size:14px;color:#374151;display:flex;'
            f'align-items:center;gap:8px;padding:5px 0">'
            f'<span>{icon}</span> {label}</li>'
            for icon, label in CAPABILITIES
        )
        st.markdown(f"""
        <div class="welcome-card">
            <div style="font-size:22px;font-weight:700;color:#111827;margin-bottom:.4rem">
                Welcome to Manual Assistant</div>
            <div style="font-size:14px;color:#6B7280;margin-bottom:1.25rem">
                Upload a technical manual and start asking questions.</div>
            <div style="font-size:16px;font-weight:700;color:#111827;margin-bottom:.75rem">
                What can I help with?</div>
            <ul style="list-style:none;padding:0;margin:0;
                       display:grid;grid-template-columns:1fr 1fr;gap:4px">
                {caps}
            </ul>
        </div>""", unsafe_allow_html=True)

    # Chat messages
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f"""
            <div class="msg-user">
                <div class="msg-user-bubble">{msg['content']}</div>
            </div>""", unsafe_allow_html=True)
        else:
            sources = msg.get("sources", [])
            pills   = "".join(
                f'<span class="source-pill">Page {s["page"]}</span>'
                for s in sources[:5]
            )
            src_row = f'<div style="margin-top:6px">{pills}</div>' if pills else ""
            content = msg["content"].replace("\n", "<br>")
            st.markdown(f"""
            <div class="msg-assistant">
                <div class="msg-avatar">📘</div>
                <div>
                    <div class="msg-assistant-bubble">{content}</div>
                    {src_row}
                </div>
            </div>""", unsafe_allow_html=True)

    # Input
    st.markdown("<hr>", unsafe_allow_html=True)
    with st.form("chat_form", clear_on_submit=True):
        col1, col2 = st.columns([9, 1])
        with col1:
            user_input = st.text_input(
                "Question",
                placeholder = ("Ask a question from the manual…"
                               if st.session_state.manual_ready
                               else "Upload a manual above to start…"),
                label_visibility = "collapsed",
                disabled = not st.session_state.manual_ready,
            )
        with col2:
            submitted = st.form_submit_button(
                "➤", use_container_width=True,
                disabled=not st.session_state.manual_ready
            )

    if submitted and user_input.strip():
        query = user_input.strip()
        st.session_state.messages.append({"role":"user","content":query,"sources":[]})
        st.session_state.pending_query = query   # flag for next rerun to process
        st.rerun()

    # ── Process pending query (runs AFTER rerun, so user msg is already shown) ──
    if st.session_state.get("pending_query"):
        query = st.session_state.pending_query
        st.session_state.pending_query = None

        if len(st.session_state.messages) == 1:
            save_session(st.session_state.session_id, query[:60])
            st.session_state.active_session = st.session_state.session_id
        save_message(st.session_state.session_id, "user", query)

        with st.spinner("Searching manual…"):
            retrieve_fn, build_prompt_fn = get_retriever()
            if retrieve_fn:
                history = st.session_state.messages[:-1]
                sources = retrieve_fn(query, top_k=5)
                prompt  = build_prompt_fn(query, sources, history=history)
            else:
                sources = []
                prompt  = f"Answer this question about the plasma cutter: {query}"

            from llm import generate
            answer = generate(prompt)

        st.session_state.messages.append({
            "role":"assistant","content":answer,"sources":sources
        })
        save_message(st.session_state.session_id, "assistant", answer, sources)
        st.rerun()