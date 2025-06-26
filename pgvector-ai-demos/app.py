import streamlit as st
import os
import warnings
from pgvector_db_setup import setup_pgvector, create_database

warnings.filterwarnings("ignore", category=DeprecationWarning)

# Initialize database on first run
if 'db_initialized' not in st.session_state:
    try:
        create_database()
        setup_pgvector()
        st.session_state.db_initialized = True
    except Exception as e:
        st.error(f"Database initialization failed: {e}")
        st.stop()

# Page setup
st.set_page_config(
    layout="wide",
    page_icon="images/opensearch_mark_default.png",
    page_title="AI Search Demos with pgvector"
)

st.markdown("""
    <div id="home-page">
""", unsafe_allow_html=True)

st.markdown("""
    <style>
    /* Import Amazon Ember font */
    @import url('https://fonts.cdnfonts.com/css/amazon-ember');

    /* Global layout tweaks */
    html, body, .main {
        background: linear-gradient(135deg, #0d0d0d 0%, #1a1a1a 30%, #102132 100%);
        height: 100vh;
        overflow: hidden;
        color: white;
        font-family: 'Amazon Ember', sans-serif;
    }

    .block-container {
        padding-top: 2rem;
    }

    /* Hide Streamlit UI elements */
    #MainMenu, header, footer,
    button[title="View fullscreen"] {
        visibility: hidden;
    }

    /* Title styling */
    .title {
        font-size: 40px;
        color: #FF9900;
        font-family: 'Amazon Ember Display 500', sans-serif;
        margin-bottom: 10px;
    }

    .card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(8px);
        border-radius: 12px;
        padding: 24px 16px;
        width: 100%;
        margin: 0 auto;
        height: 350px;
        color: white;
        box-shadow: 0 0 10px rgba(0,0,0,0.3);
        transition: transform 0.3s ease-in-out, box-shadow 0.3s ease-in-out, border 0.3s ease-in-out;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        position: relative;
        z-index: 1;
        overflow: hidden;
    }

    .card:hover {
        transform: scale(1.06);
        transition: all 0.3s ease-in-out;
        border: 2px solid rgba(255, 255, 255, 0.2);
        box-shadow:
            0 0 20px rgba(228, 110, 8, 0.6),
            0 0 30px rgba(72, 61, 255, 0.4),
            0 0 40px rgba(0, 255, 255, 0.2);
    }

    .card-header {
        font-size: 100px;
        margin: 20px auto 12px auto;
        display: flex;
        justify-content: center;
        align-items: center;
        height: 60px;
        width: 60px;
        transition: all 0.3s ease-in-out;
    }

    .card-text {
        font-size: 45px;
        font-weight: bold;
        flex-grow: 1;
        display: flex;
        align-items: center;
        justify-content: center;
        text-align: center;
        padding: 0 8px;
    }

    .card:hover .card-text{
        opacity: 1;
        color: #e46e08;
        font-size: 48px;
        font-weight: bold;
        transform: scale(1.2);
        transition: all 0.3s ease-in-out;
    }

    .card-arrow {
        position: absolute;
        bottom: 12px;
        left: 50%;
        transform: translateX(-50%) scale(1);
        font-size: 22px;
        font-weight: bold;
        opacity: 0;
        transition: all 0.3s ease-in-out;
        color: #ffffff;
    }
    
    .card-description {
        font-size: 16px;
        color: #ccc;
        text-align: center;
        white-space: normal;
        line-height: 1.4;
        margin-top: auto;
        padding: 0 12px 8px;
    }
    </style>
""", unsafe_allow_html=True)

# Header with logo and title
col_logo, col_title = st.columns([20, 80])
with col_logo:
    if os.path.exists("images/OS_AI_1_cropped.png"):
        st.image("images/OS_AI_1_cropped.png", use_column_width=True)
    else:
        st.markdown("## 🔍 pgvector AI")

with col_title:
    st.markdown("<h1 class='title'>AI Search Demos with pgvector</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #ccc; font-size: 18px;'>Powered by PostgreSQL pgvector extension</p>", unsafe_allow_html=True)

spacer_col = st.columns(1)[0]
with spacer_col:
    st.markdown("<div style='height: 50px;'></div>", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    st.markdown("""
        <a href="/Semantic_Search" target="_self" style="text-decoration: none;">
            <div class="card">
                <div class="card-header">🔍</div>
                <div class="card-text" style="font-size: 31px; color: #e46e08;">AI Search</div>
                <div class="card-description" style="font-size: 16px; color: #ccc; margin-top: 6px;text-align: center;white-space: normal;">
                    Explore ML search types, Re-ranking and Query rewriting on retail data</div>
            </div>
        </a>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
        <a href="/Multimodal_Conversational_Search" target="_self" style="text-decoration: none;">
            <div class="card">
                <div class="card-header">💬</div>
                <div class="card-text" style="font-size: 31px; color: #e46e08;">Multimodal RAG</div>
                <div class="card-description" style="font-size: 16px; color: #ccc; margin-top: 6px;text-align: center;white-space: normal;">
                    Explore Multimodal RAG over complex PDFs (with tables, graphs etc)</div>
            </div>
        </a>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
        <a href="/AI_Shopping_Assistant" target="_self" style="text-decoration: none;">
            <div class="card">
                <div class="card-header">🤖</div>
                <div class="card-text" style="font-size: 31px; color: #e46e08;">Agentic RAG</div>
                <div class="card-description" style="font-size: 16px; color: #ccc; margin-top: 6px;text-align: center;white-space: normal;">
                    Explore how an AI agent in front of RAG enhances product search experience</div>
            </div>
        </a>
    """, unsafe_allow_html=True)

st.markdown("""
    </div>
""", unsafe_allow_html=True)

# Database connection status
with st.sidebar:
    st.markdown("### Database Status")
    if st.session_state.get('db_initialized', False):
        st.success("✅ pgvector connected")
    else:
        st.error("❌ Database not initialized")
    
    # Add connection info
    st.markdown("### Connection Info")
    st.code(f"""
Host: {os.getenv('PGHOST', 'localhost')}
Port: {os.getenv('PGPORT', '5432')}
Database: {os.getenv('PGDATABASE', 'opensearch_ai_demo')}
    """, language="text")
