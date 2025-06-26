import streamlit as st
import uuid
import os
import re
import sys
import json
import time
from PIL import Image
import base64

# Add parent directories to path
sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(1, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "RAG"))
sys.path.insert(1, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "utilities"))

import RAG.rag_DocumentLoader as rag_DocumentLoader
import RAG.rag_DocumentSearcher as rag_DocumentSearcher
import RAG.colpali as colpali
from utilities.pgvector_search import pg_search

st.set_page_config(
    layout="wide",
    page_icon="images/opensearch_mark_default.png",
    page_title="Multimodal RAG with pgvector"
)

parent_dirname = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
USER_ICON = "images/user.png"
AI_ICON = "images/opensearch-twitter-card.png"
REGENERATE_ICON = "images/regenerate.png"

# Initialize session state
if 'user_id' not in st.session_state:
    st.session_state['user_id'] = str(uuid.uuid4())

if 'session_id' not in st.session_state:
    st.session_state['session_id'] = ""

if "questions_" not in st.session_state:
    st.session_state.questions_ = []

if "answers_" not in st.session_state:
    st.session_state.answers_ = []

if "show_columns" not in st.session_state:
    st.session_state.show_columns = False

if "input_index" not in st.session_state:
    st.session_state.input_index = "hpijan2024hometrack"

if "input_query" not in st.session_state:
    # Default queries based on index
    default_queries = {
        "globalwarming": "What is the projected energy percentage from renewable sources in future?",
        "hpijan2024hometrack": "Which city has the highest average housing price in UK?",
        "covid19ie": "How many aged above 85 years died due to covid?"
    }
    st.session_state.input_query = default_queries.get(st.session_state.input_index, "")

if "input_is_rerank" not in st.session_state:
    st.session_state.input_is_rerank = True

if "input_is_colpali" not in st.session_state:
    st.session_state.input_is_colpali = False

if "input_rag_searchType" not in st.session_state:
    st.session_state.input_rag_searchType = ["Vector Search"]

if "inputs_" not in st.session_state:
    st.session_state.inputs_ = {}

# Styling
st.markdown("""
    <style>
    [data-testid=column]:nth-of-type(2) [data-testid=stVerticalBlock]{
        gap: 0rem;
    }
    [data-testid=column]:nth-of-type(1) [data-testid=stVerticalBlock]{
        gap: 0rem;
    }
    </style>
""", unsafe_allow_html=True)

def write_top_bar():
    """Render top bar with title and clear button"""
    col1, col2 = st.columns([77, 23])
    with col1:
        st.page_link("app.py", label=":orange[Home]", icon="🏠")
        st.header("Chat with your data", divider='rainbow')
    
    with col2:
        st.write("")
        st.write("")
        clear = st.button("Clear")
    
    st.write("")
    st.write("")
    return clear

def handle_input(state=None, dummy=None):
    """Handle user input and execute search"""
    if state == 'colpali_show_similarity_map':
        st.session_state.show_columns = True
    
    print(f"Question: {st.session_state.input_query}")
    print("-" * 20)
    
    # Collect inputs
    inputs = {}
    for key in st.session_state:
        if key.startswith('input_'):
            inputs[key.removeprefix('input_')] = st.session_state[key]
    
    st.session_state.inputs_ = inputs
    
    # Add question to history
    question_with_id = {
        'question': inputs["query"],
        'id': len(st.session_state.questions_)
    }
    st.session_state.questions_.append(question_with_id)
    
    # Execute search
    if st.session_state.input_is_colpali:
        # Use ColPali multi-vector search
        result = colpali.colpali_search_rerank(st.session_state.input_query)
    else:
        # Use standard RAG search with pgvector
        result = rag_DocumentSearcher.query_(
            None,  # awsauth not needed for pgvector
            inputs,
            st.session_state['session_id'],
            st.session_state.input_rag_searchType
        )
    
    # Add answer to history
    st.session_state.answers_.append({
        'answer': result['text'],
        'source': result['source'],
        'id': len(st.session_state.questions_),
        'image': result['image'],
        'table': result['table']
    })
    
    st.session_state.input_query = ""

def write_user_message(md):
    """Display user message"""
    col1, col2 = st.columns([3, 97])
    
    with col1:
        st.image(USER_ICON, use_column_width='always')
    with col2:
        st.markdown(
            f"<div style='color:#e28743;font-size:18px;padding:3px 7px;font-style:italic;'>"
            f"{md['question']}</div>",
            unsafe_allow_html=True
        )

def render_answer(question, answer, index, res_img):
    """Render AI response with sources"""
    col1, col2, col3 = st.columns([4, 74, 22])
    
    with col1:
        st.image(AI_ICON, use_column_width='always')
    
    with col2:
        # Display answer
        st.write(answer['answer'])
        
        # ColPali similarity map button
        if st.session_state.input_is_colpali:
            rdn_key = ''.join([str(i) for i in range(10)])
            st.button("Show similarity map", key=rdn_key, 
                     on_click=handle_input, args=('colpali_show_similarity_map', True))
    
    # Sources expander
    colu1, colu2, colu3 = st.columns([4, 82, 20])
    with colu2:
        with st.expander("Relevant Sources:"):
            # Display images
            if len(res_img) > 0:
                if st.session_state.input_is_colpali:
                    # ColPali image display
                    if len(res_img) > 1:
                        cols_per_row = 3
                        row = st.columns(cols_per_row)
                        for j, item in enumerate(res_img[:cols_per_row]):
                            with row[j]:
                                st.markdown(
                                    "<div style='max-height:500px;overflow:auto;border:1px solid #444;padding:4px;'>",
                                    unsafe_allow_html=True
                                )
                                st.image(item['file'])
                                st.markdown("</div>", unsafe_allow_html=True)
                    else:
                        col3_, col4_, col5_ = st.columns([33, 33, 33])
                        with col3_:
                            st.markdown(
                                "<div style='max-height:500px;overflow:auto;border:1px solid #444;padding:4px;'>",
                                unsafe_allow_html=True
                            )
                            st.image(res_img[0]['file'])
                            st.markdown("</div>", unsafe_allow_html=True)
                else:
                    # Regular image display
                    for i, img_data in enumerate(res_img[:2]):
                        if img_data['file'].lower() != 'none':
                            img_path = os.path.join(
                                parent_dirname, "figures",
                                st.session_state.input_index,
                                f"{img_data['file'].split('.')[0]}.jpg"
                            )
                            if os.path.exists(img_path):
                                st.image(img_path)
            
            # Display tables
            if len(answer["table"]) > 0:
                import pandas as pd
                for table in answer["table"]:
                    try:
                        df = pd.read_csv(table['name'], skipinitialspace=True, 
                                       on_bad_lines='skip', delimiter='`')
                        df.fillna(method='pad', inplace=True)
                        st.table(df)
                    except Exception as e:
                        st.error(f"Error loading table: {e}")
            
            # Raw sources
            st.write("**Raw sources:**")
            st.write(answer["source"])
    
    # Regenerate button
    with col3:
        if index == len(st.session_state.questions_):
            rdn_key = ''.join([str(i) for i in range(10)])
            
            def on_button_click():
                # Check if settings changed
                current_value = (
                    ''.join(st.session_state.input_rag_searchType) +
                    str(st.session_state.input_is_rerank) +
                    st.session_state.input_index
                )
                old_value = (
                    ''.join(st.session_state.inputs_.get("rag_searchType", [])) +
                    str(st.session_state.inputs_.get("is_rerank", True)) +
                    st.session_state.inputs_.get("index", "")
                )
                
                if current_value != old_value or True:
                    st.session_state.input_query = st.session_state.questions_[-1]["question"]
                    st.session_state.answers_.pop()
                    st.session_state.questions_.pop()
                    handle_input("regenerate_", None)
            
            st.button("🔄", key=rdn_key, on_click=on_button_click)

def write_chat_message(md, q, index):
    """Write complete chat message with answer"""
    res_img = md['image']
    chat = st.container()
    with chat:
        render_answer(q, md, index, res_img)

def render_all():
    """Render all messages"""
    for i, (q, a) in enumerate(zip(st.session_state.questions_, st.session_state.answers_)):
        write_user_message(q)
        write_chat_message(a, q, i + 1)

# Main UI
clear = write_top_bar()

if clear:
    st.session_state.questions_ = []
    st.session_state.answers_ = []
    st.session_state.input_query = ""

# Render chat history
placeholder = st.empty()
with placeholder.container():
    render_all()

st.markdown("")

# Input section
col_2, col_3 = st.columns([75, 20])

with col_2:
    st.text_input("Ask here", label_visibility="collapsed", key="input_query")
with col_3:
    st.button("Go", on_click=handle_input, args=(None, None),
             help='Enter questions and click "GO"')

# Sidebar configuration
with st.sidebar:
    st.page_link("app.py", label=":orange[Home]", icon="🏠")
    
    # Sample data selection
    st.subheader(":blue[Sample Data]")
    coln_1, coln_2 = st.columns([70, 30])
    
    with coln_1:
        index_select = st.radio(
            "Choose one index",
            ["UK Housing", "Global Warming stats", "Covid19 impacts on Ireland"],
            key="input_rad_index"
        )
    
    with coln_2:
        st.markdown("<p style='font-size:15px'>Preview file</p>", unsafe_allow_html=True)
        st.write("[:eyes:](https://github.com/aws-samples/AI-search-with-amazon-opensearch-service/blob/main/rag/sample_pdfs/HPI-Jan-2024-Hometrack.pdf)")
        st.write("[:eyes:](https://github.com/aws-samples/AI-search-with-amazon-opensearch-service/blob/main/rag/sample_pdfs/global_warming.pdf)")
        st.write("[:eyes:](https://github.com/aws-samples/AI-search-with-amazon-opensearch-service/blob/main/rag/sample_pdfs/covid19_ie.pdf)")
    
    # Update index based on selection
    index_mapping = {
        "Global Warming stats": "globalwarming",
        "Covid19 impacts on Ireland": "covid19ie",
        "UK Housing": "hpijan2024hometrack"
    }
    st.session_state.input_index = index_mapping.get(index_select, "hpijan2024hometrack")
    
    # Sample questions
    with st.expander("Sample questions:"):
        st.markdown(
            "<span style='color:#FF9900;'>UK Housing</span> - "
            "Which city has the highest average housing price in UK?",
            unsafe_allow_html=True
        )
        st.markdown(
            "<span style='color:#FF9900;'>Global Warming stats</span> - "
            "What is the projected energy percentage from renewable sources in future?",
            unsafe_allow_html=True
        )
        st.markdown(
            "<span style='color:#FF9900;'>Covid19 impacts</span> - "
            "How many aged above 85 years died due to covid?",
            unsafe_allow_html=True
        )
    
    # Document upload section (optional)
    st.subheader(":blue[Your multi-modal documents]")
    pdf_doc_ = st.file_uploader(
        "Upload your PDFs here and click on 'Process'",
        accept_multiple_files=False,
        type=['pdf']
    )
    
    if st.button("Process"):
        if pdf_doc_:
            with st.spinner("Processing..."):
                # Save PDF
                pdfs_dir = os.path.join(parent_dirname, "pdfs")
                os.makedirs(pdfs_dir, exist_ok=True)
                
                pdf_name = pdf_doc_.name.replace(" ", "_")
                pdf_path = os.path.join(pdfs_dir, pdf_name)
                
                with open(pdf_path, "wb") as f:
                    f.write(pdf_doc_.getbuffer())
                
                # Process document
                request = {"key": pdf_name}
                rag_DocumentLoader.load_docs(request)
                
                # Update index name
                st.session_state.input_index = re.sub(
                    '[^A-Za-z0-9]+', '',
                    os.path.splitext(pdf_name)[0].lower()
                )
                
                st.success('Document processed! You can start searching.')
    
    # Retriever configuration
    st.subheader(":blue[Retriever]")
    search_type = st.multiselect(
        'Select the Retriever(s)',
        ['Keyword Search', 'Vector Search', 'Sparse Search'],
        ['Vector Search'],
        key='input_rag_searchType',
        help="Select search types. Multiple selections enable hybrid search."
    )
    
    re_rank = st.checkbox('Re-rank results', key='input_re_rank', value=True,
                         help="Re-rank results using a cross-encoder model")
    st.session_state.input_is_rerank = re_rank
    
    # Multi-vector retrieval
    st.subheader(":blue[Multi-vector retrieval]")
    
    colpali_search = st.checkbox(
        'Try ColPali multi-vector retrieval',
        key='input_colpali', value=False,
        help="Use ColPali for multi-vector document retrieval with MaxSim"
    )
    st.session_state.input_is_colpali = colpali_search
    
    if colpali_search:
        with st.expander("Sample questions for ColPali:"):
            st.write(
                "1. Proportion of female new hires 2021-2023?\n"
                "2. Hula hoop kid\n"
                "3. First-half 2021 return on unlisted real estate investments?\n"
                "4. Fund return percentage in 2017?\n"
                "5. Annualized gross return of the fund from 1997 to 2008?"
            )
