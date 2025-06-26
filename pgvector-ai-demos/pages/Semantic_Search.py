import streamlit as st
import math
import uuid
import os
import sys
import json
import warnings
from datetime import datetime
from PIL import Image
import base64
import re
import numpy as np
from datetime import datetime

# Add parent directories to path
sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(1, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "semantic_search"))
sys.path.insert(1, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "utilities"))

import utilities.mvectors as cb
import utilities.ubi_lambda as ubi
from utilities.pgvector_search import pg_search
import semantic_search.query_rewrite as query_rewrite
import semantic_search.amazon_rekognition as amazon_rekognition
import semantic_search.llm_eval as llm_eval
import semantic_search.all_search_execute as all_search_execute

warnings.filterwarnings("ignore", category=DeprecationWarning)

st.set_page_config(
    page_icon="images/opensearch_mark_default.png",
    page_title="AI Search with pgvector"
)

parent_dirname = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Initialize session state variables
if 'user_id' not in st.session_state:
    st.session_state['user_id'] = str(uuid.uuid4())

if 'session_id' not in st.session_state:
    st.session_state['session_id'] = f"sess_{uuid.uuid4()}"

if 'query_id' not in st.session_state:
    st.session_state["query_id"] = ""

if 'input_reranker' not in st.session_state:
    st.session_state['input_reranker'] = "None"

if "questions" not in st.session_state:
    st.session_state.questions = []

if "answers" not in st.session_state:
    st.session_state.answers = []

if "input_text" not in st.session_state:
    st.session_state.input_text = "black jacket for men"

if "input_searchType" not in st.session_state:
    st.session_state.input_searchType = ["Keyword Search"]

if "input_hybridType" not in st.session_state:
    st.session_state.input_hybridType = "pgvector Hybrid Query"

if "input_K" not in st.session_state:
    st.session_state.input_K = 5

if "input_category" not in st.session_state:
    st.session_state.input_category = None

if "input_gender" not in st.session_state:
    st.session_state.input_gender = None

if "input_price" not in st.session_state:
    st.session_state.input_price = (0, 0)

if "input_manual_filter" not in st.session_state:
    st.session_state.input_manual_filter = "False"

if "input_imageUpload" not in st.session_state:
    st.session_state.input_imageUpload = 'no'

if "input_image" not in st.session_state:
    st.session_state.input_image = ''

if "input_is_rewrite_query" not in st.session_state:
    st.session_state.input_is_rewrite_query = "disabled"

if "input_rewritten_query" not in st.session_state:
    st.session_state.input_rewritten_query = ""

if "input_mvector_rerank" not in st.session_state:
    st.session_state.input_mvector_rerank = False

if "input_multilingual" not in st.session_state:
    st.session_state.input_multilingual = False

if "input_sparse_filter" not in st.session_state:
    st.session_state.input_sparse_filter = 0.5

if "input_evaluate" not in st.session_state:
    st.session_state.input_evaluate = "disabled"

if "input_ndcg" not in st.session_state:
    st.session_state.input_ndcg = 0.0

if "ndcg_increase" not in st.session_state:
    st.session_state.ndcg_increase = " ~ "

if "img_doc" not in st.session_state:
    st.session_state.img_doc = None

if "input_rekog_label" not in st.session_state:
    st.session_state.input_rekog_label = ""

if "inputs_" not in st.session_state:
    st.session_state.inputs_ = {}

# Weight inputs for hybrid search
for search_type in ["Keyword", "Vector", "Multimodal", "NeuralSparse"]:
    key = f"input_{search_type}-weight"
    if key not in st.session_state:
        st.session_state[key] = 100 if search_type == "Keyword" else 0

# UI Constants
USER_ICON = "images/user.png"
AI_ICON = "images/opensearch-twitter-card.png"
IMAGE_ICON = "images/Image_Icon.png"
TEXT_ICON = "images/text.png"

# Available search types
search_types = [
    'Keyword Search',
    'Vector Search', 
    'Multimodal Search',
    'NeuralSparse Search'
]

# Styling
st.markdown("""
    <style>
    .block-container {
        padding-top: 2.75rem;
        padding-bottom: 0rem;
        padding-left: 5rem;
        padding-right: 5rem;
    }
    </style>
""", unsafe_allow_html=True)

def handle_input():
    """Handle search input and execute search"""
    # Reset NDCG if query changed
    if "text" in st.session_state.inputs_:
        if st.session_state.inputs_["text"] != st.session_state.input_text:
            st.session_state.input_ndcg = 0.0
    
    # Handle image upload for multimodal search
    if st.session_state.img_doc is not None or st.session_state.get('input_rad_1'):
        st.session_state.input_imageUpload = 'yes'
        
        # Process generated image selection
        if st.session_state.get('input_rad_1'):
            num_str = str(int(st.session_state.input_rad_1.strip()) - 1)
            image_path = os.path.join(parent_dirname, "gen_images", 
                                    f"{st.session_state.image_prompt}_gen_{num_str}-resized_display.jpg")
            with open(image_path, "rb") as image_file:
                st.session_state.input_image = base64.b64encode(image_file.read()).decode("utf8")
        
        # Process uploaded image
        elif st.session_state.img_doc:
            # Save and process uploaded image
            uploaded_images = os.path.join(parent_dirname, "uploaded_images")
            os.makedirs(uploaded_images, exist_ok=True)
            
            image_path = os.path.join(uploaded_images, st.session_state.img_doc.name)
            with open(image_path, "wb") as f:
                f.write(st.session_state.img_doc.getbuffer())
            
            # Resize and encode image
            with Image.open(image_path) as img:
                img.thumbnail((2048, 2048))
                resized_path = image_path.rsplit(".", 1)[0] + "-resized.jpg"
                img.save(resized_path)
            
            with open(resized_path, "rb") as image_file:
                st.session_state.input_image = base64.b64encode(image_file.read()).decode("utf8")
    else:
        st.session_state.input_imageUpload = 'no'
        st.session_state.input_image = ''
    
    # Extract image metadata if using keyword search with image
    if (st.session_state.input_imageUpload == 'yes' and 
        'Keyword Search' in st.session_state.input_searchType):
        # Use Amazon Rekognition for image analysis
        if st.session_state.img_doc:
            image_bytes = st.session_state.img_doc.getvalue()
        else:
            # Generated image
            num_str = str(int(st.session_state.input_rad_1.strip()) - 1)
            image_path = os.path.join(parent_dirname, "gen_images",
                                    f"{st.session_state.image_prompt}_gen_{num_str}.jpg")
            with open(image_path, "rb") as f:
                image_bytes = f.read()
        
        st.session_state.input_rekog_label = amazon_rekognition.extract_image_metadata(image_bytes)
        if st.session_state.input_text == "":
            st.session_state.input_text = st.session_state.input_rekog_label
    
    # Collect all inputs
    inputs = {}
    for key in st.session_state:
        if key.startswith('input_'):
            inputs[key.removeprefix('input_')] = st.session_state[key]
    
    # Handle weights for hybrid search
    weights = {}
    total_weight = 0
    for search_type in ['Keyword', 'Vector', 'Multimodal', 'NeuralSparse']:
        key = f"{search_type}-weight"
        if key in inputs:
            weights[key] = inputs[key]
            if f"{search_type} Search" in st.session_state.input_searchType:
                total_weight += inputs[key]
    
    # Normalize weights if needed
    if total_weight != 100 and len(st.session_state.input_searchType) > 1:
        for key in weights:
            weights[key] = (weights[key] / total_weight) * 100 if total_weight > 0 else 100 / len(st.session_state.input_searchType)
    
    inputs['weightage'] = weights
    st.session_state.inputs_ = inputs
    
    # Clear previous results
    st.session_state.questions = [{
        'question': inputs["text"],
        'id': 0
    }]
    
    # Apply query rewriting if enabled
    if st.session_state.input_is_rewrite_query == 'enabled':
        query_rewrite.get_new_query_res(st.session_state.input_text)
    else:
        st.session_state.input_rewritten_query = ""
    
    # Execute search
    results = all_search_execute.handler(inputs, st.session_state['session_id'])
    
    st.session_state.answers = [{
        'answer': results,
        'search_type': inputs['searchType'],
        'id': 0
    }]
    
    # Evaluate results if enabled
    if st.session_state.input_evaluate == "enabled":
        llm_eval.eval(st.session_state.questions, st.session_state.answers)

def write_top_bar():
    """Render the top search bar"""
    col1, col2, col3, col4 = st.columns([2.5, 35, 8, 7])
    
    with col1:
        st.image(TEXT_ICON, use_column_width='always')
    with col2:
        st.text_input("Ask here", label_visibility="collapsed", 
                     key="input_text", placeholder="Type your query")
    with col3:
        st.button("Search", on_click=handle_input, key="play")
    with col4:
        clear = st.button("Clear")
    
    # Image search section
    col5, col6 = st.columns([4.5, 95])
    with col5:
        st.image(IMAGE_ICON, use_column_width='always')
    with col6:
        with st.expander(':green[Search by using an image]'):
            tab2, tab1 = st.tabs(["Upload Image", "Generate Image by AI"])
            
            with tab1:
                # Image generation UI
                c1, c2 = st.columns([80, 20])
                with c1:
                    st.text_area("Text2Image:", placeholder="Enter the text prompt to generate images",
                               height=68, key="image_prompt")
                with c2:
                    st.markdown("<div style='height:43px'></div>", unsafe_allow_html=True)
                    st.button("Generate", disabled=False, key="generate")
                
                st.radio("Choose one image", ["Image 1", "Image 2", "Image 3"],
                        index=None, horizontal=True, key='image_select')
            
            with tab2:
                st.session_state.img_doc = st.file_uploader(
                    "Upload image", accept_multiple_files=False, type=['png', 'jpg'])
    
    return clear

def write_user_message(md, ans):
    """Display user query message"""
    if len(ans["answer"]) > 0:
        col1, col2, col3 = st.columns([3, 40, 20])
        
        with col1:
            st.image(USER_ICON, use_column_width='always')
        with col2:
            st.markdown(f"""<div style='fontSize:15px;'>Input Text: </div>
                          <div style='fontSize:25px;font-style:italic;color:#e28743'>
                          {md['question']}</div>""", unsafe_allow_html=True)
            
            # Show query expansion for sparse search
            if 'query_sparse' in ans["answer"][0]:
                with st.expander("Expanded Query:"):
                    query_sparse = dict(sorted(ans["answer"][0]['query_sparse'].items(), 
                                             key=lambda x: x[1], reverse=True))
                    filtered_sparse = {k: round(v, 2) for k, v in query_sparse.items()}
                    st.write(filtered_sparse)
            
            # Show rewritten query
            if (st.session_state.input_is_rewrite_query == "enabled" and 
                st.session_state.input_rewritten_query):
                with st.expander("Re-written Query:"):
                    st.json(st.session_state.input_rewritten_query, expanded=True)
        
        with col3:
            st.markdown("<div style='fontSize:15px;'>Input Image: </div>", 
                       unsafe_allow_html=True)
            
            if st.session_state.input_imageUpload == 'yes':
                # Display uploaded or generated image
                if st.session_state.get('input_rad_1'):
                    num_str = str(int(st.session_state.input_rad_1.strip()) - 1)
                    img_path = f"{parent_dirname}/gen_images/{st.session_state.image_prompt}_gen_{num_str}-resized_display.jpg"
                elif st.session_state.img_doc:
                    img_path = f"{parent_dirname}/uploaded_images/{st.session_state.img_doc.name}"
                
                st.image(img_path)
                
                if st.session_state.input_rekog_label:
                    with st.expander("Enriched Query Metadata:"):
                        st.json(st.session_state.get('input_rekog_directoutput', {}))
            else:
                st.markdown("<div style='fontSize:15px;'>None</div>", unsafe_allow_html=True)
        
        st.markdown('---')

def render_results(answer, index):
    """Render search results"""
    column1, column2 = st.columns([6, 90])
    
    with column1:
        st.image(AI_ICON, use_column_width='always')
    
    with column2:
        st.markdown("<div style='fontSize:25px;'>Results</div>", unsafe_allow_html=True)
        
        # Show NDCG score if evaluation is enabled
        if (st.session_state.input_evaluate == "enabled" and 
            st.session_state.input_ndcg > 0):
            span_color = "white"
            if "↑" in st.session_state.ndcg_increase:
                span_color = "green"
            elif "↓" in st.session_state.ndcg_increase:
                span_color = "red"
            
            st.markdown(f"""<span style='fontSize:20px;color:#e28743'>
                          Relevance: {st.session_state.input_ndcg:.3f}</span>
                          <span style='font-size:30px;color:{span_color}'>
                          {st.session_state.ndcg_increase.split('~')[0]}</span>""", 
                       unsafe_allow_html=True)
    
    # Display results
    if not answer:
        st.markdown("<p style='fontSize:20px;color:orange'>No results found</p>", 
                   unsafe_allow_html=True)
        return
    
    col_1, col_2, col_3 = st.columns([70, 10, 20])
    
    for i, result in enumerate(answer):
        with col_1:
            inner_col_1, inner_col_2 = st.columns([8, 92])
            
            with inner_col_2:
                # Display product image
                if 'image_url' in result:
                    st.image(result['image_url'])
                
                # Highlight matched tokens for vector search
                if ('max_score_dict_list_sorted' in result and 
                    'Vector Search' in st.session_state.input_searchType):
                    # Highlight matching words in description
                    highlighted_desc = highlight_matches(
                        result['desc'], 
                        result['max_score_dict_list_sorted']
                    )
                    st.markdown(highlighted_desc, unsafe_allow_html=True)
                
                # Highlight keyword matches
                elif ("highlight" in result and 
                      'Keyword Search' in st.session_state.input_searchType):
                    highlighted_desc = process_highlights(
                        result['desc'],
                        result['highlight']
                    )
                    st.markdown(highlighted_desc, unsafe_allow_html=True)
                else:
                    st.write(result['desc'])
                
                # Show sparse vector details
                if "sparse" in result:
                    with st.expander("Expanded document:"):
                        sparse_sorted = dict(sorted(result['sparse'].items(), 
                                                  key=lambda x: x[1], reverse=True))
                        filtered_sparse = {k: round(v, 2) for k, v in sparse_sorted.items() 
                                         if v >= 1.0}
                        st.write(filtered_sparse)
                
                # Product details expander
                with st.expander(f"{result['caption']}"):
                    st.write(":green[Details:]")
                    st.json({
                        "category": result.get('category', ''),
                        "price": str(result.get('price', 0)),
                        "gender_affinity": result.get('gender_affinity', ''),
                        "style": result.get('style', '')
                    }, expanded=True)
                    
                    # Log interaction
                    ubi.send_to_lambda("ubi_events", {
                        "action_name": "expander_open",
                        "item_id": result['id'],
                        "query_id": st.session_state.query_id,
                        "session_id": st.session_state.session_id,
                        "timestamp": datetime.utcnow().isoformat() + "Z",
                        "message_type": "INFO",
                        "message": f"Expander opened for item {result['id']}"
                    })
            
            with inner_col_1:
                # Show relevance indicator if evaluation is enabled
                if st.session_state.input_evaluate == "enabled":
                    if result.get('relevant', False):
                        st.write("✅")
                    else:
                        st.write("❌")
    
    # Regenerate button
    with col_3:
        if index == len(st.session_state.questions):
            rdn_key = ''.join([chr(ord('a') + i) for i in range(10)])
            
            def on_regenerate():
                # Check if settings changed
                current_settings = get_current_settings()
                old_settings = get_old_settings()
                
                if current_settings != old_settings:
                    st.session_state.input_text = st.session_state.questions[-1]["question"]
                    st.session_state.answers.pop()
                    st.session_state.questions.pop()
                    handle_input()
            
            st.button("🔄", key=rdn_key, on_click=on_regenerate,
                     help="Regenerate results with new settings")

def highlight_matches(text, matches):
    """Highlight matched tokens in text"""
    words = text.split()
    highlighted = []
    
    for word in words:
        clean_word = re.sub(r'[^a-zA-Z0-9]+', '', word).lower()
        highlighted_word = word
        
        for i, match in enumerate(matches[:3]):  # Top 3 matches
            if match['doc_token'] == clean_word:
                colors = ['#8B0001', '#C34632', '#E97452']
                highlighted_word = f"<span style='color:#ffffff;background-color:{colors[i]};font-weight:bold'>{word}</span>"
                break
        
        highlighted.append(highlighted_word)
    
    return "<p>" + " ".join(highlighted) + "</p>"

def process_highlights(text, highlights):
    """Process search highlights from keyword search"""
    highlighted_terms = []
    
    for highlight in highlights:
        # Extract highlighted terms
        matches = re.findall(r'<em>(.*?)</em>', highlight)
        highlighted_terms.extend(matches)
    
    words = text.split()
    highlighted = []
    
    for word in words:
        clean_word = re.sub(r'[^a-zA-Z0-9]+', '', word)
        if any(term in clean_word for term in highlighted_terms):
            highlighted.append(f"<span style='color:#e28743;font-weight:bold'>{word}</span>")
        else:
            highlighted.append(word)
    
    return "<p>" + " ".join(highlighted) + "</p>"

def get_current_settings():
    """Get current search settings for comparison"""
    return "".join([
        str(st.session_state.input_searchType),
        st.session_state.input_imageUpload,
        json.dumps(st.session_state.get('input_weightage', {})),
        str(st.session_state.input_K),
        st.session_state.input_reranker,
        st.session_state.input_is_rewrite_query,
        st.session_state.input_evaluate,
        st.session_state.input_manual_filter
    ])

def get_old_settings():
    """Get previous search settings"""
    if 'inputs_' not in st.session_state:
        return ""
    
    return "".join([
        str(st.session_state.inputs_.get("searchType", [])),
        st.session_state.inputs_.get("imageUpload", ""),
        str(st.session_state.inputs_.get("weightage", {})),
        str(st.session_state.inputs_.get("K", 5)),
        st.session_state.inputs_.get("reranker", "None"),
        st.session_state.inputs_.get("is_rewrite_query", "disabled"),
        st.session_state.inputs_.get("evaluate", "disabled"),
        st.session_state.inputs_.get("manual_filter", "False")
    ])

def render_all():
    """Render all messages and results"""
    for i, (q, a) in enumerate(zip(st.session_state.questions, st.session_state.answers)):
        write_user_message(q, st.session_state.answers[0])
        render_results(a['answer'], i + 1)

# Main UI
clear = write_top_bar()

if clear:
    # Clear all state
    st.session_state.questions = []
    st.session_state.answers = []
    st.session_state.input_text = ""
    st.session_state.input_rekog_label = ""
    st.session_state.img_doc = None

# Search configuration
col1, col3, col4 = st.columns([70, 18, 12])

with col1:
    search_type = st.multiselect(
        'Select the Search type(s)',
        search_types,
        ['Keyword Search'],
        max_selections=None,
        key='input_searchType',
        help="Select search types. Multiple selections enable hybrid search."
    )

with col3:
    st.number_input("No. of docs", min_value=1, max_value=50, value=5, 
                   step=5, key='input_K')

with col4:
    st.markdown("<div style='fontSize:14.5px'>Evaluate</div>", unsafe_allow_html=True)
    evaluate = st.toggle(' ', key='evaluate', disabled=False)
    st.session_state.input_evaluate = "enabled" if evaluate else "disabled"

# Sidebar configuration
with st.sidebar:
    st.page_link("app.py", label=":orange[Home]", icon="🏠")
    
    # Query rewriting
    rewrite_query = st.checkbox('Auto-apply filters', key='query_rewrite', 
                               help="Use LLM to rewrite query with filters")
    st.session_state.input_is_rewrite_query = "enabled" if rewrite_query else "disabled"
    
    if rewrite_query:
        st.multiselect('Fields for "MUST" filter',
                      ('Price', 'Gender', 'Color', 'Category', 'Style'),
                      ['Category'],
                      key='input_must')
    
    # Manual filters
    st.subheader(':blue[Filters]')
    
    def clear_filter():
        st.session_state.input_manual_filter = "False"
        st.session_state.input_category = None
        st.session_state.input_gender = None
        st.session_state.input_price = (0, 0)
        handle_input()
    
    st.selectbox("Select Category", 
                ("accessories", "books", "apparel", "footwear", "electronics", 
                 "beauty", "jewelry", "housewares", "outdoors", "furniture"),
                index=None, key="input_category")
    
    st.selectbox("Select Gender", ("male", "female"), 
                index=None, key="input_gender")
    
    st.slider("Price range", 0, 2000, (0, 0), 50, key="input_price")
    
    if (st.session_state.input_category or st.session_state.input_gender or 
        st.session_state.input_price != (0, 0)):
        st.session_state.input_manual_filter = "True"
    else:
        st.session_state.input_manual_filter = "False"
    
    st.button("Clear Filters", on_click=clear_filter)
    
    # Neural Sparse configuration
    if 'NeuralSparse Search' in st.session_state.input_searchType:
        st.subheader(':blue[Neural Sparse Search]')
        st.slider('Min token weight', 0.0, 1.0, 0.5, 0.1,
                 key='input_sparse_filter',
                 help='Filter sparse tokens by minimum weight')
    
    # Vector search configuration
    st.subheader(':blue[Vector Search]')
    
    mvector_rerank = st.checkbox(
        "Token-level vectors (ColBERT)",
        key='mvector_rerank',
        help="Use token-level embeddings with MaxSim reranking"
    )
    st.session_state.input_mvector_rerank = mvector_rerank
    
    multilingual = st.checkbox(
        "Enable multilingual mode",
        key='multilingual',
        help="Use multilingual embeddings and translation"
    )
    st.session_state.input_multilingual = multilingual
    
    # Hybrid search weights
    st.subheader(':blue[Hybrid Search]')
    with st.expander("Set query weights:"):
        st.number_input("Keyword %", 0, 100, 100, 5, key='input_Keyword-weight')
        st.number_input("Vector %", 0, 100, 0, 5, key='input_Vector-weight')
        st.number_input("Multimodal %", 0, 100, 0, 5, key='input_Multimodal-weight')
        st.number_input("NeuralSparse %", 0, 100, 0, 5, key='input_NeuralSparse-weight')
    
    # Re-ranking
    st.subheader(':blue[Re-ranking]')
    st.selectbox(
        'Choose a Re-Ranker',
        ('None', 'Cross Encoder', 'Cohere Rerank'),
        key='input_reranker',
        help='Select re-ranking method'
    )

# Render results
placeholder = st.empty()
with placeholder.container():
    render_all()
