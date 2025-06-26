import streamlit as st
import uuid
import os
import sys
import json
from PIL import Image
import base64
import requests
from io import BytesIO
import warnings

# Add parent directories to path
sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(1, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "RAG"))
sys.path.insert(1, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "utilities"))

import RAG.bedrock_agent as bedrock_agent

warnings.filterwarnings("ignore", category=DeprecationWarning)

st.set_page_config(
    layout="wide",
    page_icon="images/opensearch_mark_default.png",
    page_title="AI Shopping Assistant with pgvector"
)

parent_dirname = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
USER_ICON = "images/user.png"
AI_ICON = "images/opensearch-twitter-card.png"
REGENERATE_ICON = "images/regenerate.png"

# Initialize session state
if 'user_id' not in st.session_state:
    st.session_state['user_id'] = str(uuid.uuid4())

if 'session_id_' not in st.session_state:
    st.session_state['session_id_'] = str(uuid.uuid1())

if "questions__" not in st.session_state:
    st.session_state.questions__ = []

if "answers__" not in st.session_state:
    st.session_state.answers__ = []

if "inputs_" not in st.session_state:
    st.session_state.inputs_ = {}

if "input_shopping_query" not in st.session_state:
    st.session_state.input_shopping_query = "get me shoes suitable for trekking"

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
        st.header("AI Shopping assistant", divider='rainbow')
    
    with col2:
        st.write("")
        st.write("")
        clear = st.button("Clear")
    
    st.write("")
    st.write("")
    return clear

def handle_input():
    """Handle user input and query the agent"""
    if st.session_state.input_shopping_query == '':
        return ""
    
    # Collect inputs
    inputs = {}
    for key in st.session_state:
        if key.startswith('input_'):
            inputs[key.removeprefix('input_')] = st.session_state[key]
    
    st.session_state.inputs_ = inputs
    
    # Add question to history
    question_with_id = {
        'question': inputs["shopping_query"],
        'id': len(st.session_state.questions__)
    }
    st.session_state.questions__.append(question_with_id)
    
    # Query the agent
    try:
        result = bedrock_agent.query_(inputs)
        
        st.session_state.answers__.append({
            'answer': result['text'],
            'source': result['source'],
            'last_tool': result['last_tool'],
            'id': len(st.session_state.questions__)
        })
    except Exception as e:
        st.error(f"Error querying agent: {e}")
        st.session_state.answers__.append({
            'answer': "I'm sorry, I encountered an error processing your request. Please try again.",
            'source': {'error': str(e)},
            'last_tool': {'name': 'error', 'response': ''},
            'id': len(st.session_state.questions__)
        })
    
    # Clear input
    st.session_state.input_shopping_query = ""

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

def render_answer(question, answer, index):
    """Render agent response"""
    col1, col2, col3 = st.columns([4, 74, 22])
    
    with col1:
        st.image(AI_ICON, use_column_width='always')
    
    with col2:
        # Display answer with formatting
        ans_text = answer['answer']
        
        # Replace question tags with formatted spans
        formatted_answer = ans_text.replace(
            '<question>',
            "<span style='fontSize:18px;color:#f37709;fontStyle:italic;'>"
        ).replace('</question>', "</span>")
        
        st.markdown(f"<p>{formatted_answer}</p>", unsafe_allow_html=True)
        
        # Display tool results if applicable
        tool_name = answer['last_tool']['name']
        
        if tool_name in ["generate_images", "get_relevant_items_for_image", 
                        "get_relevant_items_for_text", "retrieve_with_hybrid_search", 
                        "retrieve_with_keyword_search", "get_any_general_recommendation"]:
            
            # Parse tool response
            try:
                tool_response = json.loads(answer['last_tool']['response'].replace("'", '"'))
            except:
                tool_response = {}
            
            # Display product recommendations
            if tool_name != 'generate_images' and tool_name != 'get_any_general_recommendation':
                if tool_name in tool_response:
                    st.write("<br><br>", unsafe_allow_html=True)
                    
                    img_col1, img_col2, img_col3 = st.columns([30, 30, 40])
                    
                    for idx, item in enumerate(tool_response[tool_name][:2]):
                        # Try to load image
                        try:
                            if item.get('image'):
                                response = requests.get(item['image'])
                                img = Image.open(BytesIO(response.content))
                                resized_img = img.resize((230, 180), Image.Resampling.LANCZOS)
                            else:
                                # Create placeholder image
                                resized_img = Image.new('RGB', (230, 180), color='gray')
                        except:
                            resized_img = Image.new('RGB', (230, 180), color='gray')
                        
                        # Display in appropriate column
                        if idx == 0:
                            with img_col1:
                                st.image(resized_img, use_column_width=True, 
                                       caption=item.get('title', 'Product'))
                        elif idx == 1:
                            with img_col2:
                                st.image(resized_img, use_column_width=True,
                                       caption=item.get('title', 'Product'))
            
            # Display generated images
            elif tool_name == "generate_images" or tool_name == "get_any_general_recommendation":
                st.write("<br>", unsafe_allow_html=True)
                
                if 'generate_images' in tool_response:
                    try:
                        # Parse S3 path
                        s3_path = tool_response['generate_images'].replace('s3://', '')
                        bucket, key = s3_path.split('/', 1)
                        
                        # Try to load from S3 (would need boto3 client)
                        # For now, create placeholder
                        img = Image.new('RGB', (230, 180), color='gray')
                        
                        gen_img_col1, gen_img_col2, gen_img_col3 = st.columns([30, 30, 30])
                        with gen_img_col1:
                            st.image(img, caption=f"Generated image for {key.split('.')[0]}",
                                   use_column_width=True)
                    except Exception as e:
                        st.info("Generated image preview not available")
                
                st.write("<br>", unsafe_allow_html=True)
    
    # Agent traces expander
    colu1, colu2, colu3 = st.columns([4, 82, 20])
    if answer['source'] != {}:
        with colu2:
            with st.expander("Agent Traces:"):
                # Format traces for readability
                formatted_traces = []
                for trace in answer['source']:
                    if isinstance(trace, dict):
                        if 'rationale' in trace:
                            formatted_traces.append(f"**Rationale:** {trace['rationale']}")
                        if 'invocationInput' in trace:
                            formatted_traces.append(
                                f"**Tool Called:** {trace['invocationInput'].get('function', 'Unknown')}"
                            )
                        if 'observation' in trace:
                            obs = trace['observation']
                            formatted_traces.append(
                                f"**Result Type:** {obs.get('type', 'Unknown')}"
                            )
                
                for trace in formatted_traces:
                    st.markdown(trace)

def write_chat_message(md, q, index):
    """Write complete chat message"""
    chat = st.container()
    with chat:
        render_answer(q, md, index)

def render_all():
    """Render all messages"""
    for q, a in zip(st.session_state.questions__, st.session_state.answers__):
        write_user_message(q)
        write_chat_message(a, q, a['id'])

# Main UI
clear = write_top_bar()

if clear:
    st.session_state.questions__ = []
    st.session_state.answers__ = []
    st.session_state.input_shopping_query = ""
    st.session_state.session_id_ = str(uuid.uuid1())
    bedrock_agent.delete_memory()

# Render chat history
placeholder = st.empty()
with placeholder.container():
    render_all()

st.markdown("")

# Input section
col_2, col_3 = st.columns([75, 20])

with col_2:
    st.text_input("Ask here", label_visibility="collapsed", 
                 key="input_shopping_query")
with col_3:
    st.button("Go", on_click=handle_input, key="play")

# Sidebar with examples
with st.sidebar:
    st.page_link("app.py", label=":orange[Home]", icon="🏠")
    
    st.subheader(":blue[AI Shopping Assistant]")
    st.markdown(
        "This demo showcases an AI agent that can help you find products "
        "using natural language queries. The agent can:\n"
        "- Search for products based on your description\n"
        "- Generate product images\n"
        "- Find similar items\n"
        "- Provide recommendations\n"
        "- Remember context from previous queries"
    )
    
    st.subheader(":blue[Example Queries]")
    examples = [
        "Get me shoes suitable for trekking",
        "I need a waterproof jacket for hiking",
        "Show me camping gear under $100",
        "Find me a gift for someone who likes cooking",
        "What do you have for outdoor activities?",
        "I'm looking for running shoes for women",
        "Show me electronics for home office",
        "Find accessories for yoga practice"
    ]
    
    for example in examples:
        if st.button(example, key=f"ex_{example[:20]}"):
            st.session_state.input_shopping_query = example
            handle_input()
    
    st.subheader(":blue[Agent Capabilities]")
    st.info(
        "The agent uses multiple tools:\n"
        "- **Text Search**: Find products by description\n"
        "- **Image Search**: Find similar products by image\n"
        "- **Hybrid Search**: Combine multiple search methods\n"
        "- **Image Generation**: Create product visualizations\n"
        "- **Recommendations**: Get personalized suggestions"
    )
