import boto3
import json
import uuid
import logging
import streamlit as st
from datetime import datetime, timezone
from pgvector_db_setup import get_db_connection
from sqlalchemy import text
import utilities.invoke_models as invoke_models
from utilities.pgvector_search import pg_search

# Setting up logging
logging.basicConfig(
    format='[%(asctime)s] p%(process)s {%(filename)s:%(lineno)d} %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Initialize Bedrock client
bedrock_agent_runtime_client = boto3.client(
    'bedrock-agent-runtime',
    aws_access_key_id=st.secrets['user_access_key_us_west_2'],
    aws_secret_access_key=st.secrets['user_secret_key_us_west_2'],
    region_name='us-west-2'
)

# Agent configuration
AGENT_ID = 'EJVGQW1BH7'
AGENT_ALIAS_ID = 'DEEEEZM2TM'

class PgvectorAgentMemory:
    """Handle agent memory storage in pgvector"""
    
    def __init__(self):
        self.engine = get_db_connection()
    
    def save_memory(self, session_id, agent_id, memory_data):
        """Save agent memory to database"""
        with self.engine.connect() as conn:
            conn.execute(
                text("""
                    INSERT INTO agent_memory (session_id, agent_id, memory_data)
                    VALUES (:session_id, :agent_id, :memory_data)
                    ON CONFLICT (session_id, agent_id) 
                    DO UPDATE SET 
                        memory_data = :memory_data,
                        updated_at = CURRENT_TIMESTAMP
                """),
                {
                    "session_id": session_id,
                    "agent_id": agent_id,
                    "memory_data": json.dumps(memory_data)
                }
            )
            conn.commit()
    
    def get_memory(self, session_id, agent_id):
        """Retrieve agent memory from database"""
        with self.engine.connect() as conn:
            result = conn.execute(
                text("""
                    SELECT memory_data 
                    FROM agent_memory 
                    WHERE session_id = :session_id AND agent_id = :agent_id
                """),
                {
                    "session_id": session_id,
                    "agent_id": agent_id
                }
            )
            row = result.fetchone()
            return json.loads(row[0]) if row else {}
    
    def delete_memory(self, session_id=None, agent_id=None):
        """Delete agent memory"""
        with self.engine.connect() as conn:
            if session_id and agent_id:
                conn.execute(
                    text("""
                        DELETE FROM agent_memory 
                        WHERE session_id = :session_id AND agent_id = :agent_id
                    """),
                    {
                        "session_id": session_id,
                        "agent_id": agent_id
                    }
                )
            elif session_id:
                conn.execute(
                    text("DELETE FROM agent_memory WHERE session_id = :session_id"),
                    {"session_id": session_id}
                )
            else:
                # Delete all memory for current session
                if 'session_id_' in st.session_state:
                    conn.execute(
                        text("""
                            DELETE FROM agent_memory 
                            WHERE session_id = :session_id AND agent_id = :agent_id
                        """),
                        {
                            "session_id": st.session_state.session_id_,
                            "agent_id": AGENT_ID
                        }
                    )
            conn.commit()

# Initialize memory handler
agent_memory = PgvectorAgentMemory()

def delete_memory():
    """Delete agent memory for current session"""
    agent_memory.delete_memory()
    
    # Also try to delete from Bedrock (if supported)
    try:
        response = bedrock_agent_runtime_client.delete_agent_memory(
            agentAliasId=AGENT_ALIAS_ID,
            agentId=AGENT_ID
        )
        logger.info("Bedrock agent memory deleted")
    except Exception as e:
        logger.warning(f"Could not delete Bedrock agent memory: {e}")

def search_products(query, filters=None):
    """Search products using pgvector"""
    # Generate embeddings
    text_embedding = invoke_models.invoke_model(query)
    
    # Perform hybrid search
    results = pg_search.hybrid_search(
        query=query,
        query_vectors={"description_vector_titan": text_embedding},
        weights={"keyword": 0.4, "vector": 0.6},
        index_name="products",
        filters=filters,
        size=10
    )
    
    return results

def generate_images_from_search(query):
    """Generate images based on search query"""
    # This would integrate with your image generation service
    # For now, return placeholder
    return {
        "generate_images": f"s3://generated-images/{query.replace(' ', '_')}.jpg"
    }

def get_relevant_items_for_text(query, category=None):
    """Get relevant items for text query"""
    filters = {"category": category} if category else None
    results = search_products(query, filters)
    
    formatted_results = []
    for result in results[:5]:
        source = result['_source']
        formatted_results.append({
            "title": source.get('caption', ''),
            "description": source.get('product_description', ''),
            "price": f"${source.get('price', 0):.2f}",
            "image": source.get('image_url', ''),
            "category": source.get('category', ''),
            "id": result['_id']
        })
    
    return {"get_relevant_items_for_text": formatted_results}

def get_relevant_items_for_image(image_data):
    """Get relevant items for image query"""
    # Generate multimodal embedding
    mm_embedding = invoke_models.invoke_model_mm("", image_data)
    
    # Search using multimodal vector
    results = pg_search.vector_search(
        query_vector=mm_embedding,
        index_name="products",
        vector_field="multimodal_vector",
        size=10
    )
    
    formatted_results = []
    for result in results[:5]:
        source = result['_source']
        formatted_results.append({
            "title": source.get('caption', ''),
            "description": source.get('product_description', ''),
            "price": f"${source.get('price', 0):.2f}",
            "image": source.get('image_url', ''),
            "category": source.get('category', ''),
            "id": result['_id']
        })
    
    return {"get_relevant_items_for_image": formatted_results}

def retrieve_with_hybrid_search(query, filters=None):
    """Perform hybrid search combining multiple methods"""
    # Generate all embeddings
    text_embedding = invoke_models.invoke_model(query)
    sparse_result = invoke_models.get_sparse_vector(query)
    
    # Perform hybrid search
    results = pg_search.hybrid_search(
        query=query,
        query_vectors={"description_vector_titan": text_embedding},
        sparse_vector=sparse_result['sparse_vector'],
        weights={"keyword": 0.3, "vector": 0.5, "sparse": 0.2},
        index_name="products",
        filters=filters,
        size=10
    )
    
    formatted_results = []
    for result in results[:5]:
        source = result['_source']
        formatted_results.append({
            "title": source.get('caption', ''),
            "description": source.get('product_description', ''),
            "price": f"${source.get('price', 0):.2f}",
            "image": source.get('image_url', ''),
            "category": source.get('category', ''),
            "id": result['_id']
        })
    
    return {"retrieve_with_hybrid_search": formatted_results}

def retrieve_with_keyword_search(query, filters=None):
    """Perform keyword search"""
    results = pg_search.keyword_search(
        query=query,
        index_name="products",
        filters=filters,
        size=10
    )
    
    formatted_results = []
    for result in results[:5]:
        source = result['_source']
        formatted_results.append({
            "title": source.get('caption', ''),
            "description": source.get('product_description', ''),
            "price": f"${source.get('price', 0):.2f}",
            "image": source.get('image_url', ''),
            "category": source.get('category', ''),
            "id": result['_id']
        })
    
    return {"retrieve_with_keyword_search": formatted_results}

def get_any_general_recommendation(category=None):
    """Get general product recommendations"""
    # Get popular or featured products
    with agent_memory.engine.connect() as conn:
        query = """
            SELECT * FROM products
            WHERE current_stock > 0
        """
        params = {}
        
        if category:
            query += " AND category = :category"
            params['category'] = category
        
        query += " ORDER BY price DESC LIMIT 5"
        
        result = conn.execute(text(query), params)
        
        formatted_results = []
        for row in result:
            formatted_results.append({
                "title": row.caption,
                "description": row.product_description,
                "price": f"${row.price:.2f}",
                "image": row.image_url,
                "category": row.category,
                "id": str(row.id)
            })
    
    return {"get_any_general_recommendation": formatted_results}

# Tool mapping for agent
TOOL_FUNCTIONS = {
    "generate_images": generate_images_from_search,
    "get_relevant_items_for_image": get_relevant_items_for_image,
    "get_relevant_items_for_text": get_relevant_items_for_text,
    "retrieve_with_hybrid_search": retrieve_with_hybrid_search,
    "retrieve_with_keyword_search": retrieve_with_keyword_search,
    "get_any_general_recommendation": get_any_general_recommendation
}

def process_tool_call(tool_name, parameters):
    """Process a tool call from the agent"""
    if tool_name in TOOL_FUNCTIONS:
        # Extract parameters
        params = {}
        for param in parameters:
            params[param['name']] = param['value']
        
        # Call the appropriate function
        result = TOOL_FUNCTIONS[tool_name](**params)
        return result
    else:
        logger.warning(f"Unknown tool: {tool_name}")
        return {}

def query_(inputs):
    """Main query function for agent interaction"""
    session_id = st.session_state.get('session_id_', str(uuid.uuid4()))
    
    # Restore memory if exists
    memory_data = agent_memory.get_memory(session_id, AGENT_ID)
    
    try:
        # Invoke the agent
        agent_response = bedrock_agent_runtime_client.invoke_agent(
            inputText=inputs['shopping_query'],
            agentId=AGENT_ID,
            agentAliasId=AGENT_ALIAS_ID,
            sessionId=session_id,
            enableTrace=True,
            endSession=False
        )
        
        logger.info("Agent invoked successfully")
        
        event_stream = agent_response['completion']
        total_context = []
        last_tool = ""
        last_tool_name = ""
        agent_answer = ""
        
        for event in event_stream:
            if 'trace' in event:
                if 'orchestrationTrace' not in event['trace']['trace']:
                    continue
                
                orchestration_trace = event['trace']['trace']['orchestrationTrace']
                context_item = {}
                
                # Process different trace types
                if 'rationale' in orchestration_trace:
                    context_item['rationale'] = orchestration_trace['rationale']['text']
                
                if 'invocationInput' in orchestration_trace:
                    invocation = orchestration_trace['invocationInput']['actionGroupInvocationInput']
                    context_item['invocationInput'] = invocation
                    last_tool_name = invocation['function']
                    
                    # Process tool call locally
                    tool_result = process_tool_call(
                        last_tool_name,
                        invocation.get('parameters', [])
                    )
                    last_tool = json.dumps(tool_result)
                
                if 'observation' in orchestration_trace:
                    observation = orchestration_trace['observation']
                    context_item['observation'] = observation
                    
                    if observation['type'] == 'ACTION_GROUP':
                        # Use our local tool result
                        last_tool = last_tool or observation['actionGroupInvocationOutput']['text']
                    elif observation['type'] == 'FINISH':
                        agent_answer = observation['finalResponse']['text']
                
                if context_item:
                    total_context.append(context_item)
        
        # Save updated memory
        agent_memory.save_memory(session_id, AGENT_ID, {
            "last_query": inputs['shopping_query'],
            "last_response": agent_answer,
            "context": total_context[-5:] if len(total_context) > 5 else total_context
        })
        
        return {
            'text': agent_answer,
            'source': total_context,
            'last_tool': {
                'name': last_tool_name,
                'response': last_tool
            }
        }
        
    except Exception as e:
        logger.error(f"Agent query failed: {e}")
        
        # Fallback to direct search
        fallback_results = get_relevant_items_for_text(inputs['shopping_query'])
        fallback_answer = f"I found {len(fallback_results['get_relevant_items_for_text'])} items for '{inputs['shopping_query']}'. Here are the top results."
        
        return {
            'text': fallback_answer,
            'source': [{"fallback": True, "error": str(e)}],
            'last_tool': {
                'name': 'get_relevant_items_for_text',
                'response': json.dumps(fallback_results)
            }
        }
