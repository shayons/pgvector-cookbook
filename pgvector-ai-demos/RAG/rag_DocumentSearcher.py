import os
import time
import streamlit as st
import base64
import re
import utilities.invoke_models as invoke_models
from utilities.pgvector_search import pg_search

parent_dirname = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def query_(awsauth, inputs, session_id, search_types):
    """Query documents for RAG using pgvector"""
    
    print(f"Using index: {st.session_state.input_index}")
    
    question = inputs['query']
    k = 5  # Number of results to retrieve
    
    # Generate embeddings
    start_time = time.time()
    embedding = invoke_models.invoke_model_mm(question, "none")
    print(f"Embedding generation took {time.time() - start_time:.2f} seconds")
    
    # Multimodal search
    print(f"Started multimodal search: {st.session_state.input_index}")
    mm_hits = pg_search.vector_search(
        embedding, 
        st.session_state.input_index, 
        'multimodal_embedding',
        size=1
    )
    print(f"Finished multimodal search: {st.session_state.input_index}")
    
    # Extract images from multimodal results
    images = []
    for hit in mm_hits:
        if hit['_source'].get('image') and hit['_source']['image'] != 'None':
            images.append({
                'file': hit['_source']['image'],
                'caption': hit['_source']['processed_element']
            })
    
    # Main search based on selected types
    num_queries = len(search_types)
    hits = []
    
    if num_queries == 1:
        search_type = search_types[0]
        
        if 'Keyword Search' in search_type:
            hits = pg_search.keyword_search(
                question,
                st.session_state.input_index,
                size=k
            )
        elif 'Vector Search' in search_type:
            text_embedding = invoke_models.invoke_model(question)
            hits = pg_search.vector_search(
                text_embedding,
                st.session_state.input_index,
                'processed_element_embedding',
                size=k
            )
        elif 'Sparse Search' in search_type:
            sparse_result = invoke_models.get_sparse_vector(question)
            hits = pg_search.sparse_search(
                sparse_result['sparse_vector'],
                st.session_state.input_index,
                min_score=0.5,
                size=k
            )
    else:
        # Hybrid search
        query_vectors = {}
        sparse_vector = None
        
        if 'Vector Search' in search_types:
            text_embedding = invoke_models.invoke_model(question)
            query_vectors['processed_element_embedding'] = text_embedding
        
        if 'Sparse Search' in search_types:
            sparse_result = invoke_models.get_sparse_vector(question)
            sparse_vector = sparse_result['sparse_vector']
        
        # Calculate weights
        weights = {
            'keyword': 1.0 / num_queries if 'Keyword Search' in search_types else 0,
            'vector': 1.0 / num_queries if 'Vector Search' in search_types else 0,
            'sparse': 1.0 / num_queries if 'Sparse Search' in search_types else 0
        }
        
        hits = pg_search.hybrid_search(
            question,
            query_vectors,
            sparse_vector,
            weights,
            st.session_state.input_index,
            size=k
        )
    
    # Apply reranking if enabled
    if st.session_state.get('input_is_rerank'):
        hits = pg_search.rerank_results(question, hits, 'cross_encoder')
    
    # Process results for context generation
    context = []
    context_tables = []
    images_2 = []
    table_refs = []
    
    for idx, hit in enumerate(hits[:5]):
        source = hit['_source']
        
        if source['raw_element_type'] == 'table':
            # Handle table results
            table_refs.append({
                'name': source.get('table_name', f'table_{idx}'),
                'text': source['processed_element']
            })
            context_tables.append(
                f"{idx + 1}: Reference from a table: {source['processed_element']}"
            )
        else:
            # Handle text and image results
            if source.get('image') and source['image'] != 'None':
                # Process image
                image_path = os.path.join(
                    parent_dirname, 'figures', 
                    st.session_state.input_index,
                    source['image'].replace('.jpg', '') + '-resized.jpg'
                )
                
                if os.path.exists(image_path):
                    with open(image_path, 'rb') as img_file:
                        encoded_img = base64.b64encode(img_file.read()).decode('utf8')
                    
                    # Generate image-specific answer
                    img_caption = invoke_models.generate_image_captions_llm(
                        encoded_img, question
                    )
                    context.append(
                        f"{idx + 1}: Reference from an image: {img_caption}"
                    )
                    images_2.append({
                        'file': source['image'],
                        'caption': source['processed_element']
                    })
            else:
                # Regular text
                context.append(
                    f"{idx + 1}: Reference from a text chunk: {source['processed_element']}"
                )
    
    # Combine all context
    total_context = context_tables + context
    
    # Generate final answer using LLM
    prompt_template = """
    The following is a friendly conversation between a human and an AI. 
    The AI is talkative and provides lots of specific details from its context.
    
    {context}
    
    Instruction: Based on the above documents, provide a detailed answer for: {question}
    Answer "don't know" if not present in the context. 
    
    Solution:"""
    
    llm_prompt = prompt_template.format(
        context="\n".join(total_context[:3]),
        question=question
    )
    
    print(f"Started LLM prompt: {st.session_state.input_index}")
    output = invoke_models.invoke_llm_model(
        f"\n\nHuman: {llm_prompt}\n\nAssistant:", 
        False
    )
    print(f"Finished LLM prompt: {st.session_state.input_index}")
    
    # Use multimodal images if no other images found
    if not images_2:
        images_2 = images
    
    return {
        'text': output,
        'source': total_context,
        'image': images_2,
        'table': table_refs
    }
