'''
pgvector version of search execution
'''

from datetime import datetime
import json
import os
import uuid
import streamlit as st
from utilities.pgvector_search import pg_search
import utilities.mvectors as cb
import utilities.ubi_lambda as ubi
import utilities.invoke_models as invoke_models

def handler(input_, session_id):
    """Main search handler that processes all search types"""
    
    # Parse inputs
    search_types = input_["searchType"]
    query = input_["text"]
    img = input_["image"]
    k_ = input_["K"]
    image_upload = input_["imageUpload"]
    
    # Get weights for hybrid search
    weights = {}
    for search_type in ['Keyword', 'Vector', 'Multimodal', 'NeuralSparse']:
        weight = input_['weightage'].get(f'{search_type}-weight', 0) / 100
        if weight > 0:
            weights[search_type.lower()] = weight
    
    # Prepare filters
    filters = {}
    
    # Apply manual filters
    if input_.get('manual_filter') == "True":
        if st.session_state.get('input_category'):
            filters['category'] = st.session_state.input_category
        if st.session_state.get('input_gender'):
            filters['gender_affinity'] = st.session_state.input_gender
        if st.session_state.get('input_price', (0, 0)) != (0, 0):
            filters['price'] = {
                'gte': st.session_state.input_price[0],
                'lte': st.session_state.input_price[1]
            }
    
    # Apply query rewriting filters
    if st.session_state.get('input_rewritten_query'):
        # Parse rewritten query filters
        rewritten = st.session_state.input_rewritten_query
        if 'query' in rewritten and 'bool' in rewritten['query']:
            must_clauses = rewritten['query']['bool'].get('must', [])
            for clause in must_clauses:
                if 'match' in clause:
                    for field, value in clause['match'].items():
                        if field in ['category', 'gender_affinity', 'color', 'style']:
                            filters[field] = value
    
    # Prepare query vectors
    query_vectors = {}
    sparse_vector = None
    
    # Single search type
    if len(search_types) == 1:
        search_type = search_types[0]
        
        if search_type == 'Keyword Search':
            if st.session_state.get('input_mvector_rerank'):
                # Use token-level search
                docs = pg_search.keyword_search(query, 'products', filters, k_ * 3)
                # Perform ColBERT reranking
                docs = cb.search(docs)
            else:
                docs = pg_search.keyword_search(query, 'products', filters, k_)
                
        elif search_type == 'Vector Search':
            if st.session_state.get('input_mvector_rerank'):
                # Use all-MiniLM-L6-v2 embeddings
                query_vector = cb.vectorise(query, False)
                docs = pg_search.vector_search(query_vector, 'products', 
                                             'description_vector', filters, k_)
            else:
                # Use Titan embeddings
                query_vector = invoke_models.invoke_model(query)
                docs = pg_search.vector_search(query_vector, 'products', 
                                             'description_vector_titan', filters, k_)
                                             
        elif search_type == 'Multimodal Search':
            if image_upload == 'yes' and query:
                # Text + Image
                query_vector = invoke_models.invoke_model_mm(query, img)
            elif image_upload == 'yes':
                # Image only
                query_vector = invoke_models.invoke_model_mm("", img)
            else:
                # Text only
                query_vector = invoke_models.invoke_model_mm(query, "none")
            
            docs = pg_search.vector_search(query_vector, 'products', 
                                         'multimodal_vector', filters, k_)
                                         
        elif search_type == 'NeuralSparse Search':
            # Get sparse vector from model
            sparse_result = invoke_models.get_sparse_vector(query)
            sparse_vector = sparse_result['sparse_vector']
            
            # Filter by minimum score
            min_score = st.session_state.get('input_sparse_filter', 0.5)
            docs = pg_search.sparse_search(sparse_vector, 'products', 
                                         min_score, filters, k_)
    
    else:
        # Hybrid search
        # Prepare all query vectors
        if 'Vector Search' in search_types:
            if st.session_state.get('input_mvector_rerank'):
                query_vectors['description_vector'] = cb.vectorise(query, False)
            else:
                query_vectors['description_vector_titan'] = invoke_models.invoke_model(query)
        
        if 'Multimodal Search' in search_types:
            if image_upload == 'yes' and query:
                query_vectors['multimodal_vector'] = invoke_models.invoke_model_mm(query, img)
            elif image_upload == 'yes':
                query_vectors['multimodal_vector'] = invoke_models.invoke_model_mm("", img)
            else:
                query_vectors['multimodal_vector'] = invoke_models.invoke_model_mm(query, "none")
        
        if 'NeuralSparse Search' in search_types:
            sparse_result = invoke_models.get_sparse_vector(query)
            sparse_vector = sparse_result['sparse_vector']
        
        # Perform hybrid search
        hybrid_weights = {
            'keyword': weights.get('keyword', 0),
            'vector': weights.get('vector', 0) + weights.get('multimodal', 0),
            'sparse': weights.get('neuralsparse', 0)
        }
        
        docs = pg_search.hybrid_search(query, query_vectors, sparse_vector,
                                     hybrid_weights, 'products', filters, k_)
    
    # Apply reranking
    if st.session_state.get('input_reranker', 'None') != 'None':
        docs = pg_search.rerank_results(query, docs, st.session_state.input_reranker)
    
    # Format results
    arr = []
    doc_ids = []
    
    for doc in docs:
        if '_source' not in doc:
            continue
            
        source = doc['_source']
        
        # Skip if image URL is missing or broken
        if not source.get('image_url') or 'b5/b5319e00' in source.get('image_url', ''):
            continue
        
        res_ = {
            "desc": source.get('product_description', ''),
            "caption": source.get('caption', ''),
            "image_url": source.get('image_url', ''),
            "category": source.get('category', ''),
            "price": source.get('price', 0),
            "gender_affinity": source.get('gender_affinity', ''),
            "style": source.get('style', ''),
            "id": doc['_id'],
            "score": doc.get('_score', 0),
            "title": source.get('caption', '')
        }
        
        # Add highlighting if available
        if 'highlight' in doc:
            res_['highlight'] = doc['highlight'].get('product_description', [])
        
        # Add sparse vector info
        if sparse_vector and source.get('sparse_vector'):
            res_['sparse'] = source['sparse_vector']
            res_['query_sparse'] = {k: v for k, v in sparse_vector.items() 
                                   if v >= st.session_state.get('input_sparse_filter', 0.5)}
        
        # Add token matching info for multi-vector search
        if 'max_score_dict_list_sorted' in doc:
            res_['max_score_dict_list_sorted'] = doc['max_score_dict_list_sorted']
        
        arr.append(res_)
        doc_ids.append(doc['_id'])
    
    # Log query for UBI
    st.session_state["query_id"] = str(uuid.uuid4())
    query_payload = {
        "client_id": session_id,
        "query_id": st.session_state["query_id"],
        "application": "Semantic Search",
        "query_response_hit_ids": doc_ids,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "user_query": json.dumps(input_),
        "query": query,
    }
    
    # Send to UBI lambda (optional - can be replaced with pgvector storage)
    try:
        status = ubi.send_to_lambda("ubi_queries", query_payload)
        if status == 202:
            print("Query sent to Lambda")
    except Exception as e:
        print(f"UBI logging failed: {e}")
    
    return arr[:k_]
