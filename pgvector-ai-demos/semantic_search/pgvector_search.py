import numpy as np
import json
from sqlalchemy import create_engine, text, and_, or_
from sqlalchemy.orm import sessionmaker
import streamlit as st
from typing import List, Dict, Any, Optional, Tuple
import asyncio
import asyncpg
from pgvector_db_setup import get_db_connection, DB_CONFIG

class PGVectorSearch:
    def __init__(self):
        self.engine = get_db_connection()
        
    def keyword_search(self, query: str, index_name: str = 'products', 
                      filters: Optional[Dict] = None, size: int = 10) -> List[Dict]:
        """Perform keyword search using PostgreSQL full-text search"""
        with self.engine.connect() as conn:
            # Build the base query
            if index_name == 'products':
                base_query = """
                    SELECT *, 
                           ts_rank(to_tsvector('english', product_description), 
                                  plainto_tsquery('english', :query)) as score
                    FROM products
                    WHERE to_tsvector('english', product_description) @@ plainto_tsquery('english', :query)
                          OR to_tsvector('english', caption) @@ plainto_tsquery('english', :query)
                """
            else:  # RAG documents
                base_query = """
                    SELECT *, 
                           ts_rank(to_tsvector('english', processed_element), 
                                  plainto_tsquery('english', :query)) as score
                    FROM rag_documents
                    WHERE index_name = :index_name
                          AND to_tsvector('english', processed_element) @@ plainto_tsquery('english', :query)
                """
            
            # Add filters
            filter_conditions = []
            params = {'query': query}
            
            if index_name != 'products':
                params['index_name'] = index_name
            
            if filters:
                for key, value in filters.items():
                    if key == 'category' and value:
                        filter_conditions.append("category = :category")
                        params['category'] = value
                    elif key == 'gender_affinity' and value:
                        filter_conditions.append("gender_affinity = :gender")
                        params['gender'] = value
                    elif key == 'price' and value:
                        if isinstance(value, dict):
                            if 'gte' in value:
                                filter_conditions.append("price >= :price_gte")
                                params['price_gte'] = value['gte']
                            if 'lte' in value:
                                filter_conditions.append("price <= :price_lte")
                                params['price_lte'] = value['lte']
            
            if filter_conditions:
                base_query += " AND " + " AND ".join(filter_conditions)
            
            base_query += " ORDER BY score DESC LIMIT :size"
            params['size'] = size
            
            result = conn.execute(text(base_query), params)
            
            # Format results
            hits = []
            for row in result:
                hit = {
                    '_id': str(row.id),
                    '_score': float(row.score) if row.score else 0.0,
                    '_source': dict(row._mapping)
                }
                # Remove internal fields
                hit['_source'].pop('id', None)
                hit['_source'].pop('score', None)
                hits.append(hit)
            
            return hits
    
    def vector_search(self, query_vector: List[float], index_name: str = 'products',
                     vector_field: str = 'description_vector', filters: Optional[Dict] = None,
                     size: int = 10) -> List[Dict]:
        """Perform vector similarity search"""
        with self.engine.connect() as conn:
            # Convert query vector to PostgreSQL array format
            vector_str = f"[{','.join(map(str, query_vector))}]"
            
            # Build the base query
            if index_name == 'products':
                base_query = f"""
                    SELECT *, 
                           1 - ({vector_field} <=> :query_vector::vector) as score
                    FROM products
                    WHERE {vector_field} IS NOT NULL
                """
            else:  # RAG documents
                vector_field = 'processed_element_embedding' if 'multimodal' not in vector_field else 'multimodal_embedding'
                base_query = f"""
                    SELECT *, 
                           1 - ({vector_field} <=> :query_vector::vector) as score
                    FROM rag_documents
                    WHERE index_name = :index_name
                          AND {vector_field} IS NOT NULL
                """
            
            # Add filters (same as keyword search)
            filter_conditions = []
            params = {'query_vector': vector_str}
            
            if index_name != 'products':
                params['index_name'] = index_name
            
            if filters:
                for key, value in filters.items():
                    if key == 'category' and value:
                        filter_conditions.append("category = :category")
                        params['category'] = value
                    elif key == 'gender_affinity' and value:
                        filter_conditions.append("gender_affinity = :gender")
                        params['gender'] = value
                    elif key == 'price' and value:
                        if isinstance(value, dict):
                            if 'gte' in value:
                                filter_conditions.append("price >= :price_gte")
                                params['price_gte'] = value['gte']
                            if 'lte' in value:
                                filter_conditions.append("price <= :price_lte")
                                params['price_lte'] = value['lte']
            
            if filter_conditions:
                base_query += " AND " + " AND ".join(filter_conditions)
            
            base_query += " ORDER BY score DESC LIMIT :size"
            params['size'] = size
            
            result = conn.execute(text(base_query), params)
            
            # Format results
            hits = []
            for row in result:
                hit = {
                    '_id': str(row.id),
                    '_score': float(row.score) if row.score else 0.0,
                    '_source': dict(row._mapping)
                }
                # Remove internal fields
                hit['_source'].pop('id', None)
                hit['_source'].pop('score', None)
                hit['_source'].pop(vector_field, None)
                hits.append(hit)
            
            return hits
    
    def sparse_search(self, sparse_vector: Dict[str, float], index_name: str = 'products',
                     min_score: float = 0.5, filters: Optional[Dict] = None,
                     size: int = 10) -> List[Dict]:
        """Perform sparse vector search using JSONB"""
        with self.engine.connect() as conn:
            # Filter sparse vector by minimum score
            filtered_sparse = {k: v for k, v in sparse_vector.items() if v >= min_score}
            
            # Build the query
            if index_name == 'products':
                # Calculate dot product using JSONB
                sparse_keys = list(filtered_sparse.keys())
                score_calculation = " + ".join([
                    f"COALESCE((sparse_vector->>'{k}')::float * {v}, 0)"
                    for k, v in filtered_sparse.items()
                ])
                
                base_query = f"""
                    SELECT *, 
                           ({score_calculation}) as score
                    FROM products
                    WHERE sparse_vector IS NOT NULL
                          AND sparse_vector ?| ARRAY[:sparse_keys]
                """
                params = {'sparse_keys': sparse_keys}
            else:
                # Similar for RAG documents
                sparse_keys = list(filtered_sparse.keys())
                score_calculation = " + ".join([
                    f"COALESCE((processed_element_embedding_sparse->>'{k}')::float * {v}, 0)"
                    for k, v in filtered_sparse.items()
                ])
                
                base_query = f"""
                    SELECT *, 
                           ({score_calculation}) as score
                    FROM rag_documents
                    WHERE index_name = :index_name
                          AND processed_element_embedding_sparse IS NOT NULL
                          AND processed_element_embedding_sparse ?| ARRAY[:sparse_keys]
                """
                params = {'index_name': index_name, 'sparse_keys': sparse_keys}
            
            # Add filters
            if filters:
                filter_conditions = []
                for key, value in filters.items():
                    if key == 'category' and value:
                        filter_conditions.append("category = :category")
                        params['category'] = value
                    elif key == 'gender_affinity' and value:
                        filter_conditions.append("gender_affinity = :gender")
                        params['gender'] = value
                    elif key == 'price' and value:
                        if isinstance(value, dict):
                            if 'gte' in value:
                                filter_conditions.append("price >= :price_gte")
                                params['price_gte'] = value['gte']
                            if 'lte' in value:
                                filter_conditions.append("price <= :price_lte")
                                params['price_lte'] = value['lte']
                
                if filter_conditions:
                    base_query += " AND " + " AND ".join(filter_conditions)
            
            base_query += " ORDER BY score DESC LIMIT :size"
            params['size'] = size
            
            result = conn.execute(text(base_query), params)
            
            # Format results
            hits = []
            for row in result:
                hit = {
                    '_id': str(row.id),
                    '_score': float(row.score) if row.score else 0.0,
                    '_source': dict(row._mapping)
                }
                # Remove internal fields
                hit['_source'].pop('id', None)
                hit['_source'].pop('score', None)
                
                # Add sparse vector info
                if index_name == 'products' and row.sparse_vector:
                    hit['_source']['sparse'] = row.sparse_vector
                
                hits.append(hit)
            
            return hits
    
    def hybrid_search(self, query: str, query_vectors: Dict[str, List[float]], 
                     sparse_vector: Optional[Dict[str, float]] = None,
                     weights: Dict[str, float] = None, index_name: str = 'products',
                     filters: Optional[Dict] = None, size: int = 10) -> List[Dict]:
        """Perform hybrid search combining multiple search methods"""
        if weights is None:
            # Default equal weights
            num_methods = 1 + len(query_vectors) + (1 if sparse_vector else 0)
            default_weight = 1.0 / num_methods
            weights = {
                'keyword': default_weight,
                'vector': default_weight,
                'sparse': default_weight if sparse_vector else 0
            }
        
        all_results = {}
        
        # Keyword search
        if weights.get('keyword', 0) > 0:
            keyword_results = self.keyword_search(query, index_name, filters, size * 3)
            for rank, hit in enumerate(keyword_results):
                doc_id = hit['_id']
                if doc_id not in all_results:
                    all_results[doc_id] = {
                        'hit': hit,
                        'scores': {}
                    }
                all_results[doc_id]['scores']['keyword'] = 1.0 / (rank + 1)
        
        # Vector searches
        for vector_field, query_vector in query_vectors.items():
            if weights.get('vector', 0) > 0:
                vector_results = self.vector_search(
                    query_vector, index_name, vector_field, filters, size * 3
                )
                for rank, hit in enumerate(vector_results):
                    doc_id = hit['_id']
                    if doc_id not in all_results:
                        all_results[doc_id] = {
                            'hit': hit,
                            'scores': {}
                        }
                    all_results[doc_id]['scores'][f'vector_{vector_field}'] = 1.0 / (rank + 1)
        
        # Sparse search
        if sparse_vector and weights.get('sparse', 0) > 0:
            sparse_results = self.sparse_search(
                sparse_vector, index_name, filters=filters, size=size * 3
            )
            for rank, hit in enumerate(sparse_results):
                doc_id = hit['_id']
                if doc_id not in all_results:
                    all_results[doc_id] = {
                        'hit': hit,
                        'scores': {}
                    }
                all_results[doc_id]['scores']['sparse'] = 1.0 / (rank + 1)
        
        # Calculate final scores using reciprocal rank fusion
        final_results = []
        for doc_id, data in all_results.items():
            final_score = 0
            for method, score in data['scores'].items():
                weight_key = 'keyword' if 'keyword' in method else 'vector' if 'vector' in method else 'sparse'
                final_score += weights.get(weight_key, 0) * score
            
            data['hit']['_score'] = final_score
            final_results.append(data['hit'])
        
        # Sort by final score and return top k
        final_results.sort(key=lambda x: x['_score'], reverse=True)
        return final_results[:size]
    
    def rerank_results(self, query: str, results: List[Dict], 
                      reranker_type: str = 'cross_encoder') -> List[Dict]:
        """Rerank search results using various methods"""
        if reranker_type == 'none' or not results:
            return results
        
        # This is a placeholder - in production, you'd integrate with actual reranking models
        # For now, we'll just return the results as-is
        return results
    
    def colbert_search(self, query_tokens: List[str], query_token_vectors: List[List[float]],
                      product_ids: Optional[List[str]] = None, size: int = 10) -> List[Dict]:
        """Perform ColBERT-style token-level search"""
        with self.engine.connect() as conn:
            final_scores = {}
            
            # Get all unique product IDs if not provided
            if not product_ids:
                result = conn.execute(text("SELECT DISTINCT product_id FROM token_embeddings"))
                product_ids = [row[0] for row in result]
            
            # For each product, calculate MaxSim score
            for product_id in product_ids:
                # Get all token embeddings for this product
                result = conn.execute(
                    text("SELECT token, embedding FROM token_embeddings WHERE product_id = :pid"),
                    {'pid': product_id}
                )
                
                doc_tokens = []
                doc_embeddings = []
                for row in result:
                    doc_tokens.append(row[0])
                    # Parse the vector string back to list
                    embedding = json.loads(row[1].replace('[', '').replace(']', '').replace(' ', '').split(','))
                    doc_embeddings.append(embedding)
                
                if not doc_embeddings:
                    continue
                
                # Calculate MaxSim score
                total_score = 0
                for query_embedding in query_token_vectors:
                    max_sim = -1
                    for doc_embedding in doc_embeddings:
                        # Cosine similarity
                        sim = np.dot(query_embedding, doc_embedding) / (
                            np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
                        )
                        max_sim = max(max_sim, sim)
                    total_score += max_sim
                
                final_scores[product_id] = total_score
            
            # Get top products and their details
            top_products = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[:size]
            
            # Fetch product details
            product_ids_str = ','.join([f"'{pid}'" for pid, _ in top_products])
            result = conn.execute(
                text(f"SELECT * FROM products WHERE product_id IN ({product_ids_str})")
            )
            
            products_dict = {row.product_id: row for row in result}
            
            # Format results
            hits = []
            for product_id, score in top_products:
                if product_id in products_dict:
                    row = products_dict[product_id]
                    hit = {
                        '_id': str(row.id),
                        '_score': float(score),
                        '_source': dict(row._mapping),
                        'total_score': float(score)
                    }
                    # Remove internal fields
                    hit['_source'].pop('id', None)
                    hits.append(hit)
            
            return hits

# Singleton instance
pg_search = PGVectorSearch()
