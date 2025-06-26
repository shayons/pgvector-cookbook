"""
Core search functionality using pgvector with Aurora PostgreSQL
"""
import os
import json
import numpy as np
from typing import List, Dict, Any, Optional
import streamlit as st
from pgvector_db_setup import get_db_connection
from utilities.invoke_models import invoke_model, invoke_model_mm
import logging

logger = logging.getLogger(__name__)

class PgVectorSearch:
    """Main search class for pgvector operations"""
    
    def __init__(self):
        self.conn = None
        self.connect()
    
    def connect(self):
        """Establish database connection"""
        self.conn = get_db_connection()
        if not self.conn:
            st.error("Failed to connect to database")
    
    def ensure_connection(self):
        """Ensure database connection is active"""
        if not self.conn or self.conn.closed:
            self.connect()
    
    def vector_search(self, query_text: str, table: str = "products", 
                     limit: int = 10, similarity_threshold: float = 0.7) -> List[Dict]:
        """Perform vector similarity search"""
        self.ensure_connection()
        if not self.conn:
            return []
        
        try:
            # Generate embedding for query
            query_embedding = invoke_model(query_text)
            if not query_embedding:
                return []
            
            with self.conn.cursor() as cur:
                # Vector similarity search with cosine distance
                sql = f"""
                    SELECT id, title, description, price, category, style, color, 
                           gender_affinity, image_url, metadata,
                           1 - (embedding <=> %s) as similarity_score
                    FROM {table}
                    WHERE 1 - (embedding <=> %s) > %s
                    ORDER BY embedding <=> %s
                    LIMIT %s;
                """
                
                cur.execute(sql, (query_embedding, query_embedding, 
                                similarity_threshold, query_embedding, limit))
                results = cur.fetchall()
                
                return [dict(row) for row in results]
                
        except Exception as e:
            logger.error(f"Vector search error: {e}")
            st.error(f"Search error: {e}")
            return []
    
    def multimodal_search(self, query_text: str = "", query_image: str = "", 
                         table: str = "products", limit: int = 10) -> List[Dict]:
        """Perform multimodal search with text and/or image"""
        self.ensure_connection()
        if not self.conn:
            return []
        
        try:
            # Generate multimodal embedding
            query_embedding = invoke_model_mm(query_text, query_image if query_image else 'none')
            if not query_embedding:
                return []
            
            with self.conn.cursor() as cur:
                sql = f"""
                    SELECT id, title, description, price, category, style, color,
                           gender_affinity, image_url, metadata,
                           1 - (multimodal_embedding <=> %s) as similarity_score
                    FROM {table}
                    WHERE multimodal_embedding IS NOT NULL
                    ORDER BY multimodal_embedding <=> %s
                    LIMIT %s;
                """
                
                cur.execute(sql, (query_embedding, query_embedding, limit))
                results = cur.fetchall()
                
                return [dict(row) for row in results]
                
        except Exception as e:
            logger.error(f"Multimodal search error: {e}")
            st.error(f"Multimodal search error: {e}")
            return []
    
    def keyword_search(self, query_text: str, table: str = "products", 
                      limit: int = 10) -> List[Dict]:
        """Perform full-text search"""
        self.ensure_connection()
        if not self.conn:
            return []
        
        try:
            with self.conn.cursor() as cur:
                # Full-text search using PostgreSQL's text search
                sql = f"""
                    SELECT id, title, description, price, category, style, color,
                           gender_affinity, image_url, metadata,
                           ts_rank_cd(to_tsvector('english', title || ' ' || description), 
                                     plainto_tsquery('english', %s)) as rank
                    FROM {table}
                    WHERE to_tsvector('english', title || ' ' || description) @@ 
                          plainto_tsquery('english', %s)
                    ORDER BY rank DESC
                    LIMIT %s;
                """
                
                cur.execute(sql, (query_text, query_text, limit))
                results = cur.fetchall()
                
                return [dict(row) for row in results]
                
        except Exception as e:
            logger.error(f"Keyword search error: {e}")
            st.error(f"Keyword search error: {e}")
            return []
    
    def hybrid_search(self, query_text: str, weights: Dict[str, float] = None,
                     table: str = "products", limit: int = 10) -> List[Dict]:
        """Perform hybrid search combining multiple search methods"""
        if weights is None:
            weights = {"vector": 0.6, "keyword": 0.4}
        
        # Perform individual searches
        vector_results = self.vector_search(query_text, table, limit * 2)
        keyword_results = self.keyword_search(query_text, table, limit * 2)
        
        # Combine and rerank results
        combined_results = self._combine_search_results(
            vector_results, keyword_results, weights, limit
        )
        
        return combined_results
    
    def _combine_search_results(self, vector_results: List[Dict], 
                               keyword_results: List[Dict], 
                               weights: Dict[str, float], limit: int) -> List[Dict]:
        """Combine multiple search results with weighted scoring"""
        # Create a mapping of all unique items
        items = {}
        
        # Add vector search results
        for i, item in enumerate(vector_results):
            item_id = item['id']
            vector_score = item.get('similarity_score', 0)
            items[item_id] = {
                **item,
                'vector_score': vector_score,
                'vector_rank': i + 1,
                'keyword_score': 0,
                'keyword_rank': len(vector_results) + 1
            }
        
        # Add keyword search results
        for i, item in enumerate(keyword_results):
            item_id = item['id']
            keyword_score = item.get('rank', 0)
            
            if item_id in items:
                items[item_id]['keyword_score'] = keyword_score
                items[item_id]['keyword_rank'] = i + 1
            else:
                items[item_id] = {
                    **item,
                    'vector_score': 0,
                    'vector_rank': len(keyword_results) + 1,
                    'keyword_score': keyword_score,
                    'keyword_rank': i + 1
                }
        
        # Calculate combined scores
        for item_id, item in items.items():
            # Normalize scores
            vector_norm = 1 / item['vector_rank'] if item['vector_rank'] > 0 else 0
            keyword_norm = 1 / item['keyword_rank'] if item['keyword_rank'] > 0 else 0
            
            # Weighted combination
            combined_score = (
                weights.get('vector', 0.6) * vector_norm +
                weights.get('keyword', 0.4) * keyword_norm
            )
            
            item['combined_score'] = combined_score
        
        # Sort by combined score and return top results
        sorted_results = sorted(items.values(), 
                              key=lambda x: x['combined_score'], 
                              reverse=True)
        
        return sorted_results[:limit]
    
    def filtered_search(self, query_text: str, filters: Dict[str, Any], 
                       search_type: str = "vector", table: str = "products", 
                       limit: int = 10) -> List[Dict]:
        """Perform search with metadata filters"""
        self.ensure_connection()
        if not self.conn:
            return []
        
        try:
            # Build filter conditions
            filter_conditions = []
            filter_params = []
            
            for key, value in filters.items():
                if value is not None:
                    if key == 'price_range':
                        filter_conditions.append("price BETWEEN %s AND %s")
                        filter_params.extend([value[0], value[1]])
                    elif key == 'categories' and isinstance(value, list):
                        filter_conditions.append(f"category = ANY(%s)")
                        filter_params.append(value)
                    else:
                        filter_conditions.append(f"{key} = %s")
                        filter_params.append(value)
            
            filter_clause = ""
            if filter_conditions:
                filter_clause = " AND " + " AND ".join(filter_conditions)
            
            # Generate query embedding for vector search
            if search_type == "vector":
                query_embedding = invoke_model(query_text)
                if not query_embedding:
                    return []
                
                with self.conn.cursor() as cur:
                    sql = f"""
                        SELECT id, title, description, price, category, style, color,
                               gender_affinity, image_url, metadata,
                               1 - (embedding <=> %s) as similarity_score
                        FROM {table}
                        WHERE embedding IS NOT NULL {filter_clause}
                        ORDER BY embedding <=> %s
                        LIMIT %s;
                    """
                    
                    params = [query_embedding] + filter_params + [query_embedding, limit]
                    cur.execute(sql, params)
                    results = cur.fetchall()
                    
                    return [dict(row) for row in results]
            
            elif search_type == "keyword":
                with self.conn.cursor() as cur:
                    sql = f"""
                        SELECT id, title, description, price, category, style, color,
                               gender_affinity, image_url, metadata,
                               ts_rank_cd(to_tsvector('english', title || ' ' || description), 
                                         plainto_tsquery('english', %s)) as rank
                        FROM {table}
                        WHERE to_tsvector('english', title || ' ' || description) @@ 
                              plainto_tsquery('english', %s) {filter_clause}
                        ORDER BY rank DESC
                        LIMIT %s;
                    """
                    
                    params = [query_text, query_text] + filter_params + [limit]
                    cur.execute(sql, params)
                    results = cur.fetchall()
                    
                    return [dict(row) for row in results]
            
        except Exception as e:
            logger.error(f"Filtered search error: {e}")
            st.error(f"Filtered search error: {e}")
            return []
    
    def get_similar_items(self, item_id: int, table: str = "products", 
                         limit: int = 5) -> List[Dict]:
        """Find similar items to a given item"""
        self.ensure_connection()
        if not self.conn:
            return []
        
        try:
            with self.conn.cursor() as cur:
                # Get the embedding of the reference item
                cur.execute(f"SELECT embedding FROM {table} WHERE id = %s", (item_id,))
                result = cur.fetchone()
                
                if not result or not result['embedding']:
                    return []
                
                reference_embedding = result['embedding']
                
                # Find similar items
                sql = f"""
                    SELECT id, title, description, price, category, style, color,
                           gender_affinity, image_url, metadata,
                           1 - (embedding <=> %s) as similarity_score
                    FROM {table}
                    WHERE id != %s AND embedding IS NOT NULL
                    ORDER BY embedding <=> %s
                    LIMIT %s;
                """
                
                cur.execute(sql, (reference_embedding, item_id, reference_embedding, limit))
                results = cur.fetchall()
                
                return [dict(row) for row in results]
                
        except Exception as e:
            logger.error(f"Similar items search error: {e}")
            st.error(f"Similar items search error: {e}")
            return []
    
    def close(self):
        """Close database connection"""
        if self.conn and not self.conn.closed:
            self.conn.close()

# Global search instance
search_engine = PgVectorSearch()