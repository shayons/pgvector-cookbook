import json
import os
import glob
import base64
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import streamlit as st
import boto3
from typing import List, Dict, Optional
from pgvector_db_setup import get_db_connection
from sqlalchemy import text
import utilities.invoke_models as invoke_models
from colpali_engine.interpretability import (
    get_similarity_maps_from_embeddings,
    plot_all_similarity_maps,
    plot_similarity_map,
)

# SageMaker endpoint configuration
COLPALI_ENDPOINT = "colpali-endpoint"
REGION = "us-east-1"

# Initialize SageMaker runtime
runtime = boto3.client(
    "sagemaker-runtime",
    aws_access_key_id=st.secrets['user_access_key'],
    aws_secret_access_key=st.secrets['user_secret_key'],
    region_name=REGION
)

class ColPaliPgvector:
    """ColPali implementation using pgvector for storage and search"""
    
    def __init__(self):
        self.engine = get_db_connection()
        self.parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
    def process_document(self, image_path: str, doc_id: str):
        """Process a document image and store embeddings in pgvector"""
        
        # Read and encode image
        with open(image_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("utf-8")
        
        # Get embeddings from ColPali endpoint
        payload = {"images": [img_b64]}
        
        response = runtime.invoke_endpoint(
            EndpointName=COLPALI_ENDPOINT,
            ContentType="application/json",
            Body=json.dumps(payload)
        )
        
        result = json.loads(response["Body"].read().decode())
        
        # Store in pgvector
        with self.engine.connect() as conn:
            # Store document info
            conn.execute(
                text("""
                    INSERT INTO colpali_documents (doc_id, image_path, page_embeddings)
                    VALUES (:doc_id, :image_path, :embeddings)
                    ON CONFLICT (doc_id) DO UPDATE SET
                        page_embeddings = EXCLUDED.page_embeddings
                """),
                {
                    "doc_id": doc_id,
                    "image_path": image_path,
                    "embeddings": json.dumps({
                        "embeddings": result["image_embeddings"],
                        "mask": result["image_mask"],
                        "patch_shape": result["patch_shape"]
                    })
                }
            )
            
            # Store individual patch embeddings for efficient search
            patch_embeddings = result["image_embeddings"][0]  # First image
            
            for idx, embedding in enumerate(patch_embeddings):
                conn.execute(
                    text("""
                        INSERT INTO rag_documents (
                            doc_id, index_name, processed_element,
                            raw_element_type, src_doc, 
                            processed_element_embedding
                        ) VALUES (
                            :doc_id, :index_name, :processed_element,
                            :type, :src_doc, :embedding
                        )
                    """),
                    {
                        "doc_id": f"{doc_id}_patch_{idx}",
                        "index_name": "colpali_patches",
                        "processed_element": f"Visual patch {idx}",
                        "type": "colpali_patch",
                        "src_doc": image_path,
                        "embedding": f"[{','.join(map(str, embedding))}]"
                    }
                )
            
            conn.commit()
    
    def search(self, query: str, top_k: int = 20) -> List[Dict]:
        """Search using ColPali multi-vector approach"""
        
        # Get query embeddings
        payload = {"queries": [query]}
        response = runtime.invoke_endpoint(
            EndpointName=COLPALI_ENDPOINT,
            ContentType="application/json",
            Body=json.dumps(payload)
        )
        
        result = json.loads(response["Body"].read().decode())
        query_embeddings = result['query_embeddings'][0]
        query_tokens = result['query_tokens'][0]['tokens']
        
        # Store for visualization
        st.session_state.query_token_vectors = query_embeddings
        st.session_state.query_tokens = result['query_tokens']
        
        # Search using average pooled query vector
        avg_query_vector = np.array(query_embeddings).mean(axis=0)
        
        with self.engine.connect() as conn:
            # Get candidate documents
            vector_str = f"[{','.join(map(str, avg_query_vector))}]"
            
            candidates = conn.execute(
                text("""
                    SELECT DISTINCT doc_id, src_doc
                    FROM rag_documents
                    WHERE index_name = 'colpali_patches'
                    ORDER BY processed_element_embedding <=> :query_vec::vector
                    LIMIT :limit
                """),
                {
                    "query_vec": vector_str,
                    "limit": top_k * 10  # Get more candidates
                }
            )
            
            # Calculate MaxSim scores for each document
            doc_scores = []
            
            for row in candidates:
                base_doc_id = row.doc_id.replace(/_patch_\d+$/, '')
                
                # Get all patches for this document
                patches = conn.execute(
                    text("""
                        SELECT processed_element_embedding
                        FROM rag_documents
                        WHERE doc_id LIKE :pattern
                        AND index_name = 'colpali_patches'
                        ORDER BY doc_id
                    """),
                    {"pattern": f"{base_doc_id}_patch_%"}
                )
                
                # Calculate MaxSim score
                total_score = 0
                for query_emb in query_embeddings:
                    max_sim = -1
                    
                    for patch in patches:
                        # Parse embedding
                        patch_emb = np.array(
                            json.loads(patch[0].replace('[', '').replace(']', '').replace(' ', '').split(','))
                        )
                        
                        # Cosine similarity
                        sim = np.dot(query_emb, patch_emb) / (
                            np.linalg.norm(query_emb) * np.linalg.norm(patch_emb)
                        )
                        max_sim = max(max_sim, sim)
                    
                    total_score += max_sim
                
                doc_scores.append({
                    "doc_id": base_doc_id,
                    "image_path": row.src_doc,
                    "score": total_score,
                    "total_score": total_score
                })
            
            # Sort by score
            doc_scores.sort(key=lambda x: x['score'], reverse=True)
            
            # Get top results
            top_results = doc_scores[:top_k]
            
            # Store top result for visualization
            if top_results:
                st.session_state.top_img = top_results[0]['image_path']
            
            return top_results
    
    def generate_similarity_maps(self, image_path: str, query_embeddings: List, 
                                query_tokens: List[str]) -> List[Dict]:
        """Generate similarity maps for visualization"""
        
        # Check if maps already exist
        img_name = os.path.basename(image_path)
        search_pattern = os.path.join(
            self.parent_dir, 
            f"similarity_maps/similarity_map_{img_name}_token_*"
        )
        
        matching_files = glob.glob(search_pattern)
        
        if matching_files:
            return [{'file': f} for f in matching_files]
        
        # Generate new maps
        with open(image_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("utf-8")
        
        # Get image embeddings
        payload = {"images": [img_b64]}
        response = runtime.invoke_endpoint(
            EndpointName=COLPALI_ENDPOINT,
            ContentType="application/json",
            Body=json.dumps(payload)
        )
        
        result = json.loads(response["Body"].read().decode())
        
        # Convert to tensors
        image_embeddings = torch.tensor(result["image_embeddings"][0])
        query_embeddings_tensor = torch.tensor(query_embeddings).unsqueeze(0)
        image_mask = torch.tensor(result["image_mask"][0]).bool().unsqueeze(0)
        
        if image_embeddings.dim() == 2:
            image_embeddings = image_embeddings.unsqueeze(0)
        
        # Get similarity maps
        n_patches = (result["patch_shape"]['height'], result["patch_shape"]['width'])
        
        batched_similarity_maps = get_similarity_maps_from_embeddings(
            image_embeddings=image_embeddings,
            query_embeddings=query_embeddings_tensor,
            n_patches=n_patches,
            image_mask=image_mask
        )
        
        similarity_maps = batched_similarity_maps[0]
        
        # Create output directory
        map_dir = os.path.join(self.parent_dir, "similarity_maps")
        os.makedirs(map_dir, exist_ok=True)
        
        # Generate plots
        image = Image.open(image_path)
        plots = plot_all_similarity_maps(
            image=image,
            query_tokens=query_tokens,
            similarity_maps=similarity_maps,
            figsize=(8, 8),
            show_colorbar=False,
            add_title=True,
        )
        
        map_images = []
        for idx, (fig, ax) in enumerate(plots):
            if idx < 3:  # Skip special tokens
                continue
                
            savepath = os.path.join(
                map_dir,
                f"similarity_map_{img_name}_token_{idx}_{query_tokens[idx]}.png"
            )
            fig.savefig(savepath, bbox_inches="tight")
            map_images.append({'file': savepath})
            plt.close(fig)
        
        return map_images
    
    def colpali_search_rerank(self, query: str) -> Dict:
        """Main search function with reranking"""
        
        # Check if showing similarity maps
        if st.session_state.get('show_columns', False):
            st.session_state.maxSimImages = self.generate_similarity_maps(
                st.session_state.top_img,
                st.session_state.query_token_vectors,
                st.session_state.query_tokens[0]['tokens']
            )
            st.session_state.show_columns = False
            
            return {
                'text': st.session_state.answers_[0]['answer'],
                'source': st.session_state.answers_[0]['source'],
                'image': st.session_state.maxSimImages,
                'table': []
            }
        
        # Perform search
        results = self.search(query)
        
        if not results:
            return {
                'text': "No relevant documents found.",
                'source': [],
                'image': [],
                'table': []
            }
        
        # Generate answer using the top result
        top_result = results[0]
        image_path = top_result['image_path']
        
        # Use Nova or other model to generate answer
        answer = self.generate_answer(image_path, query)
        
        return {
            'text': answer,
            'source': image_path,
            'image': [{'file': image_path}],
            'table': []
        }
    
    def generate_answer(self, image_path: str, query: str) -> str:
        """Generate answer using multimodal LLM"""
        
        # Read image
        with open(image_path, "rb") as f:
            image_base64 = base64.b64encode(f.read()).decode("utf-8")
        
        # Call Nova or appropriate model
        system_message = (
            "Given an image of a PDF page, answer the question accurately. "
            "If you don't find the answer in the page, say 'I don't know'."
        )
        
        prompt = f"""Looking at this document image, please answer: {query}
        
        Be specific and cite any relevant information you see in the image."""
        
        # This would call your actual multimodal LLM
        # For now, using a placeholder
        answer = f"Based on the document, here's what I found regarding '{query}'..."
        
        return answer

# Create singleton instance
colpali_search = ColPaliPgvector()

# Export the main function
def colpali_search_rerank(query: str) -> Dict:
    """Main entry point for ColPali search"""
    return colpali_search.colpali_search_rerank(query)

def process_doc(request: Dict):
    """Process a document for ColPali indexing"""
    doc_path = request.get('key', '')
    doc_id = os.path.splitext(os.path.basename(doc_path))[0]
    
    # Process with ColPali
    colpali_search.process_document(doc_path, doc_id)
    
    return {"status": "success", "doc_id": doc_id}
