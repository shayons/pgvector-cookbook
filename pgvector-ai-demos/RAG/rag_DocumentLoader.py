import os
import re
import json
import time
import shutil
from unstructured.partition.pdf import partition_pdf
import streamlit as st
import os
from PIL import Image
import base64
import utilities.invoke_models as invoke_models
from utilities.pgvector_search import pg_search
from pgvector_db_setup import get_db_connection
from sqlalchemy import text
import generate_csv_for_tables

parent_dirname = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_docs(inp):
    """Load and process documents into pgvector database"""
    
    print(f"Loading document: {inp}")
    
    data_dir = os.path.join(parent_dirname, "pdfs")
    target_files = [os.path.join(data_dir, inp["key"])]
    
    Image.MAX_IMAGE_PIXELS = 100000000
    width = 2048
    height = 2048
    
    engine = get_db_connection()
    
    for target_file in target_files:
        # Extract tables using Textract
        tables_textract = generate_csv_for_tables.main_(target_file)
        
        # Generate index name from filename
        index_ = re.sub('[^A-Za-z0-9]+', '', 
                       os.path.splitext(os.path.basename(target_file))[0].lower())
        st.session_state.input_index = index_
        
        # Create directories for figures
        figures_dir = os.path.join(parent_dirname, 'figures')
        if not os.path.exists(figures_dir):
            os.makedirs(figures_dir)
        
        image_output_dir = os.path.join(figures_dir, st.session_state.input_index)
        if os.path.exists(image_output_dir):
            shutil.rmtree(image_output_dir)
        os.makedirs(image_output_dir)
        
        print(f"Processing file: {target_file}")
        
        # Partition PDF using unstructured
        table_and_text_elements = partition_pdf(
            filename=target_file,
            extract_images_in_pdf=True,
            infer_table_structure=False,
            chunking_strategy="by_title",
            max_characters=4000,
            new_after_n_chars=3800,
            combine_text_under_n_chars=2000,
            extract_image_block_output_dir=image_output_dir,
        )
        
        tables = []
        texts = []
        
        # Process tables from Textract
        for table_name, table_content in tables_textract.items():
            print(f"Processing table: {table_name}")
            summary = invoke_models.invoke_llm_model(
                f"Summarize this table: {table_content}", False
            )
            tables.append({
                'table_name': table_name,
                'raw': table_content,
                'summary': summary
            })
            time.sleep(4)
        
        # Process text elements
        for element in table_and_text_elements:
            if "CompositeElement" in str(type(element)):
                texts.append(str(element))
        
        # Process images
        image_captions = {}
        for image_file in os.listdir(image_output_dir):
            print(f"Processing image: {image_file}")
            
            photo_full_path = os.path.join(image_output_dir, image_file)
            photo_path_no_ext = os.path.splitext(photo_full_path)[0]
            
            # Verify and resize image
            with Image.open(photo_full_path) as image:
                image.verify()
            
            with Image.open(photo_full_path) as image:
                image.thumbnail((width, height))
                image.save(f"{photo_path_no_ext}-resized.jpg")
            
            # Generate caption
            with open(f"{photo_path_no_ext}-resized.jpg", "rb") as img_file:
                encoded_image = base64.b64encode(img_file.read()).decode("utf8")
            
            caption = invoke_models.generate_image_captions_llm(
                encoded_image, "What's in this image?"
            )
            
            image_captions[image_file] = {
                "caption": caption,
                "encoding": encoded_image
            }
        
        print("Image processing complete")
        
        # Insert documents into pgvector
        with engine.connect() as conn:
            # Process and insert text documents
            for text in texts:
                embedding = invoke_models.invoke_model(text)
                
                doc_id = f"{index_}_text_{hash(text)}"
                processed_element = re.sub(r"[^a-zA-Z0-9]+", ' ', text)
                
                # Check if document exists
                result = conn.execute(
                    text("SELECT 1 FROM rag_documents WHERE doc_id = :doc_id"),
                    {"doc_id": doc_id}
                )
                
                if not result.fetchone():
                    conn.execute(
                        text("""
                            INSERT INTO rag_documents (
                                doc_id, index_name, processed_element, raw_element,
                                raw_element_type, src_doc, processed_element_embedding
                            ) VALUES (
                                :doc_id, :index_name, :processed_element, :raw_element,
                                :raw_element_type, :src_doc, :embedding
                            )
                        """),
                        {
                            "doc_id": doc_id,
                            "index_name": index_,
                            "processed_element": processed_element,
                            "raw_element": processed_element,
                            "raw_element_type": "text",
                            "src_doc": target_file,
                            "embedding": f"[{','.join(map(str, embedding))}]"
                        }
                    )
            
            # Process and insert table documents
            for table in tables:
                embedding = invoke_models.invoke_model(table['summary'])
                
                doc_id = f"{index_}_table_{table['table_name']}"
                processed_element = re.sub(r"[^a-zA-Z0-9]+", ' ', table['summary'])
                raw_element = re.sub(r"[^a-zA-Z0-9]+", ' ', table['raw'])
                
                result = conn.execute(
                    text("SELECT 1 FROM rag_documents WHERE doc_id = :doc_id"),
                    {"doc_id": doc_id}
                )
                
                if not result.fetchone():
                    conn.execute(
                        text("""
                            INSERT INTO rag_documents (
                                doc_id, index_name, processed_element, raw_element,
                                raw_element_type, src_doc, table_name, 
                                processed_element_embedding
                            ) VALUES (
                                :doc_id, :index_name, :processed_element, :raw_element,
                                :raw_element_type, :src_doc, :table_name, :embedding
                            )
                        """),
                        {
                            "doc_id": doc_id,
                            "index_name": index_,
                            "processed_element": processed_element,
                            "raw_element": raw_element,
                            "raw_element_type": "table",
                            "src_doc": target_file,
                            "table_name": table['table_name'],
                            "embedding": f"[{','.join(map(str, embedding))}]"
                        }
                    )
            
            # Process and insert image documents
            for filename, img_data in image_captions.items():
                # Text embedding
                text_embedding = invoke_models.invoke_model(img_data['caption'])
                
                # Multimodal embedding
                mm_embedding = invoke_models.invoke_model_mm(
                    img_data['caption'], img_data['encoding']
                )
                
                doc_id = f"{index_}_image_{filename}"
                processed_element = re.sub(r"[^a-zA-Z0-9]+", ' ', img_data['caption'])
                
                result = conn.execute(
                    text("SELECT 1 FROM rag_documents WHERE doc_id = :doc_id"),
                    {"doc_id": doc_id}
                )
                
                if not result.fetchone():
                    conn.execute(
                        text("""
                            INSERT INTO rag_documents (
                                doc_id, index_name, processed_element, raw_element,
                                raw_element_type, src_doc, image,
                                processed_element_embedding, multimodal_embedding
                            ) VALUES (
                                :doc_id, :index_name, :processed_element, :raw_element,
                                :raw_element_type, :src_doc, :image,
                                :text_embedding, :mm_embedding
                            )
                        """),
                        {
                            "doc_id": doc_id,
                            "index_name": index_,
                            "processed_element": processed_element,
                            "raw_element": processed_element,
                            "raw_element_type": "image",
                            "src_doc": target_file,
                            "image": filename,
                            "text_embedding": f"[{','.join(map(str, text_embedding))}]",
                            "mm_embedding": f"[{','.join(map(str, mm_embedding))}]"
                        }
                    )
            
            conn.commit()
        
        print(f"Successfully loaded document: {target_file}")
