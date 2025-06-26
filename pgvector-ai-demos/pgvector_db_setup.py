import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import os
from sqlalchemy import create_engine, text
import streamlit as st

# Database configuration
DB_CONFIG = {
    'host': os.getenv('PGHOST', 'localhost'),
    'port': os.getenv('PGPORT', '5432'),
    'database': os.getenv('PGDATABASE', 'opensearch_ai_demo'),
    'user': os.getenv('PGUSER', 'postgres'),
    'password': os.getenv('PGPASSWORD', 'postgres')
}

def create_database():
    """Create the database if it doesn't exist"""
    conn = psycopg2.connect(
        host=DB_CONFIG['host'],
        port=DB_CONFIG['port'],
        user=DB_CONFIG['user'],
        password=DB_CONFIG['password'],
        database='postgres'
    )
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cur = conn.cursor()
    
    # Check if database exists
    cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (DB_CONFIG['database'],))
    exists = cur.fetchone()
    
    if not exists:
        cur.execute(f"CREATE DATABASE {DB_CONFIG['database']}")
        print(f"Database {DB_CONFIG['database']} created.")
    
    cur.close()
    conn.close()

def setup_pgvector():
    """Set up pgvector extension and create necessary tables"""
    engine = create_engine(
        f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
    )
    
    with engine.connect() as conn:
        # Enable pgvector extension
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS pg_trgm"))  # For text search
        conn.commit()
        
        # Create products table for AI Search demo
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS products (
                id SERIAL PRIMARY KEY,
                product_id VARCHAR(255) UNIQUE,
                caption TEXT,
                product_description TEXT,
                category VARCHAR(100),
                price DECIMAL(10, 2),
                gender_affinity VARCHAR(20),
                style VARCHAR(100),
                image_url TEXT,
                current_stock INTEGER DEFAULT 0,
                color VARCHAR(50),
                description_vector vector(384),  -- for all-MiniLM-L6-v2
                description_vector_titan vector(1536),  -- for Titan embeddings
                multimodal_vector vector(1024),  -- for multimodal embeddings
                sparse_vector JSONB,  -- for sparse vectors
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """))
        
        # Create indices for vector similarity search
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_description_vector ON products 
            USING ivfflat (description_vector vector_cosine_ops)
            WITH (lists = 100)
        """))
        
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_description_vector_titan ON products 
            USING ivfflat (description_vector_titan vector_cosine_ops)
            WITH (lists = 100)
        """))
        
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_multimodal_vector ON products 
            USING ivfflat (multimodal_vector vector_cosine_ops)
            WITH (lists = 100)
        """))
        
        # Create text search indices
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_product_description_gin ON products 
            USING gin(to_tsvector('english', product_description))
        """))
        
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_caption_gin ON products 
            USING gin(to_tsvector('english', caption))
        """))
        
        # Create indices for filters
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_category ON products(category)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_gender ON products(gender_affinity)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_price ON products(price)"))
        
        # Create RAG documents table
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS rag_documents (
                id SERIAL PRIMARY KEY,
                doc_id VARCHAR(255) UNIQUE,
                index_name VARCHAR(255),
                processed_element TEXT,
                raw_element TEXT,
                raw_element_type VARCHAR(50),
                src_doc TEXT,
                image VARCHAR(255),
                table_name VARCHAR(255),
                processed_element_embedding vector(1536),
                processed_element_embedding_sparse JSONB,
                multimodal_embedding vector(1024),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """))
        
        # Create indices for RAG documents
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_rag_embedding ON rag_documents 
            USING ivfflat (processed_element_embedding vector_cosine_ops)
            WITH (lists = 100)
        """))
        
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_rag_multimodal ON rag_documents 
            USING ivfflat (multimodal_embedding vector_cosine_ops)
            WITH (lists = 100)
        """))
        
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_rag_text_gin ON rag_documents 
            USING gin(to_tsvector('english', processed_element))
        """))
        
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_rag_index_name ON rag_documents(index_name)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_rag_type ON rag_documents(raw_element_type)"))
        
        # Create token-level embeddings table for ColBERT-style search
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS token_embeddings (
                id SERIAL PRIMARY KEY,
                product_id VARCHAR(255),
                token VARCHAR(100),
                token_index INTEGER,
                embedding vector(384),
                FOREIGN KEY (product_id) REFERENCES products(product_id) ON DELETE CASCADE
            )
        """))
        
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_token_embedding ON token_embeddings 
            USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100)
        """))
        
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_token_product_id ON token_embeddings(product_id)"))
        
        # Create UBI (User Behavior Insights) tables
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS ubi_queries (
                id SERIAL PRIMARY KEY,
                client_id VARCHAR(255),
                query_id VARCHAR(255) UNIQUE,
                application VARCHAR(100),
                query_response_hit_ids TEXT[],
                timestamp TIMESTAMP,
                user_query JSONB,
                query TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """))
        
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS ubi_events (
                id SERIAL PRIMARY KEY,
                action_name VARCHAR(100),
                item_id VARCHAR(255),
                query_id VARCHAR(255),
                session_id VARCHAR(255),
                timestamp TIMESTAMP,
                message_type VARCHAR(50),
                message TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """))
        
        # Create session memory table for agents
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS agent_memory (
                id SERIAL PRIMARY KEY,
                session_id VARCHAR(255),
                agent_id VARCHAR(255),
                memory_data JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """))
        
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_agent_session ON agent_memory(session_id, agent_id)"))
        
        # Create colpali documents table
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS colpali_documents (
                id SERIAL PRIMARY KEY,
                doc_id VARCHAR(255) UNIQUE,
                image_path TEXT,
                page_embeddings JSONB,  -- Store as array of vectors
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """))
        
        conn.commit()
        
    print("Database setup completed successfully!")

def get_db_connection():
    """Get a database connection using connection pooling"""
    if 'db_engine' not in st.session_state:
        st.session_state.db_engine = create_engine(
            f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}",
            pool_size=20,
            max_overflow=0
        )
    return st.session_state.db_engine

if __name__ == "__main__":
    create_database()
    setup_pgvector()
