"""
Database setup and connection utilities for Aurora PostgreSQL with pgvector
"""
import os
import psycopg2
from psycopg2.extras import RealDictCursor
import streamlit as st
import os
import os
from urllib.parse import urlparse
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_db_connection():
    """Get database connection to Aurora PostgreSQL using environment variables"""
    try:
        # Check for DATABASE_URL first (recommended for HF Spaces)
        if 'DATABASE_URL' in os.environ:
            database_url = os.environ['DATABASE_URL']
            logger.info("Connecting to database using DATABASE_URL")
            return psycopg2.connect(database_url, cursor_factory=RealDictCursor)
        
        # Fallback to individual environment variables
        connection_params = {
            'host': os.environ.get('DB_HOST'),
            'port': os.environ.get('DB_PORT', '5432'),
            'database': os.environ.get('DB_NAME'),
            'user': os.environ.get('DB_USER'),
            'password': os.environ.get('DB_PASSWORD'),
        }
        
        # Remove None values
        connection_params = {k: v for k, v in connection_params.items() if v is not None}
        
        if not all([connection_params.get('host'), connection_params.get('database'), 
                   connection_params.get('user'), connection_params.get('password')]):
            raise ValueError("Missing required database connection parameters")
        
        logger.info(f"Connecting to database at {connection_params['host']}")
        return psycopg2.connect(cursor_factory=RealDictCursor, **connection_params)
        
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        if 'st' in globals():
            st.error(f"Database connection failed: {e}")
        return None

def init_pgvector_extension():
    """Initialize pgvector extension in the database"""
    conn = get_db_connection()
    if not conn:
        return False
    
    try:
        with conn.cursor() as cur:
            # Enable pgvector extension
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            
            # Check pgvector version
            cur.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector';")
            version = cur.fetchone()
            if version:
                logger.info(f"pgvector version: {version['extversion']}")
            
            conn.commit()
            logger.info("pgvector extension initialized successfully")
            return True
            
    except Exception as e:
        logger.error(f"Failed to initialize pgvector extension: {e}")
        return False
    finally:
        conn.close()

def create_tables():
    """Create necessary tables for the application"""
    conn = get_db_connection()
    if not conn:
        return False
    
    try:
        with conn.cursor() as cur:
            # Create documents table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id SERIAL PRIMARY KEY,
                    title TEXT,
                    content TEXT,
                    embedding vector(1536),
                    multimodal_embedding vector(1024),
                    metadata JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            # Create products table (for retail search)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS products (
                    id SERIAL PRIMARY KEY,
                    title TEXT,
                    description TEXT,
                    price DECIMAL(10,2),
                    category TEXT,
                    style TEXT,
                    color TEXT,
                    gender_affinity TEXT,
                    image_url TEXT,
                    embedding vector(1536),
                    multimodal_embedding vector(1024),
                    sparse_embedding TEXT,
                    metadata JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            # Create colbert_tokens table (for multi-vector search)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS colbert_tokens (
                    id SERIAL PRIMARY KEY,
                    document_id INTEGER REFERENCES documents(id),
                    token_index INTEGER,
                    token TEXT,
                    embedding vector(384),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            # Create user behavior table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS user_behavior (
                    id SERIAL PRIMARY KEY,
                    session_id TEXT,
                    user_id TEXT,
                    query TEXT,
                    search_type TEXT,
                    results_count INTEGER,
                    click_position INTEGER,
                    item_id TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            conn.commit()
            logger.info("Tables created successfully")
            
            # Create indexes
            create_indexes(cur)
            conn.commit()
            logger.info("Indexes created successfully")
            
            return True
            
    except Exception as e:
        logger.error(f"Failed to create tables: {e}")
        return False
    finally:
        conn.close()

def create_indexes(cur):
    """Create vector indexes for optimal performance"""
    
    # HNSW indexes for main embeddings
    indexes = [
        "CREATE INDEX IF NOT EXISTS documents_embedding_idx ON documents USING hnsw (embedding vector_cosine_ops);",
        "CREATE INDEX IF NOT EXISTS documents_multimodal_embedding_idx ON documents USING hnsw (multimodal_embedding vector_cosine_ops);",
        "CREATE INDEX IF NOT EXISTS products_embedding_idx ON products USING hnsw (embedding vector_cosine_ops);",
        "CREATE INDEX IF NOT EXISTS products_multimodal_embedding_idx ON products USING hnsw (multimodal_embedding vector_cosine_ops);",
        "CREATE INDEX IF NOT EXISTS colbert_tokens_embedding_idx ON colbert_tokens USING hnsw (embedding vector_cosine_ops);",
        
        # Regular indexes
        "CREATE INDEX IF NOT EXISTS products_category_idx ON products(category);",
        "CREATE INDEX IF NOT EXISTS products_price_idx ON products(price);",
        "CREATE INDEX IF NOT EXISTS products_gender_idx ON products(gender_affinity);",
        "CREATE INDEX IF NOT EXISTS user_behavior_session_idx ON user_behavior(session_id);",
        "CREATE INDEX IF NOT EXISTS colbert_tokens_doc_idx ON colbert_tokens(document_id);",
    ]
    
    for index_sql in indexes:
        try:
            cur.execute(index_sql)
            logger.info(f"Created index: {index_sql.split(' ')[5]}")
        except Exception as e:
            logger.warning(f"Index creation warning: {e}")

def test_connection():
    """Test database connection and pgvector functionality"""
    conn = get_db_connection()
    if not conn:
        return False
    
    try:
        with conn.cursor() as cur:
            # Test basic query
            cur.execute("SELECT version();")
            version = cur.fetchone()
            logger.info(f"PostgreSQL version: {version['version']}")
            
            # Test pgvector
            cur.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector';")
            vector_version = cur.fetchone()
            if vector_version:
                logger.info(f"pgvector version: {vector_version['extversion']}")
            else:
                logger.warning("pgvector extension not found")
                return False
            
            # Test vector operations
            cur.execute("SELECT '[1,2,3]'::vector <-> '[1,2,4]'::vector as distance;")
            distance = cur.fetchone()
            logger.info(f"Vector distance test: {distance['distance']}")
            
            return True
            
    except Exception as e:
        logger.error(f"Connection test failed: {e}")
        return False
    finally:
        conn.close()

def get_aws_credentials():
    """Get AWS credentials from environment variables (us-west-2)"""
    return {
        'aws_access_key_id': os.environ.get('AWS_ACCESS_KEY_ID'),
        'aws_secret_access_key': os.environ.get('AWS_SECRET_ACCESS_KEY'),
        'region_name': 'us-west-2'
    }

if __name__ == "__main__":
    # Initialize database when run directly
    logger.info("Initializing database...")
    
    if test_connection():
        logger.info("Database connection successful")
        
        if init_pgvector_extension():
            logger.info("pgvector extension initialized")
            
            if create_tables():
                logger.info("Database setup completed successfully")
            else:
                logger.error("Failed to create tables")
        else:
            logger.error("Failed to initialize pgvector extension")
    else:
        logger.error("Database connection failed")