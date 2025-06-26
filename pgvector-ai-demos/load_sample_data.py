import json
import random
from faker import Faker
from pgvector_db_setup import get_db_connection
from sqlalchemy import text
import utilities.invoke_models as invoke_models
import utilities.mvectors as mvectors

fake = Faker()

# Sample retail data
CATEGORIES = ["apparel", "footwear", "accessories", "electronics", "beauty", 
              "jewelry", "housewares", "books", "sports", "toys"]

COLORS = ["black", "white", "red", "blue", "green", "yellow", "brown", 
          "gray", "pink", "purple", "orange", "navy"]

STYLES = ["casual", "formal", "athletic", "vintage", "modern", "classic", 
          "trendy", "minimalist", "bold", "elegant"]

GENDER_AFFINITY = ["male", "female", "unisex"]

def generate_product_description(category, color, style, gender):
    """Generate a realistic product description"""
    templates = {
        "apparel": [
            f"This {style} {color} {category} is perfect for {gender} customers looking for comfort and style.",
            f"A {style} piece featuring {color} design, ideal for everyday wear.",
            f"Premium quality {color} {category} with {style} aesthetics."
        ],
        "footwear": [
            f"These {style} {color} shoes offer exceptional comfort and durability.",
            f"Step out in style with these {color} {style} shoes designed for {gender}.",
            f"High-quality {color} footwear combining {style} design with functionality."
        ],
        "accessories": [
            f"This {style} {color} accessory complements any outfit perfectly.",
            f"A must-have {color} {style} accessory for the fashion-conscious {gender}.",
            f"Elegant {color} accessory with {style} design elements."
        ],
        "electronics": [
            f"Advanced {color} electronic device with {style} design and cutting-edge features.",
            f"This {style} {color} gadget offers superior performance and reliability.",
            f"State-of-the-art {color} electronics with {style} aesthetics."
        ],
        "beauty": [
            f"Premium {style} beauty product designed for {gender} with {color} packaging.",
            f"This {color} beauty essential features {style} design and natural ingredients.",
            f"Luxurious {style} beauty product in elegant {color} packaging."
        ]
    }
    
    # Get templates for category or use generic
    category_templates = templates.get(category, [
        f"High-quality {color} {category} with {style} design for {gender}.",
        f"This {style} {color} product is perfect for any occasion.",
        f"Premium {color} {category} featuring {style} aesthetics."
    ])
    
    return random.choice(category_templates)

def generate_sample_products(num_products=100):
    """Generate sample retail products"""
    products = []
    
    for i in range(num_products):
        category = random.choice(CATEGORIES)
        color = random.choice(COLORS)
        style = random.choice(STYLES)
        gender = random.choice(GENDER_AFFINITY)
        
        # Generate product details
        product_name = f"{style.capitalize()} {color.capitalize()} {category.capitalize()}"
        caption = f"{product_name} - {fake.catch_phrase()}"
        description = generate_product_description(category, color, style, gender)
        
        product = {
            "product_id": f"PROD_{i+1:04d}",
            "caption": caption,
            "product_description": description,
            "category": category,
            "price": round(random.uniform(10, 500), 2),
            "gender_affinity": gender,
            "style": style,
            "color": color,
            "current_stock": random.randint(0, 100),
            "image_url": f"/images/sample_{category}_{i+1}.jpg"  # Placeholder
        }
        
        products.append(product)
    
    return products

def load_products_to_db(products):
    """Load products into pgvector database"""
    engine = get_db_connection()
    
    with engine.connect() as conn:
        for product in products:
            print(f"Loading product: {product['product_id']}")
            
            # Generate embeddings
            # Text embedding using all-MiniLM-L6-v2
            desc_vector = mvectors.vectorise(product['product_description'], False)
            
            # Titan text embedding
            desc_vector_titan = invoke_models.invoke_model(product['product_description'])
            
            # Multimodal embedding (text only for now)
            mm_vector = invoke_models.invoke_model_mm(product['product_description'], "none")
            
            # Generate sparse vector (placeholder - would use actual sparse model)
            sparse_vector = generate_sparse_vector(product['product_description'])
            
            # Check if product exists
            result = conn.execute(
                text("SELECT 1 FROM products WHERE product_id = :pid"),
                {"pid": product['product_id']}
            )
            
            if not result.fetchone():
                # Insert product
                conn.execute(
                    text("""
                        INSERT INTO products (
                            product_id, caption, product_description, category,
                            price, gender_affinity, style, color, current_stock,
                            image_url, description_vector, description_vector_titan,
                            multimodal_vector, sparse_vector
                        ) VALUES (
                            :product_id, :caption, :product_description, :category,
                            :price, :gender_affinity, :style, :color, :current_stock,
                            :image_url, :desc_vector, :desc_vector_titan,
                            :mm_vector, :sparse_vector
                        )
                    """),
                    {
                        **product,
                        "desc_vector": f"[{','.join(map(str, desc_vector))}]",
                        "desc_vector_titan": f"[{','.join(map(str, desc_vector_titan))}]",
                        "mm_vector": f"[{','.join(map(str, mm_vector))}]",
                        "sparse_vector": json.dumps(sparse_vector)
                    }
                )
                
                # Also insert token embeddings for ColBERT-style search
                if product['product_id'].endswith(('001', '010', '020')):  # Sample subset
                    tokens, token_vectors = mvectors.vectorise(
                        product['product_description'], True
                    )
                    
                    for idx, (token, vector) in enumerate(zip(tokens, token_vectors)):
                        conn.execute(
                            text("""
                                INSERT INTO token_embeddings (
                                    product_id, token, token_index, embedding
                                ) VALUES (
                                    :product_id, :token, :token_index, :embedding
                                )
                            """),
                            {
                                "product_id": product['product_id'],
                                "token": token,
                                "token_index": idx,
                                "embedding": f"[{','.join(map(str, vector))}]"
                            }
                        )
        
        conn.commit()
    
    print(f"Successfully loaded {len(products)} products")

def generate_sparse_vector(text, max_terms=20):
    """Generate a simple sparse vector (placeholder for actual sparse encoding)"""
    # In production, use actual sparse encoding model
    words = text.lower().split()
    word_freq = {}
    
    for word in words:
        if len(word) > 3:  # Skip short words
            word_freq[word] = word_freq.get(word, 0) + 1
    
    # Normalize and get top terms
    max_freq = max(word_freq.values()) if word_freq else 1
    sparse_vector = {
        word: freq / max_freq 
        for word, freq in sorted(
            word_freq.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:max_terms]
    }
    
    return sparse_vector

if __name__ == "__main__":
    print("Generating sample products...")
    products = generate_sample_products(100)
    
    print("Loading products to database...")
    load_products_to_db(products)
    
    print("Sample data loaded successfully!")
