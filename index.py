import os
import pandas as pd
import numpy as np
import faiss # type: ignore
from models.embedder import get_embedding

# Set path to your CSV
CSV_PATH = "data/product.csv"
EMBEDDING_DIM = 1024  # Adjust based on model used
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"  # Example

# Load and inspect data
df = pd.read_csv(CSV_PATH)
print("🚀 Generating embeddings...")
print("Columns in CSV:", df.columns.tolist())

# Combine text for embedding
texts = []
embeddings = []
product_ids = []

for index, row in df.iterrows():
    # Safely get and clean text fields
    product_name = str(row['product_name']) if pd.notna(row['product_name']) else ""
    product_description = str(row['product_description']) if pd.notna(row['product_description']) else ""
    text = f"{row['product_name']} {row['product_description']}".strip()
    if not text or text == "nan nan":
          print(f"⚠️ Skipping row {index}: Empty text")
          continue

    # Get embedding
    try:
        embedding = get_embedding(text)
    except Exception as e:
        print(f"❌ Error on row {index}: {e}")
        continue

    if embedding is None or len(embedding) == 0:
        print(f"⚠️ Skipping row {index}: Empty embedding")
        continue

    embeddings.append(embedding)
    texts.append(text)
    product_ids.append(row['product_id'])

print(f"✅ Total valid embeddings: {len(embeddings)}")

# Check if embeddings are valid
if not embeddings:
    raise Exception("❌ No valid embeddings generated. Check embedder or input text.")

# Convert embeddings to numpy array
embedding_matrix = np.array(embeddings, dtype=np.float32)
print(f"📐 Embedding matrix shape: {embedding_matrix.shape}")

embedding_dim = embedding_matrix.shape[1]
index = faiss.IndexFlatL2(embedding_dim)
index.add(embedding_matrix)

# 💾 Save index
faiss.write_index(index, "faiss_index/index.bin")
print(f"✅ FAISS index saved to index.bin with {len(embeddings)} vectors")