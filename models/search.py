# models/search.py

import faiss # type: ignore
import numpy as np
import pandas as pd
from models.embedder import get_embedding

class SearchEngine:
    def __init__(self, index_path: str, data_path: str):
        self.index = faiss.read_index(index_path)
        self.data = pd.read_csv(data_path)
    
    def search(self, query_embedding, k=5, threshold=0.75):
        if query_embedding is None:
            return pd.DataFrame()

        query_embedding = np.array([query_embedding]).astype("float32")
        distances, indices = self.index.search(query_embedding, k)

        results = []
        for i, dist in zip(indices[0], distances[0]):
            if i == -1:
                continue
            similarity = 1 - dist  # Assuming cosine similarity
            if similarity >= threshold:
                row = self.data.iloc[i].copy()
                row["distance"] = dist
                row["similarity"] = similarity
                results.append(row)

        return pd.DataFrame(results)
