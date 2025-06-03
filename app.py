# app.py

import streamlit as st # type: ignore
from models.embedder import get_embedding
from models.search import SearchEngine

# ✅ Must be first Streamlit command
st.set_page_config(page_title="E-Commerce Product Search", layout="wide")

# ✅ Load search engine
@st.cache_resource
def load_search_engine():
    return SearchEngine(index_path="faiss_index/index.bin", data_path="data/product.csv")

search_engine = load_search_engine()

# ✅ Streamlit layout
st.title("🛒 E-Commerce Product Search with FAISS")

# ✅ Input search
query = st.text_input("🔍 Enter your search query:")

if query:
    with st.spinner("Generating embedding and searching..."):
        embedding = get_embedding(query)
        results = search_engine.search(embedding, k=5)

        if results.empty:
            st.warning("❌ No matching product found. Try another query.")
        else:
            st.success("✅ Search completed!")
            st.subheader("📦 Top Matching Products")

            for _, row in results.iterrows():
                with st.container():
                    st.markdown(f"### {row['product_name']}")
                    st.markdown(f"**Class:** {row['product_class']}")

                    description = str(row.get("product_description", "")).strip()
                    if not description or description.lower() == "nan":
                        st.markdown("**Description:** No description available.")
                    elif len(description) > 300:
                        st.markdown(f"**Description:** {description[:300]}...")
                    else:
                        st.markdown(f"**Description:** {description}")

                    similarity = row.get('similarity', None)
                    if similarity is not None:
                        st.caption(f"Similarity score: {similarity:.2f}")
                    else:
                        st.caption("Similarity score: N/A")

                    st.markdown("---")
