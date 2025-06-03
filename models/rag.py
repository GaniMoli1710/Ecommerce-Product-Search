# rag.py

from together import Together # type: ignore

client = Together(api_key="tgp_v1_sO5cUnCToi-_tMR_eUvjCT37FVzdt7Ql0qYiaw3PgO0")  # or set TOGETHER_API_KEY env variable

def get_embedding(text: str, model: str = "togethercomputer/m2-bert-80M-32k-retrieval"):
    if not text.strip():
        print("❌ Empty input to embedding function.")
    try:
        res = client.embeddings.create(
            input=[text],
            model=model
        )
        if res and "data" in res and len(res["data"]) > 0:
            return res["data"][0]["embedding"]
    except Exception as e:
        print(f"❌ Error getting embedding: {e}")
    return None
