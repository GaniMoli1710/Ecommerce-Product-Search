from together import Together # type: ignore

client = Together(api_key="tgp_v1_sO5cUnCToi-_tMR_eUvjCT37FVzdt7Ql0qYiaw3PgO0")

def get_embedding(text):
    response = client.embeddings.create(
        model="togethercomputer/m2-bert-80M-32k-retrieval",
        input=[text]
    )
    return response.data[0].embedding if response.data else None

