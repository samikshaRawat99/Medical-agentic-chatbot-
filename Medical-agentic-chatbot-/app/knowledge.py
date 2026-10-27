import time
from functools import lru_cache
from pinecone import Pinecone, ServerlessSpec
from fastembed import TextEmbedding
from app.config import settings
from groq import Groq 

@lru_cache()
def clients():
    if not settings.groq_api_key or not settings.pinecone_api_key:
        raise ValueError("API keys for Groq and Pinecone must be set in the environment variables.")
    return Groq(api_key=settings.groq_api_key), Pinecone(api_key=settings.pinecone_api_key)

@lru_cache()
def get_embed_model():
    return TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

def ensure_index():
    _, pc = clients()
    if not pc.has_index(settings.pinecone_index):
        pc.create_index(
            name=settings.pinecone_index,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(cloud='aws', region='us-east-1')
        )
        while not pc.describe_index(settings.pinecone_index).status['ready']:
            time.sleep(1)
    return pc.Index(settings.pinecone_index)

def embed(texts):
    model = get_embed_model()
    embeddings = list(model.embed(texts))
    return [e.tolist() for e in embeddings]

def search_kb(query, top_k=5):
    index = ensure_index()
    vector = embed([query])[0]
    result = index.query(vector=vector, top_k=top_k, include_metadata=True)
    return [
        {
            'page': m.metadata.get('page'),
            'text': m.metadata.get('text', ' '),
            'score': round(m.score, 3),
            'source': m.metadata.get('file', 'the gale of encyclopedia.pdf')
        }
        for m in result.matches
        if m.score >= 0.30
    ]