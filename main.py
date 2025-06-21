from fastapi import FastAPI
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel
from typing import List

app = FastAPI(title="Embedding API", version="1.0.0")

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

class TextRequest(BaseModel):
    text: str

class TextListRequest(BaseModel):
    texts: List[str]

@app.get("/")
async def root():
    return {"message": "Embedding API with all-MiniLM-L6-v2"}

@app.post("/embed")
async def embed_text(request: TextRequest):
    embedding = model.encode(request.text)
    return {"embedding": embedding.tolist()}

@app.post("/embed_batch")
async def embed_texts(request: TextListRequest):
    embeddings = model.encode(request.texts)
    return {"embeddings": embeddings.tolist()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
