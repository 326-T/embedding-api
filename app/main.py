from fastapi import FastAPI

from app.router import embed, health, hybrid

app = FastAPI(title="Embedding API", version="1.0.0")

app.include_router(embed.router)
app.include_router(health.router)
app.include_router(hybrid.router)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
