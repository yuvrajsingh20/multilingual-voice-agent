from fastapi import FastAPI

app = FastAPI(title="Multilingual Debt Voice Agent")


@app.get("/health")
def health():
    return {"status": "ok"}
