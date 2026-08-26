from fastapi import FastAPI

app = FastAPI(title="Test Vercel", version="1.0")

@app.get("/")
def read_root():
    return {"status": "OK", "message": "¡Funciona en Vercel perfectamente!"}
