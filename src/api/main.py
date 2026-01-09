from fastapi import FastAPI

from src.api.routes import router

app = FastAPI(title="ML Stock Market Predictor", version="1.0.0")
app.include_router(router)