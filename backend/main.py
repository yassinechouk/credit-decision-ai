from fastapi import FastAPI
from api.routes import router

app = FastAPI(title="Credit Decision AI")

app.include_router(router)
