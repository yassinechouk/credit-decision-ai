from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import router

app = FastAPI(title="Credit Decision AI")

app.add_middleware(
	CORSMiddleware,
	allow_origins=[
		"http://localhost:4173",
		"http://127.0.0.1:4173",
	],
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)

app.include_router(router)
