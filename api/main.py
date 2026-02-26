"""FastAPI main application — LexAlign backend API."""

import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routers import download, finetune, align, status, files

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

app = FastAPI(title="LexAlign API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(download.router, prefix="/api")
app.include_router(finetune.router, prefix="/api")
app.include_router(align.router, prefix="/api")
app.include_router(status.router, prefix="/api")
app.include_router(files.router, prefix="/api")


@app.get("/")
async def root():
    return {"message": "LexAlign API", "docs": "/docs"}
