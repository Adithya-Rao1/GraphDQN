from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from webapp.backend.config import CORS_ORIGINS
from webapp.backend.db import Base, engine
from webapp.backend import models  # noqa: F401 -- registers ORM models on Base.metadata
from webapp.backend.routers import auth, candidates, configs, generation, meta, molecules, pareto_sweep, proteins, runs

app = FastAPI(title="Metis Molecular Optimization API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup():
    Base.metadata.create_all(bind=engine)


app.include_router(auth.router)
app.include_router(meta.router)
app.include_router(molecules.router)
app.include_router(proteins.router)
app.include_router(configs.router)
app.include_router(runs.router)
app.include_router(generation.router)
app.include_router(candidates.router)
app.include_router(pareto_sweep.router)


@app.get("/api/health")
def health():
    return {"status": "ok"}
