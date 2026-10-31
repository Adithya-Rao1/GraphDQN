"""Studio server STUB (owner: server agent replaces the internals, keeps the routes).

Routes are the contract the web UI codes against. In wave 0 every run simply replays
studio/mock/mock_run.json so the full path (validate -> create run -> SSE) works.
"""
from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from rdkit import Chem

from studio.schema import RunConfig, RunCreated, ValidateRequest, ValidateResponse

ROOT = Path(__file__).resolve().parents[1]
MOCK = ROOT / "mock" / "mock_run.json"
WEB = ROOT / "web"

app = FastAPI(title="Metis Lead Studio (stub)")
_runs: dict[str, RunConfig] = {}


@app.get("/api/health")
def health():
    return {"ok": True, "stub": True}


@app.post("/api/molecules/validate", response_model=ValidateResponse)
def validate(req: ValidateRequest):
    mol = None
    if req.smiles:
        mol = Chem.MolFromSmiles(req.smiles.strip())
    elif req.molfile:
        mol = Chem.MolFromMolBlock(req.molfile)
    if mol is None:
        return ValidateResponse(ok=False, error="Could not parse molecule")
    if mol.GetNumHeavyAtoms() > 60:
        return ValidateResponse(ok=False, error="Molecule too large (max 60 heavy atoms)")
    return ValidateResponse(ok=True, canonical_smiles=Chem.MolToSmiles(mol), heavy_atoms=mol.GetNumHeavyAtoms())


@app.post("/api/runs", response_model=RunCreated)
def create_run(cfg: RunConfig):
    run_id = uuid.uuid4().hex[:12]
    _runs[run_id] = cfg
    return RunCreated(run_id=run_id)


@app.get("/api/runs/{run_id}/events")
async def run_events(run_id: str, delay: float = 0.6):
    if run_id not in _runs:
        raise HTTPException(404, "unknown run")
    events = json.loads(MOCK.read_text())

    async def stream():
        for ev in events:
            ev = {**ev, "run_id": run_id}
            yield f"event: {ev['type']}\ndata: {json.dumps(ev)}\n\n"
            await asyncio.sleep(delay)

    return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


if WEB.exists():
    app.mount("/", StaticFiles(directory=WEB, html=True), name="web")
