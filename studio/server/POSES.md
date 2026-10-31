# DynamicBind service contract (for the lambda-ops agent)

The studio API (`studio/server/poses.py`) talks to a small HTTP service in front of DynamicBind.
Base URL: env `DYNAMICBIND_URL` (default `http://127.0.0.1:8100`). JSON in, JSON out. The API polls every
`DYNAMICBIND_POLL` s (default 2) and gives up after `DYNAMICBIND_TIMEOUT` s (default 900).

## `POST /dock`
Request: `{"target_id": "<id from studio/data/targets>", "smiles": "<canonical SMILES>"}`
Response: `{"job_id": "<opaque string>"}` (HTTP 200). The service looks up the target's cleaned protein in
`studio/data/structures/` itself.

## `GET /jobs/{job_id}`
Response:
```json
{"status": "queued|running|done|failed", "progress": 0.0,
 "ligand_sdf": "...best pose, SDF text...", "protein_pdb": "...predicted protein conformation, PDB text...",
 "apo_pdb": "...optional input structure...", "pose_confidence": 0.0, "affinity": 0.0, "note": "free text"}
```
Fields after `status` are required only when `status == "done"` (`apo_pdb` optional: the API fills it from the
input structure). On `failed`, put the reason in `note`.

Any non-200, connection error or timeout turns the pose job into `status="failed"` with an explanatory note;
the API never invents a pose. Concurrency on the API side is 1 job at a time.

## Precomputed poses
`studio/data/precomputed_poses/<target_id>/<hash>.json` holds a serialised `schema.PoseResult`, where
`<hash>` = `sha1(rdkit_canonical_smiles.encode()).hexdigest()[:16]`
(`from studio.server.poses import smiles_hash`). If present it is returned immediately with `source="precomputed"`.
Override the directory with `STUDIO_POSES_DIR`.
