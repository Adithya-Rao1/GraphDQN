# Publishing the Metis Space (owner runs this; nothing here has been pushed)

Space: `Adithya-Rao/Metis-Space` (Gradio SDK, bucket at `/metis-data`). Public URL: https://adithya-rao-metis-space.hf.space

## What is deployed
`deploy/space/` is the whole Space repo: `README.md` (YAML header), `app.py`, `requirements.txt`, `web/` (generated).
`app.py` is a FastAPI gateway: static UI at `/`, `/api/*` proxied to the Lambda backend (SSE unbuffered, 15 s gateway
heartbeats, 15 min stream cap, 256 KB body cap, 60 s request cap), `/api/health` answered locally
(`{"backend":"up"|"down"}`, 2 s probe, 3 s cache) so the UI can switch to replay.
Backend URL: `/metis-data/backend_url.txt` (if present) overrides the `BACKEND_URL` variable and is re-read on every
request, so a new cloudflared URL needs no restart. `STUDIO_API_KEY` is only sent to the backend as `X-Studio-Key`;
the browser gets a gateway-only cookie `metis_gw`, never the key and never the backend cookie.

Why `uvicorn.run` in `app.py`: the Gradio-SDK runner just executes `python app.py` and exposes port 7860, so any ASGI
server works. A minimal Gradio Blocks is mounted at `/gradio` (optional, skipped if gradio is missing) so the Space
still looks like a Gradio app. If the runner ever ignores the `__main__` block, fall back to exposing a Gradio `demo`
variable and mounting the proxy with `gr.mount_gradio_app` (not needed so far; untested on the live Space).

## ZeroGPU question
- Reports on the HF forum ([1](https://discuss.huggingface.co/t/issues-with-multiple-spaces-stuck-with-no-spaces-gpu-function-detected-during-startup/180537),
  [2](https://discuss.huggingface.co/t/does-zerogpu-not-work-for-all-spaces/113888)) show a ZeroGPU Space fails at
  startup with `Runtime Error: No @spaces.GPU function detected during startup` when no function is decorated, and that
  `import spaces` should come first. I could not find official docs stating this precisely, so treat it as likely, not certain.
- Mitigation already in `app.py`: a never-called `@spaces.GPU` no-op (guarded by try/except, so harmless on cpu-basic).
- Safe fallback if the Space still fails to start: Settings -> Hardware -> **CPU basic** (free). No SDK or code change needed;
  the gateway needs no GPU. Not changed by me. (A ZeroGPU quota is irrelevant: the GPU work runs on Lambda.)

## Pre-flight checklist
1. `deploy/build_space.sh` (copies `studio/web/` incl. `vendor/3Dmol-min.js`, `replay/`, mock run into `deploy/space/web/`).
   Warns if `studio/web/replay/` is missing: rebuild after the viewer branch with replay is merged.
2. `.venv/bin/python -m pytest studio/deploy -q -p no:cacheprovider` is green.
3. Local smoke: `BACKEND_URL=http://127.0.0.1:8000 STUDIO_API_KEY=x PORT=7861 python deploy/space/app.py` with
   `STUDIO_FORCE_MOCK=1 uvicorn studio.server.app:app` running; open http://localhost:7861 and play a run.
4. The studio backend must enforce `X-Studio-Key` (it does not in `studio/server/app.py` today; see handoff) and bind to
   the tunnel only.
5. `hf auth whoami` shows the owner; confirm hardware choice above; confirm no secrets in `deploy/space/`
   (`git grep -n -i "key\|token" deploy/space` should only show env-var names).

## Commands
```bash
export SPACE=Adithya-Rao/Metis-Space
deploy/build_space.sh

# secret + variable (the key must equal the one the Lambda backend expects)
hf spaces secrets set $SPACE STUDIO_API_KEY "<random 32+ chars>"        # or: Settings -> Variables and secrets
hf spaces variables set $SPACE BACKEND_URL "https://<your-tunnel>.trycloudflare.com"
# (CLI names vary by version: `hf spaces --help`; the web UI under Settings works equally well.)

# upload the folder as the Space contents (delete stale files so web/ stays exact)
hf upload $SPACE deploy/space . --repo-type space --delete "web/*" --commit-message "Metis gateway"
```
Rotating tunnel without restart (bucket is mounted at /metis-data in the Space):
```bash
echo "https://<new-tunnel>.trycloudflare.com" > backend_url.txt
hf buckets cp backend_url.txt hf://buckets/<owner>/<bucket-name>/backend_url.txt   # check `hf buckets --help` for exact syntax
```
Verify: `curl https://adithya-rao-metis-space.hf.space/api/health` -> `{"backend":"up",...}` (or `"down"` -> UI uses replay).

## Known limits
- Single gateway worker; SSE connections are capped at 15 min each and rely on the backend's own run limits.
- The backend sees one stable pseudo-session per browser (HMAC of the gateway cookie), not the user's IP; per-session rate limits work, per-IP ones do not.
