# Publishing the Metis Space front door (owner runs this; nothing here has been pushed or uploaded)

Space: `Adithya-Rao/Metis-Space` (Gradio SDK, bucket mounted at `/metis-data`). Public URL: https://adithya-rao-metis-space.hf.space

## What it does
`deploy/space/` is the whole Space repo: `README.md` (YAML header), `app.py`, `requirements.txt`, `web/` (generated).
`app.py` is a FastAPI app on port 7860. The Lambda box serves the real studio UI + API through a cloudflared quick
tunnel; the Space is the stable front door.

| Path | Behaviour |
|---|---|
| `/` | backend up (2 s probe, 3 s cache) -> **302** to the backend origin; down -> landing page (status chip, "Open the live studio" button, link to the recorded replay, clear "backend offline" notice) |
| `/landing` | landing page always; the chip polls `/api/health` every 5 s |
| `/open` | the button: 302 to the backend if up, else back to `/landing` |
| `/replay/?mock=1` | static studio UI from `web/` (the UI labels mock/replay data); `/mock/mock_run.json` is served for it |
| `/api/health` | local: `{"backend":"up"\|"down","gateway":"ok","proxy":bool}`; never contains the backend URL |
| `/live/{path}`, `/api/*` | OPTIONAL SSE-safe reverse proxy (heartbeats, 15 min stream cap, 256 KB body cap, 60 s request cap). **Off unless `ENABLE_PROXY=1`**; use it if the address bar must stay on hf.space |

Backend URL: **first line** of `/metis-data/backend_url.txt`, else the `BACKEND_URL` variable, read at request time (no
restart when the tunnel rotates). A candidate is accepted only if it is `https`, has no credentials/odd port, and its host
equals or is a subdomain of an entry in `ALLOWED_BACKEND_HOSTS` (default `trycloudflare.com`, comma-separated). Only the
origin is kept. A bad file value is skipped in favour of the variable; if nothing valid remains the offline page shows.

Optional: `STUDIO_API_KEY` secret is sent as `X-Studio-Key` to the backend by the health probe and proxy only; never to the browser.
`REPLAY_HREF` overrides the replay link (default `/replay/?mock=1`; point it at the recorded-replay URL once `studio/web/replay/` exists).

Why `uvicorn.run` in `app.py`: the Gradio-SDK runner executes `python app.py` and exposes port 7860, so any ASGI server
there works; a minimal Gradio Blocks is mounted at `/gradio` (skipped if gradio is missing). Not yet confirmed on the live Space.

## ZeroGPU
Forum reports ([1](https://discuss.huggingface.co/t/issues-with-multiple-spaces-stuck-with-no-spaces-gpu-function-detected-during-startup/180537),
[2](https://discuss.huggingface.co/t/does-zerogpu-not-work-for-all-spaces/113888)) say a ZeroGPU Space aborts with
`No @spaces.GPU function detected during startup` if nothing is decorated (no official doc found; treat as likely).
`app.py` contains a never-called, try/except-guarded `@spaces.GPU` no-op. If the Space still fails to start, switch hardware
to **CPU basic** (Settings -> Hardware): no SDK or code change needed, the front door needs no GPU. I changed no Space settings.

## Pre-flight
1. `deploy/build_space.sh` (copies `studio/web/` incl. `vendor/3Dmol-min.js`, any `replay/`, and the mock run into `deploy/space/web/`; warns if `replay/` is missing, so rebuild after that lands).
2. `.venv/bin/python -m pytest studio/deploy -q -p no:cacheprovider` is green (35 tests at time of writing).
3. Local smoke: `PORT=7861 BACKEND_URL=https://x.trycloudflare.com python deploy/space/app.py`, then `curl -i localhost:7861/` (offline page when the tunnel is not up) and open `/replay/?mock=1`.
4. `git grep -n -i "token\|secret" deploy/space` shows only env-var names.

## Commands
```bash
export SPACE=Adithya-Rao/Metis-Space
hf auth whoami
deploy/build_space.sh

# variables / secrets (CLI syntax from `hf spaces variables|secrets add --help`)
hf spaces variables add $SPACE -e ALLOWED_BACKEND_HOSTS=trycloudflare.com
hf spaces variables add $SPACE -e BACKEND_URL=https://<current-tunnel>.trycloudflare.com   # fallback if the bucket file is absent
# optional proxy mode (address bar stays on hf.space):
#   hf spaces variables add $SPACE -e ENABLE_PROXY=1
# optional key, only if the backend enforces X-Studio-Key (reads value from your shell env, not history):
#   export STUDIO_API_KEY=...; hf spaces secrets add $SPACE -s STUDIO_API_KEY

# publish the folder as the Space contents (--delete keeps web/ exact)
hf upload $SPACE deploy/space . --repo-type space --delete "web/*" --commit-message "Metis front door"
hf spaces wait $SPACE        # then: hf spaces logs $SPACE
curl -s https://adithya-rao-metis-space.hf.space/api/health
```

## When the tunnel URL changes
The bucket is mounted at `/metis-data`; write the new URL as the first line of `backend_url.txt` (from `hf buckets cp --help`: stdin `-` is a valid source):
```bash
hf buckets list                                   # find <owner>/<bucket> that is mounted at /metis-data
                                                  # (`hf spaces volumes list $SPACE` shows the mount)
printf '%s\n' "https://<new-tunnel>.trycloudflare.com" | hf buckets cp - hf://buckets/<owner>/<bucket>/backend_url.txt
# or from a file:  hf buckets cp backend_url.txt hf://buckets/<owner>/<bucket>/backend_url.txt
hf buckets cp hf://buckets/<owner>/<bucket>/backend_url.txt -    # verify (prints to stdout)
```
No restart is needed; the next request reads the file (probe cache is 3 s). The file contents are not secret but must pass the allowlist. If a new tunnel host would not end in `trycloudflare.com`, add its suffix to `ALLOWED_BACKEND_HOSTS` first.
I did not run any of these commands; whether the bucket's mount path/writes propagate live into the running Space is untested.

## Known limits
- Redirect mode hands the browser to the tunnel URL, so the address bar leaves hf.space; proxy mode keeps it but adds a hop and caps SSE at 15 min.
- In proxy mode the backend sees one stable pseudo-session per browser (HMAC of the gateway cookie `metis_gw`), not the client IP.
