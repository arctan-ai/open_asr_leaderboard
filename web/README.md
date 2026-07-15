# Open ASR Evaluation Console

Desktop-first operator UI for `api/run_eval.py`. The console launches API-provider evaluations, streams logs, stores run history, and exposes manifests and summaries without requiring handwritten curl commands.

The React application is served by Vite during development. In production, FastAPI serves the compiled `web/dist` assets and the API from one localhost-only process.

## Prerequisites

- Python environment for the repository
- Node.js 20 or newer and npm
- Provider and preprocessor credentials in the repository `.env`
- Hugging Face access through `HF_TOKEN` when the selected dataset is private

The browser never receives credential values. `/api/options` exposes only whether each required credential is configured or missing.

## Local setup

Run these commands from the repository root:

```bash
uv pip install -r requirements/requirements-ui.txt
npm --prefix web ci
```

Use the repository's existing Python environment. The first command adds only the console-specific backend dependencies to that environment.

Start the backend in one terminal:

```bash
uv run uvicorn api.ui_server:app \
  --host 127.0.0.1 \
  --port 8000 \
  --reload
```

Start Vite in another terminal:

```bash
npm --prefix web run dev
```

Open `http://127.0.0.1:5173`. Vite proxies `/api` requests to the backend on `127.0.0.1:8000`.

## Using the console

1. Enter a Hugging Face dataset ID, then select **Inspect** to verify access and load its configurations, splits, and audio schema.
2. Select the dataset configuration, split, provider/model, and language.
3. Configure workers and an optional sample limit. A blank sample limit evaluates the full split.
4. Select an optional audio preprocessor and VAD position:
   - **Pre VAD:** audio → VAD → preprocessor → ASR
   - **Post VAD:** audio → preprocessor → VAD → ASR
   - Without a preprocessor, pre and post both resolve to audio → VAD → ASR.
5. Enable streaming or remote URL mode when supported. URL mode is disabled whenever streaming, preprocessing, VAD, or a real-time-only model requires locally decoded audio.
6. Open **Advanced** for the Arctan chunk size, preprocessing batch size, or provider prompt.
7. Select **Run evaluation**. Runs start immediately and there is no global concurrency limit.

The header shows service health, active-run count, theme controls, and credential availability. Select a run from the history table to view its configuration, live logs, WER, RTFx, sample count, errors, downloadable artifacts, or cancellation control.

Each run is isolated under:

```text
results/ui-runs/<run-id>/
```

Run history is persisted in SQLite using WAL mode. Active runs left unfinished by a backend restart are marked `interrupted`.

## Compatibility rules

- Preprocessing and VAD require local audio and cannot be combined with URL mode.
- Streaming requires local audio.
- Real-time-only models force streaming and therefore disable URL mode.
- Missing provider or preprocessor credentials block submission.
- Silero VAD supports local 8 kHz or 16 kHz audio and preserves duration by replacing non-speech samples with zeros.
- Parallel runs are unlimited; consider provider cost and rate limits before launching large evaluations.

## Checks and production build

```bash
npm --prefix web run lint
npm --prefix web run test
npm --prefix web run build
```

The production build is written to `web/dist`. With that directory present, start the combined UI and API server with:

```bash
uv run uvicorn api.ui_server:app \
  --host 127.0.0.1 \
  --port 8080
```

## `livekit-server` deployment

The repository includes `deploy/open-asr-console.service`, configured for `/home/ubuntu/open_asr_leaderboard` and a localhost-only listener on port `8080`. The current `livekit-server` host does not provide Node.js, so build the frontend locally or in CI and copy the compiled assets to the server.

From the repository root on a machine with Node.js:

```bash
npm --prefix web ci
npm --prefix web run build
rsync -az web/dist/ \
  livekit-server:/home/ubuntu/open_asr_leaderboard/web/dist/
```

Then on `livekit-server`:

```bash
cd /home/ubuntu/open_asr_leaderboard
/home/ubuntu/.local/bin/uv pip install \
  --python .venv/bin/python \
  -r requirements/requirements-ui.txt
sudo install -m 0644 \
  deploy/open-asr-console.service \
  /etc/systemd/system/open-asr-console.service
sudo systemctl daemon-reload
sudo systemctl enable --now open-asr-console.service
sudo systemctl status open-asr-console.service
```

Access it from your machine through an SSH tunnel:

```bash
ssh -L 8080:127.0.0.1:8080 livekit-server
```

Then open `http://127.0.0.1:8080`.

Do not bind the service publicly for this MVP. It does not include public authentication or TLS termination.
