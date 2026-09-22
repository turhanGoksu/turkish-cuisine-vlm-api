# Turkish Cuisine VLM API

A containerised FastAPI service that serves [`Turhan123/turkish-cuisine-vlm`](https://huggingface.co/Turhan123/turkish-cuisine-vlm) — a LoRA adapter over Qwen2-VL-2B that identifies Turkish dishes from a photograph and answers nutrition questions about them in Turkish — and records every answer in PostgreSQL.

The point of this repository is not the model; that was trained separately. The point is the service around it: how the image is built, why the weights live where they do, which endpoint blocks and which does not, and where the failure modes are.

---

## Architecture

```
                        docker-compose private network
  HTTP                ┌────────────────────────────────────────────┐
 ──────────────────▶  │  api                        db             │
  POST /predict       │  FastAPI + uvicorn          PostgreSQL 16  │
  GET  /history       │  Qwen2-VL-2B + LoRA   ────▶ :5432          │
  GET  /health        │  :8000  (published)         (not published)│
                      └────────┬──────────────────────┬────────────┘
                               │                      │
                        volume: hf-cache        volume: pgdata
                        4.75 GB of weights      prediction history
```

Two containers, one concern each. The API owns inference, PostgreSQL owns durable state, and neither knows how the other is deployed. The database port is deliberately not published to the host: it is reachable from the `api` service over the private network and from nowhere else.

The API connects to `db:5432`, not `localhost:5432`. Each container has its own network namespace, so `localhost` inside the API container means the API container itself; `db` is the DNS name Compose creates from the service key.

---

## Endpoints

| Method | Path | Notes |
| --- | --- | --- |
| `GET` | `/health` | Reports liveness and whether the model has finished loading. Used by the container healthcheck. |
| `POST` | `/predict` | `multipart/form-data`: an image file plus an optional `question` field. |
| `GET` | `/history` | Stored predictions, newest first, with `limit` and `offset`. |
| `GET` | `/docs` | Swagger UI, generated from the Pydantic schemas. |

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@lahmacun.jpg" \
  -F "question=Bu yemek nedir ve besin değerleri nasıldır?"
```

```json
{
  "filename": "lahmacun.jpg",
  "question": "Bu yemek nedir ve besin değerleri nasıldır?",
  "answer": "Bu bir adet lahmacun. Besin değerleri:\n- Kalori: 280 kcal\n- Protein: 12g\n- Yağ: 10g\n- Karbonhidrat: 35g",
  "duration_ms": 131392
}
```

---

## Design decisions

### The model weights are not in the image

The base model and the adapter together are 4.75 GB. Baking them into the image was the first plan and it was wrong: the download step sits above `pip install` in the Dockerfile, so adding a single dependency would invalidate that layer and re-download 4.75 GB, and a ~6 GB image cannot reasonably be pushed to a registry.

Instead the weights live in a named volume mounted at `HF_HOME`. The image is **1.37 GB**, image rebuilds take ~30 seconds, and the weights are downloaded exactly once:

```bash
docker compose run --rm api python scripts/download_model.py
```

This trades away one property worth naming: the image is no longer self-sufficient. A brand-new deployment needs network access to Hugging Face for its first start. For a model this size that trade is worth it; for a 300 MB classifier it would not have been.

### Dockerfile layer ordering

Least-frequently-changed first, so that editing application code never reinstalls anything:

| Layer | Changes | Cost when it runs |
| --- | --- | --- |
| `pip install torch torchvision` (CPU index) | almost never | ~2 min |
| `pip install -r requirements.txt` | occasionally | ~30 s |
| `COPY scripts/` | rarely | instant |
| `COPY app/` | constantly | instant |

PyTorch is installed from `download.pytorch.org/whl/cpu` rather than PyPI. The default wheels bundle CUDA libraries worth roughly 2.5 GB that this CPU-only image would never load.

The base image is `python:3.11-slim`, not Alpine. Alpine uses musl libc, so the prebuilt manylinux wheels for torch and psycopg2 do not apply and would be compiled from source. The `db` service *does* use Alpine, because the official PostgreSQL image ships compiled C and has no such problem — the same technology choice goes the other way in a different context.

### `def` and `async def` are chosen per endpoint

- `POST /predict` is a plain `def`. Generation is CPU-bound, not I/O-bound; declaring it `async def` would pin the event loop for the whole two minutes and stall every other request, including the healthcheck, until Docker declared the container dead.
- `GET /health` is `async def`. It does no blocking work at all, so it stays on the event loop and keeps answering even when the thread pool is saturated with inference — which is exactly when a healthcheck matters most.
- Inference is additionally serialised behind a `threading.Lock`. Two thread-pool workers calling `generate()` at once would not be faster on CPU; they would compete for the same cores and hold two KV caches in memory.

### Database sessions are scoped to the work, not to the request

`GET /history` takes a session through `Depends(get_session)`: the query is fast, so holding a pooled connection for the request is fine.

`POST /predict` does not. A session holds a connection from a pool of five, and holding one open across two minutes of generation would exhaust the pool under trivial load. The session is opened afterwards, for the few milliseconds the insert needs.

### A failed history write does not fail the request

Writing to `predictions` is a side effect, not what the caller asked for. If the insert fails, the answer — which cost two minutes of CPU — is still returned and the failure is logged at ERROR level.

This is a judgement about *this* endpoint, not a general rule. On something like `POST /orders`, where the write **is** the request, the opposite is correct: reporting success without persisting would be a lie.

### A failed adapter load *does* fail startup

`peft` only emits a `UserWarning` when adapter weights do not match the base model's module names. The service then runs the unmodified base model while every response still looks plausible.

That is worse than an outage: the answers are confidently wrong, and unlike a transient database error it never recovers. `app/ml.py` therefore captures that warning and raises, so the container dies at startup instead of serving the wrong model.

This is not hypothetical — it happened here. `transformers` was initially pinned to 4.46.3, which names Qwen2-VL's submodules `model.layers.*`, while the adapter was trained against a newer layout that names them `model.language_model.layers.*`. Every LoRA key silently failed to match. The pin is now 4.57.3, contemporary with the `peft_version` recorded inside the adapter's own `adapter_config.json`.

The lesson cuts both ways: pinning gives reproducible builds, but pinning to the *wrong* version fails reproducibly every time.

### Startup order is enforced by health, not by `depends_on`

`depends_on` alone only guarantees that the database container was *started*, not that PostgreSQL accepts connections — and on a first run `initdb` takes tens of seconds. Since the API connects during startup to create its table, it would crash and be restarted in a loop.

The `db` service therefore defines a `pg_isready` healthcheck and the API waits on `condition: service_healthy`.

The API's own healthcheck is written in Python rather than `curl`, because `python:3.11-slim` does not ship `curl`; and it allows a 300 second `start_period`, because loading the model legitimately takes that long and Docker would otherwise kill it mid-load.

### The API does not report a confidence score

This is a generative vision-language model, not a classifier. It produces text, not a label with a probability. A `confidence` field would be a fabricated number, so the response exposes the generated `answer` and a measured `duration_ms` instead.

Likewise, the model is not asked to emit JSON. It was fine-tuned to answer in Turkish prose, and a 2B model coaxed into a format it was not trained for fails intermittently — which is the worst kind of failure.

### `bfloat16`, not `float32` or `float16`

At 2.2 B parameters, `float32` would need ~8.9 GB of weights alone and the container would be OOM-killed (exit code 137, with no Python traceback). `bfloat16` halves that to ~4.4 GB. `float16` is not the answer either: many PyTorch operations have no half-precision CPU kernel.

### Schemas and ORM models are separate files

`app/schemas.py` describes the API contract; `app/models.py` describes the table. Merging them would mean a column rename silently changes the public response. `PredictionRecord` sets `from_attributes=True` so FastAPI can serialise ORM rows without either layer leaking into the other.

---

## Measured behaviour

Running on CPU inside Docker Desktop on an Apple laptop, with `MAX_NEW_TOKENS=150`:

| Image | `MAX_PIXELS` | Prompt tokens | Answer | Latency |
| --- | --- | --- | --- | --- |
| `iskender.jpg` (720×480) | 200 704 | 284 | "et döner" ❌ | 110 s |
| `iskender.jpg` | **401 408** | 505 | "İskender Kebap" ✅ | 176 s |
| `iskender.jpg` | 12 845 056 (checkpoint default) | 773 | "İskender Kebap" ✅ | 208 s |
| `lahmacun.jpg` (632×898) | 401 408 | — | "lahmacun" ✅ | 131 s |
| `mercimek_corbasi.jpeg` (554×554) | 401 408 | — | "mercimek çorbası" ✅ | 181 s |

`MAX_PIXELS` defaults to 401 408 because that was the point where the answers still matched the uncapped setting while the prompt shrank by 35% and wall time by roughly 15%. Capping harder was measurably worse: at 200 704 the model downgraded İskender to plain döner.

Latency is dominated by sequential token generation, not by image size — the smallest sample image is the slowest request, because its answer is the longest.

---

## Running it

Requires Docker with at least 8 GB of memory available to the engine (Settings → Resources on Docker Desktop).

```bash
cp .env.example .env          # then choose a POSTGRES_PASSWORD

docker compose build
docker compose run --rm api python scripts/download_model.py   # ~4.75 GB, once
docker compose up -d
docker compose logs -f api    # wait for "Model ready"
```

Then open http://localhost:8000/docs.

To inspect the database, which is not published to the host:

```bash
docker compose exec db psql -U turkish_cuisine -d turkish_cuisine -c 'select id, filename, duration_ms from predictions;'
```

---

## Limitations, and what production would look like

This project is a demonstration of containerisation and API design, not a production inference service. Being specific about that is part of the point:

- **Two minutes per request on CPU.** Beyond most client timeouts. A real deployment would run on a GPU, or accept the upload and return a job id, with the answer fetched later.
- **One model per process.** Each uvicorn worker would load its own 4.4 GB copy, so the service cannot be scaled by adding workers. Past a certain load the model belongs behind a dedicated inference server (vLLM, TorchServe, Triton) that the API calls over HTTP.
- **`Base.metadata.create_all()` instead of migrations.** It creates missing tables but cannot alter existing ones; adding a column later would leave the table untouched and break the application. One table that will not change makes this acceptable here. Alembic is the real answer.
- **Offset pagination.** `OFFSET 500000` makes PostgreSQL read and discard half a million rows, and rows shift between pages when new ones are inserted. Keyset pagination is the fix at scale.
- **No authentication and no rate limiting.** Uploads are capped at 10 MB and decoded before use, but an unauthenticated endpoint that costs two minutes of CPU per call is trivially abusable.
- **No automated tests.** Loading is kept out of import time (`lifespan`, `lru_cache`) precisely so the modules can be imported and the model dependency overridden, but the tests themselves are not written yet.

---

## Layout

```
app/
  main.py       FastAPI app, lifespan, endpoints
  ml.py         model loading, the adapter guard, inference
  schemas.py    Pydantic request/response contract
  models.py     SQLAlchemy ORM model
  database.py   engine, session factory, session dependency
scripts/
  download_model.py   fills the weight cache volume
Dockerfile
docker-compose.yml
.env.example
```

The model itself — training data, LoRA fine-tuning notebook and evaluation — lives in [turhanGoksu/turkish-cuisine-vlm](https://github.com/turhanGoksu/turkish-cuisine-vlm).

---

## License

The code in this repository is licensed under the Apache License 2.0; see [LICENSE](LICENSE).

The weights it serves are covered separately, and running this service means accepting their terms too:

| Artifact | License |
| --- | --- |
| This repository | Apache 2.0 |
| [`Turhan123/turkish-cuisine-vlm`](https://huggingface.co/Turhan123/turkish-cuisine-vlm) (LoRA adapter) | Apache 2.0 |
| [`Qwen/Qwen2-VL-2B-Instruct`](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct) (base model) | Apache 2.0 |

No sample photographs are committed. The images used for the measurements above were collected from the web, their copyright belongs to their respective owners, and `samples/` is git-ignored for that reason.
