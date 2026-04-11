# Docker Setup

This project is containerized as two services:

- `api`: FastAPI inference backend on `http://localhost:8000`
- `demo`: Gradio frontend on `http://localhost:7860`

## Run

```bash
docker compose up --build
```

## Stop

```bash
docker compose down
```

## Notes

- The `demo` service waits for the `api` healthcheck before starting.
- Model checkpoints are included in the build context from `outputs/**/checkpoints`.
- The Gradio container talks to the backend via the internal Compose hostname `http://api:8000`.
- Current Dockerfile is CPU-oriented. If you later want GPU support, we can add an NVIDIA-compatible variant.
