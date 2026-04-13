# Docker Setup

This project is containerized as two services:

- `api`: FastAPI inference backend on `http://localhost:8000`
- `demo`: Gradio frontend on `http://localhost:7860`

## CPU run

```bash
docker compose up --build
```

## CUDA run

```bash
docker compose -f docker-compose.yml -f docker-compose.cuda.yml up --build
```

This uses `Dockerfile.cuda` and requests GPU access for the `api` service.

## Stop

```bash
docker compose down
```

For the CUDA stack:

```bash
docker compose -f docker-compose.yml -f docker-compose.cuda.yml down
```

## Notes

- The `demo` service waits for the `api` healthcheck before starting.
- Model checkpoints are included in the build context from `outputs/**/checkpoints`.
- The Gradio container talks to the backend via the internal Compose hostname `http://api:8000`.
- `Dockerfile` is the CPU-oriented image.
- `Dockerfile.cuda` is the CUDA-oriented image for NVIDIA GPU runtime.
- `requirements.docker.txt` contains shared Docker runtime dependencies except PyTorch, which is supplied by the selected image/runtime.
