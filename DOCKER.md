# Docker (Windows, macOS, Linux)

Training runs inside a **Linux** container with **CUDA-enabled PyTorch** so your **NVIDIA GPU** can be used when Docker passes it through. Install [Docker Desktop](https://docs.docker.com/desktop/) on Windows or macOS, or Docker Engine on Linux.

## GPU prerequisites

- **NVIDIA GPU** with a recent driver (CUDA 12.x–capable drivers work with the `cu124` wheels in the Dockerfile).
- **Docker GPU support**
  - **Windows:** Docker Desktop with **WSL2** backend, [NVIDIA drivers for WSL](https://docs.nvidia.com/cuda/wsl-user-guide/index.html), and GPU support enabled in Docker Desktop settings.
  - **Linux:** [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) so `docker run --gpus all` works.

`docker-compose.yml` sets `gpus: all` on `train`, `generate`, and `chat`. Training logs should show `Using device: cuda` when the GPU is visible inside the container.

## One-time: build the image

From the project root (`mini-ml`):

```bash
docker build -t mini-ml:latest .
```

Or:

```bash
docker compose build
```

## Train

Mount the project folder so `artifacts/`, `data/processed/`, and logs stay on your machine.

**Compose (recommended):**

```bash
docker compose run --rm train
```

**Manual `docker run` with GPU** — PowerShell:

```powershell
docker run --rm --gpus all -v "${PWD}:/app" mini-ml:latest
```

Windows CMD:

```cmd
docker run --rm --gpus all -v "%cd%":/app mini-ml:latest
```

macOS/Linux:

```bash
docker run --rm --gpus all -v "$(pwd):/app" mini-ml:latest
```

Training writes:

- `artifacts/tiny_llm.pt`
- `artifacts/tokenizer.model`
- `data/processed/*.txt`

## After training: generate text

**Compose:**

```bash
docker compose --profile tools run --rm generate
```

**Manual** (PowerShell):

```powershell
docker run --rm --gpus all -v "${PWD}:/app" mini-ml:latest python scripts/generate.py --prompt "hi" --tokens 80
```

(Adjust the volume flag for Windows CMD as above.)

## CPU-only image (no NVIDIA GPU)

Build with the CPU wheel and run **without** `--gpus`:

```bash
docker build --build-arg TORCH_INDEX=cpu -t mini-ml:cpu .
docker run --rm -v "${PWD}:/app" mini-ml:cpu python scripts/train.py
```

On hosts **without** an NVIDIA GPU (or where `gpus: all` makes Compose fail), remove the `gpus: all` lines from each service in `docker-compose.yml`, then rebuild with `--build-arg TORCH_INDEX=cpu` if you want a smaller CPU-only image.

## CUDA wheel version

The default Dockerfile uses PyTorch’s **`cu124`** index. To match a different stack, change the `TORCH_INDEX` build-arg (see [PyTorch Get Started](https://pytorch.org/get-started/locally/)) and rebuild.
