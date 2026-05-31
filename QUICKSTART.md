# dgm-2026.1

## Requirements

- [Docker](https://docs.docker.com/get-docker/) installed
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) (for GPU support)

---

## Usage

### 1. Clone the repository

```bash
git clone https://github.com/NuitJack/dgm-2026.1.git
cd dgm-2026.1
```

### 2. Build the image

```bash
docker build -t <image-name> .
```

> Replace `<image-name>` with whatever name you want to give the image.

### 3. Run the container

```bash
docker run \
  -it \
  --ipc=host \
  --name das_container \
  --gpus all \
  --volume $PWD:/workspace \
  --ulimit memlock=-1 \
  --memory=100g \
  --ulimit stack=67108864 \
  --rm \
  <image-name> bash
```

> The current directory (`$PWD`) will be mounted at `/workspace` inside the container.

---

## Key flags

| Flag | Description |
|------|-------------|
| `--gpus all` | Enables access to all available GPUs |
| `--ipc=host` | Shares IPC memory with the host (required by some frameworks) |
| `--volume $PWD:/workspace` | Mounts the project directory into the container |
| `--memory=100g` | Limits memory usage to 100 GB |
| `--ulimit memlock=-1` | Allows unlimited memory locking — prevents the OS from swapping CUDA pinned memory to disk |
| `--ulimit stack=67108864` | Sets the stack size limit to 64 MB — prevents stack overflows in deep learning workloads with large buffers or threaded workers |
| `--rm` | Automatically removes the container on exit |
