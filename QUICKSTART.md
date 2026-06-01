# dgm-2026.1

## Requirements

- [Docker](https://docs.docker.com/get-docker/) installed
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) (for GPU support)

---

## Usage

### 1. Clone the repository

```bash
git clone --recurse-submodules https://github.com/NuitJack/dgm-2026.1.git
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


# Singularity + SLURM Quickstart

Replace `<name>` with your container/project name throughout this guide.

---

## 1. Prepare the container

### Option A — From a Dockerfile (local machine)

```bash
docker build -t <name>:latest .
docker save <name>:latest -o <name>.tar
scp <name>.tar <user>@<cluster>:./myfolder/
```

On the cluster, build the `.sif` from the tar:

```bash
singularity build --fakeroot <name>.sif docker-archive://<name>.tar
```

### Option B — From a `.def` file (on the cluster)

```bash
singularity build --fakeroot <name>.sif <name>.def
```

---

## 2. Test interactively (optional)

Useful for debugging before submitting a job:

```bash
singularity shell --nv <name>.sif
```

Inside the shell you can run `python`, `pip list`, etc. to verify the environment is correct. Exit with `Ctrl+D`.

---

## 3. Create the SLURM job script

```bash
nano run.mpi
```

```bash
#!/bin/bash
#SBATCH --job-name=<name>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu02        # adjust to your cluster's available partition

srun singularity exec --nv <name>.sif python /path/to/your_script.py
```

> **`--nv`** exposes the node's GPUs inside the container — required for PyTorch/CUDA.

---

## 4. Submit and monitor

```bash
sbatch run.mpi               # submit the job
squeue                       # check job status
scancel <job_id>             # cancel if needed
cat slurm-<job_id>.out       # view output and errors
```

### Status codes

| Code | Meaning    |
|------|------------|
| `R`  | Running    |
| `PD` | Pending    |
| `CG` | Completing |

---

## 5. Copy files to the cluster (reference)

```bash
# Single file
scp myfile.txt <user>@<cluster>:./myfolder/

# Folder
scp -r myfolder/ <user>@<cluster>:./myfolder/
```
