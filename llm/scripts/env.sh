# Source before running any vLLM script.
# - ninja and nvcc must be on PATH: FlashInfer JIT-compiles kernels (otherwise the engine hangs silently). CUDA_HOME
#   must hold a toolkit matching torch's CUDA version (13.0).
# - spawn (not fork) for the engine process: once transformers/torch has touched CUDA in the parent, a forked
#   engine fails with "CUDA error: initialization error" or hangs. Spawn requires an `if __name__ == "__main__":` guard.
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export PATH=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/.venv/bin:$CUDA_HOME/bin:$PATH
export VLLM_WORKER_MULTIPROC_METHOD=spawn
