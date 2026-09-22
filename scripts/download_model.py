"""Pre-download the model weights into the Hugging Face cache.

The weights are deliberately NOT baked into the Docker image: at ~4.75 GB they would
make every rebuild re-download them and the image too large to push to a registry.
They live in a named volume mounted at HF_HOME instead.

Run this once to fill that volume before the first request:

    docker compose run --rm api python scripts/download_model.py

The application also downloads on demand, so this script is an optimisation, not a
requirement. Running it separately keeps the slow download out of the request path.
"""

import os

from huggingface_hub import snapshot_download

# The LoRA adapter cannot run on its own; it is applied on top of the base model,
# so both repositories have to be present in the cache.
BASE_MODEL_ID: str = os.environ.get("BASE_MODEL_ID", "Qwen/Qwen2-VL-2B-Instruct")
ADAPTER_MODEL_ID: str = os.environ.get("ADAPTER_MODEL_ID", "Turhan123/turkish-cuisine-vlm")


def main() -> None:
    """Fetch the base model and the fine-tuned adapter into the local cache."""
    for repo_id in (BASE_MODEL_ID, ADAPTER_MODEL_ID):
        print(f"Downloading {repo_id} ...")
        path = snapshot_download(repo_id=repo_id)
        print(f"  cached at {path}")


if __name__ == "__main__":
    main()
