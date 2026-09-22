"""Loading and running the Turkish cuisine vision-language model.

The served model is a LoRA adapter (Turhan123/turkish-cuisine-vlm) applied on top of
Qwen/Qwen2-VL-2B-Instruct. The adapter cannot run on its own, so both repositories are
fetched from the Hugging Face cache at startup.

Everything here is deliberately synchronous and blocking: generation is CPU-bound work,
not I/O, so it belongs in FastAPI's thread pool rather than on the event loop.
"""

import logging
import os
import threading
import time
from io import BytesIO

import torch
from peft import PeftModel
from PIL import Image, UnidentifiedImageError
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor

logger = logging.getLogger(__name__)

BASE_MODEL_ID: str = os.environ.get("BASE_MODEL_ID", "Qwen/Qwen2-VL-2B-Instruct")
ADAPTER_MODEL_ID: str = os.environ.get(
    "ADAPTER_MODEL_ID", "Turhan123/turkish-cuisine-vlm"
)
DEFAULT_QUESTION: str = "Bu yemek nedir ve besin değerleri nasıldır?"

# Each generated token costs roughly a second of CPU time, so the ceiling is a direct
# knob on worst-case request latency.
MAX_NEW_TOKENS: int = int(os.environ.get("MAX_NEW_TOKENS", "150"))


class InvalidImageError(ValueError):
    """Raised when the uploaded bytes cannot be decoded as an image."""


def decode_image(data: bytes) -> Image.Image:
    """Turn raw uploaded bytes into an RGB image.

    Args:
        data: The raw bytes of the uploaded file.

    Returns:
        The decoded image, converted to RGB.

    Raises:
        InvalidImageError: If the bytes are not a readable image.
    """
    try:
        # Pillow is tolerant of PNG, WebP and CMYK JPEGs; the model expects three
        # channels, so the conversion is not optional.
        return Image.open(BytesIO(data)).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise InvalidImageError("The uploaded file is not a readable image.") from exc


class DietitianModel:
    """A loaded vision-language model that answers questions about food images."""

    def __init__(
        self, model: PeftModel, processor: Qwen2VLProcessor
    ) -> None:
        self._model = model
        self._processor = processor
        # generate() mutates internal state and is CPU-saturating. Two threads running
        # it at once would not be faster — they would compete for the same cores and
        # hold two KV caches in memory at the same time. Serialise instead.
        self._lock = threading.Lock()

    @classmethod
    def load(cls) -> "DietitianModel":
        """Load the base model, apply the LoRA adapter, and return a ready instance."""
        logger.info("Loading base model %s", BASE_MODEL_ID)
        base_model = Qwen2VLForConditionalGeneration.from_pretrained(
            BASE_MODEL_ID,
            # bfloat16 halves the memory footprint to ~4.4 GB. float32 would need
            # ~8.9 GB and get the container OOM-killed; float16 is poorly supported
            # on CPU, where many operations have no half-precision kernel.
            torch_dtype=torch.bfloat16,
            # Stream the checkpoint shard by shard instead of materialising a full
            # extra copy of the weights while loading.
            low_cpu_mem_usage=True,
        )

        logger.info("Applying LoRA adapter %s", ADAPTER_MODEL_ID)
        model = PeftModel.from_pretrained(base_model, ADAPTER_MODEL_ID)
        # Disable dropout and other training-only behaviour.
        model.eval()

        # The processor (tokenizer + image pre-processing) belongs to the base model;
        # the adapter only changes weights, not the input format.
        processor = Qwen2VLProcessor.from_pretrained(BASE_MODEL_ID)

        logger.info("Model ready")
        return cls(model=model, processor=processor)

    def answer(self, image: Image.Image, question: str) -> tuple[str, int]:
        """Generate an answer about a food image.

        Args:
            image: The RGB image to describe.
            question: The question to ask about the image, in Turkish.

        Returns:
            A tuple of the generated answer and the elapsed time in milliseconds.
        """
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question},
                ],
            }
        ]
        # Wraps the question in the special tokens the instruction-tuned model was
        # trained on, and inserts the image placeholder in the right position.
        text_prompt = self._processor.apply_chat_template(
            conversation, add_generation_prompt=True
        )

        inputs = self._processor(
            text=[text_prompt], images=[image], padding=True, return_tensors="pt"
        )

        started = time.perf_counter()
        with self._lock:
            # inference_mode() skips autograd bookkeeping entirely: no gradients are
            # ever needed here, and tracking them would waste both time and memory.
            with torch.inference_mode():
                output_ids = self._model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    # Greedy decoding: the same image and question always produce the
                    # same answer, which makes the stored history reproducible.
                    do_sample=False,
                )
        duration_ms = int((time.perf_counter() - started) * 1000)

        # generate() returns the prompt followed by the completion; strip the prompt
        # so that only the newly generated tokens are decoded.
        generated_ids = [
            output[len(prompt) :]
            for prompt, output in zip(inputs.input_ids, output_ids)
        ]
        answer = self._processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0].strip()

        return answer, duration_ms
