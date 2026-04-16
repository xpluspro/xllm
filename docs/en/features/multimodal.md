# Multimodal Support

This document describes multimodal (image input) support in xLLM, including supported models, how backend routing works, and how to enable image input.

## Supported Models

The following models are registered as vision-language (VLM) backends and support image input:

| Model family | `model_type` in `config.json` | Typical variants |
|---|---|---|
| Qwen2.5-VL | `qwen2_5_vl` | 7B / 32B / 72B |
| Qwen3-VL | `qwen3_vl` | 2B / 4B / 8B / 32B |
| Qwen3-VL-MoE | `qwen3_vl_moe` | A3B / A22B |
| MiniCPM-V-2_6 | `minicpmv` | 7B |

## Backend Routing

xLLM uses the `model_type` field in `config.json` to automatically select the correct backend:

- **LLM backend** (`--backend=llm`): text-only models such as Qwen3.5 (`model_type: qwen3_5`).
- **VLM backend** (`--backend=vlm`): vision-language models such as Qwen3-VL (`model_type: qwen3_vl`).
- **DiT backend** (`--backend=dit`): diffusion models.

The backend is detected automatically from `config.json` when `--backend` is not specified. You can also set it explicitly:

```bash
xllm --model /path/to/Qwen3-VL-8B --backend=vlm ...
```

!!! warning "Qwen3.5-9B does not support image input"
    `Qwen3.5-9B` has `model_type: qwen3_5` and is a **text-only** LLM. It does **not** have a vision encoder and cannot process images, regardless of the `--backend` flag.

    To use image input, you must use a dedicated vision-language model such as **Qwen3-VL** (`model_type: qwen3_vl`).

    | Model | `model_type` | Image support |
    |---|---|---|
    | Qwen3.5-9B | `qwen3_5` | ❌ No |
    | Qwen3-VL-8B | `qwen3_vl` | ✅ Yes |

## Launch Example

To start xLLM with a VLM model that supports image input:

```bash
xllm \
  --model /path/to/Qwen3-VL-8B \
  --backend=vlm \
  --devices="npu:0" \
  --port 12345 \
  --master_node_addr=127.0.0.1:9748
```

## API Usage

Once the VLM service is running, send image-bearing requests via the `/v1/chat/completions` endpoint using the OpenAI multimodal format:

```python
import base64
import requests

api_url = "http://localhost:12345/v1/chat/completions"

with open("/path/to/image.jpg", "rb") as f:
    image_base64 = base64.b64encode(f.read()).decode("utf-8")

payload = {
    "model": "Qwen3-VL-8B",
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image."},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
                },
            ],
        }
    ],
    "max_completion_tokens": 256,
}

response = requests.post(api_url, json=payload,
                         headers={"Content-Type": "application/json"})
print(response.json())
```

!!! note "VLM limitations"
    - The VLM backend currently disables prefix cache and chunked prefill.
    - Images can be supplied as Base64-encoded data or as HTTP image URLs.
    - Video and audio modalities are not yet supported.
