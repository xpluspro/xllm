# 多模态支持
本文档主要介绍xLLM推理引擎中多模态的支持进展，包括支持模型及模态类型，以及离在线接口等。

## 支持模型

以下模型注册为视觉语言（VLM）后端，支持图片输入：

| 模型系列 | `config.json` 中的 `model_type` | 典型规格 |
|---|---|---|
| Qwen2.5-VL | `qwen2_5_vl` | 7B / 32B / 72B |
| Qwen3-VL | `qwen3_vl` | 2B / 4B / 8B / 32B |
| Qwen3-VL-MoE | `qwen3_vl_moe` | A3B / A22B |
| MiniCPM-V-2_6 | `minicpmv` | 7B |

## 后端路由

xLLM 通过 `config.json` 中的 `model_type` 字段自动选择对应后端：

- **LLM 后端**（`--backend=llm`）：纯文本模型，如 Qwen3.5（`model_type: qwen3_5`）。
- **VLM 后端**（`--backend=vlm`）：视觉语言模型，如 Qwen3-VL（`model_type: qwen3_vl`）。
- **DiT 后端**（`--backend=dit`）：扩散模型。

不指定 `--backend` 时，xLLM 会从 `config.json` 自动推断后端类型。也可以显式指定：

```bash
xllm --model /path/to/Qwen3-VL-8B --backend=vlm ...
```

!!! warning "Qwen3.5-9B 不支持图片输入"
    `Qwen3.5-9B` 的 `model_type` 为 `qwen3_5`，是**纯文本** LLM 模型，**没有**视觉编码器，无论如何设置 `--backend` 标志，都**无法**处理图片。

    如需图片输入，必须使用专用的视觉语言模型，例如 **Qwen3-VL**（`model_type: qwen3_vl`）。

    | 模型 | `model_type` | 支持图片 |
    |---|---|---|
    | Qwen3.5-9B | `qwen3_5` | ❌ 不支持 |
    | Qwen3-VL-8B | `qwen3_vl` | ✅ 支持 |

## 模态类型
- 图片: 支持单图、多图的输入，以及图片+Prompt组合、纯文本Prompt等输入方式。


!!! warning "注意事项"
    - 目前多模态后端不支持prefix cache以及chunk prefill，正在支持中。
    - 目前，xLLM统一基于JinJa渲染ChatTemplate，部署MiniCPM-V-2_6，模型目录需提供ChatTemplate文件。
    - 图片支持Base64输入以及图片Url。
    - 目前多模态模型主要支持了图片模态，视频、音频等模态正在推进中。
    
