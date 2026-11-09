# FlagOmni

[English](README.md) | 中文

## 简介

FlagOmni 是 [FlagOS](https://flagos.io/) 的一部分，FlagOS 是面向多种 AI 芯片的开源系统软件栈，通过软件连接模型、系统和芯片，降低跨硬件平台的模型移植和维护成本。

FlagOmni 是一个使用 [Triton](https://github.com/triton-lang/triton) 实现的全模态生成模型（视频、图像、音频）算子库。它面向主流的生成和推理框架，提供可复用的计算实现，可针对多种硬件后端进行优化以加速生成任务。

对于生成框架的开发者，FlagOmni 旨在提供与相关接口和计算语义对齐的 Triton 实现，减少重复开发和硬件适配工作。对于算子开发者，FlagOmni 提供围绕真实生成任务组织实现、测试和性能优化的协作平台。

本项目正在积极开发中：文档和组件实现会持续增加，组件和硬件兼容性尚未完全验证。

## 支持的框架

| 框架 | 状态 | 说明 |
|------|------|------|
| LightX2V | 🚧 开发中 | 轻量级视频/图像生成推理框架 |
| FastVideo | 🔮 规划中 | 统一视频生成推理与后训练框架 |
| xDiT | 🔮 规划中 | 可扩展的 Diffusion Transformer 推理引擎 |

## 安装

```bash
uv pip install -e .                  # 运行时
uv pip install -e ".[test]"          # 开发和测试
```

## 与其他 FlagOS 项目的关系

- **FlagGems**：LLM 推理通用算子库
- **FlagGems-vllm / FlagGems-sglang**：特定 LLM 推理框架的算子库
- **FlagOmni**：全模态生成模型算子库——与 FlagGems 互补

## 许可证

Apache License 2.0
