# FlagOmni

[English](README.md) | 中文

## 简介

FlagOmni 是 [FlagOS](https://flagos.io/) 的一部分，FlagOS 是面向多种 AI 芯片的开源系统软件栈，通过软件连接模型、系统和芯片，降低跨硬件平台的模型移植和维护成本。

FlagOmni 是一个使用 [Triton](https://github.com/triton-lang/triton) 实现的全模态生成模型（视频、图像、音频）算子库。它面向主流的生成和推理框架，提供可复用的计算实现，可针对多种硬件后端进行优化以加速生成任务。

对于生成框架的开发者，FlagOmni 旨在提供与相关接口和计算语义对齐的 Triton 实现，减少重复开发和硬件适配工作。对于算子开发者，FlagOmni 提供围绕真实生成任务组织实现、测试和性能优化的协作平台。

本项目正在积极开发中：文档和组件实现会持续增加，组件和硬件兼容性尚未完全验证。

## 安装

```bash
uv pip install -e .                  # 运行时
uv pip install -e ".[test]"          # 开发和测试
```

## 设计目标

- **Triton 实现生成计算**：为生成模型推理中的前向计算开发可复用的算子，包括归一化、激活函数、位置编码和融合内核等。
- **融合与性能优化**：针对生成任务优化算子融合、内存访问和并行计算，减少中间数据传输和执行开销。
- **多硬件后端**：基于兼容的 Triton 编译和运行环境，在不同硬件后端上适配功能并调优性能。
- **上游协作**：与目标生成框架的接口、计算语义和使用需求对齐，并跟进上游变化。
- **可验证的实现**：为算子提供正确性测试和性能基准，明确适用范围和验证条件。

## 框架覆盖范围

| 框架 | 关注领域 | 状态 |
|------|----------|------|
| LightX2V | 视频/图像生成推理算子（RMS 归一化、SwiGLU 激活、VAE 融合、旋转位置编码） | 🚧 开发中 |
| FastVideo | 统一视频生成推理与后训练算子 | 🔮 规划中 |
| xDiT | 可扩展 Diffusion Transformer 推理算子 | 🔮 规划中 |

以上框架构成初始覆盖范围。各算子的支持情况将按算子、数据类型和硬件后端记录在兼容性说明中。

## 与相关项目的关系

FlagOmni 与 FlagOS 生态中的算子库、编译器和推理框架协作开发，各有明确职责：

- **[FlagGems](https://github.com/flagos-ai/FlagGems)**：提供面向 LLM 推理的通用算子实现。FlagOmni 优先与 FlagGems 协作复用基础算子，减少重复实现。
- **FlagOmni**：专注于全模态生成模型的算子和融合计算，组织其实现、测试和性能优化。
- **生成框架及上游组件**：提供接口和计算语义的用例和参考。FlagOmni 为其关键计算路径开发和验证 Triton 实现。

## 目录结构

```text
FlagOmni/
├── src/flag_omni/
│   ├── lightx2v/                   # LightX2V 通用 Triton 算子（归一化、激活、融合…）
│   ├── runtime/
│   │   └── backend/                # 多硬件后端：厂商 → 框架
│   │       ├── _nvidia/
│   │       │   └── lightx2v/       # NVIDIA 硬件特化（SM90 Gluon/PTX）
│   │       └── _mthreads/
│   │           └── lightx2v/       # 摩尔线程硬件特化（TLE）
│   ├── testing/                    # 共享测试工具与 fixture
│   └── utils/                      # 设备检测、配置加载、类型转换
├── tests/
│   └── lightx2v/                   # 合约测试与数值测试
├── benchmark/
│   └── lightx2v/                   # 时延与吞吐量基准
├── conf/
│   └── operators.yaml              # 算子注册表元数据
├── docs/                           # 文档
├── tools/                          # CI 脚本、GPU 检测、代码生成
├── CONTRIBUTING.md
├── SECURITY.md
├── LICENSE
└── pyproject.toml
```

通用算子放在 `src/flag_omni/<框架>/`，硬件特化放在 `runtime/backend/_<厂商>/<框架>/`，各框架与各硬件后端的代码完全隔离。

### 新增框架

以 FastVideo 为例，新增一个生成框架只需以下步骤：

1. **通用算子**：创建 `src/flag_omni/fastvideo/` 目录，放置该框架的 Triton 算子实现
2. **硬件特化**：在每个需要特化的硬件后端下创建对应子目录
   - `src/flag_omni/runtime/backend/_nvidia/fastvideo/`
   - `src/flag_omni/runtime/backend/_mthreads/fastvideo/`
3. **测试与基准**：创建对应的测试和性能基准目录
   - `tests/fastvideo/`
   - `benchmark/fastvideo/`
4. **算子注册**：在 `conf/operators.yaml` 中添加算子条目，使用 `from: [fastvideo]` 标注来源框架

新增框架不需要修改现有框架的代码，各框架的算子实现完全隔离。

## 开发与验证

每个算子实现应包含对应的参考实现、测试用例和性能评估方法。

### 正确性测试

- 覆盖有代表性的输入形状、数据类型和边界情况。
- 对照参考实现验证结果，指定数值误差容忍度。
- 记录测试所用的软件环境和硬件后端。

### 性能测试

- 在相同硬件、输入和精度条件下比较实现。
- 描述参考实现、预热过程和计时方法。
- 根据计算特点报告执行时间、吞吐量或设备内存使用情况。

## 贡献

欢迎贡献生成算子、框架适配、性能优化、测试和文档。

- 贡献流程请参阅[贡献指南](CONTRIBUTING.md)。
- 通过 [GitHub Issues](https://github.com/flagos-ai/FlagOmni/issues) 报告问题和建议功能。
- 安全问题请按照[安全报告说明](SECURITY.md)私下报告。

## 许可证

本项目基于 [Apache License 2.0](LICENSE) 许可。任何包含的第三方代码须保留其原始许可证和版权声明。
