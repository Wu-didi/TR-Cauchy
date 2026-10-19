# TR-Cauchy

基于 [FAMO](https://github.com/Cranial-XIX/FAMO) 的多任务优化实现，提供 **TR-PCGrad / TR-CAGrad**、无需数据集的合成示例，以及 NYUv2、Cityscapes 等训练入口。

## 快速开始

需要 Python 3.9+、PyTorch 2.0+。获取本项目源码后，在项目根目录执行：

```bash
python -m pip install -e '.[dev]'
python -m examples.train_synthetic --method trcagrad --steps 3
python -m pytest -q
```

合成示例无需下载数据集，可用于检查优化模块的接入。换用 `--method trpcgrad` 可运行 TR-PCGrad。GPU 实验请使用与设备兼容的 PyTorch 构建。已执行检查的环境和结果见 [VALIDATION.md](VALIDATION.md)。

## 功能

- **梯度融合**：保留 PCGrad/CAGrad 基础方向，使用 Cauchy 位移控制共享参数更新。
- **逐任务半径**：`adaptive_tr` 自适应更新半径；`fixed_tr` 固定半径；`cosine` 为独立角度缩放消融。
- **候选评估**：同一 mini-batch 上配对重算当前与候选损失，保护随机状态及 BatchNorm 缓冲区。
- **实验接入**：NYUv2、Cityscapes、CelebA、QM9 和 toy 示例；可选安装相应依赖。

当前曲率为 `B = I`。数值有效的候选步直接提交，下降比率只控制下一次半径；共享参数直接应用该位移，外层优化器更新任务专用参数。完整语义见 [实现说明](docs/IMPLEMENTATION_SCOPE.md)。

## 运行数据集实验

安装图像预测实验依赖后，先查看参数：

```bash
python -m pip install -e '.[experiments]'
python -m experiments.nyuv2.trainer --help
python -m experiments.cityscapes.trainer --help
```

[使用说明](docs/USAGE.md) 包含依赖分组、数据要求、训练命令和各参数作用。当前默认阈值为 `eta1=0.1`、`eta2=0.8`。训练需独立准备数据；这个版本不附带论文结果对应的训练权重或完整复现记录，也不包含 BDD100K/YOLOP、PEMS04 和多任务强化学习入口。

## 文档与贡献

| 内容 | 入口 |
|---|---|
| 安装、运行与参数 | [docs/USAGE.md](docs/USAGE.md) |
| 数学模型、更新流程及适用范围 | [docs/IMPLEMENTATION_SCOPE.md](docs/IMPLEMENTATION_SCOPE.md) |
| 英文代码包说明 | [PAPER_CODE_ATTACHMENT.md](PAPER_CODE_ATTACHMENT.md) |
| 问题反馈和贡献流程 | [CONTRIBUTING.md](CONTRIBUTING.md) |
| 版本变化 | [CHANGELOG.md](CHANGELOG.md) |
| 上游来源与许可 | [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) |

欢迎提交可复现的问题报告、修复和文档改进。修改算法行为时，请同时补充针对性测试，并说明对已有配置和结果的影响。

## 致谢与许可

本项目是 FAMO 的独立扩展，保留上游 [MIT 许可](LICENSE) 及版权声明。感谢 FAMO、CAGrad、Nash-MTL、MTAN 及相关上游实现；具体来源见 [第三方说明](THIRD_PARTY_NOTICES.md)。使用对应方法开展研究时，请引用其原始工作，包括 [Fast Adaptive Multitask Optimization](https://arxiv.org/abs/2306.03792)。
