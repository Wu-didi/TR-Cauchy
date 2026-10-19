# 本版本验证记录

日期：2026-10-18。以下运行在独立的本地 macOS ARM64 CPU 环境完成，不是 3090 服务器的 GPU 训练验证。

## 已执行

| 检查 | 结果 |
|---|---|
| 完整测试 `python -m pytest -q` | 53 passed：41 核心、4 打包、8 合成示例测试 |
| 可编辑安装 `python -m pip install --no-deps --no-build-isolation -e .` | 成功；运行依赖已预先安装 |
| `python -m pip check` | No broken requirements found |
| TR-PCGrad Toy CLI | 5 个原有初始点，各 10 次迭代，退出码 0，生成 PNG |
| TR-CAGrad Toy CLI | 5 个原有初始点，各 10 次迭代，退出码 0，生成 PNG |
| NYUv2 / Cityscapes `--help` | 两个入口均正常退出 |
| 公开训练示例 | 两种融合方法 × 三种模式通过；同种子重复运行及非法步数检查通过 |
| 5 个启动脚本 | Bash 语法通过；隔离假执行器检查路径、参数透传及数据保留通过 |

Toy 检查使用默认 `adaptive_tr`，并设置 `MPLBACKEND=Agg`、`WANDB_MODE=disabled`。示例命令：

```bash
python -m experiments.toy.trainer --method trpcgrad --n-epochs 10 --out-path outputs/smoke-trpcgrad
python -m experiments.toy.trainer --method trcagrad --n-epochs 10 --out-path outputs/smoke-trcagrad
```

短程检查中发现并修复了旧 Matplotlib `tick.label` 接口导致的绘图错误。仍有继承代码的非阻断警告：`min_norm_solvers.py` 文档字符串的转义警告，以及 Toy 空图例警告。

## 分发与开源工程

- 已构建源码分发包（sdist）与 wheel，wheel 包含可运行示例、原 MIT 许可和第三方版权声明。
- 使用全新虚拟环境安装 wheel 及其声明的依赖，在源码目录之外运行合成示例；也完成了源码包的隔离构建安装与独立运行检查。
- 文档本地链接、GitHub YAML 和工作流 shell 语法检查通过。
- 已提供 Python 3.9/3.11/3.12 的 CI 配置。尚未在远程托管平台运行 CI；不将这些版本组合写成已验证。
- 本地实际验证版本如下，不能据此推断所有允许的依赖版本组合均已验证。

## 运行环境

Python 3.12.14；macOS-26.5.2-arm64-arm-64bit；CPU。

- `torch==2.14.1`
- `numpy==2.5.4`
- `scipy==1.18.1`
- `pytest==9.1.1`
- `matplotlib==3.11.2`
- `seaborn==0.13.2`
- `wandb==0.31.0`
- `tqdm==4.70.1`
- `setuptools==84.0.0`

## 验证边界

以上结果确认所列软件行为与短程入口，未执行真实数据集训练，未验证论文表格、收敛速度、显存或时间指标，也未验证所有可选基线及所有依赖版本组合。共享参数直接 Cauchy 更新、无比率拒步、阈值 0.1/0.8 等算法设置保持原副本语义。
