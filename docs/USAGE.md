# 使用说明

## 1. 安装与首次运行

在项目根目录，使用 Python 3.9+：

```bash
python -m pip install -e '.[dev]'
python -m examples.train_synthetic --method trcagrad --steps 3
python -m examples.train_synthetic --method trpcgrad --steps 3
python -m pytest -q
```

合成示例无需外部数据，展示共享网络与任务头如何接入 TR 更新。查看示例参数可运行 `python -m examples.train_synthetic --help`。

核心依赖为 PyTorch、NumPy、SciPy。仅使用方法核心时，`python -m pip install -e .` 即可安装所需依赖；数据集和开发工具按需安装：

| 可选组 | 用途 |
|---|---|
| `.[dev]` | 开发与测试 |
| `.[experiments]` | NYUv2/Cityscapes 等训练入口的公共依赖 |
| `.[toy]` | toy 实验与绘图 |
| `.[celeba]` | CelebA 图像与属性实验 |
| `.[qm9]` | QM9 / PyTorch Geometric 实验 |
| `.[baselines]` | 使用 CVXPY 的基线，例如 Nash-MTL |
| `.[profiling]` | 资源采样所需的 psutil |

可组合安装，例如 `python -m pip install -e '.[experiments,dev]'`。GPU 版本的 PyTorch、torchvision 和 PyTorch Geometric 相关算子需与实际环境匹配。依赖范围不是固定实验环境；记录结果时保存实际安装版本。

## 2. 模式与参数

下表对应 `experiments/utils.py` 的公共训练参数。示例自身支持的参数以其 `--help` 为准。

| 参数 | 默认值 | 作用 |
|---|---|---|
| `--method` | 显式选择 | `trpcgrad` 或 `trcagrad`；也可选注册的基线 |
| `--tr-mode` | `adaptive_tr` | 自适应半径、`fixed_tr` 固定半径或 `cosine` 角度缩放消融 |
| `--tr-delta0` | `0.01` | 各任务的初始半径 |
| `--tr-delta-min` | `0.0001` | 半径下界 |
| `--tr-delta-max` | `1.0` | 半径上界 |
| `--tr-eta1` | `0.1` | 下降比率低于此值时收缩下一次半径 |
| `--tr-eta2` | `0.8` | 有效下降比率高于此值时扩张下一次半径 |
| `--tr-gamma-dec` | `0.8` | 半径收缩乘数，介于 0 和 1 |
| `--tr-gamma-inc` | `1.5` | 半径扩张乘数，大于 1 |
| `--tr-curvature` | `identity` | 当前仅实现单位曲率 `B = I` |
| `--tr-damping` | `1e-8` | 曲率接口预留的数值参数；单位曲率的 Cauchy 系数解析计算 |
| `--c` | `0.4` | CAGrad 融合参数，TR-CAGrad 使用相同设置 |
| `--cagrad-rescale` | `1` | CAGrad 方向缩放模式，可选 0、1、2 |
| `--seed` | `42` | 随机种子 |

有效阈值满足 `0 < eta1 < eta2 < 1`。增大 `eta1` 会使收缩更容易触发；减小 `eta2` 会使扩张更容易触发。下降比率不用于拒绝当前步。具体公式与数值边界见 [实现说明](IMPLEMENTATION_SCOPE.md)。

## 3. 数据集实验

| 入口 | 数据/任务 | 依赖组 |
|---|---|---|
| `experiments.nyuv2.trainer` | 预处理 NYUv2；语义、深度、法向 | `experiments` |
| `experiments.cityscapes.trainer` | 预处理 Cityscapes；语义、深度 | `experiments` |
| `experiments.celeba.trainer` | CelebA 图像和属性标签 | `celeba` |
| `experiments.quantum_chemistry.trainer` | QM9 分子性质 | `qm9` |
| `experiments.toy.trainer` | toy 多目标问题与绘图 | `toy` |

NYUv2/Cityscapes 沿用 FAMO/MTAN 的预处理数据格式。数据加载要求见各目录的 `data.py`、README 和上游 [FAMO](https://github.com/Cranial-XIX/FAMO)、[MTAN](https://github.com/lorenmt/mtan) 说明。QM9 入口通过 PyTorch Geometric 数据接口准备数据；数据本身的获取与使用遵循其发布条件。

安装依赖后，先运行帮助命令：

```bash
python -m pip install -e '.[experiments]'
python -m experiments.nyuv2.trainer --help
python -m experiments.cityscapes.trainer --help
```

单次 NYUv2 训练示例，将 `/path/to/nyuv2` 换成实际预处理目录：

```bash
python -m experiments.nyuv2.trainer \
  --data-path /path/to/nyuv2 \
  --method trcagrad --tr-mode adaptive_tr \
  --tr-eta1 0.1 --tr-eta2 0.8 --tr-curvature identity \
  --model mtan --seed 42 --n-epochs 300 --batch-size 2 --lr 0.0001
```

这是显式配置的运行示例，不是论文结果的复现命令。不同实验请分别保存输出、完整命令和环境。NYUv2 原始说明的最后 10 个 epoch 平均需另行聚合；论文对照应采用对应论文实际的评估规则。

## 4. 接入自己的模型

先参考 [合成训练示例](../examples/train_synthetic.py)。接入时需要：

1. 划分共享参数和任务专用参数；两组参数不应重复。
2. 在产生梯度的前向之前捕获候选评估所需状态。
3. 计算各任务损失，并为同一 mini-batch 构造候选损失闭包。
4. 调用方法的 `backward`，由 TR 适配器直接提交共享位移。
5. 调用外层优化器更新任务专用参数。

`WeightMethods.state_dict()` 可导出 TR 控制器配置与半径。断点恢复还需要模型、优化器、调度器和随机状态等信息；按当前训练入口的实际保存内容核实。统计指标文件不能替代完整检查点。

## 5. 测试与反馈

运行 `python -m pytest -q` 检查软件行为。实际验证记录见 [VALIDATION.md](../VALIDATION.md)。报告问题时提供最小示例、完整命令、报错以及 Python/PyTorch/设备信息；贡献步骤见 [CONTRIBUTING.md](../CONTRIBUTING.md)。
