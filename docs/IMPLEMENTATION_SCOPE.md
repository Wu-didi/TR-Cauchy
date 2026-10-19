# 实现说明

## 更新流程

对共享参数计算各任务梯度 $g_i$，使用 PCGrad 或 CAGrad 得到融合方向 $\bar g$。TR 控制器维护各任务半径，以其最小值作为当前全局半径 $\Delta$，并在 `B = I` 下产生 Cauchy 位移：

$$
s=-\alpha\bar g,\qquad
\alpha=\min\left(1,\frac{\Delta}{\|\bar g\|_2}\right).
$$

该表达式适用于有限且范数高于数值零阈值的融合梯度。范数不超过该阈值及非有限数值由代码单独处理。当前未实现 Adam 对角曲率、Gauss–Newton 或经验 Fisher 曲率。

每个任务使用自己的梯度计算预测下降：

$$
\mathrm{pred}_i=-g_i^\mathsf{T}s-\frac12\|s\|_2^2.
$$

实际下降由同一 mini-batch 上配对重算的当前损失与候选损失之差得到；在预测下降有效时，计算 $\rho_i=\mathrm{ared}_i/\mathrm{pred}_i$。`adaptive_tr` 按以下规则更新各任务下一次的半径，并约束在上下界内：

- 比率低于 `eta1` 或预测/观测无效时收缩。
- 有效比率高于 `eta2` 时扩张。
- 其余情况保持；被判为零梯度的提议保持原半径。

这里“有效”要求相关损失、实际下降及预测下降有限，且 `pred_i > epsilon`（默认 `epsilon=1e-12`），除法结果也必须有限。负预测下降并不是仅在浮点异常下出现：即使任务梯度正交，融合方向上的有限步也可能使某个任务的二次模型上升。本实现将这种任务比率标为无效，并在 `adaptive_tr` 的非零方向分支收缩下一轮半径；这可能与直接计算负分母比值的规则产生不同结果。更新后的半径截断至可配置的 `[delta_min, delta_max]`，默认 `[1e-4, 1]`。

正常有限数值下，候选位移直接提交，**不存在根据 $\rho_i$ 拒绝或撤销当前位移的分支**。数值保护不等同于经典信赖域的接受/拒绝机制。

## 共享参数与任务头

共享参数直接接收 Cauchy 位移，其 `.grad` 不交给外层优化器再次更新。任务专用参数保留对应损失的梯度，由外层优化器更新。因此 Adam 的预条件和 `--lr` 不会再次乘到共享 Cauchy 位移上；使用相同外层优化器名称不表示共享参数的实际更新规则相同。

上述分组以模型传入的参数列表为准。NYUv2/Cityscapes 的 MTAN 按参数名是否含 `pred` 分组：TR 更新组除了共享骨干，还包括任务专属的 `encoder_att` / `decoder_att` 模块；只有最终预测层进入外层优化器的任务头更新组。因此这里的 `shared_parameters()` 接口名不表示其中每个参数都被所有任务共享。Toy 的全部优化变量均在 TR 更新组，外层 Adam 不改变这些变量的 Cauchy 位移。

PCGrad 默认对投影后的任务梯度求和。CAGrad 默认使用 `c=0.4, rescale=1`，融合结果除以 `1+c²` 后再乘任务数。TR 与对应基线调用相同的融合函数；基线随后还会执行梯度裁剪和外层优化器更新，TR 则由 Cauchy 规则控制位移，不能将两者称为相同的优化器更新。

## 候选评估

神经网络使用 `FunctionalCandidateEvaluator` 和 `torch.func.functional_call` 重算当前/候选损失。评估与产生梯度的前向配对随机状态，使用独立缓冲区，避免候选前向污染模型的 BatchNorm 状态；还检查共享参数顺序和损失向量形状。toy 问题使用 `TensorCandidateEvaluator`。

当前/候选损失评估时，最终预测头参数保持固定；任务头的外层优化器更新发生在此后。因此比率衡量的是 TR 参数组位移的效果，不包含随后任务头更新的效果。

## 模式与默认设置

| 模式 | 行为 |
|---|---|
| `adaptive_tr` | Cauchy 位移、候选评估、逐任务半径更新 |
| `fixed_tr` | 相同 Cauchy 位移与诊断，半径保持固定 |
| `cosine` | 以平均有限余弦一致性缩放融合方向，缩放值限制在 `[0, 1]`；作为独立消融 |

默认 `delta0=0.01`、`delta_min=0.0001`、`delta_max=1.0`、`eta1=0.1`、`eta2=0.8`、`gamma_dec=0.8`、`gamma_inc=1.5`。全部命令行含义见 [使用说明](USAGE.md)。实验中显式覆盖默认值时，应记录实际配置。

## 实验与理论范围

本项目提供核心实现、合成示例和 NYUv2、Cityscapes、CelebA、QM9、toy 训练接入。BDD100K/YOLOP、PEMS04 和多任务强化学习不在当前包内；完整论文实验还需要相应数据、配置、训练结果及统计流程。

代码行为测试不代替收敛证明。条件下降及驻点结论仍依赖理论中明确给出的光滑性、方向、步长和下界条件；PCGrad/CAGrad 的输出不能自动当作精确 MGDA 最小范数解。旧启发式 TR 的结果也不能代替本实现的训练、时间或显存结果。

## 关键入口

- [`methods/tr_cauchy.py`](../methods/tr_cauchy.py)：Cauchy 提议、半径控制器、候选评估器。
- [`methods/tr_weight_methods.py`](../methods/tr_weight_methods.py)：基础融合及共享位移提交。
- [`methods/weight_methods.py`](../methods/weight_methods.py)：方法注册和状态导出。
- [`examples/train_synthetic.py`](../examples/train_synthetic.py)：最小训练接入。
- [`tests/`](../tests/)：针对性行为验证。
