# Self-contained training example

After installing the core package (for example, `python -m pip install -e .`
from the project root), run:

```bash
python -m examples.train_synthetic --steps 3
python -m examples.train_synthetic --method trcagrad --tr-mode fixed_tr --steps 3
python -m examples.train_synthetic --method trpcgrad --tr-mode cosine --steps 3
```

The example generates 32 four-dimensional inputs and two regression targets in
memory. It trains a small shared neural network with dropout and two independent
task heads. It requires no dataset download, credentials, tracking service, or
private file. CPU is the default; `--device cuda` or `--device mps` selects an
available accelerator. `--seed` defaults to 7 and seeds Python, NumPy, and PyTorch.
Identical seeds support repeatable CPU checks in the same environment; they do
not promise identical results across devices or library versions.

The training loop demonstrates the public `WeightMethods` and
`FunctionalCandidateEvaluator` APIs:

1. Capture the model's forward state **before** computing the task losses.
2. For `adaptive_tr` and `fixed_tr`, evaluate the original and candidate shared
   parameters with the same generated batch, fixed task heads, and restored
   dropout RNG state. `cosine` does not perform this ratio evaluation.
3. Call `WeightMethods.backward`. The TR adapter commits the shared-parameter
   displacement directly and clears shared gradients.
4. Call the outer Adam optimizer. Although it contains all model parameters,
   shared parameters have no gradients and are skipped; only the task heads
   receive Adam updates, with learning rate `1e-3` and weight decay `0`.

The example uses the core controller's existing defaults, including `B = I`,
and prints them in its report. `adaptive_tr` uses task-wise ratios to adjust the
next radii, `fixed_tr` keeps the radii fixed, and `cosine` scales the fused
direction by angular agreement. There is no ratio-based rejection or rollback
of the current shared step. A task-head update occurs after the ratio
evaluation, so those ratios describe only the shared step with fixed heads.

Standard output is one JSON object containing configuration, initial and final
losses and radii, per-step diagnostics, and numerical/API checks. The
`evaluation_task_losses` fields always use the full generated batch with dropout
disabled, so the initial and final measurements are comparable. Paired losses
use the captured training-mode dropout state. Undefined ratios are represented
as JSON `null`, accompanied by `valid_ratios`; they are not silently reported as
successful decreases. The checks also verify that the outer optimizer preserves
the already committed shared parameters and that task heads actually update.

This is an executable API example, **not a reproduction of the paper's
benchmarks**. It does not require every task loss to decrease at every step and
does not establish a convergence or accuracy guarantee.

Run the targeted smoke tests with the development dependencies installed:

```bash
python -m pytest -q tests/test_synthetic_example.py
```
