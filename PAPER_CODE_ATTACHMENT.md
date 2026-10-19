# TR-Cauchy code package

TR-Cauchy extends the [FAMO codebase](https://github.com/Cranial-XIX/FAMO) with TR-PCGrad and TR-CAGrad. It is an independent extension and retains upstream attribution and license notices.

## Quick start

With Python 3.9+ and PyTorch 2.0+, run from the project root:

```bash
python -m pip install -e '.[dev]'
python -m examples.train_synthetic --method trcagrad --steps 3
python -m pytest -q
```

The synthetic example runs without an external dataset. See [VALIDATION.md](VALIDATION.md) for recorded checks and their environment, and [docs/USAGE.md](docs/USAGE.md) for dataset-specific installation and commands.

## Implemented behavior

- `trpcgrad` and `trcagrad` apply a Cauchy displacement to the corresponding fused shared-parameter gradient.
- The curvature model is explicitly `B = I`.
- `adaptive_tr` updates task-wise radii; `fixed_tr` keeps them fixed.
- `cosine` is a separate angle-scaling ablation, not TR-Cauchy.
- A numerically valid proposal is committed without a reduction-ratio rejection branch. Task-wise ratios control the next radii.
- Shared parameters receive the displacement directly. The outer optimizer updates task-specific parameters without rescaling or preconditioning the shared displacement.
- Neural-network candidate losses are evaluated on the same mini-batch through `torch.func.functional_call`, with paired RNG state and protected module modes and buffers.

Default thresholds are `eta1=0.1` and `eta2=0.8`. Full parameter definitions and implementation scope are documented in [docs/IMPLEMENTATION_SCOPE.md](docs/IMPLEMENTATION_SCOPE.md).

## Experimental scope and provenance

The package includes the core methods, tests, a synthetic example, and training integrations for NYUv2, Cityscapes, CelebA, QM9, and a toy problem. BDD100K/YOLOP, PEMS04, and multitask reinforcement-learning experiments are not included. Datasets and pretrained weights are not distributed with this package.

Software tests and short examples validate implementation behavior; they do not reproduce a manuscript table or establish unconditional convergence. To report an experiment, retain the source version, complete configuration, environment, data split, seeds, logs, and evaluation/aggregation procedure. Results from the former heuristic TR implementation must not be attributed to the present implementation.

## Source layout

| Path | Purpose |
|---|---|
| `methods/tr_cauchy.py` | Cauchy proposal, radius controller, candidate evaluators |
| `methods/tr_weight_methods.py` | PCGrad/CAGrad fusion and TR adapters |
| `methods/weight_methods.py` | Method registry and state delegation |
| `examples/train_synthetic.py` | Dataset-free integration example |
| `experiments/` | Dataset integrations and shared command-line settings |
| `tests/` | Behavioral and packaging checks |
| `docs/` | Public usage and implementation documentation |

## Distribution and contribution

```bash
python scripts/build_paper_attachment.py
```

The builder creates `TR_Cauchy_Code.zip` and prints its location. The package contains source, examples, tests, dependency declarations, public documentation, and license notices. Local datasets, training outputs, caches, and internal revision materials are excluded.

Contribution and test instructions are in [CONTRIBUTING.md](CONTRIBUTING.md). The original MIT license is preserved in [LICENSE](LICENSE); upstream sources and additional notices are recorded in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
