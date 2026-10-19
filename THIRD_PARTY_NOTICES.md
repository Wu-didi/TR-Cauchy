# Third-party notices

TR-Cauchy is based on FAMO and retains code, adaptations, and method references from the upstream projects below. The original root [LICENSE](LICENSE) is unchanged. The copyright notices below supplement it; upstream source links document provenance, not authorship of the TR-Cauchy additions.

## Upstream sources

| Project | Relationship visible in this source tree | Upstream license |
|---|---|---|
| [FAMO](https://github.com/Cranial-XIX/FAMO) | Base project, method registry, and experiment structure | [MIT](https://github.com/Cranial-XIX/FAMO/blob/main/LICENSE) |
| [CAGrad](https://github.com/Cranial-XIX/CAGrad) | Fusion method; NYUv2/Cityscapes loaders and toy adaptations credited in source | [MIT](https://github.com/Cranial-XIX/CAGrad/blob/main/LICENSE) |
| [Nash-MTL](https://github.com/AvivNavon/nash-mtl) | Baseline and experiment conventions inherited through FAMO | [MIT](https://github.com/AvivNavon/nash-mtl/blob/main/LICENSE) |
| [MTAN](https://github.com/lorenmt/mtan) | Model/data conventions and dynamic-weighting adaptation credited in source | [MIT](https://github.com/lorenmt/mtan/blob/master/LICENSE) |
| [MultiObjectiveOptimization](https://github.com/isl-org/MultiObjectiveOptimization) | Minimum-norm solver and MGDA source references | [MIT](https://github.com/isl-org/MultiObjectiveOptimization/blob/master/LICENSE) |
| [multi-task-learning-example](https://github.com/yaringal/multi-task-learning-example) | Uncertainty-weighting source reference in `methods/weight_methods.py` | [MIT](https://github.com/yaringal/multi-task-learning-example/blob/master/LICENSE) |
| [PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric) | QM9 example adaptation credited in its experiment README | [MIT](https://github.com/pyg-team/pytorch_geometric/blob/master/LICENSE) |

These repository-level license pages were checked when preparing this package. The inherited snapshot does not identify every original upstream commit, so this table is not a complete file-by-file origin audit. Preserve any file-specific notices when updating or importing code. Installed dependencies and separately obtained datasets retain their own licenses and usage conditions.

## Retained upstream copyright notices

```text
FAMO: Copyright (c) 2023 Bo Liu
CAGrad: Copyright (c) 2022 Bo Liu
Nash-MTL: Copyright (c) 2022 Aviv Navon
MTAN: Copyright (c) 2020 Shikun Liu
MultiObjectiveOptimization: Copyright (c) 2018 Ozan Sener
multi-task-learning-example: Copyright (c) 2017
PyTorch Geometric: Copyright (c) 2023 PyG Team <team@pyg.org>
```

The unnamed 2017 notice is reproduced as it appears in that upstream license; no author name has been inferred. The MIT permission and warranty terms for the notices above are reproduced below.

```text
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
