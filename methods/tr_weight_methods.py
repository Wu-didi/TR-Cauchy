"""Baseline fusion methods and task-wise TR-Cauchy update adapters."""

from __future__ import annotations

import copy
import random
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from scipy.optimize import minimize

from methods.tr_cauchy import (
    TRCauchyConfig,
    TRCauchyController,
    apply_parameter_step,
    flatten_tensors,
)


Tensor = torch.Tensor
Parameter = torch.nn.Parameter


def _parameter_list(parameters) -> List[Parameter]:
    if parameters is None:
        return []
    if isinstance(parameters, torch.Tensor):
        return [parameters]
    return list(parameters)


def _task_gradient_tuples(
    losses: Tensor, shared_parameters: Sequence[Parameter]
) -> List[Tuple[Tensor, ...]]:
    gradients: List[Tuple[Tensor, ...]] = []
    for loss in losses:
        raw = torch.autograd.grad(
            loss,
            shared_parameters,
            retain_graph=True,
            allow_unused=True,
        )
        gradients.append(
            tuple(
                torch.zeros_like(parameter) if gradient is None else gradient.detach().clone()
                for parameter, gradient in zip(shared_parameters, raw)
            )
        )
    return gradients


def _task_gradient_matrix(task_gradients: Sequence[Sequence[Tensor]]) -> Tensor:
    return torch.stack([flatten_tensors(gradients) for gradients in task_gradients])


def _set_task_specific_gradients(
    losses: Tensor, task_specific_parameters: Sequence[Parameter]
) -> None:
    if not task_specific_parameters:
        return
    gradients = torch.autograd.grad(
        losses.sum(), task_specific_parameters, allow_unused=True
    )
    for parameter, gradient in zip(task_specific_parameters, gradients):
        parameter.grad = None if gradient is None else gradient.detach().clone()


def _write_flat_gradient(parameters: Sequence[Parameter], gradient: Tensor) -> None:
    offset = 0
    for parameter in parameters:
        count = parameter.numel()
        parameter.grad = gradient[offset : offset + count].view_as(parameter).clone()
        offset += count
    if offset != gradient.numel():
        raise ValueError("gradient shape does not match parameters")


class _BaseWeightMethod:
    def __init__(self, n_tasks: int, device: torch.device, max_norm: float = 1.0):
        self.n_tasks = n_tasks
        self.device = torch.device(device)
        self.max_norm = max_norm

    def parameters(self):
        return []


class PCGrad(_BaseWeightMethod):
    """Original PCGrad projection and output semantics."""

    def __init__(
        self,
        n_tasks: int,
        device: torch.device,
        reduction: str = "sum",
        max_norm: float = 1.0,
    ) -> None:
        super().__init__(n_tasks, device, max_norm=max_norm)
        if reduction not in {"mean", "sum"}:
            raise ValueError("reduction must be 'mean' or 'sum'")
        self.reduction = reduction

    def get_weighted_loss(self, *args, **kwargs):
        del args, kwargs
        raise NotImplementedError("PCGrad writes projected gradients in backward()")

    def project_conflicting(
        self, gradients: List[Tuple[Tensor, ...]]
    ) -> List[Tensor]:
        """Apply the unchanged PCGrad random projection rule."""

        source = list(gradients)
        projected = copy.deepcopy(source)
        for gradient_i in projected:
            random.shuffle(source)
            for gradient_j in source:
                inner = sum(
                    torch.dot(left.flatten(), right.flatten())
                    for left, right in zip(gradient_i, gradient_j)
                )
                if inner < 0:
                    norm_square = flatten_tensors(gradient_j).norm().square()
                    if norm_square > 0:
                        for left, right in zip(gradient_i, gradient_j):
                            left.sub_(inner * right / norm_square)
        merged = [sum(values) for values in zip(*projected)]
        if self.reduction == "mean":
            merged = [value / self.n_tasks for value in merged]
        return merged

    def fuse_task_gradients(
        self, task_gradients: List[Tuple[Tensor, ...]]
    ) -> Tensor:
        return flatten_tensors(self.project_conflicting(task_gradients))

    # Backwards-compatible private name used by some downstream code.
    _project_conflicting = project_conflicting

    def _set_pc_grads(
        self, losses: Tensor, shared_parameters, task_specific_parameters=None
    ) -> None:
        shared = _parameter_list(shared_parameters)
        task_specific = _parameter_list(task_specific_parameters)
        fused = self.fuse_task_gradients(_task_gradient_tuples(losses, shared))
        _set_task_specific_gradients(losses, task_specific)
        _write_flat_gradient(shared, fused)

    def backward(
        self,
        losses: Tensor,
        shared_parameters=None,
        task_specific_parameters=None,
        **kwargs,
    ):
        del kwargs
        shared = _parameter_list(shared_parameters)
        self._set_pc_grads(losses, shared, task_specific_parameters)
        if self.max_norm > 0:
            torch.nn.utils.clip_grad_norm_(shared, self.max_norm)
        return None, {}


class CAGrad(_BaseWeightMethod):
    """Original CAGrad optimizer with explicit ``c`` and ``rescale`` config."""

    def __init__(
        self,
        n_tasks: int,
        device: torch.device,
        c: float = 0.4,
        rescale: int = 1,
        max_norm: float = 1.0,
    ) -> None:
        super().__init__(n_tasks, device, max_norm=max_norm)
        self.c = c
        self.alpha = c
        self.rescale = rescale

    def cagrad(self, grads: Tensor, alpha: float, rescale: int):
        """Fuse a ``[D, T]`` matrix; identical code serves baseline and TR."""

        gram = grads.t().mm(grads).detach().cpu()
        mean_norm = (gram.mean() + 1e-8).sqrt()
        start = np.ones(self.n_tasks) / self.n_tasks
        bounds = tuple((0, 1) for _ in start)
        constraint = {"type": "eq", "fun": lambda weights: 1 - sum(weights)}
        matrix = gram.numpy()
        mean_weights = start.copy()
        coefficient = (alpha * mean_norm + 1e-8).item()

        def objective(weights):
            weights = weights.reshape(1, self.n_tasks)
            first = weights.dot(matrix).dot(mean_weights.reshape(self.n_tasks, 1))
            second = coefficient * np.sqrt(
                weights.dot(matrix).dot(weights.reshape(self.n_tasks, 1)) + 1e-8
            )
            return (first + second).sum()

        result = minimize(
            objective, start, bounds=bounds, constraints=constraint
        )
        weights_cpu = result.x
        weights = torch.as_tensor(
            weights_cpu, device=grads.device, dtype=grads.dtype
        )
        weighted = (grads * weights.view(1, -1)).sum(1)
        coefficient_tensor = grads.new_tensor(coefficient)
        fused = grads.mean(1) + coefficient_tensor / (weighted.norm() + 1e-8) * weighted
        if rescale == 1:
            fused = fused / (1 + alpha**2)
        elif rescale != 0:
            fused = fused / (1 + alpha)
        return fused, gram.numpy(), weights_cpu

    def fuse_task_gradients(self, task_gradient_matrix: Tensor):
        """Return the exact flat gradient written by baseline CAGrad."""

        fused, gram, weights = self.cagrad(
            task_gradient_matrix.t(), alpha=self.c, rescale=self.rescale
        )
        return fused * self.n_tasks, gram, weights

    def get_weighted_loss(self, losses: Tensor, shared_parameters, **kwargs):
        del kwargs
        shared = _parameter_list(shared_parameters)
        matrix = _task_gradient_matrix(_task_gradient_tuples(losses, shared))
        fused, gram, weights = self.fuse_task_gradients(matrix)
        _write_flat_gradient(shared, fused)
        return gram, weights

    def backward(
        self,
        losses: Tensor,
        shared_parameters=None,
        task_specific_parameters=None,
        **kwargs,
    ):
        del kwargs
        shared = _parameter_list(shared_parameters)
        task_specific = _parameter_list(task_specific_parameters)
        gram, weights = self.get_weighted_loss(losses, shared)
        _set_task_specific_gradients(losses, task_specific)
        if self.max_norm > 0:
            torch.nn.utils.clip_grad_norm_(shared, self.max_norm)
        return None, {"GTG": gram, "weights": weights}


class _TRCauchyAdapter:
    """Shared TR behavior; subclasses only provide their baseline fusion."""

    def _init_tr_controller(
        self,
        *,
        n_tasks: int,
        device: torch.device,
        tr_mode: str,
        delta0: float,
        delta_min: float,
        delta_max: float,
        eta1: float,
        eta2: float,
        gamma_dec: float,
        gamma_inc: float,
        curvature_mode: str,
        damping: float,
    ) -> None:
        self.tr_config = TRCauchyConfig(
            mode=tr_mode,
            delta0=delta0,
            delta_min=delta_min,
            delta_max=delta_max,
            eta1=eta1,
            eta2=eta2,
            gamma_dec=gamma_dec,
            gamma_inc=gamma_inc,
            curvature_mode=curvature_mode,
            damping=damping,
        )
        self.tr_controller = TRCauchyController(
            n_tasks=n_tasks, device=device, config=self.tr_config
        )
        # Kept as a short alias for checkpoint consumers, not a second controller.
        self.tr = self.tr_controller
        self.last_proposal = None
        self.last_result = None

    @property
    def requires_candidate_evaluator(self) -> bool:
        return self.tr_config.mode in {"adaptive_tr", "fixed_tr"}

    def state_dict(self) -> Dict[str, object]:
        return {"tr_controller": self.tr_controller.state_dict()}

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        self.tr_controller.load_state_dict(state["tr_controller"])

    def _fuse_for_tr(
        self,
        task_gradients: List[Tuple[Tensor, ...]],
        task_gradient_matrix: Tensor,
    ):
        raise NotImplementedError

    def _tr_backward(
        self,
        losses: Tensor,
        shared_parameters,
        task_specific_parameters,
        candidate_evaluator,
    ):
        shared = _parameter_list(shared_parameters)
        task_specific = _parameter_list(task_specific_parameters)
        if not shared:
            raise ValueError("TR-Cauchy requires explicit shared_parameters")

        task_gradients = _task_gradient_tuples(losses, shared)
        matrix = _task_gradient_matrix(task_gradients)
        fused, fusion_diagnostics = self._fuse_for_tr(task_gradients, matrix)
        _set_task_specific_gradients(losses, task_specific)

        proposal = self.tr_controller.propose(matrix, fused)
        self.last_proposal = proposal
        diagnostics: Dict[str, object] = {
            **fusion_diagnostics,
            "mode": self.tr_config.mode,
            "task_grad_norms": matrix.norm(dim=1).detach(),
            "fused_grad": fused.detach().clone(),
            "step": proposal.step.detach().clone(),
            "step_norm": proposal.step.norm().detach(),
            "task_radii_before": self.tr_controller.task_radii.detach().clone(),
        }

        if self.requires_candidate_evaluator:
            if candidate_evaluator is None:
                raise ValueError(
                    "adaptive_tr/fixed_tr require a paired same-batch "
                    "candidate_evaluator"
                )
            if not hasattr(candidate_evaluator, "validate_shared_parameters"):
                raise TypeError(
                    "candidate_evaluator must validate its shared-parameter order"
                )
            candidate_evaluator.validate_shared_parameters(shared)
            current_losses, candidate_losses = candidate_evaluator.evaluate(
                proposal.step
            )
            result = self.tr_controller.observe_and_update(
                proposal, current_losses, candidate_losses
            )
            self.last_result = result
            diagnostics.update(result.diagnostics())
        else:
            self.last_result = None
            diagnostics.update(
                {
                    "cosine_scale": proposal.cosine_scale.detach(),
                    "task_radii_after": self.tr_controller.task_radii.detach().clone(),
                }
            )

        # Algorithm 1 always commits the proposal.  Shared .grad remains None,
        # so the outer Adam/SGD step can update task heads but cannot transform
        # the already committed shared displacement.
        apply_parameter_step(shared, proposal.step)
        diagnostics["committed_step_norm"] = proposal.step.norm().detach()
        return None, diagnostics


class TRPCGrad(_TRCauchyAdapter, PCGrad):
    def __init__(
        self,
        n_tasks: int,
        device: torch.device,
        reduction: str = "sum",
        tr_mode: str = "adaptive_tr",
        delta0: float = 1e-2,
        delta_min: float = 1e-4,
        delta_max: float = 1.0,
        eta1: float = 0.1,
        eta2: float = 0.8,
        gamma_dec: float = 0.8,
        gamma_inc: float = 1.5,
        curvature_mode: str = "identity",
        damping: float = 1e-8,
    ) -> None:
        PCGrad.__init__(
            self, n_tasks=n_tasks, device=device, reduction=reduction, max_norm=0.0
        )
        self._init_tr_controller(
            n_tasks=n_tasks,
            device=device,
            tr_mode=tr_mode,
            delta0=delta0,
            delta_min=delta_min,
            delta_max=delta_max,
            eta1=eta1,
            eta2=eta2,
            gamma_dec=gamma_dec,
            gamma_inc=gamma_inc,
            curvature_mode=curvature_mode,
            damping=damping,
        )

    def _fuse_for_tr(self, task_gradients, task_gradient_matrix):
        del task_gradient_matrix
        return self.fuse_task_gradients(task_gradients), {}

    def backward(
        self,
        losses: Tensor,
        shared_parameters=None,
        task_specific_parameters=None,
        candidate_evaluator=None,
        **kwargs,
    ):
        del kwargs
        return self._tr_backward(
            losses,
            shared_parameters,
            task_specific_parameters,
            candidate_evaluator,
        )


class TRCAGrad(_TRCauchyAdapter, CAGrad):
    def __init__(
        self,
        n_tasks: int,
        device: torch.device,
        c: float = 0.4,
        rescale: int = 1,
        tr_mode: str = "adaptive_tr",
        delta0: float = 1e-2,
        delta_min: float = 1e-4,
        delta_max: float = 1.0,
        eta1: float = 0.1,
        eta2: float = 0.8,
        gamma_dec: float = 0.8,
        gamma_inc: float = 1.5,
        curvature_mode: str = "identity",
        damping: float = 1e-8,
    ) -> None:
        CAGrad.__init__(
            self,
            n_tasks=n_tasks,
            device=device,
            c=c,
            rescale=rescale,
            max_norm=0.0,
        )
        self._init_tr_controller(
            n_tasks=n_tasks,
            device=device,
            tr_mode=tr_mode,
            delta0=delta0,
            delta_min=delta_min,
            delta_max=delta_max,
            eta1=eta1,
            eta2=eta2,
            gamma_dec=gamma_dec,
            gamma_inc=gamma_inc,
            curvature_mode=curvature_mode,
            damping=damping,
        )

    def _fuse_for_tr(self, task_gradients, task_gradient_matrix):
        del task_gradients
        fused, gram, weights = self.fuse_task_gradients(task_gradient_matrix)
        return fused, {"GTG": gram, "weights": weights}

    def backward(
        self,
        losses: Tensor,
        shared_parameters=None,
        task_specific_parameters=None,
        candidate_evaluator=None,
        **kwargs,
    ):
        del kwargs
        return self._tr_backward(
            losses,
            shared_parameters,
            task_specific_parameters,
            candidate_evaluator,
        )
