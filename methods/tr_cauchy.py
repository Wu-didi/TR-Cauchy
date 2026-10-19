"""TR-Cauchy primitives shared by TR-PCGrad and TR-CAGrad.

The implemented update rules are documented in ``docs/IMPLEMENTATION_SCOPE.md``.
Curvature is explicitly ``B = I``. Non-identity models would require an
implementation of :class:`CurvatureModel`, configuration support, and validation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import torch

try:
    from torch.func import functional_call
except ImportError:  # pragma: no cover - exercised only on old PyTorch
    functional_call = None


Tensor = torch.Tensor
LossClosure = Callable[[Callable[..., object]], Tensor]


@dataclass
class CandidateForwardState:
    """Model/RNG state captured immediately before the gradient forward."""

    model_id: int
    cpu_rng: Tensor
    cuda_rng: Optional[Tuple[Tensor, ...]]
    buffers: Dict[str, Tensor]
    training_flags: Dict[str, bool]

    @classmethod
    def capture(cls, model: torch.nn.Module) -> "CandidateForwardState":
        cuda_rng = None
        if torch.cuda.is_available():
            cuda_rng = tuple(state.clone() for state in torch.cuda.get_rng_state_all())
        return cls(
            model_id=id(model),
            cpu_rng=torch.random.get_rng_state().clone(),
            cuda_rng=cuda_rng,
            buffers={
                name: buffer.detach().clone()
                for name, buffer in model.named_buffers()
            },
            training_flags={
                name: module.training for name, module in model.named_modules()
            },
        )


@dataclass(frozen=True)
class TRCauchyConfig:
    """Serializable configuration for all TR-Cauchy adapters."""

    mode: str = "adaptive_tr"
    delta0: float = 1e-2
    delta_min: float = 1e-4
    delta_max: float = 1.0
    eta1: float = 0.1
    eta2: float = 0.8
    gamma_dec: float = 0.8
    gamma_inc: float = 1.5
    curvature_mode: str = "identity"
    damping: float = 1e-8
    epsilon: float = 1e-12

    def __post_init__(self) -> None:
        if self.mode not in {"adaptive_tr", "fixed_tr", "cosine"}:
            raise ValueError(
                "mode must be one of: adaptive_tr, fixed_tr, cosine"
            )
        if not (0.0 < self.eta1 < self.eta2 < 1.0):
            raise ValueError("expected 0 < eta1 < eta2 < 1")
        if not self.gamma_inc > 1.0:
            raise ValueError("gamma_inc must be greater than 1")
        if not (0.0 < self.gamma_dec < 1.0):
            raise ValueError("gamma_dec must satisfy 0 < gamma_dec < 1")
        if not (0.0 < self.delta_min <= self.delta0 <= self.delta_max):
            raise ValueError(
                "expected 0 < delta_min <= delta0 <= delta_max"
            )
        if self.curvature_mode != "identity":
            raise ValueError(
                "only the explicit IdentityCurvature (curvature_mode='identity') "
                "is implemented"
            )
        if self.damping <= 0.0 or self.epsilon <= 0.0:
            raise ValueError("damping and epsilon must be positive")


class CurvatureModel(ABC):
    """Interface for global and per-task positive-definite curvature models."""

    name: str

    @abstractmethod
    def apply(self, vector: Tensor, task_index: Optional[int] = None) -> Tensor:
        raise NotImplementedError

    def quadratic_form(
        self, vector: Tensor, task_index: Optional[int] = None
    ) -> Tensor:
        return torch.dot(vector, self.apply(vector, task_index=task_index))


class IdentityCurvature(CurvatureModel):
    """Explicit ``B = I`` curvature used by the current experiments."""

    name = "identity"

    def apply(self, vector: Tensor, task_index: Optional[int] = None) -> Tensor:
        del task_index
        return vector


@dataclass
class TRCauchyProposal:
    step: Tensor
    fused_grad: Tensor
    delta_global: Tensor
    alpha_curvature: Tensor
    alpha_boundary: Tensor
    alpha: Tensor
    predicted_decreases: Tensor
    valid_step: bool
    failure_reason: Optional[str] = None
    cosine_scale: Optional[Tensor] = None


@dataclass
class TRCauchyResult:
    proposal: TRCauchyProposal
    current_losses: Tensor
    candidate_losses: Tensor
    actual_decreases: Tensor
    rho: Tensor
    valid_ratios: Tensor
    task_radii_before: Tensor
    task_radii_after: Tensor
    accepted: bool = True

    def diagnostics(self) -> Dict[str, object]:
        """Detached diagnostics suitable for logging without default printing."""

        proposal = self.proposal
        return {
            "fused_grad": proposal.fused_grad.detach().clone(),
            "fused_grad_norm": proposal.fused_grad.norm().detach(),
            "step": proposal.step.detach().clone(),
            "step_norm": proposal.step.norm().detach(),
            "delta_global_before": proposal.delta_global.detach(),
            "delta_global_after": self.task_radii_after.min().detach(),
            "task_radii_before": self.task_radii_before.detach().clone(),
            "task_radii_after": self.task_radii_after.detach().clone(),
            "predicted_decreases": proposal.predicted_decreases.detach().clone(),
            "actual_decreases": self.actual_decreases.detach().clone(),
            "rho": self.rho.detach().clone(),
            "valid_ratios": self.valid_ratios.detach().clone(),
            "alpha_curvature": proposal.alpha_curvature.detach(),
            "alpha_boundary": proposal.alpha_boundary.detach(),
            "alpha": proposal.alpha.detach(),
            "accepted": self.accepted,
            "failure_reason": proposal.failure_reason,
        }


class TRCauchyController:
    """One Algorithm-1 controller with independent per-task radii.

    The current proposal is always committed by the adapter.  ``rho_i`` only
    changes the next iteration's radius, exactly as specified by Algorithm 1.
    """

    def __init__(
        self,
        n_tasks: int,
        device: torch.device,
        config: Optional[TRCauchyConfig] = None,
        curvature: Optional[CurvatureModel] = None,
    ) -> None:
        if n_tasks <= 0:
            raise ValueError("n_tasks must be positive")
        self.n_tasks = n_tasks
        self.device = torch.device(device)
        self.config = config or TRCauchyConfig()
        self.curvature = curvature or IdentityCurvature()
        if self.curvature.name != self.config.curvature_mode:
            raise ValueError("curvature model and curvature_mode do not match")
        self.task_radii = torch.full(
            (n_tasks,),
            self.config.delta0,
            device=self.device,
            dtype=torch.float64,
        )

    @property
    def delta_global(self) -> Tensor:
        return self.task_radii.min()

    @torch.no_grad()
    def propose(self, task_grads: Tensor, fused_grad: Tensor) -> TRCauchyProposal:
        """Compute the analytic Cauchy step and per-task model decreases."""

        self._validate_gradient_shapes(task_grads, fused_grad)
        task_grads = task_grads.detach()
        fused_grad = fused_grad.detach()
        if self.config.mode == "cosine":
            return self._propose_cosine(task_grads, fused_grad)

        delta = self.delta_global.to(device=fused_grad.device, dtype=fused_grad.dtype)
        norm = fused_grad.norm()
        if not torch.isfinite(fused_grad).all() or not torch.isfinite(norm):
            return self._zero_proposal(
                task_grads, fused_grad, delta, "non_finite_fused_gradient"
            )
        if norm <= self.config.epsilon:
            return self._zero_proposal(task_grads, fused_grad, delta, "zero_gradient")

        g2 = torch.dot(fused_grad, fused_grad)
        g_bg = self.curvature.quadratic_form(fused_grad)
        if not torch.isfinite(g_bg):
            return self._zero_proposal(
                task_grads, fused_grad, delta, "non_finite_curvature"
            )
        if isinstance(self.curvature, IdentityCurvature):
            # For B=I the analytic quotient ||g||^2 / (g^T B g) is exactly
            # one for every non-zero gradient.  Damping must not alter it.
            alpha_curvature = torch.ones_like(g2)
        else:
            if g_bg <= 0:
                return self._zero_proposal(
                    task_grads, fused_grad, delta, "non_positive_curvature"
                )
            alpha_curvature = g2 / g_bg.clamp_min(self.config.damping)
        alpha_boundary = delta / norm.clamp_min(self.config.epsilon)
        alpha = torch.minimum(alpha_curvature, alpha_boundary)
        step = -alpha * fused_grad

        step_norm = step.norm()
        tolerance = 10.0 * torch.finfo(step.dtype).eps * max(1.0, delta.item())
        if not torch.isfinite(step).all() or step_norm > delta + tolerance:
            return self._zero_proposal(
                task_grads, fused_grad, delta, "invalid_cauchy_step"
            )

        quadratic_terms = torch.stack(
            [
                self.curvature.quadratic_form(step, task_index=task_index)
                for task_index in range(self.n_tasks)
            ]
        )
        predicted = -(task_grads @ step) - 0.5 * quadratic_terms
        return TRCauchyProposal(
            step=step,
            fused_grad=fused_grad,
            delta_global=delta,
            alpha_curvature=alpha_curvature,
            alpha_boundary=alpha_boundary,
            alpha=alpha,
            predicted_decreases=predicted,
            valid_step=True,
        )

    @torch.no_grad()
    def observe_and_update(
        self,
        proposal: TRCauchyProposal,
        current_losses: Tensor,
        candidate_losses: Tensor,
    ) -> TRCauchyResult:
        """Compute true task-wise ratios and update the next radii."""

        current = self._loss_vector(current_losses, "current_losses")
        candidate = self._loss_vector(candidate_losses, "candidate_losses")
        before = self.task_radii.clone()
        actual = current - candidate
        predicted = proposal.predicted_decreases.to(
            device=actual.device, dtype=actual.dtype
        )

        finite_values = (
            torch.isfinite(current)
            & torch.isfinite(candidate)
            & torch.isfinite(actual)
            & torch.isfinite(predicted)
        )
        valid = finite_values & (predicted > self.config.epsilon)
        rho = torch.full_like(actual, torch.nan)
        rho[valid] = actual[valid] / predicted[valid]
        valid &= torch.isfinite(rho)
        rho[~valid] = torch.nan

        if self.config.mode == "adaptive_tr":
            after = before.clone()
            if proposal.failure_reason == "zero_gradient":
                # Algorithm 1 has no meaningful ratio for a zero proposal.
                after = before
            else:
                shrink = (~valid) | (rho < self.config.eta1)
                expand = valid & (rho > self.config.eta2)
                after[shrink.to(after.device)] *= self.config.gamma_dec
                after[expand.to(after.device)] *= self.config.gamma_inc
                after.clamp_(self.config.delta_min, self.config.delta_max)
            self.task_radii.copy_(after)
        elif self.config.mode == "fixed_tr":
            after = before
        else:
            # Cosine is an explicit non-TR ablation and owns no rho update.
            after = before

        return TRCauchyResult(
            proposal=proposal,
            current_losses=current,
            candidate_losses=candidate,
            actual_decreases=actual,
            rho=rho,
            valid_ratios=valid,
            task_radii_before=before,
            task_radii_after=after.clone(),
            accepted=True,
        )

    def state_dict(self) -> Dict[str, object]:
        return {
            "task_radii": self.task_radii.detach().cpu().clone(),
            "config": asdict(self.config),
            "n_tasks": self.n_tasks,
            "curvature": self.curvature.name,
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        if int(state["n_tasks"]) != self.n_tasks:
            raise ValueError("checkpoint n_tasks does not match controller")
        if state.get("curvature") != self.curvature.name:
            raise ValueError("checkpoint curvature does not match controller")
        checkpoint_config = TRCauchyConfig(**dict(state["config"]))
        if checkpoint_config != self.config:
            raise ValueError("checkpoint TRCauchyConfig does not match controller")
        radii = torch.as_tensor(
            state["task_radii"], device=self.device, dtype=self.task_radii.dtype
        )
        if radii.shape != (self.n_tasks,) or not torch.isfinite(radii).all():
            raise ValueError("invalid task_radii in checkpoint")
        if ((radii < self.config.delta_min) | (radii > self.config.delta_max)).any():
            raise ValueError("checkpoint task_radii are outside configured bounds")
        self.task_radii.copy_(radii)

    def _validate_gradient_shapes(self, task_grads: Tensor, fused_grad: Tensor) -> None:
        if task_grads.ndim != 2 or task_grads.shape[0] != self.n_tasks:
            raise ValueError("task_grads must have shape [num_tasks, num_parameters]")
        if fused_grad.ndim != 1 or fused_grad.shape[0] != task_grads.shape[1]:
            raise ValueError("fused_grad must have shape [num_parameters]")
        if task_grads.device != fused_grad.device:
            raise ValueError("task_grads and fused_grad must be on the same device")

    def _loss_vector(self, losses: Tensor, name: str) -> Tensor:
        value = torch.as_tensor(losses, device=self.device).detach().flatten()
        if value.shape != (self.n_tasks,):
            raise ValueError(f"{name} must contain exactly {self.n_tasks} values")
        return value

    def _zero_proposal(
        self,
        task_grads: Tensor,
        fused_grad: Tensor,
        delta: Tensor,
        reason: str,
    ) -> TRCauchyProposal:
        zero = torch.zeros_like(fused_grad)
        scalar_zero = fused_grad.new_zeros(())
        predicted = fused_grad.new_zeros(self.n_tasks)
        return TRCauchyProposal(
            step=zero,
            fused_grad=fused_grad,
            delta_global=delta,
            alpha_curvature=scalar_zero,
            alpha_boundary=scalar_zero,
            alpha=scalar_zero,
            predicted_decreases=predicted,
            valid_step=False,
            failure_reason=reason,
        )

    def _propose_cosine(
        self, task_grads: Tensor, fused_grad: Tensor
    ) -> TRCauchyProposal:
        """Paper ablation: scale the fused direction by angular agreement.

        The PDF does not specify a separate threshold formula.  We use the mean
        finite cosine agreement, clamped to ``[0, 1]``, solely as a scalar.  It
        therefore cannot change the fusion weights or the fused direction.
        """

        delta = self.delta_global.to(device=fused_grad.device, dtype=fused_grad.dtype)
        scalar_zero = fused_grad.new_zeros(())
        norm = fused_grad.norm()
        if (
            not torch.isfinite(fused_grad).all()
            or not torch.isfinite(norm)
            or norm <= self.config.epsilon
        ):
            reason = "zero_gradient" if torch.isfinite(norm) else "non_finite_fused_gradient"
            proposal = self._zero_proposal(task_grads, fused_grad, delta, reason)
            proposal.cosine_scale = fused_grad.new_zeros(())
            return proposal

        task_norms = task_grads.norm(dim=1)
        denominators = task_norms * norm
        valid = (
            torch.isfinite(task_grads).all(dim=1)
            & torch.isfinite(denominators)
            & (denominators > self.config.epsilon)
        )
        similarities = torch.zeros(
            self.n_tasks, device=fused_grad.device, dtype=fused_grad.dtype
        )
        similarities[valid] = (task_grads[valid] @ fused_grad) / denominators[valid]
        # Finite inputs can still overflow during norm or dot-product reduction.
        valid &= torch.isfinite(similarities)
        scale = similarities[valid].mean().clamp(0.0, 1.0) if valid.any() else scalar_zero
        step = -scale * fused_grad
        if not torch.isfinite(scale) or not torch.isfinite(step).all():
            proposal = self._zero_proposal(
                task_grads, fused_grad, delta, "invalid_cosine_step"
            )
            proposal.cosine_scale = scalar_zero
            return proposal
        nan_vector = torch.full(
            (self.n_tasks,), torch.nan, device=fused_grad.device, dtype=fused_grad.dtype
        )
        return TRCauchyProposal(
            step=step,
            fused_grad=fused_grad,
            delta_global=delta,
            alpha_curvature=nan_vector.new_tensor(torch.nan),
            alpha_boundary=nan_vector.new_tensor(torch.nan),
            alpha=scale,
            predicted_decreases=nan_vector,
            valid_step=True,
            cosine_scale=scale,
        )


class FunctionalCandidateEvaluator:
    """Evaluate paired current/candidate losses with ``torch.func``.

    Both forwards start from cloned buffers and the same RNG state.  This makes
    Dropout masks identical, prevents BatchNorm running-stat pollution, never
    mutates optimizer state, and preserves every module's train/eval flag.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        shared_parameters: Sequence[torch.nn.Parameter],
        loss_closure: LossClosure,
        n_tasks: int,
        forward_state: Optional[CandidateForwardState] = None,
    ) -> None:
        if functional_call is None:
            raise RuntimeError(
                "torch.func.functional_call is required by this project; "
                "upgrade PyTorch or provide a custom paired candidate evaluator"
            )
        self.model = model
        self.shared_parameters = tuple(shared_parameters)
        self.loss_closure = loss_closure
        self.n_tasks = n_tasks
        self.forward_state = forward_state
        if n_tasks <= 0:
            raise ValueError("n_tasks must be positive")
        if not self.shared_parameters:
            raise ValueError("shared_parameters cannot be empty")
        parameter_ids = [id(parameter) for parameter in self.shared_parameters]
        if len(parameter_ids) != len(set(parameter_ids)):
            raise ValueError("shared_parameters contains duplicate parameters")
        names_by_id = {id(parameter): name for name, parameter in model.named_parameters()}
        missing = [parameter for parameter in self.shared_parameters if id(parameter) not in names_by_id]
        if missing:
            raise ValueError("every shared parameter must belong to model")
        self.shared_names = tuple(names_by_id[id(p)] for p in self.shared_parameters)
        devices = {parameter.device for parameter in self.shared_parameters}
        if len(devices) != 1:
            raise ValueError("all shared parameters must be on the same device")
        self.device = next(iter(devices))
        if forward_state is not None:
            if forward_state.model_id != id(model):
                raise ValueError("forward_state was captured from a different model")
            current_buffers = {name for name, _ in model.named_buffers()}
            if current_buffers != set(forward_state.buffers):
                raise ValueError("model buffers changed after forward-state capture")

    @staticmethod
    def capture_forward_state(model: torch.nn.Module) -> CandidateForwardState:
        """Capture state before the forward that produces task gradients."""

        return CandidateForwardState.capture(model)

    def validate_shared_parameters(
        self, shared_parameters: Sequence[torch.nn.Parameter]
    ) -> None:
        candidate = tuple(shared_parameters)
        if len(candidate) != len(self.shared_parameters) or any(
            left is not right
            for left, right in zip(candidate, self.shared_parameters)
        ):
            raise ValueError(
                "candidate evaluator and TR adapter must use the same ordered "
                "shared_parameters"
            )

    @torch.no_grad()
    def evaluate(self, step: Tensor) -> Tuple[Tensor, Tensor]:
        if step.ndim != 1 or step.numel() != sum(p.numel() for p in self.shared_parameters):
            raise ValueError("step shape does not match shared parameters")
        if step.device != self.device or not step.is_floating_point():
            raise ValueError("step must be floating point on the shared-parameter device")

        continuation_cpu_rng = torch.random.get_rng_state().clone()
        continuation_cuda_rng = None
        if torch.cuda.is_available():
            continuation_cuda_rng = tuple(
                state.clone() for state in torch.cuda.get_rng_state_all()
            )
        continuation_flags = {
            name: module.training for name, module in self.model.named_modules()
        }

        if self.forward_state is None:
            evaluation_cpu_rng = continuation_cpu_rng
            evaluation_cuda_rng = continuation_cuda_rng
            source_buffers = {
                name: buffer.detach().clone()
                for name, buffer in self.model.named_buffers()
            }
            evaluation_flags = continuation_flags
        else:
            evaluation_cpu_rng = self.forward_state.cpu_rng
            evaluation_cuda_rng = self.forward_state.cuda_rng
            source_buffers = self.forward_state.buffers
            evaluation_flags = self.forward_state.training_flags

        def set_rng(cpu_rng: Tensor, cuda_rng: Optional[Tuple[Tensor, ...]]) -> None:
            torch.random.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(list(cuda_rng))

        modules_by_name = dict(self.model.named_modules())
        for name, was_training in evaluation_flags.items():
            modules_by_name[name].training = was_training

        def state_with_step(candidate_step: Optional[Tensor]) -> Dict[str, Tensor]:
            state: Dict[str, Tensor] = {
                name: buffer.detach().clone()
                for name, buffer in source_buffers.items()
            }
            offset = 0
            for name, parameter in zip(self.shared_names, self.shared_parameters):
                value = parameter.detach().clone()
                if candidate_step is not None:
                    count = parameter.numel()
                    value.add_(
                        candidate_step[offset : offset + count]
                        .view_as(parameter)
                        .to(device=parameter.device, dtype=parameter.dtype)
                    )
                    offset += count
                state[name] = value
            return state

        def call_with(state: Mapping[str, Tensor]) -> Tensor:
            def forward(*args, **kwargs):
                return functional_call(
                    self.model, state, args=args, kwargs=kwargs, strict=False
                )

            losses = self.loss_closure(forward)
            if not isinstance(losses, torch.Tensor):
                raise TypeError("loss_closure must return a torch.Tensor")
            if losses.ndim != 1 or losses.shape[0] != self.n_tasks:
                raise ValueError(
                    f"loss_closure must return shape [{self.n_tasks}]"
                )
            if not losses.is_floating_point():
                raise TypeError("task losses must have a floating-point dtype")
            if losses.device != self.device:
                raise ValueError(
                    "task losses must be on the shared-parameter device"
                )
            return losses.detach()

        try:
            set_rng(evaluation_cpu_rng, evaluation_cuda_rng)
            current = call_with(state_with_step(None))
            set_rng(evaluation_cpu_rng, evaluation_cuda_rng)
            candidate = call_with(state_with_step(step))
        finally:
            set_rng(continuation_cpu_rng, continuation_cuda_rng)
            for name, was_training in continuation_flags.items():
                modules_by_name[name].training = was_training
        if current.dtype != candidate.dtype:
            raise ValueError("current and candidate losses must have the same dtype")
        return current, candidate


class TensorCandidateEvaluator:
    """Pure evaluator for toy problems whose shared state is one flat tensor."""

    def __init__(
        self,
        parameter: Tensor,
        loss_function: Callable[[Tensor], Tensor],
        n_tasks: int,
    ):
        self.parameter = parameter
        self.loss_function = loss_function
        self.n_tasks = n_tasks

    def validate_shared_parameters(
        self, shared_parameters: Sequence[torch.nn.Parameter]
    ) -> None:
        candidate = tuple(shared_parameters)
        if len(candidate) != 1 or candidate[0] is not self.parameter:
            raise ValueError(
                "tensor evaluator and TR adapter must use the same parameter"
            )

    def _validate_losses(self, losses: object) -> Tensor:
        if not isinstance(losses, torch.Tensor):
            raise TypeError("loss_function must return a torch.Tensor")
        if losses.ndim != 1 or losses.shape[0] != self.n_tasks:
            raise ValueError(f"loss_function must return shape [{self.n_tasks}]")
        if not losses.is_floating_point() or losses.device != self.parameter.device:
            raise ValueError(
                "task losses must be floating point on the parameter device"
            )
        return losses.detach()

    @torch.no_grad()
    def evaluate(self, step: Tensor) -> Tuple[Tensor, Tensor]:
        current = self._validate_losses(
            self.loss_function(self.parameter.detach())
        )
        candidate = self._validate_losses(
            self.loss_function(self.parameter.detach() + step.view_as(self.parameter))
        )
        return current, candidate


def flatten_tensors(tensors: Iterable[Tensor]) -> Tensor:
    values = tuple(tensors)
    if not values:
        raise ValueError("cannot flatten an empty tensor collection")
    return torch.cat([value.reshape(-1) for value in values])


@torch.no_grad()
def apply_parameter_step(
    shared_parameters: Sequence[torch.nn.Parameter], step: Tensor
) -> None:
    """Commit the exact shared-parameter displacement returned by TR-Cauchy."""

    expected = sum(parameter.numel() for parameter in shared_parameters)
    if step.ndim != 1 or step.numel() != expected:
        raise ValueError("step shape does not match shared parameters")
    offset = 0
    for parameter in shared_parameters:
        count = parameter.numel()
        parameter.add_(
            step[offset : offset + count]
            .view_as(parameter)
            .to(device=parameter.device, dtype=parameter.dtype)
        )
        parameter.grad = None
        offset += count
