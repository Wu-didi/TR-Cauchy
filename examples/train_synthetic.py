"""Train a tiny two-task network on generated data and print one JSON report.

Run with ``python -m examples.train_synthetic --steps 3``. This example exercises
the training API; it does not reproduce any benchmark result from the paper.
"""

import argparse
import json
import math
import random

import numpy as np
import torch
from torch import nn

from methods.tr_cauchy import FunctionalCandidateEvaluator
from methods.weight_methods import WeightMethods


class TwoTaskNetwork(nn.Module):
    """A shared nonlinear representation and two independent regression heads."""

    def __init__(self):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(4, 8), nn.Tanh(), nn.Dropout(p=0.1)
        )
        self.heads = nn.ModuleList([nn.Linear(8, 1), nn.Linear(8, 1)])

    def forward(self, inputs):
        features = self.shared(inputs)
        return [head(features) for head in self.heads]


def task_losses(forward, inputs, targets):
    return torch.stack(
        [
            nn.functional.mse_loss(prediction, target)
            for prediction, target in zip(forward(inputs), targets)
        ]
    )


@torch.no_grad()
def evaluate(model, inputs, targets):
    """Use dropout-free evaluation for comparable initial/final measurements."""
    was_training = model.training
    model.eval()
    try:
        return task_losses(model, inputs, targets)
    finally:
        model.train(was_training)


def json_values(tensor):
    """Represent undefined diagnostics as JSON null, never NaN or Infinity."""
    values = tensor.detach().cpu().reshape(-1).tolist()
    return [float(value) if math.isfinite(value) else None for value in values]


def flat_parameters(parameters):
    return torch.cat([parameter.detach().reshape(-1) for parameter in parameters])


def run_training(method="trpcgrad", tr_mode="adaptive_tr", steps=10, seed=7, device="cpu"):
    if steps < 1:
        raise ValueError("steps must be positive")
    if method not in {"trpcgrad", "trcagrad"}:
        raise ValueError("method must be trpcgrad or trcagrad")
    if tr_mode not in {"adaptive_tr", "fixed_tr", "cosine"}:
        raise ValueError("unknown TR mode")
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is not available")

    random.seed(seed)  # PCGrad randomizes the order of gradient projections.
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Generate on CPU before transfer so the data do not depend on the device.
    inputs = torch.randn(32, 4)
    targets = [
        (inputs[:, 0] + 0.5 * inputs[:, 1] - 0.25 * inputs[:, 2].square()).unsqueeze(1),
        (-inputs[:, 0] + inputs[:, 2] + 0.25 * inputs[:, 3].square()).unsqueeze(1),
    ]
    inputs = inputs.to(device)
    targets = [target.to(device) for target in targets]
    model = TwoTaskNetwork().to(device).train()
    shared = list(model.shared.parameters())
    heads = list(model.heads.parameters())
    head_values_before = flat_parameters(heads).clone()
    weight_method = WeightMethods(method, n_tasks=2, device=device, tr_mode=tr_mode)
    # Shared gradients are cleared by the TR adapter after its direct update.
    # This outer Adam step therefore updates only the task heads.
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.0)
    initial_losses = evaluate(model, inputs, targets)
    initial_state = weight_method.state_dict()["tr_controller"]
    checks = {
        "losses_parameters_and_steps_finite": bool(torch.isfinite(initial_losses).all()),
        "shared_gradients_cleared": True,
        "direct_shared_step_matches_proposal": True,
        "outer_optimizer_preserved_shared_parameters": True,
        "paired_current_matches_gradient_forward": (
            True if weight_method.requires_candidate_evaluator else None
        ),
    }
    history = []

    for step_index in range(steps):
        optimizer.zero_grad(set_to_none=True)
        # Capture before the gradient forward, including its dropout RNG state.
        forward_state = (
            FunctionalCandidateEvaluator.capture_forward_state(model)
            if weight_method.requires_candidate_evaluator
            else None
        )
        losses = task_losses(model, inputs, targets)
        evaluator = None
        if weight_method.requires_candidate_evaluator:
            evaluator = FunctionalCandidateEvaluator(
                model,
                shared,
                lambda forward: task_losses(forward, inputs, targets),
                n_tasks=2,
                forward_state=forward_state,
            )

        shared_before = flat_parameters(shared).clone()
        _, diagnostics = weight_method.backward(
            losses=losses,
            shared_parameters=shared,
            task_specific_parameters=heads,
            candidate_evaluator=evaluator,
        )
        shared_after = flat_parameters(shared).clone()
        checks["shared_gradients_cleared"] &= all(p.grad is None for p in shared)
        checks["direct_shared_step_matches_proposal"] &= torch.allclose(
            shared_after - shared_before, diagnostics["step"], rtol=1e-4, atol=1e-7
        )
        paired = None
        if weight_method.requires_candidate_evaluator:
            result = weight_method.method.last_result
            checks["paired_current_matches_gradient_forward"] &= torch.allclose(
                result.current_losses, losses.detach(), rtol=1e-5, atol=1e-6
            )
            paired = {
                "current_losses": json_values(result.current_losses),
                "candidate_losses": json_values(result.candidate_losses),
                "ratios": json_values(result.rho),
                "valid_ratios": result.valid_ratios.detach().cpu().tolist(),
            }
            checks["losses_parameters_and_steps_finite"] &= bool(
                torch.isfinite(result.current_losses).all()
                and torch.isfinite(result.candidate_losses).all()
            )

        optimizer.step()
        checks["outer_optimizer_preserved_shared_parameters"] &= torch.equal(
            shared_after, flat_parameters(shared)
        )
        evaluation_losses = evaluate(model, inputs, targets)
        radii = weight_method.state_dict()["tr_controller"]["task_radii"]
        checks["losses_parameters_and_steps_finite"] &= bool(
            torch.isfinite(losses).all()
            and torch.isfinite(evaluation_losses).all()
            and torch.isfinite(flat_parameters(list(model.parameters()))).all()
            and torch.isfinite(diagnostics["step"]).all()
            and torch.isfinite(radii).all()
        )
        history.append(
            {
                "step": step_index + 1,
                "evaluation_task_losses": json_values(evaluation_losses),
                "task_radii": json_values(radii),
                "shared_step_norm": json_values(diagnostics["step"].norm())[0],
                "paired_shared_step_evaluation": paired,
                "cosine_scale": (
                    json_values(diagnostics["cosine_scale"])[0]
                    if tr_mode == "cosine"
                    else None
                ),
            }
        )

    checks["task_heads_updated"] = not torch.equal(
        head_values_before, flat_parameters(heads)
    )
    return {
        "config": {
            "method": method,
            "tr_mode": tr_mode,
            "steps": steps,
            "seed": seed,
            "device": str(device),
            "batch_size": 32,
            "task_count": 2,
            "outer_optimizer": "Adam",
            "head_learning_rate": 1e-3,
            "weight_decay": 0.0,
            "tr_controller": initial_state["config"],
        },
        "initial": {
            "evaluation_task_losses": json_values(initial_losses),
            "task_radii": json_values(initial_state["task_radii"]),
        },
        "final": {
            "evaluation_task_losses": history[-1]["evaluation_task_losses"],
            "task_radii": history[-1]["task_radii"],
        },
        "steps_completed": len(history),
        "checks": checks,
        "history": history,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["trpcgrad", "trcagrad"], default="trpcgrad")
    parser.add_argument(
        "--tr-mode", choices=["adaptive_tr", "fixed_tr", "cosine"], default="adaptive_tr"
    )
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    try:
        report = run_training(**vars(args))
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
