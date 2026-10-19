import copy
import random

import numpy as np
import pytest
import torch
from torch import nn

from methods.tr_cauchy import (
    FunctionalCandidateEvaluator,
    TRCauchyConfig,
    TRCauchyController,
)
from methods.tr_weight_methods import CAGrad, PCGrad, TRCAGrad, TRPCGrad
from methods.weight_methods import WeightMethods


def make_controller(n_tasks=3, mode="adaptive_tr", **overrides):
    values = dict(
        mode=mode,
        delta0=0.1,
        delta_min=0.01,
        delta_max=1.0,
        eta1=0.1,
        eta2=0.8,
        gamma_dec=0.5,
        gamma_inc=2.0,
    )
    values.update(overrides)
    return TRCauchyController(n_tasks, torch.device("cpu"), TRCauchyConfig(**values))


def test_identity_cauchy_step_matches_analytic_solution_and_radius():
    controller = make_controller(n_tasks=2, delta0=0.5)
    task_grads = torch.tensor([[3.0, 4.0], [3.0, 4.0]])
    fused = torch.tensor([3.0, 4.0])

    proposal = controller.propose(task_grads, fused)

    expected_alpha = min(1.0, 0.5 / 5.0)
    assert torch.allclose(proposal.step, -expected_alpha * fused)
    assert proposal.step.norm() <= controller.delta_global + 1e-7
    assert torch.allclose(proposal.alpha_curvature, torch.tensor(1.0))
    expected_predicted = -(task_grads @ proposal.step) - 0.5 * proposal.step.square().sum()
    assert torch.allclose(proposal.predicted_decreases, expected_predicted)


@pytest.mark.parametrize("magnitude", [1e-5, 1e-7])
def test_identity_curvature_keeps_exact_unit_coefficient_for_small_gradients(
    magnitude,
):
    controller = make_controller(n_tasks=2, delta0=1.0)
    fused = torch.tensor([magnitude, 0.0])
    proposal = controller.propose(torch.stack([fused, fused]), fused)
    assert proposal.failure_reason is None
    assert proposal.alpha_curvature.item() == pytest.approx(1.0)
    assert torch.allclose(proposal.step, -fused)


def test_taskwise_actual_rho_and_independent_radius_updates():
    controller = make_controller()
    task_grads = torch.tensor([[1.0, 0.0]] * 3)
    proposal = controller.propose(task_grads, torch.tensor([1.0, 0.0]))
    target_rho = torch.tensor([0.9, 0.05, 0.5])
    current = torch.ones(3)
    candidate = current - target_rho * proposal.predicted_decreases

    result = controller.observe_and_update(proposal, current, candidate)

    assert torch.allclose(result.actual_decreases, current - candidate)
    assert torch.allclose(result.rho, target_rho, atol=1e-6)
    assert torch.allclose(
        result.task_radii_after,
        torch.tensor([0.2, 0.05, 0.1], dtype=torch.float64),
    )
    assert controller.delta_global.item() == pytest.approx(0.05)


def test_fixed_radius_computes_ratios_but_does_not_update():
    controller = make_controller(n_tasks=2, mode="fixed_tr")
    proposal = controller.propose(
        torch.tensor([[1.0, 0.0], [1.0, 0.0]]), torch.tensor([1.0, 0.0])
    )
    before = controller.task_radii.clone()
    candidate = torch.ones(2) - proposal.predicted_decreases
    result = controller.observe_and_update(proposal, torch.ones(2), candidate)
    assert torch.allclose(result.rho, torch.ones(2))
    assert torch.equal(controller.task_radii, before)


def test_cosine_mode_is_scalar_directional_ablation():
    controller = make_controller(n_tasks=2, mode="cosine")
    fused = torch.tensor([2.0, 1.0])
    proposal = controller.propose(
        torch.tensor([[1.0, 0.0], [0.0, 1.0]]), fused
    )
    assert proposal.cosine_scale.item() == pytest.approx(3.0 / (2.0 * np.sqrt(5.0)))
    assert 0.0 <= proposal.cosine_scale.item() <= 1.0
    assert torch.allclose(proposal.step, -proposal.cosine_scale * fused)
    assert torch.isnan(proposal.predicted_decreases).all()


@pytest.mark.parametrize(
    "values,reason",
    [
        ([1e20, 1e20], "non_finite_fused_gradient"),
        ([float("nan"), 0.0], "non_finite_fused_gradient"),
        ([float("inf"), 0.0], "non_finite_fused_gradient"),
        ([0.0, 0.0], "zero_gradient"),
    ],
)
def test_cosine_mode_returns_zero_step_for_invalid_or_overflowed_fused_norm(
    values, reason
):
    controller = make_controller(n_tasks=2, mode="cosine")
    fused = torch.tensor(values, dtype=torch.float32)
    if values[0] == 1e20:
        assert torch.isfinite(fused).all()
        assert torch.isinf(fused.norm())

    proposal = controller.propose(torch.ones(2, 2), fused)

    assert not proposal.valid_step
    assert proposal.failure_reason == reason
    assert torch.equal(proposal.step, torch.zeros_like(fused))
    assert proposal.cosine_scale.item() == 0.0


@pytest.mark.parametrize("bad_value", [1e30, float("nan"), float("inf")])
def test_cosine_mode_excludes_nonfinite_task_agreements(bad_value):
    controller = make_controller(n_tasks=2, mode="cosine")
    fused = torch.tensor([1e10, 1e10], dtype=torch.float32)
    task_grads = torch.tensor(
        [[bad_value, bad_value], [1.0, 0.0]], dtype=torch.float32
    )
    assert torch.isfinite(fused.norm())
    if bad_value == 1e30:
        assert torch.isfinite(task_grads).all()
        assert torch.isinf(task_grads[0].norm())
        assert torch.isinf(torch.dot(task_grads[0], fused))

    proposal = controller.propose(task_grads, fused)

    assert proposal.valid_step
    assert torch.isfinite(proposal.step).all()
    assert proposal.cosine_scale.item() == pytest.approx(1.0 / np.sqrt(2.0))
    torch.testing.assert_close(proposal.step, -fused / np.sqrt(2.0))


def test_pcgrad_baseline_and_tr_use_identical_fusion():
    baseline = PCGrad(2, torch.device("cpu"), reduction="mean", max_norm=0)
    tr_method = TRPCGrad(2, torch.device("cpu"), reduction="mean")
    gradients = [
        (torch.tensor([1.0, -2.0]), torch.tensor([0.5])),
        (torch.tensor([-3.0, 1.0]), torch.tensor([2.0])),
    ]
    random.seed(13)
    baseline_fused = baseline.fuse_task_gradients(copy.deepcopy(gradients))
    random.seed(13)
    tr_fused = tr_method.fuse_task_gradients(copy.deepcopy(gradients))
    assert torch.equal(baseline_fused, tr_fused)


@pytest.mark.parametrize("c,rescale", [(0.2, 0), (0.4, 1), (0.7, 2)])
def test_cagrad_baseline_and_tr_use_same_c_rescale_weights_and_output(c, rescale):
    baseline = CAGrad(3, torch.device("cpu"), c=c, rescale=rescale, max_norm=0)
    tr_method = TRCAGrad(3, torch.device("cpu"), c=c, rescale=rescale)
    task_grads = torch.tensor(
        [[1.0, 0.0, 2.0], [0.0, 2.0, -1.0], [1.5, 1.0, 0.5]]
    )
    baseline_fused, baseline_gram, baseline_weights = baseline.fuse_task_gradients(
        task_grads
    )
    tr_fused, tr_gram, tr_weights = tr_method.fuse_task_gradients(task_grads)
    assert baseline.c == tr_method.c == c
    assert baseline.alpha == tr_method.alpha == c
    assert baseline.rescale == tr_method.rescale == rescale
    assert torch.allclose(baseline_fused, tr_fused)
    assert np.allclose(baseline_gram, tr_gram)
    assert np.allclose(baseline_weights, tr_weights)


class StatefulTwoTaskModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(3, 4),
            nn.BatchNorm1d(4),
            nn.ReLU(),
            nn.Dropout(p=0.5),
        )
        self.heads = nn.ModuleList([nn.Linear(4, 1), nn.Linear(4, 1)])

    def forward(self, inputs):
        representation = self.shared(inputs)
        return [head(representation) for head in self.heads]

    def shared_parameters(self):
        return self.shared.parameters()

    def task_specific_parameters(self):
        return self.heads.parameters()


def model_losses(forward, inputs, targets):
    predictions = forward(inputs)
    return torch.stack(
        [torch.nn.functional.mse_loss(prediction, target) for prediction, target in zip(predictions, targets)]
    )


def assert_nested_equal(left, right):
    if isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_nested_equal(left[key], right[key])
    elif isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    else:
        assert left == right


def test_functional_candidate_forward_preserves_model_optimizer_bn_rng_and_mode():
    torch.manual_seed(5)
    model = StatefulTwoTaskModel().train()
    inputs = torch.randn(8, 3)
    targets = [torch.randn(8, 1), torch.randn(8, 1)]
    shared = list(model.shared_parameters())
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer_before = copy.deepcopy(optimizer.state_dict())
    forward_state = FunctionalCandidateEvaluator.capture_forward_state(model)
    original_losses = model_losses(model, inputs, targets).detach()
    parameters_before = [parameter.detach().clone() for parameter in model.parameters()]
    buffers_before = [buffer.detach().clone() for buffer in model.buffers()]
    rng_before = torch.random.get_rng_state().clone()
    flags_before = [module.training for module in model.modules()]
    evaluator = FunctionalCandidateEvaluator(
        model,
        shared,
        lambda forward: model_losses(forward, inputs, targets),
        n_tasks=2,
        forward_state=forward_state,
    )

    current, candidate = evaluator.evaluate(
        torch.full((sum(parameter.numel() for parameter in shared),), 1e-3)
    )

    assert current.shape == candidate.shape == (2,)
    assert torch.allclose(current, original_losses)
    assert not torch.equal(current, candidate)
    assert all(
        torch.equal(before, after)
        for before, after in zip(parameters_before, model.parameters())
    )
    assert all(
        torch.equal(before, after)
        for before, after in zip(buffers_before, model.buffers())
    )
    assert_nested_equal(optimizer_before, optimizer.state_dict())
    assert torch.equal(rng_before, torch.random.get_rng_state())
    assert flags_before == [module.training for module in model.modules()]


def test_candidate_evaluator_exception_is_state_safe():
    model = StatefulTwoTaskModel().train()
    shared = list(model.shared_parameters())
    parameters_before = [parameter.detach().clone() for parameter in model.parameters()]
    buffers_before = [buffer.detach().clone() for buffer in model.buffers()]
    calls = 0

    def failing_closure(forward):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("candidate failure")
        outputs = forward(torch.randn(4, 3))
        return torch.stack([output.square().mean() for output in outputs])

    evaluator = FunctionalCandidateEvaluator(
        model, shared, failing_closure, n_tasks=2
    )
    with pytest.raises(RuntimeError, match="candidate failure"):
        evaluator.evaluate(torch.zeros(sum(parameter.numel() for parameter in shared)))
    assert all(
        torch.equal(before, after)
        for before, after in zip(parameters_before, model.parameters())
    )
    assert all(
        torch.equal(before, after)
        for before, after in zip(buffers_before, model.buffers())
    )
    assert model.training


class SimpleTwoTaskModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = nn.Linear(2, 2, bias=False)
        self.heads = nn.ModuleList(
            [nn.Linear(2, 1, bias=False), nn.Linear(2, 1, bias=False)]
        )

    def forward(self, inputs):
        representation = self.shared(inputs)
        return [head(representation) for head in self.heads]


def test_tr_adapter_commits_exact_step_and_task_heads_keep_gradients():
    torch.manual_seed(1)
    random.seed(1)
    model = SimpleTwoTaskModel()
    inputs = torch.tensor([[1.0, 2.0], [-1.0, 0.5]])
    targets = [torch.ones(2, 1), -torch.ones(2, 1)]
    forward_state = FunctionalCandidateEvaluator.capture_forward_state(model)
    predictions = model(inputs)
    losses = torch.stack(
        [torch.nn.functional.mse_loss(prediction, target) for prediction, target in zip(predictions, targets)]
    )
    shared = list(model.shared.parameters())
    heads = list(model.heads.parameters())
    expected_head_gradients = torch.autograd.grad(
        losses.sum(), heads, retain_graph=True
    )
    evaluator = FunctionalCandidateEvaluator(
        model,
        shared,
        lambda forward: model_losses(forward, inputs, targets),
        n_tasks=2,
        forward_state=forward_state,
    )
    method = TRPCGrad(
        2,
        torch.device("cpu"),
        reduction="mean",
        tr_mode="fixed_tr",
        delta0=0.05,
    )
    before = torch.cat([parameter.detach().flatten() for parameter in shared])

    _, diagnostics = method.backward(
        losses,
        shared_parameters=shared,
        task_specific_parameters=heads,
        candidate_evaluator=evaluator,
    )

    after = torch.cat([parameter.detach().flatten() for parameter in shared])
    assert torch.allclose(after - before, method.last_proposal.step)
    assert torch.allclose(after - before, diagnostics["step"])
    assert diagnostics["task_grad_norms"].shape == (2,)
    assert torch.allclose(diagnostics["committed_step_norm"], (after - before).norm())
    assert (after - before).norm() <= 0.05 + 1e-7
    assert all(parameter.grad is None for parameter in shared)
    assert all(parameter.grad is not None for parameter in heads)
    assert all(
        torch.allclose(parameter.grad, expected)
        for parameter, expected in zip(heads, expected_head_gradients)
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    shared_before_optimizer = [parameter.detach().clone() for parameter in shared]
    optimizer.step()
    assert all(
        torch.equal(before_step, parameter)
        for before_step, parameter in zip(shared_before_optimizer, shared)
    )


def test_checkpoint_round_trip_restores_task_radii():
    controller = make_controller()
    controller.task_radii.copy_(torch.tensor([0.03, 0.2, 0.7], dtype=torch.float64))
    checkpoint = controller.state_dict()
    restored = make_controller()
    restored.load_state_dict(checkpoint)
    assert torch.equal(restored.task_radii, controller.task_radii)
    assert restored.delta_global.item() == pytest.approx(0.03)


def test_weight_methods_checkpoint_round_trip_restores_task_radii():
    original = WeightMethods(
        "trpcgrad", n_tasks=2, device=torch.device("cpu"), tr_mode="adaptive_tr"
    )
    original.method.tr_controller.task_radii.copy_(
        torch.tensor([0.004, 0.02], dtype=torch.float64)
    )
    checkpoint = original.state_dict()
    restored = WeightMethods(
        "trpcgrad", n_tasks=2, device=torch.device("cpu"), tr_mode="adaptive_tr"
    )
    restored.load_state_dict(checkpoint)
    assert torch.equal(
        restored.method.tr_controller.task_radii,
        original.method.tr_controller.task_radii,
    )


@pytest.mark.parametrize("method_class", [TRPCGrad, TRCAGrad])
@pytest.mark.parametrize("mode", ["adaptive_tr", "fixed_tr", "cosine"])
def test_every_tr_adapter_mode_runs_a_complete_training_step(method_class, mode):
    torch.manual_seed(17)
    random.seed(17)
    model = SimpleTwoTaskModel()
    inputs = torch.tensor([[0.5, -1.0], [2.0, 1.0]])
    targets = [torch.zeros(2, 1), torch.ones(2, 1)]
    forward_state = None
    if mode != "cosine":
        forward_state = FunctionalCandidateEvaluator.capture_forward_state(model)
    predictions = model(inputs)
    losses = torch.stack(
        [
            torch.nn.functional.mse_loss(prediction, target)
            for prediction, target in zip(predictions, targets)
        ]
    )
    shared = list(model.shared.parameters())
    heads = list(model.heads.parameters())
    candidate_evaluator = None
    if mode != "cosine":
        candidate_evaluator = FunctionalCandidateEvaluator(
            model,
            shared,
            lambda forward: model_losses(forward, inputs, targets),
            n_tasks=2,
            forward_state=forward_state,
        )
    method = method_class(
        2, torch.device("cpu"), tr_mode=mode, delta0=0.05
    )
    radii_before = method.tr_controller.task_radii.clone()
    parameters_before = torch.cat(
        [parameter.detach().flatten() for parameter in shared]
    )

    _, diagnostics = method.backward(
        losses,
        shared_parameters=shared,
        task_specific_parameters=heads,
        candidate_evaluator=candidate_evaluator,
    )

    parameters_after = torch.cat(
        [parameter.detach().flatten() for parameter in shared]
    )
    torch.testing.assert_close(
        parameters_after - parameters_before,
        diagnostics["step"],
        rtol=1e-5,
        atol=5e-8,
    )
    assert all(parameter.grad is None for parameter in shared)
    assert all(parameter.grad is not None for parameter in heads)
    if mode == "adaptive_tr":
        assert method.last_result is not None
        assert diagnostics["rho"].shape == (2,)
    elif mode == "fixed_tr":
        assert method.last_result is not None
        assert torch.equal(method.tr_controller.task_radii, radii_before)
    else:
        assert method.last_result is None
        assert "cosine_scale" in diagnostics

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    committed_shared = [parameter.detach().clone() for parameter in shared]
    optimizer.step()
    assert all(
        torch.equal(committed, parameter)
        for committed, parameter in zip(committed_shared, shared)
    )


@pytest.mark.parametrize("method_class", [PCGrad, CAGrad])
def test_baseline_fusion_methods_complete_one_training_step(method_class):
    torch.manual_seed(23)
    random.seed(23)
    model = SimpleTwoTaskModel()
    inputs = torch.tensor([[1.0, -0.5], [0.2, 0.7]])
    targets = [torch.zeros(2, 1), torch.ones(2, 1)]
    losses = torch.stack(
        [
            torch.nn.functional.mse_loss(prediction, target)
            for prediction, target in zip(model(inputs), targets)
        ]
    )
    shared = list(model.shared.parameters())
    heads = list(model.heads.parameters())
    method = method_class(2, torch.device("cpu"), max_norm=0)
    method.backward(
        losses,
        shared_parameters=shared,
        task_specific_parameters=heads,
    )
    assert all(parameter.grad is not None for parameter in shared)
    assert all(parameter.grad is not None for parameter in heads)


def test_candidate_evaluator_rejects_duplicate_or_reordered_parameters():
    model = StatefulTwoTaskModel()
    shared = list(model.shared_parameters())
    closure = lambda forward: model_losses(
        forward,
        torch.ones(4, 3),
        [torch.zeros(4, 1), torch.zeros(4, 1)],
    )
    with pytest.raises(ValueError, match="duplicate"):
        FunctionalCandidateEvaluator(
            model, [shared[0], shared[0]], closure, n_tasks=2
        )
    evaluator = FunctionalCandidateEvaluator(model, shared, closure, n_tasks=2)
    with pytest.raises(ValueError, match="same ordered"):
        evaluator.validate_shared_parameters(list(reversed(shared)))


@pytest.mark.parametrize(
    "bad_closure,exception",
    [
        (lambda forward: [torch.tensor(1.0), torch.tensor(2.0)], TypeError),
        (lambda forward: torch.tensor(1.0), ValueError),
        (lambda forward: torch.tensor([1, 2]), TypeError),
    ],
)
def test_candidate_evaluator_rejects_invalid_loss_outputs(bad_closure, exception):
    model = SimpleTwoTaskModel()
    shared = list(model.shared.parameters())
    evaluator = FunctionalCandidateEvaluator(
        model, shared, bad_closure, n_tasks=2
    )
    with pytest.raises(exception):
        evaluator.evaluate(torch.zeros(sum(parameter.numel() for parameter in shared)))


def test_zero_nonpositive_and_nonfinite_safety_branches():
    zero_controller = make_controller(n_tasks=2)
    zero_proposal = zero_controller.propose(torch.zeros(2, 2), torch.zeros(2))
    before = zero_controller.task_radii.clone()
    zero_result = zero_controller.observe_and_update(
        zero_proposal, torch.ones(2), torch.ones(2)
    )
    assert torch.equal(zero_proposal.step, torch.zeros(2))
    assert torch.isnan(zero_result.rho).all()
    assert torch.equal(zero_controller.task_radii, before)

    invalid_controller = make_controller(n_tasks=2)
    invalid_proposal = invalid_controller.propose(
        torch.tensor([[-1.0, 0.0], [1.0, 0.0]]), torch.tensor([1.0, 0.0])
    )
    invalid_controller.observe_and_update(
        invalid_proposal, torch.ones(2), torch.ones(2)
    )
    assert invalid_controller.task_radii[0].item() == pytest.approx(0.05)

    nan_controller = make_controller(n_tasks=2)
    nan_proposal = nan_controller.propose(
        torch.ones(2, 2), torch.tensor([float("nan"), 0.0])
    )
    nan_controller.observe_and_update(
        nan_proposal, torch.ones(2), torch.ones(2)
    )
    assert torch.equal(nan_proposal.step, torch.zeros(2))
    assert torch.allclose(
        nan_controller.task_radii, torch.tensor([0.05, 0.05], dtype=torch.float64)
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {"eta1": 0.0},
        {"eta1": 0.9, "eta2": 0.8},
        {"eta2": 1.0},
        {"gamma_inc": 1.0},
        {"gamma_dec": 0.0},
        {"gamma_dec": 1.0},
    ],
)
def test_invalid_tr_hyperparameters_fail_fast(overrides):
    with pytest.raises(ValueError):
        TRCauchyConfig(**overrides)
