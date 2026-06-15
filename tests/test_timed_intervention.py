from types import SimpleNamespace

import torch
from torch import nn

from models.hrm.hrm_act_v1 import HierarchicalReasoningModel_ACTV1ReasoningModule


class AddOne(nn.Module):
    def forward(self, hidden_states: torch.Tensor, **_kwargs) -> torch.Tensor:
        return hidden_states + 1


def make_module(*, target_steps: list[int]) -> HierarchicalReasoningModel_ACTV1ReasoningModule:
    config = SimpleNamespace(
        layer_intervention_mode="bypass",
        layer_intervention_level="H",
        layer_intervention_layers=[0],
        layer_intervention_act_steps=target_steps,
        layer_intervention_noise_std=1.0,
    )
    return HierarchicalReasoningModel_ACTV1ReasoningModule(
        layers=[AddOne()],
        config=config,
        level_name="H",
    )


def test_timed_bypass_only_changes_selected_batch_rows():
    module = make_module(target_steps=[2])
    hidden = torch.zeros(2, 1, 1)
    injection = torch.zeros_like(hidden)

    result = module(hidden, injection, intervention_step=torch.tensor([1, 2]))

    assert torch.equal(result[0], torch.ones_like(result[0]))
    assert torch.equal(result[1], torch.zeros_like(result[1]))


def test_empty_step_selector_preserves_global_repeated_bypass():
    module = make_module(target_steps=[])
    hidden = torch.zeros(2, 1, 1)
    injection = torch.zeros_like(hidden)

    result = module(hidden, injection, intervention_step=torch.tensor([1, 9]))

    assert torch.equal(result, hidden)


def test_unselected_step_preserves_clean_transformation():
    module = make_module(target_steps=[5, 6])
    hidden = torch.zeros(2, 1, 1)
    injection = torch.zeros_like(hidden)

    result = module(hidden, injection, intervention_step=torch.tensor([3, 4]))

    assert torch.equal(result, torch.ones_like(hidden))


def test_collection_returns_each_block_input_and_output():
    config = SimpleNamespace(
        layer_intervention_mode="none",
        layer_intervention_level=None,
        layer_intervention_layers=[],
        layer_intervention_act_steps=[],
        layer_intervention_noise_std=1.0,
    )
    module = HierarchicalReasoningModel_ACTV1ReasoningModule(
        layers=[AddOne(), AddOne()],
        config=config,
        level_name="H",
    )
    hidden = torch.zeros(1, 1, 1)
    injection = torch.full_like(hidden, 2)

    result, inputs, outputs = module(hidden, injection, collect=True, collect_inputs=True)

    assert torch.equal(inputs[0], torch.full_like(hidden, 2))
    assert torch.equal(outputs[0], torch.full_like(hidden, 3))
    assert torch.equal(inputs[1], outputs[0])
    assert torch.equal(outputs[1], torch.full_like(hidden, 4))
    assert torch.equal(result, outputs[-1])
