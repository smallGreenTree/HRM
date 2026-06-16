import torch

from models.hrm.hrm_act_v1 import (
    HierarchicalReasoningModel_ACTV1Config,
    HierarchicalReasoningModel_ACTV1_Inner,
)


def make_model(
    *,
    level: str | None,
    source: str | None,
    scale: float,
    act_steps: list[int] | None = None,
) -> HierarchicalReasoningModel_ACTV1_Inner:
    config = HierarchicalReasoningModel_ACTV1Config(
        batch_size=2,
        seq_len=1,
        puzzle_emb_ndim=0,
        num_puzzle_identifiers=1,
        vocab_size=4,
        H_cycles=2,
        L_cycles=2,
        H_layers=1,
        L_layers=1,
        hidden_size=4,
        expansion=2,
        num_heads=1,
        pos_encodings="rope",
        halt_max_steps=2,
        halt_exploration_prob=0.0,
        merge_intervention_level=level,
        merge_intervention_source=source,
        merge_intervention_scale=scale,
        merge_intervention_act_steps=act_steps or [],
        forward_dtype="float32",
    )
    return HierarchicalReasoningModel_ACTV1_Inner(config).eval()


def sources() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    z_h = torch.full((2, 1, 4), 2.0)
    z_l = torch.full((2, 1, 4), 3.0)
    inputs = torch.full((2, 1, 4), 5.0)
    return z_h, z_l, inputs


def initial_carry(model: HierarchicalReasoningModel_ACTV1_Inner):
    carry = model.empty_carry(batch_size=2)
    return model.reset_carry(torch.ones(2, dtype=torch.bool), carry)


def test_l_merge_scales_h_source_before_addition():
    model = make_model(level="L", source="H", scale=0.5)
    z_h, z_l, inputs = sources()

    hidden, injection = model._prepare_level_inputs(
        level="L", z_H=z_h, z_L=z_l, input_embeddings=inputs, intervention_step=torch.tensor([1, 1])
    )

    assert torch.equal(hidden, z_l)
    assert torch.equal(injection, 0.5 * z_h + inputs)


def test_h_merge_scales_l_source_before_addition():
    model = make_model(level="H", source="L", scale=1.25)
    z_h, z_l, inputs = sources()

    hidden, injection = model._prepare_level_inputs(
        level="H", z_H=z_h, z_L=z_l, input_embeddings=inputs, intervention_step=torch.tensor([1, 1])
    )

    assert torch.equal(hidden, z_h)
    assert torch.equal(injection, 1.25 * z_l)


def test_act_selector_only_scales_selected_rows():
    model = make_model(level="L", source="input", scale=0.0, act_steps=[2])
    z_h, z_l, inputs = sources()

    _, injection = model._prepare_level_inputs(
        level="L", z_H=z_h, z_L=z_l, input_embeddings=inputs, intervention_step=torch.tensor([1, 2])
    )

    assert torch.equal(injection[0], z_h[0] + inputs[0])
    assert torch.equal(injection[1], z_h[1])


def test_scale_one_preserves_original_merges():
    model = make_model(level="L", source="H", scale=1.0)
    z_h, z_l, inputs = sources()

    l_hidden, l_injection = model._prepare_level_inputs(
        level="L", z_H=z_h, z_L=z_l, input_embeddings=inputs, intervention_step=torch.tensor([1, 2])
    )
    h_hidden, h_injection = model._prepare_level_inputs(
        level="H", z_H=z_h, z_L=z_l, input_embeddings=inputs, intervention_step=torch.tensor([1, 2])
    )

    assert torch.equal(l_hidden, z_l)
    assert torch.equal(l_injection, z_h + inputs)
    assert torch.equal(h_hidden, z_h)
    assert torch.equal(h_injection, z_l)


def test_active_intervention_matches_traced_and_untraced_forward_paths():
    torch.manual_seed(0)
    model = make_model(level="L", source="H", scale=0.75)
    batch = {
        "inputs": torch.tensor([[0], [1]], dtype=torch.int32),
        "puzzle_identifiers": torch.zeros(2, dtype=torch.int32),
    }

    clean_path = model(initial_carry(model), batch, intervention_step=torch.tensor([1, 1]))
    traced_path = model(
        initial_carry(model),
        batch,
        return_block_traces=True,
        intervention_step=torch.tensor([1, 1]),
    )

    assert torch.allclose(clean_path[0].z_H, traced_path[0].z_H)
    assert torch.allclose(clean_path[0].z_L, traced_path[0].z_L)
    assert torch.allclose(clean_path[1], traced_path[1])


def test_invalid_merge_source_fails_loudly():
    model = make_model(level="H", source="input", scale=0.5)
    batch = {
        "inputs": torch.tensor([[0], [1]], dtype=torch.int32),
        "puzzle_identifiers": torch.zeros(2, dtype=torch.int32),
    }

    try:
        model(initial_carry(model), batch, intervention_step=torch.tensor([1, 1]))
    except ValueError as exc:
        assert "Invalid merge intervention" in str(exc)
    else:
        raise AssertionError("Invalid merge source did not raise ValueError")
