import torch

from models.hrm.hrm_act_v1 import (
    HierarchicalReasoningModel_ACTV1Config,
    HierarchicalReasoningModel_ACTV1_Inner,
)


def make_inner_model() -> HierarchicalReasoningModel_ACTV1_Inner:
    config = HierarchicalReasoningModel_ACTV1Config(
        batch_size=1,
        seq_len=2,
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
        forward_dtype="float32",
    )
    return HierarchicalReasoningModel_ACTV1_Inner(config).eval()


def initial_inner_carry(model: HierarchicalReasoningModel_ACTV1_Inner):
    carry = model.empty_carry(batch_size=1)
    return model.reset_carry(torch.ones(1, dtype=torch.bool), carry)


def test_full_trace_captures_every_recurrent_occurrence_without_changing_output():
    torch.manual_seed(0)
    model = make_inner_model()
    batch = {
        "inputs": torch.tensor([[0, 1]], dtype=torch.int32),
        "puzzle_identifiers": torch.zeros(1, dtype=torch.int32),
    }

    clean = model(initial_inner_carry(model), batch)
    traced = model(initial_inner_carry(model), batch, return_block_traces=True)

    clean_carry, clean_logits = clean[0], clean[1]
    traced_carry, traced_logits, traces = traced[0], traced[1], traced[3]
    assert traces is not None
    assert len(traces["H_traces"]) == 2
    assert len(traces["L_traces"]) == 4
    assert [trace["phase"] for trace in traces["H_traces"]] == ["recurrent", "final"]
    assert [trace["phase"] for trace in traces["L_traces"]] == ["recurrent", "recurrent", "recurrent", "final"]
    assert torch.allclose(traced_logits, clean_logits)
    assert torch.allclose(traced_carry.z_H, clean_carry.z_H)
    assert torch.allclose(traced_carry.z_L, clean_carry.z_L)
