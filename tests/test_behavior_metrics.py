import torch

from inforidge.behavior import per_example_behavior


def test_per_example_behavior_reports_exact_and_graded_metrics():
    logits = torch.tensor(
        [
            [[4.0, 1.0], [1.0, 4.0]],
            [[4.0, 1.0], [4.0, 1.0]],
        ]
    )
    labels = torch.tensor([[0, 1], [0, 1]])

    metrics = per_example_behavior(logits, labels)

    assert metrics["exact_accuracy"].tolist() == [True, False]
    assert torch.allclose(metrics["cell_accuracy"], torch.tensor([1.0, 0.5]))
    assert metrics["target_nll"][0] < metrics["target_nll"][1]
    assert metrics["target_margin"][0] > metrics["target_margin"][1]
