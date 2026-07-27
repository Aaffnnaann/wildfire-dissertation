"""
Smoke test for all four ablation models on synthetic tensors — verifies shapes,
forward pass, and that gradients flow, without needing the real satellite patches.

Run:  python -m tests.smoke_models
"""
import torch
import torch.nn.functional as F

from wildfire.models.lstm import LSTMBaseline
from wildfire.models.temporal_transformer import TemporalTransformer
from wildfire.models.vit import ViTEncoder
from wildfire.models.fusion import DualBranchFusion

B, T, N_DYN, N_STAT, WIN, CH = 8, 30, 13, 14, 64, 4


def fake_batch():
    return {
        "dynamic": torch.randn(B, T, N_DYN),
        "static": torch.randn(B, N_STAT),
        "patch": torch.randn(B, WIN, WIN, CH),
        "y": torch.randint(0, 2, (B,)).float(),
        "w": torch.ones(B),
    }


def check(name, model, batch):
    model.train()
    logit = model(batch)
    assert logit.shape == (B,), f"{name}: bad output shape {logit.shape}"
    loss = (F.binary_cross_entropy_with_logits(logit, batch["y"], reduction="none")
            * batch["w"]).mean()
    loss.backward()
    n_grad = sum(p.grad is not None and p.grad.abs().sum() > 0
                 for p in model.parameters() if p.requires_grad)
    n_params = sum(p.numel() for p in model.parameters())
    extra = ""
    if isinstance(model, DualBranchFusion) and model.fusion == "cross":
        extra = f" attn={tuple(model.last_attn.shape)}"
    print(f"OK  {name:28s} params={n_params:>10,}  out={tuple(logit.shape)}  "
          f"grads_flowing={n_grad}{extra}")


def main():
    torch.manual_seed(0)
    b = fake_batch()
    check("A  ViT (satellite only)", ViTEncoder(), b)
    check("B  TemporalTransformer", TemporalTransformer(), b)
    check("0  LSTM baseline", LSTMBaseline(), b)
    check("C  DualBranch concat", DualBranchFusion(fusion="concat"), b)
    check("D  DualBranch cross-attn", DualBranchFusion(fusion="cross"), b)
    print("\nall models forward + backward cleanly")


if __name__ == "__main__":
    main()
