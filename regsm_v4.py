
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from regsm.regsm_v3 import RoleBranch, TransformerBlock


class ReGSMPrimaryBranchModel(nn.Module):

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        num_heads: int = 4,
        d_ff: int = 512,
        num_base_layers: int = 2,
        max_recurrent_steps: int = 3,
        max_branch_steps: int = 2,
        max_k: int = 1,
        primary_branch: int = 0,
        max_seq_len: int = 128,
        dropout: float = 0.1,
        branch_init_noise: float = 0.05,
    ):
        super().__init__()
        if primary_branch < 0 or primary_branch >= max_k:
            raise ValueError(
                f"primary_branch={primary_branch} is invalid for max_k={max_k}"
            )

        self.d_model = d_model
        self.max_recurrent_steps = max_recurrent_steps
        self.max_branch_steps = max_branch_steps
        self.max_k = max_k
        self.primary_branch = primary_branch

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.emb_drop = nn.Dropout(dropout)

        self.base_blocks = nn.ModuleList(
            [
                TransformerBlock(d_model, num_heads, d_ff, dropout)
                for _ in range(num_base_layers)
            ]
        )
        self.role_emb = nn.Parameter(torch.randn(max_k, d_model) * 0.02)
        self.branches = nn.ModuleList(
            [
                RoleBranch(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    num_steps=max_branch_steps,
                    dropout=dropout,
                    init_noise=branch_init_noise,
                )
                for _ in range(max_k)
            ]
        )
        self.norm_out = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

        self.trace: List[Dict] = []
        self.last_branch_deltas: List[torch.Tensor] = []
        self.last_primary_branch_idx: List[int] = []
        self.branch_loss = torch.zeros(())

    def _logits_from_hidden(self, h: torch.Tensor) -> torch.Tensor:
        return self.head(self.norm_out(h))

    @staticmethod
    def gather_at(logits: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        idx = pos.view(-1, 1, 1).expand(-1, 1, logits.size(-1))
        return logits.gather(dim=1, index=idx).squeeze(1)

    def _branch_aux_loss(
        self,
        candidates: List[torch.Tensor],
        targets: torch.Tensor,
        target_pos: torch.Tensor,
    ) -> torch.Tensor:
        losses = []
        for candidate in candidates:
            selected = self.gather_at(self._logits_from_hidden(candidate), target_pos)
            losses.append(F.cross_entropy(selected, targets, reduction="none"))
        return torch.stack(losses, dim=-1).mean()

    def forward(
        self,
        x: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        target_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch, seq_len = x.shape
        device = x.device
        self.branch_loss = torch.zeros((), device=device)
        self.trace = []
        self.last_branch_deltas = []
        self.last_primary_branch_idx = []
        branch_losses = []

        pos = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch, seq_len)
        h = self.emb_drop(self.token_emb(x) + self.pos_emb(pos))
        for block in self.base_blocks:
            h = block(h)

        for step in range(self.max_recurrent_steps):
            candidates = []
            deltas = []
            for i, branch in enumerate(self.branches):
                candidate, delta, _ = branch(h, self.role_emb[i])
                candidates.append(candidate)
                deltas.append(delta)

            if targets is not None and target_pos is not None:
                branch_losses.append(
                    self._branch_aux_loss(candidates, targets, target_pos)
                )

            h = candidates[self.primary_branch]
            self.last_branch_deltas.append(torch.stack(deltas, dim=1))
            self.last_primary_branch_idx.append(self.primary_branch)
            self.trace.append(
                {
                    "step": step,
                    "primary_branch": self.primary_branch,
                    "num_branches": len(candidates),
                }
            )

        if branch_losses:
            self.branch_loss = torch.stack(branch_losses).mean()

        return self._logits_from_hidden(h)

    def branch_diversity_loss(self) -> torch.Tensor:
        device = next(self.parameters()).device
        if not self.last_branch_deltas:
            return torch.zeros((), device=device)
        total = torch.zeros((), device=device)
        n = 0
        for deltas in self.last_branch_deltas:
            pooled = deltas.mean(dim=2).mean(dim=0)
            for i in range(pooled.size(0)):
                for j in range(i + 1, pooled.size(0)):
                    total = total + F.cosine_similarity(
                        pooled[i].unsqueeze(0), pooled[j].unsqueeze(0)
                    ).squeeze()
                    n += 1
        return total / max(n, 1)


def _demo():
    torch.manual_seed(0)
    model = ReGSMPrimaryBranchModel(
        vocab_size=100,
        d_model=64,
        num_heads=4,
        d_ff=128,
        max_recurrent_steps=2,
        max_branch_steps=1,
        max_k=1,
        max_seq_len=16,
        dropout=0.0,
    )
    x = torch.randint(0, 100, (4, 16))
    y = torch.randint(0, 100, (4,))
    target_pos = torch.full((4,), 15, dtype=torch.long)
    logits = model(x, targets=y, target_pos=target_pos)
    print("logits:", tuple(logits.shape))
    print("branch_loss:", float(model.branch_loss.detach()))
    print("diversity:", float(model.branch_diversity_loss().detach()))
    print("trace entries:", len(model.trace))


if __name__ == "__main__":
    _demo()
