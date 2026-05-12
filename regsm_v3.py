
import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerBlock(nn.Module):

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.dropout(a)
        h = self.norm2(x)
        return x + self.dropout(self.ffn(h))


class DifficultyRouter(nn.Module):

    ACTION_NAMES = ["CONTINUE", "DISPATCH"]
    ACT_CONTINUE, ACT_DISPATCH = 0, 1
    NUM_ACTIONS = 2

    def __init__(
        self,
        d_model: int,
        hidden_dim: int = 128,
        init_temp: float = 1.0,
        min_temp: float = 0.1,
        anneal_rate: float = 3e-5,
    ):
        super().__init__()
        self.action_head = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.NUM_ACTIONS),
        )
        self.register_buffer("temperature", torch.tensor(float(init_temp)))
        self.min_temp = float(min_temp)
        self.anneal_rate = float(anneal_rate)
        self.last_action_probs: Optional[torch.Tensor] = None

    def anneal_temperature(self):
        self.temperature.mul_(math.exp(-self.anneal_rate))
        self.temperature.clamp_(min=self.min_temp)

    def set_temperature(self, value: float):
        self.temperature.fill_(float(value))

    def forward(
        self,
        x: torch.Tensor,
        hard: bool = True,
        force_action: Optional[str] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        pooled = x.mean(dim=1)
        action_logits = self.action_head(pooled)
        probs = F.softmax(action_logits, dim=-1)
        self.last_action_probs = probs

        if self.training:
            tau = float(self.temperature.detach())
            action_one_hot = F.gumbel_softmax(
                action_logits, tau=tau, hard=hard, dim=-1
            )
        else:
            tau = float(self.temperature.detach())
            idx = action_logits.argmax(dim=-1)
            action_one_hot = F.one_hot(idx, num_classes=self.NUM_ACTIONS).to(
                dtype=x.dtype
            )

        if force_action is not None:
            forced = force_action.upper()
            if forced == "SPLIT":
                forced = "DISPATCH"
            forced_idx = self.ACTION_NAMES.index(forced)
            forced_one_hot = torch.zeros_like(action_one_hot)
            forced_one_hot[:, forced_idx] = 1.0
            action_one_hot = action_one_hot + (forced_one_hot - action_one_hot).detach()

        action_idx = action_one_hot.argmax(dim=-1)
        action_counts = {name: 0 for name in self.ACTION_NAMES}
        for ai in action_idx.detach().cpu().tolist():
            action_counts[self.ACTION_NAMES[int(ai)]] += 1

        info = {
            "action_counts": action_counts,
            "action_probs": [
                round(float(p), 3) for p in probs.mean(dim=0).detach().cpu()
            ],
            "temperature": round(tau, 4),
        }
        if force_action is not None:
            info["forced"] = True
        return action_idx, action_one_hot, info


class RoleBranch(nn.Module):

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        num_steps: int,
        dropout: float,
        init_noise: float = 0.05,
    ):
        super().__init__()
        self.input_proj = nn.Linear(d_model, d_model)
        with torch.no_grad():
            nn.init.eye_(self.input_proj.weight)
            self.input_proj.weight.add_(torch.randn_like(self.input_proj.weight) * init_noise)
            nn.init.zeros_(self.input_proj.bias)

        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_steps)]
        )
        self.delta_norm = nn.LayerNorm(d_model)
        self.delta_proj = nn.Linear(d_model, d_model)
        nn.init.normal_(self.delta_proj.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.delta_proj.bias)
        self.delta_scale = nn.Parameter(torch.tensor(0.1))

    def forward(
        self, trunk: torch.Tensor, role: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.input_proj(trunk) + role.view(1, 1, -1)
        for block in self.blocks:
            h = block(h)
        delta = torch.tanh(self.delta_scale) * self.delta_proj(self.delta_norm(h))
        candidate = trunk + delta
        return candidate, delta, h


class EvidenceMerger(nn.Module):

    def __init__(self, d_model: int, hidden_dim: int = 128):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(4 * d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.last_weights: Optional[torch.Tensor] = None

    def forward(
        self,
        trunk: torch.Tensor,
        candidates: List[torch.Tensor],
        deltas: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        trunk_pool = trunk.mean(dim=1)
        cand_pool = torch.stack([c.mean(dim=1) for c in candidates], dim=1)
        delta_pool = torch.stack([d.mean(dim=1) for d in deltas], dim=1)
        trunk_expand = trunk_pool.unsqueeze(1).expand_as(cand_pool)
        features = torch.cat(
            [trunk_expand, cand_pool, delta_pool, cand_pool - trunk_expand], dim=-1
        )
        scores = self.scorer(features).squeeze(-1)
        weights = F.softmax(scores, dim=-1)
        self.last_weights = weights
        stacked_delta = torch.stack(deltas, dim=1)
        merged_delta = (stacked_delta * weights.view(-1, len(deltas), 1, 1)).sum(dim=1)
        return trunk + merged_delta, weights


class ReGSMMergeModel(nn.Module):

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        num_heads: int = 4,
        d_ff: int = 512,
        num_base_layers: int = 2,
        max_recurrent_steps: int = 3,
        max_branch_steps: int = 2,
        max_k: int = 4,
        max_seq_len: int = 128,
        dropout: float = 0.1,
        continue_mode: str = "full",
        share_continue_weights: bool = True,
        dispatch_mode: str = "merge",
        branch_init_noise: float = 0.05,
        verbose: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_recurrent_steps = max_recurrent_steps
        self.max_branch_steps = max_branch_steps
        self.max_k = max_k
        self.continue_mode = continue_mode
        self.share_continue_weights = share_continue_weights
        self.dispatch_mode = dispatch_mode
        self.verbose = verbose

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.emb_drop = nn.Dropout(dropout)

        self.base_blocks = nn.ModuleList(
            [
                TransformerBlock(d_model, num_heads, d_ff, dropout)
                for _ in range(num_base_layers)
            ]
        )

        def make_continue():
            if continue_mode == "full":
                return TransformerBlock(d_model, num_heads, d_ff, dropout)
            if continue_mode == "linear":
                return nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model))
            if continue_mode == "identity":
                return nn.Identity()
            raise ValueError(f"unknown continue_mode {continue_mode!r}")

        if share_continue_weights:
            self.continue_block = make_continue()
            self.continue_blocks = None
        else:
            self.continue_block = None
            self.continue_blocks = nn.ModuleList(
                [make_continue() for _ in range(max_recurrent_steps)]
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
        self.router = DifficultyRouter(d_model)
        self.merger = EvidenceMerger(d_model)
        self.norm_out = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

        self.trace: List[Dict] = []
        self.last_main_action_probs: List[torch.Tensor] = []
        self.last_main_action_idx: List[torch.Tensor] = []
        self.last_merge_weights: List[torch.Tensor] = []
        self.last_branch_deltas: List[torch.Tensor] = []
        self.branch_loss = torch.zeros(())

    def _fixed_dispatch_branch_idx(self) -> Optional[int]:
        if self.dispatch_mode == "merge":
            return None
        if self.dispatch_mode.startswith("branch"):
            idx = int(self.dispatch_mode[len("branch") :])
            if idx < 0 or idx >= self.max_k:
                raise ValueError(
                    f"dispatch_mode {self.dispatch_mode!r} is invalid for max_k={self.max_k}"
                )
            return idx
        raise ValueError(f"unknown dispatch_mode {self.dispatch_mode!r}")

    def _logits_from_hidden(self, h: torch.Tensor) -> torch.Tensor:
        return self.head(self.norm_out(h))

    @staticmethod
    def gather_at(logits: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        idx = pos.view(-1, 1, 1).expand(-1, 1, logits.size(-1))
        return logits.gather(dim=1, index=idx).squeeze(1)

    def anneal_temperature(self):
        self.router.anneal_temperature()

    def _branch_aux_loss(
        self,
        candidates: List[torch.Tensor],
        targets: torch.Tensor,
        target_pos: torch.Tensor,
    ) -> torch.Tensor:
        losses = []
        for candidate in candidates:
            logits = self._logits_from_hidden(candidate)
            selected = self.gather_at(logits, target_pos)
            losses.append(F.cross_entropy(selected, targets, reduction="none"))
        return torch.stack(losses, dim=-1).mean()

    def forward(
        self,
        x: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        target_pos: Optional[torch.Tensor] = None,
        force_main_actions: Optional[List[str]] = None,
    ) -> torch.Tensor:
        B, T = x.shape
        device = x.device
        self.branch_loss = torch.zeros((), device=device)
        branch_losses = []

        pos = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
        h = self.emb_drop(self.token_emb(x) + self.pos_emb(pos))
        for block in self.base_blocks:
            h = block(h)

        self.trace = []
        self.last_main_action_probs = []
        self.last_main_action_idx = []
        self.last_merge_weights = []
        self.last_branch_deltas = []

        for step in range(self.max_recurrent_steps):
            forced = None
            if force_main_actions and step < len(force_main_actions):
                forced = force_main_actions[step]

            action_idx, action_one_hot, info = self.router(h, force_action=forced)
            self.last_main_action_probs.append(self.router.last_action_probs)
            self.last_main_action_idx.append(action_idx.detach())
            self.trace.append({"step": step, "stage": "main", **info})

            all_continue = bool((action_idx == DifficultyRouter.ACT_CONTINUE).all().item())
            all_dispatch = bool((action_idx == DifficultyRouter.ACT_DISPATCH).all().item())

            if self.share_continue_weights:
                h_continue = self.continue_block(h)
            else:
                h_continue = self.continue_blocks[step](h)

            if all_continue:
                h_dispatch = h
            else:
                candidates = []
                deltas = []
                for i, branch in enumerate(self.branches):
                    candidate, delta, _ = branch(h, self.role_emb[i])
                    candidates.append(candidate)
                    deltas.append(delta)

                fixed_branch_idx = self._fixed_dispatch_branch_idx()
                if fixed_branch_idx is None:
                    h_dispatch, merge_weights = self.merger(h, candidates, deltas)
                else:
                    h_dispatch = candidates[fixed_branch_idx]
                    merge_weights = torch.zeros(B, len(candidates), device=device)
                    merge_weights[:, fixed_branch_idx] = 1.0
                self.last_merge_weights.append(merge_weights)
                self.last_branch_deltas.append(torch.stack(deltas, dim=1))
                self.trace.append(
                    {
                        "step": step,
                        "stage": "merge",
                        "merge_weights": [
                            round(float(v), 3)
                            for v in merge_weights.mean(dim=0).detach().cpu()
                        ],
                    }
                )

                if targets is not None and target_pos is not None:
                    branch_losses.append(
                        self._branch_aux_loss(candidates, targets, target_pos)
                    )

            w_continue = action_one_hot[:, DifficultyRouter.ACT_CONTINUE].view(B, 1, 1)
            w_dispatch = action_one_hot[:, DifficultyRouter.ACT_DISPATCH].view(B, 1, 1)
            if all_dispatch:
                h = h_dispatch
            elif all_continue:
                h = h_continue
            else:
                h = w_continue * h_continue + w_dispatch * h_dispatch

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

    def merge_entropy(self) -> torch.Tensor:
        device = next(self.parameters()).device
        if not self.last_merge_weights:
            return torch.zeros((), device=device)
        total = torch.zeros((), device=device)
        for w in self.last_merge_weights:
            total = total - (w * torch.log(w.clamp_min(1e-9))).sum(dim=-1).mean()
        return total / len(self.last_merge_weights)


def _demo():
    torch.manual_seed(0)
    model = ReGSMMergeModel(
        vocab_size=100,
        d_model=64,
        num_heads=4,
        d_ff=128,
        num_base_layers=2,
        max_recurrent_steps=2,
        max_branch_steps=1,
        max_k=3,
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
    print("merge_entropy:", float(model.merge_entropy().detach()))
    print("trace entries:", len(model.trace))


if __name__ == "__main__":
    _demo()
