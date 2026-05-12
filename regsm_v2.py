
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


class MainRouter(nn.Module):

    ACTION_NAMES = ["CONTINUE", "DISPATCH", "HALT"]
    NUM_ACTIONS = 3
    ACT_CONTINUE, ACT_DISPATCH, ACT_HALT = 0, 1, 2

    def __init__(
        self,
        d_model: int,
        max_k: int = 4,
        hidden_dim: int = 128,
        init_temp: float = 1.0,
        min_temp: float = 0.1,
        anneal_rate: float = 3e-5,
    ):
        super().__init__()
        assert max_k >= 2
        self.max_k = max_k
        self.action_head = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.NUM_ACTIONS),
        )
        self.k_head = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, max_k - 1),
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
        disable_halt: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        pooled = x.mean(dim=1)
        action_logits = self.action_head(pooled)
        k_logits = self.k_head(pooled)
        if disable_halt:
            action_logits = action_logits.clone()
            action_logits[:, self.ACT_HALT] = float("-inf")
        tau = float(self.temperature.detach())

        action_one_hot = F.gumbel_softmax(
            action_logits, tau=tau, hard=hard, dim=-1
        )
        k_one_hot = F.gumbel_softmax(k_logits, tau=tau, hard=hard, dim=-1)

        if force_action is not None:
            forced = force_action.upper()
            if forced == "SPLIT":
                forced = "DISPATCH"
            forced_idx = self.ACTION_NAMES.index(forced)
            forced_one_hot = torch.zeros_like(action_one_hot)
            forced_one_hot[:, forced_idx] = 1.0
            action_one_hot = action_one_hot + (forced_one_hot - action_one_hot).detach()

        action_idx = action_one_hot.argmax(dim=-1)
        k_idx = k_one_hot.argmax(dim=-1) + 2
        self.last_action_probs = F.softmax(action_logits, dim=-1)

        action_counts = {name: 0 for name in self.ACTION_NAMES}
        for ai in action_idx.detach().cpu().tolist():
            action_counts[self.ACTION_NAMES[ai]] += 1

        k_counts: Dict[int, int] = {}
        for ai, ki in zip(action_idx.detach().cpu().tolist(), k_idx.detach().cpu().tolist()):
            if ai == self.ACT_DISPATCH:
                k_counts[int(ki)] = k_counts.get(int(ki), 0) + 1

        modal_action_idx = max(
            range(self.NUM_ACTIONS),
            key=lambda i: action_counts[self.ACTION_NAMES[i]],
        )
        info = {
            "action": self.ACTION_NAMES[modal_action_idx],
            "action_idx": modal_action_idx,
            "action_counts": action_counts,
            "k_counts": k_counts,
            "action_probs": [
                round(float(p), 3)
                for p in self.last_action_probs.mean(dim=0).detach().cpu()
            ],
            "k_probs": [
                round(float(p), 3)
                for p in F.softmax(k_logits, dim=-1).mean(dim=0).detach().cpu()
            ],
            "temperature": round(tau, 4),
        }
        if force_action is not None:
            info["forced"] = True
        return action_idx, action_one_hot, k_idx, k_one_hot, info


class ExpertPool(nn.Module):

    def __init__(self, num_experts: int, d_model: int, d_ff: int):
        super().__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_model, d_ff),
                    nn.GELU(),
                    nn.Linear(d_ff, d_model),
                )
                for _ in range(num_experts)
            ]
        )


class BranchExpertRouter(nn.Module):

    def __init__(self, d_model: int, num_experts: int, top_k: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(d_model, num_experts)

    def forward(self, x: torch.Tensor):
        pooled = x.mean(dim=1)
        gate_logits = self.gate(pooled)
        gate_probs = F.softmax(gate_logits, dim=-1)
        top_logits, top_idx = gate_logits.topk(self.top_k, dim=-1)
        top_w = F.softmax(top_logits, dim=-1)
        return top_idx, top_w, gate_probs


class BranchBlock(nn.Module):

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        expert_pool: ExpertPool,
        num_experts: int,
        top_k: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.expert_pool = expert_pool
        self.expert_router = BranchExpertRouter(d_model, num_experts, top_k)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        h = self.norm1(x)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.dropout(a)

        h = self.norm2(x)
        idx, w, gate_probs = self.expert_router(h)
        moe_out = torch.zeros_like(h)
        for eid, expert in enumerate(self.expert_pool.experts):
            match = idx == eid
            if not match.any():
                continue
            match_b, match_k = match.nonzero(as_tuple=True)
            weights = w[match_b, match_k]
            expert_out = expert(h[match_b])
            moe_out.index_add_(0, match_b, expert_out * weights.view(-1, 1, 1))
        return x + self.dropout(moe_out), gate_probs, idx


class BranchSelector(nn.Module):

    def __init__(
        self,
        d_model: int,
        max_k: int,
        hidden_dim: int = 128,
        init_temp: float = 1.0,
        min_temp: float = 0.1,
        anneal_rate: float = 3e-5,
    ):
        super().__init__()
        self.max_k = max_k
        self.scorer = nn.Sequential(
            nn.Linear(3 * d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.register_buffer("temperature", torch.tensor(float(init_temp)))
        self.min_temp = float(min_temp)
        self.anneal_rate = float(anneal_rate)
        self.last_probs: Optional[torch.Tensor] = None

    def anneal_temperature(self):
        self.temperature.mul_(math.exp(-self.anneal_rate))
        self.temperature.clamp_(min=self.min_temp)

    def set_temperature(self, value: float):
        self.temperature.fill_(float(value))

    def forward(
        self,
        trunk: torch.Tensor,
        branches: List[torch.Tensor],
        branch_weights: Optional[torch.Tensor] = None,
        hard: bool = True,
        detach_score_inputs: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        score_trunk = trunk.detach() if detach_score_inputs else trunk
        score_branches = [b.detach() for b in branches] if detach_score_inputs else branches
        trunk_pool = score_trunk.mean(dim=1)
        branch_pool = torch.stack([b.mean(dim=1) for b in score_branches], dim=1)
        trunk_expand = trunk_pool.unsqueeze(1).expand_as(branch_pool)
        features = torch.cat(
            [trunk_expand, branch_pool, branch_pool - trunk_expand], dim=-1
        )
        scores = self.scorer(features).squeeze(-1)
        if branch_weights is not None:
            scores = scores + torch.log(branch_weights.clamp_min(1e-9))

        tau = float(self.temperature.detach())
        select_one_hot = F.gumbel_softmax(scores, tau=tau, hard=hard, dim=-1)
        self.last_probs = F.softmax(scores, dim=-1)
        selected_idx = select_one_hot.argmax(dim=-1)
        stacked = torch.stack(branches, dim=1)
        selected = (stacked * select_one_hot.view(-1, len(branches), 1, 1)).sum(dim=1)
        return selected, select_one_hot, selected_idx


def _active_branch_weights(k_one_hot: torch.Tensor) -> torch.Tensor:
    rev_cumsum = torch.flip(
        torch.cumsum(torch.flip(k_one_hot, dims=[-1]), dim=-1),
        dims=[-1],
    )
    return torch.cat([rev_cumsum[..., :1], rev_cumsum], dim=-1)


class ReGSMSelectModel(nn.Module):

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
        num_experts: int = 32,
        top_k_experts: int = 2,
        max_seq_len: int = 128,
        dropout: float = 0.1,
        verbose: bool = False,
        continue_mode: str = "full",
        share_continue_weights: bool = True,
        diverse_branch_input: bool = True,
        diverse_branch_init_noise: float = 0.05,
        disable_halt: bool = False,
        train_oracle_select: bool = False,
        selector_stopgrad: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_recurrent_steps = max_recurrent_steps
        self.max_branch_steps = max_branch_steps
        self.max_k = max_k
        self.verbose = verbose
        self.continue_mode = continue_mode
        self.share_continue_weights = share_continue_weights
        self.disable_halt = disable_halt
        self.train_oracle_select = train_oracle_select
        self.selector_stopgrad = selector_stopgrad

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
            raise ValueError(
                "continue_mode must be 'full', 'linear', or 'identity', "
                f"got {continue_mode!r}"
            )

        if share_continue_weights:
            self.continue_block = make_continue()
            self.continue_blocks = None
        else:
            self.continue_block = None
            self.continue_blocks = nn.ModuleList(
                [make_continue() for _ in range(max_recurrent_steps)]
            )

        self.expert_pool = ExpertPool(num_experts, d_model, d_ff)
        self.branch_blocks = nn.ModuleList(
            [
                BranchBlock(
                    d_model,
                    num_heads,
                    self.expert_pool,
                    num_experts,
                    top_k=top_k_experts,
                    dropout=dropout,
                )
                for _ in range(max_k)
            ]
        )

        self.diverse_branch_input = diverse_branch_input
        if diverse_branch_input:
            self.branch_input_projs = nn.ModuleList(
                [nn.Linear(d_model, d_model) for _ in range(max_k)]
            )
            with torch.no_grad():
                for proj in self.branch_input_projs:
                    nn.init.eye_(proj.weight)
                    proj.weight.add_(
                        torch.randn_like(proj.weight) * diverse_branch_init_noise
                    )
                    nn.init.zeros_(proj.bias)
        else:
            self.branch_input_projs = None

        self.router = MainRouter(d_model, max_k=max_k)
        self.selector = BranchSelector(d_model, max_k=max_k)
        self.norm_out = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

        self.trace: List[Dict] = []
        self.last_main_action_probs: List[torch.Tensor] = []
        self.last_selector_probs: List[torch.Tensor] = []
        self.last_branch_gate_probs: List[List[List[torch.Tensor]]] = []
        self.selector_loss = torch.zeros(())
        self.branch_loss = torch.zeros(())

    def _log(self, msg: str):
        if self.verbose:
            print(msg)

    def anneal_temperature(self):
        self.router.anneal_temperature()
        self.selector.anneal_temperature()

    def _logits_from_hidden(self, h: torch.Tensor) -> torch.Tensor:
        return self.head(self.norm_out(h))

    @staticmethod
    def gather_at(logits: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        idx = pos.view(-1, 1, 1).expand(-1, 1, logits.size(-1))
        return logits.gather(dim=1, index=idx).squeeze(1)

    def _branch_supervision(
        self,
        branches: List[torch.Tensor],
        targets: torch.Tensor,
        target_pos: torch.Tensor,
        branch_weights: torch.Tensor,
        selector_probs: torch.Tensor,
        sample_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        per_branch_losses = []
        for branch in branches:
            logits = self._logits_from_hidden(branch)
            selected_logits = self.gather_at(logits, target_pos)
            losses = F.cross_entropy(selected_logits, targets, reduction="none")
            per_branch_losses.append(losses)

        loss_matrix = torch.stack(per_branch_losses, dim=-1)
        masked_loss = loss_matrix.masked_fill(branch_weights <= 1e-6, 1e9)
        best_branch = masked_loss.detach().argmin(dim=-1)
        if sample_mask is None:
            sample_mask = torch.ones_like(targets, dtype=loss_matrix.dtype)
        else:
            sample_mask = sample_mask.to(dtype=loss_matrix.dtype)
        denom = sample_mask.sum().clamp_min(1.0)

        selector_loss_per_sample = F.nll_loss(
            torch.log(selector_probs.clamp_min(1e-9)),
            best_branch,
            reduction="none",
        )
        selector_loss = (selector_loss_per_sample * sample_mask).sum() / denom

        active_loss_per_sample = (loss_matrix * branch_weights).sum(dim=-1) / (
            branch_weights.sum(dim=-1).clamp_min(1.0)
        )
        active_loss = (active_loss_per_sample * sample_mask).sum() / denom
        return selector_loss, active_loss, best_branch

    def forward(
        self,
        x: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        target_pos: Optional[torch.Tensor] = None,
        force_main_actions: Optional[List[str]] = None,
        force_oracle_select: bool = False,
    ) -> torch.Tensor:
        B, T = x.shape
        device = x.device
        zero = torch.zeros((), device=device)
        self.selector_loss = zero
        self.branch_loss = zero

        pos = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
        h = self.emb_drop(self.token_emb(x) + self.pos_emb(pos))
        for block in self.base_blocks:
            h = block(h)

        self.trace = []
        self.last_main_action_probs = []
        self.last_selector_probs = []
        self.last_branch_gate_probs = []
        selector_losses = []
        branch_losses = []

        for step in range(self.max_recurrent_steps):
            forced = None
            if force_main_actions and step < len(force_main_actions):
                forced = force_main_actions[step]

            action_idx, action_one_hot, _, k_one_hot, info = self.router(
                h, force_action=forced, disable_halt=self.disable_halt
            )
            self.last_main_action_probs.append(self.router.last_action_probs)
            self.trace.append({"step": step, "stage": "main", **info})

            all_continue = bool((action_idx == MainRouter.ACT_CONTINUE).all().item())
            all_dispatch = bool((action_idx == MainRouter.ACT_DISPATCH).all().item())
            all_halt = bool((action_idx == MainRouter.ACT_HALT).all().item())
            if all_halt:
                self._log(f"[Step {step}] all HALT")
                break

            if not (all_dispatch or all_halt):
                if self.share_continue_weights:
                    h_continue = self.continue_block(h)
                else:
                    h_continue = self.continue_blocks[step](h)
            else:
                h_continue = None

            if not (all_continue or all_halt):
                branches = []
                split_gate_probs: List[List[torch.Tensor]] = []
                for bi, block in enumerate(self.branch_blocks):
                    branch_h = (
                        self.branch_input_projs[bi](h)
                        if self.diverse_branch_input
                        else h.clone()
                    )
                    branch_probs = []
                    for bs in range(self.max_branch_steps):
                        branch_h, gate_probs, top_idx = block(branch_h)
                        branch_probs.append(gate_probs)
                        self.trace.append(
                            {
                                "step": step,
                                "stage": f"branch{bi}.{bs}",
                                "top_experts_sample0": top_idx[0].detach().cpu().tolist()
                                if B > 0
                                else [],
                            }
                        )
                    branches.append(branch_h)
                    split_gate_probs.append(branch_probs)
                self.last_branch_gate_probs.append(split_gate_probs)

                branch_weights = _active_branch_weights(k_one_hot)
                detach_selector_scores = (
                    self.selector_stopgrad
                    and self.training
                    and targets is not None
                    and target_pos is not None
                )
                h_dispatch, select_one_hot, selected_idx = self.selector(
                    h,
                    branches,
                    branch_weights=branch_weights,
                    detach_score_inputs=detach_selector_scores,
                )
                selector_probs = self.selector.last_probs
                self.last_selector_probs.append(selector_probs)

                selector_pred_idx = selected_idx
                oracle_idx = None
                dispatch_mask = action_idx == MainRouter.ACT_DISPATCH
                if targets is not None and target_pos is not None:
                    sel_loss, br_loss, best_branch = self._branch_supervision(
                        branches,
                        targets,
                        target_pos,
                        branch_weights,
                        selector_probs,
                        sample_mask=dispatch_mask,
                    )
                    selector_losses.append(sel_loss)
                    branch_losses.append(br_loss)
                    oracle_idx = best_branch
                    use_oracle_select = force_oracle_select or (
                        self.train_oracle_select and self.training
                    )
                    if use_oracle_select:
                        oracle_one_hot = F.one_hot(
                            best_branch, num_classes=len(branches)
                        ).to(dtype=branches[0].dtype)
                        stacked = torch.stack(branches, dim=1)
                        h_dispatch = (
                            stacked * oracle_one_hot.view(B, len(branches), 1, 1)
                        ).sum(dim=1)
                        selected_idx = best_branch

                selected_counts = {}
                for i in selected_idx.detach().cpu().tolist():
                    selected_counts[int(i)] = selected_counts.get(int(i), 0) + 1
                selector_pred_counts = {}
                for i in selector_pred_idx.detach().cpu().tolist():
                    selector_pred_counts[int(i)] = selector_pred_counts.get(int(i), 0) + 1
                dispatch_selected_counts = {}
                for i in selected_idx[dispatch_mask].detach().cpu().tolist():
                    dispatch_selected_counts[int(i)] = (
                        dispatch_selected_counts.get(int(i), 0) + 1
                    )
                dispatch_selector_pred_counts = {}
                for i in selector_pred_idx[dispatch_mask].detach().cpu().tolist():
                    dispatch_selector_pred_counts[int(i)] = (
                        dispatch_selector_pred_counts.get(int(i), 0) + 1
                    )
                trace_item = {
                    "step": step,
                    "stage": "selector",
                    "selected_counts": selected_counts,
                    "selector_pred_counts": selector_pred_counts,
                    "dispatch_selected_counts": dispatch_selected_counts,
                    "dispatch_selector_pred_counts": dispatch_selector_pred_counts,
                    "selector_probs": [
                        round(float(p), 3)
                        for p in selector_probs.mean(dim=0).detach().cpu()
                    ],
                }
                if oracle_idx is not None:
                    oracle_counts = {}
                    for i in oracle_idx.detach().cpu().tolist():
                        oracle_counts[int(i)] = oracle_counts.get(int(i), 0) + 1
                    trace_item["oracle_counts"] = oracle_counts
                    dispatch_oracle_counts = {}
                    for i in oracle_idx[dispatch_mask].detach().cpu().tolist():
                        dispatch_oracle_counts[int(i)] = (
                            dispatch_oracle_counts.get(int(i), 0) + 1
                        )
                    trace_item["dispatch_oracle_counts"] = dispatch_oracle_counts
                    dispatch_n = int(dispatch_mask.sum().detach().cpu())
                    if dispatch_n > 0:
                        pred_acc = (
                            selector_pred_idx[dispatch_mask] == oracle_idx[dispatch_mask]
                        ).float().mean()
                        trace_item["dispatch_selector_oracle_acc"] = round(
                            float(pred_acc.detach().cpu()), 4
                        )
                    trace_item["oracle_select"] = bool(use_oracle_select)
                self.trace.append(trace_item)
            else:
                self.last_branch_gate_probs.append([])
                h_dispatch = None

            w_continue = action_one_hot[:, MainRouter.ACT_CONTINUE].view(B, 1, 1)
            w_dispatch = action_one_hot[:, MainRouter.ACT_DISPATCH].view(B, 1, 1)
            w_halt = action_one_hot[:, MainRouter.ACT_HALT].view(B, 1, 1)

            h_next = w_halt * h
            if h_continue is not None:
                h_next = h_next + w_continue * h_continue
            else:
                h_next = h_next + w_continue * h
            if h_dispatch is not None:
                h_next = h_next + w_dispatch * h_dispatch
            else:
                h_next = h_next + w_dispatch * h
            h = h_next

        if selector_losses:
            self.selector_loss = torch.stack(selector_losses).mean()
        if branch_losses:
            self.branch_loss = torch.stack(branch_losses).mean()

        return self._logits_from_hidden(h)

    def expert_diversity_loss(self) -> torch.Tensor:
        device = next(self.parameters()).device
        if not self.last_branch_gate_probs:
            return torch.zeros((), device=device)

        total = torch.zeros((), device=device)
        n = 0
        for split in self.last_branch_gate_probs:
            per_branch_dist = []
            for branch_probs in split:
                if branch_probs:
                    per_branch_dist.append(
                        torch.stack(branch_probs, dim=0).mean(dim=(0, 1))
                    )
            for i in range(len(per_branch_dist)):
                for j in range(i + 1, len(per_branch_dist)):
                    total = total + F.cosine_similarity(
                        per_branch_dist[i].unsqueeze(0),
                        per_branch_dist[j].unsqueeze(0),
                    ).squeeze()
                    n += 1
        return total / max(n, 1)


def _demo():
    torch.manual_seed(0)
    model = ReGSMSelectModel(
        vocab_size=100,
        d_model=64,
        num_heads=4,
        d_ff=128,
        num_base_layers=2,
        max_recurrent_steps=3,
        max_branch_steps=2,
        max_k=3,
        num_experts=8,
        max_seq_len=16,
        dropout=0.0,
        verbose=True,
    )
    x = torch.randint(0, 100, (4, 16))
    y = torch.randint(0, 100, (4,))
    target_pos = torch.full((4,), 15, dtype=torch.long)
    logits = model(x, targets=y, target_pos=target_pos)
    print("logits:", tuple(logits.shape))
    print("selector_loss:", float(model.selector_loss.detach()))
    print("branch_loss:", float(model.branch_loss.detach()))
    print("trace entries:", len(model.trace))


if __name__ == "__main__":
    _demo()
