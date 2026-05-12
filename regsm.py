import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerBlock(nn.Module):

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads,
                                          dropout=dropout, batch_first=True)
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
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.dropout(attn_out)
        h = self.norm2(x)
        x = x + self.dropout(self.ffn(h))
        return x


class DynamicRouter(nn.Module):

    ACTION_NAMES = ["CONTINUE", "SPLIT", "MERGE"]
    NUM_ACTIONS = 3
    ACT_CONTINUE, ACT_SPLIT, ACT_MERGE = 0, 1, 2

    def __init__(
        self,
        d_model: int,
        max_k: int = 4,
        hidden_dim: int = 128,
        init_temp: float = 1.0,
        min_temp: float = 0.1,
        anneal_rate: float = 3e-5,
        use_task_masks: bool = False,
        task_mask_init: float = 0.98,
    ):
        super().__init__()
        assert max_k >= 2
        self.max_k = max_k
        self.d_model = d_model
        self.use_task_masks = use_task_masks

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

        if use_task_masks:
            init_p = min(max(float(task_mask_init), 1e-4), 1.0 - 1e-4)
            init_logit = math.log(init_p / (1.0 - init_p))
            self.task_mask_head = nn.Sequential(
                nn.Linear(d_model, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, max_k * d_model),
            )
            last = self.task_mask_head[-1]
            nn.init.zeros_(last.weight)
            nn.init.constant_(last.bias, init_logit)
        else:
            self.task_mask_head = None
        self.last_task_masks: Optional[torch.Tensor] = None

        self.register_buffer("temperature", torch.tensor(float(init_temp)))
        self.min_temp = float(min_temp)
        self.anneal_rate = float(anneal_rate)

    def anneal_temperature(self):
        self.temperature.mul_(math.exp(-self.anneal_rate))
        self.temperature.clamp_(min=self.min_temp)

    def set_temperature(self, value: float):
        self.temperature.fill_(float(value))

    def _summary(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=1)

    def forward(
        self,
        x: torch.Tensor,
        in_branch: bool = False,
        hard: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        s = self._summary(x)
        a_logits = self.action_head(s)
        k_logits = self.k_head(s)
        if self.task_mask_head is not None and not in_branch:
            task_mask_logits = self.task_mask_head(s).view(
                x.size(0), self.max_k, self.d_model
            )
            self.last_task_masks = torch.sigmoid(task_mask_logits)
        else:
            self.last_task_masks = None

        mask = torch.zeros_like(a_logits)
        if in_branch:
            mask[:, self.ACT_SPLIT] = float("-inf")
        else:
            mask[:, self.ACT_MERGE] = float("-inf")
        a_logits = a_logits + mask

        tau = self.temperature.item()
        a_one_hot = F.gumbel_softmax(a_logits, tau=tau, hard=hard, dim=-1)
        k_one_hot = F.gumbel_softmax(k_logits, tau=tau, hard=hard, dim=-1)

        action_idx = a_one_hot.argmax(dim=-1)
        k_idx      = k_one_hot.argmax(dim=-1) + 2

        self.last_action_probs = F.softmax(a_logits, dim=-1)

        per_sample_actions = action_idx.detach().cpu().tolist()
        action_counts = {n: 0 for n in self.ACTION_NAMES}
        for ai in per_sample_actions:
            action_counts[self.ACTION_NAMES[ai]] += 1

        modal_action_idx = max(
            range(self.NUM_ACTIONS),
            key=lambda i: action_counts[self.ACTION_NAMES[i]],
        )
        modal_action = self.ACTION_NAMES[modal_action_idx]

        k_per_sample = k_idx.detach().cpu().tolist()
        k_counts: Dict[int, int] = {}
        for ai, ki in zip(per_sample_actions, k_per_sample):
            if ai == self.ACT_SPLIT:
                k_counts[ki] = k_counts.get(ki, 0) + 1
        if k_counts:
            modal_k = int(max(k_counts, key=k_counts.get))
        else:
            modal_k = int(k_per_sample[0]) if k_per_sample else 2

        a_probs_mean = F.softmax(a_logits, dim=-1).mean(dim=0)
        k_probs_mean = F.softmax(k_logits, dim=-1).mean(dim=0)

        info = {
            "action": modal_action,
            "action_idx": modal_action_idx,
            "k": modal_k,
            "action_counts": action_counts,
            "k_counts": k_counts,
            "action_probs": [round(p, 3) for p in
                             a_probs_mean.detach().cpu().tolist()],
            "k_probs": [round(p, 3) for p in
                        k_probs_mean.detach().cpu().tolist()],
            "temperature": round(tau, 4),
            "in_branch": in_branch,
        }
        if self.last_task_masks is not None:
            with torch.no_grad():
                tm = self.last_task_masks
                branch_mean = tm.mean(dim=(0, 2)).detach().cpu().tolist()
                info["task_mask_mean"] = round(float(tm.mean().item()), 4)
                info["task_mask_branch_mean"] = [
                    round(float(v), 4) for v in branch_mean
                ]
        return action_idx, a_one_hot, k_idx, k_one_hot, info


class ExpertPool(nn.Module):

    def __init__(self, num_experts: int, d_model: int, d_ff: int):
        super().__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.GELU(),
                nn.Linear(d_ff, d_model),
            )
            for _ in range(num_experts)
        ])


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
        self.attn = nn.MultiheadAttention(d_model, num_heads,
                                          dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.expert_pool = expert_pool
        self.expert_router = BranchExpertRouter(d_model, num_experts, top_k)
        self.dropout = nn.Dropout(dropout)
        self.top_k = top_k

    def forward(self, x: torch.Tensor):
        h = self.norm1(x)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.dropout(a)

        h = self.norm2(x)
        idx, w, gate_probs = self.expert_router(h)

        moe_out = torch.zeros_like(h)
        for eid, expert in enumerate(self.expert_pool.experts):
            match = (idx == eid)
            if not match.any():
                continue
            match_b, match_k = match.nonzero(as_tuple=True)
            weights = w[match_b, match_k]
            expert_in = h[match_b]
            expert_out = expert(expert_in)
            moe_out.index_add_(
                0, match_b, expert_out * weights.view(-1, 1, 1)
            )
        x = x + self.dropout(moe_out)
        return x, gate_probs, idx


class CrossAttentionMerger(nn.Module):

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads,
                                                dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.gate = nn.Linear(d_model, 1)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        branches: List[torch.Tensor],
        branch_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        K = len(branches)
        if K == 1:
            return branches[0]

        kv = torch.cat(branches, dim=1)

        attended = []
        for q in branches:
            a, _ = self.cross_attn(q, kv, kv, need_weights=False)
            attended.append(self.norm(q + self.dropout(a)))

        gate_logits = torch.stack(
            [self.gate(a).mean(dim=1) for a in attended], dim=-1
        )

        if branch_weights is not None:
            log_w = torch.log(branch_weights.clamp_min(1e-9)).unsqueeze(1)
            gate_logits = gate_logits + log_w

        gate_w = F.softmax(gate_logits, dim=-1)

        stacked = torch.stack(attended, dim=-1)
        fused = (stacked * gate_w.unsqueeze(2)).sum(dim=-1)
        return self.out_proj(fused)


class ReGSMModel(nn.Module):

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        num_heads: int = 4,
        d_ff: int = 512,
        num_base_layers: int = 2,
        max_recurrent_steps: int = 3,
        max_branch_steps: int = 3,
        max_k: int = 4,
        num_experts: int = 64,
        top_k_experts: int = 2,
        max_seq_len: int = 128,
        dropout: float = 0.1,
        verbose: bool = False,
        continue_mode: str = "full",
        share_continue_weights: bool = True,
        diverse_branch_input: bool = False,
        diverse_branch_init_noise: float = 0.05,
        router_task_masks: bool = False,
        task_mask_init: float = 0.98,
        task_mask_residual_alpha: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_recurrent_steps = max_recurrent_steps
        self.max_branch_steps = max_branch_steps
        self.max_k = max_k
        self.verbose = verbose
        self.task_mask_residual_alpha = float(task_mask_residual_alpha)

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.emb_drop = nn.Dropout(dropout)

        self.base_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_base_layers)
        ])

        self.continue_mode = continue_mode
        self.share_continue_weights = share_continue_weights

        def _make_continue():
            if continue_mode == "full":
                return TransformerBlock(d_model, num_heads, d_ff, dropout)
            elif continue_mode == "linear":
                return nn.Sequential(
                    nn.LayerNorm(d_model),
                    nn.Linear(d_model, d_model),
                )
            elif continue_mode == "identity":
                return nn.Identity()
            else:
                raise ValueError(
                    f"continue_mode must be 'full' | 'linear' | 'identity', "
                    f"got {continue_mode!r}"
                )

        if share_continue_weights:
            self.continue_block = _make_continue()
            self.continue_blocks = None
        else:
            self.continue_blocks = nn.ModuleList(
                [_make_continue() for _ in range(max_recurrent_steps)]
            )
            self.continue_block = None

        self.expert_pool = ExpertPool(num_experts, d_model, d_ff)
        self.branch_blocks = nn.ModuleList([
            BranchBlock(d_model, num_heads, self.expert_pool,
                        num_experts, top_k=top_k_experts, dropout=dropout)
            for _ in range(max_k)
        ])

        self.diverse_branch_input = diverse_branch_input
        if diverse_branch_input:
            self.branch_input_projs = nn.ModuleList([
                nn.Linear(d_model, d_model) for _ in range(max_k)
            ])
            with torch.no_grad():
                for proj in self.branch_input_projs:
                    nn.init.eye_(proj.weight)
                    proj.weight.add_(
                        torch.randn_like(proj.weight) * diverse_branch_init_noise
                    )
                    nn.init.zeros_(proj.bias)
        else:
            self.branch_input_projs = None

        self.merger = CrossAttentionMerger(d_model, num_heads, dropout)

        self.router_task_masks = router_task_masks
        self.router = DynamicRouter(
            d_model,
            max_k=max_k,
            use_task_masks=router_task_masks,
            task_mask_init=task_mask_init,
        )

        self.norm_out = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

        self.trace: List[Dict] = []
        self.last_branch_gate_probs: List[List[List[torch.Tensor]]] = []
        self.last_main_action_probs: List[torch.Tensor] = []
        self.last_main_task_masks: List[torch.Tensor] = []

    def _log(self, msg: str):
        if self.verbose:
            print(msg)

    def forward(
        self,
        x: torch.Tensor,
        force_main_actions: Optional[List[str]] = None,
    ) -> torch.Tensor:
        B, T = x.shape
        device = x.device
        positions = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
        h = self.emb_drop(self.token_emb(x) + self.pos_emb(positions))
        self._log(f"[Embed]      shape={tuple(h.shape)}")

        for i, block in enumerate(self.base_blocks):
            h = block(h)
            self._log(f"[Base #{i}]   shape={tuple(h.shape)}")

        self.trace = []
        self.last_branch_gate_probs = []
        self.last_main_action_probs = []
        self.last_main_task_masks = []

        for step in range(self.max_recurrent_steps):
            action_idx, a_one_hot, k_idx, k_one_hot, info = \
                self.router(h, in_branch=False)
            self.last_main_action_probs.append(self.router.last_action_probs)
            task_masks = self.router.last_task_masks
            if task_masks is not None:
                self.last_main_task_masks.append(task_masks)

            if force_main_actions and step < len(force_main_actions):
                forced = force_main_actions[step].upper()
                forced_idx = DynamicRouter.ACTION_NAMES.index(forced)
                forced_one_hot = torch.zeros_like(a_one_hot)
                forced_one_hot[:, forced_idx] = 1.0
                a_one_hot = a_one_hot + (forced_one_hot - a_one_hot).detach()
                action_idx = a_one_hot.argmax(dim=-1)
                info["action"] = forced
                info["action_idx"] = forced_idx
                info["action_counts"] = {
                    n: (B if n == forced else 0)
                    for n in DynamicRouter.ACTION_NAMES
                }
                info["forced"] = True

            self.trace.append({"step": step, "stage": "main", **info})
            self._log(
                f"[Step {step}]    main router -> "
                f"action_counts={info['action_counts']} "
                f"K_dist={info['k_counts']} "
                f"tau={info['temperature']}"
            )

            all_continue = bool(
                (action_idx == DynamicRouter.ACT_CONTINUE).all().item()
            )
            all_split = bool(
                (action_idx == DynamicRouter.ACT_SPLIT).all().item()
            )

            if not all_split:
                if self.share_continue_weights:
                    h_continue = self.continue_block(h)
                else:
                    h_continue = self.continue_blocks[step](h)
            else:
                h_continue = None

            if not all_continue:
                K = self.max_k
                self._log(f"           SPLIT path uses {K} parallel branches (max_k)")
                branches = []
                for bi in range(K):
                    branch_h0 = (
                        self.branch_input_projs[bi](h)
                        if self.diverse_branch_input
                        else h.clone()
                    )
                    if task_masks is not None:
                        mask_i = task_masks[:, bi, :]
                        if self.task_mask_residual_alpha > 0.0:
                            sample_mean = task_masks.mean(dim=(1, 2)).unsqueeze(1)
                            centered = mask_i - sample_mean
                            gate = 1.0 + self.task_mask_residual_alpha * centered
                            branch_h0 = branch_h0 * gate.unsqueeze(1)
                        else:
                            branch_h0 = branch_h0 * mask_i.unsqueeze(1)
                    branches.append(branch_h0)
                split_gate_probs: List[List[torch.Tensor]] = []

                for bi in range(K):
                    block_for_branch = self.branch_blocks[bi]
                    branch_h = branches[bi]
                    branch_probs: List[torch.Tensor] = []

                    for bs in range(self.max_branch_steps):
                        b_act_idx, b_one_hot, _, _, b_info = \
                            self.router(branch_h, in_branch=True)
                        self.trace.append({
                            "step": step,
                            "stage": f"branch{bi}.{bs}",
                            **b_info,
                        })
                        self._log(
                            f"             branch[{bi}] inner {bs} -> "
                            f"action_counts={b_info['action_counts']}"
                        )

                        branch_h_next, gate_probs, top_idx = \
                            block_for_branch(branch_h)
                        branch_probs.append(gate_probs)
                        self._log(
                            f"               branch[{bi}] picked experts "
                            f"{top_idx[0].tolist()} (sample 0)"
                        )

                        w_continue = b_one_hot[:, DynamicRouter.ACT_CONTINUE].view(-1, 1, 1)
                        w_merge    = b_one_hot[:, DynamicRouter.ACT_MERGE].view(-1, 1, 1)
                        branch_h = w_merge * branch_h + w_continue * branch_h_next

                    branches[bi] = branch_h
                    split_gate_probs.append(branch_probs)

                self.last_branch_gate_probs.append(split_gate_probs)

                rev_cumsum = torch.flip(
                    torch.cumsum(torch.flip(k_one_hot, dims=[-1]), dim=-1),
                    dims=[-1],
                )
                branch_weights = torch.cat(
                    [rev_cumsum[..., :1], rev_cumsum], dim=-1
                )
                h_split = self.merger(branches, branch_weights=branch_weights)
                self._log(
                    f"           SPLIT path merged -> shape={tuple(h_split.shape)} "
                    f"(per-sample K via k_one_hot)"
                )
            else:
                self.last_branch_gate_probs.append([])
                h_split = None
                self._log("           [fast-path] all-CONTINUE, skip SPLIT")

            if all_continue:
                w_continue_main = a_one_hot[:, DynamicRouter.ACT_CONTINUE].view(-1, 1, 1)
                h = w_continue_main * h_continue
                mix_tag = "CONTINUE-only"
            elif all_split:
                w_split_main = a_one_hot[:, DynamicRouter.ACT_SPLIT].view(-1, 1, 1)
                h = w_split_main * h_split
                mix_tag = "SPLIT-only"
            else:
                w_continue_main = a_one_hot[:, DynamicRouter.ACT_CONTINUE].view(-1, 1, 1)
                w_split_main    = a_one_hot[:, DynamicRouter.ACT_SPLIT].view(-1, 1, 1)
                h = w_continue_main * h_continue + w_split_main * h_split
                mix_tag = "mixed"
            self._log(
                f"           main mix -> shape={tuple(h.shape)} ({mix_tag})"
            )

        h = self.norm_out(h)
        logits = self.head(h)
        self._log(f"[Output]     shape={tuple(logits.shape)}")
        return logits

    def expert_diversity_loss(self) -> torch.Tensor:
        device = next(self.parameters()).device
        if not self.last_branch_gate_probs:
            return torch.zeros((), device=device)

        total = torch.zeros((), device=device)
        n = 0
        for split in self.last_branch_gate_probs:
            per_branch_dist = []
            for branch_probs in split:
                if len(branch_probs) == 0:
                    continue
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

    vocab_size = 100
    batch_size = 2
    seq_len = 16

    model = ReGSMModel(
        vocab_size=vocab_size,
        d_model=64,
        num_heads=4,
        d_ff=128,
        num_base_layers=2,
        max_recurrent_steps=2,
        max_branch_steps=2,
        max_k=3,
        num_experts=8,
        top_k_experts=2,
        max_seq_len=seq_len,
        dropout=0.0,
        verbose=True,
    )

    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    print(f"Input shape: {tuple(x.shape)}\n")

    print("============ Demo 1: router-driven forward ============")
    model.train()
    out = model(x)
    print(f"\nOutput shape: {tuple(out.shape)}")

    print("\n============ Demo 2: forced SPLIT ============")
    out = model(x, force_main_actions=["SPLIT", "CONTINUE"])
    print(f"\nOutput shape: {tuple(out.shape)}")

    div = model.expert_diversity_loss()
    print(f"\n专家差异度 (avg cosine sim, 越小越分化): {div.item():.4f}")

    print(f"\n退火前 Router tau: {model.router.temperature.item():.4f}")
    for _ in range(5000):
        model.router.anneal_temperature()
    print(f"退火 5000 步后 Router tau: {model.router.temperature.item():.4f}")

    print(f"\n本次 forward 共记录 {len(model.trace)} 条路由决策, 前 5 条:")
    for t in model.trace[:5]:
        print(f"  - {t}")


if __name__ == "__main__":
    _demo()
