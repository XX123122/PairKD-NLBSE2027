"""Shared CodeBERT encoder with task-aware code/comment evidence fusion."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


@dataclass
class PackedBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    code_mask: torch.Tensor
    comment_mask: torch.Tensor
    has_comment: torch.Tensor
    labels: torch.Tensor | None = None
    indices: torch.Tensor | None = None
    comment_input_ids: torch.Tensor | None = None
    comment_attention_mask: torch.Tensor | None = None
    comment_tokens: torch.Tensor | None = None


def pack_views(tokenizer: Any, code: str, comment: str, max_length: int, comment_budget: int = 96) -> dict[str, Any]:
    """Pack code/comment as a RoBERTa pair and return explicit evidence masks."""
    code_ids = tokenizer(code or "", add_special_tokens=False)["input_ids"]
    comment_ids = tokenizer(comment or "", add_special_tokens=False)["input_ids"]
    bos = tokenizer.cls_token_id if tokenizer.cls_token_id is not None else tokenizer.bos_token_id
    sep = tokenizer.sep_token_id if tokenizer.sep_token_id is not None else tokenizer.eos_token_id
    if bos is None or sep is None:
        raise ValueError("Tokenizer must expose CLS/BOS and SEP/EOS IDs")
    has_comment = bool(comment_ids)
    if has_comment:
        budget = max_length - 4  # <s> code </s> </s> comment </s>
        comment_take = min(len(comment_ids), min(int(comment_budget), budget))
        code_take = min(len(code_ids), budget - comment_take)
        remaining = budget - code_take - comment_take
        if remaining:
            extra_comment = min(remaining, len(comment_ids) - comment_take)
            comment_take += extra_comment
            remaining -= extra_comment
        if remaining:
            code_take += min(remaining, len(code_ids) - code_take)
        ids = [bos] + code_ids[:code_take] + [sep, sep] + comment_ids[:comment_take] + [sep]
        code_mask = [0] + [1] * code_take + [0] * (2 + comment_take + 1)
        comment_mask = [0] * (2 + code_take + 1) + [1] * comment_take + [0]
    else:
        code_take = min(len(code_ids), max_length - 2)
        ids = [bos] + code_ids[:code_take] + [sep]
        code_mask = [0] + [1] * code_take + [0]
        comment_mask = [0] * len(ids)
    return {
        "input_ids": ids,
        "attention_mask": [1] * len(ids),
        "code_mask": code_mask,
        "comment_mask": comment_mask,
        "has_comment": has_comment,
        "comment_tokens": len(comment_ids),
    }


def pack_dual_views(tokenizer: Any, code: str, comment: str, code_max_length: int, comment_max_length: int) -> dict[str, Any]:
    """Pack independent code/comment sequences for the shared dual-view mode."""
    bos = tokenizer.cls_token_id if tokenizer.cls_token_id is not None else tokenizer.bos_token_id
    sep = tokenizer.sep_token_id if tokenizer.sep_token_id is not None else tokenizer.eos_token_id
    if bos is None or sep is None:
        raise ValueError("Tokenizer must expose CLS/BOS and SEP/EOS IDs")
    code_ids = tokenizer(code or "", add_special_tokens=False)["input_ids"][: max(0, code_max_length - 2)]
    comment_ids = tokenizer(comment or "", add_special_tokens=False)["input_ids"][: max(0, comment_max_length - 2)]
    has_comment = bool(comment_ids)
    code_input = [bos] + code_ids + [sep]
    comment_input = [bos] + (comment_ids if has_comment else []) + [sep]
    return {
        "code_input_ids": code_input,
        "code_attention_mask": [1] * len(code_input),
        "comment_input_ids": comment_input,
        "comment_attention_mask": [1] * len(comment_input),
        "has_comment": has_comment,
        "comment_tokens": len(comment_ids),
    }


class TaskAwareFusionModel(nn.Module):
    def __init__(self, encoder: nn.Module, hidden_size: int, dropout: float = 0.1, mode: str = "task-gated"):
        super().__init__()
        if mode not in {"task-gated", "fixed", "cls", "weakness-cls-mat-task-gated", "task-attention", "task-attention-aux", "global-evidence-concat", "global-evidence-moe", "shared-dual-cls", "hard-mat-comment", "mat-comment-global"}:
            raise ValueError(f"Unknown fusion mode: {mode}")
        self.encoder = encoder
        self.mode = mode
        self.dropout = nn.Dropout(dropout)
        self.code_gates = nn.ModuleList([nn.Linear(hidden_size, 1) for _ in range(2)])
        self.comment_gates = nn.ModuleList([nn.Linear(hidden_size, 1) for _ in range(2)])
        self.heads = nn.ModuleList([nn.Linear(hidden_size, 1) for _ in range(2)])
        if mode in {"task-attention", "task-attention-aux"}:
            self.code_attention = nn.ModuleList([nn.Linear(hidden_size, 1) for _ in range(2)])
            self.comment_attention = nn.ModuleList([nn.Linear(hidden_size, 1) for _ in range(2)])
            self.residuals = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(2)])
            self.norms = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(2)])
        if mode == "task-attention-aux":
            self.aux_weakness_code = nn.Linear(hidden_size, 1)
            self.aux_mat_comment = nn.Linear(hidden_size, 1)
        if mode == "global-evidence-concat":
            self.global_evidence_projections = nn.ModuleList([nn.Linear(hidden_size * 2, hidden_size) for _ in range(2)])
            self.global_evidence_norms = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(2)])
        if mode == "global-evidence-moe":
            self.global_heads = nn.ModuleList([nn.Linear(hidden_size, 1) for _ in range(2)])
            self.evidence_heads = nn.ModuleList([nn.Linear(hidden_size, 1) for _ in range(2)])
            self.global_evidence_gates = nn.ModuleList([nn.Linear(hidden_size * 2, 1) for _ in range(2)])
        if mode == "mat-comment-global":
            self.mat_comment_global_projection = nn.Linear(hidden_size * 2, hidden_size)
            self.mat_no_comment_projection = nn.Linear(hidden_size, hidden_size)
            self.mat_global_norm = nn.LayerNorm(hidden_size)

    @staticmethod
    def _pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weights = mask.to(hidden.dtype).unsqueeze(-1)
        denom = weights.sum(dim=1).clamp_min(1.0)
        return (hidden * weights).sum(dim=1) / denom

    @staticmethod
    def _attention_pool(hidden: torch.Tensor, mask: torch.Tensor, scorer: nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
        """Return masked-softmax weights and the corresponding weighted pool."""
        valid = mask.to(torch.bool)
        scores = scorer(hidden).squeeze(-1).masked_fill(~valid, -1e4)
        weights = torch.softmax(scores, dim=1) * valid.to(hidden.dtype)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        pooled = (hidden * weights.unsqueeze(-1)).sum(dim=1)
        return pooled, weights

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        code_mask: torch.Tensor,
        comment_mask: torch.Tensor,
        has_comment: torch.Tensor,
        return_details: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        if self.mode == "cls":
            cls = self.dropout(hidden[:, 0])
            logits = torch.stack([head(self.dropout(cls)).squeeze(-1) for head in self.heads], dim=1)
            result = (logits, torch.zeros((hidden.shape[0], 2, 2), device=hidden.device, dtype=hidden.dtype))
            return result
        if self.mode == "weakness-cls-mat-task-gated":
            cls = self.dropout(hidden[:, 0])
            weakness_logit = self.heads[0](self.dropout(cls)).squeeze(-1)
            code = self.dropout(self._pool(hidden, code_mask))
            comment = self.dropout(self._pool(hidden, comment_mask))
            gate_logits = torch.cat([self.code_gates[1](code), self.comment_gates[1](comment)], dim=1)
            gate_logits[:, 1] = gate_logits[:, 1].masked_fill(~has_comment, -1e4)
            mat_gates = torch.softmax(gate_logits, dim=1)
            mat_gates = torch.where(has_comment.unsqueeze(1), mat_gates, torch.cat([torch.ones_like(mat_gates[:, :1]), torch.zeros_like(mat_gates[:, 1:])], dim=1))
            mat_evidence = mat_gates[:, :1] * code + mat_gates[:, 1:] * comment
            mat_logit = self.heads[1](self.dropout(mat_evidence)).squeeze(-1)
            logits = torch.stack([weakness_logit, mat_logit], dim=1)
            no_weakness_evidence_gate = torch.zeros_like(mat_gates)
            gates = torch.stack([no_weakness_evidence_gate, mat_gates], dim=1)
            details = {"weakness_cls": cls, "mat_evidence": mat_evidence, "mat_gate": mat_gates}
            return (logits, gates, details) if return_details else (logits, gates)
        if self.mode in {"task-attention", "task-attention-aux"}:
            task_logits = []
            gate_values = []
            code_attentions = []
            comment_attentions = []
            code_pools = []
            comment_pools = []
            for code_scorer, comment_scorer, code_gate, comment_gate, residual, norm, head in zip(
                self.code_attention, self.comment_attention, self.code_gates, self.comment_gates,
                self.residuals, self.norms, self.heads
            ):
                code, code_weights = self._attention_pool(hidden, code_mask, code_scorer)
                comment, comment_weights = self._attention_pool(hidden, comment_mask, comment_scorer)
                gate_logits = torch.cat([code_gate(code), comment_gate(comment)], dim=1)
                gate_logits[:, 1] = gate_logits[:, 1].masked_fill(~has_comment, -1e4)
                gates = torch.softmax(gate_logits, dim=1)
                gates = torch.where(has_comment.unsqueeze(1), gates, torch.cat([torch.ones_like(gates[:, :1]), torch.zeros_like(gates[:, 1:])], dim=1))
                evidence = gates[:, :1] * code + gates[:, 1:] * comment
                z = norm(self.dropout(hidden[:, 0]) + residual(self.dropout(evidence)))
                task_logits.append(head(self.dropout(z)).squeeze(-1))
                gate_values.append(gates)
                code_attentions.append(code_weights)
                comment_attentions.append(comment_weights)
                code_pools.append(code)
                comment_pools.append(comment)
            logits = torch.stack(task_logits, dim=1)
            gates = torch.stack(gate_values, dim=1)
            details = {
                "code_attention": torch.stack(code_attentions, dim=1),
                "comment_attention": torch.stack(comment_attentions, dim=1),
                "code_pools": torch.stack(code_pools, dim=1),
                "comment_pools": torch.stack(comment_pools, dim=1),
            }
            if self.mode == "task-attention-aux":
                details["aux_logits"] = torch.cat([
                    self.aux_weakness_code(self.dropout(code_pools[0])),
                    self.aux_mat_comment(self.dropout(comment_pools[1])),
                ], dim=1)
            return (logits, gates, details) if return_details else (logits, gates)
        code = self.dropout(self._pool(hidden, code_mask))
        comment = self.dropout(self._pool(hidden, comment_mask))
        if self.mode in {"global-evidence-concat", "global-evidence-moe"}:
            cls = self.dropout(hidden[:, 0])
            task_logits = []
            gate_values = []
            mixture_values = []
            for task_idx, (code_gate, comment_gate) in enumerate(zip(self.code_gates, self.comment_gates)):
                gate_logits = torch.cat([code_gate(code), comment_gate(comment)], dim=1)
                gate_logits[:, 1] = gate_logits[:, 1].masked_fill(~has_comment, -1e4)
                gates = torch.softmax(gate_logits, dim=1)
                gates = torch.where(has_comment.unsqueeze(1), gates, torch.cat([torch.ones_like(gates[:, :1]), torch.zeros_like(gates[:, 1:])], dim=1))
                evidence = gates[:, :1] * code + gates[:, 1:] * comment
                if self.mode == "global-evidence-concat":
                    z = self.global_evidence_norms[task_idx](self.global_evidence_projections[task_idx](torch.cat([cls, evidence], dim=1)))
                    logit = self.heads[task_idx](self.dropout(z)).squeeze(-1)
                else:
                    beta = torch.sigmoid(self.global_evidence_gates[task_idx](torch.cat([cls, evidence], dim=1))).squeeze(-1)
                    global_logit = self.global_heads[task_idx](self.dropout(cls)).squeeze(-1)
                    evidence_logit = self.evidence_heads[task_idx](self.dropout(evidence)).squeeze(-1)
                    logit = beta * global_logit + (1.0 - beta) * evidence_logit
                    mixture_values.append(beta)
                task_logits.append(logit)
                gate_values.append(gates)
            logits = torch.stack(task_logits, dim=1)
            gates = torch.stack(gate_values, dim=1)
            if self.mode == "global-evidence-moe":
                details = {"mixture_beta": torch.stack(mixture_values, dim=1)}
            else:
                details = {}
            return (logits, gates, details) if return_details else (logits, gates)
        if self.mode in {"hard-mat-comment", "mat-comment-global"}:
            cls = self.dropout(hidden[:, 0])
            weakness_gate_logits = torch.cat([self.code_gates[0](code), self.comment_gates[0](comment)], dim=1)
            weakness_gate_logits[:, 1] = weakness_gate_logits[:, 1].masked_fill(~has_comment, -1e4)
            weakness_gates = torch.softmax(weakness_gate_logits, dim=1)
            weakness_gates = torch.where(has_comment.unsqueeze(1), weakness_gates, torch.cat([torch.ones_like(weakness_gates[:, :1]), torch.zeros_like(weakness_gates[:, 1:])], dim=1))
            weakness_evidence = weakness_gates[:, :1] * code + weakness_gates[:, 1:] * comment
            weakness_logit = self.heads[0](self.dropout(weakness_evidence)).squeeze(-1)
            hard_mat_gates = torch.where(has_comment.unsqueeze(1), torch.cat([torch.zeros_like(code[:, :1]), torch.ones_like(code[:, :1])], dim=1), torch.cat([torch.ones_like(code[:, :1]), torch.zeros_like(code[:, :1])], dim=1))
            if self.mode == "hard-mat-comment":
                mat_evidence = hard_mat_gates[:, :1] * code + hard_mat_gates[:, 1:] * comment
            else:
                mat_evidence = self.mat_global_norm(torch.where(has_comment.unsqueeze(1), self.mat_comment_global_projection(torch.cat([comment, cls], dim=1)), self.mat_no_comment_projection(cls)))
            mat_logit = self.heads[1](self.dropout(mat_evidence)).squeeze(-1)
            logits = torch.stack([weakness_logit, mat_logit], dim=1)
            gates = torch.stack([weakness_gates, hard_mat_gates], dim=1)
            details = {"mat_comment_route": hard_mat_gates}
            return (logits, gates, details) if return_details else (logits, gates)
        task_logits = []
        gate_values = []
        for code_gate, comment_gate, head in zip(self.code_gates, self.comment_gates, self.heads):
            if self.mode == "fixed":
                gates = torch.cat([0.5 * torch.ones_like(code[:, :1]), 0.5 * torch.ones_like(code[:, :1])], dim=1)
                gates = torch.where(has_comment.unsqueeze(1), gates, torch.cat([torch.ones_like(gates[:, :1]), torch.zeros_like(gates[:, 1:])], dim=1))
            else:
                gate_logits = torch.cat([code_gate(code), comment_gate(comment)], dim=1)
                gate_logits[:, 1] = gate_logits[:, 1].masked_fill(~has_comment, -1e4)
                gates = torch.softmax(gate_logits, dim=1)
                gates = torch.where(has_comment.unsqueeze(1), gates, torch.cat([torch.ones_like(gates[:, :1]), torch.zeros_like(gates[:, 1:])], dim=1))
            fused = gates[:, :1] * code + gates[:, 1:] * comment
            task_logits.append(head(self.dropout(fused)).squeeze(-1))
            gate_values.append(gates)
        return torch.stack(task_logits, dim=1), torch.stack(gate_values, dim=1)

    def forward_dual(
        self,
        code_input_ids: torch.Tensor,
        code_attention_mask: torch.Tensor,
        comment_input_ids: torch.Tensor,
        comment_attention_mask: torch.Tensor,
        has_comment: torch.Tensor,
        return_details: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Run code and comment through this same encoder instance separately."""
        code_hidden = self.encoder(input_ids=code_input_ids, attention_mask=code_attention_mask).last_hidden_state
        comment_hidden = self.encoder(input_ids=comment_input_ids, attention_mask=comment_attention_mask).last_hidden_state
        code = self.dropout(code_hidden[:, 0])
        comment = self.dropout(comment_hidden[:, 0])
        task_logits = []
        gate_values = []
        for code_gate, comment_gate, head in zip(self.code_gates, self.comment_gates, self.heads):
            gate_logits = torch.cat([code_gate(code), comment_gate(comment)], dim=1)
            gate_logits[:, 1] = gate_logits[:, 1].masked_fill(~has_comment, -1e4)
            gates = torch.softmax(gate_logits, dim=1)
            gates = torch.where(has_comment.unsqueeze(1), gates, torch.cat([torch.ones_like(gates[:, :1]), torch.zeros_like(gates[:, 1:])], dim=1))
            fused = gates[:, :1] * code + gates[:, 1:] * comment
            task_logits.append(head(self.dropout(fused)).squeeze(-1))
            gate_values.append(gates)
        logits = torch.stack(task_logits, dim=1)
        gates = torch.stack(gate_values, dim=1)
        details = {"code_cls": code, "comment_cls": comment}
        return (logits, gates, details) if return_details else (logits, gates)


def collate_packed(features: list[dict[str, Any]], pad_id: int) -> PackedBatch:
    width = max(len(x["input_ids"]) for x in features)
    def pad(key: str, value: int, dtype: torch.dtype = torch.long) -> torch.Tensor:
        return torch.tensor([x[key] + [value] * (width - len(x[key])) for x in features], dtype=dtype)
    labels = [x.get("labels") for x in features]
    indices = [x.get("index") for x in features]
    return PackedBatch(
        input_ids=pad("input_ids", pad_id),
        attention_mask=pad("attention_mask", 0),
        code_mask=pad("code_mask", 0),
        comment_mask=pad("comment_mask", 0),
        has_comment=torch.tensor([x["has_comment"] for x in features], dtype=torch.bool),
        comment_tokens=torch.tensor([x.get("comment_tokens", 0) for x in features], dtype=torch.long),
        labels=torch.tensor(labels, dtype=torch.float32) if labels[0] is not None else None,
        indices=torch.tensor(indices, dtype=torch.long) if indices[0] is not None else None,
    )


def collate_dual(features: list[dict[str, Any]]) -> PackedBatch:
    code_width = max(len(x["code_input_ids"]) for x in features)
    comment_width = max(len(x["comment_input_ids"]) for x in features)
    def pad(key: str, width: int) -> torch.Tensor:
        return torch.tensor([x[key] + [0] * (width - len(x[key])) for x in features], dtype=torch.long)
    labels = [x.get("labels") for x in features]
    indices = [x.get("index") for x in features]
    code_ids = pad("code_input_ids", code_width)
    return PackedBatch(
        input_ids=code_ids,
        attention_mask=pad("code_attention_mask", code_width),
        code_mask=torch.zeros_like(code_ids),
        comment_mask=torch.zeros_like(code_ids),
        has_comment=torch.tensor([x["has_comment"] for x in features], dtype=torch.bool),
        labels=torch.tensor(labels, dtype=torch.float32) if labels[0] is not None else None,
        indices=torch.tensor(indices, dtype=torch.long) if indices[0] is not None else None,
        comment_input_ids=pad("comment_input_ids", comment_width),
        comment_attention_mask=pad("comment_attention_mask", comment_width),
        comment_tokens=torch.tensor([x.get("comment_tokens", 0) for x in features], dtype=torch.long),
    )
