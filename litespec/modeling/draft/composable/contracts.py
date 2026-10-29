"""Explicit axes: N samples, A anchors, Q=1+B query slots, S context."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .backbone.gated_delta_state import GDNBackboneState

import torch


@dataclass
class LayerState:
    key_states: torch.Tensor
    value_states: torch.Tensor
    positions: torch.Tensor
    valid: torch.Tensor
    document_ids: torch.Tensor


@dataclass
class DraftState:
    layers: dict[int, LayerState] = field(default_factory=dict)
    backbone_state: Optional["GDNBackboneState"] = None
    indexer_state: Optional["IndexerKeyState"] = None

    def rollback(self, next_position: int | torch.Tensor):
        if self.backbone_state is not None:
            self.backbone_state.validate_rollback(next_position)
        if self.indexer_state is not None:
            state = self.indexer_state
            limits = torch.as_tensor(next_position, device=state.positions.device)
            if limits.ndim == 1:
                limits = limits[:, None]
            valid = state.valid & (state.positions < limits)
            columns = valid.any(0)
            self.indexer_state = IndexerKeyState(
                state.keys[:, columns], state.positions[:, columns],
                state.document_ids[:, columns], valid[:, columns], state.fingerprint,
            )
        for index, draft_state in list(self.layers.items()):
            limits = torch.as_tensor(next_position, device=draft_state.positions.device)
            if limits.ndim == 1:
                limits = limits[:, None]
            keep = draft_state.positions < limits
            # Rectangular cache, including inactive padding, supports unequal batch lengths.
            valid = draft_state.valid & keep
            columns = valid.any(dim=0)
            self.layers[index] = LayerState(
                draft_state.key_states[:, :, columns],
                draft_state.value_states[:, :, columns],
                draft_state.positions[:, columns],
                valid[:, columns],
                draft_state.document_ids[:, columns],
            )
        return self


@dataclass
class ModelInputs:
    """Raw selected target features [N,S,T*D] or [N,S,T,D], embeddings [N,A*Q,D].

    anchor_positions and context_position_ids are absolute; persistent state
    accepts only newly confirmed context. previous_token_ids has shape [N,A,B].
    """

    target_hidden_states: torch.Tensor
    draft_embeddings: torch.Tensor
    anchor_positions: torch.Tensor
    block_keep_mask: torch.Tensor
    context_position_ids: Optional[torch.Tensor] = None
    context_valid_mask: Optional[torch.Tensor] = None
    document_ids: Optional[torch.Tensor] = None
    anchor_document_ids: Optional[torch.Tensor] = None
    previous_token_ids: Optional[torch.Tensor] = None
    anchor_token_ids: Optional[torch.Tensor] = None
    draft_state: Optional[DraftState] = None
    mode: str = "teacher_forcing"
    temperature: float = 0.0
    prediction_mask: Optional[torch.Tensor] = None
    collect_indexer_teacher: bool = False
    selection: Optional["SelectionView"] = None


@dataclass(frozen=True)
class IndexerKeyState:
    keys: torch.Tensor
    positions: torch.Tensor
    document_ids: torch.Tensor
    valid: torch.Tensor
    fingerprint: tuple


@dataclass(frozen=True)
class SelectionView:
    indices: torch.Tensor
    selected_positions: torch.Tensor
    selected_valid: torch.Tensor
    scores: torch.Tensor
    context_positions: torch.Tensor
    context_documents: torch.Tensor
    context_valid: torch.Tensor
    anchor_valid: torch.Tensor
    backend: str = "gather"
    chunk_size: int = 16


@dataclass(frozen=True)
class IndexerAux:
    numerator: torch.Tensor
    denominator: torch.Tensor


@dataclass
class ContextOutput:
    context_hidden_states: list[torch.Tensor]
    inputs: ModelInputs


@dataclass
class BackboneOutput:
    draft_hidden_states: torch.Tensor
    draft_state: Optional[DraftState] = None
    indexer_aux: Optional[IndexerAux] = None


@dataclass
class DraftOutput:
    draft_hidden_states: torch.Tensor
    draft_logits: Optional[torch.Tensor] = None
    base_logits: Optional[torch.Tensor] = None
    draft_token_ids: Optional[torch.Tensor] = None
    draft_probabilities: Optional[torch.Tensor] = None
    candidate_ids: Optional[torch.Tensor] = None
    candidate_logits: Optional[torch.Tensor] = None
    backbone_hidden_states: Optional[torch.Tensor] = None
    draft_state: Optional[DraftState] = None
    indexer_aux: Optional[IndexerAux] = None


@dataclass
class SharedOps:
    embed_tokens: torch.nn.Module
    lm_head: torch.nn.Module
