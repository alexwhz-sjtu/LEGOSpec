from unittest.mock import patch

import torch
from torch import nn

from litespec.core.dflash2 import OnlineDFlash2Model
from litespec.modeling.draft.dflash2 import DFlash2DraftModel
from test_dflash_family import tiny_config


def test_public_accuracy_uses_selector_top1_and_counts_uncovered_gold():
    torch.manual_seed(31)
    model = DFlash2DraftModel(tiny_config('DFlash2DraftModel', selector_rank=4,
                                         selector_top_k=3, conv_kernel_size=2,
                                         conv_group_size=4))
    head = nn.Linear(16, 32, bias=False)
    wrapper = OnlineDFlash2Model(model, head, nn.Embedding(32, 16),
                                 mask_token_id=31, block_size=4, num_anchors=1,
                                 attention_backend='eager', objective_chunk_blocks=0)
    hidden = torch.randn(1, 4, 16, requires_grad=True)
    base = model.transform_unary_logits(head(hidden)).reshape(1, 1, 4, 32)
    candidates = base.topk(3, -1).indices
    ids = torch.zeros(1, 8, dtype=torch.long)
    ids[0, 1:3] = candidates[0, 0, 1:3, -1]
    ids[0, 3] = next(i for i in range(32) if i not in candidates[0, 0, 3])

    def final_scores(**kwargs):
        # Deliberately select the last candidate, unlike unary argmax.
        logits = kwargs['unary_logits']
        return torch.arange(logits.size(-1), device=logits.device).expand_as(logits).float()

    blocks = (torch.zeros(1, 1, dtype=torch.long), torch.ones(1, 1, dtype=torch.bool), hidden)
    with patch.object(wrapper, '_forward_draft_blocks', return_value=blocks), \
         patch.object(model.candidate_selector, 'score_candidates', side_effect=final_scores):
        output = wrapper(ids, torch.randn(1, 8, 16), torch.ones(1, 8),
                         target_last_hidden_states=torch.randn(1, 8, 16),
                         collect_detailed_metrics=True)
    terms = output.ratio_metrics['hard_label/token_accuracy']
    assert terms.numerator.item() == 2
    assert terms.denominator.item() == 3
    assert ((base.argmax(-1)[..., 1:] == ids[:, None, 1:4]).sum()).item() == 0
