"""LEGOSpec full-vocabulary selector row construction."""

import torch


@torch.compile(dynamic=True)
def selector_full_vocab_row(base, candidates, previous_candidates, previous, delta):
    matches = previous_candidates == previous[:, None]
    covered = matches.any(-1)
    rows = matches.to(torch.int32).argmax(-1)
    chosen = delta.gather(
        1, rows[:, None, None].expand(-1, 1, candidates.shape[-1])
    ).squeeze(1)
    chosen = torch.where(covered[:, None], chosen, 0.0)
    return base.float().scatter_add(-1, candidates, chosen.float())
