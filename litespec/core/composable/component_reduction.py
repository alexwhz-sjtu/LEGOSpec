"""Independent global means for compound composable block objectives."""

import torch
import torch.distributed as dist

from litespec.core.outputs import RatioTerms


def global_component_counts(local, sample_count, process_group=None):
    """One detached collective, outside activation-checkpointed operations."""
    order = [(branch, key) for branch in sorted(local) for key in sorted(local[branch])]
    reference = next(iter(next(iter(local.values())).values()))
    counts = torch.stack([local[b][k].detach().double() for b, k in order]
                         + [reference.new_tensor(float(sample_count), dtype=torch.float64)])
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(counts, group=process_group)
    return ({branch: {key: counts[i] for i, (b, key) in enumerate(order) if b == branch}
             for branch in sorted(local)}, counts[-1])


def combine_component_terms(terms, objectives, branch_weights, sample_count,
                            process_group=None, global_denominators=None,
                            reference_samples=None):
    """Encode a sum of global means for the existing FSDP window normalizer."""
    local = {b: {k: t.denominator for k, t in items.items()} for b, items in terms.items()}
    if global_denominators is None:
        if reference_samples is not None:
            raise ValueError("Component reference requires global denominators")
        global_denominators, reference_samples = global_component_counts(
            local, sample_count, process_group)
    elif reference_samples is None:
        raise ValueError("Component denominators require the window sample reference")
    first = next(iter(next(iter(terms.values())).values())).numerator
    reference = torch.as_tensor(reference_samples, device=first.device, dtype=first.dtype).detach()
    if not torch.isfinite(reference) or reference <= 0:
        raise ValueError("Component window must contain samples")
    total = first * 0.
    metrics = {}
    for branch, items in terms.items():
        objective = objectives[branch]
        weights = getattr(objective, "component_weights", {"token": 1.})
        branch_total = first * 0.
        for key, term in items.items():
            den = torch.as_tensor(global_denominators[branch][key], device=first.device,
                                  dtype=first.dtype).detach()
            if not torch.isfinite(den) or den < 0:
                raise ValueError("Component denominators must be finite and nonnegative")
            if (key == "survival" or key == "token") and den <= 0:
                raise ValueError("Component window has no complete supervised blocks")
            value = torch.where(den > 0, term.numerator / den.clamp_min(1e-12),
                                term.numerator * 0.)
            branch_total = branch_total + weights[key] * value
            if key != "token":
                metrics[branch + "/opd_survival/" + key] = term.detached()
        total = total + branch_weights[branch] * branch_total
        # Ratio logging sums across ranks/microbatches before dividing.
        metrics["objective/" + branch] = RatioTerms(
            (reference * branch_total).detach(), first.new_tensor(float(sample_count)))
    return reference * total, first.new_tensor(float(sample_count)), metrics
