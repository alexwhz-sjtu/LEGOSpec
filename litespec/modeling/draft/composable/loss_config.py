"""Pure configuration policy for composable training loss contract v2.

Inference may still deserialize historical recipes. Training must explicitly
normalize and validate them here before constructing objectives or optimizers.
"""

from copy import deepcopy
import math

LOSS_CONTRACT_VERSION = 2
DEFAULT_LOSS_DECAY_GAMMA = 4.0
TOKEN_OBJECTIVES = frozenset(("hard_ce", "soft_ce", "forward_kl", "tv", "l1", "nla", "lk_alpha"))
BLOCK_OBJECTIVES = frozenset(("dpace", "bv", "tv_survival", "opd_survival"))
OBJECTIVES = TOKEN_OBJECTIVES | BLOCK_OBJECTIVES | {"weighted_sum"}


def resolve_gamma(recipe):
    gamma = recipe.get("loss_decay_gamma", DEFAULT_LOSS_DECAY_GAMMA)
    if gamma is None:
        return None
    if isinstance(gamma, bool) or not isinstance(gamma, (int, float)) or not math.isfinite(gamma) or gamma <= 0:
        raise ValueError("objective.loss_decay_gamma must be a finite positive number or null")
    return float(gamma)


def loss_weight(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError("Loss weight must be finite and nonnegative")
    return float(value)


def normalize_objective(recipe):
    if not isinstance(recipe, dict) or recipe.get("type") not in OBJECTIVES:
        raise ValueError("Unsupported composable objective: " + str(recipe))
    result = deepcopy(recipe)
    result["loss_decay_gamma"] = resolve_gamma(result)
    if result["type"] == "opd_survival":
        # These are independent block objectives; no fixed token decay is used.
        result["beta_rej"] = loss_weight(result.get("beta_rej", 1.0))
        result["survival_weight"] = loss_weight(result.get("survival_weight", 1.0))
        decay = result.get("rejection_decay", 0.8)
        if (isinstance(decay, bool) or not isinstance(decay, (int, float))
                or not math.isfinite(decay) or not 0 < decay <= 1):
            raise ValueError("objective.rejection_decay must be finite and in (0, 1]")
        result["rejection_decay"] = float(decay)
    if result["type"] == "dpace":
        reduction = result.get("reduction", "complete_block_mean")
        if reduction != "complete_block_mean":
            raise ValueError("D-PACE requires reduction=complete_block_mean; use init_from to migrate historical weights")
        result["reduction"] = reduction
    if result["type"] == "weighted_sum":
        components = result.get("components")
        if not isinstance(components, list) or not components:
            raise ValueError("weighted_sum requires non-empty components")
        for component in components:
            if not isinstance(component, dict) or component.get("type") not in TOKEN_OBJECTIVES:
                raise ValueError("weighted_sum requires token-normalized components")
            if "loss_decay_gamma" in component:
                raise ValueError("Set loss_decay_gamma only on the outer objective, not components")
            component["weight"] = loss_weight(component.get("weight"))
        if not any(c["weight"] for c in components):
            raise ValueError("At least one component weight must be positive")
    return result


def normalize_training_recipe(recipe):
    result = deepcopy(recipe)
    proposal = result["proposal"]
    if "loss_decay_gamma" in proposal:
        raise ValueError("proposal.loss_decay_gamma was removed; migrate to objective.loss_decay_gamma")
    objective = result["objective"]
    if proposal["type"] != "selector":
        result["objective"] = normalize_objective(objective)
        return result
    if proposal.get("selector_mode", "full_vocab_residual") != "full_vocab_residual":
        raise ValueError("candidate_only is inference-only; migrate to full_vocab_residual and selector_losses for training")
    proposal["selector_mode"] = "full_vocab_residual"
    if objective.get("type") != "selector_losses":
        raise ValueError("Selector training requires objective.type=selector_losses with base and final objectives")
    gamma = resolve_gamma(objective)
    branches = {"type": "selector_losses", "loss_decay_gamma": gamma}
    for name, default_weight in (("base", 0.1), ("final", 1.0)):
        branch = deepcopy(objective.get(name, {"type": "hard_ce"}))
        if "loss_decay_gamma" in branch:
            raise ValueError("Set loss_decay_gamma only on objective, not its " + name + " branch")
        weight = loss_weight(branch.pop("weight", default_weight))
        branch = normalize_objective({**branch, "loss_decay_gamma": gamma})
        branch.pop("loss_decay_gamma")
        branches[name] = {**branch, "weight": weight}
    if not any(branches[name]["weight"] for name in ("base", "final")):
        raise ValueError("At least one Selector branch weight must be positive")
    result["objective"] = branches
    return result
