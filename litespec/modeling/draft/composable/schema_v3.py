"""Strict public recipe v3 and its private, expanded execution view.

This module is inference-only and is included in standalone exports.
"""
from copy import deepcopy
import math

from .loss_config import TOKEN_OBJECTIVES, normalize_training_recipe

ATTENTION_TYPES = ("mha", "gqa", "mla", "gdn")
ACCESS_TYPES = ("dense", "sliding_window", "dsa", "linear")
PROPOSAL_TYPES = ("parallel", "markov", "rnn", "sequential_rnn", "selector")
MLA_DEFAULTS = dict(num_attention_heads=16, num_key_value_heads=16, q_lora_rank=None,
                    kv_lora_rank=512, qk_nope_head_dim=128, qk_rope_head_dim=64,
                    v_head_dim=128, rope_interleave=True)
GDN_DEFAULTS = dict(linear_num_key_heads=16, linear_num_value_heads=32,
                    linear_key_head_dim=128, linear_value_head_dim=128,
                    query_conv_kernel_size=2, context_conv_kernel_size=4,
                    query_conv_mode="block")
DSA_DEFAULTS = dict(num_query_heads=4, head_dim=128, rope_head_dim=64,
                    index_topk=1024, anchor_chunk_size=16, stage="sparse",
                    execution="gather", enabled=True, loss_weight=1.0)
EXECUTION_DEFAULTS = dict(attention_backend="sdpa", flex_attention_backend="TRITON", gdn_backend="auto")

# The field manual and coverage tests use these exact sets.
FIELDS = {
    "recipe": {"context", "backbone", "proposal", "objective", "mask_token_id", "attention_configs", "access_configs"},
    "context": {"type", "target_layer_ids"},
    "backbone": {"type", "layers"},
    "layer": {"attention", "access", "local_operator"},
    "local_operator": {"type", "kernel_size", "group_size"},
    "mha": {"num_attention_heads", "num_key_value_heads", "head_dim"},
    "gqa": {"num_attention_heads", "num_key_value_heads", "head_dim"},
    "mla": set(MLA_DEFAULTS), "gdn": set(GDN_DEFAULTS),
    "dense": set(), "linear": set(), "sliding_window": {"window_size"},
    "dsa": set(DSA_DEFAULTS) | {"rope_theta"},
    "execution": set(EXECUTION_DEFAULTS),
    "target_attention": {"num_attention_heads", "num_key_value_heads", "head_dim"},
    "parallel": {"type"}, "markov": {"type", "rank"}, "rnn": {"type", "rank"},
    "sequential_rnn": {"type", "rank"},
    "selector": {"type", "rank", "top_k", "selector_mode", "stop_gradient"},
    "component": {"type", "weight"}, "branch": {"weight"},
    "objective": {"type", "loss_decay_gamma"},
    "weighted_sum": {"type", "loss_decay_gamma", "components"},
    "dpace": {"type", "loss_decay_gamma", "alpha", "reduction"},
    "tv_survival": {"type", "loss_decay_gamma", "survival_weight"},
    "opd_survival": {"type", "loss_decay_gamma", "beta_rej", "rejection_decay", "survival_weight"},
    "selector_losses": {"type", "loss_decay_gamma", "base", "final"},
}


def fields(value, allowed, path):
    if not isinstance(value, dict):
        raise ValueError(path + " must be an object")
    unknown = set(value) - set(allowed)
    if unknown:
        raise ValueError(path + ": unknown field(s): " + ", ".join(sorted(unknown)))


def integer(value, path, minimum=1):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(path + " must be an integer >= " + str(minimum))


def number(value, path, minimum=0, strict=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or (value <= minimum if strict else value < minimum):
        raise ValueError(path + " must be a finite number " + ("> " if strict else ">= ") + str(minimum))


def boolean(value, path):
    if not isinstance(value, bool):
        raise ValueError(path + " must be boolean")


def execution_options(value):
    fields(value, FIELDS["execution"], "execution_config")
    result = {**EXECUTION_DEFAULTS, **value}
    for key, options in {
        "attention_backend": ("eager", "sdpa", "flex_attention"),
        "flex_attention_backend": ("TRITON", "FLASH"),
        "gdn_backend": ("auto", "fla", "reference"),
    }.items():
        if result[key] not in options:
            raise ValueError("execution_config." + key + " must be one of " + str(options))
    return result


def check_objective(value, path="objective", branch=False, component=False):
    kind = value.get("type") if isinstance(value, dict) else None
    if not isinstance(kind, str) or kind not in set(TOKEN_OBJECTIVES) | {"dpace", "bv", "tv_survival", "opd_survival", "weighted_sum", "selector_losses"}:
        raise ValueError(path + ": unsupported objective")
    allowed = set(FIELDS.get(kind, FIELDS["objective"]))
    if branch or component:
        allowed.discard("loss_decay_gamma")
        allowed.update(FIELDS["branch"])
    if component:
        allowed = FIELDS["component"]
    fields(value, allowed, path)
    if "weight" in value:
        number(value["weight"], path + ".weight")
    if kind == "selector_losses":
        if branch or component:
            raise ValueError("selector_losses is an outer objective only")
        for side in ("base", "final"):
            if side in value:
                check_objective(value[side], path + "." + side, branch=True)
    if kind == "weighted_sum":
        if not isinstance(value.get("components"), list) or not value["components"]:
            raise ValueError(path + ".components must be a nonempty array")
        for index, item in enumerate(value["components"]):
            if not isinstance(item, dict) or item.get("type") not in TOKEN_OBJECTIVES:
                raise ValueError("weighted_sum supports token objectives only")
            check_objective(item, path + ".components[" + str(index) + "]", component=True)
            if "weight" not in item:
                raise ValueError("weighted_sum component requires weight")
    if kind == "dpace":
        alpha = value.get("alpha", .5)
        number(alpha, path + ".alpha")
        if alpha > 1 or value.get("reduction", "complete_block_mean") != "complete_block_mean":
            raise ValueError("dpace requires alpha in [0,1] and reduction=complete_block_mean")
    if kind in ("tv_survival", "opd_survival"):
        number(value.get("survival_weight", 1.), path + ".survival_weight")
    if kind == "opd_survival":
        number(value.get("beta_rej", 1.), path + ".beta_rej")
        decay = value.get("rejection_decay", .8)
        number(decay, path + ".rejection_decay", strict=True)
        if decay > 1:
            raise ValueError(path + ".rejection_decay must be in (0, 1]")


def normalize_v3(config):
    r = deepcopy(config.composable_config)
    fields(r, FIELDS["recipe"], "composable_config")
    for required in ("context", "backbone", "proposal", "objective", "attention_configs"):
        if required not in r:
            raise ValueError("composable_config requires " + required)
    integer(config.block_size, "block_size", 2)
    integer(config.num_hidden_layers, "num_hidden_layers")
    integer(config.num_target_layers, "num_target_layers")
    fields(r["context"], FIELDS["context"], "context")
    if r["context"].get("type") not in ("dflash", "dflare"):
        raise ValueError("context.type must be dflash or dflare")
    ids = r["context"].get("target_layer_ids")
    if ids is not None:
        if not isinstance(ids, list) or not ids:
            raise ValueError("target_layer_ids must be a nonempty array or null")
        for value in ids:
            integer(value, "target_layer_ids[]", 0)
        if ids != sorted(set(ids)) or ids[-1] >= config.num_target_layers:
            raise ValueError("target_layer_ids must be sorted, unique and within the target")
    if "mask_token_id" in r:
        integer(r["mask_token_id"], "mask_token_id", 0)
        if r["mask_token_id"] >= config.vocab_size:
            raise ValueError("mask_token_id is outside vocab_size")
    b = r["backbone"]
    fields(b, FIELDS["backbone"], "backbone")
    if b.get("type") != "layer_stack" or not isinstance(b.get("layers"), list) or not b["layers"]:
        raise ValueError("v3 requires backbone.type=layer_stack and nonempty layers")
    if len(b["layers"]) != config.num_hidden_layers:
        raise ValueError("layers length must equal num_hidden_layers")
    used, accesses = set(), set()
    for layer in b["layers"]:
        fields(layer, FIELDS["layer"], "layers[]")
        kind, access = layer.get("attention"), layer.get("access")
        if kind not in ATTENTION_TYPES or access not in ACCESS_TYPES:
            raise ValueError("each layer requires an attention type and access")
        if (kind == "gdn") != (access == "linear") or (kind == "mla" and access == "dsa"):
            raise ValueError("invalid attention/access combination: " + kind + "/" + access)
        used.add(kind); accesses.add(access)
        op = layer.get("local_operator", {"type": "none"})
        fields(op, FIELDS["local_operator"], "local_operator")
        if op.get("type") not in ("none", "dflash2_conv"):
            raise ValueError("unsupported local_operator.type")
        if op["type"] == "none" and set(op) != {"type"}:
            raise ValueError("local_operator=none has no parameters")
        if op["type"] == "dflash2_conv":
            if kind == "gdn":
                raise ValueError("GDN does not support dflash2_conv local_operator")
            integer(op.setdefault("kernel_size", 2), "local_operator.kernel_size")
            integer(op.setdefault("group_size", 16), "local_operator.group_size")
            if op["kernel_size"] > config.block_size or config.hidden_size % op["group_size"]:
                raise ValueError("local convolution requires kernel_size <= block_size and group_size dividing hidden_size")
        if "local_operator" in layer:
            layer["local_operator"] = op
    profiles = r["attention_configs"]
    fields(profiles, ATTENTION_TYPES, "attention_configs")
    if set(profiles) != used:
        raise ValueError("attention_configs must contain exactly the attention types used by layers")
    target = config.target_attention
    fields(target, FIELDS["target_attention"], "target_attention")
    for key in FIELDS["target_attention"]:
        integer(target.get(key), "target_attention." + key)
    if target["num_attention_heads"] % target["num_key_value_heads"]:
        raise ValueError("target Q heads must be divisible by target KV heads")
    for kind, profile in profiles.items():
        fields(profile, FIELDS[kind], "attention_configs." + kind)
        if kind in ("mha", "gqa"):
            q = target["num_attention_heads"]
            defaults = dict(num_attention_heads=q, num_key_value_heads=q if kind == "mha" else target["num_key_value_heads"], head_dim=target["head_dim"])
        else:
            defaults = MLA_DEFAULTS if kind == "mla" else GDN_DEFAULTS
        profile = profiles[kind] = {**defaults, **profile}
        for key, value in profile.items():
            if key == "q_lora_rank" and value is None:
                continue
            if key == "rope_interleave":
                boolean(value, kind + "." + key)
            elif key == "query_conv_mode":
                if value not in ("block", "confirmed_history"):
                    raise ValueError("query_conv_mode must be block or confirmed_history")
            else:
                integer(value, kind + "." + key)
        if kind in ("mha", "gqa"):
            if profile["num_attention_heads"] != target["num_attention_heads"]:
                raise ValueError("MHA/GQA Q heads must equal target Q heads")
            if profile["num_attention_heads"] % profile["num_key_value_heads"] or profile["head_dim"] % 2:
                raise ValueError("MHA/GQA requires KV dividing Q and even head_dim")
            if kind == "mha" and profile["num_attention_heads"] != profile["num_key_value_heads"]:
                raise ValueError("MHA requires equal Q/K/V head counts; use gqa for grouped KV")
        if kind == "mla" and (profile["num_attention_heads"] != profile["num_key_value_heads"] or profile["qk_rope_head_dim"] % 2):
            raise ValueError("expanded MLA requires equal head counts and even RoPE dimension")
        if kind == "gdn":
            if profile["linear_num_value_heads"] % profile["linear_num_key_heads"]:
                raise ValueError("GDN V heads must be a multiple of Q/K heads")
            if profile["query_conv_mode"] == "confirmed_history" and profile["query_conv_kernel_size"] != 4:
                raise ValueError("confirmed_history requires Q conv4")
    ac = r.setdefault("access_configs", {})
    fields(ac, ACCESS_TYPES, "access_configs")
    if set(ac) - accesses:
        raise ValueError("access_configs contains an unused access type")
    for access in sorted(accesses):
        profile = ac.setdefault(access, {})
        fields(profile, FIELDS[access], "access_configs." + access)
        if access == "sliding_window":
            integer(profile.setdefault("window_size", 512), "window_size")
        if access == "dsa":
            profile = ac[access] = {**DSA_DEFAULTS, **profile}
            for key in ("num_query_heads", "head_dim", "index_topk", "anchor_chunk_size"):
                integer(profile[key], "dsa." + key)
            integer(profile["rope_head_dim"], "dsa.rope_head_dim", 0)
            if profile["rope_head_dim"] % 2 or profile["rope_head_dim"] > profile["head_dim"]:
                raise ValueError("DSA rope_head_dim must be even and <= head_dim")
            if "rope_theta" in profile:
                number(profile["rope_theta"], "dsa.rope_theta", strict=True)
            number(profile["loss_weight"], "dsa.loss_weight")
            boolean(profile["enabled"], "dsa.enabled")
            if profile["stage"] not in ("warmup", "sparse") or profile["execution"] not in ("gather", "reference"):
                raise ValueError("invalid DSA stage/execution")
    p = r["proposal"]
    fields(p, set().union(*(FIELDS[name] for name in PROPOSAL_TYPES)), "proposal")
    if p.get("type") not in PROPOSAL_TYPES:
        raise ValueError("unsupported proposal.type")
    fields(p, FIELDS[p["type"]], "proposal")
    if p["type"] != "parallel":
        integer(p.setdefault("rank", 256), "proposal.rank")
    if p["type"] == "selector":
        integer(p.setdefault("top_k", 16), "selector.top_k")
        if p["top_k"] > config.vocab_size:
            raise ValueError("selector.top_k exceeds vocab_size")
        boolean(p.setdefault("stop_gradient", False), "selector.stop_gradient")
        if p.setdefault("selector_mode", "full_vocab_residual") != "full_vocab_residual":
            raise ValueError("v3 Selector requires full_vocab_residual; candidate_only is legacy inference-only")
    if "dsa" in accesses:
        if any(layer["access"] != "dsa" for layer in b["layers"]) or p["type"] == "selector":
            raise ValueError("DSA requires all layers using dsa and no Selector")
        if config.execution_config["attention_backend"] != "sdpa":
            raise ValueError("DSA currently supports SDPA only")
    if config.execution_config["attention_backend"] == "flex_attention":
        if config.attention_dropout:
            raise ValueError("FlexAttention requires attention_dropout=0")
        for kind, profile in profiles.items():
            if kind != "gdn":
                qk = profile["qk_nope_head_dim"] + profile["qk_rope_head_dim"] if kind == "mla" else profile["head_dim"]
                vd = profile["v_head_dim"] if kind == "mla" else profile["head_dim"]
                if min(qk, vd) < 16:
                    raise ValueError("FlexAttention requires QK/V dimensions >=16")
    check_objective(r["objective"])
    if p["type"] != "selector" and r["objective"].get("type") == "selector_losses":
        raise ValueError("selector_losses requires proposal=selector")
    r = normalize_training_recipe(r)
    return r


def runtime_recipe(config):
    """Expand only in memory: serialized v3 stays compact and shared."""
    if config.schema_version != "composable_v3":
        return deepcopy(config.composable_config)
    r = deepcopy(config.composable_config)
    profiles, accesses = r.pop("attention_configs"), r.pop("access_configs")
    layers = []
    for layer in r["backbone"]["layers"]:
        kind, access = layer["attention"], layer["access"]
        if kind == "gdn":
            layers.append({"attention": {"representation": "gdn", "access": "linear"},
                           "gdn_config": {**profiles[kind], "backend": config.execution_config["gdn_backend"]}})
        else:
            attention = {"representation": kind, "access": access, **profiles[kind],
                         "backend": config.execution_config["attention_backend"]}
            if attention["backend"] == "flex_attention":
                attention["flex_backend"] = config.execution_config["flex_attention_backend"]
            if access == "sliding_window":
                attention.update(accesses[access])
            layers.append({"attention": attention, **({"local_operator": layer["local_operator"]} if "local_operator" in layer else {})})
    r["backbone"] = {"type": "layer_stack", "layers": layers}
    if "dsa" in accesses:
        r["indexer"] = accesses["dsa"]
    return r
