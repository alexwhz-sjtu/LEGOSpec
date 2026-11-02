"""EAGLE3 integration: upstream equations, supervision, objectives and decoding."""

from copy import deepcopy
import importlib.util
import math
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from transformers import AutoModel, DynamicCache, Qwen3Config, Qwen3ForCausalLM

from litespec.core.composable.anchor_sampling import prepare_sample_epoch_layout
from litespec.core.composable.batch import position_weights, prepare_layout
from litespec.core.composable.objectives import build_objective
from litespec.core.composable.training_model import ComposableTrainingModel
from litespec.modeling.draft.composable import ComposableDraftConfig, ComposableDraftModel, SharedOps
from litespec.modeling.draft.composable.backbone.eagle3 import Eagle3State, training_mask
from litespec.modeling.draft.composable.eagle3_inference import (
    Eagle3Adapter, TargetFeatures, generate_eagle3, probabilities, request_seed, verify,
)
from litespec.modeling.draft.composable.export import export_composable


ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / "tests/eagle3_integration/reference"


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(123)
    torch.set_num_threads(1)


def config(objective=None, backend="sdpa", block_size=4):
    recipe = {
        "context": {"type": "eagle3", "target_layer_ids": [0, 1, 2, 3, 4]},
        "backbone": {"type": "eagle3", "layers": [{"attention": "gqa", "access": "dense"}]},
        "attention_configs": {"gqa": {}},
    }
    if objective is not None:
        recipe["objective"] = objective
    return ComposableDraftConfig(composable_config=recipe, hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_target_layers=6, num_attention_heads=2, num_key_value_heads=1,
        head_dim=8, vocab_size=19, block_size=block_size,
        execution_config={"attention_backend": backend}, attention_dropout=0.)


def components(c):
    shared = SharedOps(nn.Embedding(c.vocab_size, c.hidden_size), nn.Linear(c.hidden_size, c.vocab_size, bias=False))
    shared.embed_tokens.requires_grad_(False)
    shared.lm_head.requires_grad_(False)
    return shared


def batch(c, count=2, length=10):
    mask = torch.ones(count, length)
    mask[:, :2] = 0
    return dict(input_ids=torch.randint(c.vocab_size, (count, length)), loss_mask=mask,
        context_valid_mask=torch.ones(count, length, dtype=torch.bool),
        target_hidden_states=torch.randn(count, length, 5 * c.hidden_size),
        target_last_hidden_states=torch.randn(count, length, c.hidden_size))


def wrapper(model, shared, mode="all", chunk=2, anchors=512):
    return ComposableTrainingModel(model, shared.lm_head, shared.embed_tokens, None,
        num_anchors=anchors, objective_chunk_blocks=chunk, complete_blocks=True, supervision_mode=mode)


def upstream_model(model, shared):
    # These files are unmodified and SHA-pinned in reference/provenance.json.
    if str(REFERENCE) not in sys.path:
        sys.path.insert(0, str(REFERENCE))
    from deepspec.modeling.eagle3.qwen3.modeling import Qwen3Eagle3Model
    c = Qwen3Config(hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8, vocab_size=19)
    c.target_layer_ids = model.target_layer_ids
    c.ttt_length = model.ttt_length
    c.step_loss_decay = .8
    c._attn_implementation = "sdpa"
    ref = Qwen3Eagle3Model(c)
    ref.fc.load_state_dict(model.context.projection.state_dict())
    ref.norm.load_state_dict(model.backbone.norm.state_dict())
    ref.embed_tokens.load_state_dict(shared.embed_tokens.state_dict())
    ref.lm_head.load_state_dict(shared.lm_head.state_dict())
    source = model.backbone.layer
    for name in ("q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm"):
        getattr(ref.layers[0].self_attn, name).load_state_dict(getattr(source, name).state_dict())
    for name in ("mlp", "hidden_norm", "input_layernorm", "post_attention_layernorm"):
        getattr(ref.layers[0], name).load_state_dict(getattr(source, name).state_dict())
    return ref


def test_config_has_no_proposal_and_preserves_five_context_layers():
    c = config()
    model = ComposableDraftModel(c)
    assert not hasattr(model, "proposal")
    assert "proposal" not in c.composable_config
    assert model.context.projection.weight.shape == (16, 80)
    assert not hasattr(model.context, "norm")
    assert model.ttt_length == 3
    assert c.composable_config["objective"]["type"] == "soft_ce"
    torch.testing.assert_close(position_weights(torch.ones(1, 1, 7), -1 / math.log(.8)),
                               .8 ** torch.arange(7).float()[None, None])
    c = config()
    c.composable_config["context"].pop("target_layer_ids")
    c.num_target_layers = 36
    assert ComposableDraftModel(c).target_layer_ids == [1, 9, 17, 25, 33]


@pytest.mark.parametrize("change", ["proposal", "layers", "attention", "context", "operator", "selector"])
def test_invalid_combinations(change):
    c = config()
    recipe = c.composable_config
    if change == "proposal": recipe["proposal"] = {"type": "parallel"}
    if change == "layers": recipe["backbone"]["layers"] *= 2; c.num_hidden_layers = 2
    if change == "attention": recipe["backbone"]["layers"][0]["attention"] = "mha"
    if change == "context": recipe["context"]["type"] = "dflash"
    if change == "operator": recipe["backbone"]["layers"][0]["local_operator"] = {"type": "dflash2_conv"}
    if change == "selector": recipe["objective"] = {"type": "selector_losses"}
    with pytest.raises(ValueError): c.validate_recipe()


def test_ttt_matches_deepspec_every_hidden_kv_and_logit():
    c = config(block_size=8)
    model, shared = ComposableDraftModel(c), components(c)
    ref = upstream_model(model, shared)
    data = batch(c, count=1, length=9)
    anchors = torch.arange(1, 9)[None]
    seen = []
    handle = model.backbone.layer.register_forward_hook(lambda m, a, o: seen.append(o))
    selected = model.forward_ttt(data["target_hidden_states"], data["input_ids"],
        data["context_valid_mask"], anchors, shared)
    handle.remove()
    hidden = data["target_hidden_states"]
    ids, cache = data["input_ids"], DynamicCache()
    length = ids.size(1)
    for step in range(7):
        ids = torch.cat((ids[:, 1:], torch.zeros_like(ids[:, :1])), 1)
        out = ref(hidden_states=hidden, input_ids=ids, past_key_values=cache, use_cache=True,
            position_ids=torch.arange(length)[None], rope_cache_step_offset=True,
            attention_mask=training_mask(data["context_valid_mask"], length, step), return_logits=True)
        hidden = out.hidden_states
        torch.testing.assert_close(seen[step][0], hidden, rtol=2e-5, atol=2e-6)
        torch.testing.assert_close(seen[step][1], cache[0][0], rtol=2e-5, atol=2e-6)
        torch.testing.assert_close(seen[step][2], cache[0][1], rtol=2e-5, atol=2e-6)
        torch.testing.assert_close(selected[:, :, step], hidden[:, :-1], rtol=2e-5, atol=2e-6)
        torch.testing.assert_close(model.score_eagle3(selected[:, :, step], shared).draft_logits,
                                   out.draft_logits[:, :-1], rtol=2e-5, atol=2e-6)


def test_mask_matches_independent_upstream_callback_and_isolates_branches():
    from torch.nn.attention.flex_attention import create_mask
    c = config(); model, shared = ComposableDraftModel(c), components(c)
    # Import upstream through its model fixture before reading the callback.
    upstream_model(model, shared)
    from deepspec.modeling.eagle3.common import create_eagle3_attention_mask
    valid = torch.tensor([[1] * 6 + [0] * 2, [1] * 8], dtype=torch.bool)
    for step in range(3):
        ref = create_eagle3_attention_mask(attention_mask=valid, q_len=8, kv_len=8 * (step + 1),
                                           lck=step, device=valid.device)
        dense = create_mask(ref.mask_mod, 2, 1, 8, 8 * (step + 1), device="cpu")
        assert torch.equal(training_mask(valid, 8, step), dense)
        mask = dense[1, 0]
        assert not mask[2, 3:8].any()
        for block in range(1, step + 1):
            assert mask[2, block * 8 + 2]
            assert mask[2, block * 8:block * 8 + 8].sum() == 1


def test_sample_epoch_matches_current_sampler_and_ignores_padding():
    c = config(); data = batch(c)
    kwargs = dict(block_size=4, num_anchors=3, seed=42, epoch=2, sample_indices=[11, 12])
    standard = prepare_sample_epoch_layout(data["input_ids"], data["loss_mask"], data["context_valid_mask"], **kwargs)
    eagle = prepare_sample_epoch_layout(data["input_ids"], data["loss_mask"], data["context_valid_mask"], minimum_anchor=1, **kwargs)
    assert torch.equal(standard.anchor_positions, eagle.anchor_positions)
    ids = torch.nn.functional.pad(data["input_ids"], (0, 5))
    mask = torch.nn.functional.pad(data["loss_mask"], (0, 5))
    valid = torch.nn.functional.pad(data["context_valid_mask"], (0, 5))
    padded = prepare_sample_epoch_layout(ids, mask, valid, minimum_anchor=1, **kwargs)
    assert torch.equal(padded.anchor_positions, eagle.anchor_positions)
    again = prepare_sample_epoch_layout(ids[:1], mask[:1], valid[:1], minimum_anchor=1,
        **{**kwargs, "sample_indices": [11]})
    assert torch.equal(again.anchor_positions, eagle.anchor_positions[:1])


def test_sampled_all_only_change_supervision_not_ttt():
    c = config(); shared = components(c); data = batch(c)
    model = ComposableDraftModel(c)
    traces = []
    handle = model.backbone.layer.register_forward_hook(lambda m, a, o: traces.append(tuple(x.detach().clone() for x in o)))
    sampled = wrapper(model, shared, "sampled", anchors=2)(**data, collect_detailed_metrics=True)
    first = traces[:]; traces.clear()
    full = wrapper(model, shared, "all")(**data, collect_detailed_metrics=True)
    handle.remove()
    for a, b in zip(first, traces):
        for x, y in zip(a, b): torch.testing.assert_close(x, y, rtol=0, atol=0)
    assert sampled.sum_metrics["supervision/anchors"] == 4
    assert full.sum_metrics["supervision/anchors"] > 4
    # Selecting all candidates gives exactly the same scalar and gradient.
    a = wrapper(model, shared, "sampled", anchors=512)(**data, collect_detailed_metrics=True)
    b = wrapper(model, shared, "all")(**data, collect_detailed_metrics=True)
    torch.testing.assert_close(a.loss, b.loss)
    grads_a = torch.autograd.grad(a.loss, tuple(model.parameters()))
    grads_b = torch.autograd.grad(b.loss, tuple(model.parameters()))
    for x, y in zip(grads_a, grads_b): torch.testing.assert_close(x, y, rtol=1e-4, atol=2e-6)
    for name in a.ratio_metrics:
        x, y = a.ratio_metrics[name], b.ratio_metrics[name]
        torch.testing.assert_close(x.numerator / x.denominator, y.numerator / y.denominator)


OBJECTIVES = [
    {"type": kind} for kind in ["hard_ce", "soft_ce", "forward_kl", "tv", "l1", "nla", "lk_alpha",
                               "bv", "dpace", "tv_survival", "opd_survival"]
] + [{"type": "weighted_sum", "components": [{"type": "hard_ce", "weight": .1}, {"type": "l1", "weight": .9}]}]


@pytest.mark.parametrize("objective", OBJECTIVES, ids=lambda x: x["type"])
def test_objectives_gradients_chunking_and_metrics(objective):
    c = config(deepcopy(objective)); model, shared, data = ComposableDraftModel(c), components(c), batch(c)
    a = wrapper(model, shared, chunk=0)(**data, collect_detailed_metrics=True)
    b = wrapper(model, shared, chunk=2)(**data, collect_detailed_metrics=True)
    assert torch.isfinite(a.loss)
    torch.testing.assert_close(a.loss, b.loss, rtol=1e-5, atol=2e-6)
    ga = torch.autograd.grad(a.loss, tuple(model.parameters()))
    gb = torch.autograd.grad(b.loss, tuple(model.parameters()))
    for x, y in zip(ga, gb):
        assert torch.isfinite(x).all()
        torch.testing.assert_close(x, y, rtol=2e-4, atol=2e-6)
    assert {"hard_label/token_accuracy", "hard_label/block_accepted_length", "expected_accept_len"} <= a.ratio_metrics.keys()
    if objective["type"] == "bv":
        assert wrapper(model, shared).loss_branches.objectives["final"].annealing_score_floor is None


def test_supervision_and_teacher_offsets_and_single_decay():
    c = config(); model, shared, data = ComposableDraftModel(c), components(c), batch(c, count=1)
    layout = prepare_layout(data["input_ids"], data["loss_mask"], 4, 9,
                            complete_blocks=True, supervision_mode="all", minimum_anchor=1)
    selected = model.forward_ttt(data["target_hidden_states"], data["input_ids"],
                                 data["context_valid_mask"], layout.anchor_positions, shared)
    logits = model.score_eagle3(selected, shared).draft_logits
    index = layout.anchor_positions[..., None] + torch.arange(3)
    teacher = data["target_last_hidden_states"].gather(1, index.clamp_max(9).flatten(1)[..., None].expand(-1, -1, 16)).reshape(1, -1, 3, 16)
    weights = layout.prediction_mask * (.8 ** torch.arange(3))[None, None]
    token_ce = -(shared.lm_head(teacher).float().softmax(-1) * logits.float().log_softmax(-1)).sum(-1)
    manual = (token_ce * weights).sum() / weights.sum()
    got = wrapper(model, shared)(**data)
    torch.testing.assert_close(got.loss, manual)
    valid = layout.block_keep_mask
    assert torch.equal(layout.labels[valid], data["input_ids"][0, index[valid] + 1])


def test_checkpoint_and_cross_step_gradient():
    c = config(); model, shared, data = ComposableDraftModel(c), components(c), batch(c)
    plain = wrapper(model, shared)(**data)
    ga = torch.autograd.grad(plain.loss, tuple(model.parameters()))
    model.backbone.gradient_checkpointing = True
    checked = wrapper(model, shared)(**data)
    gb = torch.autograd.grad(checked.loss, tuple(model.parameters()))
    torch.testing.assert_close(plain.loss, checked.loss)
    for x, y in zip(ga, gb): torch.testing.assert_close(x, y, rtol=1e-5, atol=1e-6)
    model.backbone.gradient_checkpointing = False
    states = []
    handle = model.backbone.layer.register_forward_hook(lambda m, a, o: states.append(o[0]))
    selected = model.forward_ttt(data["target_hidden_states"], data["input_ids"], data["context_valid_mask"], torch.tensor([[2], [2]]), shared)
    handle.remove()
    earlier = torch.autograd.grad(selected[:, :, -1].square().sum(), states[0])[0]
    assert earlier.abs().sum() > 0


def test_padding_empty_short_sequences_and_unsupported_opd():
    c = config(); model, shared, data = ComposableDraftModel(c), components(c), batch(c)
    data["context_valid_mask"][0, 7:] = False
    data["loss_mask"][0, 7:] = 0
    out = wrapper(model, shared)(**data)
    assert torch.isfinite(out.loss)
    for length in (1, 2, 3):
        short = batch(c, length=length)
        short["loss_mask"].zero_()
        out = wrapper(model, shared)(**short)
        assert out.loss == 0 and out.loss_terms.denominator == 0
        out.loss.backward()
    with pytest.raises(ValueError, match="sampled_anchor"):
        ComposableTrainingModel(model, shared.lm_head, shared.embed_tokens, None, opd_data_mode="sampled_anchor")


def target_model(c):
    return Qwen3ForCausalLM(Qwen3Config(hidden_size=16, intermediate_size=32, num_hidden_layers=6,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8, vocab_size=19,
        eos_token_id=None, pad_token_id=0, _attn_implementation="sdpa")).eval()


@pytest.mark.parametrize("budget", [0, 1, 2, 9, 24])
def test_greedy_decode_equals_target(budget):
    c = config(); model, target = ComposableDraftModel(c).eval(), target_model(c)
    prompt = torch.tensor([[1, 4, 2, 7]])
    result = generate_eagle3(target, model, prompt, "tiny", 0., set(), budget)
    oracle = ([] if not budget else target.generate(prompt, do_sample=False, max_new_tokens=budget,
        eos_token_id=None, pad_token_id=0)[0, prompt.size(1):].tolist())
    assert result["generated_ids"] == oracle
    assert len(target.model.layers[0]._forward_hooks) == 0
    native = model.spec_generate(target, prompt, budget)
    assert native[0].tolist() == prompt[0].tolist() + oracle


@pytest.mark.parametrize("accepted", [0, 1, 2, 3])
def test_cache_rebuild_uses_true_target_features(accepted):
    c = config(); model, target = ComposableDraftModel(c).eval(), target_model(c)
    prompt = torch.tensor([[1, 4, 2, 7]])
    capture, cache = TargetFeatures(target, model.target_layer_ids), DynamicCache()
    shared = SharedOps(target.get_input_embeddings(), target.get_output_embeddings())
    try:
        _, features = capture.forward(prompt, cache)
        adapter = Eagle3Adapter(model, target)
        adapter.prefill(prompt, features, 5)
        assert adapter.state.length == 4
        tokens, _ = adapter.propose(0., 1., torch.Generator().manual_seed(2))
        _, verified = capture.forward(torch.tensor([[5] + tokens]), cache)
        adapter.update(verified, tokens[:accepted], 6)
        assert adapter.state.length == 4 + accepted + 1
        # Full prefix rebuild is an independent oracle for crop + extension.
        committed = prompt[0].tolist() + [5] + tokens[:accepted]
        full_cache = DynamicCache()
        _, full_features = capture.forward(torch.tensor([committed]), full_cache)
        shifted = torch.tensor([committed[1:] + [6]])
        full = Eagle3State()
        model.extend_eagle3(full_features, shifted, full, shared, target_features=True)
        torch.testing.assert_close(adapter.state.key, full.key, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(adapter.state.value, full.value, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(adapter.state.hidden, full.hidden, rtol=1e-5, atol=1e-6)
    finally:
        capture.close()


def test_random_decode_reproducibility_and_eos():
    c = config(); model, target = ComposableDraftModel(c).eval(), target_model(c)
    prompt = torch.tensor([[1, 2, 3]])
    a = generate_eagle3(target, model, prompt, "random", 1., set(), 16, .95)
    torch.rand(100)
    b = generate_eagle3(target, model, prompt, "random", 1., set(), 16, .95)
    assert a == b
    assert request_seed("x", 1., "proposal") != request_seed("x", 1., "verifier")
    first = generate_eagle3(target, model, prompt, "eos", 0., set(), 1)["generated_ids"][0]
    stopped = generate_eagle3(target, model, prompt, "eos", 0., {first}, 20)
    assert stopped["generated_ids"] == [first] and stopped["cycles"] == []


def test_exhaustive_one_step_rejection_preserves_target_distribution():
    p = torch.tensor([.1, .3, .6], dtype=torch.float64)
    for q in (torch.tensor([.6, .3, .1]), torch.tensor([0., .8, .2])):
        q = q.double()
        accepted = torch.minimum(p, q)
        rejected_mass = 1 - accepted.sum()
        residual = (p - q).clamp_min(0)
        marginal = accepted + rejected_mass * residual / residual.sum()
        torch.testing.assert_close(marginal, p)
    # Every nucleus distribution includes its threshold-crossing token.
    assert probabilities(torch.log(torch.tensor([.6, .3, .1])), 1., .7).tolist() == pytest.approx([2/3, 1/3, 0])


def test_runtime_verifier_matches_frozen_protocol():
    path = ROOT / "tests/acceptance/qwen3/engine.py"
    spec = importlib.util.spec_from_file_location("eagle3_benchmark_engine", path)
    engine = importlib.util.module_from_spec(spec); spec.loader.exec_module(engine)
    rows = [torch.softmax(torch.randn(19), -1) for _ in range(4)]
    q_rows = [torch.softmax(torch.randn(19), -1) for _ in range(3)]
    for seed_value in range(50):
        tokens = [int(q.argmax()) for q in q_rows]
        for temperature in (0., 1.):
            a = verify(tokens, q_rows, rows, temperature, torch.Generator().manual_seed(seed_value))
            b = engine.verify(tokens, q_rows, rows, temperature, torch.Generator().manual_seed(seed_value))
            assert a == b
    c = config(block_size=8); model, target = ComposableDraftModel(c).eval(), target_model(c)
    prompt = torch.tensor([[1, 4, 2, 7]])
    expected = generate_eagle3(target, model, prompt, "harness", 1., set(), 13, .95)
    assert engine.generate(target, model, prompt, "harness", 1., set(), 13, .95) == expected


def test_eos_and_budget_inside_an_accepted_block(monkeypatch):
    import litespec.modeling.draft.composable.eagle3_inference as runtime
    c = config(); model, target = ComposableDraftModel(c).eval(), target_model(c)
    # Deterministic cyclic target: ensure EOS can occur after the initial anchor.
    with torch.no_grad():
        for layer in target.model.layers:
            layer.self_attn.o_proj.weight.zero_()
            layer.mlp.down_proj.weight.zero_()
        target.model.embed_tokens.weight.zero_()
        target.model.embed_tokens.weight[:16].copy_(torch.eye(16))
        target.lm_head.weight.zero_()
        for token in range(16):
            target.lm_head.weight[(token + 1) % 16, token] = 1.
    prompt = torch.tensor([[1, 4, 2, 7]])
    oracle = target.generate(prompt, do_sample=False, max_new_tokens=4,
                            eos_token_id=None, pad_token_id=0)[0, -4:].tolist()

    class OracleAdapter(Eagle3Adapter):
        def prefill(self, ids, hidden, anchor):
            super().prefill(ids, hidden, anchor)
            self.prefix = ids[0].tolist() + [anchor]

        def propose(self, temperature, top_p, generator):
            self.cache_len_before = self.state.length
            prefix = list(self.prefix)
            tokens, rows = [], []
            for step in range(self.draft.ttt_length):
                out = target(torch.tensor([prefix]), use_cache=False)
                q = probabilities(out.logits[0, -1], 0., 1.)
                token = int(q.argmax()); tokens.append(token); rows.append(q); prefix.append(token)
            return tokens, rows

        def update(self, hidden, tokens, anchor):
            super().update(hidden, tokens, anchor)
            self.prefix.extend(list(tokens) + [anchor])

    monkeypatch.setattr(runtime, "Eagle3Adapter", OracleAdapter)
    result = generate_eagle3(target, model, prompt, "budget", 0., set(), 2)
    assert result["generated_ids"] == oracle[:2]
    assert result["cycles"][0]["accepted_draft_tokens"] == 3
    assert result["cycles"][0]["budget_truncated"] and result["cycles"][0]["censored"]
    # Choose the first token distinct from the initial anchor as an in-block EOS.
    eos_index = next((i for i in range(1, len(oracle)) if oracle[i] != oracle[0]), None)
    assert eos_index is not None, "Fixture must include an in-block EOS candidate"
    result = generate_eagle3(target, model, prompt, "eos-block", 0., {oracle[eos_index]}, 10)
    assert result["generated_ids"] == oracle[:eos_index + 1]
    assert result["cycles"][0]["eos"] and result["cycles"][0]["censored"]


def test_export_reload_and_cold_inference():
    # All generated files stay under tests/, including the subprocess cache.
    import tempfile
    c = config(); model, shared = ComposableDraftModel(c), components(c)
    data = batch(c)
    with tempfile.TemporaryDirectory(dir=ROOT / "tests/eagle3_integration") as folder:
        export_composable(model, folder)
        loaded = AutoModel.from_pretrained(folder, trust_remote_code=True, local_files_only=True)
        assert not hasattr(loaded, "proposal")
        assert loaded.config.composable_config == model.config.composable_config
        for key, value in model.state_dict().items(): torch.testing.assert_close(value, loaded.state_dict()[key])
        a = model.forward_ttt(data["target_hidden_states"], data["input_ids"], data["context_valid_mask"], torch.tensor([[2], [2]]), shared)
        b = loaded.forward_ttt(data["target_hidden_states"], data["input_ids"], data["context_valid_mask"], torch.tensor([[2], [2]]), shared)
        torch.testing.assert_close(a, b)
        code = '''import sys,torch
from transformers import AutoModel,Qwen3Config,Qwen3ForCausalLM
m=AutoModel.from_pretrained(sys.argv[1],trust_remote_code=True,local_files_only=True).eval()
t=Qwen3ForCausalLM(Qwen3Config(hidden_size=16,intermediate_size=32,num_hidden_layers=6,num_attention_heads=2,num_key_value_heads=1,head_dim=8,vocab_size=19,eos_token_id=None)).eval()
assert m.spec_generate(t,torch.tensor([[1,2,3]]),2).shape==(1,5)
assert not any(k.startswith('litespec') for k in sys.modules)
'''
        subprocess.run([sys.executable, "-c", code, str(Path(folder).resolve())], cwd=folder, check=True,
                       capture_output=True, text=True)


@pytest.mark.parametrize("kind", ["soft_ce", "opd_survival"])
def test_accumulated_microbatches_match_full_batch(kind):
    c = config({"type": kind})
    model, shared, data = ComposableDraftModel(c), components(c), batch(c)
    # Different complete-block counts exercise weighted/global denominators.
    data["loss_mask"][1, 7:] = 0
    train = wrapper(model, shared)
    full = train(**data)
    expected = torch.autograd.grad(full.loss, tuple(model.parameters()))
    micros = [{key: value[i:i + 1] for key, value in data.items()} for i in range(2)]
    metadata = []
    if kind == "opd_survival":
        with torch.no_grad():
            metadata = [train(**item, metadata_only=True) for item in micros]
        counts = {branch: {key: sum(m["component_denominators"][branch][key] for m in metadata)
                           for key in terms} for branch, terms in metadata[0]["component_denominators"].items()}
        outputs = [train(**item, opd_accepted_lengths=meta["accepted_lengths"],
            opd_global_denominators=counts, opd_reference_samples=torch.tensor(2.))
            for item, meta in zip(micros, metadata)]
    else:
        outputs = [train(**item) for item in micros]
    combined = sum(x.loss_terms.numerator for x in outputs) / sum(x.loss_terms.denominator for x in outputs)
    torch.testing.assert_close(combined, full.loss, rtol=1e-5, atol=2e-6)
    got = torch.autograd.grad(combined, tuple(model.parameters()))
    for x, y in zip(got, expected): torch.testing.assert_close(x, y, rtol=3e-4, atol=3e-6)


def _ddp_worker(rank, init_file):
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method="file://" + init_file, rank=rank, world_size=2)
    try:
        for kind in ("soft_ce", "opd_survival"):
            torch.manual_seed(321)
            c = config({"type": kind})
            model, shared, data = ComposableDraftModel(c), components(c), batch(c)
            data["loss_mask"][1, 7:] = 0
            reference = deepcopy(model)
            objective = wrapper(model, shared)
            ddp = DistributedDataParallel(objective)
            out = ddp(**{key: value[rank:rank + 1] for key, value in data.items()})
            global_den = out.loss_terms.denominator.detach().clone()
            dist.all_reduce(global_den)
            (out.loss_terms.numerator * 2 / global_den).backward()
            oracle = wrapper(reference, shared)
            # The independent full-batch oracle must not perform a collective.
            # Supply explicit component counts for the OPD objective.
            kwargs = {}
            if kind == "opd_survival":
                with torch.no_grad():
                    metadata = oracle(**data, metadata_only=True)
                kwargs = dict(opd_global_denominators=metadata["component_denominators"],
                              opd_reference_samples=torch.tensor(2.))
            full = oracle(**data, **kwargs)
            full.loss.backward()
            for a, b in zip(model.parameters(), reference.parameters()):
                torch.testing.assert_close(a.grad, b.grad, rtol=3e-4, atol=3e-6)
    finally:
        dist.destroy_process_group()


def test_two_rank_ddp_matches_full_batch():
    import tempfile
    import torch.multiprocessing as mp
    with tempfile.TemporaryDirectory(dir=ROOT / "tests/eagle3_integration") as directory:
        mp.spawn(_ddp_worker, args=(str(Path(directory).resolve() / "gloo-init"),), nprocs=2, join=True)


def test_resume_identity_rejects_supervision_mode_changes():
    from litespec.training.checkpoint import training_identity, validate_training_identity
    args = SimpleNamespace(algorithm="composable", tp_size=1, shard_draft_by_tp=False,
        disaggregate=False, accumulation_steps=1, batch_size=1, target_model_path="target",
        anchor_sampling="sample_epoch", seed=42, num_anchors=512,
        supervision_mode="sampled", opd_data_mode="teacher_forced")
    state = training_identity(args)
    validate_training_identity(state, args)
    args.supervision_mode = "all"
    with pytest.raises(ValueError, match="supervision_mode"):
        validate_training_identity(state, args)
    state = training_identity(args)
    validate_training_identity(state, args)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA FlexAttention kernel requires a GPU")
def test_cuda_flex_matches_sdpa_ttt_and_gradients():
    c = config()
    c.head_dim = 16
    c.target_attention["head_dim"] = 16
    c.composable_config["attention_configs"]["gqa"]["head_dim"] = 16
    c.validate_recipe()
    model, shared, data = ComposableDraftModel(c).cuda(), components(c), batch(c)
    shared.embed_tokens.cuda(); shared.lm_head.cuda()
    data = {key: value.cuda() for key, value in data.items()}
    baseline = wrapper(model, shared)(**data)
    grads = torch.autograd.grad(baseline.loss, tuple(model.parameters()))
    model.backbone.layer.backend = "flex_attention"
    flex = wrapper(model, shared)(**data)
    got = torch.autograd.grad(flex.loss, tuple(model.parameters()))
    torch.testing.assert_close(flex.loss, baseline.loss, rtol=2e-3, atol=2e-4)
    for a, b in zip(got, grads): torch.testing.assert_close(a, b, rtol=3e-3, atol=3e-4)
