"""Public regression tests for DFlare fusion, gradients and standalone export."""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import torch
from transformers import AutoModel
from litespec.modeling.draft.composable import ComposableDraftConfig, ComposableDraftModel
from litespec.modeling.draft.composable.context.hidden_gated import HiddenGatedContext
from litespec.modeling.draft.composable.export import export_composable
from litespec.training.optimizer import register_optimizer_parameters


def recipe(fusion=None):
    context = {"type": "dflare", "target_layer_ids": [0, 1, 2]}
    if fusion is not None:
        context["fusion"] = fusion
    return {"context": context,
        "backbone": {"type": "layer_stack", "layers": [
            {"attention": "gqa", "access": "dense"} for _ in range(2)]},
        "attention_configs": {"gqa": {}}, "proposal": {"type": "parallel"},
        "objective": {"type": "hard_ce"}}


def config(fusion=None):
    return ComposableDraftConfig(composable_config=recipe(fusion), hidden_size=8,
        intermediate_size=16, num_hidden_layers=2, num_target_layers=3,
        num_attention_heads=2, num_key_value_heads=1, head_dim=4,
        vocab_size=32, block_size=4)


class ContextFusionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.set_num_threads(1)

    def test_legacy_state_and_formula_unchanged(self):
        old = HiddenGatedContext(config())
        explicit = HiddenGatedContext(config("scalar_softmax"))
        self.assertEqual(set(old.state_dict()), {"layer_fusion_weights", "norm.weight"})
        explicit.load_state_dict(old.state_dict(), strict=True)
        x = torch.randn(2, 4, 24)
        inputs = SimpleNamespace(target_hidden_states=x)
        expected = [old.norm(torch.einsum("k,bskd->bsd", w, x.reshape(2,4,3,8)))
                    for w in old.layer_fusion_weights.softmax(-1)]
        for a,b,c in zip(old(inputs).context_hidden_states,
                         explicit(inputs).context_hidden_states, expected):
            self.assertTrue(torch.equal(a,b) and torch.equal(a,c))

    def test_channel_forward_and_gradients_match_scalar_reference(self):
        m = HiddenGatedContext(config("channel_softmax_shared_projection"))
        with torch.no_grad():
            m.layer_fusion_weights.normal_()
            m.shared_projection.weight.normal_(std=.2)
        x = torch.randn(2,4,24,requires_grad=True)
        got = torch.stack(m(SimpleNamespace(target_hidden_states=x)).context_hidden_states)
        weights = m.layer_fusion_weights.softmax(1)
        hidden = x.reshape(2,4,3,8)
        expected=[]
        for layer in range(2):
            channels=[sum(m.shared_projection.weight[o,d]
                * sum(weights[layer,k,d]*hidden[:,:,k,d] for k in range(3))
                for d in range(8)) for o in range(8)]
            expected.append(m.norm(torch.stack(channels,-1)))
        expected=torch.stack(expected)
        torch.testing.assert_close(got,expected,rtol=2e-5,atol=2e-6)
        probe=torch.randn_like(got);params=[x,*m.parameters()]
        a=torch.autograd.grad((got*probe).sum(),params,retain_graph=True)
        b=torch.autograd.grad((expected*probe).sum(),params)
        for v,w in zip(a,b):
            torch.testing.assert_close(v,w,rtol=5e-5,atol=1e-5)
            self.assertTrue(torch.isfinite(v).all())
            self.assertGreater(v.abs().sum().item(),0)
        torch.testing.assert_close(weights.sum(1),torch.ones(2,8))

    def test_initial_prior_and_projection_sharing(self):
        scalar=HiddenGatedContext(config())
        channel=HiddenGatedContext(config("channel_softmax_shared_projection"))
        torch.testing.assert_close(channel.layer_fusion_weights,
            scalar.layer_fusion_weights[...,None].expand(2,3,8))
        self.assertEqual(tuple(channel.shared_projection.weight.shape),(8,8))
        self.assertEqual(sum(n=="shared_projection.weight" for n,_ in channel.named_parameters()),1)
        self.assertIsNone(channel.shared_projection.bias)

    def test_optimizer_ownership(self):
        model=ComposableDraftModel(config("channel_softmax_shared_projection"))
        _,specs=register_optimizer_parameters(model)
        ownership={p.name:p.optimizer for p in specs}
        self.assertEqual(ownership["context.layer_fusion_weights"],"adamw")
        self.assertEqual(ownership["context.shared_projection.weight"],"muon")

    def test_invalid_fusion_configuration(self):
        for value in [None,False,"unknown"]:
            r=recipe();r["context"]["fusion"]=value
            with self.subTest(value=value), self.assertRaises(ValueError):
                c=config();c.composable_config=r;c.validate_recipe()
        r=recipe("channel_softmax_shared_projection");r["context"]["type"]="dflash"
        with self.assertRaises(ValueError):
            c=config();c.composable_config=r;c.validate_recipe()

    def test_export_reload_retains_variant_and_weights(self):
        for fusion in [None,"channel_softmax_shared_projection"]:
            with self.subTest(fusion=fusion):
                model=ComposableDraftModel(config(fusion))
                with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
                    export_composable(model,folder)
                    loaded=AutoModel.from_pretrained(folder,trust_remote_code=True,local_files_only=True)
                    self.assertEqual(model.config.composable_config,loaded.config.composable_config)
                    self.assertEqual(set(model.state_dict()),set(loaded.state_dict()))
                    for name,value in model.state_dict().items():
                        self.assertTrue(torch.equal(value,loaded.state_dict()[name]))


if __name__=="__main__":
    unittest.main()
