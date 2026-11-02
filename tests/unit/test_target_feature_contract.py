"""Keep pre-final-norm context separate from teacher features and logits."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM
from litespec.modeling.target.training_target_model import HFTrainingTargetModel
from litespec.modeling.target.anchor_opd import Qwen3AnchorScorer
from litespec.training.recipes import _dflash_common as common
from litespec.training import setup


class TargetFeatureTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(12);torch.set_num_threads(1)
        c=Qwen3Config(hidden_size=16,intermediate_size=32,num_hidden_layers=3,
            num_attention_heads=2,num_key_value_heads=1,head_dim=8,vocab_size=32,
            max_position_embeddings=64,attention_dropout=0.)
        c._attn_implementation="eager"
        self.model=Qwen3ForCausalLM(c).eval()
        self.ids=torch.randint(32,(2,7));self.mask=torch.ones_like(self.ids)
        self.backend=HFTrainingTargetModel(self.model)

    def reference(self):
        outputs={}
        hook=self.model.model.layers[-1].register_forward_hook(
            lambda m,a,o:outputs.update(raw=o[0] if isinstance(o,tuple) else o))
        try:
            with torch.no_grad():
                outputs["hf"]=self.model(self.ids,attention_mask=self.mask,
                    output_hidden_states=True,use_cache=False)
        finally:hook.remove()
        return outputs

    def test_all_layers_and_teacher_match_hf(self):
        expected=self.reference()
        for logits in [False,True]:
            with self.subTest(logits=logits):
                got=self.backend.generate_training_data(self.ids,self.mask,self.mask,return_logits=logits)
                self.assertEqual(len(got.hidden_states),3)
                for i in [0,1]:
                    self.assertTrue(torch.equal(got.hidden_states[i],expected["hf"].hidden_states[i+1]))
                self.assertTrue(torch.equal(got.hidden_states[2],expected["raw"]))
                self.assertTrue(torch.equal(got.final_hidden_state,expected["hf"].hidden_states[-1]))
                self.assertFalse(torch.equal(got.hidden_states[2],got.final_hidden_state))
                self.assertTrue(torch.equal(self.model.lm_head(got.final_hidden_state),expected["hf"].logits))
                if logits:self.assertTrue(torch.equal(got.logits,expected["hf"].logits))
                else:self.assertIsNone(got.logits)

    def test_selected_context_excludes_teacher_layer(self):
        self.backend.set_capture_layers([0])
        got=self.backend.generate_training_data(self.ids,self.mask,self.mask,False)
        draft=SimpleNamespace(target_layer_ids=[0],config=SimpleNamespace(num_target_layers=3))
        with patch.object(common,"hidden_states_to_cuda",lambda x:x):
            context,teacher=common.prepare_dflash_family_features({},got,draft,True)
        self.assertTrue(torch.equal(context,got.hidden_states[0]))
        self.assertTrue(torch.equal(teacher,got.final_hidden_state))

    def test_context_selection_for_layer_only_and_embedding_inclusive_tuples(self):
        values=tuple(torch.full((1,2,4),i,dtype=torch.float32) for i in range(3))
        embedding=torch.full_like(values[0],-1)
        for states in [values,(embedding,*values)]:
            got=common.concatenate_target_hidden(states,[0,1],num_target_layers=3)
            self.assertTrue(torch.equal(got,torch.cat(values[:2],-1)))

    def test_target_packet_uses_raw_context_and_normalized_teacher(self):
        self.backend.set_capture_layers([2])
        expected=self.backend.generate_training_data(self.ids,self.mask,self.mask,False)
        metadata={"target_layer_ids":[2],"num_target_layers":3,"include_target_last_hidden":True}
        packet=common._target_packet(None,SimpleNamespace(is_target_leader=True),
            self.backend,metadata,(self.ids,self.mask,self.mask))
        self.assertTrue(torch.equal(packet.target_context_hidden,expected.hidden_states[2].bfloat16()))
        self.assertTrue(torch.equal(packet.target_last_hidden,expected.final_hidden_state.bfloat16()))

    def test_hooks_are_removed_after_failure(self):
        norm=self.model.model.norm;before=len(norm._forward_pre_hooks)
        with patch.object(self.model.model,"forward",side_effect=RuntimeError("fixture failure")):
            with self.assertRaisesRegex(RuntimeError,"fixture failure"):
                self.backend.generate_training_data(self.ids,self.mask,self.mask,False)
        self.assertEqual(len(norm._forward_pre_hooks),before)

    def test_sampled_anchor_context_is_raw_and_teacher_is_normalized(self):
        scorer=Qwen3AnchorScorer(self.model)
        anchors=torch.tensor([[1,4],[2,4]]);keep=torch.ones_like(anchors,dtype=torch.bool)
        prefix=scorer.prefill_anchors(self.ids,self.mask.bool(),anchors,keep,capture_layer_ids=[0,2])
        with torch.no_grad():
            captured={}
            handle=self.model.model.norm.register_forward_pre_hook(lambda m,a:captured.update(raw=a[0]))
            try:full=self.model.model(self.ids[:,:5],attention_mask=self.mask[:,:5],output_hidden_states=True)
            finally:handle.remove()
        torch.testing.assert_close(prefix.context_hidden[2],captured["raw"],rtol=2e-5,atol=2e-6)
        expected=full.last_hidden_state.gather(1,anchors[...,None].expand(-1,-1,16))
        torch.testing.assert_close(prefix.anchor_hidden,expected,rtol=2e-5,atol=2e-6)
        prefix.release()

    def test_cached_teacher_feature_and_tensor_cuda_helper(self):
        raw=torch.randn(2,7,16);teacher=torch.randn_like(raw)
        data={"hidden_states":{2:raw},"final_hidden_state":teacher,"final_norm_layer_id":2}
        anchors=torch.tensor([[0],[1]])
        with patch.object(torch.Tensor,"cuda",lambda tensor,**kwargs:tensor):
            self.assertIs(setup.hidden_states_to_cuda(teacher),teacher)
            _,got=setup.load_cached_target_data(data,anchor_positions=anchors,block_size=3,
                lm_head=self.model.lm_head,need_logits=True)
        expected=setup.project_cached_target_logits(teacher,anchors,3,self.model.lm_head)
        self.assertTrue(torch.equal(got,expected))


if __name__=="__main__":
    unittest.main()
