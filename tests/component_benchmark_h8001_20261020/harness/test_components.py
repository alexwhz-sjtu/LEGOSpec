"""Tiny end-to-end cache/greedy tests across contexts and final proposal heads."""
import copy
import torch
from transformers import Qwen3Config,Qwen3ForCausalLM
from litespec.modeling.draft.composable import ComposableDraftConfig,ComposableDraftModel
from engine import generate


def main():
    torch.set_num_threads(1)
    for context in ['dflash','dflare']:
        for proposal,layout in [(p,l) for p in ['parallel','rnn','selector'] for l in ['mha','hybrid','conv']]:
            torch.manual_seed(42)
            dims=dict(vocab_size=32,hidden_size=32,intermediate_size=64,
                num_attention_heads=4,num_key_value_heads=2,head_dim=8,
                max_position_embeddings=256,rope_theta=10000.,tie_word_embeddings=False,
                bos_token_id=1,eos_token_id=2)
            tc=Qwen3Config(num_hidden_layers=4,**dims);tc._attn_implementation='sdpa'
            target=Qwen3ForCausalLM(tc).eval()
            recipe={'context':{'type':context,'target_layer_ids':[0,2]},
                'backbone':{'type':'transformer','layers':[
                    {'attention':{'representation':'mha','access':'dense','backend':'sdpa'}} for _ in range(2)]},
                'proposal':{'type':proposal,'rank':8,'top_k':4},
                'objective':{'type':'hard_ce'},'mask_token_id':3}
            if layout=='hybrid':
                recipe['backbone']['layers'][0]['attention'].update(access='sliding_window',window_size=3)
                recipe['backbone']['layers'][1]['attention'].update(representation='mla',kv_lora_rank=8,qk_nope_head_dim=4,qk_rope_head_dim=4,v_head_dim=8,access='sliding_window',window_size=3)
            if layout=='conv':
                for layer in recipe['backbone']['layers']:
                    layer['local_operator']={'type':'dflash2_conv','kernel_size':2,'group_size':16}
            dc=ComposableDraftConfig(num_hidden_layers=2,num_target_layers=4,
                                     composable_config=recipe,**dims)
            draft=ComposableDraftModel(dc).eval()
            if proposal=='rnn':torch.nn.init.normal_(draft.proposal.markov_head.markov_w2.weight,std=.1)
            if proposal=='selector':torch.nn.init.normal_(draft.proposal.candidate_selector.successor_codebook,std=.1)
            ids=torch.tensor([[1,4,5,6]])
            measured=generate(target,draft,ids,'tiny',0,set(),max_new_tokens=24)
            with torch.inference_mode():
                oracle=target.generate(ids,do_sample=False,max_new_tokens=24,eos_token_id=None,
                                       pad_token_id=0)[0,ids.shape[1]:].tolist()
            assert measured['generated_ids']==oracle,(context,proposal,measured['generated_ids'],oracle)
            for c in measured['cycles']:
                assert 1<=c['tau']<=8
            sampled=generate(target,draft,ids,'tiny',1,set(),max_new_tokens=24)
            assert len(sampled['generated_ids'])==24
            print(context,proposal,layout,'greedy target equivalence + random generation PASS',flush=True)


if __name__=='__main__':main()
