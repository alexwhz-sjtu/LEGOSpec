import torch
from litespec.modeling.draft.composable import ComposableDraftConfig,ModelInputs

def cfg(kind='selector', conv=True):
    types=['gdn','gqa'] if kind=='gdn' else ['gqa','gqa']
    layers=[dict(attention=t,access='linear' if t=='gdn' else ('sliding_window' if kind=='swa' and i==0 else 'dense')) for i,t in enumerate(types)]
    if conv:
        for item in layers:item['local_operator']=dict(type='dflash2_conv',kernel_size=2,group_size=4)
    proposal=dict(type='topk_selector',rank=8,top_k=4,stop_gradient=False) if kind=='selector' else dict(type='rnn',rank=8)
    mix=dict(type='weighted_sum',loss_decay_gamma=4.,components=[dict(type='hard_ce',weight=.1),dict(type='l1',weight=.9)])
    objective=dict(type='selector_losses',loss_decay_gamma=4.,base=dict(type='hard_ce',weight=1.),final=dict(type='hard_ce',weight=1.)) if kind=='selector' else mix
    recipe=dict(context=dict(type='dflash',target_layer_ids=[0,2]),backbone=dict(type='layer_stack',layers=layers),proposal=proposal,objective=objective,
                attention_configs={t:(dict(linear_num_key_heads=2,linear_num_value_heads=4,linear_key_head_dim=8,linear_value_head_dim=8) if t=='gdn' else {}) for t in types},
                access_configs={'sliding_window':dict(window_size=3)} if kind=='swa' else {})
    return ComposableDraftConfig(hidden_size=32,intermediate_size=48,vocab_size=37,head_dim=8,num_attention_heads=4,num_key_value_heads=2,
        num_hidden_layers=2,num_target_layers=4,block_size=4,max_position_embeddings=128,
        execution_config=dict(attention_backend='sdpa',gdn_backend='reference'),composable_config=recipe)

def inputs(features,embeddings,anchor,start=0,state=None,doc=0):
    return ModelInputs(features,embeddings,torch.tensor([[anchor]]),torch.ones(1,1,dtype=torch.bool),context_position_ids=torch.arange(start,anchor)[None],
        document_ids=torch.full((1,anchor-start),doc),anchor_document_ids=torch.tensor([[doc]]),draft_state=state,mode='features')
