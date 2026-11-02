"""Anchor-chunked gather attention and detached full-softmax teacher statistics."""

import torch
from torch.nn import functional as F


def gather_attention(module, query, key, value, inputs, positions, valid, documents):
    selection = inputs.selection
    if not (torch.equal(positions, selection.context_positions)
            and torch.equal(valid, selection.context_valid)
            and torch.equal(documents, selection.context_documents)):
        raise ValueError("DSA layers must have the same full-prefix context view")
    n, h, length, dim = query.shape
    a = inputs.anchor_positions.size(1)
    r = length // a
    s = positions.size(1)
    k = selection.indices.size(-1)
    outputs, teachers = [], []
    for start in range(0, a, selection.chunk_size):
        end = min(start + selection.chunk_size, a)
        c = end - start
        q = query[:, :, start*r:end*r].reshape(n,h,c,r,dim).transpose(1,2).reshape(n*c,h,r,dim)
        history_ids = selection.indices[:, start:end]
        block_ids = s + torch.arange(start*r,end*r,device=query.device).reshape(1,c,r).expand(n,-1,-1)
        ids = torch.cat((history_ids,block_ids),-1)
        def gather(states):
            expanded = states[:, None].expand(-1,c,-1,-1,-1)
            result = expanded.gather(3,ids[:,:,None,:,None].expand(-1,-1,states.size(1),-1,states.size(-1)))
            return result.reshape(n*c,states.size(1),k+r,states.size(-1))
        kc, vc = gather(key), gather(value)
        repeats = h // kc.size(1)
        if repeats != 1:
            kc, vc = kc.repeat_interleave(repeats,1), vc.repeat_interleave(repeats,1)
        block_valid = inputs.block_keep_mask[:,start:end]
        visible = torch.cat((selection.selected_valid[:,start:end],
                             block_valid[...,None].expand(-1,-1,r)),-1)
        visible = visible & block_valid[...,None]
        mask = visible.reshape(n*c,1,1,k+r)
        if inputs.collect_indexer_teacher:
            with torch.no_grad():
                logits = (q.detach().float() @ kc.detach().float().transpose(-2,-1)) * module.scaling
                logits = logits.masked_fill(~mask,-torch.inf)
                logits = torch.where(mask.any(-1,keepdim=True),logits,torch.zeros_like(logits))
                probabilities = logits.softmax(-1) * mask
                history = probabilities[...,:k].mean(1)
                history = history / history.sum(-1,keepdim=True).clamp_min(1e-30)
                pmask = (torch.ones(n,c,r-1,device=q.device) if inputs.prediction_mask is None
                         else inputs.prediction_mask[:,start:end].float())
                pmask = pmask.reshape(n*c,r-1)
                teacher = (history[:,1:] * pmask[...,None]).sum(1) / pmask.sum(-1,keepdim=True).clamp_min(1)
                teachers.append(teacher.reshape(n,c,k))
        if selection.backend == "reference":
            kd = key[:,None].expand(-1,c,-1,-1,-1).reshape(n*c,key.size(1),key.size(2),dim)
            vd = value[:,None].expand(-1,c,-1,-1,-1).reshape(n*c,value.size(1),value.size(2),value.size(-1))
            if repeats != 1:
                kd,vd=kd.repeat_interleave(repeats,1),vd.repeat_interleave(repeats,1)
            counts = torch.zeros(n*c,key.size(2),dtype=torch.long,device=q.device)
            counts.scatter_add_(1,ids.reshape(n*c,k+r),visible.reshape(n*c,k+r).long())
            result=F.scaled_dot_product_attention(q,kd,vd,attn_mask=(counts>0)[:,None,None],
                dropout_p=module.attention_dropout if module.training else 0,scale=module.scaling)
        else:
            result=F.scaled_dot_product_attention(q,kc,vc,attn_mask=mask,
                dropout_p=module.attention_dropout if module.training else 0,scale=module.scaling)
        result=result*block_valid.reshape(n*c,1,1,1)
        outputs.append(result.reshape(n,c,h,r,-1).transpose(1,2).reshape(n,h,c*r,-1))
    output=torch.cat(outputs,2).transpose(1,2).reshape(n,length,-1)
    output=module.o_proj(output)
    output=output*inputs.block_keep_mask.repeat_interleave(r,1)[...,None]
    return output, torch.cat(teachers,1) if teachers else None
