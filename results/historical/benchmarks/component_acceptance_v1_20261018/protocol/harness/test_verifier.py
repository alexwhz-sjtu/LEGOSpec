"""Distribution and sampling checks; these are engineering tests, not paper runs."""
import math
import torch
from engine import probabilities, verify, request_seed


def test_nucleus():
    logits=torch.tensor([.6,.3,.1]).log()
    torch.testing.assert_close(probabilities(logits,1,.8),torch.tensor([2/3,1/3,0.]))
    torch.testing.assert_close(probabilities(logits,0),torch.tensor([1.,0.,0.]))
    assert request_seed('x',1,'proposal')!=request_seed('x',1,'verifier')


def test_exhaustive_rejection_distribution():
    # Sum accepted probability and rejection-correction probability exactly,
    # including sparse proposals. It must reproduce the target distribution.
    for p,q in [(torch.tensor([.2,.3,.5]),torch.tensor([.8,.2,0.])),
                (torch.tensor([.6,.4]),torch.tensor([.6,.4]))]:
        accepted=torch.minimum(p,q)
        reject=1-accepted.sum()
        residual=(p-q).clamp_min(0)
        result=accepted+(reject*residual/residual.sum() if residual.sum()>0 else 0)
        torch.testing.assert_close(result,p)


def test_greedy_prefix_and_bonus():
    rng=torch.Generator().manual_seed(42)
    q=[torch.tensor([1.,0.]),torch.tensor([0.,1.])]
    p=[torch.tensor([1.,0.]),torch.tensor([1.,0.]),torch.tensor([0.,1.])]
    assert verify([0,1],q,p,0,rng)==(1,0)
    p[1]=q[1]
    assert verify([0,1],q,p,0,rng)==(2,1)


def test_random_verifier_matches_target_with_sparse_q():
    rng=torch.Generator().manual_seed(42)
    p=torch.tensor([.15,.35,.5]);q=torch.tensor([.7,.3,0.])
    counts=torch.zeros(3)
    for _ in range(20000):
        token=int(torch.multinomial(q,1,generator=rng))
        accepted,replacement=verify([token],[q],[p,p],1,rng)
        # If accepted, the first emitted candidate is token; otherwise it is
        # the rejection replacement. The independent bonus is not measured.
        counts[token if accepted else replacement]+=1
    torch.testing.assert_close(counts/counts.sum(),p,atol=.012,rtol=0)


if __name__=='__main__':
    for test in [test_nucleus,test_exhaustive_rejection_distribution,
                 test_greedy_prefix_and_bonus,test_random_verifier_matches_target_with_sparse_q]:
        test(); print(test.__name__,'PASS',flush=True)
