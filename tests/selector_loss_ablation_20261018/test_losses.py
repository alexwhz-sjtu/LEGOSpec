"""Numerical and Selector integration tests for the authorized loss ablation."""
import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

ROOT = Path('/data/temp/LiteSpec_exp')
sys.path.insert(1, str(ROOT / 'tests/final_proposal_metrics_20261017/candidate/tests/final_proposal_metrics_20261017'))
from test_implementation import config, fixture
from litespec.core.composable.objectives import build_objective
from litespec.core.composable.objectives.base import Supervision
from litespec.core.composable.batch import prepare_layout
from litespec.core.composable.training_model import ComposableTrainingModel

MIX = {'type': 'weighted_sum', 'components': [
    {'type': 'hard_ce', 'weight': 0.1}, {'type': 'tv', 'weight': 0.9}]}
RECIPES = [{'type': 'dpace', 'alpha': 0.5}, {'type': 'nla'}, MIX]


@pytest.mark.parametrize('gamma', [None, 4.0])
def test_mixed_loss_and_gradient_match_manual_formula_with_ragged_masks(gamma):
    torch.manual_seed(18)
    z = torch.randn(2, 3, 4, 11, requires_grad=True)
    teacher = torch.randn_like(z, requires_grad=True)
    labels = torch.randint(11, (2, 3, 4))
    mask = torch.tensor([[[1, 1, 0, 0], [1, 1, 1, 1], [0, 0, 0, 0]],
                         [[1, 0, 0, 0], [1, 1, 1, 0], [1, 1, 0, 0]]]).float()
    recipe = dict(MIX, loss_decay_gamma=gamma)
    obj = build_objective(recipe, 'selector')
    assert obj.needs_teacher and not obj.sample_reduction
    result = obj(SimpleNamespace(draft_logits=z), Supervision(labels, mask, teacher))
    weights = mask if gamma is None else mask * torch.exp(-torch.arange(4) / gamma)
    ce = F.cross_entropy(z.flatten(0, -2), labels.flatten(), reduction='none').reshape_as(mask)
    l1 = (z.softmax(-1) - teacher.detach().softmax(-1)).abs().sum(-1)
    expected = ((0.1 * ce + 0.9 * l1) * weights).sum() / weights.sum()
    torch.testing.assert_close(result.loss, expected)
    torch.testing.assert_close(torch.autograd.grad(result.loss, z, retain_graph=True)[0],
                               torch.autograd.grad(expected, z)[0])
    assert torch.autograd.grad(result.loss, teacher, allow_unused=True)[0] is None
    torch.testing.assert_close(obj.native_denominator(SimpleNamespace(prediction_mask=mask)), weights.sum())


def test_tv_full_l1_range_and_zero():
    p = torch.tensor([[[[30., -30.]]]])
    z = torch.tensor([[[[-30., 30.]]]], requires_grad=True)
    s = Supervision(torch.zeros(1, 1, 1, dtype=torch.long), torch.ones(1, 1, 1), p)
    obj = build_objective({'type': 'tv'})
    torch.testing.assert_close(obj(SimpleNamespace(draft_logits=z), s).loss, torch.tensor(2.))
    torch.testing.assert_close(obj(SimpleNamespace(draft_logits=p), s).loss, torch.tensor(0.))


@pytest.mark.parametrize('parts', [[], [{'type':'nla','weight':1}],
    [{'type':'hard_ce','weight':-1}], [{'type':'tv','weight':float('nan')}],
    [{'type':'tv','weight':0}], [{'type':'tv','weight':1,'loss_decay_gamma':4}]])
def test_invalid_mixes_rejected(parts):
    with pytest.raises(ValueError):
        build_objective({'type':'weighted_sum','components':parts})


@pytest.mark.parametrize('recipe', RECIPES)
def test_selector_base_objective_chunking_and_final_metrics(recipe):
    torch.manual_seed(1818)
    cfg = config(proposal='selector')
    cfg.composable_config['objective'] = copy.deepcopy(recipe)
    model, embedding, head, ids, hidden, teacher, mask = fixture(cfg)
    other = copy.deepcopy(model)
    layout = prepare_layout(ids, mask, cfg.block_size, 4)
    full = ComposableTrainingModel(model, head, embedding, 0, num_anchors=4, objective_chunk_blocks=0)
    chunked = ComposableTrainingModel(other, head, embedding, 0, num_anchors=4, objective_chunk_blocks=1)
    seen = []
    handle = full.objective.register_forward_pre_hook(lambda module, args: seen.append(args[0]))
    off = full(ids, hidden, mask, teacher, layout=layout)
    handle.remove()
    assert seen and all(x.draft_logits is x.base_logits for x in seen)
    on = chunked(ids, hidden, mask, teacher, layout=layout, collect_detailed_metrics=True)
    torch.testing.assert_close(off.loss, on.loss)
    off.loss.backward(); on.loss.backward()
    for (name, a), (_, b) in zip(model.named_parameters(), other.named_parameters()):
        assert a.grad is not None and torch.isfinite(a.grad).all(), name
        torch.testing.assert_close(a.grad, b.grad, atol=3e-6, rtol=3e-5, msg=name)
    assert 'objective/selector_ce' in on.ratio_metrics
    for key in ['hard_label/token_accuracy','hard_label/block_accepted_length','expected_accept_len']:
        terms = on.ratio_metrics[key]
        assert torch.isfinite(terms.numerator) and terms.denominator > 0


@pytest.mark.parametrize('recipe', RECIPES)
def test_empty_mask_has_finite_zero_loss_and_gradient(recipe):
    z = torch.randn(2, 3, 4, 11, requires_grad=True)
    s = Supervision(torch.zeros(2,3,4,dtype=torch.long), torch.zeros(2,3,4), torch.randn_like(z))
    result = build_objective(recipe, 'selector')(SimpleNamespace(draft_logits=z), s)
    assert result.loss == 0 and torch.isfinite(result.loss)
    result.loss.backward(); assert torch.equal(z.grad, torch.zeros_like(z))


def test_zero_teacher_weight_does_not_require_teacher():
    obj = build_objective({'type':'weighted_sum','components':[
        {'type':'hard_ce','weight':1},{'type':'tv','weight':0}]})
    assert not obj.needs_teacher
    z = torch.randn(1,1,2,3,requires_grad=True)
    assert torch.isfinite(obj(SimpleNamespace(draft_logits=z),
        Supervision(torch.zeros(1,1,2,dtype=torch.long),torch.ones(1,1,2))).loss)
