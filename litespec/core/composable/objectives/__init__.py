from .bv import BVObjective
from .ce import HardCEObjective
from .dpace import DPACEObjective
from .kl import ForwardKLObjective
from .nla import NLAObjective
from .soft_ce import SoftCEObjective
from .tv import TVObjective
from .l1 import L1Objective
from .tv_survival import TVSurvivalObjective
from .opd_survival import OPDSurvivalObjective
from .weighted_sum import WeightedSumObjective

OBJECTIVES = {
    "hard_ce": HardCEObjective,
    "soft_ce": SoftCEObjective,
    "forward_kl": ForwardKLObjective,
    "tv": TVObjective,
    "l1": L1Objective,
    "tv_survival": TVSurvivalObjective,
    "opd_survival": OPDSurvivalObjective,
    "nla": NLAObjective,
    "lk_alpha": NLAObjective,
    "dpace": DPACEObjective,
    "bv": BVObjective,
    "weighted_sum": WeightedSumObjective,
}


def build_objective(recipe, proposal_type="parallel"):
    return OBJECTIVES[recipe["type"]](recipe, proposal_type)
