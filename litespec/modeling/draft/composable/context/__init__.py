from .hidden_gated import HiddenGatedContext
from .hidden_mlp import HiddenMLPContext
from .eagle3 import Eagle3Context

CONTEXTS = {"dflash": HiddenMLPContext, "dflare": HiddenGatedContext, "eagle3": Eagle3Context}


def build_context(config):
    return CONTEXTS[config.composable_config["context"]["type"]](config)
