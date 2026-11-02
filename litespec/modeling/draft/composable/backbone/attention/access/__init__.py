from .dense import dense_visibility
from .dsa import dsa_visibility
from .sliding_window import sliding_window_visibility

ACCESS = {"dense": dense_visibility, "dsa": dsa_visibility, "sliding_window": sliding_window_visibility}


def build_access(config):
    return ACCESS[config.get("access", "dense")]
