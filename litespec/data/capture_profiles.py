"""Strict optimized-capture profiles; decoder layer IDs are zero based."""
from dataclasses import dataclass

@dataclass(frozen=True)
class CaptureProfile:
    context_layers: tuple[int, ...]
    final_layer: int
    num_layers: int
    hidden_size: int
    vocab_size: int

    @property
    def captured_layers(self):
        return (*self.context_layers, self.final_layer)

PROFILES = {
    "qwen3_4b": CaptureProfile((1, 9, 17, 25, 33), 35, 36, 2560, 151936),
    "qwen35_9b": CaptureProfile((3, 11, 19, 23, 27), 31, 32, 4096, 248320),
}

def resolve_capture_profile(config, requested="auto", target_layer_ids=None):
    shape = (config.num_hidden_layers, config.hidden_size, config.vocab_size)
    matches = [name for name, p in PROFILES.items()
               if shape == (p.num_layers, p.hidden_size, p.vocab_size)]
    if requested == "auto":
        if len(matches) != 1:
            raise ValueError("Optimized capture needs Qwen3-4B or Qwen3.5-9B; use prepare_hidden_states for other targets")
        requested = matches[0]
    profile = PROFILES[requested]
    if shape != (profile.num_layers, profile.hidden_size, profile.vocab_size):
        raise ValueError("Target dimensions do not match the selected capture profile")
    if target_layer_ids is not None and tuple(target_layer_ids) != profile.context_layers:
        raise ValueError("Selected decoder layers differ from the strict capture profile")
    return profile
