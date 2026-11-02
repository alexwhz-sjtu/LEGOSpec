"""Native draft projection policy; target modules retain their own dispatch."""

from sglang.kernels.ops.gemm.legospec_linear import legospec_cublas_linear
from sglang.srt.layers.linear import LinearBase
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from torch import nn


class _LEGOSpecLinear(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.in_features = original.in_features
        self.out_features = original.out_features
        self.weight = original.weight
        self.bias = original.bias

    def forward(self, hidden):
        return legospec_cublas_linear(hidden, self.weight, self.bias)


class _LEGOSpecLinearMethod(UnquantizedLinearMethod):
    def apply(self, layer, x, bias=None):
        return legospec_cublas_linear(x, layer.weight, bias)


def use_legospec_cublas(model):
    for parent in list(model.modules()):
        for name, child in list(parent.named_children()):
            if type(child) is nn.Linear:
                setattr(parent, name, _LEGOSpecLinear(child))
    for module in model.modules():
        if isinstance(module, LinearBase):
            if type(module.quant_method) is not UnquantizedLinearMethod:
                raise ValueError("LEGOSpec v1 requires unquantized projection methods")
            module.quant_method = _LEGOSpecLinearMethod()
