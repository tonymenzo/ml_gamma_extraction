"""
Neural network architectures for the constraint-aware gamma extraction.

Contains:
  - FlowMLP: conditioner sub-network used inside the RQ-NSF coupling layers
  - create_flow: constructs a rational-quadratic neural spline flow
  - SirenLayer: single SIREN hidden layer with sin() activation
  - HNetworkSIREN: h-network using SIREN architecture
  - SymmetricSIREN: h-network with enforced theta' -> 1-theta' symmetry
"""
import numpy as np
import torch
from torch import nn

from nflows.flows import Flow
from nflows.distributions.normal import StandardNormal
from nflows.transforms import CompositeTransform, RandomPermutation
from nflows.transforms.coupling import PiecewiseRationalQuadraticCouplingTransform
from nflows.transforms import Sigmoid, InverseTransform


# ═══════════════════════════════════════════════════════════════════════════
# Normalizing flow (must match collaborator's architecture)
# ═══════════════════════════════════════════════════════════════════════════

class FlowMLP(nn.Module):
    """Conditioner sub-network for RQ-NSF coupling layers.

    Two hidden layers with SiLU activations. Output is scaled by
    `output_scale` and zero-initialized so the spline starts as the
    identity map.
    """
    def __init__(self, in_features, out_features, hidden=64, layers=2,
                 output_scale=0.30):
        super().__init__()
        feats = [nn.Linear(in_features, hidden), nn.SiLU()]
        for _ in range(layers - 1):
            feats += [nn.Linear(hidden, hidden), nn.SiLU()]
        self.backbone = nn.Sequential(*feats)
        self.head = nn.Linear(hidden, out_features)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.output_scale = output_scale

    def forward(self, x, context=None):
        return self.head(self.backbone(x)) * self.output_scale


def create_flow(on_unit_box=True, num_flows=12, hidden_features=128,
                num_bins=24, device=None):
    """Construct a rational-quadratic neural spline flow.

    Architecture: `num_flows` coupling blocks with alternating masks and
    random permutations. Each coupling uses an RQ spline with `num_bins`
    bins and a FlowMLP conditioner with `hidden_features` units.

    Parameters
    ----------
    on_unit_box : bool
        If True, prepend an inverse-sigmoid (logit) transform to map
        [0,1]^2 inputs to R^2.
    num_flows : int
        Number of coupling blocks.
    hidden_features : int
        Hidden units in the conditioner MLP.
    num_bins : int
        Number of spline bins per dimension.
    device : str or None
        Device for tensor allocation.
    """
    if device is None:
        device = 'cpu'
    dim = 2
    transforms = []
    if on_unit_box:
        sigmoid = Sigmoid()
        if hasattr(sigmoid, 'temperature') and isinstance(sigmoid.temperature, torch.Tensor):
            sigmoid.temperature = sigmoid.temperature.to(device)
        if hasattr(sigmoid, 'eps') and isinstance(sigmoid.eps, torch.Tensor):
            sigmoid.eps = sigmoid.eps.to(device)
        transforms.append(InverseTransform(sigmoid))
    masks = [torch.tensor([1, 0], dtype=torch.bool),
             torch.tensor([0, 1], dtype=torch.bool)]
    for i in range(num_flows):
        mask = masks[i % 2]
        def conditioner(in_f, out_f, _h=hidden_features):
            return FlowMLP(in_f, out_f, hidden=_h, layers=2)
        transforms.append(PiecewiseRationalQuadraticCouplingTransform(
            mask=mask, transform_net_create_fn=conditioner, num_bins=num_bins,
            tails='linear', tail_bound=5.0, apply_unconditional_transform=False))
        transforms.append(RandomPermutation(features=dim))
    return Flow(CompositeTransform(transforms), StandardNormal(shape=[dim]))


# ═══════════════════════════════════════════════════════════════════════════
# SIREN h-network
# ═══════════════════════════════════════════════════════════════════════════

class SirenLayer(nn.Module):
    """Single SIREN hidden layer: z = sin(omega_0 * (Wz + b)).

    Weights are initialized following Sitzmann et al. (2020):
      - First layer: uniform[-1/n, 1/n]
      - Hidden layers: uniform[-sqrt(6/n)/omega_0, sqrt(6/n)/omega_0]
    """
    def __init__(self, in_features, out_features, omega_0=15.0, is_first=False):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.linear = nn.Linear(in_features, out_features)
        self._init_weights()

    def _init_weights(self):
        with torch.no_grad():
            n = self.linear.in_features
            if self.is_first:
                self.linear.weight.uniform_(-1.0 / n, 1.0 / n)
            else:
                bound = np.sqrt(6.0 / n) / self.omega_0
                self.linear.weight.uniform_(-bound, bound)

    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))


class HNetworkSIREN(nn.Module):
    """SIREN h-network: maps (m', theta') -> [-1, +1].

    A multilayer perceptron with sinusoidal activations and a tanh
    output layer that guarantees |h| <= 1 by construction.

    Parameters
    ----------
    hidden : int
        Number of units per hidden layer.
    layers : int
        Total number of layers (including first and last).
    omega_0 : float
        Frequency hyperparameter for sin() activations.
    """
    def __init__(self, hidden=256, layers=5, omega_0=15.0):
        super().__init__()
        self.layers_list = nn.ModuleList()
        self.layers_list.append(SirenLayer(2, hidden, omega_0, is_first=True))
        for _ in range(layers - 2):
            self.layers_list.append(SirenLayer(hidden, hidden, omega_0))
        self.head = nn.Linear(hidden, 1)
        with torch.no_grad():
            n = self.head.in_features
            bound = np.sqrt(6.0 / n) / omega_0
            self.head.weight.uniform_(-bound, bound)
            self.head.bias.zero_()

    def forward(self, x):
        for layer in self.layers_list:
            x = layer(x)
        return torch.tanh(self.head(x)).squeeze(-1)


class SymmetricSIREN(nn.Module):
    """SIREN with enforced theta' -> 1-theta' symmetry.

    Since cos(delta_D) is symmetric under pi+ <-> pi- exchange (which
    maps to theta' -> 1-theta' in the (2,3,1) SDP), we enforce this
    exactly by averaging the raw network output:

        h(m', theta') = [g(m', theta') + g(m', 1-theta')] / 2

    where g is the underlying HNetworkSIREN.
    """
    def __init__(self, hidden=256, layers=5, omega_0=15.0):
        super().__init__()
        self.net = HNetworkSIREN(hidden, layers, omega_0)

    def forward(self, x):
        x_flip = x.clone()
        x_flip[:, 1] = 1.0 - x_flip[:, 1]
        return 0.5 * (self.net(x) + self.net(x_flip))
