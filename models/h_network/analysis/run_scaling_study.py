#!/usr/bin/env python3
"""
Data Scaling Study: h-network performance at 20k / 500k / 2M CP events.

Uses the SAME architecture (SIREN 256x5, omega_0=15) from v16 at each data
size for a fair comparison. MC grid size scales with data.

Saves results to scaling_results.json and produces scaling_study.png.

Data sizes map to experimental programs (see docs/experimental_yields.md):
  - 20k:  BESIII full dataset
  - 500k: STCF early running
  - 2M:   STCF full dataset
"""
import sys, os, time, json, warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", message="Inputs to the softmax are not scaled down")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(SCRIPT_DIR, '..')
sys.path.insert(0, ROOT)

import torch
from torch import nn
from sklearn.model_selection import train_test_split

from nflows.flows import Flow
from nflows.distributions.normal import StandardNormal
from nflows.transforms import CompositeTransform, RandomPermutation
from nflows.transforms.coupling import PiecewiseRationalQuadraticCouplingTransform
from nflows.transforms import Sigmoid, InverseTransform

from Amplitude import SquareDalitzPlot2, BKpp
from DKpp import DKpp, DKppCorrelated, AmpSample

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
sys.stdout.flush()

# ===========================================================================
# Flow architecture (for frozen flavor flow)
# ===========================================================================
class FlowMLP(nn.Module):
    def __init__(self, in_features, out_features, hidden=64, layers=2, output_scale=0.30):
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

def create_flow(on_unit_box=True, num_flows=8, hidden_features=64, num_bins=8, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dim = 2
    transforms = []
    if on_unit_box:
        sigmoid = Sigmoid()
        if hasattr(sigmoid, 'temperature') and isinstance(sigmoid.temperature, torch.Tensor):
            sigmoid.temperature = sigmoid.temperature.to(device)
        if hasattr(sigmoid, 'eps') and isinstance(sigmoid.eps, torch.Tensor):
            sigmoid.eps = sigmoid.eps.to(device)
        transforms.append(InverseTransform(sigmoid))
    masks = [torch.tensor([1, 0], dtype=torch.bool), torch.tensor([0, 1], dtype=torch.bool)]
    for i in range(num_flows):
        mask = masks[i % 2]
        def conditioner(in_features, out_features, _hidden=hidden_features):
            return FlowMLP(in_features, out_features, hidden=_hidden, layers=2)
        transforms.append(PiecewiseRationalQuadraticCouplingTransform(
            mask=mask, transform_net_create_fn=conditioner, num_bins=num_bins,
            tails="linear", tail_bound=5.0, apply_unconditional_transform=False))
        transforms.append(RandomPermutation(features=dim))
    return Flow(CompositeTransform(transforms), StandardNormal(shape=[dim]))

# ===========================================================================
# Coordinate transforms
# ===========================================================================
M_D0, m_KS, m_pip, m_pim = 1.86484, 0.497611, 0.13957, 0.13957
sdp_obj = SquareDalitzPlot2(M_D0, m_KS, m_pip, m_pim)
IDX = (1, 2, 3)

def sdp_to_dp(points_sdp, sdp_obj, idx=(1,2,3)):
    i, j, k = idx
    out = np.empty_like(points_sdp, dtype=float)
    for n, (mp, th) in enumerate(points_sdp):
        sij, sik = sdp_obj.M_from_MpT(mp, th, i, j, k)
        out[n, 0], out[n, 1] = sij, sik
    return out

def dp_to_sdp(points_dp, sdp_obj, idx=(1,2,3)):
    i, j, k = idx
    s12, s13 = points_dp[:, 0], points_dp[:, 1]
    mp = np.vectorize(lambda a, b: sdp_obj.MpfromM(a, b, i, j, k), otypes=[float])(s12, s13)
    tp = np.vectorize(lambda a, b: sdp_obj.TfromM(a, b, i, j, k), otypes=[float])(s12, s13)
    return np.column_stack([mp, tp])

def swap_to_other_pair_sdp(s12, s13, sdp_obj, pair_swap=(1,3,2)):
    i2, j2, k2 = pair_swap
    s12, s13 = np.asarray(s12, dtype=float), np.asarray(s13, dtype=float)
    mp13 = np.empty_like(s12)
    th13 = np.empty_like(s12)
    for n in range(s12.size):
        mp13[n] = sdp_obj.MpfromM(s13[n], s12[n], i2, j2, k2)
        th13[n] = sdp_obj.TfromM(s13[n], s12[n], i2, j2, k2)
    return np.column_stack([mp13, th13])

def compute_sdp_jacobian(points_sdp, sdp_obj, idx=(1,2,3)):
    dp = sdp_to_dp(points_sdp, sdp_obj, idx=idx)
    jac_inv = np.empty(len(dp))
    for n in range(len(dp)):
        j = float(sdp_obj.jacobian(dp[n, 0], dp[n, 1], *idx))
        jac_inv[n] = 1.0 / max(j, 1e-30)
    return jac_inv

def eval_flavor_log_prob(points_sdp, flow, device, batch_size=50000):
    pts = torch.from_numpy(np.ascontiguousarray(points_sdp)).float().to(device)
    log_probs = []
    with torch.no_grad():
        for i in range(0, len(pts), batch_size):
            log_probs.append(flow.log_prob(pts[i:i+batch_size]).cpu())
    return torch.cat(log_probs).numpy()

def precompute_flavor_terms(points_sdp, flow, sdp_obj, device, idx=(1,2,3)):
    N = len(points_sdp)
    dp = sdp_to_dp(points_sdp, sdp_obj, idx=idx)
    s12, s13 = dp[:, 0], dp[:, 1]
    J_123 = np.empty(N)
    for n in range(N):
        J_123[n] = max(float(sdp_obj.jacobian(s12[n], s13[n], *idx)), 1e-30)
    logp = eval_flavor_log_prob(points_sdp, flow, device)
    p_sdp = np.exp(np.clip(logp, -50, 50))
    K_dp = p_sdp * J_123
    swap_sdp = swap_to_other_pair_sdp(s12, s13, sdp_obj, pair_swap=(1,3,2))
    J_132 = np.empty(N)
    for n in range(N):
        J_132[n] = max(float(sdp_obj.jacobian(s13[n], s12[n], 1, 3, 2)), 1e-30)
    logp_sw = eval_flavor_log_prob(swap_sdp.astype(np.float32), flow, device)
    p_sdp_sw = np.exp(np.clip(logp_sw, -50, 50))
    Kbar_dp = p_sdp_sw * J_132
    alpha_dp = np.sqrt(np.maximum(K_dp * Kbar_dp, 0.0))
    return K_dp, Kbar_dp, alpha_dp

# ===========================================================================
# SIREN h-network (same architecture as v16)
# ===========================================================================
class SirenLayer(nn.Module):
    def __init__(self, in_features, out_features, omega_0=15.0, is_first=False):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.linear = nn.Linear(in_features, out_features)
        self._init_weights()
    def _init_weights(self):
        with torch.no_grad():
            if self.is_first:
                n = self.linear.in_features
                self.linear.weight.uniform_(-1.0 / n, 1.0 / n)
            else:
                n = self.linear.in_features
                bound = np.sqrt(6.0 / n) / self.omega_0
                self.linear.weight.uniform_(-bound, bound)
    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))

class HNetworkSIREN(nn.Module):
    def __init__(self, hidden=256, layers=5, omega_0=15.0):
        super().__init__()
        self.layers_list = nn.ModuleList()
        self.layers_list.append(SirenLayer(2, hidden, omega_0=omega_0, is_first=True))
        for _ in range(layers - 2):
            self.layers_list.append(SirenLayer(hidden, hidden, omega_0=omega_0, is_first=False))
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

# ===========================================================================
# Load frozen flavor flow (shared across all runs)
# ===========================================================================
print("\n=== Loading flavor flow ===")
FLOW_CONFIG = {'num_flows': 12, 'hidden_features': 128, 'num_bins': 24}
flow_flavor = create_flow(**FLOW_CONFIG, device=str(device))
state = torch.load(os.path.join(ROOT, 'test_ensemble_2e6/trial_seed1.pth'),
                   map_location=device, weights_only=False)
flow_flavor.load_state_dict(state)
flow_flavor = flow_flavor.to(device).eval()
for p in flow_flavor.parameters():
    p.requires_grad = False
print(f"Flavor flow loaded. Parameters: {sum(p.numel() for p in flow_flavor.parameters()):,}")
sys.stdout.flush()

# ===========================================================================
# Precompute truth grid (shared across all runs)
# ===========================================================================
print("Precomputing truth grid for RMSE checks...")
N_check = 80
m_check = np.linspace(0.02, 0.98, N_check)
t_check = np.linspace(0.02, 0.98, N_check)
mm_c, tt_c = np.meshgrid(m_check, t_check)
check_pts = np.column_stack([mm_c.ravel(), tt_c.ravel()]).astype(np.float32)
check_pts_t = torch.from_numpy(check_pts).to(device)

dkpp = DKpp()
dp_check = sdp_to_dp(check_pts, sdp_obj, idx=IDX)
A12_check = dkpp.full(np.column_stack([dp_check[:, 0], dp_check[:, 1]]))
A13_check = dkpp.full(np.column_stack([dp_check[:, 1], dp_check[:, 0]]))
cos_dd_check = np.real(A12_check * np.conj(A13_check)) / (np.abs(A12_check) * np.abs(A13_check) + 1e-30)

# Fine grid for diagnostics
N_grid = 200
m_g = np.linspace(0.02, 0.98, N_grid)
t_g = np.linspace(0.02, 0.98, N_grid)
mm, tt = np.meshgrid(m_g, t_g)
grid_pts = np.column_stack([mm.ravel(), tt.ravel()]).astype(np.float32)
dp_grid = sdp_to_dp(grid_pts, sdp_obj, idx=IDX)
A12_grid = dkpp.full(np.column_stack([dp_grid[:, 0], dp_grid[:, 1]]))
A13_grid = dkpp.full(np.column_stack([dp_grid[:, 1], dp_grid[:, 0]]))
cos_dd_true = np.real(A12_grid * np.conj(A13_grid)) / (np.abs(A12_grid) * np.abs(A13_grid) + 1e-30)
alpha_grid_true = np.abs(A12_grid) * np.abs(A13_grid)
alpha_norm_grid = alpha_grid_true / alpha_grid_true.sum()

# Load B+/B- data (shared for gamma fits)
from iminuit import Minuit

bpm_path = os.path.join(ROOT, 'BpBm_samples.npz')
if os.path.exists(bpm_path):
    bdata = np.load(bpm_path)
    dataM_sdp, dataP_sdp = bdata['dataM_sdp'], bdata['dataP_sdp']
    mcM_sdp, mcP_sdp = bdata['mcM_sdp'], bdata['mcP_sdp']
else:
    print("Generating B+/B- pseudo-data...")
    true_rB, true_deltaB_deg, true_gamma_deg = 0.1, 130.0, 70.0
    bkpp_M = BKpp(rB=true_rB, deltaB=np.radians(true_deltaB_deg), gamma=np.radians(true_gamma_deg), Bcharge=-1)
    bkpp_P = BKpp(rB=true_rB, deltaB=np.radians(true_deltaB_deg), gamma=np.radians(true_gamma_deg), Bcharge=+1)
    dataM_dp = AmpSample(bkpp_M).generate(50000, nbatch=20000)
    dataP_dp = AmpSample(bkpp_P).generate(50000, nbatch=20000)
    dataM_sdp = dp_to_sdp(dataM_dp, sdp_obj, idx=IDX)
    dataP_sdp = dp_to_sdp(dataP_dp, sdp_obj, idx=IDX)
    eps_mc = 1e-6
    mcM_sdp = np.random.rand(500000, 2) * (1 - 2*eps_mc) + eps_mc
    mcP_sdp = np.random.rand(500000, 2) * (1 - 2*eps_mc) + eps_mc
    np.savez(bpm_path, dataM_sdp=dataM_sdp, dataP_sdp=dataP_sdp,
             mcM_sdp=mcM_sdp, mcP_sdp=mcP_sdp,
             dataM_dp=dataM_dp, dataP_dp=dataP_dp)

true_rB, true_deltaB_deg, true_gamma_deg = 0.1, 130.0, 70.0
print(f"B- data: {len(dataM_sdp)}, B+ data: {len(dataP_sdp)}")
sys.stdout.flush()

# ===========================================================================
# Loss / training / evaluation functions
# ===========================================================================
def compute_loss_with_health(h_net, even_pts, odd_pts, alpha_even, alpha_odd,
                             KpKbar_even, KpKbar_odd, mc_pts, w_mc, Gamma_KKbar,
                             eps=1e-8):
    h_even = h_net(even_pts)
    h_odd  = h_net(odd_pts)
    h_mc = h_net(mc_pts)
    I_h = torch.mean(w_mc * h_mc)
    Gamma_plus  = torch.clamp(Gamma_KKbar + 2.0 * I_h, min=eps)
    Gamma_minus = torch.clamp(Gamma_KKbar - 2.0 * I_h, min=eps)
    p_tilde_plus_raw  = KpKbar_even + 2.0 * alpha_even * h_even
    p_tilde_minus_raw = KpKbar_odd  - 2.0 * alpha_odd  * h_odd
    p_tilde_plus  = torch.clamp(p_tilde_plus_raw, min=eps)
    p_tilde_minus = torch.clamp(p_tilde_minus_raw, min=eps)
    nll = -torch.mean(torch.log(p_tilde_plus) - torch.log(Gamma_plus)) \
          -torch.mean(torch.log(p_tilde_minus) - torch.log(Gamma_minus))
    with torch.no_grad():
        n_clamp = (p_tilde_plus_raw < eps).sum().item() + (p_tilde_minus_raw < eps).sum().item()
        n_total = len(p_tilde_plus_raw) + len(p_tilde_minus_raw)
        clamp_frac = n_clamp / n_total
        h_all = torch.cat([h_even, h_odd])
        h_abs_mean = h_all.abs().mean().item()
        h_saturated = (h_all.abs() > 0.99).float().mean().item()
    health = {
        'I_h': I_h.item(), 'Gamma_plus': Gamma_plus.item(),
        'Gamma_minus': Gamma_minus.item(), 'clamp_frac': clamp_frac,
        'h_abs_mean': h_abs_mean, 'h_saturated_frac': h_saturated,
    }
    return nll, health

def _finite_pos(x, eps=1e-14):
    x = np.asarray(x)
    x = np.where(np.isfinite(x), x, 0.0)
    return np.maximum(x, eps)

def compute_interference_terms_from_h(points_sdp, flow_flavor, h_net, dkpp_model,
                                       sdp_obj, idx=(1,2,3), device=None):
    if device is None:
        device = next(flow_flavor.parameters()).device
    N = len(points_sdp)
    dp = sdp_to_dp(points_sdp, sdp_obj, idx=idx)
    s12, s13 = dp[:, 0], dp[:, 1]
    logp = eval_flavor_log_prob(points_sdp, flow_flavor, device)
    p_sdp = np.exp(np.clip(logp, -50, 50))
    J_123 = np.empty(N)
    for n in range(N):
        J_123[n] = max(float(sdp_obj.jacobian(s12[n], s13[n], *idx)), 1e-30)
    pF = p_sdp * J_123
    swap_sdp = swap_to_other_pair_sdp(s12, s13, sdp_obj, pair_swap=(1,3,2))
    logp_sw = eval_flavor_log_prob(swap_sdp.astype(np.float32), flow_flavor, device)
    p_sdp_sw = np.exp(np.clip(logp_sw, -50, 50))
    J_132 = np.empty(N)
    for n in range(N):
        J_132[n] = max(float(sdp_obj.jacobian(s13[n], s12[n], 1, 3, 2)), 1e-30)
    pFsw = p_sdp_sw * J_132
    invJ = 1.0 / J_123
    abJ = np.sqrt(np.maximum(pF * pFsw, 0.0))
    pts_t = torch.from_numpy(np.ascontiguousarray(points_sdp).astype(np.float32)).to(device)
    h_net.eval()
    with torch.no_grad():
        h_v = h_net(pts_t).cpu().numpy()
    C = abJ * h_v
    absS = abJ * np.sqrt(np.maximum(1.0 - h_v**2, 0.0))
    A12 = dkpp_model.full(np.column_stack([s12, s13]))
    A13 = dkpp_model.full(np.column_stack([s13, s12]))
    dphi = (np.angle(A12) - np.angle(A13) + np.pi) % (2*np.pi) - np.pi
    S = np.sign(np.sin(dphi)) * absS
    return dict(pF=_finite_pos(pF), pFsw=_finite_pos(pFsw),
                C=np.where(np.isfinite(C), C, 0.0),
                S=np.where(np.isfinite(S), S, 0.0),
                abJ=abJ, J=_finite_pos(invJ))

def run_gamma_fit(h_net, flow_flavor, dkpp, sdp_obj, IDX, device,
                  dataM_sdp, dataP_sdp, mcM_sdp, mcP_sdp,
                  true_rB, true_deltaB_deg, true_gamma_deg):
    tM = compute_interference_terms_from_h(dataM_sdp, flow_flavor, h_net, dkpp, sdp_obj, IDX, device)
    tP = compute_interference_terms_from_h(dataP_sdp, flow_flavor, h_net, dkpp, sdp_obj, IDX, device)
    mM = compute_interference_terms_from_h(mcM_sdp,   flow_flavor, h_net, dkpp, sdp_obj, IDX, device)
    mP = compute_interference_terms_from_h(mcP_sdp,   flow_flavor, h_net, dkpp, sdp_obj, IDX, device)

    N_expM, N_expP = len(tM['pF']), len(tP['pF'])

    def nll(rB, delta, gamma):
        thM, thP = np.radians(delta - gamma), np.radians(delta + gamma)
        pBm = _finite_pos(tM['pFsw'] + rB**2*tM['pF'] + 2*rB*(np.cos(thM)*tM['C'] - np.sin(thM)*tM['S']))
        pBp = _finite_pos(tP['pF'] + rB**2*tP['pFsw'] + 2*rB*(np.cos(thP)*tP['C'] + np.sin(thP)*tP['S']))
        mBm = _finite_pos(mM['J']*(mM['pFsw'] + rB**2*mM['pF'] + 2*rB*(np.cos(thM)*mM['C'] - np.sin(thM)*mM['S'])))
        mBp = _finite_pos(mP['J']*(mP['pF'] + rB**2*mP['pFsw'] + 2*rB*(np.cos(thP)*mP['C'] + np.sin(thP)*mP['S'])))
        return -np.log(pBm).sum() + N_expM*np.log(mBm.mean()) - np.log(pBp).sum() + N_expP*np.log(mBp.mean())

    m = Minuit(nll, rB=true_rB*0.2, delta=true_deltaB_deg+30, gamma=true_gamma_deg+30)
    m.limits['rB'] = (0, 1)
    m.limits['delta'] = (0, 360)
    m.limits['gamma'] = (0, 360)
    m.errors['rB'], m.errors['delta'], m.errors['gamma'] = 0.01, 2.0, 2.0
    m.errordef = Minuit.LIKELIHOOD
    m.migrad()
    m.hesse()

    rB_pull = abs(m.values['rB'] - true_rB) / m.errors['rB']
    delta_pull = abs(m.values['delta'] - true_deltaB_deg) / m.errors['delta']
    gamma_pull = abs(m.values['gamma'] - true_gamma_deg) / m.errors['gamma']

    return {
        'rB': m.values['rB'], 'rB_err': m.errors['rB'], 'rB_pull': rB_pull,
        'delta': m.values['delta'], 'delta_err': m.errors['delta'], 'delta_pull': delta_pull,
        'gamma': m.values['gamma'], 'gamma_err': m.errors['gamma'], 'gamma_pull': gamma_pull,
        'valid': m.valid, 'accurate': m.accurate,
    }

# ===========================================================================
# Scaling study configuration
# ===========================================================================
STUDY_CONFIGS = [
    {'n_cp': 20_000,    'n_mc': 50_000,  'label': '20k (BESIII full)'},
    {'n_cp': 500_000,   'n_mc': 250_000, 'label': '500k (STCF early)'},
    {'n_cp': 2_000_000, 'n_mc': 500_000, 'label': '2M (STCF full)'},
]

# Fixed architecture (same as v16)
HIDDEN    = 256
LAYERS    = 5
OMEGA_0   = 15.0
EPOCHS    = 100
LR        = 1e-4
BATCH_SIZE = 20000
EPS       = 1e-8
PATIENCE  = 30
MAX_CLAMP_FRAC = 0.05
MAX_LOSS = 10.0

results = []

for cfg_idx, cfg in enumerate(STUDY_CONFIGS):
    n_cp = cfg['n_cp']
    n_mc = cfg['n_mc']
    label = cfg['label']
    nk = n_cp // 1000

    print(f"\n{'='*70}")
    print(f"SCALING STUDY [{cfg_idx+1}/{len(STUDY_CONFIGS)}]: {label}")
    print(f"  N_CP={n_cp:,}, N_MC={n_mc:,}, SIREN {HIDDEN}x{LAYERS}")
    print(f"{'='*70}")
    sys.stdout.flush()

    # --- Load/generate data ---
    even_path = os.path.join(ROOT, f'D_Kspipi_even_SDP_{nk}k.npy')
    odd_path  = os.path.join(ROOT, f'D_Kspipi_odd_SDP_{nk}k.npy')

    if os.path.exists(even_path) and os.path.exists(odd_path):
        data_even = np.load(even_path)
        data_odd  = np.load(odd_path)
        print(f"Loaded cached: {even_path} {data_even.shape}")
    else:
        print(f"ERROR: Data files not found. Run: python generate_data.py --n-events {n_cp}")
        print(f"  Missing: {even_path}")
        continue

    even_train, even_val = train_test_split(data_even, test_size=0.1, random_state=42)
    odd_train,  odd_val  = train_test_split(data_odd,  test_size=0.1, random_state=42)
    print(f"Train: {len(even_train)} even, {len(odd_train)} odd | Val: {len(even_val)}, {len(odd_val)}")

    np.random.seed(123)
    mc_grid = np.random.rand(n_mc, 2).astype(np.float32)

    # --- Precompute flavor terms ---
    print("Precomputing flavor terms...")
    t0 = time.time()
    K_even_train, Kbar_even_train, alpha_even_train = precompute_flavor_terms(even_train, flow_flavor, sdp_obj, device, IDX)
    K_odd_train,  Kbar_odd_train,  alpha_odd_train  = precompute_flavor_terms(odd_train, flow_flavor, sdp_obj, device, IDX)
    K_even_val,   Kbar_even_val,   alpha_even_val   = precompute_flavor_terms(even_val, flow_flavor, sdp_obj, device, IDX)
    K_odd_val,    Kbar_odd_val,    alpha_odd_val    = precompute_flavor_terms(odd_val, flow_flavor, sdp_obj, device, IDX)
    K_mc, Kbar_mc, alpha_mc = precompute_flavor_terms(mc_grid, flow_flavor, sdp_obj, device, IDX)
    jac_inv_mc = compute_sdp_jacobian(mc_grid, sdp_obj, idx=IDX)
    precomp_time = time.time() - t0
    print(f"  Precomputation: {precomp_time:.1f}s")

    w_mc = jac_inv_mc * alpha_mc
    w_K_plus_Kbar = jac_inv_mc * (K_mc + Kbar_mc)
    Gamma_KKbar = np.mean(w_K_plus_Kbar)
    print(f"  Gamma(K+Kbar) = {Gamma_KKbar:.6f}")

    # Tensors
    even_train_t = torch.from_numpy(even_train.astype(np.float32)).to(device)
    odd_train_t  = torch.from_numpy(odd_train.astype(np.float32)).to(device)
    even_val_t   = torch.from_numpy(even_val.astype(np.float32)).to(device)
    odd_val_t    = torch.from_numpy(odd_val.astype(np.float32)).to(device)
    alpha_even_train_t = torch.from_numpy(alpha_even_train.astype(np.float32)).to(device)
    alpha_odd_train_t  = torch.from_numpy(alpha_odd_train.astype(np.float32)).to(device)
    alpha_even_val_t   = torch.from_numpy(alpha_even_val.astype(np.float32)).to(device)
    alpha_odd_val_t    = torch.from_numpy(alpha_odd_val.astype(np.float32)).to(device)
    KpKbar_even_train_t = torch.from_numpy((K_even_train + Kbar_even_train).astype(np.float32)).to(device)
    KpKbar_odd_train_t  = torch.from_numpy((K_odd_train + Kbar_odd_train).astype(np.float32)).to(device)
    KpKbar_even_val_t   = torch.from_numpy((K_even_val + Kbar_even_val).astype(np.float32)).to(device)
    KpKbar_odd_val_t    = torch.from_numpy((K_odd_val + Kbar_odd_val).astype(np.float32)).to(device)
    mc_grid_t  = torch.from_numpy(mc_grid).to(device)
    w_mc_t     = torch.from_numpy(w_mc.astype(np.float32)).to(device)
    Gamma_KKbar_t = torch.tensor(Gamma_KKbar, dtype=torch.float32, device=device)

    # --- Fresh h-network ---
    torch.manual_seed(42)
    h_net = HNetworkSIREN(hidden=HIDDEN, layers=LAYERS, omega_0=OMEGA_0).to(device)
    n_params = sum(p.numel() for p in h_net.parameters() if p.requires_grad)

    optimizer = torch.optim.Adam(h_net.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=15, min_lr=1e-6)

    N_even_tr, N_odd_tr = len(even_train_t), len(odd_train_t)
    N_batches = max(N_even_tr, N_odd_tr) // BATCH_SIZE
    best_val_loss, best_state, patience_counter = float('inf'), None, 0
    training_aborted = False

    print(f"\n--- Training SIREN {HIDDEN}x{LAYERS} on {nk}k data ({EPOCHS} epochs) ---")
    sys.stdout.flush()
    train_t0 = time.time()

    for epoch in range(1, EPOCHS + 1):
        h_net.train()
        perm_even = torch.randperm(N_even_tr, device=device)
        perm_odd  = torch.randperm(N_odd_tr, device=device)
        epoch_loss, n_steps = 0.0, 0

        for b in range(N_batches):
            idx_e = perm_even[b*BATCH_SIZE:(b+1)*BATCH_SIZE]
            idx_o = perm_odd[b*BATCH_SIZE:(b+1)*BATCH_SIZE]
            if len(idx_e) == 0 or len(idx_o) == 0:
                continue

            loss, health = compute_loss_with_health(
                h_net, even_train_t[idx_e], odd_train_t[idx_o],
                alpha_even_train_t[idx_e], alpha_odd_train_t[idx_o],
                KpKbar_even_train_t[idx_e], KpKbar_odd_train_t[idx_o],
                mc_grid_t, w_mc_t, Gamma_KKbar_t, eps=EPS)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(h_net.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_steps += 1

        avg_train = epoch_loss / max(n_steps, 1)

        h_net.eval()
        with torch.no_grad():
            val_loss, val_health = compute_loss_with_health(
                h_net, even_val_t, odd_val_t,
                alpha_even_val_t, alpha_odd_val_t,
                KpKbar_even_val_t, KpKbar_odd_val_t,
                mc_grid_t, w_mc_t, Gamma_KKbar_t, eps=EPS)
        vl = val_loss.item()
        scheduler.step(vl)

        if vl < best_val_loss:
            best_val_loss = vl
            best_state = {k: v.cpu().clone() for k, v in h_net.state_dict().items()}
            patience_counter = 0
            marker = ' *'
        else:
            patience_counter += 1
            marker = ''

        # Health checks
        abort_reason = None
        if not np.isfinite(avg_train) or avg_train > MAX_LOSS:
            abort_reason = f"Loss diverged: train={avg_train:.4f}"
        if val_health['clamp_frac'] > MAX_CLAMP_FRAC:
            abort_reason = f"Clamped: {val_health['clamp_frac']*100:.1f}%"
        if abs(val_health['I_h']) > 1.5:
            abort_reason = f"I_h out of range: {val_health['I_h']:.4f}"

        if epoch % 20 == 0 or epoch == 1 or abort_reason:
            lr = optimizer.param_groups[0]['lr']
            print(f"  Epoch {epoch:3d} | train {avg_train:.4f} | val {vl:.4f} | "
                  f"I_h {val_health['I_h']:+.5f} | clamp {val_health['clamp_frac']*100:.2f}% | "
                  f"lr {lr:.1e}{marker}")
            sys.stdout.flush()

        if abort_reason:
            print(f"\n  *** ABORTED: {abort_reason} ***")
            training_aborted = True
            break

        if patience_counter >= PATIENCE:
            print(f"  Early stopping at epoch {epoch}")
            break

    if best_state is not None:
        h_net.load_state_dict(best_state)
        h_net = h_net.to(device)
    train_time = time.time() - train_t0
    stopped_epoch = epoch
    print(f"  Training done in {train_time:.1f}s (stopped at epoch {stopped_epoch}). Best val: {best_val_loss:.4f}")

    # --- Diagnostics ---
    h_net.eval()
    with torch.no_grad():
        h_vals = h_net(torch.from_numpy(grid_pts).to(device)).cpu().numpy()

    residuals = h_vals - cos_dd_true
    rmse = np.sqrt(np.mean(residuals**2))
    rmse_weighted = np.sqrt(np.sum(alpha_norm_grid * residuals**2))
    print(f"  RMSE={rmse:.4f}, weighted RMSE={rmse_weighted:.4f}")

    # --- Gamma fit ---
    print(f"  Running gamma fit...")
    t0 = time.time()
    fit_result = run_gamma_fit(h_net, flow_flavor, dkpp, sdp_obj, IDX, device,
                                dataM_sdp, dataP_sdp, mcM_sdp, mcP_sdp,
                                true_rB, true_deltaB_deg, true_gamma_deg)
    fit_time = time.time() - t0
    print(f"  rB={fit_result['rB']:.4f}[{fit_result['rB_pull']:.1f}s] "
          f"gamma={fit_result['gamma']:.2f}[{fit_result['gamma_pull']:.1f}s] "
          f"delta={fit_result['delta']:.2f}[{fit_result['delta_pull']:.1f}s] "
          f"(fit: {fit_time:.1f}s)")

    # Save model
    model_path = os.path.join(SCRIPT_DIR, f'h_network_scaling_{nk}k.pth')
    torch.save(h_net.state_dict(), model_path)

    result = {
        'n_cp': n_cp,
        'n_mc': n_mc,
        'label': label,
        'n_params': n_params,
        'rmse': float(rmse),
        'rmse_weighted': float(rmse_weighted),
        'val_loss': float(best_val_loss),
        'stopped_epoch': stopped_epoch,
        'train_time_s': float(train_time),
        'precomp_time_s': float(precomp_time),
        'fit_time_s': float(fit_time),
        'aborted': training_aborted,
        **{f'fit_{k}': float(v) if isinstance(v, (float, np.floating)) else v
           for k, v in fit_result.items()},
    }
    results.append(result)
    sys.stdout.flush()

# ===========================================================================
# Save results
# ===========================================================================
results_path = os.path.join(SCRIPT_DIR, 'scaling_results.json')
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved results to {results_path}")

# ===========================================================================
# Summary table
# ===========================================================================
print(f"\n{'='*80}")
print(f"SCALING STUDY SUMMARY — SIREN {HIDDEN}x{LAYERS} ({n_params:,} params)")
print(f"{'='*80}")
print(f"{'N_CP':>10s} | {'Label':>20s} | {'RMSE':>7s} | {'w-RMSE':>7s} | "
      f"{'rB':>8s} | {'gamma':>10s} | {'val':>7s} | {'time':>6s}")
print(f"{'-'*80}")
for r in results:
    nk = r['n_cp'] // 1000
    print(f"{nk:>8d}k | {r['label']:>20s} | {r['rmse']:.4f} | {r['rmse_weighted']:.4f} | "
          f"{r['fit_rB']:.4f}[{r['fit_rB_pull']:.1f}s] | "
          f"{r['fit_gamma']:.2f}[{r['fit_gamma_pull']:.1f}s] | "
          f"{r['val_loss']:.4f} | {r['train_time_s']:.0f}s")
print(f"{'='*80}")

# ===========================================================================
# Comparison plot
# ===========================================================================
if len(results) >= 2:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    n_cp_vals = [r['n_cp'] for r in results]
    rmse_vals = [r['rmse'] for r in results]
    wrmse_vals = [r['rmse_weighted'] for r in results]
    gamma_vals = [r['fit_gamma'] for r in results]
    gamma_errs = [r['fit_gamma_err'] for r in results]
    rB_vals = [r['fit_rB'] for r in results]
    rB_errs = [r['fit_rB_err'] for r in results]
    labels = [r['label'].split('(')[0].strip() for r in results]

    # RMSE vs data size
    axes[0].plot(n_cp_vals, rmse_vals, 'o-', color='blue', label='RMSE', markersize=8)
    axes[0].plot(n_cp_vals, wrmse_vals, 's--', color='orange', label='Weighted RMSE', markersize=8)
    axes[0].set_xscale('log')
    axes[0].set_xlabel('N (CP events per tag)')
    axes[0].set_ylabel('RMSE')
    axes[0].set_title('h-network RMSE vs Data Size')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    for i, lab in enumerate(labels):
        axes[0].annotate(lab, (n_cp_vals[i], rmse_vals[i]),
                         textcoords="offset points", xytext=(0, 10), ha='center', fontsize=8)

    # Gamma vs data size
    axes[1].errorbar(n_cp_vals, gamma_vals, yerr=gamma_errs, fmt='o-', color='red',
                     markersize=8, capsize=5)
    axes[1].axhline(true_gamma_deg, color='gray', ls='--', alpha=0.7, label=f'True ({true_gamma_deg})')
    axes[1].set_xscale('log')
    axes[1].set_xlabel('N (CP events per tag)')
    axes[1].set_ylabel('gamma (deg)')
    axes[1].set_title('Extracted gamma vs Data Size')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # rB vs data size
    axes[2].errorbar(n_cp_vals, rB_vals, yerr=rB_errs, fmt='o-', color='green',
                     markersize=8, capsize=5)
    axes[2].axhline(true_rB, color='gray', ls='--', alpha=0.7, label=f'True ({true_rB})')
    axes[2].set_xscale('log')
    axes[2].set_xlabel('N (CP events per tag)')
    axes[2].set_ylabel('rB')
    axes[2].set_title('Extracted rB vs Data Size')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.suptitle(f'Data Scaling Study: SIREN {HIDDEN}x{LAYERS} h-network', fontsize=14)
    plt.tight_layout()
    plot_path = os.path.join(SCRIPT_DIR, 'scaling_study.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {plot_path}")

print("\nDone!")
