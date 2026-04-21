#!/usr/bin/env python3
"""
Symmetric h-network development and tuning.

Compares symmetric vs non-symmetric SIREN across hyperparameters
using quick training runs (small data, few epochs) on seed 0.

Usage:
    python run_tuning.py
"""
import sys, os, time, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
H_DIR = os.path.join(SCRIPT_DIR, '..')
ROOT = os.path.join(H_DIR, '..')
sys.path.insert(0, ROOT)

import torch
from torch import nn
from sklearn.model_selection import train_test_split

from nflows.flows import Flow
from nflows.distributions.normal import StandardNormal
from nflows.transforms import CompositeTransform, RandomPermutation
from nflows.transforms.coupling import PiecewiseRationalQuadraticCouplingTransform
from nflows.transforms import Sigmoid, InverseTransform

from Amplitude import SquareDalitzPlot2
from DKpp import DKpp

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ── Physics ───────────────────────────────────────────────────────────────
M_D0, m_KS, m_pip, m_pim = 1.86484, 0.497611, 0.13957, 0.13957
sdp_obj = SquareDalitzPlot2(M_D0, m_KS, m_pip, m_pim)
IDX = (2, 3, 1)
S_TOTAL = M_D0**2 + m_KS**2 + 2 * m_pip**2

# ── Flow architecture ─────────────────────────────────────────────────────
class FlowMLP(nn.Module):
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
    if device is None: device = 'cpu'
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

# ── SIREN layers ──────────────────────────────────────────────────────────
class SirenLayer(nn.Module):
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
    """Standard (non-symmetric) SIREN h-network."""
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
    """SIREN with enforced θ'→1-θ' symmetry: h(m',θ') = h(m',1-θ')."""
    def __init__(self, hidden=256, layers=5, omega_0=15.0):
        super().__init__()
        self.net = HNetworkSIREN(hidden, layers, omega_0)

    def forward(self, x):
        x_flip = x.clone()
        x_flip[:, 1] = 1.0 - x_flip[:, 1]
        return 0.5 * (self.net(x) + self.net(x_flip))

# ── Coordinate helpers ────────────────────────────────────────────────────
def sdp_to_dp(points_sdp):
    out = np.empty_like(points_sdp, dtype=float)
    for n, (mp, th) in enumerate(points_sdp):
        out[n, 0], out[n, 1] = sdp_obj.M_from_MpT(mp, th, *IDX)
    return out

def eval_flow_log_prob(points_sdp, flow, batch_size=50_000):
    pts = torch.from_numpy(np.ascontiguousarray(points_sdp)).float().to(device)
    lps = []
    with torch.no_grad():
        for i in range(0, len(pts), batch_size):
            lps.append(flow.log_prob(pts[i:i+batch_size]).cpu())
    return torch.cat(lps).numpy()

def precompute_flavor_terms(points_sdp, flow):
    N = len(points_sdp)
    dp = sdp_to_dp(points_sdp)
    s_ij, s_ik = dp[:, 0], dp[:, 1]
    J = np.empty(N)
    for n in range(N):
        J[n] = max(float(sdp_obj.jacobian(s_ij[n], s_ik[n], *IDX)), 1e-30)
    logp = eval_flow_log_prob(points_sdp, flow)
    K = np.exp(np.clip(logp, -50, 50)) * J
    swap = points_sdp.copy()
    swap[:, 1] = 1.0 - swap[:, 1]
    logp_sw = eval_flow_log_prob(swap, flow)
    Kbar = np.exp(np.clip(logp_sw, -50, 50)) * J
    alpha = np.sqrt(np.maximum(K * Kbar, 0.0))
    return K, Kbar, alpha, J

# ── Loss ──────────────────────────────────────────────────────────────────
EPS = 1e-8

def compute_loss(h_net, even_pts, odd_pts, alpha_even, alpha_odd,
                 KpKbar_even, KpKbar_odd, mc_pts, w_mc, Gamma_KKbar):
    h_even = h_net(even_pts)
    h_odd  = h_net(odd_pts)
    h_mc   = h_net(mc_pts)
    I_h = torch.mean(w_mc * h_mc)
    Gamma_plus  = torch.clamp(Gamma_KKbar + 2.0 * I_h, min=EPS)
    Gamma_minus = torch.clamp(Gamma_KKbar - 2.0 * I_h, min=EPS)
    p_plus  = torch.clamp(KpKbar_even + 2.0 * alpha_even * h_even, min=EPS)
    p_minus = torch.clamp(KpKbar_odd  - 2.0 * alpha_odd  * h_odd,  min=EPS)
    nll = (-torch.mean(torch.log(p_plus)  - torch.log(Gamma_plus))
           -torch.mean(torch.log(p_minus) - torch.log(Gamma_minus)))
    return nll

# ── Training loop ─────────────────────────────────────────────────────────
def train(h_net, even_tr_t, odd_tr_t, even_val_t, odd_val_t,
          a_etr_t, a_otr_t, a_evl_t, a_ovl_t,
          KK_etr_t, KK_otr_t, KK_evl_t, KK_ovl_t,
          mc_t, w_mc_t, G_t, lr=1e-4, epochs=30, batch_size=20_000,
          patience=20):
    optimizer = torch.optim.Adam(h_net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6)
    N_e, N_o = len(even_tr_t), len(odd_tr_t)
    N_batches = max(N_e, N_o) // batch_size
    best_val, best_state, pat_ctr = float('inf'), None, 0

    for epoch in range(1, epochs + 1):
        h_net.train()
        perm_e = torch.randperm(N_e, device=device)
        perm_o = torch.randperm(N_o, device=device)
        for b in range(N_batches):
            ie = perm_e[b*batch_size:(b+1)*batch_size]
            io = perm_o[b*batch_size:(b+1)*batch_size]
            if len(ie) == 0 or len(io) == 0: continue
            loss = compute_loss(h_net, even_tr_t[ie], odd_tr_t[io],
                                a_etr_t[ie], a_otr_t[io],
                                KK_etr_t[ie], KK_otr_t[io],
                                mc_t, w_mc_t, G_t)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(h_net.parameters(), 1.0)
            optimizer.step()

        h_net.eval()
        with torch.no_grad():
            vl = compute_loss(h_net, even_val_t, odd_val_t,
                              a_evl_t, a_ovl_t, KK_evl_t, KK_ovl_t,
                              mc_t, w_mc_t, G_t).item()
        scheduler.step(vl)
        if vl < best_val:
            best_val = vl
            best_state = {k: v.cpu().clone() for k, v in h_net.state_dict().items()}
            pat_ctr = 0
        else:
            pat_ctr += 1
        if pat_ctr >= patience:
            break

    if best_state is not None:
        h_net.load_state_dict(best_state)
        h_net = h_net.to(device)
    return best_val

# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════
print("Loading flavor flow (seed 0)...")
flow = create_flow(num_flows=12, hidden_features=128, num_bins=24, device=str(device))
flow.load_state_dict(torch.load(os.path.join(ROOT, 'flavor_SDP_sym_2e6/trial_seed0.pth'),
                                map_location=device, weights_only=False))
flow = flow.to(device).eval()
for p in flow.parameters():
    p.requires_grad = False

# Truth grid
N_check = 80
m_c = np.linspace(0.02, 0.98, N_check)
t_c = np.linspace(0.02, 0.98, N_check)
mm_c, tt_c = np.meshgrid(m_c, t_c)
check_pts = np.column_stack([mm_c.ravel(), tt_c.ravel()]).astype(np.float32)
check_pts_t = torch.from_numpy(check_pts).to(device)

dkpp = DKpp()
dp_check = sdp_to_dp(check_pts)
s23_c, s12_c = dp_check[:, 0], dp_check[:, 1]
s13_c = S_TOTAL - s23_c - s12_c
A12_c = dkpp.full(np.column_stack([s12_c, s13_c]))
A13_c = dkpp.full(np.column_stack([s13_c, s12_c]))
cos_dd_check = np.real(A12_c * np.conj(A13_c)) / (np.abs(A12_c)*np.abs(A13_c) + 1e-30)

# Verify truth symmetry
cos_dd_sym = cos_dd_check.reshape(N_check, N_check)
print(f"Truth symmetry check: max|h(m',θ') - h(m',1-θ')| = "
      f"{np.max(np.abs(cos_dd_sym - cos_dd_sym[::-1, :])):.2e}")

# Load CP data (seed 0, subsample to 100k for speed)
N_CP = 100_000
rng = np.random.RandomState(42)
even_full = np.load(os.path.join(ROOT, 'data/even_symmetric_datasets_sdp/dataset_000.npy'))
odd_full  = np.load(os.path.join(ROOT, 'data/odd_symmetric_datasets_sdp/dataset_000.npy'))
even_sdp = even_full[rng.choice(len(even_full), N_CP, replace=False)].astype(np.float32)
odd_sdp  = odd_full[rng.choice(len(odd_full),  N_CP, replace=False)].astype(np.float32)
print(f"CP data: {N_CP} per tag")

# Train/val split
even_tr, even_vl = train_test_split(even_sdp, test_size=0.1, random_state=42)
odd_tr,  odd_vl  = train_test_split(odd_sdp,  test_size=0.1, random_state=42)

# Precompute flavor terms
print("Precomputing flavor terms...")
t0 = time.time()
K_etr, Kb_etr, a_etr, _ = precompute_flavor_terms(even_tr, flow)
K_otr, Kb_otr, a_otr, _ = precompute_flavor_terms(odd_tr,  flow)
K_evl, Kb_evl, a_evl, _ = precompute_flavor_terms(even_vl, flow)
K_ovl, Kb_ovl, a_ovl, _ = precompute_flavor_terms(odd_vl,  flow)

N_MC = 50_000
np.random.seed(123)
mc_grid = (np.random.rand(N_MC, 2) * (1 - 2e-6) + 1e-6).astype(np.float32)
K_mc, Kb_mc, a_mc, J_mc = precompute_flavor_terms(mc_grid, flow)
invJ_mc = 1.0 / J_mc
w_mc = invJ_mc * a_mc
Gamma_KKbar = np.mean(invJ_mc * (K_mc + Kb_mc))
print(f"  Done ({time.time()-t0:.0f}s), Gamma(K+Kbar) = {Gamma_KKbar:.4f}")

# Tensors
def _t(x): return torch.from_numpy(x.astype(np.float32)).to(device)
even_tr_t, odd_tr_t     = _t(even_tr), _t(odd_tr)
even_vl_t, odd_vl_t     = _t(even_vl), _t(odd_vl)
a_etr_t, a_otr_t        = _t(a_etr), _t(a_otr)
a_evl_t, a_ovl_t        = _t(a_evl), _t(a_ovl)
KK_etr_t = _t(K_etr + Kb_etr); KK_otr_t = _t(K_otr + Kb_otr)
KK_evl_t = _t(K_evl + Kb_evl); KK_ovl_t = _t(K_ovl + Kb_ovl)
mc_t = _t(mc_grid); w_mc_t = _t(w_mc)
G_t = torch.tensor(Gamma_KKbar, dtype=torch.float32, device=device)

# ── Hyperparameter grid ──────────────────────────────────────────────────
configs = [
    # (label, symmetric, hidden, layers, omega_0, lr, epochs)
    ('SIREN 256x5 w=15',          False, 256, 5, 15.0, 1e-4, 30),
    ('Sym-SIREN 256x5 w=15',      True,  256, 5, 15.0, 1e-4, 30),
    ('SIREN 256x5 w=20',          False, 256, 5, 20.0, 1e-4, 30),
    ('Sym-SIREN 256x5 w=20',      True,  256, 5, 20.0, 1e-4, 30),
    ('SIREN 256x5 w=30',          False, 256, 5, 30.0, 1e-4, 30),
    ('Sym-SIREN 256x5 w=30',      True,  256, 5, 30.0, 1e-4, 30),
    ('SIREN 128x4 w=15',          False, 128, 4, 15.0, 1e-4, 30),
    ('Sym-SIREN 128x4 w=15',      True,  128, 4, 15.0, 1e-4, 30),
    ('SIREN 320x6 w=15',          False, 320, 6, 15.0, 1e-4, 30),
    ('Sym-SIREN 320x6 w=15',      True,  320, 6, 15.0, 1e-4, 30),
    ('Sym-SIREN 256x5 w=10',      True,  256, 5, 10.0, 1e-4, 30),
    ('Sym-SIREN 256x5 w=40',      True,  256, 5, 40.0, 1e-4, 30),
]

results = []
print(f"\n{'='*70}")
print(f"TUNING: {len(configs)} configurations, {N_CP//1000}k data, seed 0")
print(f"{'='*70}\n")

for label, symmetric, hidden, layers, omega, lr, epochs in configs:
    torch.manual_seed(42)
    if symmetric:
        h_net = SymmetricSIREN(hidden, layers, omega).to(device)
    else:
        h_net = HNetworkSIREN(hidden, layers, omega).to(device)
    n_params = sum(p.numel() for p in h_net.parameters() if p.requires_grad)

    t0 = time.time()
    best_val = train(h_net, even_tr_t, odd_tr_t, even_vl_t, odd_vl_t,
                     a_etr_t, a_otr_t, a_evl_t, a_ovl_t,
                     KK_etr_t, KK_otr_t, KK_evl_t, KK_ovl_t,
                     mc_t, w_mc_t, G_t, lr=lr, epochs=epochs)
    train_s = time.time() - t0

    h_net.eval()
    with torch.no_grad():
        h_pred = h_net(check_pts_t).cpu().numpy()
    rmse = np.sqrt(np.mean((h_pred - cos_dd_check)**2))

    # Check symmetry violation
    h_2d = h_pred.reshape(N_check, N_check)
    sym_viol = np.max(np.abs(h_2d - h_2d[::-1, :]))

    r = {'label': label, 'symmetric': symmetric, 'hidden': hidden,
         'layers': layers, 'omega': omega, 'n_params': n_params,
         'rmse': float(rmse), 'val_loss': float(best_val),
         'sym_violation': float(sym_viol), 'train_s': float(train_s)}
    results.append(r)

    tag = 'SYM' if symmetric else '   '
    print(f"  [{tag}] {label:30s} | RMSE={rmse:.4f} | val={best_val:.4f} | "
          f"sym_viol={sym_viol:.4f} | {train_s:.0f}s | {n_params:,} params")
    sys.stdout.flush()

# Save results
results_path = os.path.join(SCRIPT_DIR, 'tuning_results.json')
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved {results_path}")

# ── Summary plot ──────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

labels = [r['label'] for r in results]
rmses  = [r['rmse'] for r in results]
vals   = [r['val_loss'] for r in results]
syms   = [r['symmetric'] for r in results]
colors = ['C1' if s else 'C0' for s in syms]

y_pos = np.arange(len(results))
ax1.barh(y_pos, rmses, color=colors, alpha=0.7)
ax1.set_yticks(y_pos)
ax1.set_yticklabels(labels, fontsize=9)
ax1.set_xlabel('RMSE vs truth')
ax1.set_title('RMSE (lower is better)')
ax1.invert_yaxis()
ax1.grid(True, alpha=0.3, axis='x')

ax2.barh(y_pos, vals, color=colors, alpha=0.7)
ax2.set_yticks(y_pos)
ax2.set_yticklabels(labels, fontsize=9)
ax2.set_xlabel('Validation loss')
ax2.set_title('Val loss (lower is better)')
ax2.invert_yaxis()
ax2.grid(True, alpha=0.3, axis='x')

# Legend
from matplotlib.patches import Patch
ax1.legend([Patch(color='C0', alpha=0.7), Patch(color='C1', alpha=0.7)],
           ['Standard', 'Symmetric'], loc='lower right')

plt.suptitle(f'h-network tuning: symmetric vs standard SIREN\n'
             f'(seed 0, {N_CP//1000}k CP data, {configs[0][-1]} epochs)',
             fontsize=13)
plt.tight_layout()
plot_path = os.path.join(SCRIPT_DIR, 'tuning_results.pdf')
plt.savefig(plot_path, bbox_inches='tight')
plt.close()
print(f"Saved {plot_path}")

print("\nDone!")
