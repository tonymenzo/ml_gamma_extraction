#!/usr/bin/env python3
"""
Ensemble h-network comparison with 3-flow method.

For each flavor flow seed, trains one SIREN h-network on CP-tagged data
using the symmetric SDP convention idx=(2,3,1), evaluates the gamma fit
on shared B± pseudo-data, and saves results.

Serial design: train + evaluate one network at a time, so generalises
naturally to any number of trials.

Usage:
    python run_ensemble_comparison.py [--n-trials 5] [--n-cp 500000]
"""
import sys, os, time, json, argparse, warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", message="Inputs to the softmax are not scaled down")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(SCRIPT_DIR, '..')
ROOT = os.path.join(MODELS_DIR, '..')
sys.path.insert(0, ROOT)

import torch
from torch import nn
from sklearn.model_selection import train_test_split
from iminuit import Minuit

from Amplitude import SquareDalitzPlot2
from DKpp import DKpp
from models.h_network.models import (FlowMLP, create_flow, SirenLayer,
                                      HNetworkSIREN, SymmetricSIREN)

# ── CLI ───────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--n-trials', type=int, default=5)
parser.add_argument('--n-cp',     type=int, default=None,
                    help='Subsample CP events per tag (default: use all 2M)')
parser.add_argument('--n-mc',     type=int, default=250_000,
                    help='MC grid size for normalisation integral')
parser.add_argument('--epochs',   type=int, default=100)
parser.add_argument('--resume',   action='store_true',
                    help='Skip seeds whose results already exist')
parser.add_argument('--symmetric', action='store_true',
                    help='Use SymmetricSIREN (enforces theta->1-theta symmetry)')
parser.add_argument('--hidden',   type=int, default=None,
                    help='Override hidden size')
parser.add_argument('--layers',   type=int, default=None,
                    help='Override number of layers')
parser.add_argument('--omega',    type=float, default=None,
                    help='Override omega_0')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
sys.stdout.flush()

# ── Physics constants ─────────────────────────────────────────────────────
M_D0, m_KS, m_pip, m_pim = 1.86484, 0.497611, 0.13957, 0.13957
sdp_obj = SquareDalitzPlot2(M_D0, m_KS, m_pip, m_pim)
IDX = (2, 3, 1)          # symmetric SDP: π+π− resonant pair
S_TOTAL = M_D0**2 + m_KS**2 + 2 * m_pip**2

true_rB        = 0.1
true_deltaB_deg = 130.0
true_gamma_deg  = 70.0

# Flow and h-network architectures imported from h_network.models

FLOW_CONFIG = dict(num_flows=12, hidden_features=128, num_bins=24)

def load_flavor_flow(seed):
    path = os.path.join(ROOT, f'models/flavor_flow/weights/trial_seed{seed}.pth')
    flow = create_flow(**FLOW_CONFIG, device=str(device))
    flow.load_state_dict(torch.load(path, map_location=device, weights_only=False))
    flow = flow.to(device).eval()
    for p in flow.parameters():
        p.requires_grad = False
    return flow

# ── Coordinate helpers (symmetric SDP) ────────────────────────────────────
def sdp_to_dp(points_sdp, sdp_obj=sdp_obj, idx=IDX):
    """(m',θ') → (s_ij, s_ik).  With idx=(2,3,1): returns (s23, s12)."""
    i, j, k = idx
    out = np.empty_like(points_sdp, dtype=float)
    for n, (mp, th) in enumerate(points_sdp):
        out[n, 0], out[n, 1] = sdp_obj.M_from_MpT(mp, th, i, j, k)
    return out

def dp_to_sdp(points_dp, sdp_obj=sdp_obj, idx=IDX):
    """(s_ij, s_ik) → (m',θ')."""
    i, j, k = idx
    sij, sik = points_dp[:, 0], points_dp[:, 1]
    mp = np.vectorize(
        lambda a, b: sdp_obj.MpfromM(a, b, i, j, k), otypes=[float])(sij, sik)
    tp = np.vectorize(
        lambda a, b: sdp_obj.TfromM(a, b, i, j, k), otypes=[float])(sij, sik)
    return np.column_stack([mp, tp])

def eval_flow_log_prob(points_sdp, flow, batch_size=50_000):
    pts = torch.from_numpy(np.ascontiguousarray(points_sdp)).float().to(device)
    lps = []
    with torch.no_grad():
        for i in range(0, len(pts), batch_size):
            lps.append(flow.log_prob(pts[i:i+batch_size]).cpu())
    return torch.cat(lps).numpy()

def precompute_flavor_terms(points_sdp, flow):
    """Return K, Kbar, alpha at each SDP point using symmetric convention.

    With idx=(2,3,1) and m_π+=m_π−, the swap is θ'→1−θ' and J is symmetric.
    """
    N = len(points_sdp)
    dp = sdp_to_dp(points_sdp)
    s_ij = dp[:, 0]     # s23
    s_ik = dp[:, 1]     # s12

    # Jacobian (same for swapped point by symmetry)
    J = np.empty(N)
    for n in range(N):
        J[n] = max(float(sdp_obj.jacobian(s_ij[n], s_ik[n], *IDX)), 1e-30)

    # K  = p_flow(m', θ') · J
    logp = eval_flow_log_prob(points_sdp, flow)
    K = np.exp(np.clip(logp, -50, 50)) * J

    # Kbar = p_flow(m', 1−θ') · J   (same J by symmetry)
    swap = points_sdp.copy()
    swap[:, 1] = 1.0 - swap[:, 1]
    logp_sw = eval_flow_log_prob(swap, flow)
    Kbar = np.exp(np.clip(logp_sw, -50, 50)) * J

    alpha = np.sqrt(np.maximum(K * Kbar, 0.0))
    return K, Kbar, alpha, J


# ── h-network training hyperparameters ────────────────────────────────────
H_HIDDEN   = args.hidden or (320 if args.symmetric else 256)
H_LAYERS   = args.layers or (6 if args.symmetric else 5)
H_OMEGA    = args.omega or 15.0

def load_hnet(path):
    """Load a trained h-network, auto-detecting architecture from checkpoint."""
    sd = torch.load(path, map_location=device, weights_only=False)
    n_hidden = len([k for k in sd.keys() if 'layers_list' in k and 'weight' in k])
    hidden = sd['net.layers_list.0.linear.weight'].shape[0]
    h = SymmetricSIREN(hidden=hidden, layers=n_hidden+1, omega_0=H_OMEGA).to(device)
    h.load_state_dict(sd)
    h.eval()
    return h
LR         = 1e-4
BATCH_SIZE = 20_000
PATIENCE   = 30
EPS        = 1e-8

# ── Loss function ─────────────────────────────────────────────────────────
def compute_loss(h_net, even_pts, odd_pts, alpha_even, alpha_odd,
                 KpKbar_even, KpKbar_odd, mc_pts, w_mc, Gamma_KKbar):
    h_even = h_net(even_pts)
    h_odd  = h_net(odd_pts)
    h_mc   = h_net(mc_pts)
    I_h = torch.mean(w_mc * h_mc)
    Gamma_plus  = torch.clamp(Gamma_KKbar + 2.0 * I_h, min=EPS)
    Gamma_minus = torch.clamp(Gamma_KKbar - 2.0 * I_h, min=EPS)
    p_plus_raw  = KpKbar_even + 2.0 * alpha_even * h_even
    p_minus_raw = KpKbar_odd  - 2.0 * alpha_odd  * h_odd
    p_plus  = torch.clamp(p_plus_raw, min=EPS)
    p_minus = torch.clamp(p_minus_raw, min=EPS)
    nll = (-torch.mean(torch.log(p_plus)  - torch.log(Gamma_plus))
           -torch.mean(torch.log(p_minus) - torch.log(Gamma_minus)))
    with torch.no_grad():
        n_clamp = ((p_plus_raw < EPS).sum() + (p_minus_raw < EPS).sum()).item()
        n_total = len(p_plus_raw) + len(p_minus_raw)
        clamp_frac = n_clamp / n_total
    return nll, {'I_h': I_h.item(), 'clamp_frac': clamp_frac}

# ── Interference terms from h-network (for gamma fit) ────────────────────
def _finite_pos(x):
    x = np.asarray(x)
    return np.maximum(np.where(np.isfinite(x), x, 0.0), 1e-14)

def interference_from_h(points_sdp, flow, h_net):
    """Compute pF, pFsw, C, S, abJ, J (inverse) at each point."""
    N = len(points_sdp)
    dp = sdp_to_dp(points_sdp)
    s23, s12 = dp[:, 0], dp[:, 1]
    s13 = S_TOTAL - s23 - s12

    # Flavor density at (m', θ') and (m', 1−θ')
    logp = eval_flow_log_prob(points_sdp, flow)
    swap = points_sdp.copy(); swap[:, 1] = 1.0 - swap[:, 1]
    logp_sw = eval_flow_log_prob(swap, flow)

    J = np.empty(N)
    for n in range(N):
        J[n] = max(float(sdp_obj.jacobian(s23[n], s12[n], *IDX)), 1e-30)

    pF   = np.exp(np.clip(logp,    -50, 50)) * J      # K
    pFsw = np.exp(np.clip(logp_sw, -50, 50)) * J      # Kbar
    abJ  = np.sqrt(np.maximum(pF * pFsw, 0.0))        # alpha

    # h-network evaluation
    pts_t = torch.from_numpy(points_sdp.astype(np.float32)).to(device)
    h_net.eval()
    with torch.no_grad():
        h_v = h_net(pts_t).cpu().numpy()
    C    = abJ * h_v
    absS = abJ * np.sqrt(np.maximum(1.0 - h_v**2, 0.0))

    # Sign of S from isobar model
    dkpp = DKpp()
    A12 = dkpp.full(np.column_stack([s12, s13]))
    A13 = dkpp.full(np.column_stack([s13, s12]))
    dphi = (np.angle(A12) - np.angle(A13) + np.pi) % (2*np.pi) - np.pi
    S = np.sign(np.sin(dphi)) * absS

    invJ = 1.0 / J
    return dict(pF=_finite_pos(pF), pFsw=_finite_pos(pFsw),
                C=np.where(np.isfinite(C), C, 0.0),
                S=np.where(np.isfinite(S), S, 0.0),
                abJ=abJ, J=_finite_pos(invJ))

# ── Gamma fit ─────────────────────────────────────────────────────────────
def run_gamma_fit(h_net, flow, dataM_sdp, dataP_sdp, mcM_sdp, mcP_sdp):
    tM = interference_from_h(dataM_sdp, flow, h_net)
    tP = interference_from_h(dataP_sdp, flow, h_net)
    mM = interference_from_h(mcM_sdp,   flow, h_net)
    mP = interference_from_h(mcP_sdp,   flow, h_net)

    N_M, N_P = len(tM['pF']), len(tP['pF'])

    def nll(rB, delta, gamma):
        thM = np.radians(delta - gamma)
        thP = np.radians(delta + gamma)
        pBm = _finite_pos(tM['pFsw'] + rB**2*tM['pF']
                          + 2*rB*(np.cos(thM)*tM['C'] - np.sin(thM)*tM['S']))
        pBp = _finite_pos(tP['pF'] + rB**2*tP['pFsw']
                          + 2*rB*(np.cos(thP)*tP['C'] + np.sin(thP)*tP['S']))
        # MC normalization in SDP measure (no Jacobian — K already includes it)
        mBm = _finite_pos(mM['pFsw'] + rB**2*mM['pF']
                          + 2*rB*(np.cos(thM)*mM['C'] - np.sin(thM)*mM['S']))
        mBp = _finite_pos(mP['pF'] + rB**2*mP['pFsw']
                          + 2*rB*(np.cos(thP)*mP['C'] + np.sin(thP)*mP['S']))
        return (-np.log(pBm).sum() + N_M*np.log(mBm.mean())
                -np.log(pBp).sum() + N_P*np.log(mBp.mean()))

    m = Minuit(nll, rB=true_rB*0.2,
               delta=true_deltaB_deg + 30, gamma=true_gamma_deg + 30)
    m.limits['rB']    = (0, 1)
    m.limits['delta']  = (0, 360)
    m.limits['gamma']  = (0, 360)
    m.errors['rB'], m.errors['delta'], m.errors['gamma'] = 0.01, 2.0, 2.0
    m.errordef = Minuit.LIKELIHOOD
    m.migrad()
    m.hesse()
    return {k: float(m.values[k]) for k in ('rB','delta','gamma')}, \
           {k: float(m.errors[k]) for k in ('rB','delta','gamma')}, \
           m.valid, m.accurate

# ── Load pre-computed CP data in symmetric SDP ───────────────────────────
def load_cp_data(seed, n_cp=None):
    """Load pre-computed CP-even/odd data for given seed from data/ directory.

    These datasets (2M events each, idx=(2,3,1) SDP) are the same ones used
    to train the collaborator's CP flows, ensuring an apples-to-apples comparison.
    If n_cp is given, subsample to that size.
    """
    even_path = os.path.join(ROOT, f'data/cp_tagged/even_symmetric_datasets_sdp/dataset_{seed:03d}.npy')
    odd_path  = os.path.join(ROOT, f'data/cp_tagged/odd_symmetric_datasets_sdp/dataset_{seed:03d}.npy')
    even_sdp = np.load(even_path).astype(np.float32)
    odd_sdp  = np.load(odd_path).astype(np.float32)
    if n_cp is not None and n_cp < len(even_sdp):
        rng = np.random.RandomState(42)
        even_sdp = even_sdp[rng.choice(len(even_sdp), n_cp, replace=False)]
        odd_sdp  = odd_sdp[rng.choice(len(odd_sdp),  n_cp, replace=False)]
    return even_sdp, odd_sdp

# ── Train one h-network ──────────────────────────────────────────────────
def train_one(seed, even_sdp, odd_sdp, mc_grid, n_epochs):
    """Load flavor flow `seed`, train h-network, run gamma fit. Returns dict."""
    print(f"\n{'='*70}")
    print(f"TRIAL seed={seed}")
    print(f"{'='*70}")
    sys.stdout.flush()

    flow = load_flavor_flow(seed)
    n_flow_params = sum(p.numel() for p in flow.parameters())
    print(f"  Flavor flow loaded ({n_flow_params:,} params)")

    # Train/val split
    even_tr, even_val = train_test_split(even_sdp, test_size=0.1, random_state=42)
    odd_tr,  odd_val  = train_test_split(odd_sdp,  test_size=0.1, random_state=42)
    print(f"  Train: {len(even_tr)} even, {len(odd_tr)} odd")

    # Precompute flavor terms
    t0 = time.time()
    K_etr, Kb_etr, a_etr, _ = precompute_flavor_terms(even_tr, flow)
    K_otr, Kb_otr, a_otr, _ = precompute_flavor_terms(odd_tr,  flow)
    K_evl, Kb_evl, a_evl, _ = precompute_flavor_terms(even_val, flow)
    K_ovl, Kb_ovl, a_ovl, _ = precompute_flavor_terms(odd_val,  flow)
    K_mc,  Kb_mc,  a_mc, J_mc = precompute_flavor_terms(mc_grid, flow)
    precomp_s = time.time() - t0
    print(f"  Precomputation: {precomp_s:.0f}s")

    invJ_mc = 1.0 / J_mc
    w_mc = invJ_mc * a_mc
    Gamma_KKbar = np.mean(invJ_mc * (K_mc + Kb_mc))
    print(f"  Gamma(K+Kbar) = {Gamma_KKbar:.6f}")

    # → tensors
    def _t(x): return torch.from_numpy(x.astype(np.float32)).to(device)
    even_tr_t, odd_tr_t     = _t(even_tr), _t(odd_tr)
    even_val_t, odd_val_t   = _t(even_val), _t(odd_val)
    a_etr_t, a_otr_t        = _t(a_etr), _t(a_otr)
    a_evl_t, a_ovl_t        = _t(a_evl), _t(a_ovl)
    KK_etr_t                = _t(K_etr + Kb_etr)
    KK_otr_t                = _t(K_otr + Kb_otr)
    KK_evl_t                = _t(K_evl + Kb_evl)
    KK_ovl_t                = _t(K_ovl + Kb_ovl)
    mc_t                    = _t(mc_grid)
    w_mc_t                  = _t(w_mc)
    G_t = torch.tensor(Gamma_KKbar, dtype=torch.float32, device=device)

    # Fresh h-network
    torch.manual_seed(42)
    if args.symmetric:
        h_net = SymmetricSIREN(H_HIDDEN, H_LAYERS, H_OMEGA).to(device)
    else:
        h_net = HNetworkSIREN(H_HIDDEN, H_LAYERS, H_OMEGA).to(device)
    n_params = sum(p.numel() for p in h_net.parameters() if p.requires_grad)
    sym_tag = 'Sym-' if args.symmetric else ''
    print(f"  h-network: {sym_tag}SIREN {H_HIDDEN}x{H_LAYERS} ({n_params:,} params)")

    optimizer = torch.optim.Adam(h_net.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=15, min_lr=1e-6)

    N_even_tr, N_odd_tr = len(even_tr_t), len(odd_tr_t)
    N_batches = max(N_even_tr, N_odd_tr) // BATCH_SIZE
    best_val, best_state, patience_ctr = float('inf'), None, 0

    t_train = time.time()
    for epoch in range(1, n_epochs + 1):
        h_net.train()
        perm_e = torch.randperm(N_even_tr, device=device)
        perm_o = torch.randperm(N_odd_tr,  device=device)
        epoch_loss, n_steps = 0.0, 0

        for b in range(N_batches):
            ie = perm_e[b*BATCH_SIZE:(b+1)*BATCH_SIZE]
            io = perm_o[b*BATCH_SIZE:(b+1)*BATCH_SIZE]
            if len(ie) == 0 or len(io) == 0:
                continue
            loss, _ = compute_loss(
                h_net, even_tr_t[ie], odd_tr_t[io],
                a_etr_t[ie], a_otr_t[io],
                KK_etr_t[ie], KK_otr_t[io],
                mc_t, w_mc_t, G_t)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(h_net.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_steps += 1

        avg_train = epoch_loss / max(n_steps, 1)

        # Validation
        h_net.eval()
        with torch.no_grad():
            val_loss, val_h = compute_loss(
                h_net, even_val_t, odd_val_t,
                a_evl_t, a_ovl_t, KK_evl_t, KK_ovl_t,
                mc_t, w_mc_t, G_t)
        vl = val_loss.item()
        scheduler.step(vl)

        if vl < best_val:
            best_val = vl
            best_state = {k: v.cpu().clone() for k, v in h_net.state_dict().items()}
            patience_ctr = 0
            marker = ' *'
        else:
            patience_ctr += 1
            marker = ''

        # Abort checks
        if not np.isfinite(avg_train) or avg_train > 10:
            print(f"  *** ABORT: loss diverged ({avg_train:.4f})")
            break
        if val_h['clamp_frac'] > 0.05:
            print(f"  *** ABORT: clamp {val_h['clamp_frac']*100:.1f}%")
            break

        if epoch % 20 == 0 or epoch == 1:
            lr = optimizer.param_groups[0]['lr']
            print(f"  Ep {epoch:3d} | trn {avg_train:.4f} | val {vl:.4f} | "
                  f"I_h {val_h['I_h']:+.4f} | lr {lr:.1e}{marker}")
            sys.stdout.flush()

        if patience_ctr >= PATIENCE:
            print(f"  Early stop at epoch {epoch}")
            break

    train_s = time.time() - t_train
    if best_state is not None:
        h_net.load_state_dict(best_state)
        h_net = h_net.to(device)
    print(f"  Training: {train_s:.0f}s, best val={best_val:.4f}")

    # RMSE on truth grid
    h_net.eval()
    with torch.no_grad():
        h_pred = h_net(check_pts_t).cpu().numpy()
    rmse = np.sqrt(np.mean((h_pred - cos_dd_check)**2))
    print(f"  RMSE vs truth: {rmse:.4f}")

    # Save model
    tag = 'sym_' if args.symmetric else ''
    model_path = os.path.join(SCRIPT_DIR, f'weights/h_ensemble_{tag}seed{seed}.pth')
    torch.save(h_net.state_dict(), model_path)

    # Gamma fit
    print(f"  Running gamma fit...")
    t0 = time.time()
    vals, errs, valid, accurate = run_gamma_fit(
        h_net, flow, dataM_sdp, dataP_sdp, mcM_sdp, mcP_sdp)
    fit_s = time.time() - t0
    print(f"  rB={vals['rB']:.4f}±{errs['rB']:.4f}  "
          f"delta={vals['delta']:.2f}±{errs['delta']:.2f}  "
          f"gamma={vals['gamma']:.2f}±{errs['gamma']:.2f}  ({fit_s:.0f}s)")
    sys.stdout.flush()

    return {
        'seed': seed, 'rmse': float(rmse),
        'val_loss': float(best_val),
        'train_s': float(train_s), 'precomp_s': float(precomp_s),
        'rB': vals['rB'], 'rB_err': errs['rB'],
        'delta': vals['delta'], 'delta_err': errs['delta'],
        'gamma': vals['gamma'], 'gamma_err': errs['gamma'],
        'valid': valid, 'accurate': accurate,
    }

# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════
print("\n=== Ensemble h-network comparison ===")
cp_label = f"{args.n_cp:,}" if args.n_cp else "full (2M)"
sym_label = 'Sym-' if args.symmetric else ''
print(f"  Trials: {args.n_trials}, CP data: {cp_label}, MC: {args.n_mc:,}")
print(f"  Epochs: {args.epochs}, {sym_label}SIREN {H_HIDDEN}x{H_LAYERS} w={H_OMEGA}")

# Truth grid for RMSE (shared)
N_check = 80
m_c = np.linspace(0.02, 0.98, N_check)
t_c = np.linspace(0.02, 0.98, N_check)
mm_c, tt_c = np.meshgrid(m_c, t_c)
check_pts = np.column_stack([mm_c.ravel(), tt_c.ravel()]).astype(np.float32)
check_pts_t = torch.from_numpy(check_pts).to(device)

dkpp_truth = DKpp()
dp_check = sdp_to_dp(check_pts)
s23_c, s12_c = dp_check[:, 0], dp_check[:, 1]
s13_c = S_TOTAL - s23_c - s12_c
A12_c = dkpp_truth.full(np.column_stack([s12_c, s13_c]))
A13_c = dkpp_truth.full(np.column_stack([s13_c, s12_c]))
cos_dd_check = np.real(A12_c * np.conj(A13_c)) / (np.abs(A12_c)*np.abs(A13_c) + 1e-30)
print(f"  Truth grid: {len(check_pts)} points")

# Shared B± pseudo-data
bpm = np.load(os.path.join(ROOT, 'data/b_pseudo/BpBm_samples_rB0.1.npz'))
dataM_sdp = bpm['dataM_sdp'].astype(np.float32)
dataP_sdp = bpm['dataP_sdp'].astype(np.float32)
mcM_sdp   = bpm['mcM_sdp'].astype(np.float32)
mcP_sdp   = bpm['mcP_sdp'].astype(np.float32)
print(f"  B- data: {len(dataM_sdp)}, B+ data: {len(dataP_sdp)}, "
      f"MC: {len(mcM_sdp)}")

# MC grid for normalisation
np.random.seed(123)
eps_mc = 1e-6
mc_grid = (np.random.rand(args.n_mc, 2) * (1 - 2*eps_mc) + eps_mc).astype(np.float32)
print(f"  MC grid: {len(mc_grid)}")
sys.stdout.flush()

# Load existing results if resuming
rtag = 'symmetric_' if args.symmetric else ''
results_path = os.path.join(SCRIPT_DIR, f'results/ensemble_comparison_{rtag}results.json')
all_results = []
done_seeds = set()
if args.resume and os.path.exists(results_path):
    with open(results_path) as f:
        all_results = json.load(f)
    done_seeds = {r['seed'] for r in all_results}
    print(f"  Resuming: {len(done_seeds)} seeds already done")

# Run trials — each seed loads its own pre-computed CP data
for trial_idx in range(args.n_trials):
    seed = trial_idx
    if seed in done_seeds:
        print(f"\n  Skipping seed {seed} (already done)")
        continue

    even_sdp, odd_sdp = load_cp_data(seed, n_cp=args.n_cp)
    print(f"  CP data seed {seed}: {len(even_sdp)} even, {len(odd_sdp)} odd")

    result = train_one(seed, even_sdp, odd_sdp, mc_grid, args.epochs)
    all_results.append(result)

    # Save incrementally
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)

# ═══════════════════════════════════════════════════════════════════════════
# Summary & Comparison Plot
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("ENSEMBLE RESULTS")
print(f"{'='*70}")

h_rB    = np.array([r['rB']    for r in all_results])
h_delta = np.array([r['delta'] for r in all_results])
h_gamma = np.array([r['gamma'] for r in all_results])
h_rB_e  = np.array([r['rB_err']    for r in all_results])
h_delta_e = np.array([r['delta_err'] for r in all_results])
h_gamma_e = np.array([r['gamma_err'] for r in all_results])

print(f"h-network (N={len(all_results)}):")
print(f"  rB:    {h_rB.mean():.4f} ± {h_rB.std():.4f}  (mean err {h_rB_e.mean():.4f})")
print(f"  delta: {h_delta.mean():.2f} ± {h_delta.std():.2f}  (mean err {h_delta_e.mean():.2f})")
print(f"  gamma: {h_gamma.mean():.2f} ± {h_gamma.std():.2f}  (mean err {h_gamma_e.mean():.2f})")

# Load collaborator's 3-flow results
three_flow = np.load(os.path.join(SCRIPT_DIR, 'results/fit_results_symmetric.npz'))
tf_rB    = three_flow['results_rB']
tf_delta = three_flow['results_delta']
tf_gamma = three_flow['results_gamma']
tf_rB_e  = three_flow['errors_rB']
tf_delta_e = three_flow['errors_delta']
tf_gamma_e = three_flow['errors_gamma']

print(f"\n3-flow (N={len(tf_gamma)}):")
print(f"  rB:    {tf_rB.mean():.4f} ± {tf_rB.std():.4f}  (mean err {tf_rB_e.mean():.4f})")
print(f"  delta: {tf_delta.mean():.2f} ± {tf_delta.std():.2f}  (mean err {tf_delta_e.mean():.2f})")
print(f"  gamma: {tf_gamma.mean():.2f} ± {tf_gamma.std():.2f}  (mean err {tf_gamma_e.mean():.2f})")

# ── Comparison figure ─────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
params  = [('rB', true_rB), ('delta', true_deltaB_deg), ('gamma', true_gamma_deg)]
h_arrs  = [h_rB, h_delta, h_gamma]
tf_arrs = [tf_rB, tf_delta, tf_gamma]
h_errs  = [h_rB_e, h_delta_e, h_gamma_e]
tf_errs = [tf_rB_e, tf_delta_e, tf_gamma_e]
labels_p = [r'$r_B$', r'$\delta_B$ (deg)', r'$\gamma$ (deg)']

for ax, (pname, truth), h_v, tf_v, h_e, tf_e, lab in zip(
        axes, params, h_arrs, tf_arrs, h_errs, tf_errs, labels_p):

    # 3-flow results (all 25)
    ax.errorbar(np.arange(len(tf_v)), tf_v, yerr=tf_e,
                fmt='o', color='C0', alpha=0.5, markersize=4, capsize=2,
                label=f'3-flow (N={len(tf_v)})')
    # h-network results
    ax.errorbar(np.arange(len(h_v)) + len(tf_v) + 1, h_v, yerr=h_e,
                fmt='s', color='C1', markersize=5, capsize=2,
                label=f'h-network (N={len(h_v)})')
    ax.axhline(truth, color='k', ls='--', lw=1, label=f'Truth = {truth}')

    # Mean lines
    ax.axhline(tf_v.mean(), color='C0', ls=':', alpha=0.6)
    ax.axhline(h_v.mean(),  color='C1', ls=':', alpha=0.6)

    ax.set_ylabel(lab)
    ax.set_xlabel('Trial index')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

plt.suptitle(r'h-network vs 3-flow: extraction of $\gamma$ '
             f'(rB={true_rB}, $N_{{B\\pm}}$={len(dataM_sdp)//1000}k each)',
             fontsize=13)
plt.tight_layout()
plot_path = os.path.join(SCRIPT_DIR, f'figures/ensemble_comparison_{rtag[:-1] if rtag else "standard"}.pdf')
plt.savefig(plot_path, bbox_inches='tight')
plt.close()
print(f"\nSaved {plot_path}")

# ── Bias summary ──────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print("BIAS COMPARISON")
print(f"{'='*70}")
for pname, truth, h_v, tf_v in zip(
        ['rB', 'delta', 'gamma'], [true_rB, true_deltaB_deg, true_gamma_deg],
        h_arrs, tf_arrs):
    h_bias  = h_v.mean() - truth
    tf_bias = tf_v.mean() - truth
    print(f"  {pname:>6s}:  3-flow bias = {tf_bias:+.4f},  h-network bias = {h_bias:+.4f}")

print("\nDone!")
