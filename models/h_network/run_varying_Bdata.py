#!/usr/bin/env python3
"""
Fit h-networks on multiple B± pseudo-data realizations.

For each B± dataset, runs all available h-networks and saves per-seed
results (not just means), so new seeds can be appended without rerunning.

Usage:
    python run_varying_Bdata.py --bdata-dir /path/to/varying_Bdata_results
    python run_varying_Bdata.py --bdata-dir /path/to/varying_Bdata_results --resume
"""
import sys, os, time, json, argparse, warnings
import numpy as np
warnings.filterwarnings("ignore", message="Inputs to the softmax")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(SCRIPT_DIR, '..')
ROOT = os.path.join(MODELS_DIR, '..')
sys.path.insert(0, ROOT)

import torch
from iminuit import Minuit
from Amplitude import SquareDalitzPlot2
from DKpp import DKpp
from models.h_network.models import create_flow, SymmetricSIREN

parser = argparse.ArgumentParser()
parser.add_argument('--bdata-dir', type=str, required=True,
                    help='Directory containing Bdata_*.npz files')
parser.add_argument('--resume', action='store_true',
                    help='Skip (bdata, seed) pairs already in results')
parser.add_argument('--n-hnet', type=int, default=None,
                    help='Number of h-networks to use (default: all available)')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

M_D0, m_KS, m_pip, m_pim = 1.86484, 0.497611, 0.13957, 0.13957
sdp_obj = SquareDalitzPlot2(M_D0, m_KS, m_pip, m_pim)
IDX = (2, 3, 1)
S_TOTAL = M_D0**2 + m_KS**2 + 2*m_pip**2
FLOW_CONFIG = dict(num_flows=12, hidden_features=128, num_bins=24)

def sdp_to_dp(pts):
    out = np.empty_like(pts, dtype=float)
    for n, (mp, th) in enumerate(pts):
        out[n,0], out[n,1] = sdp_obj.M_from_MpT(mp, th, *IDX)
    return out

def eval_lp(pts, flow):
    pts_t = torch.from_numpy(np.ascontiguousarray(pts)).float().to(device)
    lps = []
    with torch.no_grad():
        for i in range(0, len(pts_t), 50000):
            lps.append(flow.log_prob(pts_t[i:i+50000]).cpu())
    return torch.cat(lps).numpy()

def _fp(x): return np.maximum(np.where(np.isfinite(np.asarray(x)), x, 0.0), 1e-14)

def load_hnet(seed):
    path = os.path.join(SCRIPT_DIR, f'weights/h_ensemble_sym_seed{seed}.pth')
    sd = torch.load(path, map_location=device, weights_only=False)
    n_layers = len([k for k in sd.keys() if 'layers_list' in k and 'weight' in k])
    hidden = sd['net.layers_list.0.linear.weight'].shape[0]
    h = SymmetricSIREN(hidden=hidden, layers=n_layers+1, omega_0=15.0).to(device)
    h.load_state_dict(sd)
    h.eval()
    return h

# Discover available h-networks
available_seeds = sorted([
    int(f.replace('h_ensemble_sym_seed', '').replace('.pth', ''))
    for f in os.listdir(os.path.join(SCRIPT_DIR, 'weights'))
    if f.startswith('h_ensemble_sym_seed') and f.endswith('.pth')
])
n_hnet = args.n_hnet or len(available_seeds)
seeds_to_use = available_seeds[:n_hnet]
print(f"Using {len(seeds_to_use)} h-networks: seeds {seeds_to_use[0]}-{seeds_to_use[-1]}")

# Discover B± datasets with event data
bdata_files = sorted([
    f for f in os.listdir(args.bdata_dir)
    if f.startswith('Bdata_') and f.endswith('.npz')
])
bdata_with_events = []
for f in bdata_files:
    d = np.load(os.path.join(args.bdata_dir, f))
    if 'dataM_sdp' in d.keys():
        idx = int(f.replace('Bdata_', '').replace('.npz', ''))
        bdata_with_events.append(idx)
print(f"Found {len(bdata_with_events)} B± datasets with event data: {bdata_with_events}")

# Load existing results if resuming
results_path = os.path.join(SCRIPT_DIR, 'results', 'varying_Bdata_per_seed.json')
all_results = []
done_pairs = set()
if args.resume and os.path.exists(results_path):
    all_results = json.load(open(results_path))
    done_pairs = {(r['bdata_idx'], r['seed']) for r in all_results}
    print(f"Resuming: {len(done_pairs)} (bdata, seed) pairs already done")

# Preload models
print("Loading h-networks...", end=' '); sys.stdout.flush()
h_nets = {s: load_hnet(s) for s in seeds_to_use}
print("done")

print("Loading flavor flows...", end=' '); sys.stdout.flush()
flows = {}
for s in seeds_to_use:
    f = create_flow(**FLOW_CONFIG, device=str(device))
    f.load_state_dict(torch.load(
        os.path.join(ROOT, f'models/flavor_flow/weights/trial_seed{s}.pth'),
        map_location=device, weights_only=False))
    f.eval()
    for p in f.parameters(): p.requires_grad = False
    flows[s] = f
print("done")

dkpp = DKpp()

for bdata_idx in bdata_with_events:
    bdata = np.load(os.path.join(args.bdata_dir, f'Bdata_{bdata_idx:03d}.npz'))
    dataM = bdata['dataM_sdp'].astype(np.float32)
    dataP = bdata['dataP_sdp'].astype(np.float32)
    mcM = bdata['mcM_sdp'].astype(np.float32)
    mcP = bdata['mcP_sdp'].astype(np.float32)
    all_pts = [dataM, dataP, mcM, mcP]

    # Precompute DKpp signs once per B± dataset
    signs = []
    for pts in all_pts:
        dp = sdp_to_dp(pts)
        s23, s12 = dp[:,0], dp[:,1]
        s13 = S_TOTAL - s23 - s12
        A12 = dkpp.full(np.column_stack([s12, s13]))
        A13 = dkpp.full(np.column_stack([s13, s12]))
        dphi = (np.angle(A12) - np.angle(A13) + np.pi) % (2*np.pi) - np.pi
        signs.append(np.sign(np.sin(dphi)))

    print(f"\nBdata_{bdata_idx:03d}: ", end=''); sys.stdout.flush()

    for seed in seeds_to_use:
        if (bdata_idx, seed) in done_pairs:
            print('.', end=''); sys.stdout.flush()
            continue

        flow = flows[seed]
        h_net = h_nets[seed]

        terms = []
        for k, pts in enumerate(all_pts):
            logp = eval_lp(pts, flow)
            pF = np.exp(np.clip(logp, -50, 50))
            swap = pts.copy(); swap[:,1] = 1.0 - swap[:,1]
            pFsw = np.exp(np.clip(eval_lp(swap, flow), -50, 50))
            abJ = np.sqrt(np.maximum(pF * pFsw, 0.0))
            with torch.no_grad():
                h_v = h_net(torch.from_numpy(pts).to(device)).cpu().numpy()
            C = abJ * h_v
            absS = abJ * np.sqrt(np.maximum(1.0 - h_v**2, 0.0))
            S = signs[k] * absS
            terms.append(dict(pF=_fp(pF), pFsw=_fp(pFsw),
                              C=np.where(np.isfinite(C), C, 0.0),
                              S=np.where(np.isfinite(S), S, 0.0)))

        tM, tP, mM, mP = terms
        N_M, N_P = len(tM['pF']), len(tP['pF'])

        def nll(rB, delta, gamma):
            thM, thP = np.radians(delta-gamma), np.radians(delta+gamma)
            pBm = _fp(tM['pFsw']+rB**2*tM['pF']+2*rB*(np.cos(thM)*tM['C']-np.sin(thM)*tM['S']))
            pBp = _fp(tP['pF']+rB**2*tP['pFsw']+2*rB*(np.cos(thP)*tP['C']+np.sin(thP)*tP['S']))
            mBm = _fp(mM['pFsw']+rB**2*mM['pF']+2*rB*(np.cos(thM)*mM['C']-np.sin(thM)*mM['S']))
            mBp = _fp(mP['pF']+rB**2*mP['pFsw']+2*rB*(np.cos(thP)*mP['C']+np.sin(thP)*mP['S']))
            return (-np.log(pBm).sum()+N_M*np.log(mBm.mean())-np.log(pBp).sum()+N_P*np.log(mBp.mean()))

        m = Minuit(nll, rB=0.02, delta=160, gamma=100)
        m.limits['rB']=(0,1); m.limits['delta']=(0,360); m.limits['gamma']=(0,360)
        m.errors['rB'],m.errors['delta'],m.errors['gamma'] = 0.01, 2.0, 2.0
        m.errordef = Minuit.LIKELIHOOD; m.migrad(); m.hesse()

        all_results.append({
            'bdata_idx': bdata_idx,
            'seed': seed,
            'rB': float(m.values['rB']), 'rB_err': float(m.errors['rB']),
            'delta': float(m.values['delta']), 'delta_err': float(m.errors['delta']),
            'gamma': float(m.values['gamma']), 'gamma_err': float(m.errors['gamma']),
            'valid': m.valid,
        })
        print('+', end=''); sys.stdout.flush()

    # Save incrementally after each B± dataset
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)

print(f"\n\nSaved {len(all_results)} per-seed results to {results_path}")

# ----- Aggregate per-realization, then across realizations -----
bdata_indices = sorted(set(r['bdata_idx'] for r in all_results))

realiz = {}
for bi in bdata_indices:
    seeds_done = [r for r in all_results if r['bdata_idx'] == bi and r.get('valid', True)]
    if not seeds_done:
        continue
    realiz[bi] = {
        'n_seeds': len(seeds_done),
        'rB_mean':       np.mean([r['rB']        for r in seeds_done]),
        'rB_meanerr':    np.mean([r['rB_err']    for r in seeds_done]),
        'delta_mean':    np.mean([r['delta']     for r in seeds_done]),
        'delta_meanerr': np.mean([r['delta_err'] for r in seeds_done]),
        'gamma_mean':    np.mean([r['gamma']     for r in seeds_done]),
        'gamma_meanerr': np.mean([r['gamma_err'] for r in seeds_done]),
    }

print(f"\nPer-realization (mean over seeds  ±  ⟨HESSE⟩ over seeds):")
for bi in bdata_indices:
    r = realiz.get(bi)
    if r is None: continue
    print(f"  Bdata_{bi:03d}: {r['n_seeds']:2d} seeds  "
          f"rB={r['rB_mean']:.4f}±{r['rB_meanerr']:.4f}  "
          f"δ={r['delta_mean']:6.2f}±{r['delta_meanerr']:.2f}  "
          f"γ={r['gamma_mean']:6.2f}±{r['gamma_meanerr']:.2f}")

# Across-realization σ_total = √(realization-spread² + ⟨per-realization fit-error⟩²)
print(f"\nAcross {len(realiz)} realizations  (σ_total = √(σ_realization² + ⟨fit_err⟩²)):")
for name, fmt in [('rB', '.4f'), ('delta', '.3f'), ('gamma', '.3f')]:
    means = np.array([realiz[bi][f'{name}_mean']    for bi in realiz])
    errs  = np.array([realiz[bi][f'{name}_meanerr'] for bi in realiz])
    spread, mean_err = means.std(ddof=0), errs.mean()
    sig_tot = np.sqrt(spread**2 + mean_err**2)
    print(f"  {name:>5s}: mean={means.mean():{fmt}}  "
          f"σ_realization={spread:{fmt}}  ⟨fit_err⟩={mean_err:{fmt}}  "
          f"σ_total={sig_tot:{fmt}}")

# ----- Verification: cross-check 2 realizations vs varying_Bdata_hnet.json -----
ref_path = os.path.join(SCRIPT_DIR, 'results', 'varying_Bdata_hnet.json')
if os.path.exists(ref_path):
    ref = {r['bdata_idx']: r for r in json.load(open(ref_path))}
    common = sorted(set(realiz) & set(ref.keys()))
    if len(common) >= 2:
        print(f"\nVerification vs varying_Bdata_hnet.json (h_mean_*  on first 2 common):")
        for bi in common[:2]:
            r_new, r_ref = realiz[bi], ref[bi]
            print(f"  Bdata_{bi:03d}:")
            for name in ['rB', 'delta', 'gamma']:
                new_v = r_new[f'{name}_mean']
                ref_v = r_ref[f'h_mean_{name}']
                print(f"    {name:>5s}: new={new_v:.4f}  ref={ref_v:.4f}  Δ={new_v-ref_v:+.4f}")
