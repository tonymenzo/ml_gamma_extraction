#!/usr/bin/env python3
"""
Fit the analytic (isobar) amplitude model on multiple B± pseudo-data realizations.

Mirrors run_varying_Bdata.py but replaces the flow + h-network with the exact
amplitude DKpp.full(...). One fit per realization (no seeds), so the per-realization
HESSE error IS the apples-to-apples analog of <fit_err> for the seeded methods.

Convention: SDP measure (pF, pFsw, C, S have units of SDP density), matching
run_varying_Bdata.py. Achieved by dividing |A|^2 and Re/Im(A12 A13*) by
J = |∂(m', θ')/∂(s_ij, s_ik)| at each point.

If verification (printed at the end) shows the central values disagree with
'exact_*' in varying_Bdata_hnet.json, the lost generator script likely used
DP measure (multiply by J instead) — toggle the line marked DP/SDP.

Usage:
    python run_varying_Bdata_amp.py --bdata-dir /path/to/varying_Bdata_results
    python run_varying_Bdata_amp.py --bdata-dir /path/to/varying_Bdata_results --resume
"""
import sys, os, json, argparse, warnings
import numpy as np
warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(SCRIPT_DIR, '..')
ROOT = os.path.join(MODELS_DIR, '..')
sys.path.insert(0, ROOT)

from iminuit import Minuit
from Amplitude import SquareDalitzPlot2
from DKpp import DKpp

parser = argparse.ArgumentParser()
parser.add_argument('--bdata-dir', type=str, required=True,
                    help='Directory containing Bdata_*.npz files')
parser.add_argument('--resume', action='store_true',
                    help='Skip realizations already in results')
args = parser.parse_args()

M_D0, m_KS, m_pip, m_pim = 1.86484, 0.497611, 0.13957, 0.13957
sdp_obj = SquareDalitzPlot2(M_D0, m_KS, m_pip, m_pim)
IDX = (2, 3, 1)
S_TOTAL = M_D0**2 + m_KS**2 + 2*m_pip**2

dkpp = DKpp()

def sdp_to_dp(pts):
    out = np.empty_like(pts, dtype=float)
    for n, (mp, th) in enumerate(pts):
        out[n, 0], out[n, 1] = sdp_obj.M_from_MpT(mp, th, *IDX)
    return out

def jacobian_vec(s_ij_arr, s_ik_arr):
    out = np.empty(len(s_ij_arr))
    for n in range(len(s_ij_arr)):
        out[n] = sdp_obj.jacobian(float(s_ij_arr[n]), float(s_ik_arr[n]), *IDX)
    return np.maximum(out, 1e-30)

def _fp(x):
    return np.maximum(np.where(np.isfinite(np.asarray(x)), x, 0.0), 1e-14)

def amp_terms(pts_sdp):
    """Return (pF, pFsw, C, S) at SDP points using the exact amplitude.
    Convention: SDP measure (matches run_varying_Bdata.py)."""
    dp = sdp_to_dp(pts_sdp)
    s23, s12 = dp[:, 0], dp[:, 1]
    s13 = S_TOTAL - s23 - s12
    A12 = dkpp.full(np.column_stack([s12, s13]))
    A13 = dkpp.full(np.column_stack([s13, s12]))
    J = jacobian_vec(s23, s12)
    cross = A12 * np.conjugate(A13)
    # SDP measure: divide DP-density quantities by J. (For DP measure, multiply instead.)
    pF, pFsw = (np.abs(A12)**2) / J, (np.abs(A13)**2) / J
    C, S = np.real(cross) / J, np.imag(cross) / J
    return dict(pF=_fp(pF), pFsw=_fp(pFsw),
                C=np.where(np.isfinite(C), C, 0.0),
                S=np.where(np.isfinite(S), S, 0.0))

# Discover B± datasets with event data
bdata_files = sorted([f for f in os.listdir(args.bdata_dir)
                      if f.startswith('Bdata_') and f.endswith('.npz')])
bdata_with_events = []
for f in bdata_files:
    d = np.load(os.path.join(args.bdata_dir, f))
    if 'dataM_sdp' in d.keys():
        bdata_with_events.append(int(f.replace('Bdata_', '').replace('.npz', '')))
print(f"Found {len(bdata_with_events)} B± datasets with event data: {bdata_with_events}")

# Resume support
results_path = os.path.join(SCRIPT_DIR, 'results', 'varying_Bdata_amp.json')
all_results = []
done_idx = set()
if args.resume and os.path.exists(results_path):
    all_results = json.load(open(results_path))
    done_idx = {r['bdata_idx'] for r in all_results}
    print(f"Resuming: {len(done_idx)} realizations already done")

for bdata_idx in bdata_with_events:
    if bdata_idx in done_idx:
        continue
    bdata = np.load(os.path.join(args.bdata_dir, f'Bdata_{bdata_idx:03d}.npz'))
    print(f"Bdata_{bdata_idx:03d}: ", end=''); sys.stdout.flush()

    tM = amp_terms(bdata['dataM_sdp'].astype(np.float64))
    tP = amp_terms(bdata['dataP_sdp'].astype(np.float64))
    mM = amp_terms(bdata['mcM_sdp'].astype(np.float64))
    mP = amp_terms(bdata['mcP_sdp'].astype(np.float64))
    N_M, N_P = len(tM['pF']), len(tP['pF'])

    def nll(rB, delta, gamma):
        thM, thP = np.radians(delta - gamma), np.radians(delta + gamma)
        pBm = _fp(tM['pFsw'] + rB**2*tM['pF'] + 2*rB*(np.cos(thM)*tM['C'] - np.sin(thM)*tM['S']))
        pBp = _fp(tP['pF']   + rB**2*tP['pFsw'] + 2*rB*(np.cos(thP)*tP['C'] + np.sin(thP)*tP['S']))
        mBm = _fp(mM['pFsw'] + rB**2*mM['pF']   + 2*rB*(np.cos(thM)*mM['C'] - np.sin(thM)*mM['S']))
        mBp = _fp(mP['pF']   + rB**2*mP['pFsw'] + 2*rB*(np.cos(thP)*mP['C'] + np.sin(thP)*mP['S']))
        return (-np.log(pBm).sum() + N_M*np.log(mBm.mean())
                -np.log(pBp).sum() + N_P*np.log(mBp.mean()))

    m = Minuit(nll, rB=0.02, delta=160, gamma=100)
    m.limits['rB']    = (0, 1)
    m.limits['delta'] = (0, 360)
    m.limits['gamma'] = (0, 360)
    m.errors['rB'], m.errors['delta'], m.errors['gamma'] = 0.01, 2.0, 2.0
    m.errordef = Minuit.LIKELIHOOD
    m.migrad(); m.hesse()

    rec = {
        'bdata_idx': bdata_idx,
        'rB': float(m.values['rB']),       'rB_err':    float(m.errors['rB']),
        'delta': float(m.values['delta']), 'delta_err': float(m.errors['delta']),
        'gamma': float(m.values['gamma']), 'gamma_err': float(m.errors['gamma']),
        'valid': bool(m.valid),
    }
    all_results.append(rec)
    print(f"rB={rec['rB']:.4f}±{rec['rB_err']:.4f}  "
          f"δ={rec['delta']:6.2f}±{rec['delta_err']:.2f}  "
          f"γ={rec['gamma']:6.2f}±{rec['gamma_err']:.2f}  "
          f"{'✓' if rec['valid'] else '✗'}")

    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)

print(f"\nSaved {len(all_results)} results to {results_path}")

# ----- Across-realization summary -----
all_results.sort(key=lambda r: r['bdata_idx'])
valid = [r for r in all_results if r.get('valid', True)]
print(f"\nAcross {len(valid)} realizations  (σ_total = √(σ_realization² + ⟨fit_err⟩²)):")
for name, fmt in [('rB', '.4f'), ('delta', '.3f'), ('gamma', '.3f')]:
    vals = np.array([r[name]          for r in valid])
    errs = np.array([r[f'{name}_err'] for r in valid])
    spread, mean_err = vals.std(ddof=0), errs.mean()
    sig_tot = np.sqrt(spread**2 + mean_err**2)
    print(f"  {name:>5s}: mean={vals.mean():{fmt}}  "
          f"σ_realization={spread:{fmt}}  ⟨fit_err⟩={mean_err:{fmt}}  "
          f"σ_total={sig_tot:{fmt}}")

# ----- Verification vs varying_Bdata_hnet.json (exact_*) -----
ref_path = os.path.join(SCRIPT_DIR, 'results', 'varying_Bdata_hnet.json')
if os.path.exists(ref_path):
    ref = {r['bdata_idx']: r for r in json.load(open(ref_path))}
    new_by_idx = {r['bdata_idx']: r for r in all_results}
    common = sorted(set(new_by_idx) & set(ref.keys()))
    if len(common) >= 2:
        print(f"\nVerification vs varying_Bdata_hnet.json (exact_*  on first 2 common):")
        for bi in common[:2]:
            r_new, r_ref = new_by_idx[bi], ref[bi]
            print(f"  Bdata_{bi:03d}:")
            for name in ['rB', 'delta', 'gamma']:
                new_v = r_new[name]
                ref_v = r_ref[f'exact_{name}']
                print(f"    {name:>5s}: new={new_v:.4f}  ref={ref_v:.4f}  Δ={new_v-ref_v:+.4f}")
