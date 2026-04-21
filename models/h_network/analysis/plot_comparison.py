#!/usr/bin/env python3
"""
Produce apples-to-apples comparison plots:
  - Sym-SIREN h-network vs 3-flow, seed by seed
  - Amplitude model baseline
  - Mean ± total uncertainty bands
"""
import numpy as np, json, sys, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
H_DIR = os.path.join(SCRIPT_DIR, '..')
ROOT = os.path.join(H_DIR, '..')

true_rB, true_delta, true_gamma = 0.1, 130.0, 70.0

# Load results
RESULTS_DIR = os.path.join(H_DIR, 'results')
am = np.load(os.path.join(RESULTS_DIR, 'amplitude_model_fit.npz'))
h_results = json.load(open(os.path.join(RESULTS_DIR, 'ensemble_comparison_symmetric_results.json')))
d = np.load(os.path.join(RESULTS_DIR, 'fit_results_symmetric.npz'))
seeds = [r['seed'] for r in h_results]
n_seeds = len(seeds)

print(f"Loaded {n_seeds} h-network seeds, 25 3-flow seeds")

params = [
    ('rB',    r'$r_B$',             true_rB,    'rB',    'rB_err'),
    ('delta', r'$\delta_B$ (deg)',  true_delta, 'delta', 'delta_err'),
    ('gamma', r'$\gamma$ (deg)',    true_gamma, 'gamma', 'gamma_err'),
]

# ── Figure 1: Seed-by-seed comparison ─────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 5))

for ax, (pname, label, truth, key, ekey) in zip(axes, params):
    am_v, am_e = float(am[key]), float(am[ekey])

    # True value
    ax.axhline(truth, color='k', ls='--', lw=1.5, zorder=0, label='Truth')

    # Amplitude model
    ax.errorbar([-0.5], [am_v], yerr=[am_e], fmt='D', color='green',
                markersize=9, capsize=4, capthick=1.5, zorder=5,
                label='Amplitude model')
    ax.axhspan(am_v - am_e, am_v + am_e, color='green', alpha=0.08, zorder=0)

    # 3-flow (matching seeds only)
    x = np.arange(n_seeds)
    tf_v = np.array([d[f'results_{pname}'][s] for s in seeds])
    tf_e = np.array([d[f'errors_{pname}'][s] for s in seeds])
    ax.errorbar(x + 1, tf_v, yerr=tf_e, fmt='o', color='C0',
                markersize=7, capsize=3, capthick=1.2, zorder=4, label='3-flow')

    # h-network
    h_v = np.array([r[key] for r in h_results])
    h_e = np.array([r[ekey] for r in h_results])
    ax.errorbar(x + 1.3, h_v, yerr=h_e, fmt='s', color='C1',
                markersize=7, capsize=3, capthick=1.2, zorder=4, label='h-network')

    # 3-flow mean ± total uncertainty
    tf_mean = tf_v.mean()
    tf_sig = np.sqrt(tf_v.std()**2 + tf_e.mean()**2)
    ax.axhline(tf_mean, color='C0', ls='--', lw=1, alpha=0.7, zorder=2)
    ax.axhspan(tf_mean - tf_sig, tf_mean + tf_sig, color='C0', alpha=0.08, zorder=1)

    # h-network mean ± total uncertainty
    h_mean = h_v.mean()
    h_sig = np.sqrt(h_v.std()**2 + h_e.mean()**2)
    ax.axhline(h_mean, color='C1', ls='--', lw=1, alpha=0.7, zorder=2)
    ax.axhspan(h_mean - h_sig, h_mean + h_sig, color='C1', alpha=0.08, zorder=1)

    ax.set_ylabel(label, fontsize=13)
    ax.set_xticks([-0.5] + list(x + 1.15))
    ax.set_xticklabels(['Amp.\nmodel'] + [str(s+1) for s in seeds], fontsize=10)
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.2, axis='y')

plt.suptitle(f'$B^\\pm \\to DK^\\pm$ fit comparison: amplitude model vs 3-flow vs h-network'
             f'\n(shared pseudo-data, $r_B$={true_rB}, 100k $B^+$ + 100k $B^-$, 5 datasets)',
             fontsize=13)
plt.tight_layout()
p1 = os.path.join(H_DIR, 'figures', 'method_comparison_symmetric.pdf')
plt.savefig(p1, bbox_inches='tight')
plt.close()
print(f"Saved {p1}")

# ── Summary table ─────────────────────────────────────────────────────────
print(f"\n{'='*95}")
print(f"  SEED-BY-SEED COMPARISON (Sym-SIREN 256x5 vs 3-flow)")
print(f"  Truth: rB={true_rB}, delta={true_delta}, gamma={true_gamma}")
print(f"{'='*95}")
print(f"  {'Seed':>4s}  {'Method':>10s}  {'rB (pull)':>14s}  {'delta (pull)':>16s}  {'gamma (pull)':>16s}  {'RMSE':>6s}")
print(f"  {'-'*85}")

for r in h_results:
    s = r['seed']
    rB_p = (r['rB'] - true_rB) / r['rB_err']
    d_p  = (r['delta'] - true_delta) / r['delta_err']
    g_p  = (r['gamma'] - true_gamma) / r['gamma_err']
    tf_rB_p = (d['results_rB'][s] - true_rB) / d['errors_rB'][s]
    tf_d_p  = (d['results_delta'][s] - true_delta) / d['errors_delta'][s]
    tf_g_p  = (d['results_gamma'][s] - true_gamma) / d['errors_gamma'][s]
    print(f"   {s}    h-net  {r['rB']:.4f}({rB_p:+5.1f}σ)  {r['delta']:6.2f}({d_p:+5.1f}σ)  {r['gamma']:6.2f}({g_p:+5.1f}σ)  {r['rmse']:.4f}")
    print(f"         3-flow {d['results_rB'][s]:.4f}({tf_rB_p:+5.1f}σ)  {d['results_delta'][s]:6.2f}({tf_d_p:+5.1f}σ)  {d['results_gamma'][s]:6.2f}({tf_g_p:+5.1f}σ)")
    print()

# Aggregates
h_rB = np.array([r['rB'] for r in h_results])
h_delta = np.array([r['delta'] for r in h_results])
h_gamma = np.array([r['gamma'] for r in h_results])
h_rB_e = np.array([r['rB_err'] for r in h_results])
h_delta_e = np.array([r['delta_err'] for r in h_results])
h_gamma_e = np.array([r['gamma_err'] for r in h_results])

tf_rB_s = np.array([d['results_rB'][s] for s in seeds])
tf_delta_s = np.array([d['results_delta'][s] for s in seeds])
tf_gamma_s = np.array([d['results_gamma'][s] for s in seeds])
tf_rB_e_s = np.array([d['errors_rB'][s] for s in seeds])
tf_delta_e_s = np.array([d['errors_delta'][s] for s in seeds])
tf_gamma_e_s = np.array([d['errors_gamma'][s] for s in seeds])

print(f"  {'='*85}")
print(f"  AGGREGATE ({n_seeds} seeds)")
print(f"  {'='*85}")
for name, truth, h_v, h_e, tf_v, tf_e in [
    ('rB', true_rB, h_rB, h_rB_e, tf_rB_s, tf_rB_e_s),
    ('delta', true_delta, h_delta, h_delta_e, tf_delta_s, tf_delta_e_s),
    ('gamma', true_gamma, h_gamma, h_gamma_e, tf_gamma_s, tf_gamma_e_s)]:
    h_sig_tot = np.sqrt(h_v.std()**2 + h_e.mean()**2)
    tf_sig_tot = np.sqrt(tf_v.std()**2 + tf_e.mean()**2)
    print(f"  {name:>6s}:")
    print(f"    h-net:  mean={h_v.mean():.4f}  bias={h_v.mean()-truth:+.4f}  "
          f"σ_flow={h_v.std():.4f}  σ_fit={h_e.mean():.4f}  σ_tot={h_sig_tot:.4f}")
    print(f"    3-flow: mean={tf_v.mean():.4f}  bias={tf_v.mean()-truth:+.4f}  "
          f"σ_flow={tf_v.std():.4f}  σ_fit={tf_e.mean():.4f}  σ_tot={tf_sig_tot:.4f}")
    print()

print("Done!")
