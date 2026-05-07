#!/usr/bin/env python3
"""
Plot comparison of amplitude model, 3-flow, and h-network across
multiple B± pseudo-data realizations.

Usage:
    python plot_varying_Bdata.py
"""
import numpy as np, json, os, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
H_DIR = os.path.join(SCRIPT_DIR, '..')
RESULTS_DIR = os.path.join(H_DIR, 'results')
FIGURES_DIR = os.path.join(H_DIR, 'figures')

plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
    'font.serif': ['Computer Modern Roman'],
    'text.latex.preamble': r'\usepackage{amsmath}',
    'axes.labelsize': 18,
    'axes.titlesize': 16,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'figure.dpi': 300,
})

results = json.load(open(os.path.join(RESULTS_DIR, 'varying_Bdata_hnet.json')))
true_rB, true_delta, true_gamma = 0.1, 130.0, 70.0

x = np.arange(len(results))
labels = [f'{i+1}' for i in range(len(results))]

exact_rB = np.array([r['exact_rB'] for r in results])
exact_delta = np.array([r['exact_delta'] for r in results])
exact_gamma = np.array([r['exact_gamma'] for r in results])

flow_rB = np.array([r['flow_mean_rB'] for r in results])
flow_delta = np.array([r['flow_mean_delta'] for r in results])
flow_gamma = np.array([r['flow_mean_gamma'] for r in results])
flow_rB_std = np.array([r['flow_std_rB'] for r in results])
flow_delta_std = np.array([r['flow_std_delta'] for r in results])
flow_gamma_std = np.array([r['flow_std_gamma'] for r in results])

h_rB = np.array([r['h_mean_rB'] for r in results])
h_delta = np.array([r['h_mean_delta'] for r in results])
h_gamma = np.array([r['h_mean_gamma'] for r in results])
h_rB_std = np.array([r['h_std_rB'] for r in results])
h_delta_std = np.array([r['h_std_delta'] for r in results])
h_gamma_std = np.array([r['h_std_gamma'] for r in results])

# Clip extreme error bars for display
flow_gamma_std_clip = np.minimum(flow_gamma_std, 5.0)
flow_delta_std_clip = np.minimum(flow_delta_std, 5.0)
flow_rB_std_clip = np.minimum(flow_rB_std, 0.005)

fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
offset = 0.15

params = [
    (axes[0], r'$r_B$', true_rB, exact_rB, flow_rB, flow_rB_std_clip, h_rB, h_rB_std,
     (0.092, 0.104)),
    (axes[1], r'$\delta_B\;[^\circ]$', true_delta, exact_delta, flow_delta, flow_delta_std_clip, h_delta, h_delta_std,
     (125, 135)),
    (axes[2], r'$\gamma\;[^\circ]$', true_gamma, exact_gamma, flow_gamma, flow_gamma_std_clip, h_gamma, h_gamma_std,
     (64, 77)),
]

param_fmts = [
    ('{:.4f}', '{:.4f}', '{:.4f}'),  # rB
    ('{:.1f}$^\\circ$', '{:.1f}$^\\circ$', '{:.1f}$^\\circ$'),  # delta
    ('{:.1f}$^\\circ$', '{:.1f}$^\\circ$', '{:.1f}$^\\circ$'),  # gamma
]

for idx, (ax, ylabel, truth, exact, fl_mean, fl_std, h_mean, h_std, ylim) in enumerate(params):
    ax.axhline(truth, color='k', ls='--', lw=1, alpha=0.4, label='Benchmark')
    ax.errorbar(x, exact, fmt='D', color='#2ca02c', markersize=6, zorder=5,
                label='Amplitude model')
    ax.errorbar(x - offset, fl_mean, yerr=fl_std, fmt='o', color='#1f77b4',
                markersize=5, capsize=3, capthick=0.8, elinewidth=0.8, alpha=0.7,
                zorder=3, label=r'3-flow (mean $\pm\,\sigma_{\rm flow}$)')
    ax.errorbar(x + offset, h_mean, yerr=h_std, fmt='s', color='#ff7f0e',
                markersize=5, capsize=3, capthick=0.8, elinewidth=0.8, alpha=0.7,
                zorder=4, label=r'$h$-network (mean $\pm\,\sigma_{\rm flow}$)')
    # Average lines across B± realizations
    ax.axhline(exact.mean(), color='#2ca02c', ls=':', lw=1.2, alpha=0.5)
    ax.axhline(fl_mean.mean(), color='#1f77b4', ls=':', lw=1.2, alpha=0.5)
    ax.axhline(h_mean.mean(), color='#ff7f0e', ls=':', lw=1.2, alpha=0.5)

    ax.set_ylabel(ylabel)
    ax.set_ylim(*ylim)
    ax.grid(True, alpha=0.15)
    if ax == axes[2]:
        ax.legend(fontsize=13, loc='upper center')

    # Per-panel averages text box with std/sqrt(N) uncertainty
    N_real = len(exact)
    fmt_v = param_fmts[idx][0]
    # Build format string for value ± error
    if idx == 0:  # rB
        def vpm(arr): return f'{arr.mean():.4f} $\\pm$ {arr.std()/np.sqrt(N_real):.4f}'
    else:  # angles
        def vpm(arr): return f'{arr.mean():.1f} $\\pm$ {arr.std()/np.sqrt(N_real):.1f}$^\\circ$'
    avg_box = (
        f'Avg {ylabel}:  '
        f'Amp.: {vpm(exact)},  '
        f'3-flow: {vpm(fl_mean)},  '
        f'$h$-net: {vpm(h_mean)}'
    )
    ax.text(0.5, 0.06, avg_box, transform=ax.transAxes,
            fontsize=17, ha='center', va='bottom',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='0.7', alpha=0.9))

axes[2].set_xlabel(r'$B^\pm$ pseudo-data realization')
axes[2].set_xticks(x)
axes[2].set_xticklabels(labels)

plt.tight_layout()
for ext in ['pdf', 'png']:
    path = os.path.join(FIGURES_DIR, f'varying_Bdata_comparison.{ext}')
    plt.savefig(path, bbox_inches='tight', dpi=300)
    print(f"Saved {path}")
plt.close()

# Summary
print(f"\nAcross {len(results)} B± realizations:")
print(f"  Exact:   γ mean={exact_gamma.mean():.2f} std={exact_gamma.std():.2f}")
print(f"  3-flow:  γ mean={flow_gamma.mean():.2f} std={flow_gamma.std():.2f}")
print(f"  h-net:   γ mean={h_gamma.mean():.2f} std={h_gamma.std():.2f}")
