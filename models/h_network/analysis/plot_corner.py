#!/usr/bin/env python3
"""
Generate corner plots comparing h-network vs 3-flow gamma extraction.

Produces two versions:
  - corner_comparison_v2.pdf:      all 25 3-flow + 5 h-network datasets
  - corner_comparison_matched.pdf: only the 5 matched datasets

Usage:
    python plot_corner.py [--three-flow-results PATH]
"""
import numpy as np, json, os, sys, argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
H_DIR = os.path.join(SCRIPT_DIR, '..')
RESULTS_DIR = os.path.join(H_DIR, 'results')
FIGURES_DIR = os.path.join(H_DIR, 'figures')

parser = argparse.ArgumentParser()
parser.add_argument('--three-flow-results', type=str,
                    default=os.path.join(RESULTS_DIR, 'fit_results_4_15.npz'),
                    help='Path to 3-flow fit results .npz')
args = parser.parse_args()

plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
    'font.serif': ['Computer Modern Roman'],
    'text.latex.preamble': r'\usepackage{amsmath}',
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'figure.dpi': 300,
})

true_rB, true_delta, true_gamma = 0.1, 130.0, 70.0

am = np.load(os.path.join(RESULTS_DIR, 'amplitude_model_fit.npz'))
h_results = json.load(open(os.path.join(RESULTS_DIR, 'ensemble_comparison_symmetric_results_v2.json')))
d = np.load(args.three_flow_results)

seeds = [r['seed'] for r in h_results]

h_rB = np.array([r['rB'] for r in h_results])
h_delta = np.array([r['delta'] for r in h_results])
h_gamma = np.array([r['gamma'] for r in h_results])
h_rB_e = np.array([r['rB_err'] for r in h_results])
h_delta_e = np.array([r['delta_err'] for r in h_results])
h_gamma_e = np.array([r['gamma_err'] for r in h_results])

def sig_tot(vals, errs):
    return np.sqrt(vals.std()**2 + errs.mean()**2)

def make_corner(tf_rB, tf_delta, tf_gamma, tf_rB_e, tf_delta_e, tf_gamma_e,
                n_tf_label, save_path, rB_ylim=None):
    pairs = [
        (h_gamma, tf_gamma, h_gamma_e, tf_gamma_e, float(am['gamma']), float(am['gamma_err']), true_gamma, r'$\gamma\;[^\circ]$'),
        (h_delta, tf_delta, h_delta_e, tf_delta_e, float(am['delta']), float(am['delta_err']), true_delta, r'$\delta_B\;[^\circ]$'),
        (h_rB, tf_rB, h_rB_e, tf_rB_e, float(am['rB']), float(am['rB_err']), true_rB, r'$r_B$'),
    ]
    n = len(pairs)
    fig, axes = plt.subplots(n-1, n-1, figsize=(8, 8))

    for i in range(n-1):
        for j in range(n-1):
            ax = axes[i, j]
            if j > i:
                ax.axis('off')
                continue
            h_x, tf_x, h_xe, tf_xe, am_x, am_xe, true_x, label_x = pairs[j]
            h_y, tf_y, h_ye, tf_ye, am_y, am_ye, true_y, label_y = pairs[i+1]

            ax.axhline(true_y, color='k', ls='--', lw=0.7, alpha=0.35)
            ax.axvline(true_x, color='k', ls='--', lw=0.7, alpha=0.35)

            ax.errorbar(am_x, am_y, xerr=am_xe, yerr=am_ye,
                        fmt='D', color='#2ca02c', markersize=7,
                        capsize=3, capthick=1.0, elinewidth=1.0, zorder=8)
            ax.errorbar(tf_x, tf_y, xerr=tf_xe, yerr=tf_ye,
                        fmt='o', color='#1f77b4', markersize=3.5 if len(tf_x)>5 else 4.5,
                        capsize=1.2, capthick=0.5, elinewidth=0.5, alpha=0.45 if len(tf_x)>5 else 0.55, zorder=3)
            ax.errorbar(h_x, h_y, xerr=h_xe, yerr=h_ye,
                        fmt='s', color='#ff7f0e', markersize=4.5,
                        capsize=1.5, capthick=0.6, elinewidth=0.6, alpha=0.6, zorder=4)

            tf_sx, tf_sy = sig_tot(tf_x, tf_xe), sig_tot(tf_y, tf_ye)
            h_sx, h_sy = sig_tot(h_x, h_xe), sig_tot(h_y, h_ye)
            ax.errorbar(tf_x.mean(), tf_y.mean(), xerr=tf_sx, yerr=tf_sy,
                        fmt='P', color='#1f77b4', markersize=9, capsize=4, capthick=1.5,
                        elinewidth=1.5, zorder=7, markeredgecolor='white', markeredgewidth=1.0)
            ax.errorbar(h_x.mean(), h_y.mean(), xerr=h_sx, yerr=h_sy,
                        fmt='P', color='#ff7f0e', markersize=9, capsize=4, capthick=1.5,
                        elinewidth=1.5, zorder=7, markeredgecolor='white', markeredgewidth=1.0)

            if j == 0: ax.set_ylabel(label_y)
            else: ax.set_yticklabels([])
            if i == n-2: ax.set_xlabel(label_x)
            else: ax.set_xticklabels([])

    if rB_ylim:
        axes[1,0].set_ylim(*rB_ylim)
        axes[1,1].set_ylim(*rB_ylim)

    legend_handles = [
        Line2D([0],[0], marker='D', color='#2ca02c', markersize=7, ls='none', label='Amplitude model'),
        Line2D([0],[0], marker='o', color='#1f77b4', markersize=5, ls='none', alpha=0.5, label=f'3-flow ($N={n_tf_label}$)'),
        Line2D([0],[0], marker='s', color='#ff7f0e', markersize=5, ls='none', alpha=0.6, label=r'$h$-network ($N=5$)'),
        Line2D([0],[0], marker='P', color='#1f77b4', markersize=8, ls='none', markeredgecolor='white', markeredgewidth=0.8, label=r'3-flow mean $\pm\,\sigma_{\rm tot}$'),
        Line2D([0],[0], marker='P', color='#ff7f0e', markersize=8, ls='none', markeredgecolor='white', markeredgewidth=0.8, label=r'$h$-net mean $\pm\,\sigma_{\rm tot}$'),
        Line2D([0],[0], color='k', ls='--', lw=0.7, alpha=0.4, label='Benchmark'),
    ]
    axes[0,1].legend(handles=legend_handles, loc='center', fontsize=12,
                     frameon=True, fancybox=False, edgecolor='0.8')
    axes[0,1].set_visible(True); axes[0,1].axis('off')

    plt.tight_layout(h_pad=0.4, w_pad=0.4)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved {save_path}")

# Version 1: all 25 3-flow + 5 h-network
make_corner(d['results_rB'], d['results_delta'], d['results_gamma'],
            d['errors_rB'], d['errors_delta'], d['errors_gamma'],
            n_tf_label=len(d['results_gamma']),
            save_path=os.path.join(FIGURES_DIR, 'corner_comparison_v2.pdf'),
            rB_ylim=(0.093, 0.108))

# Version 2: matched 5 datasets only
tf_rB_m = np.array([d['results_rB'][s] for s in seeds])
tf_delta_m = np.array([d['results_delta'][s] for s in seeds])
tf_gamma_m = np.array([d['results_gamma'][s] for s in seeds])
tf_rB_e_m = np.array([d['errors_rB'][s] for s in seeds])
tf_delta_e_m = np.array([d['errors_delta'][s] for s in seeds])
tf_gamma_e_m = np.array([d['errors_gamma'][s] for s in seeds])

make_corner(tf_rB_m, tf_delta_m, tf_gamma_m,
            tf_rB_e_m, tf_delta_e_m, tf_gamma_e_m,
            n_tf_label=5,
            save_path=os.path.join(FIGURES_DIR, 'corner_comparison_matched.pdf'),
            rB_ylim=(0.096, 0.106))

print("Done!")
