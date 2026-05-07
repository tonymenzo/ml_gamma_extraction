#!/usr/bin/env python3
"""
Extract per-realization fit results (3-flow ensemble + analytic amplitude model)
from San's Bdata_*.npz files into JSONs that live in our repo.

Each Bdata_NNN.npz already contains:
  flow_rB, flow_delta, flow_gamma                 -- (25,) per-seed 3-flow values
  flow_err_rB, flow_err_delta, flow_err_gamma     -- (25,) per-seed HESSE errors
  exact_rB, exact_delta, exact_gamma              -- scalar amplitude-model values
  exact_err_rB, exact_err_delta, exact_err_gamma  -- scalar amplitude-model HESSE errors

These were originally produced by Gamma-fit-pipeline-SDP-varying-B-data.ipynb in
the Extraction-of-Gamma-with-Normalising-flows repo. We mirror them here so the
fit values + errors remain referenceable even if San's repo moves.

Outputs (under models/h_network/results/):
  varying_Bdata_3flow_per_seed.json  -- one entry per (bdata_idx, seed)
  varying_Bdata_amp.json             -- one entry per bdata_idx (amplitude model)

Usage:
    python extract_san_results.py [--bdata-dir <path>]
"""
import os, json, argparse
import numpy as np

DEFAULT_BDATA = ('/Users/ynot/code/research/Extraction-of-Gamma-with-Normalising-flows/'
                 'varying_Bdata_results')

parser = argparse.ArgumentParser()
parser.add_argument('--bdata-dir', default=DEFAULT_BDATA)
args = parser.parse_args()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

bdata_files = sorted(f for f in os.listdir(args.bdata_dir)
                     if f.startswith('Bdata_') and f.endswith('.npz'))

three_flow, amp = [], []
for fname in bdata_files:
    path = os.path.join(args.bdata_dir, fname)
    bdata_idx = int(fname.replace('Bdata_', '').replace('.npz', ''))
    d = np.load(path)
    if 'flow_rB' not in d.keys():
        print(f'  {fname}: skipped (no fit results)')
        continue
    n_seeds = len(d['flow_rB'])
    for s in range(n_seeds):
        three_flow.append({
            'bdata_idx': bdata_idx, 'seed': s,
            'rB':        float(d['flow_rB'][s]),     'rB_err':    float(d['flow_err_rB'][s]),
            'delta':     float(d['flow_delta'][s]),  'delta_err': float(d['flow_err_delta'][s]),
            'gamma':     float(d['flow_gamma'][s]),  'gamma_err': float(d['flow_err_gamma'][s]),
        })
    amp.append({
        'bdata_idx': bdata_idx,
        'rB':        float(d['exact_rB']),     'rB_err':    float(d['exact_err_rB']),
        'delta':     float(d['exact_delta']),  'delta_err': float(d['exact_err_delta']),
        'gamma':     float(d['exact_gamma']),  'gamma_err': float(d['exact_err_gamma']),
    })
    print(f'  {fname}: {n_seeds} 3-flow seeds + 1 amp fit')

print(f'\nExtracted from {len(bdata_files)} Bdata files in {args.bdata_dir}:')
print(f'  3-flow: {len(three_flow)} per-seed entries '
      f'({len(set((r["bdata_idx"], r["seed"]) for r in three_flow))} unique)')
print(f'  Amp:    {len(amp)} per-realization entries')

flow_path = os.path.join(RESULTS_DIR, 'varying_Bdata_3flow_per_seed.json')
amp_path  = os.path.join(RESULTS_DIR, 'varying_Bdata_amp.json')
with open(flow_path, 'w') as f:
    json.dump(three_flow, f, indent=2)
with open(amp_path, 'w') as f:
    json.dump(amp, f, indent=2)
print(f'\nSaved:\n  {flow_path}\n  {amp_path}')
