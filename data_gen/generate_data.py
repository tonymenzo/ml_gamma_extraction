#!/usr/bin/env python3
"""
Generate CP-tagged D -> Kspipi data in Square Dalitz Plot coordinates.

Usage:
    python generate_data.py --n-events 20000      # 20k CP-even + 20k CP-odd
    python generate_data.py --n-events 2000000     # 2M each
    python generate_data.py --n-events 500000 --force  # regenerate even if cached

Output files follow the naming convention:
    D_Kspipi_{even,odd}_SDP_{N}k.npy

Reuses amplitude code from generate_test_data.py.
"""
import argparse
import os
import sys
import time

import numpy as np

from DKpp import DKppCorrelated, AmpSample
from generate_test_data import dp_to_sdp, sdp_obj


def size_label(n_events):
    """Convert event count to filename label: 20000 -> '20k', 2000000 -> '2000k'."""
    return f"{n_events // 1000}k"


def generate_cp_data(cp, n_events, nbatch=50000):
    """Generate CP-tagged events with accept-reject sampling."""
    tag = "even" if cp == +1 else "odd"
    print(f"Generating {n_events:,} CP-{tag} events (nbatch={nbatch})...")
    sys.stdout.flush()
    t0 = time.time()
    sampler = AmpSample(DKppCorrelated(cp=cp))
    points_dp = sampler.generate(n_events, nbatch=nbatch)
    points_sdp = dp_to_sdp(points_dp, sdp_obj)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s")
    return points_sdp


def main():
    parser = argparse.ArgumentParser(description="Generate CP-tagged D->Kspipi SDP data")
    parser.add_argument("--n-events", type=int, required=True,
                        help="Number of events per CP tag (e.g. 20000, 500000, 2000000)")
    parser.add_argument("--force", action="store_true",
                        help="Regenerate even if cached files exist")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    args = parser.parse_args()

    label = size_label(args.n_events)
    even_path = f"D_Kspipi_even_SDP_{label}.npy"
    odd_path = f"D_Kspipi_odd_SDP_{label}.npy"

    # Check cache
    if not args.force and os.path.exists(even_path) and os.path.exists(odd_path):
        even = np.load(even_path)
        odd = np.load(odd_path)
        print(f"Cached files found:")
        print(f"  {even_path}: {even.shape}")
        print(f"  {odd_path}: {odd.shape}")
        print("Use --force to regenerate.")
        return

    # Use larger batches for large datasets to speed up accept-reject
    nbatch = 100000 if args.n_events >= 1000000 else 50000

    np.random.seed(args.seed)

    print("=" * 60)
    print(f"Generating {args.n_events:,} events per CP tag")
    print(f"Output: {even_path}, {odd_path}")
    print("=" * 60)

    data_even = generate_cp_data(cp=+1, n_events=args.n_events, nbatch=nbatch)
    data_odd = generate_cp_data(cp=-1, n_events=args.n_events, nbatch=nbatch)

    np.save(even_path, data_even)
    np.save(odd_path, data_odd)

    print(f"\nSaved:")
    print(f"  {even_path}: {data_even.shape}")
    print(f"  {odd_path}: {data_odd.shape}")

    for name, data in [("CP-even", data_even), ("CP-odd", data_odd)]:
        print(f"  {name}: m' [{data[:,0].min():.4f}, {data[:,0].max():.4f}], "
              f"theta' [{data[:,1].min():.4f}, {data[:,1].max():.4f}]")

    print("\nDone!")


if __name__ == "__main__":
    main()
