# plots.py
#!/usr/bin/env python3
"""
Plot the mean value (pedestal, raw sigma, or sigma) vs time (with TAG 0xC00 as a reference), for all QLs.

Usage :
    python plots.py ql parameter DATAFOLDER 
"""
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import List
import sys

# Define the RMS of mean formula
def rms_of_mean(rms_series):
    n = len(rms_series)
    sum_sq = np.sum(rms_series ** 2)
    return np.sqrt(sum_sq) / n


# ───────────────────────────── Core maths / data-prep ──────────────────────────────────────

def plot_csv(csv_path: str, title: str, xlabel: str, ylabel: str) -> None:
    """
    Load the CSV, aggregate data by TAG, and plot the mean value vs mean time.
    """

    # Load the CSV
    df = pd.read_csv(csv_path)

    # Group by TAG
    grouped = df.groupby('TAG').agg(
        mean_time=('mean_time', 'mean'),
        mean_value=('mean_value', 'mean'),
        rms_mean_value=('rms_value', rms_of_mean)
    ).reset_index()

    plt.figure(figsize=(10, 6))
    plt.errorbar(
        grouped['mean_time'],
        grouped['mean_value'],
        yerr=grouped['rms_mean_value'],
        fmt='o',
        markersize=4,
        capsize=3,
        alpha=0.7
    )

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ───────────────────────────── CLI ──────────────────────────────────────────
def _parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="plots",
        description="Plot the mean value (pedestal, raw sigma, or sigma) vs time (with TAG 0xC00 as a reference), for all QLs.",
    )
    parser.add_argument(
        "ql",
        choices=["QL-L1", "QL-L2", "QL-L3", "QL-L4", "QL-R1", "QL-R2", "QL-R3", "QL-R4", "ALL"],
        help="Which QL to plot",
    )
    parser.add_argument(
        "parameter",
        choices=["pedestal", "raw_sigma", "sigma", "all"],
        help="Which parameter to plot",
    )
    parser.add_argument(
        "--datafolder",
        help="Path to the folder containing the CSV files",
        type=str,
        default="stats",
        required=False
    )
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> None:
    args = _parse_args(argv or sys.argv[1:])
    
    if args.ql != "ALL":
        ql = args.ql
        csv_path = f"{args.datafolder}/{ql}/{args.parameter}_diff_{ql}.csv"
        print(csv_path)
        title = f"{ql} {args.parameter} vs Time"
        xlabel = "Time (s)"
        ylabel = f"{args.parameter} (ADC)"
        plot_csv(csv_path, title, xlabel, ylabel)

if __name__ == "__main__":
    main()
