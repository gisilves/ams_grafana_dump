#!/usr/bin/env python3
"""
Output folder structure:

output/
├── TAG_<tag>
│   ├── <test>
│   │   ├── <LEF>
│   │   │   ├── <LEF>P.csv
│   │   │   ├── <LEF>R.csv
│   │   │   └── <LEF>S.csv

Script to locate all TAG folders that contain a specific LEFP.csv file and
build a *diff table* that, for every channel, holds the time and pedestal
differences between each run and the first (reference) run.
"""

import os
import sys
import re
import numpy as np
import pandas as pd
from typing import List, Tuple
import matplotlib.pyplot as plt


def find_calibs(lef: str, kind: str = "pedestal") -> List[Tuple[str, str]]:
    suffix_map = {"pedestal": "P.csv", "raw_sigma": "R.csv", "sigma": "S.csv"}
    search_suffix = suffix_map[kind]

    calibs: List[Tuple[str, str]] = []
    for tags_folder in os.listdir("output"):
        if not tags_folder.startswith("TAG_"):
            continue

        tag_root = os.path.join("output", tags_folder)
        for root, _dirs, files in os.walk(tag_root):
            for f in files:
                if f.endswith(lef + search_suffix):
                    calibs.append((tags_folder, os.path.join(root, f)))

    # deterministic ordering ⇒ sort so the reference is always the earliest tag
    calibs.sort(key=lambda t: t[0])
    return calibs


def read_calib(path: str) -> pd.DataFrame:
    """Return raw dataframe exactly as read from CSV (no parsing)."""
    return pd.read_csv(
        path,
        dtype={
            "Time": str,
            "channel": int,
            "pedestal": float,
            "raw_sigma": float,
            "sigma": float,
        },
    )

def _core_df(path: str, type: str) -> pd.DataFrame:
    """Load <LEF>P.csv and return a {channel → (Time, pedestal)} frame
    with a *unique* channel index.
    """
    df = read_calib(path)
    df["Time"] = pd.to_datetime(df["Time"], format="%Y-%m-%d %H:%M:%S")

    if df["channel"].duplicated().any():
        dup_ch = df.loc[df["channel"].duplicated(), "channel"].unique()
        print(f"⚠️  {path} contains duplicate channels {dup_ch}. "
              f"Keeping first occurrence for each.")
        df = df.drop_duplicates(subset="channel", keep="first")
    
    return df.set_index("channel")[["Time", type]].sort_index()

def build_diff_table(tags: List[Tuple[str, str]], type: str):
    # reference
    ref_tag, ref_path = tags[0]
    ref_core = _core_df(ref_path, type)

    # master table + index tracker
    diff_table = pd.DataFrame(index=ref_core.index)

    for tagname, path in tags[1:]:
        cur_core = _core_df(path, type)

        # include any extra channels appearing only in this run
        all_channels = diff_table.index.union(cur_core.index)
        diff_table = diff_table.reindex(all_channels)

        # align both cores on the union
        ref_aligned = ref_core.reindex(all_channels)
        cur_aligned = cur_core.reindex(all_channels)

        time_diff = (cur_aligned["Time"] - ref_aligned["Time"]).dt.total_seconds()
        if type == "pedestal":
            cal_diff = cur_aligned["pedestal"] - ref_aligned["pedestal"]
        elif type == "raw_sigma":
            cal_diff = cur_aligned["raw_sigma"] - ref_aligned["raw_sigma"]
        elif type == "sigma":
            cal_diff = cur_aligned["sigma"] - ref_aligned["sigma"]
        else:
            raise ValueError(f"Invalid Type argument. Valid options: pedestal, raw_sigma, sigma")
        
        diff_table[f"TimeDiff_{tagname}"] = time_diff
        if type == "pedestal":
            diff_table[f"PedDiff_{tagname}"] = cal_diff
        elif type == "raw_sigma":
            diff_table[f"RawSigmaDiff_{tagname}"] = cal_diff
        elif type == "sigma":
            diff_table[f"SigmaDiff_{tagname}"] = cal_diff
            
    diff_table = diff_table.reset_index().rename(columns={"channel": "Channel"})
    return diff_table

def plot_channel(df, channel,
                 time_prefix="TimeDiff_TAG_",
                 val_prefix="PedDiff_TAG_"):
    """
    Plot a scatter plot of pedestal/raw_sigma/sigmadiff vs. timediff for a given channel. 
    """
    row = df.loc[df["Channel"] == channel]
    if row.empty:
        raise ValueError(f"Channel {channel} not found.")
    row = row.iloc[0]

    x_vals, y_vals, tags = [], [], []
    for col in df.columns:
        if col.startswith(time_prefix):
            tag = col[len(time_prefix):]
            ped_col = f"{val_prefix}{tag}"
            if ped_col in df.columns:
                x, y = row[col], row[ped_col]

                if pd.isna(x) or pd.isna(y):
                    continue

                x_vals.append(x)
                y_vals.append(y)
                tags.append(tag)

    if not x_vals:
        print(f"All values for Channel {channel} are NaN.")
        return

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(x_vals, y_vals)
    ax.set_xlabel("TimeDiff (s)")
    if sys.argv[2] == "pedestal":
        ax.set_ylabel("PedestalDiff (ADC)")
    elif sys.argv[2] == "raw_sigma":
        ax.set_ylabel("RawSigmaDiff (ADC)")
    elif sys.argv[2] == "sigma":
        ax.set_ylabel("SigmaDiff (ADC)")
    ax.set_title(f"Channel {channel}")
    plt.tight_layout()
    plt.show()
    
    
def compute_diff_stats(df,
                       time_prefix="TimeDiff_TAG_",
                       val_prefix="PedDiff_TAG_",
                       use_rms=False,
                       type="pedestal"):
    """
    For every TAG present, compute mean and RMS of TimeDiff and Ped/RawSigma/SigmaDiff
    over all channels (rows).
    """
    tags = set()
    for col in df.columns:
        m = re.match(fr"{re.escape(time_prefix)}(.+)", col)
        if m is not None:
            tags.add(m.group(1))

    records = []
    
    for tag in sorted(tags):                      # deterministic order

        tcol = f"{time_prefix}{tag}"
        pcol = f"{val_prefix}{tag}"
        
        if tcol not in df.columns or pcol not in df.columns:
            continue                              # skip incomplete pairs

        tvals = df[tcol].astype(float)            # ensure numeric
        pvals = df[pcol].astype(float)

        mean_t = tvals.mean(skipna=True)
        mean_p = pvals.mean(skipna=True)

        if use_rms:
            rms_t = np.sqrt(np.nanmean(tvals**2))
            rms_p = np.sqrt(np.nanmean(pvals**2))
        else:                                     # “RMS” = standard deviation
            rms_t = tvals.std(skipna=True, ddof=1)
            rms_p = pvals.std(skipna=True, ddof=1)
    
        if type == "pedestal":
            records.append(dict(TAG=tag,
                                mean_time=mean_t, rms_time=rms_t,
                                mean_ped=mean_p, rms_ped=rms_p))
        elif type == "raw_sigma":
            records.append(dict(TAG=tag,
                                mean_time=mean_t, rms_time=rms_t,
                                mean_raw_sigma=mean_p, rms_raw_sigma=rms_p))
        elif type == "sigma":
            records.append(dict(TAG=tag,
                                mean_time=mean_t, rms_time=rms_t,
                                mean_sigma=mean_p, rms_sigma=rms_p))
        else:
            raise ValueError(f"Invalid Type argument. Valid options: pedestal, raw_sigma, sigma")

    return pd.DataFrame(records)


def plot_tag_stats(stats_df, type):
    """
    Scatter plot of mean(TimeDiff) vs mean(Ped/RawSigma/SigmaDiff) with ±RMS error bars.
    """
    
    if stats_df.empty:
        raise ValueError("stats_df has no rows - did the TAG pattern match?")

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlabel("Mean TimeDiff")
    
    if type == "pedestal":
        ax.errorbar(stats_df["mean_time"], stats_df["mean_ped"],
                    xerr=stats_df["rms_time"], yerr=stats_df["rms_ped"],
                    fmt="o", ms=4, alpha=0.5, label="TAGs")
        ax.set_ylabel("Mean PedDiff")
        ax.set_title("Per-TAG mean pedestal ± RMS across all channels for LEF " + sys.argv[1])
    elif type == "raw_sigma":
        ax.errorbar(stats_df["mean_time"], stats_df["mean_raw_sigma"],
                    xerr=stats_df["rms_time"], yerr=stats_df["rms_raw_sigma"],
                    fmt="o", ms=4, alpha=0.5, label="TAGs")
        ax.set_ylabel("Mean RawSigmaDiff")
        ax.set_title("Per-TAG mean raw sigma ± RMS across all channels for LEF " + sys.argv[1])
    elif type == "sigma":
        ax.errorbar(stats_df["mean_time"], stats_df["mean_sigma"],
                    xerr=stats_df["rms_time"], yerr=stats_df["rms_sigma"],
                    fmt="o", ms=4, alpha=0.5, label="TAGs")
        ax.set_ylabel("Mean SigmaDiff")
        ax.set_title("Per-TAG mean sigma ± RMS across all channels for LEF " + sys.argv[1])
        
    ax.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.show()
    
def main():
    if len(sys.argv) != 3:
        sys.exit("Usage: python3 calib_diff.py <LEF> <Type (pedestal/raw_sigma/sigma)>")

    lef = sys.argv[1]
    if sys.argv[2] == "pedestal":
        print(f"Searching for TAG folders that contain {lef}P.csv …")
        tags = find_calibs(lef, "pedestal")
    elif sys.argv[2] == "raw_sigma":
        print(f"Searching for TAG folders that contain {lef}R.csv …")
        tags = find_calibs(lef, "raw_sigma")
    elif sys.argv[2] == "sigma":
        print(f"Searching for TAG folders that contain {lef}S.csv …")
        tags = find_calibs(lef, "sigma")
    else:
        sys.exit("Invalid Type argument. Valid options: pedestal, raw_sigma, sigma")

    if not tags:
        sys.exit("No matching calibrations found - nothing to do.")

    print(f"Found {len(tags)} TAG folders:")
    for tag, _ in tags:
        print(f"  • {tag}")
    print(f"\nUsing {tags[0][0]} as reference run.\n")

    diff_table = build_diff_table(tags, sys.argv[2])
        
    if sys.argv[2] == "pedestal":
        stats_df = compute_diff_stats(diff_table, val_prefix="PedDiff_TAG_", type="pedestal")
        plot_tag_stats(stats_df, type="pedestal")
    elif sys.argv[2] == "raw_sigma":
        stats_df = compute_diff_stats(diff_table, val_prefix="RawSigmaDiff_TAG_", type="raw_sigma")
        plot_tag_stats(stats_df, type="raw_sigma")
    elif sys.argv[2] == "sigma":
        stats_df = compute_diff_stats(diff_table, val_prefix="SigmaDiff_TAG_", type="sigma")
        plot_tag_stats(stats_df, type="sigma")
        
if __name__ == "__main__":
    main()
