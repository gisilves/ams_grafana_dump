# calib_diff.py
#!/usr/bin/env python3
"""
Compute per-channel and per-TAG differences between calibration runs for a given LEF.

Usage :
    python calib_diff.py LEF [pedestal|raw_sigma|sigma] [OPTIONS]
"""
import argparse
import os
import sys
import re
import pathlib
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ── Mapping for QLs to LEFs ─────────────────────────────────────────────────
ql_mapping = {
    # -------- LEFT QLs --------
    "QL-L1": [
        "12F36", "12F38", "12F43",
        "12F35", "12F19", "10F10",
        "10F07", "08F06", "08F02",
    ],
    "QL-L2": [
        "12F41", "12F14", "12F24",
        "12F18", "12F06", "10F03",
        "10F14", "08F10", "08F13",
    ],
    "QL-L3": [
        "12F30", "12F26", "12F09",
        "12F42", "12F05", "10F13",
        "10F05", "08F08", "08F07",
    ],
    "QL-L4": [
        "12F33", "12F12", "12F31",
        "12F15", "12F17", "10F12",
        "10F04", "08F09", "08F16",
    ],

    # -------- RIGHT QLs --------
    "QL-R1": [
        "12F37", "12F40", "12F39",
        "12F21", "12F20", "10F16",
        "10F11", "08F12", "08F11",
    ],
    "QL-R2": [
        "12F04", "12F10", "12F34",
        "12F02", "12F27", "10F08",
        "10F06", "08F15", "08F04",
    ],
    "QL-R3": [
        "12F16", "12F03", "12F29",
        "12F46", "12F45", "10F18",
        "10F17", "08F05", "08F18",
    ],
    "QL-R4": [
        "12F01", "12F28", "12F08",
        "12F11", "12F25", "10F15",
        "10F02", "08F17", "08F14",
    ],
}


# ───────────────────────────────────────────────────────────────────────────────────


# ───────────────────────────── I/O helpers ─────────────────────────────────────────

def find_calibs(lef: str, kind: str = "pedestal", root: pathlib.Path | str = "output") -> List[Tuple[str, str]]:
    """Return a list ``[(tag_name, csv_path), …]`` sorted alphabetically by tag.

    Parameters
    ----------
    lef
        LEF identifier, e.g. ``"LEF10"``.
    kind
        One of ``"pedestal"``, ``"raw_sigma"``, or ``"sigma"``.
    root
        Path to the directory that contains all *TAG_* folders.  Defaults to
        the local ``output`` folder but can be overridden (useful for testing).
    """
    suffix_map: Dict[str, str] = {
        "pedestal": "P.csv",
        "raw_sigma": "R.csv",
        "sigma": "S.csv",
    }
    try:
        search_suffix = suffix_map[kind]
    except KeyError as exc:
        raise ValueError("kind must be 'pedestal', 'raw_sigma' or 'sigma'") from exc

    calibs: List[Tuple[str, str]] = []
    root = pathlib.Path(root)
    if not root.exists():
        return calibs  # nothing to find

    for tags_folder in root.iterdir():
        if not tags_folder.name.startswith("TAG_"):
            continue

        for path in tags_folder.rglob(f"*{lef}{search_suffix}"):
            calibs.append((tags_folder.name, str(path)))

    # deterministic ordering → reference is always the earliest tag alphabetically
    calibs.sort(key=lambda t: t[0])
    return calibs


def read_calib(path: str | pathlib.Path) -> pd.DataFrame:
    """Read a calibration CSV exactly as-is (no column munging)."""
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


# ───────────────────────────── Core maths / data-prep ──────────────────────────────────────

def _core_df(path: str | pathlib.Path, col: str) -> pd.DataFrame:
    """Return a frame indexed by *channel* with columns ``Time`` and ``col``.

    Duplicate channels are silently de-duped (first occurrence wins).
    """
    df = read_calib(path)
    df["Time"] = pd.to_datetime(df["Time"], format="%Y-%m-%d %H:%M:%S")

    if df["channel"].duplicated().any():
        dup_ch = df.loc[df["channel"].duplicated(), "channel"].unique()
        print(
            f"⚠️  {path} contains duplicate channels {dup_ch}. Keeping first occurrence.",
            file=sys.stderr,
        )
        df = df.drop_duplicates(subset="channel", keep="first")

    return df.set_index("channel")[["Time", col]].sort_index()


def build_diff_table(tags: List[Tuple[str, str]], kind: str) -> pd.DataFrame:
    """Create the wide diff table described in the docstring."""

    col_map = {
        "pedestal": "pedestal",
        "raw_sigma": "raw_sigma",
        "sigma": "sigma",
    }
    val_col = col_map[kind]

    # reference run
    ref_tag, ref_path = tags[0]
    ref_core = _core_df(ref_path, val_col)

    diff_table = pd.DataFrame(index=ref_core.index)

    for tagname, path in tags[1:]:
        cur_core = _core_df(path, val_col)

        all_channels = diff_table.index.union(cur_core.index)
        diff_table = diff_table.reindex(all_channels)

        ref_aligned = ref_core.reindex(all_channels)
        cur_aligned = cur_core.reindex(all_channels)

        time_diff = (cur_aligned["Time"] - ref_aligned["Time"]).dt.total_seconds()
        val_diff = cur_aligned[val_col] - ref_aligned[val_col]

        diff_table[f"TimeDiff_{tagname}"] = time_diff
        prefix = {
            "pedestal": "PedDiff_",
            "raw_sigma": "RawSigmaDiff_",
            "sigma": "SigmaDiff_",
        }[kind]
        diff_table[f"{prefix}{tagname}"] = val_diff

    return diff_table.reset_index().rename(columns={"channel": "Channel"})


def compute_diff_stats(
    df: pd.DataFrame,
    kind: str,
    use_rms: bool = False,
) -> pd.DataFrame:
    """Per-TAG mean and RMS (or stdev) of Δtime and Δvalue across all channels."""

    time_prefix = "TimeDiff_TAG_"
    val_prefix = {
        "pedestal": "PedDiff_TAG_",
        "raw_sigma": "RawSigmaDiff_TAG_",
        "sigma": "SigmaDiff_TAG_",
    }[kind]

    tags = {m.group(1) for col in df.columns if (m := re.match(fr"{time_prefix}(.+)", col))}

    records = []
    for tag in sorted(tags):  # deterministic order → easier to compare runs
        tcol, vcol = f"{time_prefix}{tag}", f"{val_prefix}{tag}"
        if tcol not in df.columns or vcol not in df.columns:
            continue  # skip incomplete pairs

        tvals = df[tcol].astype(float)
        vvals = df[vcol].astype(float)

        if use_rms:
            rms_t = np.sqrt(np.nanmean(tvals ** 2))
            rms_v = np.sqrt(np.nanmean(vvals ** 2))
        else:
            rms_t = tvals.std(skipna=True, ddof=1)
            rms_v = vvals.std(skipna=True, ddof=1)

        records.append(
            {
                "TAG": tag,
                "mean_time": tvals.mean(skipna=True),
                "rms_time": rms_t,
                "mean_value": vvals.mean(skipna=True),
                "rms_value": rms_v,
            }
        )

    return pd.DataFrame(records)

# ───────────────────────────── Plotting helpers ─────────────────────────────────────────

def plot_channel(
    df: pd.DataFrame,
    channel: int,
    kind: str,
    ax: plt.Axes | None = None,
) -> None:
    """Scatter Δvalue vs Δtime for *one* channel across all TAGs."""
    time_prefix = "TimeDiff_TAG_"
    val_prefix = {
        "pedestal": "PedDiff_TAG_",
        "raw_sigma": "RawSigmaDiff_TAG_",
        "sigma": "SigmaDiff_TAG_",
    }[kind]

    row = df.loc[df["Channel"] == channel]
    if row.empty:
        raise ValueError(f"Channel {channel} not found in diff table.")
    row = row.iloc[0]

    x_vals, y_vals, tags = [], [], []
    for col in df.columns:
        if col.startswith(time_prefix):
            tag = col[len(time_prefix) :]
            vcol = f"{val_prefix}{tag}"
            if vcol in df.columns:
                x, y = row[col], row[vcol]
                if pd.isna(x) or pd.isna(y):
                    continue
                x_vals.append(x)
                y_vals.append(y)
                tags.append(tag)

    if not x_vals:
        print(f"All values for Channel {channel} are NaN.")
        return

    ax = ax or plt.gca()
    ax.scatter(x_vals, y_vals)
    ax.set_xlabel("TimeDiff (s)")
    y_label = {
        "pedestal": "PedestalDiff (ADC)",
        "raw_sigma": "RawSigmaDiff (ADC)",
        "sigma": "SigmaDiff (ADC)",
    }[kind]
    ax.set_ylabel(y_label)
    ax.set_title(f"Channel {channel}")


def plot_tag_stats(stats_df: pd.DataFrame, kind: str) -> None:
    """Scatter mean Δtime vs mean Δvalue with ±RMS error bars for every TAG."""
    if stats_df.empty:
        raise ValueError("stats_df has no rows - did the TAG pattern match?")

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlabel("Mean TimeDiff (s)")
    ax.set_ylabel({
        "pedestal": "Mean PedestalDiff (ADC)",
        "raw_sigma": "Mean RawSigmaDiff (ADC)",
        "sigma": "Mean SigmaDiff (ADC)",
    }[kind])

    ax.errorbar(
        stats_df["mean_time"],
        stats_df["mean_value"],
        xerr=stats_df["rms_time"],
        yerr=stats_df["rms_value"],
        fmt="o",
        ms=4,
        alpha=0.6,
    )

    ax.set_title(f"Per-TAG mean {kind.replace('_', ' ')} ± RMS across all channels")
    ax.grid(True, ls="--", alpha=0.3)
    fig.tight_layout()
    plt.show()
    
def save_tag_stats(stats_df: pd.DataFrame, kind: str, output: str | pathlib.Path) -> None:
    """Write per-TAG mean and RMS to a CSV file."""
    if stats_df.empty:
        raise ValueError("stats_df has no rows - did the TAG pattern match?")
    
    stats_df.to_csv(output, index=False)
    print(f"Stats table written")


# ───────────────────────────── CLI ──────────────────────────────────────────
def _parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="calib_diff",
        description="Compute time/value drifts between calibration TAGs.",
    )
    parser.add_argument(
        "level",
        choices=["LEF", "QL"],
        help="Which level to analyse (single LEF or whole QL).",
    )
    parser.add_argument(
        "kind",
        choices=["pedestal", "raw_sigma", "sigma"],
        help="Which quantity to analyse.",
    )
    parser.add_argument(
        "--name",
        help="Name of the object (e.g. LEF name or QL name).",
        required=False,
        default=None,
        type=str
    )
    parser.add_argument(
        "--plots",
        action="store_true",
        help="Interactive plotting (useful in CI environments).",
    )
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Only compute the per-TAG statistics table (skip diff table).",
    )
    parser.add_argument(
        "--cumulative",
        action="store_true",
        help="Compute cumulative statistics (useful for plotting) for a QL",
    )
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> None:
    args = _parse_args(argv or sys.argv[1:])

    if args.level == "LEF":
        print("Finding TAGs at LEF level")
        if args.name is None:
            lefname = input("Enter LEF name: ")
        else:
            lefname = args.name
        tags = find_calibs(lefname, args.kind)
        if not tags:
            sys.exit("No matching calibrations found - nothing to do.")
        
        print(f"Found {len(tags)} TAG folders (reference = {tags[0][0]}).")
        
        diff_table = build_diff_table(tags, args.kind)
        
        if args.stats_only:
            stats_df = compute_diff_stats(diff_table, args.kind)
            print(stats_df)
            return
        
        # Always compute stats - plots may or may not be shown depending on flag
        stats_df = compute_diff_stats(diff_table, args.kind)
        
        if args.plots:
            plot_tag_stats(stats_df, args.kind)
        else:
            savefile = "stats/" + args.kind + "_diff_LEF" + lefname + ".csv"
            save_tag_stats(stats_df, args.kind, savefile)
            
    elif args.level == "QL":
        print("Finding TAGs at QL level")
        if args.name is None:
            qlname = input("Enter QL name: ")
        else:
            qlname = args.name
        
        # Check if the QL is in the mapping
        if qlname not in ql_mapping:
            sys.exit(f"QL {qlname} not found in mapping.")
        
        # Create folder for this QL in stats/
        if not os.path.exists("stats/" + qlname):
            os.makedirs("stats/" + qlname)
            
        # Find all LEFs in the QL from the mapping
        lefs = ql_mapping[qlname]
        print(f"Found {len(lefs)} LEFs in QL {qlname}.")
        for lef in lefs:
            print(f"  • {lef}")
        
        
        cumulative_df = pd.DataFrame()
        if args.cumulative:
            print("Computing cumulative statistics for QL") 
            
        cumulative_frames = []

        for lef in lefs:
            tags = find_calibs(lef, args.kind)
            if not tags:
                print(f"[warn] LEF {lef}: no matching calibrations - skipping")
                continue

            print(f"Found {len(tags)} TAG folders (reference = {tags[0][0]}).")

            diff_table = build_diff_table(tags, args.kind)

            # Compute stats no matter what
            stats_df = compute_diff_stats(diff_table, args.kind)

            if args.stats_only:
                print(stats_df)
                continue

            if args.plots:
                plot_tag_stats(stats_df, args.kind)
            else:
                savefile = f"stats/{qlname}/{args.kind}_diff_LEF{lef}.csv"
                os.makedirs(os.path.dirname(savefile), exist_ok=True)
                save_tag_stats(stats_df, args.kind, savefile)

            if args.cumulative:
                cumulative_frames.append(
                    stats_df.assign(lef=lef)
                )

        if args.cumulative and cumulative_frames:
            cumulative_df = pd.concat(cumulative_frames, ignore_index=True)
            savefile = f"stats/{qlname}/{args.kind}_diff_{qlname}.csv"
            os.makedirs(os.path.dirname(savefile), exist_ok=True)
            cumulative_df.to_csv(savefile, index=False)
            print(f"Cumulative stats table written → {savefile}")
            print(cumulative_df)
        elif args.cumulative:
            print("[info] No LEFs produced stats; nothing to write cumulatively.")

if __name__ == "__main__":
    main()