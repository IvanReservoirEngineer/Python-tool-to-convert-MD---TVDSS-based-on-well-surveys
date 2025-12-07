"""
MD <-> TVDSS converter based on well trajectories.

Inputs:
1) Text file with trajectories (same format as in Day 1):
   well  X  Y  MD  TVDSS  INCL  AZIM   (space / tab separated)

2) Excel file with a list of wells and MD or TVDSS values:
   - must contain a column with well names/numbers (e.g. 'Well', 'well')
   - and either:
       a) column with MD (e.g. 'MD', 'MD, m')  -> mode: md2tvdss
       b) column with TVDSS (e.g. 'TVDSS', 'TVDSS, m') -> mode: tvdss2md

The script creates a NEW Excel file with an extra column
('TVDSS_from_MD' or 'MD_from_TVDSS') next to input data.
"""

import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


# ---------- Helpers for reading trajectories ----------

def load_trajectories(path: Path) -> pd.DataFrame:
    """
    Load trajectory database from a text file.
    The file must have a header row with at least:
    well, X, Y, MD, TVDSS, INCL, AZIM
    """
    df = pd.read_csv(path, sep=r"\s+")

    expected_cols = {"well", "X", "Y", "MD", "TVDSS", "INCL", "AZIM"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns in file: {missing}")

    return df


def build_curves(df: pd.DataFrame) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    For each well build arrays (MD, TVDSS), sorted by MD.
    Returns dict: well -> (md_array, tvdss_array)
    """
    curves = {}
    for well, grp in df.groupby("well"):
        grp_sorted = grp.sort_values("MD")
        md = grp_sorted["MD"].to_numpy(dtype=float)
        tvd = grp_sorted["TVDSS"].to_numpy(dtype=float)
        curves[str(well)] = (md, tvd)
    return curves


# ---------- Linear interpolation along a single trajectory ----------

def interp_md_to_tvdss(md_arr: np.ndarray,
                       tvd_arr: np.ndarray,
                       md_target: float) -> tuple[float, bool]:
    """
    Interpolate TVDSS at md_target.
    Returns (value, is_extrapolated).
    If extrapolated, value is a linear extrapolation using the edge segment.
    """
    if np.isnan(md_target) or len(md_arr) < 2:
        return np.nan, False

    md_min, md_max = md_arr[0], md_arr[-1]

    # Inside range -> pure interpolation
    if md_min <= md_target <= md_max:
        val = float(np.interp(md_target, md_arr, tvd_arr))
        return val, False

    # Outside range -> extrapolation
    if md_target < md_min:
        x1, x2 = md_arr[0], md_arr[1]
        y1, y2 = tvd_arr[0], tvd_arr[1]
    else:  # md_target > md_max
        x1, x2 = md_arr[-2], md_arr[-1]
        y1, y2 = tvd_arr[-2], tvd_arr[-1]

    if x2 == x1:
        return np.nan, True

    val = y1 + (y2 - y1) * (md_target - x1) / (x2 - x1)
    return float(val), True


def interp_tvdss_to_md(md_arr: np.ndarray,
                       tvd_arr: np.ndarray,
                       tvd_target: float) -> tuple[float, bool]:
    """
    Interpolate MD at tvd_target.
    Returns (value, is_extrapolated).
    Works by sorting points by TVDSS and interpolating/extrapolating along TVDSS.
    """
    if np.isnan(tvd_target) or len(md_arr) < 2:
        return np.nan, False

    # Sort by TVDSS to get a monotonic independent variable
    order = np.argsort(tvd_arr)
    tvd_sorted = tvd_arr[order]
    md_sorted = md_arr[order]

    tvd_min, tvd_max = tvd_sorted[0], tvd_sorted[-1]

    # Inside range -> interpolation
    if tvd_min <= tvd_target <= tvd_max:
        val = float(np.interp(tvd_target, tvd_sorted, md_sorted))
        return val, False

    # Outside range -> extrapolation using edge segment
    if tvd_target < tvd_min:
        t1, t2 = tvd_sorted[0], tvd_sorted[1]
        m1, m2 = md_sorted[0], md_sorted[1]
    else:  # tvd_target > tvd_max
        t1, t2 = tvd_sorted[-2], tvd_sorted[-1]
        m1, m2 = md_sorted[-2], md_sorted[-1]

    if t2 == t1:
        return np.nan, True

    val = m1 + (m2 - m1) * (tvd_target - t1) / (t2 - t1)
    return float(val), True


# ---------- Helpers for Excel columns ----------

def find_column(df: pd.DataFrame, *keywords) -> str:
    """
    Find first column whose cleaned name contains ANY of keywords.
    Cleaning: lower-case, remove spaces and commas.
    """
    kw = [k.lower() for k in keywords]
    for col in df.columns:
        clean = col.strip().lower().replace(" ", "").replace(",", "")
        if any(k in clean for k in kw):
            return col
    raise ValueError(f"Could not find column for keywords: {keywords}")


# ---------- Main conversion logic ----------

def process_excel_md2tvdss(df_excel: pd.DataFrame,
                           curves: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> pd.DataFrame:
    well_col = find_column(df_excel, "well")
    md_col = find_column(df_excel, "md")

    values = []
    comments = []

    for _, row in df_excel.iterrows():
        well = str(row[well_col])
        md_val = row[md_col]
        curve = curves.get(well)

        if curve is None:
            values.append(np.nan)
            comments.append("")  # or "no trajectory" if you prefer
        else:
            md_arr, tvd_arr = curve
            val, is_extrap = interp_md_to_tvdss(md_arr, tvd_arr, md_val)
            values.append(val)
            comments.append("extrapolated" if is_extrap else "")

    # Round TVDSS to 2 decimal places
    df_excel["TVDSS_from_MD"] = np.round(values, 2)
    df_excel["TVDSS_from_MD_comment"] = comments
    return df_excel


def process_excel_tvdss2md(df_excel: pd.DataFrame,
                           curves: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> pd.DataFrame:
    well_col = find_column(df_excel, "well")
    tvd_col = find_column(df_excel, "tvd", "tvdss")

    values = []
    comments = []

    for _, row in df_excel.iterrows():
        well = str(row[well_col])
        tvd_val = row[tvd_col]
        curve = curves.get(well)

        if curve is None:
            values.append(np.nan)
            comments.append("")  # or "no trajectory"
        else:
            md_arr, tvd_arr = curve
            val, is_extrap = interp_tvdss_to_md(md_arr, tvd_arr, tvd_val)
            values.append(val)
            comments.append("extrapolated" if is_extrap else "")

    df_excel["MD_from_TVDSS"] = np.round(values, 2)
    df_excel["MD_from_TVDSS_comment"] = comments
    return df_excel



# ---------- CLI ----------
def main():
    parser = argparse.ArgumentParser(
        description="Convert MD <-> TVDSS for selected wells using trajectories."
    )
    parser.add_argument(
        "traj_file",
        type=str,
        help="Path to the trajectories .txt file",
    )
    parser.add_argument(
        "excel_file",
        type=str,
        help="Path to the Excel file with wells and MD/TVDSS values",
    )
    parser.add_argument(
        "--mode",
        choices=["md2tvdss", "tvdss2md"],
        required=True,
        help=(
            "Conversion mode: "
            "'md2tvdss' – Excel содержит MD, посчитать TVDSS; "
            "'tvdss2md' – Excel содержит TVDSS, посчитать MD."
        ),
    )
    args = parser.parse_args()

    traj_path = Path(args.traj_file)
    excel_path = Path(args.excel_file)

    if not traj_path.is_file():
        raise SystemExit(f"ERROR: trajectories file not found: {traj_path}")
    if not excel_path.is_file():
        raise SystemExit(f"ERROR: Excel file not found: {excel_path}")

    # 1) Trajectories
    df_traj = load_trajectories(traj_path)
    curves = build_curves(df_traj)

    # 2) Excel
    #df_excel = pd.read_excel(excel_path)
    df_excel = pd.read_excel(excel_path, header=1)  # headers are in 2nd line
    
    # Drop any technical columns without names ("Unnamed: 0", "Unnamed: 1", etc.)
    df_excel = df_excel.loc[:, ~df_excel.columns.str.contains(r"^Unnamed")]


    # 3) Conversion
    if args.mode == "md2tvdss":
        df_out = process_excel_md2tvdss(df_excel, curves)
        suffix = "_with_TVDSS.xlsx"
    else:
        df_out = process_excel_tvdss2md(df_excel, curves)
        suffix = "_with_MD.xlsx"

    # 4) Save result
    out_path = excel_path.with_name(excel_path.stem + suffix)
    df_out.to_excel(out_path, index=False)
    
    print(f"Done. Result saved to: {out_path}")


if __name__ == "__main__":
    main()