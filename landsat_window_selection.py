import argparse
from pathlib import Path
import numpy as np
import pandas as pd

WINDOWS = [27, 30, 33, 36, 39, 42]
KEEP_RATIOS = [0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15, 0.10]
AGB_COL = "agbd_live_dry_mg_ha"
REQUIRE_HAS_VALID_LANDSAT = True
MIN_IMAGES_USED = 1
MIN_VALID_FRAC = 0.90
MIN_OBS_COUNT_MEAN = 1.0
HETERO_COMPONENTS = ["NDVI_std", "SWIR1_std", "NBR_std"]


def to_numeric_safe(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df


def safe_zscore(series):
    s = pd.to_numeric(series, errors='coerce')
    mean = s.mean(skipna=True)
    std = s.std(skipna=True, ddof=0)
    if pd.isna(std) or std == 0:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - mean) / std


def minmax_scale(series):
    s = pd.to_numeric(series, errors='coerce')
    s_min = s.min(skipna=True)
    s_max = s.max(skipna=True)
    if pd.isna(s_min) or pd.isna(s_max) or s_max == s_min:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - s_min) / (s_max - s_min)


def preprocess_master(df):
    required_cols = ['sample_id', 'INVYR', AGB_COL, 'n_images_used', 'has_valid_landsat']
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f'Missing required column: {c}')
    numeric_cols = ['INVYR', AGB_COL, 'n_images_used', 'has_valid_landsat']
    for w in WINDOWS:
        prefix = f'w{w}_'
        numeric_cols.extend([prefix + 'valid_frac', prefix + 'valid_pixel_n', prefix + 'obs_count_mean', prefix + 'obs_count_min', prefix + 'obs_count_max', prefix + 'NDVI_std', prefix + 'SWIR1_std', prefix + 'NBR_std'])
    df = to_numeric_safe(df, numeric_cols)
    before = len(df)
    df = df.drop_duplicates(subset=['sample_id']).copy()
    print(f'Dropped duplicated sample_id rows: {before - len(df)}')
    before = len(df)
    df = df.dropna(subset=[AGB_COL]).copy()
    print(f'Dropped rows with missing AGB: {before - len(df)}')
    return df


def filter_base_valid_samples(df):
    out = df.copy()
    if REQUIRE_HAS_VALID_LANDSAT:
        out = out[out['has_valid_landsat'] == 1].copy()
    out = out[out['n_images_used'] >= MIN_IMAGES_USED].copy()
    print(f'Rows after base filter: {len(out)}')
    return out


def prepare_window_dataframe(base_df, window):
    prefix = f'w{window}_'
    needed_cols = ['sample_id', 'INVYR', AGB_COL, 'n_images_used', 'time_window_used']
    for suffix in ['valid_frac', 'valid_pixel_n', 'obs_count_mean', 'obs_count_min', 'obs_count_max', 'NDVI_std', 'SWIR1_std', 'NBR_std']:
        needed_cols.append(prefix + suffix)
    d = base_df[needed_cols].copy()
    for c in d.columns:
        if c.startswith(prefix):
            d.loc[d[c] == -9999, c] = np.nan
    quality_mask = d[prefix + 'valid_frac'].notna() & d[prefix + 'obs_count_mean'].notna() & (d[prefix + 'valid_frac'] >= MIN_VALID_FRAC) & (d[prefix + 'obs_count_mean'] >= MIN_OBS_COUNT_MEAN)
    d = d.loc[quality_mask].copy()
    z_ndvi = safe_zscore(d[prefix + 'NDVI_std'])
    z_swir1 = safe_zscore(d[prefix + 'SWIR1_std'])
    z_nbr = safe_zscore(d[prefix + 'NBR_std'])
    d[prefix + 'hetero_score'] = z_ndvi + z_swir1 + z_nbr
    d = d.sort_values(prefix + 'hetero_score', ascending=True).reset_index(drop=True)
    d['hetero_rank'] = np.arange(1, len(d) + 1)
    return d


def scan_keep_ratios_for_window(d_window, window, n_base):
    prefix = f'w{window}_'
    if d_window.empty:
        return (pd.DataFrame(), {})
    rows = []
    passed_dict = {}
    for keep_ratio in KEEP_RATIOS:
        n_keep = max(1, int(np.floor(len(d_window) * keep_ratio)))
        d_keep = d_window.iloc[:n_keep].copy()
        passed_dict[keep_ratio] = d_keep
        retention_rate = len(d_keep) / n_base if n_base > 0 else np.nan
        hetero_threshold = d_keep[prefix + 'hetero_score'].max()
        rows.append({'window': window, 'keep_ratio': keep_ratio, 'n_base_samples': n_base, 'n_after_quality': len(d_window), 'n_retained': len(d_keep), 'retention_rate': retention_rate, 'hetero_threshold': hetero_threshold, 'hetero_threshold_p10': d_keep[prefix + 'hetero_score'].quantile(0.1), 'hetero_threshold_p25': d_keep[prefix + 'hetero_score'].quantile(0.25), 'hetero_threshold_p50': d_keep[prefix + 'hetero_score'].quantile(0.5), 'hetero_threshold_p75': d_keep[prefix + 'hetero_score'].quantile(0.75), 'hetero_threshold_p90': d_keep[prefix + 'hetero_score'].quantile(0.9)})
    summary_df = pd.DataFrame(rows)
    summary_df['hetero_threshold_scaled'] = minmax_scale(summary_df['hetero_threshold'])
    return (summary_df, passed_dict)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    args = parser.parse_args()
    frame = preprocess_master(pd.read_csv(args.input, low_memory=False))
    base = filter_base_valid_samples(frame)
    summaries = []
    selected = None
    for window in WINDOWS:
        window_frame = prepare_window_dataframe(base, window)
        summary, passed = scan_keep_ratios_for_window(window_frame, window, len(base))
        if not summary.empty:
            summaries.append(summary)
        if window == 33:
            selected = passed.get(0.75)
    if selected is None:
        raise ValueError("No samples pass the 33-pixel window quality criteria.")
    for value in (args.output, args.summary):
        Path(value).parent.mkdir(parents=True, exist_ok=True)
    selected_rows = selected[["sample_id"]].merge(frame, on="sample_id", how="left", validate="one_to_one")
    selected_rows.to_csv(args.output, index=False)
    pd.concat(summaries, ignore_index=True).to_csv(args.summary, index=False)


if __name__ == "__main__":
    main()
