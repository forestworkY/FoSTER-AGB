import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from .config import TARGET, GROUP_COL, ALL_FEATURE_COLS, GEDI_COLS, SPECTRAL_COLS, ROBUST_SPECTRAL_COLS, ENV_COLS, ModelConfig

CFG = ModelConfig()
AGB_STRATA_BINS = 8
TEST_MIN_L3_SIZE = 50
TEST_RATIO = 0.20
VAL_RATIO = 0.15
SPLIT_SEED = 6


def derive_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    eps = 1e-06
    NIR = df['w33_NIR_mean']
    SWIR1 = df['w33_SWIR1_mean']
    SWIR2 = df['w33_SWIR2_mean']
    Red = df['w33_Red_mean']
    NDVI = df['NDVI'].clip(-1, 1)
    NDMI = df['NDMI'].clip(-1, 1)
    NBR = df['NBR'].clip(-1, 1)
    df['NBR2'] = (SWIR1 - SWIR2) / (SWIR1 + SWIR2 + eps)
    df['NDMI_over_NDVI'] = NDMI / (NDVI.abs() + eps)
    df['NBR_over_NDVI'] = NBR / (NDVI.abs() + eps)
    df['NDMI_minus_NBR'] = NDMI - NBR
    df['SWIR1_over_NIR'] = SWIR1 / (NIR + eps)
    df['SWIR2_over_NIR'] = SWIR2 / (NIR + eps)
    df['SWIR1_over_SWIR2'] = SWIR1 / (SWIR2 + eps)
    df['Red_over_NIR'] = Red / (NIR + eps)
    bands = df[['w33_Blue_mean', 'w33_Green_mean', 'w33_Red_mean', 'w33_NIR_mean', 'w33_SWIR1_mean', 'w33_SWIR2_mean']]
    band_sum = bands.abs().sum(axis=1) + eps
    df['bp_Blue'] = df['w33_Blue_mean'] / band_sum
    df['bp_Green'] = df['w33_Green_mean'] / band_sum
    df['bp_Red'] = df['w33_Red_mean'] / band_sum
    df['bp_NIR'] = df['w33_NIR_mean'] / band_sum
    df['bp_SWIR1'] = df['w33_SWIR1_mean'] / band_sum
    df['bp_SWIR2'] = df['w33_SWIR2_mean'] / band_sum
    return df


def clip_shape_complexity(df, lo_val=None, hi_val=None, fit=False):
    col = 'Shape_Complexity'
    if fit:
        lo_val = df[col].quantile(CFG.clip_lo)
        hi_val = df[col].quantile(CFG.clip_hi)
    df = df.copy()
    df[col] = df[col].clip(lo_val, hi_val)
    return (df, lo_val, hi_val)


def preprocess(train_df, val_df, test_df, feature_cols):
    train_df, lo, hi = clip_shape_complexity(train_df, fit=True)
    val_df, _, _ = clip_shape_complexity(val_df, lo_val=lo, hi_val=hi)
    test_df, _, _ = clip_shape_complexity(test_df, lo_val=lo, hi_val=hi)
    scaler = RobustScaler()
    X_train = scaler.fit_transform(train_df[feature_cols].values.astype(np.float32))
    X_val = scaler.transform(val_df[feature_cols].values.astype(np.float32))
    X_test = scaler.transform(test_df[feature_cols].values.astype(np.float32))
    y_train = np.log1p(train_df[TARGET].values.astype(np.float32))
    y_val = np.log1p(val_df[TARGET].values.astype(np.float32))
    y_test = np.log1p(test_df[TARGET].values.astype(np.float32))
    y_test_raw = test_df[TARGET].values.astype(np.float32)
    return (X_train, y_train, X_val, y_val, X_test, y_test, y_test_raw, scaler)


class AGBDataset(Dataset):

    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y).unsqueeze(1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (self.X[idx], self.y[idx])


def make_loaders(X_train, y_train, X_val, y_val, batch_size=512):
    train_ds = AGBDataset(X_train, y_train)
    val_ds = AGBDataset(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False, num_workers=0, pin_memory=False)
    return (train_loader, val_loader)


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df[GROUP_COL] = df[GROUP_COL].astype(str).str.strip()
    df = derive_features(df)
    df = df[df[TARGET] > 0].reset_index(drop=True)
    print(f'[data] 加载完成：{len(df)} 行，{df[GROUP_COL].nunique()} 个 L3')
    return df


def _agb_bins(y, n_bins=AGB_STRATA_BINS):
    y = pd.Series(y).astype(float)
    bins = min(int(n_bins), max(2, int(y.notna().sum())))
    while bins >= 2:
        try:
            return pd.qcut(y, q=bins, labels=False, duplicates='drop').astype('Int64').astype(str)
        except ValueError:
            bins -= 1
    return pd.Series(['0'] * len(y), index=y.index, dtype=str)


def _combined_agb_l3_strata(df_part, min_count=2):
    agb_bin = _agb_bins(df_part[TARGET], AGB_STRATA_BINS).reset_index(drop=True).astype(str)
    l3 = df_part[GROUP_COL].astype(str).reset_index(drop=True)
    combo = agb_bin + '__' + l3
    combo_counts = combo.value_counts()
    strata = combo.where(combo.map(combo_counts) >= min_count, agb_bin)
    strata = strata.where(strata.map(strata.value_counts()) >= min_count, 'all')
    if strata.nunique() < 2 or strata.value_counts().min() < min_count:
        agb_counts = agb_bin.value_counts()
        strata = agb_bin.where(agb_bin.map(agb_counts) >= min_count, 'all')
    return strata.astype(str).to_numpy()


def _stratified_sample_split(df_part, test_size, seed):
    strata = _combined_agb_l3_strata(df_part, min_count=2)
    idx = np.arange(len(df_part))
    try:
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        train_idx, hold_idx = next(splitter.split(idx, strata))
    except ValueError:
        train_idx, hold_idx = train_test_split(idx, test_size=test_size, random_state=seed, shuffle=True)
    return (train_idx, hold_idx, strata)


def _active_split_seed():
    return CFG.seed if SPLIT_SEED is None else int(SPLIT_SEED)


def _l3_agb_bins(l3_stats, n_bins=AGB_STRATA_BINS):
    bins = min(int(n_bins), max(2, len(l3_stats)))
    while bins >= 2:
        try:
            return pd.qcut(l3_stats['agb_median'], q=bins, labels=False, duplicates='drop').astype('Int64').astype(str)
        except ValueError:
            bins -= 1
    return pd.Series(['0'] * len(l3_stats), index=l3_stats.index, dtype=str)


def _choose_l3_holdout_by_agb(df0):
    l3_stats = df0.groupby(GROUP_COL, dropna=False)[TARGET].agg(n='size', agb_mean='mean', agb_median='median').reset_index()
    l3_stats[GROUP_COL] = l3_stats[GROUP_COL].astype(str)
    eligible = l3_stats[l3_stats['n'] >= TEST_MIN_L3_SIZE].copy()
    small = l3_stats[l3_stats['n'] < TEST_MIN_L3_SIZE].copy()
    if eligible.empty:
        raise ValueError(f'No L3 has at least TEST_MIN_L3_SIZE={TEST_MIN_L3_SIZE} samples; cannot build an unseen-L3 test split.')
    eligible['agb_bin'] = _l3_agb_bins(eligible, AGB_STRATA_BINS)
    rng = np.random.default_rng(_active_split_seed())
    selected = set()
    target_total = len(df0) * TEST_RATIO
    for _, bin_df in eligible.groupby('agb_bin', sort=True):
        bin_df = bin_df.copy()
        order = rng.permutation(len(bin_df))
        bin_df = bin_df.iloc[order]
        bin_target = bin_df['n'].sum() * TEST_RATIO
        picked_n = 0
        bin_selected = []
        for _, row in bin_df.iterrows():
            if not bin_selected or abs(picked_n + row['n'] - bin_target) <= abs(picked_n - bin_target):
                bin_selected.append(str(row[GROUP_COL]))
                picked_n += int(row['n'])
            if picked_n >= bin_target:
                break
        selected.update(bin_selected)
    selected_n = int(eligible.loc[eligible[GROUP_COL].isin(selected), 'n'].sum())
    improved = True
    while improved:
        improved = False
        current_gap = abs(selected_n - target_total)
        unselected = eligible[~eligible[GROUP_COL].isin(selected)].copy()
        if not unselected.empty:
            unselected['gap'] = (selected_n + unselected['n'] - target_total).abs()
            best_add = unselected.sort_values(['gap', 'n']).iloc[0]
            if float(best_add['gap']) < current_gap:
                selected.add(str(best_add[GROUP_COL]))
                selected_n += int(best_add['n'])
                improved = True
                continue
        selected_df = eligible[eligible[GROUP_COL].isin(selected)].copy()
        if len(selected_df) > 1:
            selected_df['gap'] = (selected_n - selected_df['n'] - target_total).abs()
            best_remove = selected_df.sort_values(['gap', 'n'], ascending=[True, False]).iloc[0]
            if float(best_remove['gap']) < current_gap:
                selected.remove(str(best_remove[GROUP_COL]))
                selected_n -= int(best_remove['n'])
                improved = True
    held_out_l3s = sorted(selected)
    if not held_out_l3s:
        raise ValueError('L3 holdout selection produced an empty test split.')
    return (held_out_l3s, eligible, small)


def _agb_quantile_text(df_part):
    qs = df_part[TARGET].astype(float).quantile([0.1, 0.5, 0.9]).to_dict()
    return f'p10={qs.get(0.1, np.nan):.2f}, p50={qs.get(0.5, np.nan):.2f}, p90={qs.get(0.9, np.nan):.2f}'


def build_split(df):
    df0 = df.reset_index(drop=True).copy()
    df0[GROUP_COL] = df0[GROUP_COL].astype(str)
    held_out_l3s, eligible_l3, small_l3 = _choose_l3_holdout_by_agb(df0)
    test_mask = df0[GROUP_COL].isin(held_out_l3s)
    train_full = df0.loc[~test_mask].copy().reset_index(drop=True)
    test_df = df0.loc[test_mask].copy().reset_index(drop=True)
    train_idx, val_idx, inner_strata = _stratified_sample_split(train_full, VAL_RATIO, _active_split_seed() + 1)
    train_df = train_full.iloc[train_idx].copy().reset_index(drop=True)
    val_df = train_full.iloc[val_idx].copy().reset_index(drop=True)
    train_l3 = set(train_df[GROUP_COL].astype(str))
    val_l3 = set(val_df[GROUP_COL].astype(str))
    test_l3 = set(test_df[GROUP_COL].astype(str))
    overlap_train_test = sorted(train_l3 & test_l3)
    overlap_val_test = sorted(val_l3 & test_l3)
    if overlap_train_test or overlap_val_test:
        raise RuntimeError(f'Invalid split: held-out L3 leaked into modeling data. train/test={overlap_train_test[:5]}, val/test={overlap_val_test[:5]}')
    all_l3s = sorted(df0[GROUP_COL].unique().tolist())
    test_counts = test_df[GROUP_COL].value_counts()
    print('\n[split] L3-group holdout test split, AGB-stratified at L3 level')
    print(f'       split_seed={_active_split_seed()}  train_seed={CFG.seed}')
    print(f'       ratios target: train+val={1.0 - TEST_RATIO:.2f}  test={TEST_RATIO:.2f}; val within modeling={VAL_RATIO:.2f}')
    print(f'       samples: train={len(train_df)}  val={len(val_df)}  test={len(test_df)} (test_ratio={len(test_df) / len(df0):.3f})')
    print(f'       L3: total={len(all_l3s)}  eligible_for_test={len(eligible_l3)}  too_small_kept_in_modeling={len(small_l3)}  held_out_test={len(held_out_l3s)}')
    print(f'       test L3 sample count: min={int(test_counts.min())}  median={float(test_counts.median()):.1f}  max={int(test_counts.max())}  threshold={TEST_MIN_L3_SIZE}')
    print(f'       L3 overlap check: train/test={len(overlap_train_test)}  val/test={len(overlap_val_test)}')
    print(f'       AGB quantiles: train({_agb_quantile_text(train_df)}); val({_agb_quantile_text(val_df)}); test({_agb_quantile_text(test_df)})')
    print(f'       inner val strata={len(np.unique(inner_strata))}')
    print(f'       held-out L3: {held_out_l3s}')
    return (train_df, val_df, test_df, held_out_l3s)


def build_feature_index():
    cols = ALL_FEATURE_COLS
    idx_G = [cols.index(c) for c in GEDI_COLS if c in cols]
    idx_S = [cols.index(c) for c in SPECTRAL_COLS + ROBUST_SPECTRAL_COLS if c in cols]
    idx_E = [cols.index(c) for c in ENV_COLS if c in cols]
    idx_S_robust = [cols.index(c) for c in ROBUST_SPECTRAL_COLS if c in cols]
    return (idx_G, idx_S, idx_E, idx_S_robust)
