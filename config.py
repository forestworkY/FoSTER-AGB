from dataclasses import dataclass


TARGET = 'agbd_live_dry_mg_ha'


GROUP_COL = 'US_L3CODE'


GEDI_COLS = ['RH25', 'RH50', 'RH75', 'RH95', 'RH98', 'RH100', 'Thick_Upper100', 'Thick_Upper98', 'Thick_Mid', 'Thick_Lower', 'Thick_Ground', 'Norm_RH25', 'Norm_RH50', 'Norm_RH75', 'Rel_Crown_Depth', 'Shape_Complexity']


SPECTRAL_COLS = ['w33_Blue_mean', 'w33_Green_mean', 'w33_Red_mean', 'w33_NIR_mean', 'w33_SWIR1_mean', 'w33_SWIR2_mean', 'NDVI', 'NDMI', 'NBR', 'RDVI', 'OSAVI', 'MCARI1_MTVI2', 'SIPI', 'NBR2']


ROBUST_SPECTRAL_COLS = ['NDMI_over_NDVI', 'NBR_over_NDVI', 'NDMI_minus_NBR', 'SWIR1_over_NIR', 'SWIR2_over_NIR', 'SWIR1_over_SWIR2', 'Red_over_NIR', 'bp_Blue', 'bp_Green', 'bp_Red', 'bp_NIR', 'bp_SWIR1', 'bp_SWIR2']


TERRAIN_COLS = ['elev_point_m', 'elev_mean_m', 'elev_median_m', 'elev_min_m', 'elev_max_m', 'elev_std_m']


QUALITY_COLS = ['w33_valid_frac', 'w33_obs_count_mean', 'w33_obs_count_min', 'w33_obs_count_max', 'w33_cirrus_ratio_mean', 'w33_clear_ratio_mean', 'w33_cloud_ratio_mean', 'w33_shadow_ratio_mean', 'w33_snow_ratio_mean', 'n_candidate_images', 'n_images_used', 'n_qualified_images']


ENV_COLS = TERRAIN_COLS + QUALITY_COLS


ALL_FEATURE_COLS = GEDI_COLS + SPECTRAL_COLS + ROBUST_SPECTRAL_COLS + ENV_COLS


@dataclass
class ModelConfig:
    hidden_dim: int = 128
    z_dim: int = 64
    zc_dim: int = 64
    dropout: float = 0.1
    t_d_model: int = 64
    t_n_heads: int = 4
    t_n_layers: int = 2
    t_ffn_mult: int = 2
    m3_state_summary_cols: tuple = ("elev_point_m", "NDMI", "Rel_Crown_Depth", "RH95", "w33_clear_ratio_mean")
    seed: int = 42
    batch_size: int = 512
    clip_lo: float = 0.01
    clip_hi: float = 0.99
    lr: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 200
    patience: int = 20
    m3_lr: float = 1e-3
    m3_weight_decay: float = 1e-2
    m3_patience: int = 30
    m3_warmup_epochs: int = 10
    m3_eta_min: float = 1e-5
    m3_state_aux_weight: float = 0.0
