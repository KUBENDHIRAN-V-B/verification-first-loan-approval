"""
Statistical Validation Module for V-Loan research framework.
Computes non-parametric bootstrap confidence intervals, repeated seed statistics, and hypothesis tests.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Any, List, Tuple, Optional

from src.utils import save_table, setup_logger

logger = setup_logger("statistics")

def compute_bootstrap_ci(
    data: np.ndarray,
    stat_func=np.mean,
    n_bootstraps: int = 2000,
    confidence_level: float = 0.95,
    random_seed: int = 42
) -> Tuple[float, float, float]:
    """
    Compute empirical bootstrap confidence interval.
    
    Returns:
        point_estimate (float): Sample statistic
        ci_lower (float): Lower bound
        ci_upper (float): Upper bound
    """
    arr = np.asarray(data)
    if len(arr) == 0:
        return 0.0, 0.0, 0.0
        
    point_est = float(stat_func(arr))
    rng = np.random.RandomState(random_seed)
    n = len(arr)
    
    if stat_func == np.mean or stat_func is np.mean:
        samples = rng.choice(arr, size=(n_bootstraps, n), replace=True)
        boot_stats = np.mean(samples, axis=1)
    else:
        boot_stats = np.empty(n_bootstraps)
        for i in range(n_bootstraps):
            sample = rng.choice(arr, size=n, replace=True)
            boot_stats[i] = stat_func(sample)
        
    alpha = (1.0 - confidence_level) / 2.0
    ci_lower = float(np.percentile(boot_stats, alpha * 100))
    ci_upper = float(np.percentile(boot_stats, (1.0 - alpha) * 100))
    
    return point_est, ci_lower, ci_upper

def aggregate_seed_results(
    seed_data_list: List[Dict[str, Any]]
) -> pd.DataFrame:
    """
    Aggregate metrics across multiple random seeds (e.g. seeds 42, 52, 62, 72, 82).
    Calculates Mean, Standard Deviation, Min, Max, and 95% CI across runs.
    """
    df_seeds = pd.DataFrame(seed_data_list)
    
    summary_records = []
    metric_cols = [c for c in df_seeds.columns if c not in ["seed", "dataset", "model", "method"]]
    
    group_cols = [c for c in ["dataset", "model", "method"] if c in df_seeds.columns]
    
    if group_cols:
        for grp_keys, grp_df in df_seeds.groupby(group_cols):
            if not isinstance(grp_keys, tuple):
                grp_keys = (grp_keys,)
            grp_dict = dict(zip(group_cols, grp_keys))
            
            for m in metric_cols:
                vals = grp_df[m].dropna().values.astype(float)
                if len(vals) == 0:
                    continue
                mean_val = float(np.mean(vals))
                std_val = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
                _, ci_low, ci_high = compute_bootstrap_ci(vals)
                
                summary_records.append({
                    **grp_dict,
                    "metric": m,
                    "n_seeds": len(vals),
                    "mean": mean_val,
                    "std": std_val,
                    "min": float(np.min(vals)),
                    "max": float(np.max(vals)),
                    "ci_95_str": f"[{ci_low:.4f}, {ci_high:.4f}]"
                })
    else:
        for m in metric_cols:
            vals = df_seeds[m].dropna().values.astype(float)
            if len(vals) == 0:
                continue
            mean_val = float(np.mean(vals))
            std_val = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            _, ci_low, ci_high = compute_bootstrap_ci(vals)
            
            summary_records.append({
                "metric": m,
                "n_seeds": len(vals),
                "mean": mean_val,
                "std": std_val,
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
                "ci_95_str": f"[{ci_low:.4f}, {ci_high:.4f}]"
            })
            
    df_agg = pd.DataFrame(summary_records)
    save_table(df_agg, "repeated_seed_results")
    return df_agg
