"""
Utility functions for V-Loan research framework.
"""

import os
import json
import logging
import random
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

def get_project_root() -> Path:
    """Return the absolute path to the project root directory."""
    return Path(__file__).resolve().parent.parent

def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration YAML file."""
    if config_path is None:
        config_path = str(get_project_root() / "configs" / "config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_models_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load models configuration YAML file."""
    if config_path is None:
        config_path = str(get_project_root() / "configs" / "models.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_experiment_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load experiment configuration YAML file."""
    if config_path is None:
        config_path = str(get_project_root() / "configs" / "experiment.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def ensure_directories(config: Optional[Dict[str, Any]] = None) -> None:
    """Create all required project directories if they do not exist."""
    root = get_project_root()
    dirs = [
        root / "data" / "raw",
        root / "data" / "processed",
        root / "models",
        root / "results" / "raw",
        root / "results" / "tables",
        root / "results" / "figures",
        root / "results" / "logs",
        root / "results" / "summaries",
        root / "results" / "verification_cards",
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def setup_logger(name: str = "vloan", log_file: Optional[str] = None) -> logging.Logger:
    """Configure a standard logger for experiments."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        formatter = logging.Formatter(
            "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(log_file, encoding="utf-8")
            fh.setFormatter(formatter)
            logger.addHandler(fh)
    return logger

def save_table(df: pd.DataFrame, base_filename: str, output_dir: Optional[str] = None) -> Dict[str, str]:
    """Save a DataFrame both as CSV and as formatted Markdown."""
    root = get_project_root()
    if output_dir is None:
        target_dir = root / "results" / "tables"
    else:
        target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    
    if base_filename.endswith(".csv"):
        base_filename = base_filename[:-4]
    elif base_filename.endswith(".md"):
        base_filename = base_filename[:-3]
        
    csv_path = target_dir / f"{base_filename}.csv"
    md_path = target_dir / f"{base_filename}.md"
    
    df.to_csv(csv_path, index=False)
    
    try:
        md_content = df.to_markdown(index=False)
    except (ImportError, Exception):
        # Fallback tabulate or simple markdown formatter
        md_content = df.to_string(index=False)
        
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# Table: {base_filename.replace('_', ' ').title()}\n\n")
        f.write(md_content)
        f.write("\n")
        
    return {"csv": str(csv_path), "md": str(md_path)}

def save_json(data: Any, filepath: str) -> None:
    """Save data to JSON file with indentation and float formatting."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    def convert(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient="records")
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        return str(obj)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=convert)
