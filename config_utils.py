import json
from pathlib import Path


DEFAULT_CONFIG = {
    "train": {
        "episodes": 40000,
        "log_every_episodes": 20,
    },
    "train_optimized": {
        "episodes": 40000,
        "batch_size": 32,
        "update_frequency": 4,
        "log_every_episodes": 20,
        "agent_overrides": {
            "batch_size": 256,
            "learn_every": 4,
            "updates_per_step": 1,
        },
    },
    "agent": {
        "batch_size": 128,
        "burnin": 2000,
        "learn_every": 1,
        "updates_per_step": 4,
        "sync_every": 10_000,
        "gamma": 0.9,
        "learning_rate": 0.00025,
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def load_config(config_path: str | Path = "config.json") -> dict:
    path = Path(config_path)
    config = json.loads(json.dumps(DEFAULT_CONFIG))  # deep copy
    if path.exists():
        try:
            with path.open("r", encoding="utf-8") as f:
                user_cfg = json.load(f)
            _deep_merge(config, user_cfg)
        except Exception:
            # If config is malformed, fall back to defaults silently
            pass
    return config


