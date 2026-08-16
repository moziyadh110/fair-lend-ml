"""Evaluate current metric values against configs/monitoring_config.yaml's
threshold rules and return which alerts fired.
"""
from pathlib import Path

import yaml

CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "monitoring_config.yaml"


def load_rules(config_path: Path = CONFIG_PATH) -> list[dict]:
    return yaml.safe_load(config_path.read_text())["alerts"]


def evaluate_alerts(metric_values: dict, rules: list[dict] | None = None) -> list[dict]:
    """metric_values: {metric_name: current_value}. Returns fired alerts with their action."""
    rules = rules if rules is not None else load_rules()
    fired = []
    for rule in rules:
        value = metric_values.get(rule["metric"])
        if value is None:
            continue
        breached = value < rule["threshold"] if rule["direction"] == "below" else value > rule["threshold"]
        if breached:
            fired.append({**rule, "current_value": value})
    return fired
