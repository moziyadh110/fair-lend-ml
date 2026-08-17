"""Evaluate current metric values against configs/monitoring_config.yaml's
threshold rules, and notify a real channel when something fires — computing
a breach and never telling anyone was a real gap (evaluate_alerts alone
just returns a list nobody would see in production).
"""
import os
from pathlib import Path

import requests
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


def format_alert_message(alert: dict) -> str:
    direction = "below" if alert["direction"] == "below" else "above"
    return (f":rotating_light: *{alert['name']}* — `{alert['metric']}` = {alert['current_value']:.4f} "
            f"({direction} threshold {alert['threshold']}). Action: `{alert['action']}`")


def send_alerts(fired_alerts: list[dict], webhook_url: str | None = None) -> int:
    """POSTs each fired alert to a Slack incoming webhook (SLACK_WEBHOOK_URL
    in .env). Uses `requests` directly rather than a Slack SDK — one POST to
    one webhook URL doesn't need a client library. No-ops (returns 0) if no
    webhook is configured, so this is safe to call even in environments that
    haven't set one up yet. Returns the count of alerts successfully sent."""
    webhook_url = webhook_url or os.environ.get("SLACK_WEBHOOK_URL")
    if not webhook_url:
        return 0

    sent = 0
    for alert in fired_alerts:
        resp = requests.post(webhook_url, json={"text": format_alert_message(alert)}, timeout=10)
        resp.raise_for_status()
        sent += 1
    return sent
