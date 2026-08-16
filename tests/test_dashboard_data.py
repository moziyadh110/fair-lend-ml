import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from monitoring.dashboard import risk_tier_breakdown


def test_risk_tier_breakdown_counts_each_tier():
    df = pd.DataFrame({"risk_tier": ["low", "low", "medium", "high", "very_high", "high"]})
    counts = risk_tier_breakdown(df)
    assert counts["low"] == 2
    assert counts["high"] == 2
    assert counts["medium"] == 1
    assert counts["very_high"] == 1
