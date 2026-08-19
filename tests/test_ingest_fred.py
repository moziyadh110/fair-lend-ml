import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data_engineering.ingest_fred import fetch_series


def test_fetch_series_parses_observations_and_drops_missing_values():
    fake_response = MagicMock()
    fake_response.json.return_value = {
        "observations": [
            {"date": "2020-01-01", "value": "3.5"},
            {"date": "2020-02-01", "value": "."},  # FRED's missing-value marker
            {"date": "2020-03-01", "value": "4.4"},
        ]
    }
    fake_response.raise_for_status = MagicMock()

    with patch("data_engineering.ingest_fred.requests.get", return_value=fake_response) as mock_get:
        df = fetch_series("UNRATE", api_key="fake_key")

    assert len(df) == 2  # the "." row was dropped
    assert list(df["value"]) == [3.5, 4.4]
    assert (df["series_id"] == "UNRATE").all()
    mock_get.assert_called_once()
    assert mock_get.call_args.kwargs["params"]["series_id"] == "UNRATE"
