import subprocess
import pytest
from pathlib import Path

from prefect_flows.export_processed_data import export_processed


@pytest.fixture
def set_prefect_server_url():
    """
    Fixture to set Prefect server URL using the CLI command.
    """
    url = "http://192.168.0.41:4200/api"
    result = subprocess.run(
        ["prefect", "config", "set", f"PREFECT_API_URL={url}"],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        pytest.fail(f"Failed to set Prefect API URL: {result.stderr}")
    print(f"Prefect API URL set to {url}")
    yield


def test_export_processed_data(set_prefect_server_url):
    coordinates = [
        [-157.75156, 21.981161],
        [-157.74477, 21.981161],
        [-157.74477, 22.000805],
        [-157.75156, 22.000805],
        [-157.75156, 21.981161]
    ]

    cruise_id = "SD_TPOS2023_v03"

    short_pulse_ds, long_pulse_ds = export_processed(cruise_id=cruise_id, coordinates=coordinates)
    print('short_pulse_ds', short_pulse_ds)
    print('long_pulse_ds', long_pulse_ds)


