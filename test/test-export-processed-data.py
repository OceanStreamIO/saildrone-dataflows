import subprocess
import pytest
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
        [
            -154.9754093230061,
            18.0249775317156
        ],
        [
            -154.93628125789695,
            18.0249775317156
        ],
        [
            -154.93628125789695,
            17.978556131284293
        ],
        [
            -154.9754093230061,
            17.978556131284293
        ],
        [
            -154.9754093230061,
            18.0249775317156
        ]
    ]

    coordinates2 = [
        [
            -155.47070647789164,
            17.80055053577573
        ],
        [
            -154.6663012654895,
            17.80055053577573
        ],
        [
            -154.6663012654895,
            17.347733975514398
        ],
        [
            -155.47070647789164,
            17.347733975514398
        ],
        [
            -155.47070647789164,
            17.80055053577573
        ]
    ]

    cruise_id = "SD_TPOS2023_v03"
    filters = {
        "mask_transient_noise": {
            "func": "nanmean",
            "depth_bin": "10m",
            "num_side_pings": "25",
            "exclude_above": "250.0m",
            "transient_noise_threshold": "12.0dB",
            "range_var": "depth",
            "use_index_binning": False
        },
        "mask_impulse_noise": {
            "depth_bin": 5,
            "num_side_pings": 2,
            "impulse_noise_threshold": 10,
            "range_var": "depth",
            "use_index_binning": False
        }
    }

    filters2 = {
        "mask_impulse_noise": {
            "depth_bin": 5,
            "num_side_pings": 2,
            "impulse_noise_threshold": 10,
            "range_var": "depth",
            "use_index_binning": False
        },
        "mask_attenuated_signal": {
            "upper_limit_sl": 400,
            "lower_limit_sl": 500,
            "num_side_pings": 15,
            "attenuation_signal_threshold": 10,
            "start": 0,
            "range_var": "depth"
        },
        "remove_background_noise": {
            "ping_num": "10",
            "range_sample_num": "50",
            "background_noise_max": "",
            "SNR_threshold": "3.0dB"
        }
    }

    result = export_processed(cruise_id=cruise_id, coordinates=coordinates, depth_offset=0.1, filters=filters2,
                              container_name="export-test")
    print('result:', result)


