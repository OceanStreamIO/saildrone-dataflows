import numpy as np

from saildrone.utils import load_values_from_xlsx


def apply_calibration(echodata, file_path='./calibration/calibration_values.xlsx'):
    """
    Apply calibration to the dataset.

    Args:
        echodata: The dataset to apply calibration to.
        file_path: The path to the calibration file.

    Returns:
        The calibrated dataset.
    """
    cal = load_values_from_xlsx(file_path)
    repeat_times = echodata["Vendor_specific"].pulse_length_bin.shape[0]

    channels_len = len(echodata["Platform"].channel)

    # Gain correction
    value1_array = np.tile(clean_float(cal.iloc[1, 2]), repeat_times)
    value2_array = np.tile(clean_float(cal.iloc[1, 3]), repeat_times)
    gain_correction = np.vstack([value1_array, value2_array])

    # Beamwidths

    # along beamwidth
    alongship_beamwidth_value = clean_float(cal.iloc[2, 2])
    alongship_beamwidth = np.array([alongship_beamwidth_value])
    alongship_beamwidth = np.resize(alongship_beamwidth, echodata.beam.beamwidth_twoway_alongship.shape)

    # athwarth beam width
    athwarth_beamwidth_value = clean_float(cal.iloc[3, 2])
    athwarth_beamwidth = np.array([athwarth_beamwidth_value])
    athwarth_beamwidth = np.resize(athwarth_beamwidth, echodata.beam.beamwidth_twoway_athwartship.shape)

    # Alongship offset
    alongship_offset_value = clean_float(cal.iloc[4, 2])
    alongship_offset = np.array([alongship_offset_value])
    alongship_offset = np.resize(alongship_offset, echodata.beam.angle_offset_alongship.shape)

    # Athwartship offset
    athwarth_offset_value = clean_float(cal.iloc[5, 2])
    athwarth_offset = np.array([athwarth_offset_value])
    athwarth_offset = np.resize(athwarth_offset, echodata.beam.angle_offset_athwartship.shape)

    # Sa correction
    value1_array = np.tile(clean_float(cal.iloc[6, 2]), 5)
    value2_array = np.tile(clean_float(cal.iloc[6, 3]), 5)
    sa_correction = np.vstack([value1_array, value2_array])

    # apply calibration values
    echodata["Vendor_specific"].gain_correction.values = gain_correction
    echodata.beam.beamwidth_twoway_alongship.values = alongship_beamwidth
    echodata.beam.beamwidth_twoway_athwartship.values = athwarth_beamwidth
    echodata.beam.angle_offset_alongship.values = alongship_offset
    echodata.beam.angle_offset_athwartship.values = athwarth_offset
    echodata["Vendor_specific"].sa_correction.values = sa_correction

    return echodata


def clean_float(x):
    """Convert string to float safely, handling non-breaking spaces and whitespace."""
    return float(str(x).replace('\xa0', '').strip())