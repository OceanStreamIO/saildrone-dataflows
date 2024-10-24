import numpy as np

from saildrone.utils import load_values_from_xlsx


def apply_calibration(echodata, file_path='./utils/calibration_values.xlsx'):
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

    # gain
    value1_array = np.tile(cal.iloc[1, 2], repeat_times)
    value2_array = np.tile(cal.iloc[1, 3], repeat_times)
    gain_correction = np.vstack([value1_array, value2_array])

    # along beamwidth
    alongship_beamwidth = np.array([cal.iloc[2, 2]])
    alongship_beamwidth = np.resize(alongship_beamwidth, echodata.beam.beamwidth_twoway_alongship.shape)

    # athwarth beam width
    athwarth_beamwidth = np.array([cal.iloc[3, 2]])
    athwarth_beamwidth = np.resize(athwarth_beamwidth, echodata.beam.beamwidth_twoway_athwartship.shape)

    # Alongship offset
    alongship_offset = np.array([cal.iloc[4, 2]])
    alongship_offset = np.resize(alongship_offset, echodata.beam.angle_offset_alongship.shape)

    # Athwartship offset
    athwarth_offset = np.array([cal.iloc[5, 2]])
    athwarth_offset = np.resize(athwarth_offset, echodata.beam.angle_offset_athwartship.shape)

    # Sa_corr
    value1_array = np.tile(cal.iloc[6, 2], 5)
    value2_array = np.tile(cal.iloc[6, 3], 5)
    sa_correction = np.vstack([value1_array, value2_array])
    sa_correction = np.vectorize(lambda x: float(str(x).replace('\xa0', '').strip()))(sa_correction)

    # apply calibration values
    echodata["Vendor_specific"].gain_correction.values = gain_correction
    echodata.beam.beamwidth_twoway_alongship.values = alongship_beamwidth
    echodata.beam.beamwidth_twoway_athwartship.values = athwarth_beamwidth
    echodata.beam.angle_offset_alongship.values = alongship_offset
    echodata.beam.angle_offset_athwartship.values = athwarth_offset
    echodata["Vendor_specific"].sa_correction.values = sa_correction

    return echodata
