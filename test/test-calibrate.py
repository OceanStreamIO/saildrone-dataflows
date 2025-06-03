import numpy as np
import echopype as ep
from echopype.calibrate import compute_Sv

from saildrone.calibrate import apply_calibration
import pytest


# @pytest.mark.skip(reason="Temporarily")
def test_apply_calibration_defaults_short_pulse():
    cal_file = 'calibration/calibration_values.xlsx'
    raw_file = 'test/data/SD_TPOS2023_v03-Phase0-D20230601-T015958-0.raw'
    ed = ep.open_raw(raw_file=str(raw_file), sonar_model="EK80")
    result = apply_calibration(ed, cal_file)

    assert result is ed

    pl_bins = ed["Vendor_specific"].pulse_length_bin.shape[0]
    gain = ed["Vendor_specific"].gain_correction.values

    assert gain.shape == (2, pl_bins)

    assert np.allclose(gain[0], 19.24)
    assert np.allclose(gain[1], 18.5015)

    # Beamwidths
    along = ed.beam.beamwidth_twoway_alongship.values
    athw = ed.beam.beamwidth_twoway_athwartship.values
    assert along.shape == (2,)
    assert athw.shape == (2,)
    assert np.isclose(along[0], 17.36)
    assert np.isclose(along[1], 16.4942)

    assert np.isclose(athw[0], 16.91)
    assert np.isclose(athw[1], 16.0478)

    off_a = ed.beam.angle_offset_alongship.values
    off_b = ed.beam.angle_offset_athwartship.values
    assert off_a.shape == (2,)
    assert off_b.shape == (2,)

    assert np.isclose(off_a[0], 0.32)
    assert np.isclose(off_a[1], 0.3196)
    assert np.isclose(off_b[0], -0.27)
    assert np.isclose(off_b[1], -0.4255)

    sa_bins = ed["Vendor_specific"].sa_correction.shape[1]
    sa = ed["Vendor_specific"].sa_correction.values

    assert sa.shape == (2, sa_bins)
    assert np.allclose(sa[0], -0.07)
    assert np.allclose(sa[1], -0.2592)


# @pytest.mark.skip(reason="Temporarily")
def test_apply_calibration_defaults_long_pulse():
    cal_file = 'calibration/calibration_values.xlsx'
    raw_file = 'test/data/SD_TPOS2023_v03-Phase0-D20230813-T094204-0.raw'
    ed = ep.open_raw(raw_file=str(raw_file), sonar_model="EK80")
    result = apply_calibration(ed, cal_file)

    assert result is ed

    pl_bins = ed["Vendor_specific"].pulse_length_bin.shape[0]
    gain = ed["Vendor_specific"].gain_correction.values
    assert gain.shape[0] == 2 and gain.shape[1] == pl_bins
    assert np.allclose(gain, 18.5)

    # Beamwidths
    bw_along = ed.beam.beamwidth_twoway_alongship.values
    bw_athw = ed.beam.beamwidth_twoway_athwartship.values
    assert np.allclose(np.asarray(bw_along, dtype=float), 16.49)
    assert np.allclose(np.asarray(bw_athw, dtype=float), 16.05)

    # Angle offsets
    off_a = ed.beam.angle_offset_alongship.values
    off_b = ed.beam.angle_offset_athwartship.values
    assert np.allclose(np.asarray(off_a, dtype=float), 0.32)
    assert np.allclose(np.asarray(off_b, dtype=float), -0.43)

    # Sa_corr
    sa = ed["Vendor_specific"].sa_correction.values
    assert np.allclose(np.asarray(sa, dtype=float), -0.26)


# @pytest.mark.skip(reason="Temporarily")
def test_apply_calibration_invalid_str_excel():
    cal_file = 'calibration/calibration_values.xlsx'
    raw_file = 'test/data/SD_TPOS2023_v03-Phase0-D20231007-T235959-0.raw'
    ed = ep.open_raw(raw_file=str(raw_file), sonar_model="EK80")
    ed = apply_calibration(ed, cal_file)

    sv_dataset = compute_Sv(ed, waveform_mode='CW', encode_mode='complex')
    sv_dataset = sv_dataset.dropna(dim='ping_time', how='all', subset=['Sv'])

    assert sv_dataset is not None

