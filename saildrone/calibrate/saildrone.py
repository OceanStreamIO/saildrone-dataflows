import numpy as np
import pandas as pd


SHORT_MS = 1.024e-3        # 1.024 ms  → “short”
LONG_MS  = 2.048e-3        # 2.048 ms  → “long”


def load_values_from_xlsx(file_path):
    cal = pd.read_excel(file_path, sheet_name="Sheet1", header=1, usecols="A:D")

    # Correct the spelling and set column names
    cal.columns = ["Variable", "38k short pulse", "38k long pulse", "200k short pulse"]

    return cal


def _channel_pulse_mode(echodata, atol=5e-6):
    """
    Return a list the same length as channel dimension, containing
    'short' or 'long' for each channel, decided from
    transmit_duration_nominal.

    If a channel switches pulse length mid-file, the statistical *mode*
    of all pings is used.
    """
    td = echodata["Sonar/Beam_group1"].transmit_duration_nominal  # (ping_time, channel)
    # Use modal value along ping_time
    first = td.isel(ping_time=0).values.astype(float)

    td_vals = td.values.astype(float)
    for ch, col in enumerate(td_vals):
        if not np.allclose(col, first[ch], atol=atol):
            uniq = np.unique(np.round(col, 6))
            if len(uniq) == 1:
                first[ch] = uniq[0]
            else:
                raise ValueError(
                    f"Channel {ch} has multiple pulse durations {uniq}; "
                    "cannot decide short vs long automatically."
                )

    pulse_mode = []
    for d in first:
        if np.isclose(d, 1.024e-3, atol=atol):
            pulse_mode.append("short")
        elif np.isclose(d, 2.048e-3, atol=atol) or d > 1.5e-3:
            pulse_mode.append("long")
        else:
            raise ValueError(f"Unknown pulse duration {d:.6f}s")

    return pulse_mode


def calibrate(echodata, cal_file):
    """
    Apply calibration from `cal` (a pandas.DataFrame) to an Echopype `echodata` Dataset.
    cal must have columns:
      [Variable, '38k short pulse', '38k long pulse', '200k short pulse']
    and rows in this order:
      0: pulse_length (ms),
      1: Gain (dB),
      2: beamwidth_alongship (°),
      3: beamwidth_athwartship (°),
      4: angle_offset_alongship (°),
      5: angle_offset_athwartship (°),
      6: Sa_corr (dB).
    """
    cal = load_values_from_xlsx(cal_file)

    gain_var = echodata["Vendor_specific"].gain_correction
    n_chans, n_pl_bins = gain_var.shape
    n_sa_bins = echodata["Vendor_specific"].sa_correction.shape[1]

    freqs = echodata["Sonar/Beam_group1"].frequency_nominal.values.tolist()
    modes = _channel_pulse_mode(echodata)

    cols = []
    for f, mode in zip(freqs, modes):
        if np.isclose(f, 38_000.0):
            cols.append("38k short pulse" if mode == "short" else "38k long pulse")
        elif np.isclose(f, 200_000.0):
            cols.append("200k short pulse")  # Saildrone never uses long at 200 kHz
        else:
            raise ValueError(f"Unsupported frequency {f}")

    if len(cols) == 1 and n_chans > 1:
        cols = cols * n_chans

    def tile_row(row_idx, tgt_bins):  # helper
        return np.vstack([
            np.full(tgt_bins, cal.at[row_idx, c]) for c in cols
        ])

    # Gain (row 1) –  (n_chans × n_pl_bins)
    echodata["Vendor_specific"].gain_correction.values = tile_row(1, n_pl_bins)

    # Beamwidths & angle-offsets – 1-D
    bw_shape = echodata.beam.beamwidth_twoway_alongship.shape[0]
    ao_shape = echodata.beam.angle_offset_alongship.shape[0]

    echodata.beam.beamwidth_twoway_alongship.values = [
        cal.at[2, c] for c in cols[:bw_shape]
    ]
    echodata.beam.beamwidth_twoway_athwartship.values = [
        cal.at[3, c] for c in cols[:bw_shape]
    ]

    echodata.beam.angle_offset_alongship.values = [
        cal.at[4, c] for c in cols[:ao_shape]
    ]
    echodata.beam.angle_offset_athwartship.values = [
        cal.at[5, c] for c in cols[:ao_shape]
    ]

    # Sa_corr (row 6) – (n_chans × n_sa_bins)
    echodata["Vendor_specific"].sa_correction.values = tile_row(6, n_sa_bins)

    return echodata
