from __future__ import annotations

from copy import deepcopy
from typing import Mapping, Any, Dict, Optional
from pydantic import BaseModel, Field


class DenoiseOptions(BaseModel):
    def get(self, key, default_value=None):
        return getattr(self, key, default_value)


class ReprocessingOptions(DenoiseOptions):
    reprocess: bool = Field(default=False,
                            description='Load and reprocess the raw data for files that '
                                        'have been already marked as processed.')
    skip_processed: bool = Field(default=False,
                                 description='Skip processing and only generate outputs'
                                             ' (e.g., echograms, NetCDF files); requires previously processed files.')


# class MaskImpulseNoise(DenoiseOptions):
#     depth_bin: int = Field(default=10, description="Downsampling bin size along vertical range variable (`range_var`) in meters.")
#     num_side_pings: int = Field(default=2, description="Number of side pings to look at for the two-side comparison.")
#     threshold: float = Field(default=10, description="Impulse noise threshold value (in dB) for the two-side comparison.")
#     range_var: str = Field(default='depth', description="Vertical Axis Range Variable. Can be either \"depth\" or \"echo_range\".")


class MaskAttenuatedSignal(BaseModel):
    """
    Options for masking attenuated signals based on side-ping comparisons.

    Parameters *per acoustic frequency* as a **simple dict-of-dicts**. n\n
    Valid options are:\n
    - **range_coord**          : str   – vertical coordinate name ("echo_range"/"depth")\n
    - **upper_limit_sl**       : int   – upper limit of the scattering layer range (m, default 400)\n
    - **lower_limit_sl**       : int   – lower limit of the scattering layer range (m, default 550)\n
    - **num_side_pings**       : int – number of side pings to include in the comparison (default 10)\n
    - **threshold**            : float – threshold for the side-ping comparison (dB, default -3.0)\n

    Parameters that are omitted will either be filled in with the values from the `"38000"` (38 kHz frequency), the first frequency in the dictionary, or default values.\n
    \n\n
    Example:

    ```
    \n
    {\n
      "38000":  {\n
        "upper_limit_sl": 400,\n
        "lower_limit_sl": 550,\n
        "num_side_pings": 10,\n
        "range_coord": "depth",\n
        "threshold": -3.0\n
       },\n
      "200000": {\n
        "upper_limit_sl": 20,\n
        "lower_limit_sl": 120,\n
        "num_side_pings": 15,\n
        "range_coord": "depth",\n
        "threshold": -2.5\n
      }\n
    }\n
    ```
    """
    frequencies: Dict[str, dict] = Field(
        ...,
        description="Parameter dictionary for each frequency (Hz)."
    )


class TransientNoiseMask(DenoiseOptions):
    """
    Options for masking transient noise based on side-ping comparisons.

    Parameters *per acoustic frequency* as a **simple dict-of-dicts**. n\n
    Valid options are:\n
    - **range_coord**          : str   – vertical coordinate name ("echo_range"/"depth")\n
    - **ping_window**          : int   – half-width (pings) for the rolling window (default 5)\n
    - **range_window**         : int   – half-width (samples) for the rolling window (default 3)\n
    - **threshold**            : float – dB above block statistic (default 6)\n
    - **exclude_above**        : float – minimum range to apply (m, default 250)\n
    - **percentile**           : int   – percentile for block statistic (default 15)\n

    Parameters that are omitted will either be filled in with the values from the `"38000"` (38 kHz frequency), the first frequency in the dictionary, or default values.\n
    \n\n
    Example:

    ```
    \n
    {\n
      "38000":  {\n
        "ping_window": 5,\n
        "range_window": 3,\n
        "threshold": 12.0,\n
        "percentile": 15,\n
        "exclude_above": 2.0\n
       },\n
      "200000": {\n
        "ping_window": 5,\n
        "range_window": 3,\n
        "threshold": 12.0,\n
        "percentile": 15,\n
        "exclude_above": 2.0\n
      }\n
    }\n
    ```
    """
    frequencies: Dict[str, dict] = Field(
        ...,
        description="Parameter dictionary for each frequency (Hz)."
    )


class RemoveBackgroundNoise(BaseModel):
    """
    Options for removing noise by using estimates of background noise from mean calibrated power of a collection of pings.

    Parameters *per acoustic frequency* as a **simple dict-of-dicts**. n\n
    Valid options are:\n
    - **range_coord**          : str   – vertical coordinate name ("echo_range"/"depth")\n
    - **range_window**         : int   – # of range-samples in blocking window (default 20)\n
    - **ping_window**          : int   – # of pings in blocking window (default 50)\n
    - **background_noise_max** : float – lowest allowed background Sv (default -125.0)\n
    - **SNR_threshold**        : float – minimum SNR (dB, default 3.0)\n
    - **sound_absorption**     : float – sound absorption coefficient (dB m⁻¹, default 0.001)\n
    - **minimal_linear**       : float – floor for linear power before log10 (default 1e-30)\n

    Parameters that are omitted will either be filled in with the values from the `"38000"` (38 kHz frequency), the first frequency in the dictionary, or default values.\n
    \n\n
    Example:

    ```
    \n
    {\n
      "38000":  {\n
        "range_window": 5,\n
        "ping_window": 20,\n
        "background_noise_max": -125.0,\n
        "range_coord": "depth",\n
        "SNR_threshold": 3.0,\n
        "sound_absorption": 9e-6\n
       },\n
      "200000": {\n
        "sound_absorption": 3.8e-4\n
      }\n
    }\n
    ```
    """

    frequencies: Dict[str, dict] = Field(
        ...,
        description="Parameter dictionary for each frequency (Hz)."
    )


class MaskImpulseNoise(BaseModel):
    """
    Parameters *per acoustic frequency* as a **simple dict-of-dicts**.\n\n

    Valid options are:\n
    - **range_coord**          : str   – vertical coordinate name ("echo_range"/"depth")\n
    - **vertical_bin_size**    : int   – # samples in vertical mean (≥1, default 1 → none)\n
    - **ping_lags**            : tuple – side-ping offsets, e.g. (1, 2, 3)\n
    - **threshold_db**         : float – Sv diff threshold (dB, default 10)\n
    - **exclude_shallow_above**: float – optional range cut-off (m) to skip processing\n\n

    Parameters that are omitted will either be filled in with the values from the `"38000"` (38 kHz frequency), the first frequency in the dictionary, or default values.\n
\n\n
    Example:

    ```
    \n
    {\n
      "38000":  {\n
        "ping_lags": [1, 2],\n
        "threshold": 10.0,\n
        "range_coord": "depth",\n
        "vertical_bin_size": "2m",\n
        "exclude_shallow_above": 5.0\n
       },\n
      "200000": {\n
        "ping_lags": [1],\n
        "threshold": 10.0,\n
        "range_coord": "depth",\n
        "vertical_bin_size": "2m",\n
        "exclude_shallow_above": 4.0\n
      }\n
    }\n
    ```
    """

    frequencies: Dict[str, dict] = Field(
        ...,
        description="Parameter dictionary for each frequency (Hz)."
    )


def fill_missing_frequency_params(
    freq_params: Mapping[str, Any] | Any  # accept Pydantic/BaseModel too
) -> Dict[str, Optional[Dict]]:
    """
    Build a new per-frequency param map where:
      - Keys are coerced to strings.
      - None means the frequency is disabled and is left as None.
      - If the chosen template frequency has short/long pulse subdicts, the
        *short_pulse* subdict is used as the default template base.
      - For a frequency that itself has short/long pulse subdicts:
          short_pulse := template_base ⊕ user.short_pulse
          long_pulse  := short_pulse  ⊕ user.long_pulse   (inherits SP defaults)
      - For a frequency with a flat dict: merged := template_base ⊕ user.
      - If input is empty or all disabled, return as-is (no merge).
    """

    # --- accept Pydantic models or plain mappings
    if hasattr(freq_params, "frequencies"):            # our PerFrequency model
        raw_map = freq_params.frequencies
    elif hasattr(freq_params, "model_dump"):           # any BaseModel
        raw_map = freq_params.model_dump()
    else:
        raw_map = dict(freq_params)

    if not raw_map:
        return {}

    # --- normalize: str keys; None stays None; mappings to plain dict
    norm: Dict[str, Optional[Dict]] = {}
    for k, v in raw_map.items():
        sk = str(k)
        if v is None:
            norm[sk] = None
        elif isinstance(v, Mapping):
            norm[sk] = dict(v)
        else:
            norm[sk] = {}  # treat other truthy values as empty enabled config

    # Enabled entries (non-None) are candidates for templating
    enabled_keys = [k for k, v in norm.items() if v is not None]
    if not enabled_keys:
        # all disabled -> return unchanged
        return norm

    # --- choose template key (prefer 38 kHz if enabled)
    template_key = "38000" if "38000" in enabled_keys else enabled_keys[0]
    template_val = deepcopy(norm[template_key]) or {}

    # Template base: if template has short/long, use its short_pulse as base
    if isinstance(template_val, Mapping) and (
        "short_pulse" in template_val or "long_pulse" in template_val
    ):
        template_base = deepcopy(template_val.get("short_pulse", {}))
    else:
        template_base = deepcopy(template_val)

    # --- merge per frequency
    filled: Dict[str, Optional[Dict]] = {}
    for fk, fval in norm.items():
        if fval is None:
            filled[fk] = None
            continue

        if "short_pulse" in fval or "long_pulse" in fval:
            # Nested pulse configs
            user_sp = dict(fval.get("short_pulse", {}))
            user_lp = dict(fval.get("long_pulse", {}))

            sp = deepcopy(template_base)
            sp.update(user_sp)

            lp = deepcopy(sp)          # inherit SP as default
            lp.update(user_lp)

            filled[fk] = {"short_pulse": sp, "long_pulse": lp}
        else:
            # Flat config
            merged = deepcopy(template_base)
            merged.update(fval)
            filled[fk] = merged

    return filled
