from .pathdata import map_file_paths, load_local_files, get_metadata_for_files
from .fix_calibration import load_values_from_xlsx

__all__ = [
    "frequency_nominal_to_channel",
    "load_values_from_xlsx",
    "map_file_paths",
    "load_local_files",
    "get_metadata_for_files"
]


def frequency_nominal_to_channel(source_Sv, frequency_nominal: int):
    """
    Given a value for a nominal frequency, returns the channel associated with it
    """
    channels = source_Sv["frequency_nominal"].coords["channel"].values
    freqs = source_Sv["frequency_nominal"].values
    chan = channels[freqs == frequency_nominal]
    assert len(chan) == 1, "Frequency not uniquely identified"
    channel = chan[0]
    return channel
