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


class MaskImpulseNoise(DenoiseOptions):
    depth_bin: int = Field(default=10, description="Downsampling bin size along vertical range variable (`range_var`) in meters.")
    num_side_pings: int = Field(default=2, description="Number of side pings to look at for the two-side comparison.")
    threshold: float = Field(default=10, description="Impulse noise threshold value (in dB) for the two-side comparison.")
    range_var: str = Field(default='depth', description="Vertical Axis Range Variable. Can be either \"depth\" or \"echo_range\".")


class MaskAttenuatedSignal(DenoiseOptions):
    upper_limit_sl: int = Field(default=180, description="Upper limit of deep scattering layer line (m).")
    lower_limit_sl: int = Field(default=300, description="Lower limit of deep scattering layer line (m).")
    num_side_pings: int = Field(default=15, description="Number of preceding & subsequent pings defining the block.")
    threshold: float = Field(default=10, description="Attenuation signal threshold value (dB) for the ping-block comparison.")
    range_var: str = Field(default='range_sample', description="Vertical Axis Range Variable. Can be either `depth` or `echo_range`.")


class TransientNoiseMask(DenoiseOptions):
    operation: str = Field(default='nanmedian', description="Pooling function used in the pooled Sv aggregation, either 'nanmedian' or 'nanmean'.")
    depth_bin: int = Field(default=10, description="Bin size for depth calculation.")
    num_side_pings: int = Field(default=25, description="Number of side pings to include.")
    exclude_above: float = Field(default=250.0, description="Exclude data above this depth value.")
    threshold: float = Field(default=12.0, description="Transient noise threshold value (in dB) for the pooling comparison.")
    range_var: str = Field(default='depth', description="Vertical Range Variable. Can be either `depth` or `echo_range`.")


class RemoveBackgroundNoise(DenoiseOptions):
    ping_num: int = Field(default=5, description="Number of pings to obtain noise estimates")
    range_sample_num: int = Field(default=30, description="Number of range samples to consider.")
    background_noise_max: float = Field(default=-125, description="Maximum allowable background noise estimation (in dB).")
    SNR_threshold: float = Field(default=3.0, description="Signal-to-noise ratio threshold for background noise removal.")
