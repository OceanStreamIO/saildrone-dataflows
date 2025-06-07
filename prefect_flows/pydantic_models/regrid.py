from pydantic import BaseModel, Field


class BaseOptions(BaseModel):
    def get(self, key, default_value=None):
        return getattr(self, key, default_value)


class NASC_Compute_Options(BaseOptions):
    range_bin: str = Field(default="10m",
                           description='bin size along depth in meters (m)')
    dist_bin: str = Field(default="0.5nmi",
                          description="bin size along distance in nautical miles (nmi).")
    closed: str = Field(default="left",
                        description="Which side of bin interval is closed, either 'left' or 'right'.")


class MVBS_Compute_Options(BaseOptions):
    range_var: str = Field(default="depth",
                           description='The variable to use for range binning. Must be either "echo_range" or "depth".')
    range_bin: str = Field(default="20m",
                           description="bin size along echo_range or depth in meters.")
    ping_time_bin: str = Field(default="5s",
                               description="bin size along 'ping_time' in seconds (s).")
    closed: str = Field(default="left",
                        description="Which side of bin interval is closed, either 'left' or 'right'.")
