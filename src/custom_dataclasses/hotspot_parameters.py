from dataclasses import dataclass


# This class represents the parameters of the hotspot generation
# Each dataset will have a pandas dataframe where each row will have this information
@dataclass
class HotspotParameters:
    add_hotspot_flag: bool
    max_intensity_multiplier: float
    hotspot_size_x: int
    hotspot_size_y: int
    sigma: float
    hotspot_place_x: int
    hotspot_place_y: int

@dataclass
class HotspotParametersRanges:
    hotspot_probability: float
    max_intensity_multiplier_range: tuple
    hotspot_size_x_range: tuple
    hotspot_size_y_range: tuple
    sigma_range: tuple