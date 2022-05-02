from pydantic import BaseModel


class RailwayAttritionRecord(BaseModel):
    Track_age_yrs: float = 22.0
    Track_degradation: str = "TRUE"
    Load_cycles_daily: float = 177.0
    Load_amplitude_avg_N: float = 2400.0
    Balast_thickness_inmm: float = 385.0
    track_confining_pressure_kPa:  float = 189.0
    curve_radius_mm: float = 497.0
    tilting_wagon: str = "FALSE"
    fracture_strength_level:  float = 5
    fracture_strength_value:  float = 316
    track_fouling:  str = "TRUE"
    track_misalignment:  str = "FALSE"
    rail_material_inhomogenous:  str = "FALSE"
    humidty_percent:  float = 95.0
    corrosion_index_iso:  float = 10
    speed_mph:  float = 240.0
    Accumulated_tonnage_kgpmeter:  float = 59.0
    rail_fall_in_mm:  float = 71.0
    track_quality_index:  float = 5.0
    inspection_interval_yrs:  float = 6.0

    
