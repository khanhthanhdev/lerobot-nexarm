"""LeRobot plugin for DEM. Importing it registers the `dem` policy type."""

from .configuration_dem import DEMConfig
from .processor_dem import make_dem_pre_post_processors

__all__ = ["DEMConfig", "make_dem_pre_post_processors"]
