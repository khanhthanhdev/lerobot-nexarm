"""LeRobot plugin for B-spline Policy. Importing it registers the `bspline` policy type."""

from .configuration_bspline import BsplineConfig
from .processor_bspline import make_bspline_pre_post_processors

__all__ = ["BsplineConfig", "make_bspline_pre_post_processors"]
