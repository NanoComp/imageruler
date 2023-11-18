"""Imageruler."""

__version__ = '1.0'

import imageruler.imageruler
import imageruler.regular_shapes
from .imageruler import (
    KernelShape,
    PaddingMode,
    binary_close,
    binary_dilate,
    binary_erode,
    binary_open,
    get_kernel,
    length_violation,
    length_violation_solid,
    length_violation_void,
    minimum_length,
    minimum_length_solid,
    minimum_length_solid_void,
    minimum_length_void,
)
