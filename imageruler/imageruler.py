"""Imageruler for measuring minimum lengthscales in binary images."""

import enum
from typing import Callable, Optional, Tuple, Union
import warnings
import cv2 as cv
import numpy as np

warnings.simplefilter('always')

# Threshold used for binarization.
_BINARIZATION_THRESHOLD = 0.5

PhysicalSize = Tuple[float, ...]
PixelSize = Tuple[float, ...]
MarginSize = Tuple[Tuple[float, float], ...]
PeriodicAxes = Tuple[int, ...]


@enum.unique
class PaddingMode(enum.Enum):
  EDGE = 'edge'
  SOLID = 'solid'
  VOID = 'void'


def minimum_length_solid(
    array: np.ndarray,
    phys_size: Optional[PhysicalSize] = None,
    periodic_axes: Optional[PeriodicAxes] = None,
    margin_size: Optional[MarginSize] = None,
    pad_mode: PaddingMode = PaddingMode.SOLID,
    warn_cusp: bool = False,
) -> float:
  """Computes the minimum length scale of the solid regions of an image.

  Args:
    array: The image as a 1d or 2d array.
    phys_size: The physical dimensions of the image. Default is None.
    periodic_axes: The axes which are periodic. x is 0 and y is 1. Default is
      None.
    margin_size: The physical dimensions of the image margins. If specified,
      this subregion is excluded from the result. Default is None.
    pad_mode: The padding mode, defaults to solid.
    warn_cusp: Whether to warn about the presence of sharp corners or cusps
      detected in the image. Default is False (no warning).

  Returns:
    The minimum length scale of the solid regions. The units are the same
    as `phys_size`. If `phys_size` is None, the units are in number of
    pixels.
  """

  array, pixel_size, short_pixel_side, short_entire_side = _ruler_initialize(
      array, phys_size, periodic_axes, warn_cusp
  )

  # If all of the elements in the array are the same,
  # the shorter length of the image is considered to
  # be its minimum length scale.
  if len(np.unique(array)) == 1:
    return short_entire_side

  if array.ndim == 1:
    if margin_size is not None:
      array = _trim(array, margin_size, pixel_size)
    solid_min_length, _ = _minimum_length_1d(array)
    return solid_min_length * short_pixel_side

  def _interior_pixel_number(diameter: float, array: np.ndarray) -> bool:
    """Determines whether an image violates a given length scale."""
    return _length_violation_solid(
        array, diameter, pixel_size, margin_size, pad_mode
    ).any()

  min_len, _ = _search(
      (short_pixel_side, short_entire_side),
      min(pixel_size) / 2,
      lambda d: _interior_pixel_number(d, array),
  )

  return min_len


def minimum_length_void(
    array: np.ndarray,
    phys_size: Optional[PhysicalSize] = None,
    periodic_axes: Optional[PeriodicAxes] = None,
    margin_size: Optional[MarginSize] = None,
    pad_mode: PaddingMode = PaddingMode.VOID,
    warn_cusp: bool = False,
) -> float:
  """Computes the minimum length scale of the void regions in an image.

  Args:
    array: The image as a 1d or 2d array.
    phys_size: The physical dimensions of the image. Default is None.
    periodic_axes: The axes which are periodic. x is 0 and y is 1. Default is
      None.
    margin_size: The physical dimensions of the image margins. If specified,
      this subregion is excluded from the result. Default is None.
    pad_mode: The padding mode, defaults to void.
    warn_cusp: Whether to warn about the presence of sharp corners or cusps
      detected in the image. Default is False (no warning).

  Returns:
    The minimum length scale of the void regions. The units are the same
    as `phys_size`. If `phys_size` is None, the units are in number of
    pixels.
  """
  array, _, _, _ = _ruler_initialize(array, phys_size)
  if pad_mode is PaddingMode.SOLID:
    pad_mode = PaddingMode.VOID
  elif pad_mode is PaddingMode.VOID:
    pad_mode = PaddingMode.SOLID
  else:
    pad_mode = PaddingMode.EDGE

  return minimum_length_solid(
      ~array, phys_size, periodic_axes, margin_size, pad_mode, warn_cusp
  )


def minimum_length_solid_void(
    array: np.ndarray,
    phys_size: Optional[PhysicalSize] = None,
    periodic_axes: Optional[PeriodicAxes] = None,
    margin_size: Optional[MarginSize] = None,
    pad_mode: Tuple[PaddingMode, PaddingMode] = (
        PaddingMode.SOLID,
        PaddingMode.VOID,
    ),
    warn_cusp: bool = False,
) -> Tuple[float, float]:
  """Computes the minimum length scale of an image.

  Args:
    array: The image as a 1d or 2d array.
    phys_size: The physical dimensions of the image. Default is None.
    periodic_axes: The axes which are periodic. x is 0 and y is 1. Default is
      None.
    margin_size: The physical dimensions of the image margins. If specified,
      this subregion is excluded from the result. Default is None.
    pad_mode: The padding mode to apply to the solid and void phases.
    warn_cusp: A boolean value that specifies whether to warn about sharp
      corners or cusps. If True, a warning will be given when arr is likely to
      contain sharp corners or cusps. Default is False (no warning).

  Returns:
    The minimum length scale of the solid and void regions in the same unit
    as `phys_size`. If `phys_size` is None, the units are in number of
    pixels.
  """
  return minimum_length_solid(
      array, phys_size, periodic_axes, margin_size, pad_mode[0], warn_cusp
  ), minimum_length_void(
      array, phys_size, periodic_axes, margin_size, pad_mode[1]
  )


def minimum_length(
    array: np.ndarray,
    phys_size: Optional[PhysicalSize] = None,
    periodic_axes: Optional[PeriodicAxes] = None,
    margin_size: Optional[MarginSize] = None,
    pad_mode: Union[PaddingMode, Tuple[PaddingMode, PaddingMode]] = (
        PaddingMode.SOLID,
        PaddingMode.VOID,
    ),
    warn_cusp: bool = False,
) -> float:
  """Computes the minimum length scale of an image.

  This is the smaller of the minimum length scale of the solid and void
  regions. For a 2d image, this is computed using the difference between
  morphological opening and closing. For a 1d image, this is computed using
  a brute-force search.

  Args:
    array: The image as a 1d or 2d array.
    phys_size: The physical dimensions of the image. Default is None.
    periodic_axes: The axes which are periodic. x is 0 and y is 1. Default is
      None.
    margin_size: The physical dimensions of the image margins. If specified,
      this subregion is excluded from the result. Default is None.
    pad_mode: The padding mode to apply to the solid and void phases.
    warn_cusp: A boolean value that specifies whether to warn about sharp
      corners or cusps. If True, a warning will be given when arr is likely to
      contain sharp corners or cusps. Default is no warning (False).

  Returns:
    The minimum length scale of the solid and void regions in the same unit
    as `phys_size`. If `phys_size` is None, the units are in number of
    pixels.
  """
  array, pixel_size, short_pixel_side, short_entire_side = _ruler_initialize(
      array, phys_size, periodic_axes, warn_cusp
  )

  # If all of the elements in the array are the same,
  # the shorter length of the image is considered to
  # be its minimum length scale.
  if len(np.unique(array)) == 1:
    return short_entire_side

  if array.ndim == 1:
    if margin_size is not None:
      array = _trim(array, margin_size, pixel_size)
    solid_min_length, void_min_length = _minimum_length_1d(array)
    return min(solid_min_length, void_min_length) * short_pixel_side

  if isinstance(pad_mode, PaddingMode):
    pad_mode = (pad_mode, pad_mode)

  def _interior_pixel_number(diameter: float, arr: np.ndarray) -> bool:
    """Determines whether an image violates a given length scale."""
    return _length_violation(
        arr, diameter, pixel_size, margin_size, pad_mode
    ).any()

  min_len, _ = _search(
      (short_pixel_side, short_entire_side),
      min(pixel_size) / 2,
      lambda d: _interior_pixel_number(d, array),
  )

  return min_len


def _length_violation_solid(
    array: np.ndarray,
    diameter: float,
    pixel_size: PixelSize,
    margin_size: Optional[Tuple[Tuple[float, float], ...]] = None,
    pad_mode: PaddingMode = PaddingMode.SOLID,
) -> np.ndarray:
  """Identifies the solid subregions which contain length scale violations."""

  open_diff = binary_open(array, diameter, pixel_size, pad_mode) ^ array
  interior_diff = open_diff & _get_interior(array, 'in', pad_mode)
  if margin_size is not None:
    interior_diff = _trim(interior_diff, margin_size, pixel_size)

  return interior_diff


def _length_violation(
    array: np.ndarray,
    diameter: float,
    pixel_size: PixelSize,
    margin_size: Optional[MarginSize] = None,
    pad_mode: Tuple[PaddingMode, PaddingMode] = (
        PaddingMode.SOLID,
        PaddingMode.VOID,
    ),
) -> np.ndarray:
  """Identifies the subregions which contain length scale violations."""

  close_open_diff = binary_open(
      array, diameter, pixel_size, pad_mode[0]
  ) ^ binary_close(array, diameter, pixel_size, pad_mode[1])
  interior_diff = close_open_diff & _get_interior(array, 'both', pad_mode)
  if margin_size is not None:
    interior_diff = _trim(interior_diff, margin_size, pixel_size)

  return interior_diff


def length_violation_solid(
    array: np.ndarray,
    diameter: float,
    phys_size: Optional[PhysicalSize] = None,
    periodic_axes: Optional[PeriodicAxes] = None,
    margin_size: Optional[MarginSize] = None,
    pad_mode: PaddingMode = PaddingMode.SOLID,
) -> np.ndarray:
  """Identifies the solid subregions which contain length scale violations.

  Args:
    array: The image as a 2d array.
    diameter: The diameter of the kernel (or "probe").
    phys_size: The physical dimensions of the image. Default is None.
    periodic_axes: The axes which are periodic. x is 0 and y is 1. Default is
      None.
    margin_size: The physical dimensions of the image margins. If specified,
      this subregion is excluded from the result. Default is None.
    pad_mode: The padding mode, defaults to solid.

  Returns:
      A 2d array of the violations. The array dimensions are the same as the
      original image.
  """

  array, pixel_size, _, _ = _ruler_initialize(array, phys_size, periodic_axes)
  assert array.ndim == 2
  return _length_violation_solid(
      array, diameter, pixel_size, margin_size, pad_mode
  )


def length_violation_void(
    array: np.ndarray,
    diameter: float,
    phys_size: Optional[PhysicalSize] = None,
    periodic_axes: Optional[PeriodicAxes] = None,
    margin_size: Optional[MarginSize] = None,
    pad_mode: str = 'void',
) -> np.ndarray:
  """Identifies the void subregions which contain length scale violations.

  Args:
    array: The image as a 2d array.
    diameter: The diameter of the kernel (or "probe").
    phys_size: The physical dimensions of the image. Default is None.
    periodic_axes: The axes which are periodic. x is 0 and y is 1. Default is
      None.
    margin_size: The physical dimensions of the image margins. If specified,
      this subregion is excluded from the result. Default is None.
    pad_mode: A string that represents the padding mode, which can be 'solid',
      'void', or 'edge'. Default is 'solid'.

  Returns:
      A 2d array of the violations. The array dimensions are the same as the
      original image.
  """

  array, pixel_size, _, _ = _ruler_initialize(array, phys_size, periodic_axes)
  assert array.ndim == 2

  if pad_mode is PaddingMode.SOLID:
    pad_mode = PaddingMode.VOID
  elif pad_mode is PaddingMode.VOID:
    pad_mode = PaddingMode.SOLID
  else:
    pad_mode = PaddingMode.EDGE

  return _length_violation_solid(
      ~array, diameter, pixel_size, margin_size, pad_mode
  )


def length_violation(
    array: np.ndarray,
    diameter: float,
    phys_size: Optional[PhysicalSize] = None,
    periodic_axes: Optional[PeriodicAxes] = None,
    margin_size: Optional[MarginSize] = None,
    pad_mode: Tuple[PaddingMode, PaddingMode] = (
        PaddingMode.SOLID,
        PaddingMode.VOID,
    ),
) -> np.ndarray:
  """Identifies the subregions which contain length scale violations.

  Args:
      array: The image as a 2d array.
      diameter: The diameter of the kernel (or "probe").
      phys_size: The physical dimensions of the image. Default is None.
      periodic_axes: The axes which are periodic. x is 0 and y is 1. Default is
        None.
      margin_size: The physical dimensions of the image margins. If specified,
        this subregion is excluded from the result. Default is None.
      pad_mode: The padding mode, defaults to solid.

  Returns:
      A 2d array of the violations. The array dimensions are the same as the
      original image.
  """

  array, pixel_size, _, _ = _ruler_initialize(array, phys_size, periodic_axes)
  assert array.ndim == 2
  if isinstance(pad_mode, str):
    pad_mode = (pad_mode, pad_mode)

  return _length_violation(array, diameter, pixel_size, margin_size, pad_mode)


def _ruler_initialize(
    array: np.ndarray,
    phys_size: PhysicalSize,
    periodic_axes: Optional[PeriodicAxes] = None,
    warn_cusp: bool = False,
) -> Tuple[np.ndarray, PixelSize, float, float]:
  """Initializes the ruler.

  This function converts the input array to a boolean array without redundant
  dimensions and computes some basic information about the image.

  Args:
      array: An array that represents an image.
      phys_size: A tuple, list, array, or number that represents the physical
        size of the image.
      periodic_axes: A tuple of axes (x, y = 0, 1) treated as periodic (default
        is None: all axes are non-periodic).
      warn_cusp: A boolean value that determines whether to warn about sharp
        corners or cusps. If True, warning will be given when the input 2d image
        is likely to contain sharp corners or cusps; if False, warning will not
        be given.

  Returns:
      A tuple with four elements. The first is a Boolean array obtained by
      squeezing and binarizing the input array, the second is an array that
      contains the pixel size, the third is the length of the shorter side of
      the pixel, and the fourth is the length of the shorter side of the image.

  Raises:
      ValueError: If the physical size `phys_size` does not have the
      expected format or the length of `phys_size` does not match the dimension
      of the input array.
  """

  array = np.squeeze(array)

  if (
      isinstance(phys_size, np.ndarray)
      or isinstance(phys_size, list)
      or isinstance(phys_size, tuple)
  ):
    phys_size = np.squeeze(phys_size)
    phys_size = phys_size[phys_size.nonzero()]  # keep nonzero elements only
  elif isinstance(phys_size, float) or isinstance(phys_size, int):
    phys_size = [phys_size]
  elif phys_size is None:
    phys_size = array.shape
  else:
    raise ValueError('Invalid format of the physical size.')

  assert array.ndim == len(
      phys_size
  ), 'The physical size and the dimension of the input array do not match.'
  assert array.ndim in (
      1,
      2,
  ), 'The current version of imageruler only supports 1d and 2d.'

  short_entire_side = min(phys_size)  # shorter side of the entire design region
  pixel_size = _get_pixel_size(array, phys_size)
  short_pixel_side = min(pixel_size)  # shorter side of a pixel
  array = _binarize(array)  # Boolean array

  if periodic_axes is not None:
    if array.ndim == 2:
      periodic_axes = np.array(periodic_axes)
      reps = (2 if 0 in periodic_axes else 1, 2 if 1 in periodic_axes else 1)
      array = np.tile(array, reps)
      phys_size = np.array(phys_size) * reps
      short_entire_side = min(
          phys_size
      )  # shorter side of the entire design region
    else:  # arr.ndim == 1
      array = np.tile(array, 2)
      short_entire_side *= 2

  if warn_cusp and array.ndim == 2:
    harris = cv.cornerHarris(
        array.astype(np.uint8), blockSize=5, ksize=5, k=0.04
    )
    if np.max(harris) > 5e-10:
      warnings.warn('This image may contain sharp corners or cusps.')

  return array, pixel_size, short_pixel_side, short_entire_side


def _search(
    arg_range: Tuple[float, float],
    arg_threshold: float,
    function: Callable[[float], bool],
) -> Tuple[float, bool]:
  """Binary search.

  Args:
      arg_range: Initial range of the argument under search.
      arg_threshold: Threshold of the argument range, below which the search
        stops.
      function: A function that returns True if the viariable is large enough
        but False if the variable is not large enough.

  Returns:
      A tuple with two elements. The first is a float that represents the search
      result. The second is a Boolean value, which is True if the search indeed
      happens, False if the condition for starting search is not satisfied in
      the beginning.

  Raises:
      RuntimeError: If `function` returns True at a smaller input viariable
      but False at a larger input viariable.
  """

  args = [min(arg_range), (min(arg_range) + max(arg_range)) / 2, max(arg_range)]

  if not function(args[0]) and function(args[2]):
    while abs(args[0] - args[2]) > arg_threshold:
      arg = args[1]
      if not function(arg):
        args[0], args[1] = (
            arg,
            (arg + args[2]) / 2,
        )  # The current value is too small
      else:
        args[1], args[2] = (
            arg + args[0]
        ) / 2, arg  # The current value is still large
    return args[1], True
  elif not function(args[0]) and not function(args[2]):
    return args[2], False
  elif function(args[0]) and function(args[2]):
    return args[0], False
  else:
    raise RuntimeError('The function is not monotonically increasing.')


def _minimum_length_1d(array: np.ndarray) -> Tuple[int, int]:
  """Search the minimum lengths of solid and void segments in a 1d array.

  Args:
    array: A 1d Boolean array.

  Returns:
    A tuple of two integers. The first and second intergers represent the
    numbers of pixels in the shortest solid and void segments, respectively.
  """

  array = np.append(array, ~array[-1])
  solid_lengths, void_lengths = [], []
  counter = 0

  for idx in range(len(array) - 1):
    counter += 1

    if array[idx] != array[idx + 1]:
      if array[idx]:
        solid_lengths.append(counter)
      else:
        void_lengths.append(counter)
      counter = 0

  if len(solid_lengths) > 0:
    solid_min_length = min(solid_lengths)
  else:
    solid_min_length = 0

  if len(void_lengths) > 0:
    void_min_length = min(void_lengths)
  else:
    void_min_length = 0

  return solid_min_length, void_min_length


def _get_interior(array: np.ndarray, direction: str, pad_mode: PaddingMode):
  """Gets inner borders, outer borders, or union of inner and outer borders.

  Args:
    array: A 2d array that represents an image.
    direction: A string that can be "in", "out", or "both" to indicate inner
      borders, outer borders, and union of inner and outer borders.
    pad_mode: The padding mode to use.

  Returns:
      A Boolean array in which all True elements are at and only at borders.

  Raises:
      ValueError: If the option provided to `direction` is not 'in', 'out',
      or 'both'.
  """

  pixel_size = (1,) * array.ndim
  # With this pixel size and diameter, the resulting kernel has the shape of a
  # plus sign.
  diameter = 2.8

  if direction == 'in':  # interior of solid regions
    return binary_erode(array, diameter, pixel_size, pad_mode)
  elif direction == 'out':  # interior of void regions
    return ~binary_dilate(array, diameter, pixel_size, pad_mode)
  elif direction == 'both':  # union of interiors of solid and void regions
    eroded = binary_erode(array, diameter, pixel_size, pad_mode[0])
    dilated = binary_dilate(array, diameter, pixel_size, pad_mode[1])
    return ~dilated | eroded
  else:
    raise ValueError(
        'The direction at the border can only be in, out, or both.'
    )


def _get_pixel_size(array: np.ndarray, phys_size: PhysicalSize) -> PixelSize:
  """Compute the physical size of a single pixel.

  Args:
      array: An array that represents an image.
      phys_size: A tuple that represents the physical size of the image.

  Returns:
      An array of floats. It represents the physical size of a single pixel.
  """
  return tuple(p / s for p, s in zip(phys_size, array.shape))


def _binarize(array: np.ndarray) -> np.ndarray:
  """Binarize the input array according to the threshold.

  Args:
      array: An array that represents an image.

  Returns:
      An Boolean array.
  """

  return array > _BINARIZATION_THRESHOLD * max(array.flatten()) + (
      1 - _BINARIZATION_THRESHOLD
  ) * min(array.flatten())


def _get_kernel(diameter: float, pixel_size: PixelSize) -> np.ndarray:
  """Get the kernel with a given diameter and pixel size.

  Args:
      diameter: A float that represents the diameter of the kernel, which acts
        like a probe.
      pixel_size: A tuple, list, or array that represents the physical size of
        one pixel in the image.

  Returns:
      An array of unsigned integers 0 and 1. It represent the kernel for
      morpological operations.
  """

  pixel_size = np.array(pixel_size)
  se_shape = np.array(np.round(diameter / pixel_size), dtype=int)

  if se_shape[0] <= 2 and se_shape[1] <= 2:
    return np.ones(se_shape, dtype=np.uint8)

  rounded_size = np.round(diameter / pixel_size - 1) * pixel_size

  x_tick = np.linspace(-rounded_size[0] / 2, rounded_size[0] / 2, se_shape[0])
  y_tick = np.linspace(-rounded_size[1] / 2, rounded_size[1] / 2, se_shape[1])

  x, y = np.meshgrid(
      x_tick, y_tick, sparse=True, indexing='ij'
  )  # grid over the entire design region
  structuring_element = x**2 + y**2 <= diameter**2 / 4

  return np.array(structuring_element, dtype=np.uint8)


def binary_open(
    array: np.ndarray,
    diameter: float,
    pixel_size: PixelSize = (1.0, 1.0),
    pad_mode: PaddingMode = PaddingMode.EDGE,
) -> np.ndarray:
  """Applies a binary morphological opening.

  Args:
    array: A binarized 2d array that represents a binary image.
    diameter: A float that represents the diameter of the kernel, which acts
      like a probe.
    pixel_size: A tuple, list, or array that represents the physical size of one
      pixel in the image.
    pad_mode: A string that represents the padding mode, which can be 'solid',
      'void', or 'edge'.

  Returns:
    A Boolean array that represents the outcome of morphological opening.
  """
  kernel = _get_kernel(diameter, pixel_size)
  array = _proper_pad(array, kernel, pad_mode)
  opened = cv.morphologyEx(src=array, kernel=kernel, op=cv.MORPH_OPEN)
  return _proper_unpad(opened, kernel).astype(bool)


def binary_close(
    arr: np.ndarray,
    diameter: float,
    pixel_size: PixelSize = (1.0, 1.0),
    pad_mode: PaddingMode = PaddingMode.EDGE,
) -> np.ndarray:
  """Applies a binary morphological closing.

  Args:
    arr: A binarized 2d array that represents a binary image.
    diameter: A float that represents the diameter of the kernel, which acts
      like a probe.
    pixel_size: A tuple, list, or array that represents the physical size of one
      pixel in the image.
    pad_mode: A string that represents the padding mode, which can be 'solid',
      'void', or 'edge'.

  Returns:
      A Boolean array that represents the outcome of morphological closing.
  """
  kernel = _get_kernel(diameter, pixel_size)
  arr = _proper_pad(arr, kernel, pad_mode)
  closed = cv.morphologyEx(src=arr, kernel=kernel, op=cv.MORPH_CLOSE)
  return _proper_unpad(closed, kernel).astype(bool)


def binary_erode(
    array: np.ndarray,
    diameter: float,
    pixel_size: PixelSize = (1.0, 1.0),
    pad_mode: PaddingMode = PaddingMode.EDGE,
) -> np.ndarray:
  """Applies a binary morphological erosion.

  Args:
    array: A binarized 2d array that represents a binary image.
    diameter: A float that represents the diameter of the kernel, which acts
      like a probe.
    pixel_size: A tuple, list, or array that represents the physical size of one
      pixel in the image.
    pad_mode: A string that represents the padding mode, which can be 'solid',
      'void', or 'edge'.

  Returns:
    A Boolean array that represents the outcome of morphological erosion.
  """

  kernel = _get_kernel(diameter, pixel_size)
  array = _proper_pad(array, kernel, pad_mode)
  eroded = cv.erode(array, kernel)

  return _proper_unpad(eroded, kernel).astype(bool)


def binary_dilate(
    array: np.ndarray,
    diameter: float,
    pixel_size: PixelSize = (1.0, 1.0),
    pad_mode: PaddingMode = PaddingMode.EDGE,
) -> np.ndarray:
  """Applies a binary morphological dilation.

  Args:
    array: A binarized 2d array that represents a binary image.
    diameter: A float that represents the diameter of the kernel, which acts
      like a probe.
    pixel_size: A tuple, list, or array that represents the physical size of one
      pixel in the image.
    pad_mode: A string that represents the padding mode, which can be 'solid',
      'void', or 'edge'.

  Returns:
    A Boolean array that represents the outcome of morphological dilation.
  """

  kernel = _get_kernel(diameter, pixel_size)
  array = _proper_pad(array, kernel, pad_mode)
  dilated = cv.dilate(array, kernel)

  return _proper_unpad(dilated, kernel).astype(bool)


def _proper_pad(
    array: np.ndarray, kernel: np.ndarray, pad_mode: PaddingMode
) -> np.ndarray:
  """Pad the input array properly according to the size of the kernel.

  Args:
    array: A binarized 2d array that represents a binary image.
    kernel: A 2d array that represents the kernel of morphological operations.
    pad_mode: A string that represents the padding mode, which can be 'solid',
      'void', or 'edge'.

  Returns:
    A padded array composed of unsigned integers 0 and 1.
  """

  ((top, bottom), (left, right)) = (
      (kernel.shape[0],) * 2,
      (kernel.shape[1],) * 2,
  )

  if pad_mode is PaddingMode.EDGE:
    return cv.copyMakeBorder(
        array.view(np.uint8),
        top=top,
        bottom=bottom,
        left=left,
        right=right,
        borderType=cv.BORDER_REPLICATE,
    )
  elif pad_mode is PaddingMode.SOLID:
    return cv.copyMakeBorder(
        array.view(np.uint8),
        top=top,
        bottom=bottom,
        left=left,
        right=right,
        borderType=cv.BORDER_CONSTANT,
        value=1,
    )
  elif pad_mode is PaddingMode.VOID:
    return cv.copyMakeBorder(
        array.view(np.uint8),
        top=top,
        bottom=bottom,
        left=left,
        right=right,
        borderType=cv.BORDER_CONSTANT,
        value=0,
    )
  else:
    raise ValueError(f'Unknown padding mode: {pad_mode.name}.')


def _proper_unpad(array: np.ndarray, kernel: np.ndarray) -> np.ndarray:
  """Removes padding according to the size of the kernel.

  The code is copied from Martin F. Schubert's code at
  https://github.com/mfschubert/topology/blob/main/metrics.py

  Args:
    array: A 2d array that has extra padding.
    kernel: A 2d array that represents the kernel of morphological operations.

  Returns:
    A 2d array without padding.
  """

  unpad_width = (
      (
          kernel.shape[0] + (kernel.shape[0] + 1) % 2,
          kernel.shape[0] - (kernel.shape[0] + 1) % 2,
      ),
      (
          kernel.shape[1] + (kernel.shape[1] + 1) % 2,
          kernel.shape[1] - (kernel.shape[1] + 1) % 2,
      ),
  )

  slices = tuple(
      [
          slice(pad_lo, dim - pad_hi)
          for (pad_lo, pad_hi), dim in zip(unpad_width, array.shape)
      ]
  )
  return array[slices]


def _trim(
    array: np.ndarray, margin_size: MarginSize, pixel_size: PixelSize
) -> np.ndarray:
  """Trims margins from an array.

  Args:
    array: The image as a 1d or 2d array.
    margin_size: The physical dimensions of the image margins. If specified,
      this subregion is excluded from the result. No default.
    pixel_size: The physical size of one pixel in the image. No default.

  Returns:
    A subarray of the input array.
  """

  array = np.squeeze(array)
  arr_dim = array.ndim
  margin_size = abs(np.reshape(margin_size, (-1, 2)))
  margin_dim = len(margin_size)

  if margin_dim > arr_dim:
    raise ValueError(
        'The number of rows of margin_size should not '
        'exceed the dimension of the input array.'
    )

  pixel_size = np.asarray(pixel_size)
  margin_number = np.array(margin_size) / pixel_size[
      0 : len(margin_size)
  ].reshape(len(margin_size), 1)
  margin_number = np.round(margin_number).astype(
      int
  )  # numbers of pixels of marginal regions

  if (
      np.array(array.shape)[0:margin_dim] - np.sum(margin_number, axis=1) < 2
  ).all():
    raise ValueError(
        'The design region is too narrow or contains '
        'margins which are too wide.'
    )

  if margin_dim == 1:
    return array[margin_number[0][0] : -margin_number[0][1]]
  elif margin_dim == 2:
    return array[
        margin_number[0][0] : -margin_number[0][1],
        margin_number[1][0] : -margin_number[1][1],
    ]
  else:
    raise ValueError('The input array has too many dimensions.')
