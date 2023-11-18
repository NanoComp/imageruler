"""Imageruler for measuring minimum lengthscales in binary images."""

import enum
from typing import Callable, Optional, Tuple, Union
import warnings
import cv2 as cv
import numpy as np

warnings.simplefilter('always')

# Threshold used for binarization.
_BINARIZATION_THRESHOLD = 0.5

_PLUS_KERNEL = np.array(
    [
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0],
    ],
    dtype=np.uint8,
)

Array = np.ndarray
PhysicalSize = Tuple[float, ...]
PixelSize = Tuple[float, ...]
MarginSize = Tuple[Tuple[float, float], ...]
PeriodicAxes = Tuple[int, ...]
Padding = Tuple[Tuple[int, int], Tuple[int, int]]


@enum.unique
class Direction(enum.Enum):
  IN = 'in'
  OUT = 'out'
  BOTH = 'both'


@enum.unique
class PaddingMode(enum.Enum):
  EDGE = 'edge'
  SOLID = 'solid'
  VOID = 'void'


@enum.unique
class KernelShape(enum.Enum):
  CIRCLE = 'circle'
  RECTANGLE = 'rectangle'


def minimum_length_solid(
    array: Array,
    phys_size: Optional[PhysicalSize] = None,
    periodic_axes: Optional[PeriodicAxes] = None,
    margin_size: Optional[MarginSize] = None,
    pad_mode: PaddingMode = PaddingMode.SOLID,
    kernel_shape: KernelShape = KernelShape.CIRCLE,
    warn_cusp: bool = False,
) -> float:
  """Computes the minimum length scale of the solid regions of an image.

  Args:
    array: The 1D or 2D binarized image array.
    phys_size: The extent of the image in physical units. If not specified,
      pixel units are used for the returned length scale.
    periodic_axes: The axes which are periodic. x is 0 and y is 1. If not
      specified, no axes are treated as periodic.
    margin_size: The physical dimensions of the image margins. If specified,
      this subregion is excluded from the length scale measurement. Default is
      no margins.
    pad_mode: The padding mode to use at the image boundaries. Defaults to
      solid.
    kernel_shape: The kernel shape to use for probing the length scale. Defaults
      to a circular kernel.
    warn_cusp: Whether to warn about the presence of sharp corners or cusps
      detected in the image. Default is False (no warning).

  Returns:
    The minimum length scale of the solid regions in the input image. The units
    are the same as those of `phys_size`. If `phys_size` is not specified, pixel
    units are used.
  """

  array, pixel_size, short_pixel_side, short_entire_side = _initialize_ruler(
      array, phys_size, periodic_axes, warn_cusp
  )

  # If all of the elements in the array are the same,
  # the shorter length of the image is considered to
  # be its minimum length scale.
  if len(np.unique(array)) == 1:
    return short_entire_side

  if array.ndim == 1:
    if margin_size is not None:
      array = _trim_margins(array, margin_size, pixel_size)
    solid_min_length, _ = _minimum_length_1d(array)
    return solid_min_length * short_pixel_side

  def _interior_pixel_number(diameter: float, array: Array) -> bool:
    """Determines whether an image violates a given length scale."""
    return _length_violation_solid(
        array=array,
        diameter=diameter,
        pixel_size=pixel_size,
        margin_size=margin_size,
        pad_mode=pad_mode,
        kernel_shape=kernel_shape,
    ).any()

  min_len, _ = _search(
      (short_pixel_side, short_entire_side),
      min(pixel_size) / 2,
      lambda d: _interior_pixel_number(d, array),
  )

  return min_len


def minimum_length_void(
    array: Array,
    phys_size: Optional[PhysicalSize] = None,
    periodic_axes: Optional[PeriodicAxes] = None,
    margin_size: Optional[MarginSize] = None,
    pad_mode: PaddingMode = PaddingMode.VOID,
    kernel_shape: KernelShape = KernelShape.CIRCLE,
    warn_cusp: bool = False,
) -> float:
  """Computes the minimum length scale of the void regions of an image.

  Args:
    array: The 1D or 2D binarized image array.
    phys_size: The extent of the image in physical units. If not specified,
      pixel units are used for the returned length scale.
    periodic_axes: The axes which are periodic. x is 0 and y is 1. If not
      specified, no axes are treated as periodic.
    margin_size: The physical dimensions of the image margins. If specified,
      this subregion is excluded from the length scale measurement. Default is
      no margins.
    pad_mode: The padding mode to use at the image boundaries. Defaults to void.
    kernel_shape: The kernel shape to use for probing the length scale. Defaults
      to a circular kernel.
    warn_cusp: Whether to warn about the presence of sharp corners or cusps
      detected in the image. Default is False (no warning).

  Returns:
    The minimum length scale of the void regions in the input image. The units
    are the same as those of `phys_size`. If `phys_size` is not specified, pixel
    units are used.
  """
  array, _, _, _ = _initialize_ruler(array, phys_size)
  if pad_mode is PaddingMode.SOLID:
    pad_mode = PaddingMode.VOID
  elif pad_mode is PaddingMode.VOID:
    pad_mode = PaddingMode.SOLID
  else:
    pad_mode = PaddingMode.EDGE

  return minimum_length_solid(
      array=~array,
      phys_size=phys_size,
      periodic_axes=periodic_axes,
      margin_size=margin_size,
      pad_mode=pad_mode,
      kernel_shape=kernel_shape,
      warn_cusp=warn_cusp,
  )


def minimum_length_solid_void(
    array: Array,
    phys_size: Optional[PhysicalSize] = None,
    periodic_axes: Optional[PeriodicAxes] = None,
    margin_size: Optional[MarginSize] = None,
    pad_mode: Tuple[PaddingMode, PaddingMode] = (
        PaddingMode.SOLID,
        PaddingMode.VOID,
    ),
    kernel_shape: KernelShape = KernelShape.CIRCLE,
    warn_cusp: bool = False,
) -> Tuple[float, float]:
  """Computes the minimum length scale for both phases of an image.

  Args:
    array: The 1D or 2D binarized image array.
    phys_size: The extent of the image in physical units. If not specified,
      pixel units are used for the returned length scale.
    periodic_axes: The axes which are periodic. x is 0 and y is 1. If not
      specified, no axes are treated as periodic.
    margin_size: The physical dimensions of the image margins. If specified,
      this subregion is excluded from the length scale measurement. Default is
      no margins.
    pad_mode: The padding mode to use at the image boundaries for each phase.
      Defaults to solid for the solid phase and void for the void phase.
    kernel_shape: The kernel shape to use for probing the length scale. Defaults
      to a circular kernel.
    warn_cusp: Whether to warn about the presence of sharp corners or cusps
      detected in the image. Default is False (no warning).

  Returns:
    The minimum length scale of the solid and void phases in the input image.
    The units are the same as those of `phys_size`. If `phys_size` is not
    specified, pixel units are used.
  """
  return minimum_length_solid(
      array=array,
      phys_size=phys_size,
      periodic_axes=periodic_axes,
      margin_size=margin_size,
      pad_mode=pad_mode[0],
      kernel_shape=kernel_shape,
      warn_cusp=warn_cusp,
  ), minimum_length_void(
      array=array,
      phys_size=phys_size,
      periodic_axes=periodic_axes,
      margin_size=margin_size,
      pad_mode=pad_mode[1],
      kernel_shape=kernel_shape,
      warn_cusp=warn_cusp,
  )


def minimum_length(
    array: Array,
    phys_size: Optional[PhysicalSize] = None,
    periodic_axes: Optional[PeriodicAxes] = None,
    margin_size: Optional[MarginSize] = None,
    pad_mode: Union[PaddingMode, Tuple[PaddingMode, PaddingMode]] = (
        PaddingMode.SOLID,
        PaddingMode.VOID,
    ),
    kernel_shape: KernelShape = KernelShape.CIRCLE,
    warn_cusp: bool = False,
) -> float:
  """Computes the minimum length scale of an image.

  Th returned value is the smaller of the minimum length scale of the solid and
  void regions. For a 2D image, this is computed using the difference between
  morphological opening and closing. For a 1d image, this is computed using
  a brute-force search.

  Args:
    array: The 1D or 2D binarized image array.
    phys_size: The extent of the image in physical units. If not specified,
      pixel units are used for the returned length scale.
    periodic_axes: The axes which are periodic. x is 0 and y is 1. If not
      specified, no axes are treated as periodic.
    margin_size: The physical dimensions of the image margins. If specified,
      this subregion is excluded from the length scale measurement. Default is
      no margins.
    pad_mode: The padding mode to use at the image boundaries for each phase.
      Defaults to solid for the solid phase and void for the void phase.
    kernel_shape: The kernel shape to use for probing the length scale. Defaults
      to a circular kernel.
    warn_cusp: Whether to warn about the presence of sharp corners or cusps
      detected in the image. Default is False (no warning).

  Returns:
    The minimum length scale of the solid and void phases in the input image.
    The units are the same as those of `phys_size`. If `phys_size` is not
    specified, pixel units are used.
  """
  array, pixel_size, short_pixel_side, short_entire_side = _initialize_ruler(
      array, phys_size, periodic_axes, warn_cusp
  )

  # If all of the elements in the array are the same,
  # the shorter length of the image is considered to
  # be its minimum length scale.
  if len(np.unique(array)) == 1:
    return short_entire_side

  if array.ndim == 1:
    if margin_size is not None:
      array = _trim_margins(array, margin_size, pixel_size)
    solid_min_length, void_min_length = _minimum_length_1d(array)
    return min(solid_min_length, void_min_length) * short_pixel_side

  if isinstance(pad_mode, PaddingMode):
    pad_mode = (pad_mode, pad_mode)

  def _interior_pixel_number(diameter: float, arr: Array) -> bool:
    """Determines whether an image violates a given length scale."""
    return _length_violation(
        array=arr,
        diameter=diameter,
        pixel_size=pixel_size,
        margin_size=margin_size,
        pad_mode=pad_mode,
        kernel_shape=kernel_shape,
    ).any()

  min_len, _ = _search(
      (short_pixel_side, short_entire_side),
      min(pixel_size) / 2,
      lambda d: _interior_pixel_number(d, array),
  )

  return min_len


def _length_violation_solid(
    array: Array,
    diameter: float,
    pixel_size: PixelSize,
    margin_size: Optional[Tuple[Tuple[float, float], ...]] = None,
    pad_mode: PaddingMode = PaddingMode.SOLID,
    kernel_shape: KernelShape = KernelShape.CIRCLE,
) -> Array:
  """Identifies the solid subregions which contain length scale violations."""
  kernel = get_kernel(diameter, pixel_size, kernel_shape)
  open_diff = binary_open(array, kernel, pad_mode) ^ array
  interior_diff = open_diff & _get_interior(array, Direction.IN, pad_mode)
  if margin_size is not None:
    interior_diff = _trim_margins(interior_diff, margin_size, pixel_size)

  return interior_diff


def _length_violation(
    array: Array,
    diameter: float,
    pixel_size: PixelSize,
    margin_size: Optional[MarginSize] = None,
    pad_mode: Tuple[PaddingMode, PaddingMode] = (
        PaddingMode.SOLID,
        PaddingMode.VOID,
    ),
    kernel_shape: KernelShape = KernelShape.CIRCLE,
) -> Array:
  """Identifies the subregions which contain length scale violations."""
  kernel = get_kernel(diameter, pixel_size, kernel_shape)
  close_open_diff = binary_open(array, kernel, pad_mode[0]) ^ binary_close(
      array, kernel, pad_mode[1]
  )
  interior_diff = close_open_diff & _get_interior(
      array, Direction.BOTH, pad_mode
  )
  if margin_size is not None:
    interior_diff = _trim_margins(interior_diff, margin_size, pixel_size)
  return interior_diff


def length_violation_solid(
    array: Array,
    diameter: float,
    phys_size: Optional[PhysicalSize] = None,
    periodic_axes: Optional[PeriodicAxes] = None,
    margin_size: Optional[MarginSize] = None,
    pad_mode: PaddingMode = PaddingMode.SOLID,
    kernel_shape: KernelShape = KernelShape.CIRCLE,
) -> Array:
  """Identifies the solid subregions which contain length scale violations.

  Args:
    array: The 2D binarized image array.
    diameter: The diameter of the kernel for detecting length scale violations.
    phys_size: The extent of the image in physical units. If not specified,
      pixel units are used for the returned length scale.
    periodic_axes: The axes which are periodic. x is 0 and y is 1. If not
      specified, no axes are treated as periodic.
    margin_size: The physical dimensions of the image margins. If specified,
      this subregion is excluded from the length scale measurement. Default is
      no margins.
    pad_mode: The padding mode to use at the image boundaries. Defaults to void.
    kernel_shape: The kernel shape to use for probing the length scale. Defaults
      to a circular kernel.

  Returns:
    An array indicating the solid length scale violations in the input image.
  """

  array, pixel_size, _, _ = _initialize_ruler(array, phys_size, periodic_axes)
  assert array.ndim == 2
  return _length_violation_solid(
      array=array,
      diameter=diameter,
      pixel_size=pixel_size,
      margin_size=margin_size,
      pad_mode=pad_mode,
      kernel_shape=kernel_shape,
  )


def length_violation_void(
    array: Array,
    diameter: float,
    phys_size: Optional[PhysicalSize] = None,
    periodic_axes: Optional[PeriodicAxes] = None,
    margin_size: Optional[MarginSize] = None,
    pad_mode: str = 'void',
    kernel_shape: KernelShape = KernelShape.CIRCLE,
) -> Array:
  """Identifies the void subregions which contain length scale violations.

  Args:
    array: The 2D binarized image array.
    diameter: The diameter of the kernel for detecting length scale violations.
    phys_size: The extent of the image in physical units. If not specified,
      pixel units are used for the returned length scale.
    periodic_axes: The axes which are periodic. x is 0 and y is 1. If not
      specified, no axes are treated as periodic.
    margin_size: The physical dimensions of the image margins. If specified,
      this subregion is excluded from the length scale measurement. Default is
      no margins.
    pad_mode: The padding mode to use at the image boundaries. Defaults to void.
    kernel_shape: The kernel shape to use for probing the length scale. Defaults
      to a circular kernel.

  Returns:
    An array indicating the void length scale violations in the input image.
  """

  array, pixel_size, _, _ = _initialize_ruler(array, phys_size, periodic_axes)
  assert array.ndim == 2

  if pad_mode is PaddingMode.SOLID:
    pad_mode = PaddingMode.VOID
  elif pad_mode is PaddingMode.VOID:
    pad_mode = PaddingMode.SOLID
  else:
    pad_mode = PaddingMode.EDGE

  return _length_violation_solid(
      array=~array,
      diameter=diameter,
      pixel_size=pixel_size,
      margin_size=margin_size,
      pad_mode=pad_mode,
      kernel_shape=kernel_shape,
  )


def length_violation(
    array: Array,
    diameter: float,
    phys_size: Optional[PhysicalSize] = None,
    periodic_axes: Optional[PeriodicAxes] = None,
    margin_size: Optional[MarginSize] = None,
    pad_mode: Tuple[PaddingMode, PaddingMode] = (
        PaddingMode.SOLID,
        PaddingMode.VOID,
    ),
    kernel_shape: KernelShape = KernelShape.CIRCLE,
) -> Array:
  """Identifies the subregions which contain length scale violations.

  Args:
    array: The 2D binarized image array.
    diameter: The diameter of the kernel for detecting length scale violations.
    phys_size: The extent of the image in physical units. If not specified,
      pixel units are used for the returned length scale.
    periodic_axes: The axes which are periodic. x is 0 and y is 1. If not
      specified, no axes are treated as periodic.
    margin_size: The physical dimensions of the image margins. If specified,
      this subregion is excluded from the length scale measurement. Default is
      no margins.
    pad_mode: The padding mode to use at the image boundaries for each phase.
      Defaults to solid for the solid phase and void for the void phase.
    kernel_shape: The kernel shape to use for probing the length scale. Defaults
      to a circular kernel.

  Returns:
    An array indicating the length scale violations in the input image.
  """

  array, pixel_size, _, _ = _initialize_ruler(array, phys_size, periodic_axes)
  assert array.ndim == 2
  if isinstance(pad_mode, PaddingMode):
    pad_mode = (pad_mode, pad_mode)

  return _length_violation(
      array=array,
      diameter=diameter,
      pixel_size=pixel_size,
      margin_size=margin_size,
      pad_mode=pad_mode,
      kernel_shape=kernel_shape,
  )


def _initialize_ruler(
    array: Array,
    phys_size: PhysicalSize,
    periodic_axes: Optional[PeriodicAxes] = None,
    warn_cusp: bool = False,
) -> Tuple[Array, PixelSize, float, float]:
  """Initializes the ruler.

  This function converts the input array to a boolean array without redundant
  dimensions and computes some basic information about the image.

  Args:
    array: The 2D binarized image array.
    phys_size: The extent of the image in physical units. If not specified,
      pixel units are used for the returned length scale.
    periodic_axes: The axes which are periodic. x is 0 and y is 1. If not
      specified, no axes are treated as periodic.
    warn_cusp: Whether to warn about the presence of sharp corners or cusps
      detected in the image. Default is False (no warning).

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
      isinstance(phys_size, Array)
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
  """Performs a binary search.

  Args:
    arg_range: Initial range of the argument under search.
    arg_threshold: Threshold of the argument range, below which the search
      stops.
    function: A function that returns True if the viariable is large enough but
      False if the variable is not large enough.

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


def _minimum_length_1d(array: Array) -> Tuple[int, int]:
  """Searches the minimum lengths of solid and void segments in a 1d array.

  Args:
    array: A 1D boolean array.

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


def _get_interior(
    array: Array,
    direction: Direction,
    pad_mode: Union[PaddingMode, Tuple[PaddingMode, PaddingMode]],
) -> Array:
  """Gets inner borders, outer borders, or union of inner and outer borders.

  Args:
    array: A 2D array that represents an image.
    direction: The direction indicating inner borders, outer borders, or a union
      of inner and outer borders.
    pad_mode: The padding mode to use.

  Returns:
    A Boolean array in which all True elements are at and only at borders.
  """
  if direction is Direction.IN:
    return binary_erode(array, _PLUS_KERNEL, pad_mode)
  elif direction is Direction.OUT:
    return ~binary_dilate(array, _PLUS_KERNEL, pad_mode)
  elif direction is Direction.BOTH:
    eroded = binary_erode(array, _PLUS_KERNEL, pad_mode[0])
    dilated = binary_dilate(array, _PLUS_KERNEL, pad_mode[1])
    return ~dilated | eroded
  else:
    raise ValueError(f'Unknown direction: {direction.name}.')


def _get_pixel_size(array: Array, phys_size: PhysicalSize) -> PixelSize:
  """Gets the pixel size from an array and physical size."""
  return tuple(p / s for p, s in zip(phys_size, array.shape))


def _binarize(array: Array) -> Array:
  """Binarizes the input array."""

  return array > _BINARIZATION_THRESHOLD * max(array.flatten()) + (
      1 - _BINARIZATION_THRESHOLD
  ) * min(array.flatten())


def get_kernel(
    diameter: float,
    pixel_size: PixelSize = (1.0, 1.0),
    kernel_shape: KernelShape = KernelShape.CIRCLE,
) -> Array:
  """Gets the kernel with a given diameter and pixel size.

  Args:
    diameter: A float that represents the diameter of the kernel, which acts
      like a probe.
    pixel_size: A tuple, list, or array that represents the physical size of one
      pixel in the image.
    kernel_shape: The kernel shape to use.

  Returns:
    An array of unsigned integers 0 and 1 representing the kernel for
    morpological operations.
  """

  pixel_size = np.asarray(pixel_size)
  shape = np.array(np.round(diameter / pixel_size), dtype=int)

  if shape[0] <= 2 and shape[1] <= 2:
    return np.ones(shape, dtype=np.uint8)

  rounded_size = np.round(diameter / pixel_size - 1) * pixel_size

  if kernel_shape is KernelShape.CIRCLE:
    x_tick = np.linspace(-rounded_size[0] / 2, rounded_size[0] / 2, shape[0])
    y_tick = np.linspace(-rounded_size[1] / 2, rounded_size[1] / 2, shape[1])
    x, y = np.meshgrid(x_tick, y_tick, sparse=True, indexing='ij')
    return np.array(x**2 + y**2 <= diameter**2 / 4, dtype=np.uint8)
  elif kernel_shape is KernelShape.RECTANGLE:
    return np.ones(shape, dtype=np.uint8)
  else:
    raise ValueError(f'Unknown kernel shape: {kernel_shape.name}.')


def _get_padding_for_kernel(kernel: Array) -> Tuple[Padding, Padding]:
  """Gets padding and unpadding width for a given kernel."""
  shape = kernel.shape
  pad_width = ((shape[0],) * 2, (shape[1],) * 2)
  unpad_width = (
      (
          pad_width[0][0] + (shape[0] + 1) % 2,
          pad_width[0][1] - (shape[0] + 1) % 2,
      ),
      (
          pad_width[1][0] + (shape[1] + 1) % 2,
          pad_width[1][1] - (shape[1] + 1) % 2,
      ),
  )
  return pad_width, unpad_width


def binary_open(
    array: Array,
    kernel: Array,
    pad_mode: PaddingMode = PaddingMode.EDGE,
) -> Array:
  """Applies a binary morphological opening.

  Args:
    array: A binarized 2D array that represents a binary image.
    kernel: The kernel to use.
    pad_mode: The padding mode.

  Returns:
    A boolean array that represents the outcome of morphological opening.
  """
  pad_width, unpad_width = _get_padding_for_kernel(kernel)
  array = _apply_padding(array, pad_width, pad_mode)
  opened = cv.morphologyEx(src=array, kernel=kernel, op=cv.MORPH_OPEN)
  return _remove_padding(opened, unpad_width).view(bool)


def binary_close(
    arr: Array,
    kernel: Array,
    pad_mode: PaddingMode = PaddingMode.EDGE,
) -> Array:
  """Applies a binary morphological closing.

  Args:
    arr: A binarized 2D array that represents a binary image.
    kernel: The kernel to use.
    pad_mode: The padding mode.

  Returns:
    A boolean array that represents the outcome of morphological closing.
  """
  pad_width, unpad_width = _get_padding_for_kernel(kernel)
  arr = _apply_padding(arr, pad_width, pad_mode)
  closed = cv.morphologyEx(src=arr, kernel=kernel, op=cv.MORPH_CLOSE)
  return _remove_padding(closed, unpad_width).view(bool)


def binary_erode(
    array: Array,
    kernel: Array,
    pad_mode: PaddingMode = PaddingMode.EDGE,
) -> Array:
  """Applies a binary morphological erosion.

  Args:
    array: A binarized 2D array that represents a binary image.
    kernel: The kernel to use.
    pad_mode: The padding mode.

  Returns:
    A boolean array that represents the outcome of morphological erosion.
  """
  pad_width, unpad_width = _get_padding_for_kernel(kernel)
  array = _apply_padding(array, pad_width, pad_mode)
  eroded = cv.erode(array, kernel)
  return _remove_padding(eroded, unpad_width).view(bool)


def binary_dilate(
    array: Array,
    kernel: Array,
    pad_mode: PaddingMode = PaddingMode.EDGE,
) -> Array:
  """Applies a binary morphological dilation.

  Args:
    array: A binarized 2D array that represents a binary image.
    kernel: The kernel to use.
    pad_mode: The padding mode.

  Returns:
    A boolean array that represents the outcome of morphological dilation.
  """
  pad_width, unpad_width = _get_padding_for_kernel(kernel)
  array = _apply_padding(array, pad_width, pad_mode)
  dilated = cv.dilate(array, kernel)
  return _remove_padding(dilated, unpad_width).view(bool)


def _apply_padding(
    array: Array, pad_width: Padding, pad_mode: PaddingMode
) -> Array:
  """Applies padding to the input array."""
  ((top, bottom), (left, right)) = pad_width
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


def _remove_padding(array: Array, pad_width: Padding) -> Array:
  """Removes padding from the input array."""
  slices = tuple(
      slice(pad_lo, dim - pad_hi)
      for (pad_lo, pad_hi), dim in zip(pad_width, array.shape)
  )
  return array[slices]


def _trim_margins(
    array: Array, margin_size: MarginSize, pixel_size: PixelSize
) -> Array:
  """Trims margins from an array.

  Args:
    array: The 1D or 2D binarized image array.
    margin_size: The physical dimensions of the image margins. If specified,
      this subregion is excluded from the length scale measurement. Default is
      no margins.
    pixel_size: The physical size of one pixel in the image.

  Returns:
    The input array with the sepecified margins trimmed.
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
