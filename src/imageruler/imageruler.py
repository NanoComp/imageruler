"""Imageruler for measuring minimum lengthscales in binary images."""

import dataclasses
import enum
import functools
from typing import Any, Callable, Tuple

import cv2
import numpy as onp

NDArray = onp.ndarray[Any, Any]


PLUS_3_KERNEL = onp.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
PLUS_5_KERNEL = onp.array(
    [
        [0, 1, 1, 1, 0],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [0, 1, 1, 1, 0],
    ],
    dtype=bool,
)
SQUARE_3_KERNEL = onp.ones((3, 3), dtype=bool)

# Kernels for detecting horizontal and vertical edges.
N_EDGE_KERNEL = onp.array(
    [
        [-1, -1, -1],
        [0, 1, 0],
    ],
    dtype=onp.int8,
)
W_EDGE_KERNEL = onp.rot90(N_EDGE_KERNEL, k=1)
S_EDGE_KERNEL = onp.rot90(N_EDGE_KERNEL, k=2)
E_EDGE_KERNEL = onp.rot90(N_EDGE_KERNEL, k=3)

# Kernels for detecting corners, i.e. "diagonal edges".
NE_CORNER_KERNEL = onp.array(
    [
        [-1, -1],
        [1, -1],
    ],
    dtype=onp.int8,
)
NW_CORNER_KERNEL = onp.rot90(NE_CORNER_KERNEL, k=1)
SW_CORNER_KERNEL = onp.rot90(NE_CORNER_KERNEL, k=2)
SE_CORNER_KERNEL = onp.rot90(NE_CORNER_KERNEL, k=3)


@enum.unique
class IgnoreScheme(enum.Enum):
    """Enumerates schemes for ignoring length scale violations.

    Supported schemes are:

      - `NONE`: a strict scheme in which no violations are ignored.
      - `EDGES`: ignores violations on the edges of all features.
      - `LARGE_FEATURE_EDGES`: ignores violations at the edges of large features only.
        A pixel is on the edge of a large feature if it is on the edge of the feature,
        and adjacent to an interior pixel. Here, interior pixels are those not on any
        edges.
      - `LARGE_FEATURE_EDGES_STRICT`: similar to `LARGE_FEATURE_EDGES`, but uses a more
        strict algorithm to detect edges and does not ignore checkerboard patterns.
    """

    NONE = "none"
    EDGES = "edges"
    LARGE_FEATURE_EDGES = "large_feature_edges"
    LARGE_FEATURE_EDGES_STRICT = "large_feature_edges_strict"


DEFAULT_IGNORE_SCHEME = IgnoreScheme.LARGE_FEATURE_EDGES_STRICT
DEFAULT_FEASIBILITY_GAP_ALLOWANCE = 10


@enum.unique
class PaddingMode(enum.Enum):
    """Enumerates padding modes for arrays."""

    EDGE = "edge"
    SOLID = "solid"
    VOID = "void"


# ------------------------------------------------------------------------------
# Exported functions related to the length scale metric.
# ------------------------------------------------------------------------------


def minimum_length_scale(
    x: NDArray,
    periodic: Tuple[bool, bool] = (False, False),
    ignore_scheme: IgnoreScheme = DEFAULT_IGNORE_SCHEME,
    feasibility_gap_allowance: int = DEFAULT_FEASIBILITY_GAP_ALLOWANCE,
) -> Tuple[int, int]:
    """Identifies the minimum length scale of solid and void features in `x`.

    The minimum length scale for solid (void) features defines the largest brush
    which can be used to recreate the solid (void) features in `x`, by convolving
    an array of "touches" with the brush kernel. In general if an array can be
    created with a given brush, then its solid and void features are unchanged by
    binary opening operations with that brush.

    In some cases, an array that can be created with a brush of size `n` cannot
    be created with the smaller brush if size `n - 1`. Further, small pixel-scale
    violations at edges of features may be unimportant. Some allowance for these
    is provided via optional arguments to this function.

    Args:
        x: Bool-typed rank-2 array containing the features.
        periodic: Specifies which of the two axes are to be regarded as periodic.
        ignore_scheme: Specifies what pixels are ignored when detecting violations.
        feasibility_gap_allowance: In checking whether `x` is feasible with a brush
            of size `n`, we also check for feasibility with larger brushes, since
            e.g. some features realizable with a brush `n + k` may not be realizable
            with the brush of size `n`. The `feasibility_gap_allowance is the
            maximum value of `k` checked. For arrays with very large features,
            particularly when ignoring no violations, larger values may be needed.

    Returns:
        The detected minimum length scales `(length_scale_solid, length_scale_void)`.
    """
    # Note that when the minimum of solid and void length scale is desired,
    # a faster implementation involving comparison between binary opening and
    # closing of designs is possible. This could improve performance by a factor
    # of two or greater.
    if x.ndim != 2:
        raise ValueError(f"`x` must be 2-dimensional, but got shape {x.shape}.")
    if not isinstance(x, onp.ndarray):
        raise ValueError(f"`x` must be a numpy array but got {type(x)}.")
    if x.dtype != bool:
        raise ValueError(f"`x` must be of type `bool` but got {x.dtype}.")
    if not isinstance(periodic[0], bool) or not isinstance(periodic[1], bool):
        raise ValueError(
            f"`periodic` must be a length-2 tuple of `bool` but got {periodic}."
        )

    # Use a dedicated codepath for arrays with a singleton axis.
    if 1 in x.shape:
        idx, squeeze_idx = (1, 0) if x.shape[0] == 1 else (0, 1)
        return minimum_length_scale_1d(
            onp.squeeze(x, axis=squeeze_idx), periodic=periodic[idx]
        )

    return (
        minimum_length_scale_solid(
            x, periodic, ignore_scheme, feasibility_gap_allowance
        ),
        minimum_length_scale_solid(
            ~x, periodic, ignore_scheme, feasibility_gap_allowance
        ),
    )


def minimum_length_scale_solid(
    x: NDArray,
    periodic: Tuple[bool, bool] = (False, False),
    ignore_scheme: IgnoreScheme = DEFAULT_IGNORE_SCHEME,
    feasibility_gap_allowance: int = DEFAULT_FEASIBILITY_GAP_ALLOWANCE,
) -> int:
    """Identifies the minimum length scale of solid features in `x`.

    Args:
        x: Bool-typed rank-2 array containing the features.
        periodic: Specifies which of the two axes are to be regarded as periodic.
        ignore_scheme: Specifies what pixels are ignored when detecting violations.
        feasibility_gap_allowance: In checking whether `x` is feasible with a brush
            of size `n`, we also check for feasibility with larger brushes, since
            e.g. some features realizable with a brush `n + k` may not be realizable
            with the brush of size `n`. The `feasibility_gap_allowance is the
            maximum value of `k` checked. For arrays with very large features,
            particularly when ignoring no violations, larger values may be needed.

    Returns:
        The detected minimum length scale of solid features.
    """
    assert x.dtype == bool

    def test_fn(length_scale: int) -> bool:
        return ~onp.any(  # type: ignore
            length_scale_violations_solid(
                x=x,
                length_scale=length_scale,
                periodic=periodic,
                ignore_scheme=ignore_scheme,
                feasibility_gap_allowance=feasibility_gap_allowance,
            )
        )

    return maximum_true_arg(
        nearly_monotonic_fn=test_fn,
        min_arg=1,
        max_arg=max(x.shape),
        non_monotonic_allowance=feasibility_gap_allowance,
    )


def length_scale_violations_solid(
    x: NDArray,
    length_scale: int,
    periodic: Tuple[bool, bool] = (False, False),
    ignore_scheme: IgnoreScheme = DEFAULT_IGNORE_SCHEME,
    feasibility_gap_allowance: int = DEFAULT_FEASIBILITY_GAP_ALLOWANCE,
) -> NDArray:
    """Computes the length scale violations, allowing for the feasibility gap.

    Args:
        x: Bool-typed rank-2 array containing the features.
        length_scale: The length scale for which violations are sought.
        periodic: Specifies which of the two axes are to be regarded as periodic.
        ignore_scheme: Specifies what pixels are ignored when detecting violations.
        feasibility_gap_allowance: In checking whether `x` is feasible with a brush
            of size `n`, we also check for feasibility with larger brushes, since
            e.g. some features realizable with a brush `n + k` may not be realizable
            with the brush of size `n`. The `feasibility_gap_allowance is the
            maximum value of `k` checked. For arrays with very large features,
            particularly when ignoring no violations, larger values may be needed.

    Returns:
        The array containing violations.
    """
    violations = []
    for scale in range(length_scale, length_scale + feasibility_gap_allowance):
        violations.append(
            length_scale_violations_solid_strict(x, scale, periodic, ignore_scheme)
        )
    length_scale_violations: NDArray = onp.all(violations, axis=0)
    return length_scale_violations


# ------------------------------------------------------------------------------
# Non-exported functions related to the length scale metric.
# ------------------------------------------------------------------------------


def length_scale_violations_solid_strict(
    x: NDArray,
    length_scale: int,
    periodic: Tuple[bool, bool],
    ignore_scheme: IgnoreScheme,
) -> NDArray:
    """Identifies length scale violations of solid features in `x`.

    Args:
        x: Bool-typed rank-2 array containing the features.
        length_scale: The length scale for which violations are sought.
        periodic: Specifies which of the two axes are to be regarded as periodic.
        ignore_scheme: Specifies what pixels are ignored when detecting violations.

    Returns:
        The array containing violations.
    """
    violations = _length_scale_violations_solid_strict(
        wrapped_x=_HashableArray(x),
        length_scale=length_scale,
        periodic=periodic,
        ignore_scheme=ignore_scheme,
    )
    assert violations.shape == x.shape
    return violations


@dataclasses.dataclass
class _HashableArray:
    """Hashable wrapper for numpy arrays."""

    array: NDArray

    def __hash__(self) -> int:
        return hash((self.array.dtype, self.array.shape, self.array.tobytes()))

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, _HashableArray):
            return False
        return onp.all(self.array == other.array) and (  # type: ignore
            self.array.dtype == other.array.dtype
        )


@functools.lru_cache(maxsize=128)
def _length_scale_violations_solid_strict(
    wrapped_x: _HashableArray,
    length_scale: int,
    periodic: Tuple[bool, bool],
    ignore_scheme: IgnoreScheme,
) -> NDArray:
    """Identifies length scale violations of solid features in `x`.

    This function is strict, in the sense that no violations are ignored.

    Args:
        wrapped_x: The wrapped bool-typed rank-2 array containing the features.
        length_scale: The length scale for which violations are sought.
        periodic: Specifies which of the two axes are to be regarded as periodic.
        ignore_scheme: Specifies what pixels are ignored when detecting violations.

    Returns:
        The array containing violations.
    """
    x = wrapped_x.array
    kernel = kernel_for_length_scale(length_scale)
    violations_solid = x & ~binary_opening(x, kernel, periodic, PaddingMode.SOLID)

    ignored = ignored_pixels(x, periodic, ignore_scheme)
    violations_solid = violations_solid & ~ignored
    return onp.asarray(violations_solid)


def kernel_for_length_scale(length_scale: int) -> NDArray:
    """Returns an approximately circular kernel for the given `length_scale`.

    The kernel has shape `(length_scale, length_scale)`, and is `True` for pixels
    whose centers lie within the circle of radius `length_scale / 2` centered on
    the kernel. This yields a pixelated circle, which for length scales less than
    `3` will actually be square.

    Args:
        length_scale: The length scale for which the kernel is sought.

    Returns:
        The approximately circular kernel.
    """
    assert length_scale > 0
    centers = onp.arange(-length_scale / 2 + 0.5, length_scale / 2)
    squared_distance = centers[:, onp.newaxis] ** 2 + centers[onp.newaxis, :] ** 2
    kernel = squared_distance < (length_scale / 2) ** 2
    # Ensure that kernels larger than `2` can be realized with a width-3 brush.
    if length_scale > 2:
        kernel = binary_opening(
            kernel,
            kernel=PLUS_3_KERNEL,
            periodic=(False, False),
            padding_mode=PaddingMode.VOID,
        )
    return kernel


def ignored_pixels(
    x: NDArray,
    periodic: Tuple[bool, bool],
    ignore_scheme: IgnoreScheme,
) -> NDArray:
    """Returns an array indicating locations at which violations are to be ignored.

    Args:
        x: The array for which ignored locations are to be identified.
        periodic: Specifies which of the two axes are to be regarded as periodic.
        ignore_scheme: Specifies the manner in which ignored locations are identified.

    Returns:
        The array indicating locations to be ignored.
    """
    assert x.dtype == bool
    if ignore_scheme == IgnoreScheme.NONE:
        return onp.zeros_like(x)
    elif ignore_scheme == IgnoreScheme.EDGES:
        return onp.asarray(
            x & ~binary_erosion(x, PLUS_3_KERNEL, periodic, PaddingMode.SOLID)
        )
    elif ignore_scheme == IgnoreScheme.LARGE_FEATURE_EDGES:
        return onp.asarray(x & ~erode_large_features(x, periodic))
    elif ignore_scheme == IgnoreScheme.LARGE_FEATURE_EDGES_STRICT:
        return onp.asarray(x & ~erode_large_features_strict(x, periodic))
    else:
        raise ValueError(f"Unknown `ignore_scheme`, got {ignore_scheme}.")


# ------------------------------------------------------------------------------
# Array-manipulating functions backed by `cv2`.
# ------------------------------------------------------------------------------


def binary_opening(
    x: NDArray, kernel: NDArray, periodic: Tuple[bool, bool], padding_mode: PaddingMode
) -> NDArray:
    """Performs binary opening with the given `kernel` and padding mode."""
    assert x.ndim == 2
    assert x.dtype == bool
    assert kernel.ndim == 2
    assert kernel.dtype == bool
    # The `cv2` convention for binary opening yields a shifted output with
    # even-shape kernels, requiring padding and unpadding to differ.
    pad_width, unpad_width = _pad_width_for_kernel_shape(kernel.shape)
    opened = cv2.morphologyEx(
        src=pad_2d(x, pad_width, periodic, padding_mode).view(onp.uint8),
        kernel=kernel.view(onp.uint8),
        op=cv2.MORPH_OPEN,
    )
    return unpad(opened.view(bool), unpad_width)


def binary_erosion(
    x: NDArray, kernel: NDArray, periodic: Tuple[bool, bool], padding_mode: PaddingMode
) -> NDArray:
    """Performs binary erosion with structuring element `kernel`."""
    assert x.dtype == bool
    assert kernel.dtype == bool
    pad_width = ((kernel.shape[0],) * 2, (kernel.shape[1],) * 2)
    eroded = cv2.erode(
        src=pad_2d(x, pad_width, periodic, padding_mode).view(onp.uint8),
        kernel=kernel.view(onp.uint8),
    )
    return unpad(eroded.view(bool), pad_width)


def binary_dilation(
    x: NDArray, kernel: NDArray, periodic: Tuple[bool, bool], padding_mode: PaddingMode
) -> NDArray:
    """Performs binary dilation with structuring element `kernel`."""
    assert x.dtype == bool
    assert kernel.dtype == bool
    # The `cv2` convention for binary dilation yields a shifted output with
    # even-shape kernels, requiring padding and unpadding to differ.
    pad_width, unpad_width = _pad_width_for_kernel_shape(kernel.shape)
    dilated = cv2.dilate(
        src=pad_2d(x, pad_width, periodic, padding_mode).view(onp.uint8),
        kernel=kernel.view(onp.uint8),
    )
    return unpad(dilated.view(bool), unpad_width)


_Padding = Tuple[Tuple[int, int], Tuple[int, int]]


def _pad_width_for_kernel_shape(shape: Tuple[int, ...]) -> Tuple[_Padding, _Padding]:
    """Prepares `pad_width` and `unpad_width` for the given kernel shape."""
    assert len(shape) == 2
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


def hitmiss(
    x: NDArray,
    kernel: NDArray,
    anchor_ij: Tuple[int, int] = (-1, -1),
) -> NDArray:
    """Applies the hitmiss transformation to `x`."""
    anchor_y, anchor_x = anchor_ij
    return cv2.morphologyEx(
        x.view(onp.uint8),
        kernel=kernel,
        op=cv2.MORPH_HITMISS,
        anchor=(anchor_x, anchor_y),
        borderType=cv2.BORDER_REPLICATE,
    ).view(bool)


def edges_n(x: NDArray) -> NDArray:
    """Detect northeast corners of solid features."""
    return hitmiss(x, kernel=N_EDGE_KERNEL, anchor_ij=(1, 1)) & x


def edges_w(x: NDArray) -> NDArray:
    """Detect northwest corners of solid features."""
    return hitmiss(x, kernel=W_EDGE_KERNEL, anchor_ij=(1, 1)) & x


def edges_s(x: NDArray) -> NDArray:
    """Detect southwest corners of solid features."""
    return hitmiss(x, kernel=S_EDGE_KERNEL, anchor_ij=(0, 1)) & x


def edges_e(x: NDArray) -> NDArray:
    """Detect southeast corners of solid features."""
    return hitmiss(x, kernel=E_EDGE_KERNEL, anchor_ij=(1, 0)) & x


def corners_ne(x: NDArray) -> NDArray:
    """Detect northeast corners of solid features."""
    return hitmiss(x, kernel=NE_CORNER_KERNEL, anchor_ij=(1, 0))


def corners_nw(x: NDArray) -> NDArray:
    """Detect northwest corners of solid features."""
    return hitmiss(x, kernel=NW_CORNER_KERNEL, anchor_ij=(1, 1))


def corners_sw(x: NDArray) -> NDArray:
    """Detect southwest corners of solid features."""
    return hitmiss(x, kernel=SW_CORNER_KERNEL, anchor_ij=(0, 1))


def corners_se(x: NDArray) -> NDArray:
    """Detect southeast corners of solid features."""
    return hitmiss(x, kernel=SE_CORNER_KERNEL, anchor_ij=(0, 0))


def detect_edges(
    x: NDArray,
    periodic: Tuple[bool, bool],
) -> NDArray:
    """Idetifies edges of solid features in `x`.

    The edge of a solid feature may either be horizontal (north or south) or vertical
    (east or west), or can be a corner (northeast, northwest, southeast, southwest).

    Args:
        x: Bool-typed rank-2 array where corners are to be detected.
        periodic: Specifies which of the two axes are to be regarded as periodic.

    Returns:
        The array with identified corners.
    """
    x = pad_2d(
        x, pad_width=((2, 2), (2, 2)), periodic=periodic, padding_mode=PaddingMode.EDGE
    )
    edges = (
        edges_n(x)
        | edges_w(x)
        | edges_s(x)
        | edges_e(x)
        | corners_ne(x)
        | corners_nw(x)
        | corners_sw(x)
        | corners_se(x)
    )
    return edges[2:-2, 2:-2]


def erode_large_features(x: NDArray, periodic: Tuple[bool, bool]) -> NDArray:
    """Erodes large features while leaving small features untouched.

    Note that this operation can change the topology of `x`, i.e. it
    may create two disconnected solid features where originally there
    was a single contiguous feature.

    Args:
        x: Bool-typed rank-2 array to be eroded.
        periodic: Specifies which of the two axes are to be regarded as periodic.

    Returns:
        The array with eroded features.
    """
    assert x.dtype == bool

    # Identify interior solid pixels, which should not be removed. Pixels for
    # which the neighborhood sum equals `9` are interior pixels.
    neighborhood_sum = _filter_2d(x, SQUARE_3_KERNEL, periodic, PaddingMode.EDGE)
    interior_pixels = neighborhood_sum == 9

    # Identify solid pixels that are adjacent to interior pixels.
    adjacent_to_interior = (
        x
        & ~interior_pixels
        & binary_dilation(
            x=interior_pixels,
            kernel=PLUS_5_KERNEL,
            periodic=periodic,
            padding_mode=PaddingMode.EDGE,
        )
    )

    removed_by_erosion = x & ~binary_erosion(
        x, PLUS_3_KERNEL, periodic, PaddingMode.EDGE
    )
    should_remove = adjacent_to_interior & removed_by_erosion
    return onp.asarray(x & ~should_remove)


def erode_large_features_strict(x: NDArray, periodic: Tuple[bool, bool]) -> NDArray:
    """Erodes large features while leaving small features untouched.

    This function uses a more strict algorithm than `erode_large_features`, and will
    not remove the corners in a checkerboard pattern.

    Args:
        x: Bool-typed rank-2 array to be eroded.
        periodic: Specifies which of the two axes are to be regarded as periodic.

    Returns:
        The array with eroded features.
    """
    assert x.dtype == bool

    neighborhood_sum = _filter_2d(x, SQUARE_3_KERNEL, periodic, PaddingMode.EDGE)
    interior_pixels = neighborhood_sum == 9

    edge_pixels = detect_edges(x, periodic=periodic)

    # Identify solid pixels that are adjacent to interior pixels.
    adjacent_to_interior = (
        x
        & ~interior_pixels
        & binary_dilation(
            x=interior_pixels,
            kernel=PLUS_5_KERNEL,
            periodic=periodic,
            padding_mode=PaddingMode.EDGE,
        )
    )
    should_remove = adjacent_to_interior & edge_pixels
    return onp.asarray(x & ~should_remove)


def _filter_2d(
    x: NDArray, kernel: NDArray, periodic: Tuple[bool, bool], padding_mode: PaddingMode
) -> NDArray:
    """Convolves `x` with `kernel`."""
    assert x.dtype == bool
    assert kernel.dtype == bool
    pad_width, unpad_width = _pad_width_for_kernel_shape(kernel.shape)
    filtered = cv2.filter2D(
        src=pad_2d(x, pad_width, periodic, padding_mode).view(onp.uint8),
        kernel=kernel.view(onp.uint8),
        ddepth=cv2.CV_32F,
        borderType=cv2.BORDER_REPLICATE,
    )
    filtered = onp.around(onp.asarray(filtered)).astype(int)
    return unpad(filtered, unpad_width)


def pad_2d(
    x: NDArray,
    pad_width: Tuple[Tuple[int, int], Tuple[int, int]],
    periodic: Tuple[bool, bool],
    padding_mode: PaddingMode,
) -> NDArray:
    """Pads rank-2 boolean array `x` with the specified mode.

    Padding may take values from the edge pixels, or be entirely solid or
    void, determined by the `mode` parameter.

    Args:
        x: The array to be padded.
        pad_width: The extent of the padding, `((i_lo, i_hi), (j_lo, j_hi))`.
        periodic: Specifies which of the two axes are to be regarded as periodic.
        padding_mode: Specifies the padding mode to be used.

    Returns:
        The padded array.
    """
    assert x.dtype == bool
    ((top, bottom), (left, right)) = pad_width

    pad_value = 1 if padding_mode == PaddingMode.SOLID else 0

    if periodic[0]:
        border_type_i = cv2.BORDER_WRAP
    elif padding_mode == PaddingMode.EDGE:
        border_type_i = cv2.BORDER_REPLICATE
    else:
        border_type_i = cv2.BORDER_CONSTANT
    x = cv2.copyMakeBorder(  # type: ignore[call-overload]
        src=x.view(onp.uint8),
        top=top,
        bottom=bottom,
        left=0,
        right=0,
        borderType=border_type_i,
        value=pad_value,
    )

    if periodic[1]:
        border_type_j = cv2.BORDER_WRAP
    elif padding_mode == PaddingMode.EDGE:
        border_type_j = cv2.BORDER_REPLICATE
    else:
        border_type_j = cv2.BORDER_CONSTANT
    x = cv2.copyMakeBorder(  # type: ignore[call-overload]
        src=x,
        top=0,
        bottom=0,
        left=left,
        right=right,
        borderType=border_type_j,
        value=pad_value,
    ).view(bool)
    return onp.asarray(x)


def unpad(
    x: NDArray,
    pad_width: Tuple[Tuple[int, int], ...],
) -> NDArray:
    """Undoes a pad operation."""
    slices = tuple(
        slice(pad_lo, dim - pad_hi) for (pad_lo, pad_hi), dim in zip(pad_width, x.shape)
    )
    return x[slices]


# ------------------------------------------------------------------------------
# Functions that find thresholds of nearly-monotonic functions.
# ------------------------------------------------------------------------------


def maximum_true_arg(
    nearly_monotonic_fn: Callable[[int], bool],
    min_arg: int,
    max_arg: int,
    non_monotonic_allowance: int,
) -> int:
    """Searches for the maximum integer for which `nearly_monotonic_fn` is `True`.

    This requires `nearly_monotonic_fn` to be approximately monotonically
    decreasing, i.e. it should be `True` for small arguments and then `False` for
    large arguments. Some allowance for "noisy" behavior at the transition is
    controlled by `non_monotonic_allowance`.

    The input argument is checked in the range `[min_arg, max_arg]`, where both
    values are positive. If `test_fn` is never `True`, `min_arg` is returned.

    Note that the algorithm here assumes that `nearly_monotonic_fn` is expensive
    to evaluate with large arguments, and so a "small first" search strategy is
    employed. For this reason, `min_arg` must be positive.

    Args:
        nearly_monotonic_fn: The function for which the maximum `True` argument is
            sought.
        min_arg: The minimum argument. Must be positive.
        max_arg: The maximum argument. Must be greater than `min_arg.`
        non_monotonic_allowance: The number of candidate arguments where the
            function evaluates to `False` to be considered before concluding that the
            maximum `True` argument is smaller than the candidates. Must be positive.

    Returns:
        The maximum `True` argument, or `min_arg`.
    """
    assert min_arg > 0
    assert min_arg < max_arg
    assert non_monotonic_allowance > 0

    max_true_arg = min_arg - 1

    while min_arg <= max_arg:
        # We double `min_arg` rather than bisecting, as this requires fewer
        # evaluations when the minimum `True` value is close to `min_arg`.
        test_arg_start = min(min_arg * 2, max_arg)
        test_arg_stop = min(test_arg_start + non_monotonic_allowance, max_arg + 1)
        for test_arg in range(test_arg_start, test_arg_stop):
            result = nearly_monotonic_fn(test_arg)
            if result:
                break
        if result:
            min_arg = test_arg + 1
            max_true_arg = max(max_true_arg, test_arg)
        else:
            max_arg = test_arg_start - 1
    return max_true_arg


# ------------------------------------------------------------------------------
# Functions that handle the special case of 1D patterns.
# ------------------------------------------------------------------------------


def minimum_length_scale_1d(x: NDArray, periodic: bool) -> Tuple[int, int]:
    """Return the minimum solid and void length scale for a 1D array."""
    assert x.dtype == bool
    x = x.astype(int)

    # Find interior locations within `x` where the pattern transistions from
    # solid to void, or vice versa.
    xpad = onp.pad(x, (1, 1), mode="edge")
    delta = onp.roll(xpad, 1) - xpad
    delta = delta[1:-1]
    (idxs,) = onp.where(onp.abs(delta) > 0)

    # Determine the dimensions of each region.
    idxs = onp.concatenate([[0], idxs, [x.size]])
    dims = idxs[1:] - idxs[:-1]

    if periodic:
        # Wrap the starting region in the case of a periodic pattern.
        if x[0] == x[-1]:
            dims = onp.concatenate([[dims[0] + dims[-1]], dims[1:-1]])
            dims = onp.minimum(dims, x.size)
    else:
        # If not periodic, discount the first and last regions.
        dims = dims[1:-1]

    # Find the minimum length scale.
    min_dim_first_region = int(onp.amin(dims[::2]) if len(dims) > 0 else x.size)
    min_dim_second_region = int(onp.amin(dims[1::2]) if len(dims) > 1 else x.size)

    starts_with_solid = bool(x[0])
    starts_with_void = not starts_with_solid
    if starts_with_solid and periodic or (starts_with_void and not periodic):
        return min_dim_first_region, min_dim_second_region
    else:
        return min_dim_second_region, min_dim_first_region
