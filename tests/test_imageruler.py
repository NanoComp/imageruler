"""Tests for `imageruler`."""

import itertools
import unittest

import numpy as onp
from parameterized import parameterized
from scipy import ndimage

from imageruler import imageruler

TEST_ARRAY_4_5 = onp.array(
    [  # Solid features feasible with circle-4, void with circle-5.
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1],
        [0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1],
        [0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
    ],
    dtype=bool,
)
TEST_ARRAY_5_5 = onp.array(
    [  # Solid features feasible with circle-5, void with circle-5.
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1],
        [0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1],
        [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
    ],
    dtype=bool,
)
TEST_ARRAY_5_3 = onp.array(
    [  # Solid features feasible with circle-5, void with circle-3.
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1],
        [0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1],
        [0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
    ],
    dtype=bool,
)
TEST_ARRAY_5_WITH_DEFECT = onp.array(
    [  # Mostly feasible with diameter-4 brush.
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1],  # One pixel here is defective.
        [0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1],
        [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
    ],
    dtype=bool,
)
TEST_ARRAYS = [TEST_ARRAY_4_5, TEST_ARRAY_5_5, TEST_ARRAY_5_3, TEST_ARRAY_5_WITH_DEFECT]


TEST_KERNEL_4 = onp.array(
    [[0, 1, 1, 0], [1, 1, 1, 1], [1, 1, 1, 1], [0, 1, 1, 0]], dtype="bool"
)
TEST_KERNEL_5 = onp.array(
    [
        [0, 1, 1, 1, 0],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [0, 1, 1, 1, 0],
    ],
    dtype="bool",
)
TEST_KERNEL_4_3_ASYMMETRIC = onp.array(
    [[0, 1, 0], [1, 1, 1], [1, 1, 1], [0, 1, 0]], dtype="bool"
)
TEST_KERNELS = [TEST_KERNEL_4, TEST_KERNEL_5, TEST_KERNEL_4_3_ASYMMETRIC]


class LengthScaleTest(unittest.TestCase):
    @parameterized.expand(
        [
            (TEST_ARRAY_4_5, 4, 5),
            (TEST_ARRAY_5_5, 5, 5),
            (TEST_ARRAY_5_3, 5, 3),
        ]
    )
    def test_length_scale_matches_expected(self, x, expected_solid, expected_void):
        length_scale_solid, length_scale_void = imageruler.minimum_length_scale(
            x, ignore_scheme=imageruler.IgnoreScheme.NONE
        )
        self.assertEqual(length_scale_solid, expected_solid)
        self.assertEqual(length_scale_void, expected_void)

    @parameterized.expand(
        list(
            itertools.product(
                [i for i in range(1, 20)],
                [
                    imageruler.IgnoreScheme.NONE,
                    imageruler.IgnoreScheme.LARGE_FEATURE_EDGES,
                ],
            )
        )
    )
    def test_circle_has_expected_length_scale(self, length_scale, ignore_scheme):
        # With `NONE` and `LARGE_FEATURE_EDGES` ignore schemes, even small length
        # scales will correctly be identified.
        x = imageruler.kernel_for_length_scale(length_scale)
        x = onp.pad(x, ((1, 1), (1, 1)), mode="constant")
        with self.subTest("solid_circle"):
            length_scale_solid, length_scale_void = imageruler.minimum_length_scale(
                x, ignore_scheme=ignore_scheme
            )
            self.assertEqual(length_scale_solid, length_scale)
            self.assertEqual(length_scale_void, min(x.shape))
        with self.subTest("void_circle"):
            length_scale_solid, length_scale_void = imageruler.minimum_length_scale(
                ~x, ignore_scheme=ignore_scheme
            )
            self.assertEqual(length_scale_void, length_scale)
            self.assertEqual(length_scale_solid, min(x.shape))

    @parameterized.expand([[i] for i in range(5, 20)])
    def test_circle_has_expected_length_scale_ignore_edges(self, length_scale):
        # With the `EDGES` ignore scheme, we will be blind to small features.
        # Check only for features larger than 5 pixels in size.
        x = imageruler.kernel_for_length_scale(length_scale)
        x = onp.pad(x, ((1, 1), (1, 1)), mode="constant")
        with self.subTest("solid_circle"):
            length_scale_solid, length_scale_void = imageruler.minimum_length_scale(
                x, ignore_scheme=imageruler.IgnoreScheme.EDGES
            )
            self.assertEqual(length_scale_solid, length_scale)
            self.assertEqual(length_scale_void, min(x.shape))
        with self.subTest("void_circle"):
            length_scale_solid, length_scale_void = imageruler.minimum_length_scale(
                ~x, ignore_scheme=imageruler.IgnoreScheme.EDGES
            )
            self.assertEqual(length_scale_void, length_scale)
            self.assertEqual(length_scale_solid, min(x.shape))

    @parameterized.expand(
        list(
            itertools.product(
                list(range(1, 20)),
                [
                    imageruler.IgnoreScheme.NONE,
                    imageruler.IgnoreScheme.LARGE_FEATURE_EDGES,
                    imageruler.IgnoreScheme.LARGE_FEATURE_EDGES_STRICT,
                ],
            )
        )
    )
    def test_line_has_expected_length_scale(self, length_scale, ignore_scheme):
        # Make an array that has a single void circular feature with diameter equal
        # to the length scale. Pad to make sure the feature is isolated.
        x = onp.zeros((40, 40), dtype=bool)
        x[:, 10 : (10 + length_scale)] = True
        length_scale_solid, length_scale_void = imageruler.minimum_length_scale(
            x, ignore_scheme=ignore_scheme
        )
        self.assertEqual(length_scale_solid, length_scale)
        self.assertEqual(length_scale_void, min(x.shape))

    @parameterized.expand(
        [[imageruler.IgnoreScheme.EDGES], [imageruler.IgnoreScheme.LARGE_FEATURE_EDGES]]
    )
    def test_brush_violations_with_interface_defects(self, ignore_scheme):
        # Assert that there are violations in the defective array.
        assert onp.any(
            imageruler.length_scale_violations_solid_strict(
                TEST_ARRAY_5_WITH_DEFECT,
                length_scale=4,
                periodic=(False, False),
                ignore_scheme=imageruler.IgnoreScheme.NONE,
            )
        )
        # Assert that there are no violations if we ignore interfaces.
        assert not onp.any(
            imageruler.length_scale_violations_solid_strict(
                TEST_ARRAY_5_WITH_DEFECT,
                length_scale=4,
                periodic=(False, False),
                ignore_scheme=ignore_scheme,
            )
        )

    @parameterized.expand([[s] for s in imageruler.IgnoreScheme])
    def test_solid_feature_shallow_incidence(self, ignore_scheme):
        # Checks that the length scale for a design having a solid feature that
        # is incident on the design edge with a very shallow angle has a length
        # scale equal to the size of the design.
        x = onp.ones((70, 70), dtype=bool)
        x[-1, 10:] = 0
        x[-2, 20:] = 0
        length_scale_solid, length_scale_void = imageruler.minimum_length_scale(
            x, ignore_scheme=ignore_scheme
        )
        self.assertEqual(length_scale_solid, 70)
        self.assertEqual(length_scale_void, 70)

    def test_feasibility_gap(self):
        # Tests that if we have features feasible with size 6 kernel and size 7
        # kernel, but not both, that a minimum feature size of 6 is reported.
        circle6 = onp.pad(imageruler.kernel_for_length_scale(6), ((2, 1), (2, 1)))
        circle7 = onp.pad(imageruler.kernel_for_length_scale(7), ((1, 1), (1, 1)))
        # Check that the `circle6` is feasible with `6` but infeasible with `7`.
        self.assertFalse(
            onp.any(
                imageruler.length_scale_violations_solid_strict(
                    circle6, 6, (False, False), imageruler.IgnoreScheme.NONE
                )
            )
        )
        self.assertTrue(
            onp.any(
                imageruler.length_scale_violations_solid_strict(
                    circle6, 7, (False, False), imageruler.IgnoreScheme.NONE
                )
            )
        )
        # Check that the `circle7` is infeasible with `6` but feasible with `7`.
        self.assertTrue(
            onp.any(
                imageruler.length_scale_violations_solid_strict(
                    circle7, 6, (False, False), imageruler.IgnoreScheme.NONE
                )
            )
        )
        self.assertFalse(
            onp.any(
                imageruler.length_scale_violations_solid_strict(
                    circle7, 7, (False, False), imageruler.IgnoreScheme.NONE
                )
            )
        )
        # Check that putting both features in the same design makes it infeasible
        # for `6` and `7`.
        merged = onp.concatenate([circle6, circle7])
        self.assertTrue(
            onp.any(
                imageruler.length_scale_violations_solid_strict(
                    merged, 6, (False, False), imageruler.IgnoreScheme.NONE
                )
            )
        )
        self.assertTrue(
            onp.any(
                imageruler.length_scale_violations_solid_strict(
                    merged, 7, (False, False), imageruler.IgnoreScheme.NONE
                )
            )
        )
        # Check that when allowing for the feasibility gap, the merged design is
        # considered feasible with `6` but infeasible with `7`.
        self.assertFalse(
            onp.any(
                imageruler.length_scale_violations_solid(
                    merged,
                    6,
                    (False, False),
                    imageruler.IgnoreScheme.NONE,
                    feasibility_gap_allowance=2,
                )
            )
        )
        self.assertTrue(
            onp.any(
                imageruler.length_scale_violations_solid(
                    merged,
                    7,
                    (False, False),
                    imageruler.IgnoreScheme.NONE,
                    feasibility_gap_allowance=2,
                )
            )
        )


class PeriodicLengthScale(unittest.TestCase):
    @parameterized.expand(range(1, 10))
    def test_circle_array_length_scale(self, pad_amount):
        x = imageruler.kernel_for_length_scale(20)
        x = onp.pad(x, ((pad_amount, 0), (pad_amount, 0)))
        length_scale_not_periodic = imageruler.minimum_length_scale(
            x, periodic=(False, False), ignore_scheme=imageruler.IgnoreScheme.NONE
        )
        onp.testing.assert_array_equal(length_scale_not_periodic, (20, x.shape[0]))

        length_scale_periodic_both = imageruler.minimum_length_scale(
            x, periodic=(True, True), ignore_scheme=imageruler.IgnoreScheme.NONE
        )
        onp.testing.assert_array_equal(length_scale_periodic_both, (20, pad_amount))

    @parameterized.expand(range(1, 10))
    def test_line_array_length_scale(self, pad_amount):
        x = onp.ones((20, 20))
        x = onp.pad(x, ((pad_amount, 0), (0, 0))).astype(bool)
        length_scale_not_periodic = imageruler.minimum_length_scale(
            x, periodic=(False, False), ignore_scheme=imageruler.IgnoreScheme.NONE
        )
        onp.testing.assert_array_equal(
            length_scale_not_periodic, (x.shape[0], x.shape[0])
        )

        length_scale_periodic_x = imageruler.minimum_length_scale(
            x, periodic=(True, False), ignore_scheme=imageruler.IgnoreScheme.NONE
        )
        onp.testing.assert_array_equal(length_scale_periodic_x, (20, pad_amount))

        length_scale_periodic_y = imageruler.minimum_length_scale(
            x, periodic=(False, True), ignore_scheme=imageruler.IgnoreScheme.NONE
        )
        onp.testing.assert_array_equal(
            length_scale_periodic_y, (x.shape[0], x.shape[0])
        )

        length_scale_periodic_both = imageruler.minimum_length_scale(
            x, periodic=(True, True), ignore_scheme=imageruler.IgnoreScheme.NONE
        )
        onp.testing.assert_array_equal(length_scale_periodic_both, (20, pad_amount))


class OneDimensionalLengthScaleTest(unittest.TestCase):
    @parameterized.expand(
        [
            [[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], (False, False), (10, 10)],
            [[[0, 0, 1, 1, 0, 0, 0, 0, 0, 0]], (False, False), (2, 10)],
            [[[0, 0, 1, 1, 0, 1, 1, 0, 0, 0]], (False, False), (2, 1)],
            [[[0, 0, 1, 1, 0, 0, 0, 0, 0, 0]], (False, True), (2, 8)],
            [[[0, 0, 1, 1, 0, 0, 0, 0, 0, 0]], (True, False), (2, 10)],
        ]
    )
    def test_length_scale_matches_expected(self, x, periodic, expected):
        mls = imageruler.minimum_length_scale(
            x=onp.asarray(x, dtype=bool),
            periodic=periodic,
        )
        onp.testing.assert_array_equal(mls, expected)


class KernelTest(unittest.TestCase):
    @parameterized.expand([(4, TEST_KERNEL_4), (5, TEST_KERNEL_5)])
    def test_kernel_matches_expected(self, length_scale, expected):
        onp.testing.assert_array_equal(
            imageruler.kernel_for_length_scale(length_scale), expected
        )

    def test_length_scale_1(self):
        onp.testing.assert_array_equal(
            imageruler.kernel_for_length_scale(1), onp.ones((1, 1))
        )

    def test_length_scale_2(self):
        onp.testing.assert_array_equal(
            imageruler.kernel_for_length_scale(2), onp.ones((2, 2))
        )

    def test_length_scale_3(self):
        onp.testing.assert_array_equal(
            imageruler.kernel_for_length_scale(3), imageruler.PLUS_3_KERNEL
        )


# ------------------------------------------------------------------------------
# Tests for array-manipulating functions.
# ------------------------------------------------------------------------------


class MorphologyOperationsTest(unittest.TestCase):
    @parameterized.expand(
        list(itertools.product(TEST_ARRAYS, TEST_KERNELS, imageruler.PaddingMode))
    )
    def test_erosion_matches_scipy(self, x, kernel, padding_mode):
        pad_width = ((kernel.shape[0],) * 2, (kernel.shape[1],) * 2)
        if padding_mode == imageruler.PaddingMode.EDGE:
            x_padded = onp.pad(x, pad_width, mode="edge")
        elif padding_mode == imageruler.PaddingMode.SOLID:
            x_padded = onp.pad(x, pad_width, mode="constant", constant_values=True)
        elif padding_mode == imageruler.PaddingMode.VOID:
            x_padded = onp.pad(x, pad_width, mode="constant", constant_values=False)
        expected = ndimage.binary_erosion(x_padded, kernel)
        expected = expected[
            pad_width[0][0] : expected.shape[0] - pad_width[0][1],
            pad_width[1][0] : expected.shape[1] - pad_width[1][1],
        ]
        actual = imageruler.binary_erosion(
            x, kernel, periodic=(False, False), padding_mode=padding_mode
        )
        onp.testing.assert_array_equal(expected, actual)

    @parameterized.expand(
        list(itertools.product(TEST_ARRAYS, TEST_KERNELS, imageruler.PaddingMode))
    )
    def test_dilation_matches_scipy(self, x, kernel, padding_mode):
        pad_width = ((kernel.shape[0],) * 2, (kernel.shape[1],) * 2)
        if padding_mode == imageruler.PaddingMode.EDGE:
            x_padded = onp.pad(x, pad_width, mode="edge")
        elif padding_mode == imageruler.PaddingMode.SOLID:
            x_padded = onp.pad(x, pad_width, mode="constant", constant_values=True)
        elif padding_mode == imageruler.PaddingMode.VOID:
            x_padded = onp.pad(x, pad_width, mode="constant", constant_values=False)
        expected = ndimage.binary_dilation(x_padded, kernel)
        expected = expected[
            pad_width[0][0] : expected.shape[0] - pad_width[0][1],
            pad_width[1][0] : expected.shape[1] - pad_width[1][1],
        ]
        actual = imageruler.binary_dilation(
            x, kernel, periodic=(False, False), padding_mode=padding_mode
        )
        onp.testing.assert_array_equal(expected, actual)

    @parameterized.expand(
        list(itertools.product(TEST_ARRAYS, TEST_KERNELS, imageruler.PaddingMode))
    )
    def test_opening_matches_scipy(self, x, kernel, padding_mode):
        pad_width = ((kernel.shape[0],) * 2, (kernel.shape[1],) * 2)
        if padding_mode == imageruler.PaddingMode.EDGE:
            x_padded = onp.pad(x, pad_width, mode="edge")
        elif padding_mode == imageruler.PaddingMode.SOLID:
            x_padded = onp.pad(x, pad_width, mode="constant", constant_values=True)
        elif padding_mode == imageruler.PaddingMode.VOID:
            x_padded = onp.pad(x, pad_width, mode="constant", constant_values=False)
        expected = ndimage.binary_opening(x_padded, kernel)
        expected = expected[
            pad_width[0][0] : expected.shape[0] - pad_width[0][1],
            pad_width[1][0] : expected.shape[1] - pad_width[1][1],
        ]
        actual = imageruler.binary_opening(
            x, kernel, periodic=(False, False), padding_mode=padding_mode
        )
        onp.testing.assert_array_equal(expected, actual)

    def test_detect_edges(self):
        x = onp.array(
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1],
                [0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1],
                [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ],
            dtype=bool,
        )
        expected = onp.array(
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1],
                [0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ],
            dtype=bool,
        )
        onp.testing.assert_array_equal(
            imageruler.detect_edges(x, periodic=(False, False)), expected
        )

    def test_detect_edges_periodic(self):
        x = onp.array(
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1],
                [0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1],
                [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ],
            dtype=bool,
        )
        expected = onp.array(
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1],
                [0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1],
                [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ],
            dtype=bool,
        )
        onp.testing.assert_array_equal(
            imageruler.detect_edges(x, periodic=(True, True)), expected
        )

    def test_opening_removes_small_features(self):
        # Test that a feature that is feasible with a size-4 brush is eliminated
        # by opening with the size-5 brush.
        actual = imageruler.binary_opening(
            TEST_ARRAY_4_5,
            TEST_KERNEL_5,
            periodic=(False, False),
            padding_mode=imageruler.PaddingMode.EDGE,
        )
        expected = onp.array(
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
            ],
            dtype=bool,
        )
        onp.testing.assert_array_equal(expected, actual)

    @parameterized.expand(
        [
            [[(0, 0, 1, 0, 0)], [(0, 0, 1, 0, 0)]],
            [[(0, 0, 1, 1, 0)], [(0, 0, 1, 1, 0)]],
            [[(0, 0, 1, 1, 1, 0, 0, 0)], [(0, 0, 0, 1, 0, 0, 0, 0)]],
            [[(0, 0, 0, 0, 0, 0, 0, 1)], [(0, 0, 0, 0, 0, 0, 0, 1)]],
        ]
    )
    def test_erode_large_features_1d(self, x, expected):
        result = imageruler.erode_large_features(
            onp.asarray(x, dtype=bool), periodic=(False, False)
        )
        onp.testing.assert_array_equal(result, onp.asarray(expected, dtype=bool))

    @parameterized.expand(
        [
            (
                [
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1],
                ],
                [
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1],
                ],
            ),
            (
                [
                    [1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 0, 0, 0, 0],
                    [1, 1, 1, 1, 0, 0, 0, 0],
                    [1, 1, 1, 1, 0, 0, 0, 0],
                    [1, 1, 0, 0, 1, 0, 0, 0],
                    [1, 1, 0, 0, 0, 0, 0, 0],
                ],
                [
                    [1, 1, 1, 1, 0, 1, 1, 1],
                    [1, 1, 1, 0, 0, 0, 0, 0],
                    [1, 1, 1, 0, 0, 0, 0, 0],
                    [1, 1, 0, 0, 0, 0, 0, 0],
                    [1, 0, 0, 0, 1, 0, 0, 0],
                    [1, 0, 0, 0, 0, 0, 0, 0],
                ],
            ),
        ]
    )
    def test_erode_large_features_2d(self, x, expected):
        result = imageruler.erode_large_features(
            onp.asarray(x, dtype=bool), periodic=(False, False)
        )
        onp.testing.assert_array_equal(result, onp.asarray(expected, dtype=bool))


class PaddingOperationsTest(unittest.TestCase):
    @parameterized.expand(
        [
            [((0, 0), (0, 0))],
            [((1, 5), (2, 4))],
            [((4, 2), (5, 1))],
        ],
    )
    def test_pad_2d_matches_numpy(self, pad_width):
        onp.random.seed(0)
        x = onp.random.rand(20, 30) > 0.5  # Random binary array.
        with self.subTest("edge"):
            expected = onp.pad(x, pad_width, mode="edge")
            actual = imageruler.pad_2d(
                x, pad_width, (False, False), imageruler.PaddingMode.EDGE
            )
            onp.testing.assert_array_equal(expected, actual)
        with self.subTest("solid"):
            expected = onp.pad(x, pad_width, constant_values=True)
            actual = imageruler.pad_2d(
                x, pad_width, (False, False), imageruler.PaddingMode.SOLID
            )
            onp.testing.assert_array_equal(expected, actual)
        with self.subTest("void"):
            expected = onp.pad(x, pad_width, constant_values=False)
            actual = imageruler.pad_2d(
                x, pad_width, (False, False), imageruler.PaddingMode.VOID
            )
            onp.testing.assert_array_equal(expected, actual)

    @parameterized.expand(
        [
            [((0, 0), (0, 0))],
            [((1, 5), (2, 4))],
            [((4, 2), (5, 1))],
        ],
    )
    def test_unpad(self, pad_width):
        x = onp.arange(200).reshape(10, 20)
        expected = x[
            pad_width[0][0] : x.shape[0] - pad_width[0][1],
            pad_width[1][0] : x.shape[1] - pad_width[1][1],
        ]
        actual = imageruler.unpad(x, pad_width)
        onp.testing.assert_equal(expected.shape, actual.shape)
        onp.testing.assert_array_equal(expected, actual)


# ------------------------------------------------------------------------------
# Tests for `maximum_true_arg`.
# ------------------------------------------------------------------------------


class MaximumTrueArgTest(unittest.TestCase):
    @parameterized.expand(
        [
            # 1  2  3  4  5  6  7  8  9  10 11
            ([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0], 1, 11, 1, 10),
            ([1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0], 1, 11, 1, 3),
            ([1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0], 1, 11, 11, 10),
            ([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1], 1, 8, 1, 0),  # No `True` values.
            ([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 1, 8, 1, 8),  # All `True` values.
        ]
    )
    def test_finds_maximum_true_arg(
        self,
        sequence,
        min_arg,
        max_arg,
        allowance,
        expected,
    ):
        fn = lambda i: bool(sequence[i - 1])
        actual = imageruler.maximum_true_arg(fn, min_arg, max_arg, allowance)
        assert expected == actual


class OneDimensionalPatternTest(unittest.TestCase):
    @parameterized.expand(
        [
            [[0, 0, 1, 1, 1, 0, 0, 0, 0, 0], False, (3, 10)],
            [[0, 0, 1, 1, 1, 0, 0, 0, 0, 0], True, (3, 7)],
            [[0, 0, 1, 1, 1, 0, 0, 0, 1, 1], True, (2, 2)],
            [[0, 1, 1, 1, 0, 0, 0, 1, 1, 0], True, (2, 2)],
            [[0, 1, 0, 1, 0, 0, 0, 1, 1, 0], True, (1, 1)],
            [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], False, (10, 10)],
            [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], True, (10, 10)],
            [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1], False, (10, 10)],
            [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1], True, (10, 10)],
        ]
    )
    def test_minimum_feature_size_matches_expected(
        self, pattern, periodic, expected_minimum_length
    ):
        minimum_length = imageruler.minimum_length_scale_1d(
            onp.asarray(pattern, dtype=bool), periodic
        )
        onp.testing.assert_array_equal(minimum_length, expected_minimum_length)
