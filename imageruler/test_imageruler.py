import numpy as np
import unittest
import imageruler
from regular_shapes import disc, rounded_square

resolution = 1
phys_size = (200, 200)
diameters = np.linspace(2, 200, 11)


def duality_dilation_erosion(pattern, diameter):
    """
    Test whether the duality relations between dilation and erosion are satisfied.

    Args:
        arr: A 2d array.
        diameter: A float that represents the diameter of the kernel.

    Returns:
        A Boolean value that indicates whether the duality relations between erosion and dilation are satisfied.
    """
    erosion_of_negation = imageruler.binary_erode(~pattern, diameter)
    negation_of_dilation = ~imageruler.binary_dilate(pattern, diameter)
    dilation_of_negation = imageruler.binary_dilate(~pattern, diameter)
    negation_of_erosion = ~imageruler.binary_erode(pattern, diameter)
    return (erosion_of_negation
            == negation_of_dilation).all() and (negation_of_erosion
                                                == dilation_of_negation).all()


def duality_closing_opening(pattern, diameter):
    """
    Test whether the duality relations between closing and opening are satisfied.

    Args:
        arr: A 2d array.
        diameter: A float that represents the diameter of the kernel.

    Returns:
        A Boolean value that indicates whether the duality relations between closing and opening are satisfied.
    """
    opening_of_negation = imageruler.binary_open(~pattern, diameter)
    negation_of_closing = ~imageruler.binary_close(pattern, diameter)
    closing_of_negation = imageruler.binary_close(~pattern, diameter)
    negation_of_opening = ~imageruler.binary_open(pattern, diameter)
    return (opening_of_negation
            == negation_of_closing).all() and (closing_of_negation
                                               == negation_of_opening).all()


class TestDuality(unittest.TestCase):
    def test_rounded_square(self):
        print("------ Testing duality on rounded squares ------")

        for angle in range(0, 90, 10):
            print("Rotation angle of the rounded square: ", angle)
            pattern = rounded_square(resolution, phys_size, 50, angle)
            for diameter in diameters:
                print("Kernel diameter: " + str(diameter))
                assert duality_dilation_erosion(
                    pattern, diameter
                ), 'The duality between dilation and erosion does not hold.'
                assert duality_closing_opening(
                    pattern, diameter
                ), 'The duality between closing and opening does not hold.'

    def test_disc(self):
        print("------ Testing duality on a disc ------")

        pattern = disc(resolution, phys_size, 50)
        for diameter in diameters:
            print("Kernel diameter: " + str(diameter))
            assert duality_dilation_erosion(
                pattern, diameter
            ), 'The duality between dilation and erosion does not hold.'
            assert duality_closing_opening(
                pattern, diameter
            ), 'The duality between closing and opening does not hold.'

    def test_ring(self):
        print("------ Testing duality on concentric circles ------")
        outer_diameter, inner_diameter = 120, 50
        declared_solid_mls, declared_void_mls = (
            outer_diameter - inner_diameter) / 2, inner_diameter
        print("Declared minimum length scale: ", declared_solid_mls,
              "(solid), ", declared_void_mls, "(void)")

        solid_disc = disc(resolution, phys_size, diameter=outer_diameter)
        void_disc = disc(resolution, phys_size, diameter=inner_diameter)
        pattern = solid_disc ^ void_disc  # ring

        for diameter in diameters:
            print("Kernel diameter: " + str(diameter))
            assert duality_dilation_erosion(
                pattern, diameter
            ), 'The duality between dilation and erosion does not hold.'
            assert duality_closing_opening(
                pattern, diameter
            ), 'The duality between closing and opening does not hold.'


message = "The estimated minimum length scale is too far from the declared minimum length scale."


class TestMinimumLengthScale(unittest.TestCase):
    def test_rounded_square(self):
        print("------ Testing minimum length scale on rounded squares ------")
        declared_mls = 50
        print("Declared minimum length scale: ", declared_mls)
        delta = 4

        for angle in range(0, 90, 10):
            pattern = rounded_square(resolution, phys_size, declared_mls,
                                     angle)
            solid_mls = imageruler.minimum_length_solid(pattern)
            print("Rotation angle of the rounded square: ", angle)
            print("Estimated minimum length scale: ", solid_mls)
            # check if values are almost equal
            self.assertAlmostEqual(solid_mls, declared_mls, None, message,
                                   delta)

    def test_disc(self):
        print("------ Testing minimum length scale on a disc ------")
        diameter = 50
        print("Declared minimum length scale: ", diameter)
        pattern = disc(resolution, phys_size, diameter)
        solid_mls = imageruler.minimum_length_solid(pattern, phys_size)
        print("Estimated minimum length scale: ", solid_mls)
        # check if values are almost equal
        self.assertAlmostEqual(solid_mls, diameter, None, message, delta=1)

    def test_ring(self):
        print("------ Testing minimum length scale on concentric circles ------")
        outer_diameter, inner_diameter = 120, 50
        declared_solid_mls, declared_void_mls = (
            outer_diameter - inner_diameter) / 2, inner_diameter
        print("Declared minimum length scale: ", declared_solid_mls,
              "(solid), ", declared_void_mls, "(void)")

        solid_disc = disc(resolution, phys_size, diameter=outer_diameter)
        void_disc = disc(resolution, phys_size, diameter=inner_diameter)
        pattern = solid_disc ^ void_disc  # ring

        solid_mls = imageruler.minimum_length_solid(pattern)
        void_mls = imageruler.minimum_length_void(pattern)
        dual_mls = imageruler.minimum_length(pattern)
        print("Estimated minimum length scale: ", solid_mls, "(solid), ",
              void_mls, "(void)")

        delta = 1
        # check if values are almost equal
        self.assertAlmostEqual(solid_mls, declared_solid_mls, None, message,
                               delta)
        self.assertAlmostEqual(void_mls, declared_void_mls, None, message,
                               delta)
        self.assertAlmostEqual(dual_mls,
                               min(declared_solid_mls, declared_void_mls),
                               None, message, delta)
        self.assertEqual(dual_mls, min(solid_mls, void_mls))


if __name__ == "__main__":
    unittest.main()