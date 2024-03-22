"""Tests for imageruler based on regular shapes."""

import unittest

import numpy as np
from parameterized import parameterized

import imageruler
from imageruler import imageruler
from imageruler import regular_shapes as shapes

# properties of design weights array
RESOLUTION = 1
PHYS_SIZE = (200, 200)


class TestDuality(unittest.TestCase):
    def duality_dilation_erosion(self, weights: np.ndarray, diam: int) -> bool:
        """Checks the duality of the dilation and erosion operators for solid

        and void features of the design weights.

        Args:
            weights: the design weights as a 2d array.
            diam: the diameter of the circular-ball kernel.

        Returns:
            A Boolean indicating whether duality is satisfied or not.
        """
        kernel = imageruler.kernel_for_length_scale(diam)
        erosion_of_negation = imageruler.binary_erosion(
            ~weights,
            kernel,
            periodic=(False, False),
            padding_mode=imageruler.PaddingMode.SOLID,
        )
        negation_of_dilation = ~imageruler.binary_dilation(
            weights,
            kernel,
            periodic=(False, False),
            padding_mode=imageruler.PaddingMode.VOID,
        )
        dilation_of_negation = imageruler.binary_dilation(
            ~weights,
            kernel,
            periodic=(False, False),
            padding_mode=imageruler.PaddingMode.SOLID,
        )
        negation_of_erosion = ~imageruler.binary_erosion(
            weights,
            kernel,
            periodic=(False, False),
            padding_mode=imageruler.PaddingMode.VOID,
        )
        return (erosion_of_negation == negation_of_dilation).all() and (
            negation_of_erosion == dilation_of_negation
        ).all()

    @parameterized.expand((3, 45, 51, 81, 101))
    def test_rounded_square(self, length_scale):
        print("------ Testing duality property using rounded squares ------")

        declared_mls: float = 50.0
        for angle in [0, 10.5, 44.3, 73.9]:
            print(f"Rotation angle of the rounded square: {angle}°")
            pattern = shapes.rounded_square(
                RESOLUTION, PHYS_SIZE, declared_mls, angle=angle
            )
            print(f"Kernel diameter: {length_scale:.2f}")
            self.assertTrue(self.duality_dilation_erosion(pattern, length_scale))

    @parameterized.expand((3, 45, 51, 81, 101))
    def test_disc(self, length_scale):
        print("------ Testing duality property using a shapes.disc ------")

        diam = 50.0
        pattern = shapes.disc(RESOLUTION, PHYS_SIZE, diam)
        print(f"Kernel diameter: {length_scale:.2f}")
        self.assertTrue(self.duality_dilation_erosion(pattern, length_scale))

    @parameterized.expand((3, 45, 51, 81, 101))
    def test_ring(self, length_scale):
        print("------ Testing duality property using concentric circles ------")

        outer_diameter, inner_diameter = 120, 50
        declared_solid_mls, declared_void_mls = (
            outer_diameter - inner_diameter
        ) / 2, inner_diameter
        print(
            f"Declared minimum length scale: {declared_solid_mls:.2f} "
            f"(solid), {declared_void_mls:.2f} (void)"
        )

        solid_disc = shapes.disc(RESOLUTION, PHYS_SIZE, diameter=outer_diameter)
        void_disc = shapes.disc(RESOLUTION, PHYS_SIZE, diameter=inner_diameter)
        pattern = solid_disc ^ void_disc  # ring

        print(f"Kernel diameter: {length_scale:.2f}")
        self.assertTrue(self.duality_dilation_erosion(pattern, length_scale))


class TestMinimumLengthScale(unittest.TestCase):
    def test_rounded_square(self):
        print("------ Testing minimum length scale of rounded squares ------")

        declared_mls = 50
        print("Declared minimum length scale: ", declared_mls)

        for angle in [5.6, 25.2, 49.3, 69.5]:
            pattern = shapes.rounded_square(
                RESOLUTION, PHYS_SIZE, declared_mls, angle=angle
            )
            solid_mls = imageruler.minimum_length_scale_solid(pattern)
            print(f"Rotation angle of the rounded square: {angle}°")
            print(f"Estimated minimum length scale: {solid_mls:.2f}")
            self.assertAlmostEqual(solid_mls, declared_mls, delta=4)

    def test_disc(self):
        print("------ Testing minimum length scale of a shapes.disc ------")

        diameter = 50
        print(f"Declared minimum length scale: {diameter}")
        pattern = shapes.disc(RESOLUTION, PHYS_SIZE, diameter)
        solid_mls = imageruler.minimum_length_scale_solid(pattern)
        print(f"Estimated minimum length scale: {solid_mls:.2f}")
        self.assertAlmostEqual(solid_mls, diameter, delta=1)

    def test_ring(self):
        print("------ Testing minimum length scale of concentric circles ------")

        outer_diameter, inner_diameter = 120, 50
        declared_solid_mls, declared_void_mls = (
            outer_diameter - inner_diameter
        ) / 2, inner_diameter
        print(
            f"Declared minimum length scale: {declared_solid_mls:.2f} "
            f"(solid), {declared_void_mls:.2f} (void)"
        )

        solid_disc = shapes.disc(RESOLUTION, PHYS_SIZE, diameter=outer_diameter)
        void_disc = shapes.disc(RESOLUTION, PHYS_SIZE, diameter=inner_diameter)
        pattern = solid_disc ^ void_disc  # ring

        solid_mls = imageruler.minimum_length_scale_solid(pattern)
        void_mls = imageruler.minimum_length_scale_solid(~pattern)
        print(
            f"Estimated minimum length scale: {solid_mls:.2f} (solid), ",
            f"{void_mls:.2f} (void)",
        )

        self.assertAlmostEqual(solid_mls, declared_solid_mls, delta=1)
        self.assertAlmostEqual(void_mls, declared_void_mls, delta=1)

    def test_periodicity(self):
        print("------ Testing minimum length scale on a periodic image ------")

        shapes.stripe_width = 50
        print(f"Declared minimum length scale: {shapes.stripe_width}")
        pattern = shapes.stripe(
            RESOLUTION, PHYS_SIZE, shapes.stripe_width, center=(0, -PHYS_SIZE[1] / 2)
        ) | shapes.stripe(
            RESOLUTION, PHYS_SIZE, shapes.stripe_width, center=(0, PHYS_SIZE[1] / 2)
        )
        solid_mls = imageruler.minimum_length_scale_solid(
            pattern, periodic=(False, True)
        )
        print(f"Estimated minimum length scale: {solid_mls:.2f}")
        self.assertAlmostEqual(solid_mls, shapes.stripe_width, delta=1)
