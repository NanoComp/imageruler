"""Tests for imageruler."""

import unittest
import imageruler
import numpy as np

from imageruler import imageruler
from imageruler import regular_shapes as shapes


# properties of design weights array
resolution = 1
phys_size = (200, 200)


class TestDuality(unittest.TestCase):
    diameters = [2.2, 45.2, 50.1, 80.2, 101.5]

    def duality_dilation_erosion(self, weights: np.ndarray, diam: float) -> bool:
        """Checks the duality of the dilation and erosion operators for solid

        and void features of the design weights.

        Args:
            weights: the design weights as a 2d array.
            diam: the diameter of the circular-ball kernel.

        Returns:
            A Boolean indicating whether duality is satisfied or not.
        """
        kernel = imageruler.get_kernel(diam)
        erosion_of_negation = imageruler.binary_erode(~weights, kernel)
        negation_of_dilation = ~imageruler.binary_dilate(weights, kernel)
        dilation_of_negation = imageruler.binary_dilate(~weights, kernel)
        negation_of_erosion = ~imageruler.binary_erode(weights, kernel)
        return (erosion_of_negation == negation_of_dilation).all() and (
            negation_of_erosion == dilation_of_negation
        ).all()

    def duality_closing_opening(self, weights: np.ndarray, diam: float) -> bool:
        """Checks the duality of the opening and closing morphological transform

        operators for solid and void features of the design weights.

        Args:
            weights: the design weights as a 2d array.
            diam: the diameter of the circular-ball kernel.

        Returns:
            A Boolean indicating whether duality is satisfied or not.
        """
        kernel = imageruler.get_kernel(diam)
        opening_of_negation = imageruler.binary_open(~weights, kernel)
        negation_of_closing = ~imageruler.binary_close(weights, kernel)
        closing_of_negation = imageruler.binary_close(~weights, kernel)
        negation_of_opening = ~imageruler.binary_open(weights, kernel)
        return (opening_of_negation == negation_of_closing).all() and (
            closing_of_negation == negation_of_opening
        ).all()

    def test_rounded_square(self):
        print("------ Testing duality property using rounded squares ------")

        declared_mls: float = 50.0
        for angle in [0, 10.5, 44.3, 73.9]:
            print(f"Rotation angle of the rounded square: {angle}°")
            pattern = shapes.rounded_square(
                resolution, phys_size, declared_mls, angle=angle
            )
            for diameter in self.diameters:
                print(f"Kernel diameter: {diameter:.2f}")
                self.assertTrue(self.duality_dilation_erosion(pattern, diameter))
                self.assertTrue(self.duality_closing_opening(pattern, diameter))

    def test_disc(self):
        print("------ Testing duality property using a shapes.disc ------")

        diam = 50.0
        pattern = shapes.disc(resolution, phys_size, diam)
        for diameter in self.diameters:
            print(f"Kernel diameter: {diameter:.2f}")
            self.assertTrue(self.duality_dilation_erosion(pattern, diameter))
            self.assertTrue(self.duality_closing_opening(pattern, diameter))

    def test_ring(self):
        print("------ Testing duality property using concentric circles ------")

        outer_diameter, inner_diameter = 120, 50
        declared_solid_mls, declared_void_mls = (
            outer_diameter - inner_diameter
        ) / 2, inner_diameter
        print(
            f"Declared minimum length scale: {declared_solid_mls:.2f} "
            f"(solid), {declared_void_mls:.2f} (void)"
        )

        solid_disc = shapes.disc(resolution, phys_size, diameter=outer_diameter)
        void_disc = shapes.disc(resolution, phys_size, diameter=inner_diameter)
        pattern = solid_disc ^ void_disc  # ring

        for diameter in self.diameters:
            print(f"Kernel diameter: {diameter:.2f}")
            self.assertTrue(self.duality_dilation_erosion(pattern, diameter))
            self.assertTrue(self.duality_closing_opening(pattern, diameter))


class TestMinimumLengthScale(unittest.TestCase):
    def test_rounded_square(self):
        print("------ Testing minimum length scale of rounded squares ------")

        declared_mls = 50
        print("Declared minimum length scale: ", declared_mls)

        for angle in [5.6, 25.2, 49.3, 69.5]:
            pattern = shapes.rounded_square(
                resolution, phys_size, declared_mls, angle=angle
            )
            solid_mls = imageruler.minimum_length_solid(pattern)
            print(f"Rotation angle of the rounded square: {angle}°")
            print(f"Estimated minimum length scale: {solid_mls:.2f}")
            self.assertAlmostEqual(solid_mls, declared_mls, delta=4)

    def test_disc(self):
        print("------ Testing minimum length scale of a shapes.disc ------")

        diameter = 50
        print(f"Declared minimum length scale: {diameter}")
        pattern = shapes.disc(resolution, phys_size, diameter)
        solid_mls = imageruler.minimum_length_solid(pattern, phys_size)
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

        solid_disc = shapes.disc(resolution, phys_size, diameter=outer_diameter)
        void_disc = shapes.disc(resolution, phys_size, diameter=inner_diameter)
        pattern = solid_disc ^ void_disc  # ring

        solid_mls = imageruler.minimum_length_solid(pattern)
        void_mls = imageruler.minimum_length_void(pattern)
        dual_mls = imageruler.minimum_length(pattern)
        print(
            f"Estimated minimum length scale: {solid_mls:.2f} (solid), ",
            f"{void_mls:.2f} (void)",
        )

        self.assertAlmostEqual(solid_mls, declared_solid_mls, delta=1)
        self.assertAlmostEqual(void_mls, declared_void_mls, delta=1)
        self.assertAlmostEqual(
            dual_mls, min(declared_solid_mls, declared_void_mls), delta=1
        )
        self.assertEqual(dual_mls, min(solid_mls, void_mls))

    def test_periodicity(self):
        print("------ Testing minimum length scale on a periodic image ------")

        shapes.stripe_width = 50
        print(f"Declared minimum length scale: {shapes.stripe_width}")
        pattern = shapes.stripe(
            resolution, phys_size, shapes.stripe_width, center=(0, -phys_size[1] / 2)
        ) | shapes.stripe(
            resolution, phys_size, shapes.stripe_width, center=(0, phys_size[1] / 2)
        )
        solid_mls = imageruler.minimum_length_solid(pattern, phys_size, periodic_axes=1)
        print(f"Estimated minimum length scale: {solid_mls:.2f}")
        self.assertAlmostEqual(solid_mls, shapes.stripe_width, delta=1)
