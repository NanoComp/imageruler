"""Command line interface for imageruler.

Invoke by running `python imageruler FILENAME` from the imageruler directory.
"""

import argparse

import numpy as onp

import imageruler


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "A python package for estimating minimum lengthscales in binary" " images"
        )
    )
    parser.add_argument(
        "file", type=str, help="Path of the text file containing the image array."
    )
    args = parser.parse_args()

    design = onp.loadtxt(args.file, delimiter=",")
    binarized_design = design > 0.5
    solid_mls, void_mls = imageruler.minimum_length_scale(binarized_design)
    print("Minimum lengthscales of solid and void regions: ", solid_mls, void_mls)


if __name__ == "__main__":
    main()
