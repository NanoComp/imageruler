"""Command line interface for imageruler.

Invoke by running `python cli` from the imageruler directory.
"""

import argparse
import imageruler
import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description=(
            "A python package for estimating minimum lengthscales in binary" " images"
        )
    )
    parser.add_argument(
        "file", type=str, help="Path of the text file containing the image array."
    )
    args = parser.parse_args()

    solid_mls, void_mls = imageruler.minimum_length_solid_void(np.loadtxt(args.file))
    print("Minimum lengthscales of solid and void regions: ", solid_mls, void_mls)


if __name__ == "__main__":
    main()
