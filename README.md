[![Build Status](https://github.com/NanoComp/imageruler/workflows/CI/badge.svg)](https://github.com/NanoComp/imageruler/actions)
`v0.3.0`

Imageruler is a free Python program to compute the minimum length scale of binary images which are typically designs produced by topology optimization. The algorithm is described in Section 2 of [J. Optical Society of America B, Vol. 42, pp. A161-A176 (2024)](https://opg.optica.org/josab/abstract.cfm?uri=josab-41-2-A161) and is based on morphological transformations implemented using [OpenCV](https://github.com/opencv/opencv). Imageruler also supports 1d binary images.

## Algorithm for Determining Minimum Length Scale

The procedure used by Imageruler for determining the minimum length scale of the solid regions in a binary image involves four steps:

1. Binarize the 1d or 2d array $\rho$ representing the image such that each of its elements is a Boolean value for solid (true) and void (false).
2. For a circular-ball [kernel](https://en.wikipedia.org/wiki/Kernel_(image_processing)) with diameter $d$, compute the morphological opening $\mathcal{O}_d(\rho)$ and obtain its difference with the original array via $\mathcal{O}_d(\rho) \oplus \rho$, where $\oplus$ denotes the exclusive-or operator. In the strictest sense, solid pixels in $\mathcal{O}_d(\rho) \oplus \rho$ are violations of the length scale $d$.
3. Identify pixels where violations are to be ignored. The scheme used to identify ignored violating pixels is specified by the `IgnoreScheme`; by default, these include pixels at the edges of large features. Remove any violations to be ignored from consideration.
4. Check whether there are any remaining violations in $\mathcal{O}_d(\rho) \oplus \rho$. If there are no violations pixels, $d$ is less than or equal to the minimum length scale of solid regions.
5. Use a [binary search](https://en.wikipedia.org/wiki/Binary_search_algorithm) and repeat Steps 2 and 3 to find the largest $d$ for which no violating pixels exist. The search has some allowance for non-monotonicity, i.e. situations where there are violations for $d$ but not $d + 1$. This largest value of $d$ for which there are no violations is considered to be the minimum length scale of the solid regions.

To estimate the minimum length scale of the void regions, the binary image is inverted after the binarization of Step 1: $\rho \rightarrow \neg \rho$ such that the solid and void regions are interchanged. The remaining Steps 2-5 are unchanged. This approach is equivalent to computing $\mathcal{C}_d(\rho) \oplus \rho$ in Step 2 and then checking its overlap with the interior pixels of the void regions of $\rho$ in Step 3. $\mathcal{C}_d(\rho)$ denotes morphological closing.

The minimum length scale of $\rho$ is the smaller of the minimum length scales of the solid and void regions. Rather than determining these separately, it is possible in principle to compute their minimum simultaneously using $\mathcal{O}_d(\rho) \oplus \mathcal{C}_d(\rho)$ and then to check its overlap with the union of the interior pixels of the solid and void regions of $\rho$. This approach involves a single binary search rather than two.

## Schemes for Identifying Ignored Violations
The `ignore_scheme` is an optional argument to top-level functions such as `imageruler.minimum_length_scale`. The choice may affect the length scale value reported for a given design; the possible values are as follows:

- `IgnoreScheme.NONE`: a strict scheme in which no violations are ignored.
- `IgnoreScheme.EDGES`: ignores violations for any solid pixel removed by erosion.
- `IgnoreScheme.LARGE_FEATURE_EDGES`: ignores violations at the edges of large features only. A pixel is on the edge of a large feature if it removed by erosion and adjacent to an interior pixel. Interior pixels are those which are solid and surrounded on all sides (in an 8-connected sense) by solid pixels.
- `IgnoreScheme.LARGE_FEATURE_EDGES_STRICT`: the default choice. Similar to `LARGE_FEATURE_EDGES`, but uses a more strict algorithm to detect edges and does not ignore checkerboard patterns.

## Note on Accuracy

The accuracy of the minimum length scale computed by Imageruler is limited by the finite resolution of the input image. A fundamental feature is that length scales smaller than a single pixel cannot be measured. Also, in certain situations, length scales of a few pixels may be indistinguishable from discretization artifacts such as the "staircasing" of curved surfaces. As an example, a sharp 90° corner corresponds to a length scale (radius of curvature) of *zero* at infinite resolution. However, at a finite resolution, a sharp corner is indistinguishable from one that is rounded with a radius of curvature of a few pixels. Imageruler will therefore return a length scale proportional to the pixel size rather than zero. In general, the measured length scale should be viewed as having an "error bar" on the few-pixel level.
