[![Build Status](https://github.com/NanoComp/imageruler/workflows/CI/badge.svg)](https://github.com/NanoComp/imageruler/actions)

Imageruler is a free Python program to compute the minimum length scale of binarized images which are typically designs produced by topology optimization. The algorithm is based on morphological transformations [1,2] as implemented using the OpenCV library [3]. Imageruler also supports 1d designs.

For examples of using Imageruler on a variety of structures, see this [notebook](notebooks/examples.ipynb). Documentation is currently provided by the docstrings. A user manual is under development.

## Algorithm for Determining Minimum Length Scale

The procedure used by Imageruler for determining the minimum length scale of the solid regions in a binary image involves four steps:

1. Binarize the 2d array $\rho$ representing the image such that each of its elements is a Boolean value for solid (true) and void (false).
2. For a circular [kernel](https://en.wikipedia.org/wiki/Kernel_(image_processing)) with diameter $d$, compute the morphological opening $\mathcal{O}_d(\rho)$ and obtain its difference with the original array via $\mathcal{O}_d(\rho) \oplus \rho$, where $\oplus$ denotes the exclusive-or operator.
3. Check whether $\mathcal{O}_d(\rho) \oplus \rho$ contains a solid pixel within the interior solid regions of $\rho$. If no, $d$ is less than the minimum length scale of solid regions. If yes, $d$ is equal or greater than the minimum length scale of the solid regions. The interior of the solid regions of $\rho$ is obtained by morphological erosion using a "cross" kernel of size $3\times3$ pixels.
4. Use a [binary search](https://en.wikipedia.org/wiki/Binary_search_algorithm) and repeat Steps 2 and 3 to find the smallest $d$ for which the check in Step 3 evaluates to true. This value of $d$ is considered to be the minimum length scale of the solid regions.

To estimate the minimum length scale of the void regions, the binary image is inverted after the binarization of Step 1: $\rho \rightarrow \neg \rho$ such that the solid and void regions are interchanged. The remaining Steps 2-4 are unchanged. This approach is equivalent to computing $\mathcal{C}_d(\rho) \oplus \rho$ in Step 2 and then checking its overlap with the interior pixels of the void regions of $\rho$ in Step 3. $\mathcal{C}_d(\rho)$ denotes morphological closing.

The minimum length scale of $\rho$ is the smaller of the minimum length scales of the solid and void regions which must be determined separately. A more efficient approach is to compute $\mathcal{O}_d(\rho) \oplus \mathcal{C}_d(\rho)$ in Step 2 and then to check its overlap with the union of the interior pixels of the solid and void regions of $\rho$ in Step 3. This approach involves a single binary search rather than two.

For a 1d binary image, the algorithm simply finds the minimum length among all solid or void segments.

## Accuracy

The accuracy of the minimum length scale computed by Imageruler is limited by the finite resolution of the input image. A fundamental feature is that length scales smaller than a single pixel cannot be measured. Also, in certain situations, length scales of a few pixels may be indistinguishable from discretization artifacts such as the "staircasing" of curved surfaces. As one example, a sharp 90° corner corresponds to a length scale (radius of curvature) of *zero* at infinite resolution. However, at a finite resolution, a sharp corner will tend to be rounded with a radius of curvature of a few pixels. In this case, the sharp and rounded corners are indistinguishable and thus Imageruler will return a length scale proportional to the pixel size. In general, the measured length scale should be viewed as having an "error bar" on the few-pixel level.

## References

[1] Linus Hägg and Eddie Wadbro, On minimum length scale control in density based topology optimization, Struct. Multidisc Optim. 58(3), 1015–1032 (2018).  
[2] Rafael C. Gonzalez and Richard E. Woods, Digital Image Processing (Fourth Edition), Chapter 9 (Pearson, 2017).  
[3] Gary Bradski. The OpenCV Library. Dr. Dobb’s Journal of Software Tools (2000).
