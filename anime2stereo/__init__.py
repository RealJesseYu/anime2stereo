"""anime2stereo: lift 2D cel animation into stereoscopic 3D.

Pipeline (per shot):
    frames -> deflicker -> global-motion (pan) estimation
           -> clean background plate (temporal median + inpainting)
           -> cel segmentation (frame vs. plate residual, line-art aware refinement)
           -> depth (ordinal cel layers | monocular ML | hybrid)
           -> layered depth-image-based rendering (left/right eye)
           -> side-by-side / anaglyph / wiggle output
"""

__version__ = "0.1.0"
