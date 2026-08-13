"""
stage2_bilateral_filter.py — Edge-preserving denoising

WHY WE NEED THIS
─────────────────────────────────────────────────────────────────────────────
Underwater images contain speckle noise from light scattering through
suspended water particles. A standard Gaussian blur would remove that noise,
but it also blurs coral texture edges — and those edges are EXACTLY what the
CNN needs to distinguish healthy vs bleached coral (healthy coral has
regular ridge/polyp patterns; bleached coral loses that structural detail).
Denoising and edge-preservation are in direct tension here, which is why we
can't just use a cheap Gaussian blur.

HOW BILATERAL FILTERING WORKS
─────────────────────────────────────────────────────────────────────────────
A normal blur averages every pixel with ALL its spatial neighbors, full
stop. Bilateral filtering adds a second condition: it only averages
neighbors that are ALSO similar in intensity/color to the center pixel.

Two parameters control this:
    sigmaSpace — how far away (in pixels) a neighbor can be and still count
    sigmaColor — how DIFFERENT in intensity a neighbor can be and still count

At an edge (e.g. the boundary of a coral branch), pixels on one side are
bright and pixels on the other are dark — sigmaColor excludes them from
averaging with each other, so the edge survives sharply. In a flat, uniform
region (open water, smooth coral surface), neighboring pixels are all
similar, so the filter averages freely and removes noise there.

    d=5           — diameter of the pixel neighbourhood considered
    sigmaColor=40 — intensity similarity tolerance
                    (higher = smoother but less edge-preserving)
    sigmaSpace=40 — spatial distance tolerance
                    (higher = farther pixels can influence the result)
These are gentle values — safe for microscopic ridge textures, since a more
aggressive setting risks smoothing away the exact patterns the CNN relies on.
"""

import cv2
import numpy as np


def apply_bilateral_filter(img_bgr: np.ndarray, d: int = 5,
                            sigma_color: int = 40, sigma_space: int = 40) -> np.ndarray:
    """
    Args:
        img_bgr: Blue-shift corrected BGR image, uint8 (run stage 1 first)
    Returns:
        Noise-reduced BGR image, uint8, with edges preserved
    """
    return cv2.bilateralFilter(img_bgr, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)


# ─────────────────────────────────────────────────────────────────────────────
# Sample test case — run this file directly: python stage2_bilateral_filter.py
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    import os

    if len(sys.argv) > 1:
        img = cv2.imread(sys.argv[1])
        if img is None:
            raise FileNotFoundError(f"Could not load image: {sys.argv[1]}")
    else:
        print("No image path given — generating a synthetic noisy test image with a sharp edge.")
        # Half-black / half-white image with added Gaussian noise — lets us
        # directly check whether the edge in the middle survives filtering.
        img = np.zeros((300, 300, 3), dtype=np.uint8)
        img[:, 150:, :] = 255
        noise = np.random.normal(0, 25, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    gaussian = cv2.GaussianBlur(img, (9, 9), 0)
    bilateral = apply_bilateral_filter(img)

    # Compare edge SPREAD (not just jump size): walk a row across the
    # boundary and count how many pixels it takes to go from 10% to 90% of
    # the full black->white swing. A blurred edge spreads over more pixels;
    # a preserved edge stays a narrow, near-vertical transition.
    def transition_width(gray_row, lo_frac=0.1, hi_frac=0.9):
        lo, hi = gray_row.min(), gray_row.max()
        lo_val, hi_val = lo + (hi - lo) * lo_frac, lo + (hi - lo) * hi_frac
        idx_lo = np.argmax(gray_row > lo_val)
        idx_hi = np.argmax(gray_row > hi_val)
        return abs(idx_hi - idx_lo)

    gray_gauss = cv2.cvtColor(gaussian, cv2.COLOR_BGR2GRAY).astype(float)
    gray_bilat = cv2.cvtColor(bilateral, cv2.COLOR_BGR2GRAY).astype(float)
    row = 150
    width_gauss = transition_width(gray_gauss[row])
    width_bilat = transition_width(gray_bilat[row])

    print(f"Edge transition width after Gaussian blur : {width_gauss} px")
    print(f"Edge transition width after bilateral filter: {width_bilat} px")
    print("Expect: bilateral width should be noticeably SMALLER (sharper, less spread edge).")

    os.makedirs("test_outputs", exist_ok=True)
    cv2.imwrite("test_outputs/stage2_original.jpg", img)
    cv2.imwrite("test_outputs/stage2_gaussian.jpg", gaussian)
    cv2.imwrite("test_outputs/stage2_bilateral.jpg", bilateral)
    print("Saved test_outputs/stage2_original.jpg, stage2_gaussian.jpg, stage2_bilateral.jpg")
