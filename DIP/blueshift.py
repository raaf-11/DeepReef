"""
stage1_blue_shift.py — Underwater blue-shift correction

WHY WE NEED THIS
─────────────────────────────────────────────────────────────────────────────
Water absorbs light unevenly by wavelength. Red light (long wavelength, low
energy) is absorbed first and is almost entirely gone by ~5m depth. Blue
light (short wavelength) penetrates deepest. So the deeper/murkier the water,
the more the ENTIRE photo shifts toward blue-green — regardless of what
color the coral actually is. If we feed this raw blue-tinted image straight
into a CNN, the model has to separately learn "ignore depth-dependent color
cast" on top of "detect bleaching," which wastes model capacity and hurts
generalization across photos taken at different depths/turbidity.

THE FIX — Gray World assumption in LAB colorspace
─────────────────────────────────────────────────────────────────────────────
We convert to LAB colorspace instead of correcting RGB directly, because LAB
cleanly separates:
    L = Lightness        (brightness only, untouched here)
    A = green <-> red axis
    B = blue <-> yellow axis
Editing A and B only re-balances COLOR without touching brightness — you
can't do this cleanly in RGB because all 3 channels mix color and brightness
together.

The "Gray World" assumption says: averaged over a natural scene, colors
should roughly balance out to neutral gray (A ≈ 128, B ≈ 128 is the "no
color cast" center in 8-bit LAB). So we just measure how far the image's
average A/B has drifted from 128, and shift the whole image back by that
amount.

LIMITATION TO KNOW: this assumes the average scene content is roughly
color-neutral. A shot that's genuinely dominated by one true color (e.g.
a frame that's almost entirely yellow sponge, no coral, no background)
will get incorrectly pulled toward gray. It's a cheap, calibration-free
heuristic — not a physically exact color correction.
"""

import cv2
import numpy as np


def correct_blue_shift(img_bgr: np.ndarray) -> np.ndarray:
    """
    Args:
        img_bgr: OpenCV BGR image, uint8
    Returns:
        Blue-shift corrected BGR image, uint8
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    L, A, B = cv2.split(lab)

    # Re-center A and B channels so their mean sits at neutral (128)
    A_corrected = np.clip(A - A.mean() + 128.0, 0, 255)
    B_corrected = np.clip(B - B.mean() + 128.0, 0, 255)

    lab_corrected = cv2.merge([L, A_corrected, B_corrected]).astype(np.uint8)
    return cv2.cvtColor(lab_corrected, cv2.COLOR_LAB2BGR)


# ─────────────────────────────────────────────────────────────────────────────
# Sample test case — run this file directly: python stage1_blue_shift.py
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    import os

    # Usage: python stage1_blue_shift.py path/to/coral_photo.jpg
    # If no path is given, we synthesize a fake "blue-tinted" test image so
    # the script still runs end-to-end without any external file.
    if len(sys.argv) > 1:
        path = sys.argv[1]
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Could not load image: {path}")
    else:
        print("No image path given — generating a synthetic blue-tinted test image.")
        # A neutral gray 300x300 image, artificially pushed blue —
        # simulates what a real underwater photo's color cast looks like.
        img = np.full((300, 300, 3), 150, dtype=np.uint8)  # neutral gray BGR
        img[:, :, 0] = np.clip(img[:, :, 0].astype(int) + 60, 0, 255)  # boost Blue channel
        img[:, :, 2] = np.clip(img[:, :, 2].astype(int) - 40, 0, 255)  # reduce Red channel

    corrected = correct_blue_shift(img)

    b, g, r = img[:, :, 0].mean(), img[:, :, 1].mean(), img[:, :, 2].mean()
    cb, cg, cr = corrected[:, :, 0].mean(), corrected[:, :, 1].mean(), corrected[:, :, 2].mean()

    print(f"Before correction — mean B:{b:.1f} G:{g:.1f} R:{r:.1f}")
    print(f"After  correction — mean B:{cb:.1f} G:{cg:.1f} R:{cr:.1f}")
    print("Expect: B/R gap should shrink after correction (closer to neutral).")

    os.makedirs("test_outputs", exist_ok=True)
    cv2.imwrite("test_outputs/stage1_before.jpg", img)
    cv2.imwrite("test_outputs/stage1_after.jpg", corrected)
    print("Saved test_outputs/stage1_before.jpg and stage1_after.jpg")
