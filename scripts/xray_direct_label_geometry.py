"""Measure boxes around visually selected points; never detect or accept labels.

The visual reviewer selects the points first. This module standardizes the
measurement of their visible dark core and adds a small annotation margin.
It does not use existing pseudo labels, model predictions, or colored originals.
"""
from __future__ import annotations

import cv2
import numpy as np

GEOMETRY_VERSION = "dark_core_margin3_v1"


def measured_box(gray, cx, cy):
    """Return a proposed box and flags, in original image pixels.

    Local lower-quartile background reduces contamination by the bright area
    beside a bar. Long components touching the local window remain unresolved.
    Three pixels of context surround the measured dark core, with minimum 8px
    sides. This is an annotation convention, not an object detector or safety
    confidence score. Flagged boxes require a separate visual boundary decision.
    """
    h, w = gray.shape
    cx, cy = int(round(cx)), int(round(cy))
    x0, y0, x2, y2 = max(0, cx-8), max(0, cy-8), min(w, cx+9), min(h, cy+9)
    patch = cv2.GaussianBlur(gray.astype(np.float32), (3, 3), .6)[y0:y2, x0:x2]
    yy, xx = np.indices(patch.shape)
    radius = np.sqrt((xx-(cx-x0))**2+(yy-(cy-y0))**2)
    background = float(np.percentile(patch[(radius >= 4) & (radius <= 7)], 25))
    core = float(patch[cy-y0, cx-x0])
    flags = []
    if background-core < 4:
        flags.append("low_local_contrast")
    threshold = core + max(2., (background-core)*.6)
    mask = ((patch <= threshold) & (radius <= 7)).astype(np.uint8)
    _, lab, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    k = int(lab[cy-y0, cx-x0])
    if not k:
        return [cx-5, cy-5, cx+5, cy+5], ["core_not_isolated"]
    px, py, pw, ph, area = map(int, stats[k])
    if max(pw, ph) > 10 or max(pw, ph) > 3*max(1, min(pw, ph)) or area > 65:
        flags.append("core_merges_background")
    center_x, center_y = x0+px+pw/2, y0+py+ph/2
    if abs(center_x-cx) > 2.5 or abs(center_y-cy) > 2.5:
        flags.append("core_center_shift")
    width, height = max(8, pw+6), max(8, ph+6)
    box = [max(0., center_x-width/2), max(0., center_y-height/2),
           min(float(w), center_x+width/2), min(float(h), center_y+height/2)]
    return box, flags
