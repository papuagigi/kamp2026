"""Geometry shared by role-specific preparation and inference; no labels at runtime."""
import numpy as np


def around(cx, cy, width, height, side=96):
    w, h = min(side, width), min(side, height)
    x = min(max(0, int(round(cx - w / 2))), width - w)
    y = min(max(0, int(round(cy - h / 2))), height - h)
    return (x, y, x + w, y + h)


def tiles(width, height, side=256, overlap=64):
    def starts(length):
        if length <= side:
            return [0]
        return sorted(set(list(range(0, length - side + 1, side - overlap)) + [length - side]))
    return [(x, y, min(width, x + side), min(height, y + side))
            for y in starts(height) for x in starts(width)]


def clipped_labels(boxes, window):
    """Return labels and whether any GT is partially truncated (training only)."""
    x0, y0, x1, y1 = window
    result, partial = [], False
    for x, y, w, h in boxes:
        a, b = max(x0, x-w/2), max(y0, y-h/2)
        c, d = min(x1, x+w/2), min(y1, y+h/2)
        if c <= a or d <= b:
            continue
        visible = (c-a)*(d-b)/(w*h)
        partial |= visible < .999
        result.append([0, ((a+c)/2-x0)/(x1-x0), ((b+d)/2-y0)/(y1-y0),
                       (c-a)/(x1-x0), (d-b)/(y1-y0)])
    return result, partial


def remap(raw, window):
    raw = np.asarray(raw, dtype=float).reshape(-1, 5).copy()
    if len(raw):
        raw[:, [0, 2]] = np.clip(raw[:, [0, 2]], 0, window[2]-window[0]) + window[0]
        raw[:, [1, 3]] = np.clip(raw[:, [1, 3]], 0, window[3]-window[1]) + window[1]
        raw = raw[(raw[:, 2] > raw[:, 0]) & (raw[:, 3] > raw[:, 1])]
    return raw


def suppress(raw):
    import torch
    from torchvision.ops import nms
    raw = np.asarray(raw, dtype=float).reshape(-1, 5)
    if not len(raw):
        return raw
    keep = nms(torch.tensor(raw[:, :4], dtype=torch.float32),
               torch.tensor(raw[:, 4], dtype=torch.float32), .5).numpy()
    return raw[keep]
