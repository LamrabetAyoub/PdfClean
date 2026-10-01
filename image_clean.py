#!/usr/bin/env python3
"""
image_clean.py — scanned-document cleaning for the OCR API.

Per-page pipeline (numpy + scipy, already installed):

1. Render the page at CLEAN_DPI.
2. Detect the paper / background: per-channel low-frequency fields
   (``_background_maps``) — a grey closing swallows the ink, a large Gaussian
   keeps only the paper tint / illumination, so the estimate tracks the bright
   paper instead of being dragged down by text.
3. Detect folds / strong illumination changes (``_fold_exposure_map``): the
   reconstructed paper is compared with its own smooth sheet; rows that deviate
   by more than a margin are a fold / page-curl / platen edge — wide, unreliable
   surface.
4. Identify foreground ink (``_residual_dark_mask`` + chroma): pixels clearly
   darker than the local paper, or carrying strong colour. Faded, pale and grey
   content is only kept if it is clearly off the paper field.
5. Clean the background: flat-field division per channel (``_flat_field``)
   removes cast + illumination while preserving ink hue; then
   reconstruct-and-whiten (``_whiten_luma``) and a residual wash
   (``_residual_wash``) lift smooth tinted patches — fold bands, ink stains,
   shadows — to clean paper. Sharp-edged pixels are content and spared.
6. Selectively suppress ink inside the folded / exposed region
   (``_fold_ink``): inside those zones the surface is unreliable, so *smooth,
   neutral, shallow* dark pixels (crease shadow, dust, compressed fold noise)
   are lifted to paper while genuine text — deep or sharp-edged — survives.
7. Despeckle: isolated dark spots and long-thin crease/scratch lines
   (``_tiny_specks``, ``_stray_marks``) are stripped, then the page is emitted
   in gray / colour / binary and rebuilt into a brand-new PDF with img2pdf.

The original PDF is never touched: pages are rendered, cleaned and rebuilt.
"""

import collections
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import img2pdf
import numpy as np
from pdf2image import convert_from_path
from pypdf import PdfReader
import scipy.ndimage as ndi
from PIL import Image

import config

_LUMA_WEIGHTS = np.array([0.299, 0.587, 0.114], dtype=np.float32)
_STRUCT = np.ones((3, 3), dtype=np.uint8)
_WHITE = np.array([255, 255, 255], dtype=np.uint8)

# After reconstruct-and-whiten (see ``_whiten_luma``), paper is ~1.0 * the
# background model. Pixels at least this bright are rounding *near-background*
# only: the plateau turns a uniform paper surface into exactly 255. Faded ink,
# pale stamps and every colour-carrying pixel sit well below it (the ratio
# collapses shading, so a global 190-cutoff is no longer needed). All content
# below is spared via the chroma guard below when colour is present.
_PAPER_LUMA = 247.0
_PAPER_CHROMA = 16.0

# Mode vocabulary exposed to the API and the UI.
MODES = ("gray", "color", "binary", "document")

# Residual background wash thresholds (see config.py).
_WASH_BRIGHT_FLOOR = float(config.CLEAN_WASH_BRIGHT)
_WASH_CHROMA_FLOOR = float(config.CLEAN_WASH_CHROMA)
_WASH_GRAD_FLOOR = float(config.CLEAN_WASH_GRAD)
_WASH_GRAD_DILATE = int(config.CLEAN_WASH_GRAD_DILATE)
_WASH_NEUTRAL_CHROMA = float(config.CLEAN_WASH_NEUTRAL_CHROMA)

# Fold / strong-illumination zone detection (see config.py). The pipe is the
# *original* scan: a fold, page curl or scanner edge shows as the paper field
# deviating from its own smooth sheet before any cleaning touches it.
_FOLD_DEV = float(config.CLEAN_FOLD_DEV)
_FOLD_BG_SIGMA = float(config.CLEAN_FOLD_BG_SIGMA)
_FOLD_ERODE = int(config.CLEAN_FOLD_ERODE)
_FOLD_INK_FLOOR = float(config.CLEAN_FOLD_INK_FLOOR)

# Local-gradient resolution (see config.py). The gradient is the single most
# expensive filter here and it is only ever thresholded, never used as a value.
_GRAD_MAX_EDGE = int(config.CLEAN_GRAD_MAX_EDGE)

# Resolution of the fold sheet (see config.py).
_SHEET_MAX_EDGE = int(config.CLEAN_SHEET_MAX_EDGE)


class CleanError(Exception):
    """Cleaning failed with a machine-readable code and a French-safe message."""

    def __init__(self, code, message):
        super().__init__(message)
        self.code = code
        self.message = message


def _rgb(image):
    return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _luma(rgb):
    rgb = rgb.astype(np.float32)
    return (
        rgb[..., 0] * _LUMA_WEIGHTS[0]
        + rgb[..., 1] * _LUMA_WEIGHTS[1]
        + rgb[..., 2] * _LUMA_WEIGHTS[2]
    )


def _percentile(image, p):
    return float(np.percentile(image, p))


def _as_l8(image):
    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)
    return arr


def _grad_decision(image, sigma, threshold, above):
    """``gradient_magnitude > threshold`` (or ``<=``) as a full-size bool mask.

    Every consumer of the local gradient in this module compares it against a
    threshold and then keeps the boolean, so the magnitude itself never has to
    be materialised at full resolution. It is measured on a box-downscaled copy
    and the *decision* is upscaled with nearest sampling, which keeps the mask
    exactly as blocky-edged as the tests that produced it already were.

    Downscaling by ``factor`` shrinks a gradient of a step edge by ``factor``
    and widens its spatial scale by ``factor`` pixels, so sigma and the
    threshold are both divided by it: the comparison is scale-invariant.
    """
    h, w = image.shape[:2]
    factor = max(1, int(max(h, w) / _GRAD_MAX_EDGE)) if _GRAD_MAX_EDGE else 1

    if factor > 1:
        small = np.asarray(
            Image.fromarray(_as_l8(image), "L").resize(
                (max(1, w // factor), max(1, h // factor)), Image.BOX
            ),
            dtype=np.float32,
        )
        grad = ndi.gaussian_gradient_magnitude(small, sigma=sigma / factor)
        limit = threshold / factor
        mask = grad > limit if above else grad <= limit
        return np.asarray(
            Image.fromarray(mask.astype(np.uint8) * 255, "L").resize(
                (w, h), Image.NEAREST
            )
        ) > 127

    grad = ndi.gaussian_gradient_magnitude(
        np.asarray(image, dtype=np.float32), sigma=sigma
    )
    return grad > threshold if above else grad <= threshold


def _grad_above(image, sigma, floor):
    """True where the local gradient exceeds ``floor``: an edge, i.e. content."""
    return _grad_decision(image, sigma, floor, above=True)


def _grad_below(image, sigma, ceiling):
    """True where the local gradient stays under ``ceiling``: a smooth surface."""
    return _grad_decision(image, sigma, ceiling, above=False)


# --------------------------------------------------------------------------- #
# background estimation & flat-field correction
# --------------------------------------------------------------------------- #
def _background_maps(rgb):
    """Per-channel low-frequency background maps at full resolution.

    Computed once on a downscaled copy: a grey closing swallows the dark ink,
    then a Gaussian blur keeps only the paper tint / illumination gradient.
    """
    h, w = rgb.shape[:2]
    factor = max(1, int(max(h, w) / config.CLEAN_BG_MAX_EDGE))
    small = rgb[::factor, ::factor].astype(np.float32)
    kernel = max(3, min(41, (min(small.shape[:2]) // 48) | 1))

    maps = {}
    sh, sw = small.shape[:2]
    for channel in range(3):
        plane = small[..., channel]
        bg = ndi.grey_closing(plane, size=(kernel, kernel))
        bg = ndi.gaussian_filter(
            bg, sigma=config.CLEAN_BG_SIGMA, mode="nearest"
        )
        full = np.asarray(
            Image.fromarray(bg.astype(np.uint8), "L").resize(
                (w, h), Image.BILINEAR
            ),
            dtype=np.float32,
        )
        maps[channel] = full + 1.0  # +1 guards against division noise
    del small
    return maps


def _flat_field(rgb, maps):
    """Divide each channel by its background: white paper, no shadow, no cast."""
    corrected = np.empty_like(rgb, dtype=np.float32)
    for channel in range(3):
        corrected[..., channel] = np.clip(
            rgb[..., channel].astype(np.float32) / maps[channel] * 255.0,
            0.0, 255.0,
        )
    return corrected, _luma(corrected)


def _background_luma(luma):
    """Per-pixel reconstruction of the paper level (smooth, full resolution).

    The paper tint / illumination is a slow field: it is resampled down, a wide
    grey-closing swallows every drop of ink (closing erases dark objects
    narrower than the kernel), a heavy Gaussian keeps only that low-frequency
    envelope, and a bilinear upscale brings it back to full size. This is the
    background *model* the whitening step divides against — nothing here paints
    over "gray spots", it estimates what the paper alone looks like at every
    pixel.
    """
    h, w = luma.shape
    factor = max(1, int(max(h, w) / config.CLEAN_BG_MAX_EDGE))
    small = luma[::factor, ::factor].astype(np.float32)
    kernel = max(9, (min(small.shape[:2]) // 28) | 1)
    bg = ndi.grey_closing(small, size=(kernel, kernel))
    bg = ndi.gaussian_filter(bg, sigma=config.CLEAN_BG_SIGMA, mode="nearest")
    full = np.asarray(
        Image.fromarray(bg.astype(np.uint8), "L").resize(
            (w, h), Image.BILINEAR
        ),
        dtype=np.float32,
    )
    return full + 1.0  # +1 guards against division noise


def _whiten_luma(luma, bg=None, beta=0.9):
    """Reconstruct-and-whiten: every pixel is written as luma / local paper.

    ``ratio = luma / bg`` makes the paper itself ~1 everywhere — shadow, fold
    band, stained patch, vignette all collapse to the same whiteness because
    each pixel is compared with its own reconstructed background, not a global
    cutoff. ``255 * ratio**beta`` maps paper to ~255; slow shading makes it
    slide further and further above the ink so faint stains vanish *by
    reconstruction*. A final tight plateau only rounds near-paper pixels to
    exactly 255; faded ink lies well below it and survives.
    """
    if bg is None:
        bg = _background_luma(luma)
    ratio = np.clip(luma.astype(np.float32) / bg, 0.0, 1.0)
    l2 = np.clip(255.0 * np.power(ratio, beta), 0.0, 255.0)
    l2[l2 >= _PAPER_LUMA] = 255.0
    return l2


def _residual_wash(l2, chroma=None):
    """Second background pass: lift smooth tinted patches up to clean paper.

    This is a *field* correction, not a spot-removal loop. The first flat
    field removes the slow cast; what it leaves is the mid-frequency
    discoloration — a fold band, an ink-soaked patch, a page-curvature
    shadow. Those are wide and smooth, so their texture is essentially
    edge-free: the brightness changes gradually over tens of pixels.

    Real content is the opposite: text, lines, handwriting and logos all have
    *sharp* edges. So the wash classifies by local gradient, not by absolute
    brightness or by deviation from the reconstructed field (folds deviate
    from their field too):

    * smooth + bright pixels  → lifted by exactly the local deficit and (in
      colour mode) their tint is dropped. Stains, folds and shadows are
      smooth, so they are washed away.
    * sharp-edged pixels → content, left at the flat-field value. In colour
      mode the privilege is only granted to *neutral* content (chroma below
      the neutral floor): a grey logo or light handwriting must not be eaten
      just because it is bright. Coloured canvas stains and ink-bleeds still
      carry chroma, so they are washed even across their boundary, while
      genuinely coloured ink (stamps, signatures) sits far above the chroma
      floor and is always kept.
    """
    bg2 = _background_luma(l2)
    bright_ref = max(float(np.percentile(bg2, 99)) + 2.0, _PAPER_LUMA)
    lift = np.clip(bright_ref - bg2, 0.0, 255.0)

    edge = _grad_above(l2, 1.2, _WASH_GRAD_FLOOR)
    if _WASH_GRAD_DILATE > 0:
        edge = ndi.binary_dilation(edge, iterations=_WASH_GRAD_DILATE)

    background = l2 >= _WASH_BRIGHT_FLOOR
    if chroma is not None:
        background = background & (chroma < _WASH_CHROMA_FLOOR)
        # edge protection only for visually neutral content (grey logo,
        # light handwriting); coloured stains keep their tint, so their
        # chroma marks them as background to wash regardless of boundary
        background = background & ~(edge & (chroma < _WASH_NEUTRAL_CHROMA))
    else:
        background = background & ~edge

    out = np.asarray(l2, dtype=np.float32).copy()
    out[background] += lift[background]
    np.clip(out, 0.0, 255.0, out=out)
    return out, background


def _fold_exposure_map(luma, bg=None):
    """Wide zones where the reconstructed paper leaves its own smooth sheet.

    A fold / page curl / scanner platen edge is a *local* departure of the
    paper background from its own sheet: the background model ``bg`` and its
    Gaussian low-pass ``sheet`` agree on clean paper and disagree across the
    fold, whose shadow (or ridge) does not move at the sheet's scale. A global
    shading ramp moves at the sheet's scale and is NOT flagged.

    This runs on the original luma, before any cleaning, exactly as the
    pipeline wants: the fold is a property of the paper, so it is measured on
    the raw background. Thin crease *lines* are too narrow to survive the 1px
    opening and are left to ``_stray_marks``.

    ``bg`` is the caller's already-computed background for this same luma; the
    reconstruction is the most expensive step in the module, so it is reused
    rather than redone.
    """
    if bg is None:
        bg = _background_luma(luma)
    h, w = bg.shape
    factor = max(1, int(max(h, w) / _SHEET_MAX_EDGE)) if _SHEET_MAX_EDGE else 1

    if factor > 1:
        # The sheet is a sigma-~32 Gaussian: sampling it per pixel is pure
        # oversampling, so it is built on a downscaled copy. Downscaling
        # divides every deviation by the same factor, so the deviation is
        # scaled back before it is compared with the threshold.
        small = np.asarray(
            Image.fromarray(_as_l8(bg), "L").resize(
                (max(1, w // factor), max(1, h // factor)), Image.BOX
            ),
            dtype=np.float32,
        )
        sheet = ndi.gaussian_filter(
            small, sigma=_FOLD_BG_SIGMA / factor, mode="nearest"
        )
        dev = np.abs(small - sheet) * factor
        zone = np.asarray(
            Image.fromarray((dev > _FOLD_DEV).astype(np.uint8) * 255, "L").resize(
                (w, h), Image.NEAREST
            )
        ) > 127
    else:
        sheet = ndi.gaussian_filter(bg, sigma=_FOLD_BG_SIGMA, mode="nearest")
        zone = np.abs(bg - sheet) > _FOLD_DEV
    if _FOLD_ERODE > 0:
        zone = ndi.binary_opening(zone, iterations=_FOLD_ERODE)
    return zone


def _fold_ink(l2, fold, smooth, chroma=None):
    """Twilight-band ink inside a folded / exposed zone: shadow, not text.

    After clean-background the paper is ~255 and real ink is *deep* (well
    below ``_FOLD_INK_FLOOR``), so the luminance band between "definitely
    real ink" and "definitely background" (the wash floor) is the twilight
    band a fold leaves behind: compressed mid-tone pixels — crease shadow,
    dust in the fold, crushed paper gradient — that are only lightly darker
    than their surroundings.

    Those are suppressed selectively: they must sit *inside* the folded /
    exposed zone and be smooth (a glyph's edge exempts it, exactly like the
    wash). It is also required that they be neutral — a coloured marker or
    felt-pen burn inside a fold is real and is never touched.

    ``smooth`` is the precomputed low-gradient mask for this page (see
    ``_grad_below``), so the same test is shared by every caller.
    """
    twilight = (l2 >= _FOLD_INK_FLOOR) & (l2 < _WASH_BRIGHT_FLOOR)
    zone = fold & twilight & smooth
    if chroma is not None:
        zone = zone & (chroma < _WASH_CHROMA_FLOOR)
    return zone


# --------------------------------------------------------------------------- #
# conservative grayscale artifact guards
# --------------------------------------------------------------------------- #
def _corner_fold_mask(luma, bg=None):
    """Detect only smooth dark regions physically attached to page corners.

    A dog-ear is treated as a geometric surface artifact, not as "dark ink":
    it must touch a real image corner, occupy a limited fraction of a corner
    patch, and be substantially darker than the reconstructed paper while
    remaining low-gradient. This protects normal text elsewhere on the page.
    """
    if not config.CLEAN_CORNER_ENABLED:
        return np.zeros_like(luma, dtype=bool)
    h, w = luma.shape
    if min(h, w) < 64:
        return np.zeros_like(luma, dtype=bool)
    if bg is None:
        bg = _background_luma(luma)

    margin = max(24, int(round(min(h, w) * config.CLEAN_CORNER_MARGIN_RATIO)))
    margin = min(margin, max(24, min(h, w) // 3))
    smooth = _grad_below(luma, 1.2, config.CLEAN_CORNER_GRAD)
    shadow = ((bg - luma) >= config.CLEAN_CORNER_DEV) & smooth

    out = np.zeros_like(shadow)
    corners = (
        (slice(0, margin), slice(0, margin), "tl"),
        (slice(0, margin), slice(w - margin, w), "tr"),
        (slice(h - margin, h), slice(0, margin), "bl"),
        (slice(h - margin, h), slice(w - margin, w), "br"),
    )
    for ys, xs, name in corners:
        patch = shadow[ys, xs]
        if not patch.any():
            continue
        labels, count = ndi.label(patch, structure=np.ones((3, 3), dtype=np.uint8))
        if count == 0:
            continue
        sizes = ndi.sum(patch, labels, index=np.arange(1, count + 1))
        patch_area = float(patch.size)
        min_area = patch_area * config.CLEAN_CORNER_MIN_AREA_RATIO
        max_area = patch_area * config.CLEAN_CORNER_MAX_AREA_RATIO

        for label_id, area in enumerate(sizes, 1):
            area = float(area)
            if area < min_area or area > max_area:
                continue
            component = labels == label_id
            yy, xx = np.where(component)
            if len(yy) == 0:
                continue

            # The dog-ear must actually reach the relevant two corner edges.
            touches = {
                "tl": yy.min() == 0 and xx.min() == 0,
                "tr": yy.min() == 0 and xx.max() == margin - 1,
                "bl": yy.max() == margin - 1 and xx.min() == 0,
                "br": yy.max() == margin - 1 and xx.max() == margin - 1,
            }[name]
            if not touches:
                continue

            # Require a compact corner wedge rather than a full corner block.
            bh = yy.max() - yy.min() + 1
            bw = xx.max() - xx.min() + 1
            fill = area / float(max(1, bh * bw))
            if fill < 0.12 or fill > 0.92:
                continue

            target = np.zeros_like(out)
            target[ys, xs] = component
            out |= target

    # Never erase a high-contrast edge: text/logos close to a corner survive.
    out &= smooth
    return out


def _grayscale_band_mask(luma, bg=None):
    """Detect long, shallow grayscale streaks without treating text as noise.

    Horizontal/vertical responses are obtained from a morphological opening of
    the *dark residual* against the local paper model. Only shallow tones with
    low local gradient and page-scale continuity qualify. Deep black content is
    deliberately excluded, and the length is page-relative rather than DPI-
    dependent. This targets scanner-roller/toner banding, not text rules.
    """
    if not config.CLEAN_BAND_ENABLED:
        return np.zeros_like(luma, dtype=bool)
    h, w = luma.shape
    if min(h, w) < 128:
        return np.zeros_like(luma, dtype=bool)
    if bg is None:
        bg = _background_luma(luma)

    residual = np.clip(bg - luma, 0.0, 255.0)
    candidate = (
        (residual >= config.CLEAN_BAND_MIN_DARK)
        & (residual <= config.CLEAN_BAND_MAX_DARK)
        & (luma >= config.CLEAN_BAND_MAX_LUMA)
    )
    smooth = _grad_below(luma, 1.0, config.CLEAN_BAND_GRAD)
    candidate &= smooth

    hlen = max(31, int(round(w * config.CLEAN_BAND_MIN_LENGTH_RATIO)))
    vlen = max(31, int(round(h * config.CLEAN_BAND_MIN_LENGTH_RATIO)))
    hlen = min(hlen, max(31, w))
    vlen = min(vlen, max(31, h))

    # Opening the residual preserves structures that remain dark continuously
    # along the long axis while rejecting isolated glyphs and speckles.
    h_response = ndi.grey_opening(residual, size=(1, hlen))
    v_response = ndi.grey_opening(residual, size=(vlen, 1))
    response = np.maximum(h_response, v_response)

    band = (
        candidate
        & (response >= config.CLEAN_BAND_MIN_DARK)
        & (response <= config.CLEAN_BAND_MAX_DARK)
    )

    # A small closing joins JPEG/noise gaps but does not expand across normal
    # text rows. Keep the correction itself conservative.
    band = ndi.binary_closing(
        band, structure=np.ones((3, 3), dtype=bool), iterations=1
    )
    return band



# --------------------------------------------------------------------------- #
# denoising
# --------------------------------------------------------------------------- #
def _disk(radius):
    """Boolean symmetric disk structuring element."""
    y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    return x * x + y * y <= radius * radius


def _tiny_specks(dark):
    """Isolated 1px dust: residue of a 1px opening that hugs no real ink.

    A plain opening residue is NOT enough: on anti-aliased scans the thin
    fringe around every glyph is also 1px-wide, and deleting it shreds the
    text (OCR drops to zero while the page still "looks" fine). So the
    residue is kept unless it floats at least a couple of pixels away from
    the surviving core ink — glyph fringes hug their core and survive, dust
    in open paper goes.
    """
    if not dark.any():
        return dark
    opened = ndi.binary_opening(dark, structure=_disk(1))
    residue = dark & ~opened
    if not residue.any():
        return residue
    # Halo over ALL dark ink (not the opened core): a thin anti-aliased shard
    # is itself destroyed by the opening, so the safe test is "does this
    # residue pixel sit near ANY ink at all" — glyph fringes always do, dust
    # floating in open paper does not.
    halo = ndi.binary_dilation(dark, structure=_disk(2), iterations=1)
    return residue & ~halo


def _stray_marks(luma, dark=None):
    """Long-thin straight marks (crease lines, scratches, pen strokes).

    Scans of folded paper show a thin dark *line a few pixels wide but long*:
    the crease centre, a scratched gutter edge, a long pen stroke. Text never
    matches that shape — glyphs are blobs, a text row is too thick (its short
    side is many tens of pixels), vertical letter stems are too short.

    Detection runs on the Sauvola ink mask (keeps antialiased glyphs
    connected). Limit values mean only *very* long straight marks are touched
    (page-length creases/scratch lines), so nothing a cursor finger can reach
    — form field rules, signature curls, stamp rings — ever gets removed.
    """
    if dark is not None:
        mask = dark
    else:
        mask = _sauvola(luma)
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)
    lab, n = ndi.label(mask, structure=np.ones((3, 3)))
    if n == 0:
        return np.zeros_like(mask, dtype=bool)
    slices = ndi.find_objects(lab)
    sizes = ndi.sum(np.ones_like(lab), lab, range(1, n + 1))
    remove = np.zeros_like(lab, dtype=bool)

    for i in range(1, n + 1):
        area = int(sizes[i - 1])
        o = slices[i - 1]
        h = o[0].stop - o[0].start
        w = o[1].stop - o[1].start
        long_side, short_side = max(h, w), min(h, w)
        if (area >= config.CLEAN_MARK_MIN_AREA
                and long_side >= config.CLEAN_MARK_MIN_LEN
                and short_side <= config.CLEAN_MARK_MAX_WIDTH
                and long_side / float(max(1, short_side)) >= config.CLEAN_MARK_RATIO):
            remove |= lab == i
    return remove


# --------------------------------------------------------------------------- #
# output builders
# --------------------------------------------------------------------------- #
def _residual_dark_mask(luma, bg=None):
    """Pixels that are clearly darker than the local paper model.

    This is intentionally conservative: real gray, blue or dark content is kept
    unless it is small isolated noise or a very long crease/scratch. The goal is
    to preserve content by default; only obvious speck / scratch artifacts are
    stripped, never an entire glyph or mark simply because it is dark.
    """
    if bg is None:
        bg = _background_luma(luma)
    # Only pixels meaningfully below the reconstructed paper are considered for
    # denoising. A flat 62% cutoff on the whitened result is too aggressive and
    # turns legitimate gray text into paper.
    return (luma < np.clip(bg * 0.94, 0.0, 255.0)) & (bg > 8.0)


def _cam_scanner_reconstruct_gray(luma, beta=0.84):
    """Reconstruct a clean white document page from a scanned image.

    This follows the same general class of processing as scanner apps:
    estimate the sheet/background, correct illumination, separate foreground
    from paper using local contrast, rebuild the paper as white, then restore
    high-confidence text/graphics. It is not a copy of any proprietary
    CamScanner implementation.
    """
    bg = _background_luma(luma)
    ratio = np.clip(luma.astype(np.float32) / np.maximum(bg, 1.0), 0.0, 1.0)
    normalized = np.clip(255.0 * np.power(ratio, beta), 0.0, 255.0)

    local_bg = ndi.gaussian_filter(
        normalized, sigma=config.CLEAN_DOCUMENT_LOCAL_SIGMA, mode="nearest"
    )
    local_contrast = local_bg - normalized
    sharp = _grad_above(normalized, 1.0, config.CLEAN_DOCUMENT_EDGE_GRAD)

    foreground = (
        (normalized < config.CLEAN_DOCUMENT_FOREGROUND_FLOOR)
        | (local_contrast > config.CLEAN_DOCUMENT_LOCAL_CONTRAST)
    )
    foreground &= (
        (normalized < config.CLEAN_DOCUMENT_DARK_FLOOR) | sharp
    )

    out = np.full_like(normalized, 255.0)
    out[foreground] = normalized[foreground]

    corner = _corner_fold_mask(luma, bg)
    bands = _grayscale_band_mask(luma, bg)
    fold = _fold_exposure_map(luma, bg)
    fold_ink = _fold_ink(
        out, fold, _grad_below(normalized, 1.0, _WASH_GRAD_FLOOR)
    )
    residual = _residual_dark_mask(luma, bg)
    dirty = _tiny_specks(residual) | _stray_marks(luma, residual)
    out[corner | bands | fold_ink | dirty] = 255.0

    out[out >= config.CLEAN_DOCUMENT_WHITE_FLOOR] = 255.0
    return np.clip(out, 0.0, 255.0).astype(np.uint8)


def _clean_gray(luma, beta=0.9):
    bg = _background_luma(luma)
    fold = _fold_exposure_map(luma, bg)
    out, _ = _residual_wash(_whiten_luma(luma, bg=bg, beta=beta))

    # CamScanner-style surface cleanup: correct only high-confidence grayscale
    # surface artifacts after the main illumination normalization. Text is
    # protected by the shallow-tone + low-gradient + long-continuity tests.
    corner = _corner_fold_mask(luma, bg)
    bands = _grayscale_band_mask(luma, bg)

    dirty = _fold_ink(
        out, fold, _grad_below(out, 1.2, _WASH_GRAD_FLOOR)
    )
    residual = _residual_dark_mask(luma, bg)
    dirty |= _tiny_specks(residual) | _stray_marks(luma, residual)
    out[corner | bands] = 255.0
    out[dirty] = 255.0
    return out.astype(np.uint8)


def _clean_color(rgb, luma, beta=0.9):
    """Saturation-preserving colour output.

    Lumas are reconstructed-and-whitened against the per-pixel paper model
    (``_whiten_luma``); each channel's colour offset relative to luma
    (channel - luma) is added back untouched. That way the *paper* is
    whitened by reconstruction while seals, stamps and signature ink keep
    their colour and their relative weight.
    """
    bg = _background_luma(luma)
    l2 = _whiten_luma(luma, bg=bg, beta=beta)
    
    scale = l2 / np.maximum(luma, 1.0)
    max_scale = 255.0 / np.maximum(rgb.max(axis=2).astype(np.float32), 1.0)
    scale = np.minimum(scale, max_scale)
    
    out = rgb.astype(np.float32) * scale[..., None]
    out = np.clip(out, 0.0, 255.0)

    chroma = np.abs(out).max(axis=2) - np.abs(out).min(axis=2)
    # Residual wash: lift bright low-chroma tint (folds, ink patches, page
    # shadow) up to clean paper, and knock their tint out entirely. Coloured
    # ink (stamps, signatures) has enough chroma to stay off this mask.
    l3, washed = _residual_wash(l2, chroma)
    out[washed] = l3[washed, None]
    
    scale3 = l3 / np.maximum(luma, 1.0)
    scale3 = np.minimum(scale3, max_scale)
    out[~washed] = np.clip(
        rgb[~washed].astype(np.float32) * scale3[~washed, None], 0.0, 255.0)

    # Selectively suppress shallow neutral ink inside the folded / exposed
    # region (crease shadow, dust in the fold, compressed fold noise). Colour
    # ink is never touched here.
    fold = _fold_exposure_map(luma, bg)
    fold_ink = _fold_ink(
        l3, fold, _grad_below(l3, 1.2, _WASH_GRAD_FLOOR), chroma
    )
    out[fold_ink] = [255.0, 255.0, 255.0]

    # Apply the same conservative grayscale artifact masks to colour pages.
    # The masks are derived from luma and only target neutral surface defects;
    # genuine coloured stamps/signatures remain protected by the existing
    # chroma logic.
    corner = _corner_fold_mask(luma, bg)
    bands = _grayscale_band_mask(luma, bg)
    out[corner | bands] = [255.0, 255.0, 255.0]

    paper = (l3 >= _PAPER_LUMA) & (luma >= (bg * 0.94)) & (chroma < _PAPER_CHROMA)
    out[paper] = [255.0, 255.0, 255.0]
    residual = _residual_dark_mask(luma, bg)
    dirty = _tiny_specks(residual) | _stray_marks(luma, residual)
    out[dirty] = 255.0
    return out.astype(np.uint8)


def _sauvola(luma, k=0.25, radius=128, window=None):
    """Sauvola adaptive-threshold mask: dark ink where luma < local bound."""
    h, w = luma.shape
    if window is None:
        window = max(15, (min(h, w) // 16) | 1)
    mean = ndi.uniform_filter(luma.astype(np.float32), size=window)
    sq = ndi.uniform_filter(luma.astype(np.float32) ** 2, size=window)
    var = np.clip(sq - mean * mean, 0.0, None)
    std = np.sqrt(var)
    threshold = mean * (1.0 + k * (std / radius - 1.0))
    return luma < threshold


def _fill_holes(binary):
    """Remove tiny holes inside ink so thin displaced strokes stay connected."""
    inv = ~binary
    label, count = ndi.label(inv, _STRUCT)
    if count == 0:
        return binary
    border = np.unique(np.concatenate([
        label[0, :], label[-1, :], label[:, 0], label[:, -1]
    ]))
    border = border[border > 0]
    sizes = ndi.sum(inv, label, index=np.arange(1, count + 1))
    inside = (sizes < max(8, int(0.00006 * binary.size))) & ~np.isin(
        np.arange(1, count + 1), border
    )
    if inside.any():
        binary = binary | np.isin(label, np.flatnonzero(inside) + 1)
    return binary


def _clean_binary(luma):
    ink = _sauvola(luma)
    ink = _fill_holes(ink)
    ink = ink & ~(_tiny_specks(ink) | _stray_marks(luma, ink))
    out = np.where(ink, 0, 255).astype(np.uint8)
    return out, ink


# --------------------------------------------------------------------------- #
# quality metrics & output decision
# --------------------------------------------------------------------------- #
def _page_metrics(luma, rgb, maps):
    """Measure corrected background uniformity, noise and colourfulness."""
    paper = luma >= (0.85 * 255.0)
    paper_std = float(luma[paper].std()) / 255.0 if paper.sum() > 0 else 1.0

    bg_stack = np.stack([maps[0], maps[1], maps[2]])
    bg_spread = float(bg_stack.max() - bg_stack.min()) / 255.0

    ink = luma < (0.9 * 255.0)
    if paper.sum() > 0:
        paper_mid = _percentile(luma[paper], 50) / 255.0
    else:
        paper_mid = 1.0
    ink_mid = (
        _percentile(luma[ink], 30) / 255.0 if ink.sum() > 1000 else 0.9
    )
    contrast = round(paper_mid - ink_mid, 4)

    diff = rgb.astype(np.int16)
    chroma = diff.max(axis=2).astype(np.int16) - diff.min(axis=2).astype(np.int16)
    chroma_mean = 0.0
    if ink.sum() > 0:
        chroma_mean = float(chroma[ink].mean()) / 255.0

    return {
        "paperStd": round(paper_std, 4),
        "backgroundSpread": round(bg_spread, 4),
        "contrast": round(contrast, 4),
        "chroma": round(chroma_mean, 4),
    }


# --------------------------------------------------------------------------- #
# public: one page
# --------------------------------------------------------------------------- #
def clean_page_image(image, user_mode=None):
    """Clean one PIL page. Returns (PIL image, info dict)."""
    if not config.CLEAN_ENABLED:
        raise CleanError("cleaning_disabled", "Cleaning is disabled by configuration.")

    original_edge = max(image.size)
    downscaled = False
    if original_edge > config.CLEAN_IMAGE_MAX_EDGE:
        scale = config.CLEAN_IMAGE_MAX_EDGE / float(max(image.size))
        image = image.resize(
            (max(1, round(image.width * scale)),
             max(1, round(image.height * scale))),
            Image.LANCZOS,
        )
        downscaled = True

    rgb = _rgb(image)
    maps = _background_maps(rgb)
    corrected, luma = _flat_field(rgb, maps)

    metrics = _page_metrics(luma, corrected, maps)
    user_mode = (user_mode or config.CLEAN_OUTPUT or "auto").lower()
    if user_mode == "auto":
        scattered = metrics["paperStd"] > 0.06 or metrics["backgroundSpread"] > 0.35
        if scattered and metrics["chroma"] <= 0.05:
            mode = "binary"  # no colour at all AND the paper is badly distorted
        elif metrics["chroma"] > 0.05:
            mode = "color"
        else:
            mode = "gray"
    elif user_mode in MODES:
        mode = user_mode
    else:
        mode = "gray"

    if mode == "document":
        array = _cam_scanner_reconstruct_gray(luma)
        image_out = Image.fromarray(array, "L").convert("RGB")
    elif mode == "binary":
        out, _ = _clean_binary(luma)
        image_out = Image.fromarray(out, "L").convert("RGB")
    elif mode == "color":
        array = _clean_color(corrected, luma)
        
        # Preserve original colorful regions (charts, pale maps)
        diff = rgb.astype(np.float32)
        chroma_orig = diff.max(axis=2) - diff.min(axis=2)
        blend_chroma = np.clip((chroma_orig - 22.0) / 10.0, 0.0, 1.0)
        
        # Preserve original wide, dark regions (satellite photos, deep graphics)
        bg_luma = maps[0] * 0.3 + maps[1] * 0.6 + maps[2] * 0.1
        blend_bg = np.clip((180.0 - bg_luma) / 30.0, 0.0, 1.0)
        
        blend = np.maximum(blend_chroma, blend_bg)
        array = (array * (1.0 - blend[..., None]) + rgb * blend[..., None]).astype(np.uint8)
        image_out = Image.fromarray(array, "RGB")
    else:
        array = _clean_gray(luma)
        image_out = Image.fromarray(array, "L").convert("RGB")

    del rgb, maps, corrected, luma

    metrics["mode"] = mode
    metrics["scaled"] = downscaled
    metrics["originalEdge"] = int(original_edge)
    return image_out, metrics


# --------------------------------------------------------------------------- #
# public: whole document
# --------------------------------------------------------------------------- #
def clean_document(pdf_path, work_dir, index=0, progress=None,
                   preview_pages=None, user_mode=None):
    """Render, clean and rebuild a PDF. Returns metadata + the new PDF path.

    ``preview_pages`` limits how many pages get small original/cleaned JPEG
    thumbnails written into the work directory for the preview endpoint.
    """
    if not config.CLEAN_ENABLED:
        raise CleanError("cleaning_disabled", "Cleaning is disabled by configuration.")

    try:
        total = len(PdfReader(pdf_path).pages)
    except Exception as exc:  # noqa: BLE001
        raise CleanError("clean_pdf_unreadable", "The PDF could not be read.") from exc
    if total == 0:
        raise CleanError("clean_pdf_empty", "The PDF has no pages.")

    preview_dir = os.path.join(work_dir, "previews", f"{index:03d}")
    os.makedirs(preview_dir, exist_ok=True)

    preview_max = max(0, int(preview_pages or config.CLEAN_PREVIEW_PAGES_MAX))
    preview_edge = int(config.CLEAN_PREVIEW_EDGE)
    page_infos = []
    saved_pages = []
    decision_counts = collections.Counter()

    # Original physical page size per page (points): the cleaned rebuild must
    # keep it, otherwise a page that was downscaled during processing shrinks
    # to half its real size (an A4 turns into an A5) and OCR collapses.
    try:
        original_page_sizes = [
            (float(p.mediabox.width), float(p.mediabox.height))
            for p in PdfReader(pdf_path).pages
        ]
    except Exception:  # noqa: BLE001
        original_page_sizes = []

    dpi = max(96, int(config.CLEAN_DPI))

    def _clean_one_page(number):
        """Render, clean and store one page. Returns (info, page_path).

        Pages never look at each other, so this is safe to run on a pool: the
        cleaning maths, the JPEG quality and the preview thumbnails are all
        per-page, and the only shared state is the read-only config.
        """
        try:
            rendered = convert_from_path(
                pdf_path, dpi=dpi, first_page=number, last_page=number,
                fmt="jpeg", thread_count=1,
            )
            original = rendered[0]
        except Exception as exc:  # noqa: BLE001
            raise CleanError(
                "clean_render_failed",
                f"Could not render page {number} of the PDF.",
            ) from exc

        cleaned, info = clean_page_image(original, user_mode=user_mode)
        info["page"] = number

        # If clean_page_image downscaled for processing, restore the full
        # rendered resolution so the output PDF keeps the original pixel
        # density (text stays crisp, OCR has full detail).
        if info.get("scaled"):
            cleaned = cleaned.resize(original.size, Image.LANCZOS)

        page_path = os.path.join(work_dir, f"clean_{index:03d}_p{number:04d}.jpg")
        cleaned.save(page_path, "JPEG", quality=int(config.CLEAN_JPEG_QUALITY),
                     optimize=True)

        if preview_max and number <= preview_max:
            for side, source in (("original", original), ("cleaned", cleaned)):
                if max(source.size) > preview_edge:
                    scale = preview_edge / float(max(source.size))
                    thumb = source.resize(
                        (max(1, round(source.width * scale)),
                         max(1, round(source.height * scale))),
                        Image.BILINEAR,
                    )
                else:
                    thumb = source
                thumb.convert("RGB").save(
                    os.path.join(preview_dir, f"p{number:04d}_{side}.jpg"),
                    "JPEG", quality=82, optimize=True,
                )
        original.close()
        cleaned.close()
        return info, page_path

    # Pages are cleaned concurrently, then reassembled in page order: the
    # rebuilt PDF must list them in the original order for _layout_fun to pick
    # the right page box for each one.
    workers = max(1, int(config.CLEAN_PAGE_CONCURRENCY))
    numbers = list(range(1, total + 1))
    collected = {}
    if workers > 1 and total > 1:
        with ThreadPoolExecutor(max_workers=min(workers, total),
                                thread_name_prefix="cleanpage") as pool:
            futures = {
                pool.submit(_clean_one_page, number): number for number in numbers
            }
            for future in as_completed(futures):
                number = futures[future]
                collected[number] = future.result()
                if progress:
                    progress(number, total, f"cleaning page {number}")
    else:
        for number in numbers:
            collected[number] = _clean_one_page(number)
            if progress:
                progress(number, total, f"cleaning page {number}")

    for number in numbers:
        info, page_path = collected[number]
        page_infos.append(info)
        saved_pages.append(page_path)
        decision_counts[info["mode"]] += 1

    cleaned_pdf = os.path.join(work_dir, f"cleaned_{index:03d}.pdf")

    def _layout_fun(img_width, img_height, _ndpi):  # noqa: ANN001, ANN202
        """Page size that preserves the original document's geometry.

        img2pdf expects ``(page_w, page_h, image_w_pdf, image_h_pdf)``.
        """
        if _layout_fun.index < len(original_page_sizes):
            pw, ph = original_page_sizes[_layout_fun.index]
            _layout_fun.index += 1
            # Fit the (aspect-preserved) cleaned image inside the original
            # page box: identical aspect means exact size, rotated pages gain
            # the minimal margin instead of breaking.
            if pw > 0 and ph > 0 and img_width > 0 and img_height > 0:
                scale = min(pw / float(img_width), ph / float(img_height))
                iw = img_width * scale
                ih = img_height * scale
                return (iw, ih, iw, ih)
        return img2pdf.get_fixed_dpi_layout_fun((dpi, dpi))(
            img_width, img_height, _ndpi)

    _layout_fun.index = 0

    try:
        with open(cleaned_pdf, "wb") as fh:
            fh.write(img2pdf.convert(
                saved_pages,
                # Size each PDF page from its pixel counts at the cleaning
                # DPI (aspect-ratio preserved). get_layout_fun(pagesize) in
                # img2pdf 0.6.1 fixes the *page* to a square "300x300 pts",
                # which squashes non-square scans and wrecks OCR.
                layout_fun=_layout_fun,
            ))
    except Exception as exc:  # noqa: BLE001
        raise CleanError("clean_export_failed",
                         "The cleaned PDF could not be built.") from exc
    finally:
        for page_path in saved_pages:
            try:
                os.remove(page_path)
            except OSError:
                pass

    avg = lambda key: round(       # noqa: E731
        sum(p.get(key, 0.0) for p in page_infos) / max(1, len(page_infos)), 4
    )
    mode = decision_counts.most_common(1)[0][0] if decision_counts else "gray"
    return {
        "pdf_path": cleaned_pdf,
        "mode": mode,
        "pages": page_infos,
        "preview_pages": [p["page"] for p in page_infos][:preview_max],
        "average": {
            "paperStd": avg("paperStd"),
            "backgroundSpread": avg("backgroundSpread"),
            "contrast": avg("contrast"),
            "chroma": avg("chroma"),
        },
        "dpi": dpi,
        "scaled": any(p.get("scaled", False) for p in page_infos),
    }