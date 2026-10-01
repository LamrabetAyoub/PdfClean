#!/usr/bin/env python3
"""
config.py — OCR-only settings, read from the environment once.

The OCR half of this project deliberately has no options: language, DPI and
rotation are detected per document. These variables only cover the HTTP API,
job lifecycle and the optional local Arabic OCR text alternatives. The vision
and form layers of the original OCR-Formulaire project are not part of this
reusable OCR API.
"""

import os
import re

from dotenv import load_dotenv

load_dotenv(override=False)


def _int(name, default):
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _float(name, default):
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _str(name, default):
    value = os.environ.get(name, default)
    return value.strip() if isinstance(value, str) else default


APP_VERSION = _str("APP_VERSION", "2.28.1")


# --------------------------------------------------------------------------- #
# Arabic OCR alternatives (text-only Tesseract passes, no AI involved)
# --------------------------------------------------------------------------- #
OCR_ARABIC_VARIANTS_ENABLED = _str("OCR_ARABIC_VARIANTS_ENABLED", "1") == "1"
OCR_ARABIC_VARIANT_PSMS = tuple(
    int(part) for part in re.split(r"[,;\s]+", _str("OCR_ARABIC_VARIANT_PSMS", "6,11"))
    if part.isdigit() and 0 <= int(part) <= 13
)
OCR_ARABIC_VARIANT_DPI = _int("OCR_ARABIC_VARIANT_DPI", 220)


# --------------------------------------------------------------------------- #
# scanned-document cleaning (image_clean.py, numpy + scipy)
# --------------------------------------------------------------------------- #
CLEAN_ENABLED = _str("CLEAN_ENABLED", "1") == "1"
# Render resolution used when cleaning. 300 DPI keeps thin Arabic strokes and
# table rules safe while remaining fast enough for multi-page jobs.
CLEAN_DPI = _int("CLEAN_DPI", 300)
# Safety cap: a huge page is scaled down before the numerical passes so memory
# stays bounded. 2600 px is still ~215 DPI on A4, plenty for cleaning.
CLEAN_IMAGE_MAX_EDGE = _int("CLEAN_IMAGE_MAX_EDGE", 2600)
CLEAN_JPEG_QUALITY = _int("CLEAN_JPEG_QUALITY", 92)

# Stray-mark removal (scan creases, scratches, pen strokes) — tuned for
# 300-dpi cleaning. Only *very long, thin, straight* components are removed
# (page-length crease lines, scratched gutters, long pen strokes). Text rows
# are too thick, glyphs are too short, signatures curl and stamp rings are
# round, so nothing legitimate ever matches these limits. Kept deliberately
# narrow: miscounting here eats the page.
CLEAN_MARK_MIN_AREA = _int("CLEAN_MARK_MIN_AREA", 600)
CLEAN_MARK_MIN_LEN = _int("CLEAN_MARK_MIN_LEN", 320)
CLEAN_MARK_MAX_WIDTH = _int("CLEAN_MARK_MAX_WIDTH", 14)
CLEAN_MARK_RATIO = _float("CLEAN_MARK_RATIO", 12.0)
# Background (paper tint / shadow) estimation runs on a downscaled map: the
# low-frequency field does not need full resolution, and this keeps the
# per-page cost flat regardless of page size.
CLEAN_BG_MAX_EDGE = _int("CLEAN_BG_MAX_EDGE", 900)
# Local gradient magnitude decides every "is this an edge or a smooth surface?"
# test: it is what protects stamp and signature colour from the residual wash.
# It therefore stays at native resolution by default. Measuring it on a
# box-downscaled copy is ~3x faster overall, but area-averaging attenuates the
# high-frequency energy at ink edges, which pushes borderline pixels under the
# threshold and washes tint out of stamps — the flat masks also read as smooth.
# Left tunable (0 = native); only lower it if the speed is worth the artefacts.
CLEAN_GRAD_MAX_EDGE = _int("CLEAN_GRAD_MAX_EDGE", 0)
# The fold detector compares the reconstructed paper with its own smooth
# "sheet" (a wide Gaussian, sigma ~32). Sampling a Gaussian that wide at full
# resolution is pure oversampling: the kernel is ~161 taps, so the pass costs
# two billion multiply-adds per page. Measured on the downscaled sheet and
# upscaled, the deviation field is unchanged to well under a level.
CLEAN_SHEET_MAX_EDGE = _int("CLEAN_SHEET_MAX_EDGE", 800)
CLEAN_BG_SIGMA = _float("CLEAN_BG_SIGMA", 6.0)
# auto = pick gray / color / binary per page from measured background quality.
# gray | color | binary force one output style for every page.
CLEAN_OUTPUT = _str("CLEAN_OUTPUT", "auto").lower()
# CamScanner-style document reconstruction: rebuild the paper as white while
# preserving high-confidence foreground through local contrast.
CLEAN_DOCUMENT_LOCAL_SIGMA = _float("CLEAN_DOCUMENT_LOCAL_SIGMA", 7.0)
CLEAN_DOCUMENT_LOCAL_CONTRAST = _float("CLEAN_DOCUMENT_LOCAL_CONTRAST", 8.0)
CLEAN_DOCUMENT_FOREGROUND_FLOOR = _float("CLEAN_DOCUMENT_FOREGROUND_FLOOR", 225.0)
CLEAN_DOCUMENT_DARK_FLOOR = _float("CLEAN_DOCUMENT_DARK_FLOOR", 185.0)
CLEAN_DOCUMENT_EDGE_GRAD = _float("CLEAN_DOCUMENT_EDGE_GRAD", 12.0)
CLEAN_DOCUMENT_WHITE_FLOOR = _float("CLEAN_DOCUMENT_WHITE_FLOOR", 248.0)
CLEAN_PREVIEW_PAGES_MAX = _int("CLEAN_PREVIEW_PAGES_MAX", 10)
CLEAN_PREVIEW_EDGE = _int("CLEAN_PREVIEW_EDGE", 900)
# Pages inside one document are independent, so they are cleaned in parallel —
# the same idea the OCR stage already uses. This is a pure wall-clock win: the
# per-page result is bit-identical, only the order of completion changes.
# Peak memory scales with it (a 300-DPI A4 colour page costs ~0.5 GB of numpy
# intermediates while in flight), so keep it modest. 1 = sequential.
CLEAN_PAGE_CONCURRENCY = _int("CLEAN_PAGE_CONCURRENCY", 4)
# Residual background wash (second pass in image_clean.py): after the flat
# field removes the slow gray cast, fold bands, ink-soaked patches and page
# shadows survive as *bright mid-chroma* regions. WASH_BRIGHT is the whitened
# luminance floor above which a pixel is treated as background (real ink is
# darker); WASH_CHROMA is the colour span below which a bright pixel is
# considered a tint rather than coloured ink. Measured on real scans, genuine
# stamp / pen ink has chroma >= 60 at mid-to-dark luminance, while bleed
# stains stay at chroma 20-50 on bright paper — the two thresholds separate
# them cleanly.
#
# A wash driven only by those two thresholds would also swallow bright graphism
# (a grey logo, light handwriting) the moment it passes the brightness floor.
# Real content is distinguished by having *edges*: text, lines, logos and
# handwriting are sharp, while stains, folds and shadows are smooth. WASH_GRAD
# is the local gradient above which a pixel is treated as content and spared;
# WASH_GRAD_DILATE spreads that protection over anti-aliased fringes. Edge
# protection is only granted to *neutral* pixels (chroma < WASH_NEUTRAL_CHROMA):
# coloured canvas stains and bleed are washed even where they have a boundary,
# while genuinely coloured ink (chroma >= WASH_CHROMA) is always kept.
CLEAN_WASH_BRIGHT = _int("CLEAN_WASH_BRIGHT", 210)
CLEAN_WASH_CHROMA = _int("CLEAN_WASH_CHROMA", 60)
CLEAN_WASH_GRAD = _int("CLEAN_WASH_GRAD", 14)
CLEAN_WASH_GRAD_DILATE = _int("CLEAN_WASH_GRAD_DILATE", 2)
CLEAN_WASH_NEUTRAL_CHROMA = _int("CLEAN_WASH_NEUTRAL_CHROMA", 14)
# Fold / strong-illumination detection (image_clean._fold_exposure_map). A fold,
# page curl or scanner platen edge shows as a *local* deviation of the paper
# background from its own smooth sheet: FOLD_DEV is that deviation above which
# the sheet counts as folded/exposed; FOLD_BG_SIGMA is the sheet scale;
# FOLD_ERODE drops 1px zones (thin crease *lines* are handled by stray marks).
# In the zone the surface is unreliable, so _fold_ink suppresses the twilight
# band — pixels between FOLD_INK_FLOOR (clearly real ink) and the wash floor
# (clearly background) that are smooth and neutral: crease shadow, dust in the
# fold, crushed paper gradient.
CLEAN_FOLD_DEV = _int("CLEAN_FOLD_DEV", 10)
CLEAN_FOLD_BG_SIGMA = _float("CLEAN_FOLD_BG_SIGMA", 32.0)
CLEAN_FOLD_ERODE = _int("CLEAN_FOLD_ERODE", 2)
CLEAN_FOLD_INK_FLOOR = _int("CLEAN_FOLD_INK_FLOOR", 150)

# CamScanner-style white-document cleanup. These conservative grayscale
# artifact guards run after illumination normalization. They are intentionally
# expressed as page-relative ratios so 150/220/300/600 DPI scans behave alike.
# Corner folds are only accepted when a smooth dark region touches a physical
# page corner; streaks require a long, low-gradient, shallow grayscale response.
CLEAN_CORNER_ENABLED = _str("CLEAN_CORNER_ENABLED", "1") == "1"
CLEAN_CORNER_MARGIN_RATIO = _float("CLEAN_CORNER_MARGIN_RATIO", 0.075)
CLEAN_CORNER_MIN_AREA_RATIO = _float("CLEAN_CORNER_MIN_AREA_RATIO", 0.002)
CLEAN_CORNER_MAX_AREA_RATIO = _float("CLEAN_CORNER_MAX_AREA_RATIO", 0.22)
CLEAN_CORNER_DEV = _float("CLEAN_CORNER_DEV", 18.0)
CLEAN_CORNER_GRAD = _float("CLEAN_CORNER_GRAD", 9.0)
CLEAN_BAND_ENABLED = _str("CLEAN_BAND_ENABLED", "1") == "1"
CLEAN_BAND_MIN_LENGTH_RATIO = _float("CLEAN_BAND_MIN_LENGTH_RATIO", 0.28)
CLEAN_BAND_MIN_DARK = _float("CLEAN_BAND_MIN_DARK", 7.0)
CLEAN_BAND_MAX_DARK = _float("CLEAN_BAND_MAX_DARK", 42.0)
CLEAN_BAND_MAX_LUMA = _float("CLEAN_BAND_MAX_LUMA", 205.0)
CLEAN_BAND_GRAD = _float("CLEAN_BAND_GRAD", 8.0)


# --------------------------------------------------------------------------- #
# context helpers used by text_context.py
# --------------------------------------------------------------------------- #
MAX_LLM_INPUT_CHARACTERS = _int("MAX_LLM_INPUT_CHARACTERS", 18000)
MAX_LLM_INPUT_TOKENS = _int("MAX_LLM_INPUT_TOKENS", 3500)
CONTEXT_LINES_BEFORE = _int("CONTEXT_LINES_BEFORE", 4)
CONTEXT_LINES_AFTER = _int("CONTEXT_LINES_AFTER", 6)


# --------------------------------------------------------------------------- #
# concurrency
# --------------------------------------------------------------------------- #
DOCUMENT_WORKER_CONCURRENCY = _int("DOCUMENT_WORKER_CONCURRENCY", 1)
OCR_PAGE_CONCURRENCY = _int("OCR_PAGE_CONCURRENCY", 4)


# --------------------------------------------------------------------------- #
# job lifecycle
# --------------------------------------------------------------------------- #
OCR_MAX_MB = _int("OCR_MAX_MB", 100)
OCR_MAX_BATCH_MB = max(OCR_MAX_MB, _int("OCR_MAX_BATCH_MB", 500))
OCR_JOB_TTL_MIN = _int("OCR_JOB_TTL_MIN", 30)
OCR_JOB_MAX_AGE_MIN = _int("OCR_JOB_MAX_AGE_MIN", 240)


# --------------------------------------------------------------------------- #
# network binding
# --------------------------------------------------------------------------- #
# The UI (tester.html) is served by this same process, so it always works
# whatever the host. HOST only decides who can reach the server:
#   127.0.0.1  this machine only (safe default)
#   0.0.0.0    every interface, so a deployed server answers on its real IP /
#              public hostname instead of refusing connections from anywhere
HOST = _str("HOST", "127.0.0.1")
PORT = _int("PORT", 8000)


def _allowed_origins():
    """Resolve OCR_ALLOWED_ORIGINS.

    The value is a comma-separated allow-list of exact origins. The literal
    ``*`` means "any origin", which is what a public deployment wants: the UI is
    same-origin, and without a CORS allow-list a client served from a different
    host is rejected. Nothing here grants access to the filesystem, so the API
    is reachable either way — only browser pre-flights are affected.
    """
    raw = _str("OCR_ALLOWED_ORIGINS", "http://localhost:5173,http://127.0.0.1:8000,http://localhost:8000")
    origins = [o.strip() for o in raw.split(",") if o.strip()]
    if "*" in origins:
        return "*"
    return origins


OCR_ALLOWED_ORIGINS = _allowed_origins()


def snapshot():
    """Non-secret configuration, for logging at startup."""
    return {
        "version": APP_VERSION,
        "documentWorkers": DOCUMENT_WORKER_CONCURRENCY,
        "ocrPageConcurrency": OCR_PAGE_CONCURRENCY,
        "arabicOcrVariants": OCR_ARABIC_VARIANTS_ENABLED,
        "arabicOcrVariantPsms": list(OCR_ARABIC_VARIANT_PSMS),
        "maxUploadMb": OCR_MAX_MB,
        "maxBatchUploadMb": OCR_MAX_BATCH_MB,
        "ocrJobTtlSeconds": OCR_JOB_TTL_MIN * 60,
        "host": HOST,
        "port": PORT,
        "cleanEnabled": CLEAN_ENABLED,
        "cleanDpi": CLEAN_DPI,
        "cleanOutput": CLEAN_OUTPUT,
        "documentLocalSigma": CLEAN_DOCUMENT_LOCAL_SIGMA,
        "documentLocalContrast": CLEAN_DOCUMENT_LOCAL_CONTRAST,
        "documentForegroundFloor": CLEAN_DOCUMENT_FOREGROUND_FLOOR,
        "documentWhiteFloor": CLEAN_DOCUMENT_WHITE_FLOOR,
        "cleanPreviewPagesMax": CLEAN_PREVIEW_PAGES_MAX,
        "washBright": CLEAN_WASH_BRIGHT,
        "washChroma": CLEAN_WASH_CHROMA,
        "washGrad": CLEAN_WASH_GRAD,
        "washGradDilate": CLEAN_WASH_GRAD_DILATE,
        "washNeutralChroma": CLEAN_WASH_NEUTRAL_CHROMA,
        "foldDev": CLEAN_FOLD_DEV,
        "foldBgSigma": CLEAN_FOLD_BG_SIGMA,
        "foldErode": CLEAN_FOLD_ERODE,
        "foldInkFloor": CLEAN_FOLD_INK_FLOOR,
        "cornerEnabled": CLEAN_CORNER_ENABLED,
        "cornerMarginRatio": CLEAN_CORNER_MARGIN_RATIO,
        "bandEnabled": CLEAN_BAND_ENABLED,
        "bandMinLengthRatio": CLEAN_BAND_MIN_LENGTH_RATIO,
    }