#!/usr/bin/env python3
"""
ocr_pdf.py — make a PDF searchable.

Renders each page, has Tesseract lay an invisible text layer over it, and
merges everything back into one PDF. The pages look identical; the text
becomes selectable and findable.

By default nothing needs configuring — language, resolution, page rotation
and image encoding are detected from the file (see autodetect.py). The flags
below only exist for when the detection gets it wrong.

    python ocr_pdf.py scan.pdf searchable.pdf
    python ocr_pdf.py scan.pdf out.pdf --lang fra+ara    # override detection
    python ocr_pdf.py scan.pdf out.pdf --dpi 500 --force

Requirements: tesseract-ocr, poppler-utils, and pip install -r requirements.txt
"""

import argparse
import io
import os
from concurrent.futures import ThreadPoolExecutor
import subprocess
import sys
import threading
import tempfile
from pathlib import Path

from PIL import Image, ImageOps
from pdf2image import convert_from_path, pdfinfo_from_path
from pypdf import PdfReader, PdfWriter

import autodetect
import config
import deps
from deps import DependencyError

MIN_CHARS_FOR_TEXT_PAGE = 20   # below this, a page counts as scanned
DETECT_DPI = 150               # cheap render used only for detection
DETECT_SAMPLE_PAGES = 3        # samples used for long documents
DETECT_ALL_PAGES_LIMIT = 12    # detect every page for ordinary documents


def _sample_pages(pages, count):
    """Spread `count` samples across the pages needing OCR.

    First, middle and last rather than the first three: a document that changes
    language does so partway through, and three consecutive pages from the top
    would miss it entirely.
    """
    if not pages:
        return []
    if len(pages) <= count:
        return list(pages)
    step = (len(pages) - 1) / (count - 1)
    picked = {pages[round(i * step)] for i in range(count)}
    return sorted(picked)


def _language_detection_pages(scanned, requested_lang):
    """Pages used to choose OCR languages.

    Small and medium documents are checked page by page. Long files keep the
    old first/middle/last sampling strategy to avoid a large detection penalty.
    A user-supplied language override disables automatic per-page selection.
    """
    if requested_lang != "auto":
        return _sample_pages(scanned, DETECT_SAMPLE_PAGES)
    try:
        limit = int(os.environ.get("OCR_DETECT_ALL_PAGES_LIMIT", DETECT_ALL_PAGES_LIMIT))
    except ValueError:
        limit = DETECT_ALL_PAGES_LIMIT
    limit = max(1, min(limit, 100))
    if len(scanned) <= limit:
        return list(scanned)
    return _sample_pages(scanned, DETECT_SAMPLE_PAGES)


def _nearest_finding(page_number, findings):
    """Exact page finding, or the closest sampled page for a long document."""
    if not findings:
        return None
    exact = next((item for item in findings if item.get("page") == page_number), None)
    if exact:
        return exact
    return min(findings, key=lambda item: abs(int(item.get("page", 1)) - page_number))


def _primary_page_language(finding, fallback="eng"):
    """Choose the Tesseract pack used for one page.

    Mixing a Latin pack into an Arabic-dominant page can turn a valid Arabic
    token into a Latin/Arabic hybrid. For an Arabic-dominant page we therefore
    use one Arabic pack only, preferring ``ara``. Other pages retain the
    detector's selected pack or pack combination.
    """
    if not finding:
        return fallback
    detected = str(finding.get("lang") or fallback)
    primary_script = str(finding.get("primary_script") or finding.get("script") or "")
    packs = [part for part in detected.split("+") if part]
    if primary_script.lower() == "arabic":
        if "ara" in packs:
            return "ara"
        arabic_packs = set(autodetect.SCRIPT_LANGS.get("Arabic", []))
        candidate = next((pack for pack in packs if pack in arabic_packs), None)
        return candidate or ("ara" if "ara" in autodetect.installed_languages() else detected)
    return detected


def _language_for_page(page_number, requested_lang, findings, fallback):
    if requested_lang != "auto":
        return requested_lang
    return _primary_page_language(_nearest_finding(page_number, findings), fallback)


# --------------------------------------------------------------------------- #
# environment
# --------------------------------------------------------------------------- #
def check_dependencies(lang="eng"):
    """Locate the system tools and validate the language pack.

    Raises DependencyError rather than calling sys.exit(): this runs inside a
    worker thread in the web app, and SystemExit would kill that thread without
    tripping `except Exception`, leaving the job stuck at "working" forever.
    """
    deps.ensure(strict=True)

    if lang == "auto":
        return
    have = autodetect.installed_languages()
    for code in lang.split("+"):
        if code not in have:
            raise DependencyError(
                f"The language pack '{code}' is not installed.\n"
                f"Installed: {', '.join(sorted(have))}\n"
                f"On Windows, re-run the Tesseract installer and tick that "
                f"language. On Linux: sudo apt install tesseract-ocr-{code}")


# --------------------------------------------------------------------------- #
# page helpers
# --------------------------------------------------------------------------- #
def pages_with_text(pdf_path):
    """Indices of pages that already carry a usable text layer."""
    found = set()
    try:
        for i, page in enumerate(PdfReader(pdf_path).pages):
            if len((page.extract_text() or "").strip()) >= MIN_CHARS_FOR_TEXT_PAGE:
                found.add(i)
    except Exception:
        pass
    return found


def prepare(img, rotate=0):
    """Return the greyscaled, auto-contrasted copy for Tesseract to read.

    The original image is never modified — it stays in the source PDF and is
    extracted directly, so there is zero quality loss in the output.
    """
    if rotate:
        img = img.rotate(-rotate, expand=True, fillcolor="white")
    return ImageOps.autocontrast(img.convert("L"))


def ocr_page_to_pdf(src_path, page_num, read_img, lang, dpi, psm=3,
                    mono=False, return_text=False):
    """One page: the original PDF page with an invisible text layer over it.

    The original page is extracted directly from the source PDF — no
    re-encoding, no JPEG compression, no quality loss. Only the invisible
    text layer from Tesseract is added on top.
    """
    with tempfile.TemporaryDirectory() as tmp:
        read_path = os.path.join(tmp, "read.png")
        out_base = os.path.join(tmp, "overlay")
        read_img.save(read_path, "PNG", dpi=(dpi, dpi))

        proc = subprocess.run(
            ["tesseract", read_path, out_base, "-l", lang, "--dpi", str(dpi),
             "--psm", str(psm), "-c", "textonly_pdf=1", "pdf", "txt"],
            capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"Tesseract failed: {proc.stderr.strip()}")

        overlay = PdfReader(out_base + ".pdf").pages[0]
        try:
            ocr_text = Path(out_base + ".txt").read_text(encoding="utf-8", errors="replace")
        except OSError:
            ocr_text = ""

    # Extract the original page directly — no re-encoding, no quality loss.
    original_page = PdfReader(src_path).pages[page_num]

    if mono:
        # For mono output, encode the greyscaled image (the only case where
        # we re-encode, because the user explicitly chose to discard colour).
        buf = io.BytesIO()
        flat = read_img.convert("L").point(lambda p: 255 if p > 128 else 0, mode="1")
        flat.save(buf, "PNG", optimize=True)
        import img2pdf
        page_pdf = img2pdf.convert(buf.getvalue(),
                                   layout_fun=img2pdf.get_fixed_dpi_layout_fun((dpi, dpi)))
        page = PdfReader(io.BytesIO(page_pdf)).pages[0]
    else:
        page = original_page

    if abs(float(page.mediabox.width) - float(overlay.mediabox.width)) > 1.5:
        raise RuntimeError("Text layer does not match the page size.")

    page.merge_page(overlay)
    out = PdfWriter()
    out.add_page(page)
    buf = io.BytesIO()
    out.write(buf)
    pdf_bytes = buf.getvalue()
    return (pdf_bytes, ocr_text) if return_text else pdf_bytes


def ocr_image_to_text(read_img, lang, dpi, psm):
    """Run a text-only Tesseract pass for a prepared page image.

    Used only for local Arabic layout alternatives. It does not create another
    searchable-PDF layer and never calls Qwen. Returning an empty string on a
    failed alternative keeps the primary OCR result authoritative.
    """
    with tempfile.TemporaryDirectory() as tmp:
        read_path = os.path.join(tmp, "read.png")
        out_base = os.path.join(tmp, "ocr")
        read_img.save(read_path, "PNG", dpi=(dpi, dpi))
        proc = subprocess.run(
            ["tesseract", read_path, out_base, "-l", lang, "--dpi", str(dpi),
             "--psm", str(psm), "txt"],
            capture_output=True, text=True,
        )
        if proc.returncode != 0:
            return ""
        try:
            return Path(out_base + ".txt").read_text(
                encoding="utf-8", errors="replace"
            )
        except OSError:
            return ""


def _arabic_text_variants(read_img, page_lang, dpi, primary_psm, enabled=True):
    """Alternative Arabic OCR texts for different page-layout assumptions."""
    if (not enabled or not config.OCR_ARABIC_VARIANTS_ENABLED
            or page_lang != "ara"):
        return []
    target_dpi = max(120, min(dpi, config.OCR_ARABIC_VARIANT_DPI))
    variant_img = read_img
    if dpi > target_dpi:
        scale = target_dpi / dpi
        variant_img = read_img.resize(
            (max(1, round(read_img.width * scale)),
             max(1, round(read_img.height * scale))),
            Image.Resampling.LANCZOS,
        )
    variants = []
    seen = set()
    for psm in config.OCR_ARABIC_VARIANT_PSMS:
        if psm == primary_psm or psm in seen:
            continue
        seen.add(psm)
        text = ocr_image_to_text(variant_img, "ara", target_dpi, psm)
        if text.strip():
            variants.append({"psm": psm, "text": text})
    return variants


# --------------------------------------------------------------------------- #
# main routine
# --------------------------------------------------------------------------- #
def default_jobs():
    """Tesseract is single-threaded per page, so pages run in parallel instead.

    Work happens in subprocesses (pdftoppm, tesseract), so threads are fine here
    — Python releases the GIL while waiting on them. Capped at 4: each worker
    holds a full-resolution page in memory, and the gains flatten out anyway
    once disk and RAM bandwidth become the limit.

    OCR_PAGE_CONCURRENCY overrides the cap. Worth lowering on a machine that is
    also hosting Ollama, where OCR and the model compete for the same cores.
    """
    try:
        cap = int(os.environ.get("OCR_PAGE_CONCURRENCY", 4))
    except ValueError:
        cap = 4
    return max(1, min(max(1, cap), (os.cpu_count() or 2)))


def make_searchable(src, dst, lang="auto", dpi=0, force=False, psm=3,
                    sidecar=None, quiet=False, progress=None,
                    jobs=None, mono=False, arabic_variants=False):
    """lang='auto' and dpi=0 mean detect. progress: callback(done, total, note)."""
    requested_lang = lang
    check_dependencies(requested_lang)

    total = pdfinfo_from_path(src)["Pages"]
    skip = set() if force else pages_with_text(src)
    settings = {"rotate": 0, "lang": lang, "dpi": dpi, "summary": ""}

    # ---- detection: cheap renders of a few pages needing OCR ----------------
    # More than one page, because a document does not have to be uniform. A
    # bilingual file often runs French for six pages and Arabic for the next
    # six; deciding from page one alone reads half the file with the wrong
    # language pack. Rotation and resolution still come from the first sample —
    # those really are properties of the scan — but the language packs are
    # unioned across the samples.
    scanned = [n for n in range(1, total + 1) if (n - 1) not in skip]
    samples = _language_detection_pages(scanned, requested_lang)
    findings = []

    if samples and (requested_lang == "auto" or not dpi):
        if progress:
            progress(0, total, "examining the document")

        for number in samples:
            page_img = convert_from_path(src, dpi=DETECT_DPI,
                                         first_page=number, last_page=number)[0]
            finding = autodetect.inspect(src, page_img)
            finding["page"] = number
            findings.append(finding)

        first = findings[0]
        settings.update(rotate=first["rotate"],
                        summary=autodetect.describe(first, findings))
        if requested_lang == "auto":
            # Kept as a document-level summary/fallback. The real OCR pass below
            # chooses the language again for every page.
            settings["lang"] = autodetect.merge_languages(findings)
        if not dpi:
            settings["dpi"] = first["dpi"]
        if not quiet:
            print(f"  {settings['summary']} ({settings['dpi']} DPI)")
    else:
        settings["lang"] = "eng" if requested_lang == "auto" else requested_lang
        settings["dpi"] = dpi or 300

    fallback_lang, dpi = settings["lang"], settings["dpi"]
    settings["page_languages"] = {
        n: _language_for_page(n, requested_lang, findings, fallback_lang)
        for n in scanned
    }

    if skip and not quiet:
        print(f"  {len(skip)}/{total} page(s) already contain text — left as they are.")

    # ---- the actual pass ----------------------------------------------------
    todo = [n for n in range(1, total + 1) if (n - 1) not in skip]
    results = {}                      # page number -> one-page PDF bytes
    page_texts = {}                   # page number -> direct Tesseract UTF-8 text
    page_text_variants = {}          # page number -> Arabic text-only OCR alternatives
    workers = max(1, min(jobs or default_jobs(), len(todo) or 1))
    done_count = [len(skip)]
    lock = threading.Lock()

    def handle(n):
        img = convert_from_path(src, dpi=dpi, first_page=n, last_page=n)[0]
        read = prepare(img, settings["rotate"])
        page_lang = settings["page_languages"].get(n, fallback_lang)
        data, direct_text = ocr_page_to_pdf(
            src, n - 1, read, page_lang, dpi, psm, mono, return_text=True
        )
        text_variants = _arabic_text_variants(
            read, page_lang, dpi, psm, enabled=arabic_variants
        )
        with lock:
            results[n] = data
            page_texts[n] = direct_text
            if text_variants:
                page_text_variants[n] = text_variants
            done_count[0] += 1
            if progress:
                progress(done_count[0], total, "reading text")
            if not quiet:
                print(
                    f"  page {done_count[0]}/{total} done "
                    f"(Tesseract: {page_lang})",
                    flush=True,
                )

    if not quiet and workers > 1:
        print(f"  {workers} pages at a time")
    if progress:
        progress(len(skip), total, "reading text")

    if todo:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            # list() re-raises the first worker exception here rather than
            # silently producing a PDF with pages missing
            list(pool.map(handle, todo))

    original = PdfReader(src)
    writer = PdfWriter()
    dump = []
    for n in range(1, total + 1):
        idx = n - 1
        if idx in skip:
            page = original.pages[idx]
            page_texts[n] = page.extract_text() or ""
        else:
            page = PdfReader(io.BytesIO(results[n])).pages[0]
        writer.add_page(page)
        if sidecar:
            dump.append(page_texts.get(n, ""))

    meta = dict(original.metadata or {})
    meta["/Producer"] = "ocr_pdf.py (Tesseract)"
    writer.add_metadata({k: str(v) for k, v in meta.items()})
    with open(dst, "wb") as fh:
        writer.write(fh)

    if sidecar:
        with open(sidecar, "w", encoding="utf-8") as fh:
            fh.write("\n\n".join(dump))
    if not quiet:
        print(f"\nSaved {dst} ({os.path.getsize(dst)/1_048_576:.1f} MB)")

    # Keep the direct Tesseract text for downstream extraction. Reading the
    # invisible PDF layer back through pypdf can reverse or corrupt Arabic even
    # when Tesseract produced the correct logical text.
    settings["page_texts"] = dict(page_texts)
    settings["page_text_variants"] = dict(page_text_variants)
    return settings


def cli():
    p = argparse.ArgumentParser(
        description="Make a scanned PDF searchable. Detects its own settings.")
    p.add_argument("input")
    p.add_argument("output")
    p.add_argument("--lang", default="auto",
                   help="override detection, e.g. fra or ara+fra")
    p.add_argument("--dpi", type=int, default=0,
                   help="override detected resolution (150-600)")
    p.add_argument("--psm", type=int, default=3,
                   help="page layout: 3 auto, 6 single block, 4 columns")
    p.add_argument("--force", action="store_true",
                   help="re-OCR pages that already have text")
    p.add_argument("--sidecar", metavar="FILE", help="also write the plain text here")
    p.add_argument("--mono", action="store_true",
                   help="force black and white output — much smaller, discards colour")
    p.add_argument("--jobs", type=int, default=0,
                   help=f"pages to process in parallel (default {default_jobs()})")
    p.add_argument("--quiet", action="store_true")
    a = p.parse_args()

    try:
        make_searchable(a.input, a.output, a.lang, a.dpi, a.force, a.psm,
                        a.sidecar, a.quiet,
                        jobs=a.jobs or None, mono=a.mono)
    except DependencyError as exc:
        sys.exit(f"\n{exc}\n")


if __name__ == "__main__":
    cli()
