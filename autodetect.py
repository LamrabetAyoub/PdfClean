#!/usr/bin/env python3
"""
autodetect.py — work out good OCR settings from the file itself.

Every knob ocr_pdf.py exposes can be inferred:

  rotation  Tesseract's orientation-and-script detection (OSD) reports how far
            the page is turned. Sideways scans are common and users rarely
            think to mention them.
  dpi       An embedded scan has a native resolution. Rendering above it just
            interpolates; rendering below it throws away detail. Read it from
            the PDF and clamp into the range Tesseract is happiest with.
  language  OSD names the script (Latin, Arabic, Cyrillic...). Within that
            script, run a fast low-resolution pass in each installed candidate
            and keep whichever Tesseract is most confident about.
  encoding  A clean document scan is essentially two-tone, so 1-bit PNG beats
            JPEG on size and is lossless as well. Photos and grubby scans need
            JPEG. Measure the histogram and choose.
"""

import re
import subprocess

from PIL import Image, ImageOps
import pytesseract

# Languages worth trying per script, roughly by how often they turn up.
SCRIPT_LANGS = {
    "Latin": ["eng", "fra", "spa", "deu", "por", "ita", "nld"],
    "Arabic": ["ara", "fas", "urd"],
    "Cyrillic": ["rus", "ukr", "bul"],
    "Han": ["chi_sim", "chi_tra"],
    "HanS": ["chi_sim"], "HanT": ["chi_tra"],
    "Japanese": ["jpn"], "Hangul": ["kor"],
    "Greek": ["ell"], "Hebrew": ["heb"], "Thai": ["tha"], "Devanagari": ["hin"],
}
# Characters that prove a script is present, for counting what a probe pass
# actually recognised rather than trusting its confidence score.
SCRIPT_PATTERNS = {
    "Latin": re.compile(r"[A-Za-z\u00C0-\u024F]"),
    "Arabic": re.compile(r"[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]"),
    "Cyrillic": re.compile(r"[\u0400-\u04FF]"),
    "Greek": re.compile(r"[\u0370-\u03FF]"),
    "Hebrew": re.compile(r"[\u0590-\u05FF]"),
}

# Which other script routinely shares a page with each one. OSD reports a
# single script per page, so a bilingual document — an Arabic heading over a
# French body, which is the norm for administrative paperwork across the
# Maghreb — gets classified as whichever script won, and the other half is then
# read with the wrong language pack and comes out as noise. After OSD names the
# primary script we probe its usual partner and combine when the evidence is
# there.
MIXED_SCRIPT_PARTNERS = {
    "Latin": ["Arabic"],
    "Arabic": ["Latin"],
    "Cyrillic": ["Latin"],
    "Greek": ["Latin"],
    "Hebrew": ["Latin"],
}

PROBE_DPI = 150          # detection only; the real pass renders properly
BILEVEL_THRESHOLD = 0.92  # share of near-black/near-white pixels
MIN_SECONDARY_CHARS = 12   # below this, a few stray glyphs are not a language
MIN_SECONDARY_CONF = 55    # Tesseract confidence floor for counted characters
MIN_SECONDARY_SHARE = 0.25  # relative to the primary script's character count

# The absolute floor alone is not enough. An Arabic pack turned loose on a
# French page finds a handful of Arabic-looking glyphs in the accents, and an
# English pack on an Arabic page finds a few Latin ones in the numerals — on
# measured samples that noise runs at 7-9% of the primary script's character
# count, while a genuinely bilingual page runs near 100%. The share threshold
# is what separates those two cases; the absolute floor only guards against
# dividing tiny numbers.


def installed_languages():
    try:
        out = subprocess.run(["tesseract", "--list-langs"],
                             capture_output=True, text=True).stdout
        # First line is a header sentence; real codes are one per line after it.
        codes = {ln.strip() for ln in out.splitlines()[1:]
                 if ln.strip() and " " not in ln.strip()}
        return {c for c in codes if c not in ("osd", "equ")} or {"eng"}
    except Exception:
        return {"eng"}


def native_dpi(pdf_path, default=300, lo=300, hi=450, raw=False):
    """Resolution of the images actually embedded in the PDF.

    Below `lo` Tesseract starts losing small type, so we oversample rather than
    trust a 96 DPI scan. Above `hi` we gain nothing and pay in time and size.
    """
    try:
        out = subprocess.run(["pdfimages", "-list", pdf_path],
                             capture_output=True, text=True).stdout
        ppi = []
        for line in out.splitlines()[2:]:
            parts = line.split()
            if len(parts) > 13 and parts[2] == "image":
                try:
                    ppi.append(int(parts[12]))
                except ValueError:
                    pass
        if ppi:
            ppi.sort()
            median = ppi[len(ppi) // 2]
            if median > 0:
                return median if raw else max(lo, min(hi, median))
    except Exception:
        pass
    return default


def orientation_and_script(img):
    """Returns (clockwise_degrees_to_correct, script_name)."""
    try:
        osd = pytesseract.image_to_osd(img)
        rotate = int(re.search(r"Rotate: (\d+)", osd).group(1))
        script = re.search(r"Script: (\w+)", osd).group(1)
        conf = float(re.search(r"Orientation confidence: ([\d.]+)", osd).group(1))
        # Low confidence usually means a sparse page; leave it alone rather
        # than confidently turning an upright page on its side.
        return (rotate if conf >= 2.0 else 0), script
    except Exception:
        return 0, "Latin"


def score_language(img, lang, script=None):
    """Mean Tesseract confidence, weighted by how much text it found.

    Confidence alone is misleading: a pass that reads three words at 95% is
    worse than one that reads three hundred at 80%.

    `script` restricts the count to words containing that script's characters.
    On a bilingual page this matters: the Arabic half read by a Latin pack
    produces confident nonsense, and letting that nonsense into the score means
    whichever Latin pack hallucinates most enthusiastically wins the comparison
    between French and English on the half of the page that is actually French.
    """
    pattern = SCRIPT_PATTERNS.get(script) if script else None
    try:
        data = pytesseract.image_to_data(
            img, lang=lang, output_type=pytesseract.Output.DICT)
        confs, chars = [], 0
        for c, txt in zip(data["conf"], data["text"]):
            c = float(c)
            if c <= 0 or not txt.strip():
                continue
            counted = len(pattern.findall(txt)) if pattern else len(txt.strip())
            if counted:
                confs.append(c)
                chars += counted
        if not confs:
            return 0.0
        mean = sum(confs) / len(confs)
        return mean * min(chars, 400) / 400
    except Exception:
        return 0.0


def script_evidence(img, lang, script):
    """How many characters of `script` a pass in `lang` actually recognised.

    Counting characters is a far better signal than mean confidence for the
    question "is this script on the page at all". Tesseract will happily
    hallucinate Latin words out of Arabic text at 80% confidence — that is
    exactly what "Sah Sah" is — but it cannot hallucinate Arabic codepoints out
    of Latin text, because they are not in the model's output alphabet for a
    Latin pass. So we ask the candidate pack to read the page and count how
    much of its own script comes back.
    """
    pattern = SCRIPT_PATTERNS.get(script)
    if pattern is None:
        return 0
    try:
        data = pytesseract.image_to_data(
            img, lang=lang, output_type=pytesseract.Output.DICT)
    except Exception:
        return 0

    found = 0
    for conf, text in zip(data["conf"], data["text"]):
        try:
            conf = float(conf)
        except (TypeError, ValueError):
            continue
        if conf >= MIN_SECONDARY_CONF and text.strip():
            found += len(pattern.findall(text))
    return found


def script_groups(available):
    """Installed packs grouped by script, for scripts we can verify by evidence."""
    groups = {}
    for script, langs in SCRIPT_LANGS.items():
        if script not in SCRIPT_PATTERNS:
            continue                    # no pattern means no way to count it
        have = [c for c in langs if c in available]
        if have:
            groups[script] = have
    return groups


def measure_scripts(probe, groups):
    """Characters of each script the page actually yields. {script: count}."""
    return {
        script: max(script_evidence(probe, lang, script) for lang in langs[:2])
        for script, langs in groups.items()
    }


def _pick(scored):
    """Best language from (score, lang) pairs, or the top two when it is close.

    Two packs scoring alike means the evidence does not separate them — which
    on short text is the normal case for English against French. Loading both
    costs time; guessing wrong costs the value. On a measured bilingual sample
    the wrong guess turned "1 août 2026" into "1 200] 6", so this errs towards
    loading both.
    """
    best_score, best = scored[0]
    runner_up = scored[1][0] if len(scored) > 1 else 0
    if runner_up > best_score * 0.93:
        return f"{best}+{scored[1][1]}"
    return best


def languages_for_script(probe, script, langs, alongside=None):
    """Best pack (or two, when it is close) within one script."""
    if len(langs) == 1:
        return langs[0]
    scored = sorted(
        ((score_language(probe, f"{c}+{alongside}" if alongside else c, script), c)
         for c in langs[:4]), reverse=True)
    return _pick(scored)


def detect_language(img, script, available=None):
    """Returns (tesseract lang string, whether it was chosen, scripts present).

    `script` is Tesseract's OSD guess. It is a hint, not the answer: OSD is a
    weak classifier on noisy or bilingual pages and its verdict moves with the
    render resolution — the same Arabic/French contract that OSD calls Latin at
    300 DPI it calls Cyrillic at the 150 DPI the detection pass actually uses.
    Acting on that directly meant falling back to English and reading the whole
    Arabic half as noise. So the script is established by measuring what each
    installed pack can actually recover from the page, and OSD only breaks ties.
    """
    available = available or installed_languages()
    probe = img.copy()
    probe.thumbnail((1100, 1100))          # small = fast; enough for scoring

    groups = script_groups(available)
    if not groups:
        # Only packs we cannot verify by pattern (CJK and friends). Fall back
        # to trusting OSD, which is all the information there is.
        langs = [c for c in SCRIPT_LANGS.get(script, []) if c in available]
        if langs:
            return languages_for_script(probe, script, langs), len(langs) > 1, [script]
        return sorted(available)[0], False, [script]

    # ---- which scripts are actually on this page? --------------------------
    measured = measure_scripts(probe, groups)
    ranked = sorted(measured.items(), key=lambda kv: kv[1], reverse=True)
    primary_script, primary_chars = ranked[0]

    # OSD gets the benefit of the doubt only when the measurement is close.
    if script in measured and measured[script] >= primary_chars * 0.8:
        primary_script, primary_chars = script, measured[script]

    if primary_chars == 0:
        fallback = "eng" if "eng" in available else sorted(available)[0]
        return fallback, False, [primary_script]

    # ---- a second script sharing the page ----------------------------------
    partners = MIXED_SCRIPT_PARTNERS.get(primary_script, [])
    secondary_script = next(
        (s for s, chars in ranked
         if s != primary_script and s in partners
         and chars >= MIN_SECONDARY_CHARS
         and chars >= primary_chars * MIN_SECONDARY_SHARE),
        None)

    if secondary_script is None:
        lang = languages_for_script(probe, primary_script, groups[primary_script])
        return lang, len(groups[primary_script]) > 1, [primary_script]

    # ---- choose packs for both, each judged alongside the other ------------
    # Judging a Latin pack on a bilingual page without the Arabic pack loaded
    # means the Arabic half contributes confident nonsense — an English model
    # reading Arabic emits Latin words — and whichever pack hallucinates most
    # wins. Loading the other script lets it consume its own half so the
    # comparison runs on text that is genuinely in the script being judged.
    secondary_hint = groups[secondary_script][0]
    primary_lang = languages_for_script(probe, primary_script,
                                        groups[primary_script],
                                        alongside=secondary_hint)
    secondary_lang = languages_for_script(probe, secondary_script,
                                          groups[secondary_script],
                                          alongside=primary_lang.split("+")[0])

    # Tesseract weights the first pack most heavily, so the dominant script
    # leads. Recognition also slows roughly linearly in the number of packs, so
    # three is the ceiling: two for a genuinely mixed page, one spare for an
    # unresolved English-or-French tie.
    combined = primary_lang.split("+") + secondary_lang.split("+")
    combined = list(dict.fromkeys(combined))[:3]
    return "+".join(combined), True, [primary_script, secondary_script]


def has_colour(img, threshold=16, sample=400):
    """True if the page carries real colour, not just greyscale in an RGB file.

    Scans are routinely saved as RGB with identical channels; treating those as
    colour would cost file size for nothing.
    """
    if img.mode in ("L", "1"):
        return False
    small = img.convert("RGB").copy()
    small.thumbnail((sample, sample))
    grey = small.convert("L").convert("RGB")
    from PIL import ImageChops
    diff = ImageChops.difference(small, grey)
    return (diff.getextrema()[0][1] > threshold or
            diff.getextrema()[1][1] > threshold or
            diff.getextrema()[2][1] > threshold)


def is_bilevel(img):
    """True for ordinary black-text-on-white document scans."""
    grey = ImageOps.autocontrast(img.convert("L"))
    hist = grey.histogram()
    total = sum(hist) or 1
    return (sum(hist[:26]) + sum(hist[230:])) / total > BILEVEL_THRESHOLD


def inspect(pdf_path, sample_img):
    """One page's worth of evidence -> settings for the whole document."""
    rotate, script = orientation_and_script(sample_img)
    if rotate:
        sample_img = sample_img.rotate(-rotate, expand=True)

    lang, chose, scripts = detect_language(sample_img, script)
    return {
        "dpi": native_dpi(pdf_path),
        # what the scan actually holds, before clamping up for recognition
        "native_dpi": native_dpi(pdf_path, raw=True),
        "rotate": rotate,
        "script": script,
        # ``scripts`` is ordered by measured dominance. Keeping the first one
        # lets the OCR pipeline choose one primary pack for pages whose scripts
        # should not be mixed (notably Arabic pages where a Latin pack can
        # corrupt names).
        "primary_script": scripts[0] if scripts else script,
        "scripts": scripts,
        "mixed": len(scripts) > 1,
        "lang": lang,
        "lang_detected": chose,
        # bilevel describes the ORIGINAL page, not the cleaned-up copy used for
        # recognition — it decides how the page is stored, so it must not be
        # measured on an image we already flattened.
        "bilevel": is_bilevel(sample_img) and not has_colour(sample_img),
        "colour": has_colour(sample_img),
    }


def merge_languages(findings, limit=3):
    """One Tesseract language string covering every sampled page.

    Ordered by how many samples wanted each pack, so the language that
    dominates the document leads — Tesseract weights the first one most.
    """
    counts = {}
    for order, found in enumerate(findings):
        for position, lang in enumerate(found["lang"].split("+")):
            # Earlier in a page's own list, and earlier in the document, both
            # count for more.
            counts[lang] = counts.get(lang, 0) + 10 - position - (order * 0.1)
    ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    return "+".join(lang for lang, _ in ranked[:limit])


def describe(s, findings=None):
    """Plain sentence for the interface — no jargon."""
    bits = []
    if s["rotate"]:
        bits.append(f"rotated {s['rotate']}° to straighten it")

    langs = merge_languages(findings) if findings else s["lang"]
    mixed = s.get("mixed") or (
        findings is not None
        and len({f["lang"] for f in findings}) > 1)

    if mixed:
        bits.append(f"found more than one script, reading it as "
                    f"{langs.replace('+', ' and ')}")
    elif s["lang_detected"]:
        bits.append(f"detected {langs.replace('+', ' and ')}")
    else:
        bits.append(f"read as {langs}")
    bits.append("kept in colour" if s.get("colour") else "black and white original")
    return "Auto: " + ", ".join(bits) + "."
