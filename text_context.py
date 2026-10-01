#!/usr/bin/env python3
"""
text_context.py — decide which part of the document the model actually sees.

The local vision model has a finite context budget, and every token
spent on boilerplate is a token not spent on the paragraph that holds the
answer. For a two-page letter the right answer is "send all of it". For a
forty-page contract it is "send the neighbourhood of each field label".

Two things here are easy to get wrong and expensive to get wrong:

Budgeting in characters. A character cap is a proxy for a token cap, and the
proxy breaks on non-Latin scripts. English runs about 4 characters per token;
French with accents about 3; Arabic about 1.8. A 30 000-character budget is
7 500 English tokens but 16 000 Arabic tokens, which overflows num_ctx and
makes Ollama truncate from the front — dropping the system prompt first. So
the budget is computed in tokens and converted to characters using a ratio
estimated from the text itself.

Losing page numbers. The API contract promises a page number per extracted
value, so page identity has to survive selection. Every excerpt is emitted
under a `<<<PAGE n>>>` marker, and the model is told in the system prompt that
those markers are inserted by the system and are not part of the document.
"""

import re
import unicodedata

import config

PAGE_MARKER = "<<<PAGE {n}>>>"
GAP_MARKER = "[...]"

# How many lines from the top of the first page are always included. Names,
# reference numbers and dates cluster in letterheads, and a label search will
# not always find them because a letterhead often has no labels at all.
HEAD_LINES = 12

_ARABIC = re.compile(r"[\u0600-\u06FF\u0750-\u077F]")
_CJK = re.compile(r"[\u3000-\u9FFF\uF900-\uFAFF]")
_DIACRITICS = re.compile(r"[\u064B-\u0652\u0640]")
_BIDI_MARKS = re.compile(r"[\u200e\u200f\u202a-\u202e\u2066-\u2069]")
_ASCII_TOKEN = re.compile(r"[A-Za-z0-9@._+:/,\-]+")

# PyPDF frequently reads the invisible Arabic text layer produced by Tesseract
# in reverse logical order. The PDF still looks correct on screen, but a line
# such as "الاسم الكامل نادية بنعلي" can arrive as
# "يلعنب ةيدان لماكلا مسالا". These common document words let us choose
# between the original and a repaired candidate without reversing Arabic text
# that was already extracted correctly.
_ARABIC_DOCUMENT_WORDS = {
    "الاسم", "الكامل", "الموظف", "الموظفة", "الشركة", "المشغل",
    "العنوان", "الهاتف", "البريد", "الالكتروني", "الوظيفة", "العقد",
    "تاريخ", "بداية", "الميلاد", "الاجر", "الشهري", "الاجمالي",
    "فترة", "التجربة", "رقم", "البطاقة", "الوطنية", "العربية",
}


# --------------------------------------------------------------------------- #
# normalising — shared with result_validation
# --------------------------------------------------------------------------- #
def normalise_for_match(text):
    """Fold away everything that differs between what OCR produced and what a
    human would call "the same text": Unicode form, case, accents, Arabic
    diacritics and tatweel, and whitespace runs.

    Used both for finding field labels in the document and for checking that
    the model's quoted sourceText really occurs in it. Exact matching fails on
    almost every French and Arabic document; this is what makes the check
    meaningful rather than decorative.
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = _DIACRITICS.sub("", text)
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    text = unicodedata.normalize("NFC", text).casefold()
    return re.sub(r"\s+", " ", text).strip()




def _arabic_word_score(text):
    folded = normalise_for_match(text)
    return sum(1 for word in _ARABIC_DOCUMENT_WORDS if word in folded)


def _reverse_pdf_arabic_line(line):
    """Repair one Arabic-dominant line while preserving Latin/digit tokens."""
    candidate = _BIDI_MARKS.sub("", line)[::-1]
    # Reversing the whole line restores Arabic order but also reverses dates,
    # IDs and email fragments. Reverse those neutral tokens a second time.
    return _ASCII_TOKEN.sub(lambda match: match.group(0)[::-1], candidate)


def repair_extracted_text(text):
    """Repair reversed Arabic lines from a Tesseract-generated PDF layer.

    The transformation is deliberately conservative: a line is changed only
    when the repaired candidate contains more recognisable document vocabulary
    than the original. Real Arabic text that is already in logical order is
    therefore left untouched.
    """
    if not text or not _ARABIC.search(text):
        return text or ""

    repaired = []
    for line in text.splitlines():
        clean = _BIDI_MARKS.sub("", line)
        arabic_count = len(_ARABIC.findall(clean))
        if arabic_count < 3:
            repaired.append(clean)
            continue
        candidate = _reverse_pdf_arabic_line(clean)
        if _arabic_word_score(candidate) > _arabic_word_score(clean):
            repaired.append(candidate.strip())
        else:
            repaired.append(clean)
    return "\n".join(repaired)


def chars_per_token(text, sample=4000):
    """Rough characters-per-token ratio for the script in use.

    Deliberately an estimate: importing a tokeniser to count exactly would add
    a heavy dependency to save a margin we can simply leave in place instead.
    The numbers err on the pessimistic side.
    """
    probe = text[:sample]
    if not probe:
        return 4.0
    arabic = len(_ARABIC.findall(probe))
    cjk = len(_CJK.findall(probe))
    total = max(1, len(probe))
    if cjk / total > 0.10:
        return 1.2
    if arabic / total > 0.10:
        return 1.8
    non_ascii = sum(1 for c in probe if ord(c) > 127)
    if non_ascii / total > 0.05:      # accented Latin: French, Spanish...
        return 3.0
    return 4.0


def budget_characters(text):
    """Character ceiling for this document's script, honouring both caps."""
    ratio = chars_per_token(text)
    from_tokens = int(config.MAX_LLM_INPUT_TOKENS * ratio)
    ceiling = min(config.MAX_LLM_INPUT_CHARACTERS, from_tokens)
    # A floor keeps a pathological ratio from starving the prompt, but it can
    # never override an explicitly configured character cap.
    return max(ceiling, min(1000, config.MAX_LLM_INPUT_CHARACTERS))


# --------------------------------------------------------------------------- #
# page handling
# --------------------------------------------------------------------------- #
def normalise_pages(pages):
    """Clean the output of extract_pages_text into tidy per-page line lists.

    Tesseract's text layer arrives with ragged spacing and empty lines where
    the page had white space. Collapsing that is worth real tokens.
    """
    out = []
    for entry in pages or []:
        number = entry.get("page")
        raw = repair_extracted_text(entry.get("text") or "")
        lines = []
        for line in raw.splitlines():
            line = re.sub(r"[ \t\u00a0]+", " ", line).strip()
            if line:
                lines.append(line)
        out.append({"page": number, "lines": lines})
    return out


def _render(pages_lines):
    """Whole document, page-marked."""
    chunks = []
    for page in pages_lines:
        if not page["lines"]:
            continue
        chunks.append(PAGE_MARKER.format(n=page["page"]))
        chunks.append("\n".join(page["lines"]))
    return "\n".join(chunks)


# --------------------------------------------------------------------------- #
# selection
# --------------------------------------------------------------------------- #
def _windows_for_field(pages_lines, field, before, after):
    """Line ranges around every occurrence of a field's search terms."""
    hits = []
    terms = field.search_terms()
    if (field.language_mode == "specific_language"
            and (field.language or "").split("-", 1)[0] == "ar"):
        # A French UI label must not pull the French page into an Arabic-only
        # request. Prefer Arabic aliases; script windows below provide a safe
        # fallback when the user did not enter an alias.
        terms = [term for term in terms if _ARABIC.search(term)]
    for term in terms:
        needle = normalise_for_match(term)
        if len(needle) < 3:
            continue
        for page in pages_lines:
            for index, line in enumerate(page["lines"]):
                if needle in normalise_for_match(line):
                    hits.append((page["page"],
                                 max(0, index - before),
                                 min(len(page["lines"]), index + after + 1)))
        if hits:
            # The longest matching term already found something; shorter,
            # vaguer terms would only widen the selection with noise.
            break
    return hits


def _windows_for_requested_script(pages_lines, field, before, after,
                                  max_hits=24):
    """Add useful excerpts when the UI label is not in the target language.

    A user may define the field as "Nom en arabe" without knowing the Arabic
    label used by the document. Label matching then finds the French page, not
    the Arabic page. For Arabic-only textual fields, retain windows around
    Arabic-script lines so the model actually sees candidate values.
    """
    if (field.language_mode != "specific_language"
            or (field.language or "").split("-", 1)[0] != "ar"
            or field.type not in ("string", "array")):
        return []

    hits = []
    for page in pages_lines:
        for index, line in enumerate(page["lines"]):
            if _ARABIC.search(line):
                hits.append((page["page"],
                             max(0, index - before),
                             min(len(page["lines"]), index + after + 1)))
                if len(hits) >= max_hits:
                    return hits
    return hits


def _merge(windows):
    """Collapse overlapping and adjacent ranges, per page."""
    by_page = {}
    for page, start, end in windows:
        by_page.setdefault(page, []).append((start, end))

    merged = []
    for page in sorted(by_page):
        ranges = sorted(by_page[page])
        current_start, current_end = ranges[0]
        for start, end in ranges[1:]:
            if start <= current_end + 1:          # touching counts as overlapping
                current_end = max(current_end, end)
            else:
                merged.append((page, current_start, current_end))
                current_start, current_end = start, end
        merged.append((page, current_start, current_end))
    return merged


def build_context(pages, fields):
    """Return the text to send to the model, plus what it covers.

    {
      "text":       str,          # page-marked excerpts, or the whole document
      "strategy":   "full" | "selected",
      "pages":      [1, 2, 4],    # pages actually represented in `text`
      "characters": int,
      "budget":     int,
      "truncated":  bool,
    }
    """
    pages_lines = normalise_pages(pages)
    whole = _render(pages_lines)
    budget = budget_characters(whole)

    if not whole.strip():
        return {"text": "", "strategy": "full", "pages": [], "characters": 0,
                "budget": budget, "truncated": False}

    language_focused = bool(fields) and all(
        field.language_mode == "specific_language" for field in fields)

    # Short documents normally go in whole. A language-specific request is the
    # exception: showing a 1.5B model three copies of the same name in FR/AR/EN
    # makes it prefer the Latin one even when Arabic was requested.
    if len(whole) <= budget and not language_focused:
        return {"text": whole, "strategy": "full",
                "pages": [p["page"] for p in pages_lines if p["lines"]],
                "characters": len(whole), "budget": budget, "truncated": False}

    # ---- gather candidate windows, most valuable first ---------------------
    ordered = []
    first_page = next((p for p in pages_lines if p["lines"]), None)
    if first_page and not language_focused:
        ordered.append((first_page["page"], 0, min(HEAD_LINES,
                                                   len(first_page["lines"]))))
    for field in fields:
        ordered.extend(_windows_for_field(pages_lines, field,
                                          config.CONTEXT_LINES_BEFORE,
                                          config.CONTEXT_LINES_AFTER))
        ordered.extend(_windows_for_requested_script(
            pages_lines, field,
            config.CONTEXT_LINES_BEFORE,
            config.CONTEXT_LINES_AFTER))

    if not ordered:
        # No field label or requested-script line matched anywhere. Rather than
        # send an unrelated first page, send as much as fits and let validation
        # reject a value in the wrong script.
        clipped = whole[:budget]
        return {"text": clipped, "strategy": "selected",
                "pages": [p["page"] for p in pages_lines if p["lines"]],
                "characters": len(clipped), "budget": budget,
                "truncated": len(whole) > budget}

    # ---- fill the budget in priority order, then emit in reading order -----
    lines_by_page = {p["page"]: p["lines"] for p in pages_lines}
    kept, used, truncated = [], 0, False
    for window in ordered:
        page, start, end = window
        size = sum(len(l) + 1 for l in lines_by_page[page][start:end]) + 16
        if used + size > budget:
            truncated = True
            continue
        kept.append(window)
        used += size

    chunks, seen_pages, last_page = [], [], None
    for page, start, end in _merge(kept):
        if page != last_page:
            chunks.append(PAGE_MARKER.format(n=page))
            last_page = page
            seen_pages.append(page)
        else:
            chunks.append(GAP_MARKER)
        chunks.append("\n".join(lines_by_page[page][start:end]))

    text = "\n".join(chunks)
    return {"text": text, "strategy": "selected", "pages": seen_pages,
            "characters": len(text), "budget": budget, "truncated": truncated}


def flatten(pages):
    """All page text as one string — for sourceText verification."""
    return "\n".join((entry.get("text") or "") for entry in pages or [])
