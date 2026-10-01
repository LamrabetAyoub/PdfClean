#!/usr/bin/env python3
"""
archive.py — accept a .zip of PDFs as safely as a single upload.

A zip from an untrusted browser is not just a folder. Three things have to be
refused before anything is written to disk: entry names that escape the target
directory, archives that expand to far more than they claim, and archives with
so many entries that the queue is effectively a denial of service.
"""

import os
import zipfile

MAX_FILES = 100
MAX_TOTAL_BYTES = 500 * 1024 * 1024      # uncompressed ceiling
MAX_RATIO = 200                          # uncompressed / compressed


class ArchiveError(Exception):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def is_zip(filename):
    return filename.lower().endswith(".zip")


def _wanted(info):
    """PDF entries only, ignoring directories and archiver metadata."""
    name = info.filename
    if info.is_dir():
        return False
    base = os.path.basename(name)
    if not base or base.startswith("."):          # ._resource forks
        return False
    if name.startswith("__MACOSX/") or "/.," in name:
        return False
    return base.lower().endswith(".pdf")


def _safe_name(name, taken):
    """Flatten to a bare filename. Nested paths are the whole zip-slip problem,
    and the folder structure carries no meaning for OCR anyway."""
    base = os.path.basename(name.replace("\\", "/")).strip()
    base = "".join(c for c in base if c not in '<>:"|?*').strip(". ")
    if not base:
        base = "document.pdf"

    stem, ext = os.path.splitext(base)
    candidate, n = base, 2
    while candidate.lower() in taken:            # zips can hold a.pdf and A.pdf
        candidate = f"{stem} ({n}){ext}"
        n += 1
    taken.add(candidate.lower())
    return candidate


def extract_pdfs(zip_path, dest_dir):
    """Returns [(display_name, path_on_disk)] in archive order.

    Raises ArchiveError with a machine-readable code on anything suspicious.
    """
    try:
        zf = zipfile.ZipFile(zip_path)
    except zipfile.BadZipFile:
        raise ArchiveError("corrupt_zip", "This ZIP file could not be opened.")

    with zf:
        if zf.testzip() is not None:
            raise ArchiveError("corrupt_zip", "This ZIP file is damaged.")

        entries = [i for i in zf.infolist() if _wanted(i)]
        if not entries:
            raise ArchiveError("empty_zip", "The ZIP contains no PDF files.")
        if len(entries) > MAX_FILES:
            raise ArchiveError(
                "too_many_files",
                f"The ZIP contains {len(entries)} PDFs; the limit is {MAX_FILES}.")

        declared = sum(i.file_size for i in entries)
        packed = sum(i.compress_size for i in entries) or 1
        if declared > MAX_TOTAL_BYTES:
            raise ArchiveError(
                "zip_too_large",
                f"The PDFs unpack to {declared // 1048576} MB; "
                f"the limit is {MAX_TOTAL_BYTES // 1048576} MB.")
        if declared / packed > MAX_RATIO:
            raise ArchiveError("suspicious_zip",
                               "This ZIP expands far beyond its size and was rejected.")

        results, taken, written = [], set(), 0
        for info in entries:
            name = _safe_name(info.filename, taken)
            target = os.path.join(dest_dir, name)

            with zf.open(info) as src, open(target, "wb") as out:
                # Stream and count: the header's file_size is a claim, not a fact.
                while True:
                    chunk = src.read(65536)
                    if not chunk:
                        break
                    written += len(chunk)
                    if written > MAX_TOTAL_BYTES:
                        raise ArchiveError(
                            "zip_too_large",
                            "The ZIP expands beyond the size limit.")
                    out.write(chunk)
            results.append((name, target))

    return results
