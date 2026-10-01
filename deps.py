#!/usr/bin/env python3
"""
deps.py — find Tesseract and Poppler, or explain exactly what's missing.

The Windows installers for both tools do not add themselves to PATH, and almost
nobody does it by hand, so "installed" and "findable" are different questions
there. This looks in the usual install locations before giving up.
"""

import glob
import os
import platform
import shutil

WINDOWS = platform.system() == "Windows"


class DependencyError(Exception):
    """Missing system tool. A normal Exception on purpose — raising SystemExit
    from a worker thread kills it silently, because SystemExit derives from
    BaseException and slips past `except Exception`."""


def _windows_candidates():
    local = os.environ.get("LOCALAPPDATA", "")
    home = os.path.expanduser("~")

    tess = [
        r"C:\Program Files\Tesseract-OCR",
        r"C:\Program Files (x86)\Tesseract-OCR",
        os.path.join(local, "Programs", "Tesseract-OCR"),
        os.path.join(local, "Tesseract-OCR"),
    ]

    # Poppler ships as a versioned folder, wherever the user unzipped it —
    # and each package manager parks it somewhere different.
    popp = []
    roots = [r"C:\Program Files", r"C:\Program Files (x86)", "C:\\",
             local, home, os.path.join(home, "Desktop"),
             os.path.join(home, "Downloads")]
    for root in roots:
        if root and os.path.isdir(root):
            popp += glob.glob(os.path.join(root, "poppler*", "Library", "bin"))
            popp += glob.glob(os.path.join(root, "poppler*", "bin"))

    # winget unpacks portable packages here and the folder name carries a hash
    if local:
        winget = os.path.join(local, "Microsoft", "WinGet", "Packages")
        popp += glob.glob(os.path.join(winget, "*Poppler*", "**", "bin"),
                          recursive=True)
        tess += glob.glob(os.path.join(winget, "*Tesseract*", "**"))
    # scoop and chocolatey
    popp += glob.glob(os.path.join(home, "scoop", "apps", "poppler", "*", "bin"))
    popp += glob.glob(r"C:\ProgramData\chocolatey\lib\poppler*\tools\**\bin",
                      recursive=True)

    return tess, popp


def _locate(exe, extra_dirs):
    """PATH first, then the known install locations."""
    found = shutil.which(exe)
    if found:
        return os.path.dirname(found)
    name = exe + ".exe" if WINDOWS else exe
    for d in extra_dirs:
        if d and os.path.isfile(os.path.join(d, name)):
            return d
    return None


def ensure(strict=True):
    """Locate both tools, adding them to PATH for this process if needed.

    Returns a dict describing what was found. With strict=True, raises
    DependencyError listing what to install instead of returning half-ready.
    """
    tess_dirs, popp_dirs = _windows_candidates() if WINDOWS else ([], [])

    tess = _locate("tesseract", tess_dirs)
    popp = _locate("pdftoppm", popp_dirs)

    for d in (tess, popp):
        if d and d not in os.environ.get("PATH", ""):
            os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")

    if tess:
        try:  # pytesseract looks this up separately from PATH
            import pytesseract
            pytesseract.pytesseract.tesseract_cmd = os.path.join(
                tess, "tesseract.exe" if WINDOWS else "tesseract")
        except ImportError:
            pass

    report = {"tesseract": tess, "poppler": popp,
              "ok": bool(tess and popp), "missing": []}
    if not tess:
        report["missing"].append("tesseract")
    if not popp:
        report["missing"].append("poppler")

    if strict and not report["ok"]:
        raise DependencyError(explain(report))
    return report


def explain(report):
    """Actionable install text for whatever is missing."""
    lines = []
    system = platform.system()
    for tool in report["missing"]:
        if system == "Windows":
            if tool == "tesseract":
                lines.append(
                    "Tesseract is missing. It is a program, not a Python package —\n"
                    "'pip install tesseract' installs something unrelated.\n\n"
                    "  winget install -e --id UB-Mannheim.TesseractOCR\n\n"
                    "Or download the installer from\n"
                    "  https://github.com/UB-Mannheim/tesseract/wiki\n"
                    "keeping the default folder, and tick your language packs.")
            else:
                lines.append(
                    "Poppler is missing. Also a program, not a Python package.\n\n"
                    "  winget install -e --id oschwartz10612.Poppler\n\n"
                    "Or download the zip from\n"
                    "  https://github.com/oschwartz10612/poppler-windows/releases\n"
                    "and extract it to C:\\poppler (so C:\\poppler\\Library\\bin exists).")
        elif system == "Darwin":
            lines.append(f"{tool} is missing. Run:  brew install "
                         f"{'tesseract tesseract-lang' if tool == 'tesseract' else 'poppler'}")
        else:
            lines.append(f"{tool} is missing. Run:  sudo apt install "
                         f"{'tesseract-ocr' if tool == 'tesseract' else 'poppler-utils'}")
    if system == "Windows":
        lines.append("Then close this terminal and open a new one, so it picks "
                     "up the change.")
    return "\n\n".join(lines)


def language_help(installed, wanted=("fra", "ara")):
    """Install text for language packs that are missing, or None.

    Worth surfacing rather than leaving to the documentation: a missing pack
    does not fail, it silently reads the document in the wrong language. An
    Arabic page OCR'd as English returns confident Latin nonsense, and the only
    symptom is that the extracted values are wrong.
    """
    missing = [code for code in wanted if code not in installed]
    if not missing:
        return None

    names = {"fra": "French", "ara": "Arabic", "spa": "Spanish",
             "deu": "German", "por": "Portuguese", "ita": "Italian",
             "nld": "Dutch", "rus": "Russian"}
    listed = ", ".join(names.get(code, code) for code in missing)
    system = platform.system()

    if system == "Windows":
        return (f"{listed} language data is not installed, so pages in "
                f"{listed} will be read as English.\nRe-run the Tesseract "
                f"installer and tick the missing languages under "
                f"'Additional language data'.")
    if system == "Darwin":
        return (f"{listed} language data is not installed, so pages in "
                f"{listed} will be read as English.\nRun:  brew install "
                f"tesseract-lang")
    return (f"{listed} language data is not installed, so pages in {listed} "
            f"will be read as English.\nRun:  sudo apt install "
            + " ".join(f"tesseract-ocr-{code}" for code in missing))


if __name__ == "__main__":
    r = ensure(strict=False)
    print("tesseract:", r["tesseract"] or "NOT FOUND")
    print("poppler:  ", r["poppler"] or "NOT FOUND")

    # Do not import this package: old releases contain Python 2 syntax and can
    # raise SyntaxError during the dependency check. Detect it without executing
    # any of its code.
    import importlib.util
    if importlib.util.find_spec("tesseract") is not None:
        print("\nNote: the PyPI package 'tesseract' is installed. It is not the "
              "OCR engine\nand is not used here. Remove it with: "
              "python -m pip uninstall tesseract")

    if not r["ok"]:
        print("\n" + explain(r))
    else:
        import autodetect
        langs = sorted(autodetect.installed_languages())
        print("languages:", ", ".join(langs))
        if langs == ["eng"]:
            print("\nOnly English is installed. Documents in other languages "
                  "will be read\nas English, quietly and less accurately.")
        print("\nReady. Run: python api.py")
