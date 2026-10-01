#!/usr/bin/env python3
"""
api.py — OCR-only JSON API for making scanned PDFs searchable.

This is the reusable OCR half, separated from the form/extraction layer of the
original OCR-Formulaire project. It has no AI model and no form editor: upload
a PDF, several PDFs from a folder, or a ZIP of PDFs and get back a searchable
PDF (or a ZIP of searchable PDFs) plus the per-page OCR text. With cleaning
enabled that one file is cleaned as well: there is never a second output.

    pip install -r requirements.txt
    python api.py                                # dev,  127.0.0.1:8000
    HOST=0.0.0.0 PORT=8000 python api.py          # reachable from other machines

For a deployed server, bind HOST=0.0.0.0 so it answers on its real IP or
public hostname rather than refusing every remote connection, and put a reverse
proxy in front of it for TLS. The wait line is a plain WSGI app:

    waitress-serve --listen=0.0.0.0:8000 api:app  # production, Windows
    gunicorn -w 1 --threads 8 -b 0.0.0.0:8000 api:app  # production, Linux

Optional scanned-document cleaning (image_clean.py) runs BEFORE OCR so dirty
pages (gray/blue cast, shadows, spots, paper texture) reach Tesseract already
whitened. Send the form flags:

    clean=1        clean each page, then OCR the cleaned PDF into one result
    clean_mode=auto|gray|color|binary   force an output style (default auto)

Cleaning happens before OCR, so a cleaned run yields a single PDF that is both
whitened and searchable. Nothing is written twice and there is no second
artifact to download.

Jobs run in the background; poll GET /api/jobs/<id> until it reports ``done``,
then download the result. Originals are never modified: cleaning writes a new
PDF into the job work directory.

Configuration lives in config.py, which reads the environment. See .env.example.
"""

import os
import shutil
import socket
import tempfile
import threading
import time
import uuid
import zipfile

from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from pypdf import PdfReader
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename

import archive
import autodetect
import config
import deps
import image_clean
import job_queue
import text_context
from archive import ArchiveError
from deps import DependencyError
from image_clean import CleanError
from ocr_pdf import make_searchable

MAX_MB = config.OCR_MAX_MB
MAX_BATCH_MB = config.OCR_MAX_BATCH_MB
TTL = config.OCR_JOB_TTL_MIN * 60
MAX_AGE = config.OCR_JOB_MAX_AGE_MIN * 60
ORIGINS = config.OCR_ALLOWED_ORIGINS

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_BATCH_MB * 1024 * 1024

# Content-Disposition must be exposed, or fetch() cannot read the filename.
# A wildcard allow-list cannot be combined with credentials, and none is used.
CORS(app, resources={r"/api/*": {"origins": ORIGINS}},
     expose_headers=["Content-Disposition"],
     supports_credentials=False)

WORK_DIR = tempfile.mkdtemp(prefix="ocrapi_")
JOBS = {}
LOCK = threading.Lock()
DEPS = deps.ensure(strict=False)
QUEUE = job_queue.create(workers=config.DOCUMENT_WORKER_CONCURRENCY)


# --------------------------------------------------------------------------- #
# errors — every failure leaves as JSON with a stable code
# --------------------------------------------------------------------------- #
def fail(code, message, status=400):
    return jsonify(error={"code": code, "message": message}), status


@app.errorhandler(RequestEntityTooLarge)
def too_large(_):
    return fail("file_too_large",
                f"Maximum upload size is {MAX_MB} MB for one file "
                f"or {MAX_BATCH_MB} MB for a batch.", 413)


@app.errorhandler(404)
def no_route(_):
    return fail("not_found", "No such endpoint.", 404)


@app.errorhandler(Exception)
def unhandled(exc):
    app.logger.exception(exc)
    return fail("internal_error", "Unexpected server error.", 500)


# --------------------------------------------------------------------------- #
# text extraction
# --------------------------------------------------------------------------- #
def read_text(path):
    """Whole document as one string."""
    try:
        return "\n\n".join((p.extract_text() or "") for p in PdfReader(path).pages)
    except Exception:
        return ""


def read_pages_text(path):
    """Same text, with page boundaries intact."""
    try:
        return [{
            "page": index + 1,
            "text": text_context.repair_extracted_text(page.extract_text() or ""),
        } for index, page in enumerate(PdfReader(path).pages)]
    except Exception:
        return []


def classify(exc):
    text = str(exc).lower()
    if "password" in text or "encrypt" in text:
        return "password_protected", "This PDF is password protected."
    if "eof" in text or "damaged" in text or "startxref" in text:
        return "corrupt_pdf", "This PDF could not be read; it may be damaged."
    if isinstance(exc, CleanError):
        return exc.code, exc.message
    return "processing_failed", f"OCR failed: {str(exc)[:200]}"


def form_flag(name, default=False):
    """Read a boolean form field sent as 1/true/yes/on."""
    raw = request.form.get(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


# --------------------------------------------------------------------------- #
# worker
# --------------------------------------------------------------------------- #
def process(job_id, upload_path, is_archive, work, sources=None,
             cleanup_paths=None):
    job = JOBS.get(job_id)
    if job is None:                 # deleted while it sat in the queue
        shutil.rmtree(work, ignore_errors=True)
        return

    job["status"] = "running"
    started = time.time()

    try:
        # ---- gather the documents to work through -------------------------
        if sources is None:
            if is_archive:
                try:
                    sources = archive.extract_pdfs(upload_path, work)
                except ArchiveError as exc:
                    job.update(status="error", errorCode=exc.code,
                               errorMessage=str(exc), finishedAt=time.time())
                    return
            else:
                sources = [(job["filename"], upload_path)]

        job["documentCount"] = len(sources)
        job["documents"] = [
            {"index": i, "filename": name, "status": "pending"}
            for i, (name, _) in enumerate(sources)
        ]

        # ---- one document at a time; pages within each run in parallel ----
        for index, (name, src) in enumerate(sources):
            if job_id not in JOBS:                    # cancelled mid-run
                return
            doc = job["documents"][index]
            doc["status"] = "running"
            job["currentDocument"] = index

            def on_progress(done, total, note, _d=doc):
                _d.update(pagesDone=done, pageCount=total, note=note)
                job["elapsedSeconds"] = round(time.time() - started, 1)

            out = os.path.join(work, f"out_{index:03d}.pdf")
            ocr_source = src
            try:
                before = len((read_text(src) or "").strip())

                # ---- optional cleaning stage (runs BEFORE OCR) ------------
                if job.get("clean"):
                    job["stage"] = "cleaning"
                    clean_result = image_clean.clean_document(
                        src, work, index=index, progress=on_progress,
                        user_mode=job.get("cleanMode") or None,
                    )
                    # The uploaded original is never written to: cleaning
                    # renders, cleans and rebuilds into a brand-new PDF, which
                    # then feeds the OCR stage as its page source.
                    ocr_source = clean_result["pdf_path"]
                    doc["cleaning"] = {
                        "status": "done",
                        "mode": clean_result["mode"],
                        "dpi": clean_result["dpi"],
                        "average": clean_result["average"],
                        "pageCount": len(clean_result["pages"]),
                        "previewPages": clean_result["preview_pages"],
                    }
                    # Full per-page metrics stay internal (small summary goes
                    # to clients through public()).
                    doc["cleaningPages"] = clean_result["pages"]

                # ---- OCR stage on the (possibly cleaned) pages -------------
                job["stage"] = "ocr"
                used = make_searchable(
                    ocr_source, out, quiet=True, progress=on_progress,
                    jobs=config.OCR_PAGE_CONCURRENCY,
                    arabic_variants=False,
                )
                direct_page_texts = used.get("page_texts") or {}
                direct_page_variants = used.get("page_text_variants") or {}
                if direct_page_texts:
                    pages_from_ocr = [
                        {
                            "page": number,
                            "text": text_context.repair_extracted_text(
                                str(direct_page_texts.get(number) or "")
                            ),
                            "textVariants": [
                                {
                                    "psm": variant.get("psm"),
                                    "text": text_context.repair_extracted_text(
                                        str(variant.get("text") or "")
                                    ),
                                }
                                for variant in direct_page_variants.get(number, [])
                                if isinstance(variant, dict) and str(variant.get("text") or "").strip()
                            ],
                        }
                        for number in range(1, int(doc.get("pageCount") or len(direct_page_texts)) + 1)
                    ]
                    text = "\n\n".join(page["text"] for page in pages_from_ocr)
                else:
                    pages_from_ocr = []
                    text = read_text(out)
                doc.update(status="done", path=out, text=text,
                           charsBefore=before, charsAfter=len(text.strip()),
                           bytesIn=os.path.getsize(src),
                           bytesOut=os.path.getsize(out),
                           pageCount=doc.get("pageCount", 0),
                           searchable=len(text.strip()) > before + 20,
                           detected=used.get("summary", ""),
                           language=used.get("lang"),
                           pageLanguages=used.get("page_languages", {}))
            except DependencyError:
                raise
            except BaseException as exc:                       # noqa: BLE001
                code, msg = classify(exc)
                doc.update(status="error", error={"code": code, "message": msg})

            job["documentsDone"] = index + 1

        # ---- one PDF, or a zip of them ------------------------------------
        good = [d for d in job["documents"] if d["status"] == "done"]
        if not good:
            first = job["documents"][0].get("error", {})
            job.update(status="error",
                       errorCode=first.get("code", "processing_failed"),
                       errorMessage=first.get("message", "No document could be read."),
                       finishedAt=time.time())
            return

        if is_archive:
            bundle = os.path.join(work, "searchable.zip")
            with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as zf:
                for d in good:
                    stem = os.path.splitext(d["filename"])[0]
                    zf.write(d["path"], f"{stem}-searchable.pdf")
            job["path"] = bundle
        else:
            job["path"] = good[0]["path"]

        job.update(status="done",
                   elapsedSeconds=round(time.time() - started, 1),
                   charsAfter=sum(d.get("charsAfter", 0) for d in good),
                   searchable=any(d.get("searchable") for d in good),
                   finishedAt=time.time())

    except DependencyError as exc:
        print(f"\n{exc}\n", flush=True)
        job.update(status="error", errorCode="dependency_missing",
                   errorMessage=str(exc), finishedAt=time.time())
    except BaseException as exc:                               # noqa: BLE001
        # BaseException on purpose: a SystemExit raised in here would otherwise
        # kill the thread silently and strand the job at "running" forever.
        code, msg = classify(exc)
        job.update(status="error", errorCode=code, errorMessage=msg,
                   finishedAt=time.time())
    finally:
        paths = cleanup_paths or ([upload_path] if upload_path else [])
        for path in paths:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass
        if job.get("status") in ("running", "queued"):
            job.update(status="error", errorCode="internal_error",
                       errorMessage="Processing stopped unexpectedly.",
                       finishedAt=time.time())


def reap():
    """Delete finished jobs after the TTL — and only finished ones.

    TTL runs from completion so a long ZIP never loses its working directory
    while its own thread is still writing into it; the absolute ceiling keeps
    a wedged job collectable eventually.
    """
    while True:
        time.sleep(60)
        now = time.time()
        with LOCK:
            for jid, job in list(JOBS.items()):
                finished = job.get("finishedAt")
                expired = finished and now - finished > TTL
                too_old = now - job["createdAt"] > MAX_AGE
                if expired or too_old:
                    shutil.rmtree(job["work"], ignore_errors=True)
                    JOBS.pop(jid, None)


threading.Thread(target=reap, daemon=True).start()


def public(job, include_text=False):
    """Only what the caller needs — never internal paths."""
    docs = []
    for d in job.get("documents", []):
        item = {k: d.get(k) for k in
                ("index", "filename", "status", "pagesDone", "pageCount",
                 "charsAfter", "bytesIn", "bytesOut", "searchable",
                 "detected", "language", "note")}
        if d.get("error"):
            item["error"] = d["error"]
        if d.get("cleaning"):
            item["cleaning"] = d["cleaning"]
        if include_text and d.get("text"):
            item["text"] = d["text"]
        docs.append(item)

    out = {
        "id": job["id"],
        "status": job["status"],
        "filename": job["filename"],
        "kind": "zip" if job["isArchive"] else "pdf",
        "documentCount": job.get("documentCount", 0),
        "documentsDone": job.get("documentsDone", 0),
        "currentDocument": job.get("currentDocument", 0),
        "elapsedSeconds": job.get("elapsedSeconds", 0),
        "stage": job.get("stage", "ocr"),
        "clean": bool(job.get("clean")),
        "cleanMode": job.get("cleanMode", "auto"),
        "documents": docs,
    }
    if job["status"] == "done":
        out["searchable"] = job.get("searchable", False)
        out["charsAfter"] = job.get("charsAfter", 0)
        out["resultKind"] = "zip" if job["isArchive"] else "pdf"
    if job["status"] == "error":
        out["error"] = {"code": job.get("errorCode", "processing_failed"),
                        "message": job.get("errorMessage", "")}
    return out


@app.after_request
def disable_browser_cache(response):
    """Prevent stale job JSON from masking code updates."""
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    response.headers["X-OCR-Extract-Version"] = config.APP_VERSION
    return response


# --------------------------------------------------------------------------- #
# routes
# --------------------------------------------------------------------------- #
@app.get("/api/health")
def health():
    languages = sorted(autodetect.installed_languages()) if DEPS["ok"] else []
    return jsonify({
        "ok": DEPS["ok"],
        "version": config.APP_VERSION,
        "languages": languages,
        "languageHelp": deps.language_help(languages) if DEPS["ok"] else None,
        "maxUploadMb": MAX_MB,
        "maxBatchUploadMb": MAX_BATCH_MB,
        "maxFilesPerBatch": archive.MAX_FILES,
        "maxFilesPerZip": archive.MAX_FILES,
        "jobTtlSeconds": TTL,
        "setupHelp": None if DEPS["ok"] else deps.explain(DEPS),
        "ocr": "available" if DEPS["ok"] else "unavailable",
        "documentWorkers": QUEUE.capacity(),
        "documentsActive": QUEUE.active_count(),
        "cleaning": "available" if config.CLEAN_ENABLED else "disabled",
        "cleanDpi": config.CLEAN_DPI,
        "cleanOutput": config.CLEAN_OUTPUT,
        "cleanModes": list(image_clean.MODES) + ["auto"],
        "cleanPreviewPagesMax": config.CLEAN_PREVIEW_PAGES_MAX,
    })


@app.get("/")
def bench():
    """Minimal browser test bench for this API (OCR + cleaning)."""
    page = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tester.html")
    if not os.path.exists(page):
        return fail("not_found", "tester.html is not next to api.py.", 404)
    return send_file(page, mimetype="text/html")


@app.post("/api/jobs")
def create_job():
    if not DEPS["ok"]:
        return fail("dependency_missing", deps.explain(DEPS), 503)

    uploads = [upload for upload in request.files.getlist("file")
               if upload and upload.filename]
    if not uploads:
        return fail("no_file",
                    "Attach a PDF, a ZIP, or multiple PDFs as the 'file' field.",
                    400)
    if len(uploads) > archive.MAX_FILES:
        return fail("too_many_files",
                    f"At most {archive.MAX_FILES} PDFs can be uploaded at once.", 400)

    parsed = []
    taken = set()
    for upload in uploads:
        raw_name = os.path.basename((upload.filename or "").replace("\\", "/")).strip()
        if not raw_name:
            raw_name = "document.pdf"
        lower = raw_name.lower()
        if lower.endswith(".pdf"):
            file_is_zip = False
        elif lower.endswith(".zip"):
            file_is_zip = True
        else:
            return fail("unsupported_type",
                        "Upload PDFs, or one ZIP containing PDFs.", 415)

        if len(uploads) > 1:
            name = archive._safe_name(raw_name, taken)
        else:
            name = secure_filename(raw_name) or ""
            if not name.lower().endswith((".pdf", ".zip")):
                name = archive._safe_name(raw_name, taken)
            else:
                taken.add(name.lower())
        parsed.append((upload, name, file_is_zip))

    if len(parsed) > 1 and any(is_zip for _, _, is_zip in parsed):
        return fail("mixed_uploads",
                    "Upload either one ZIP or multiple PDFs, not a mixture.", 400)

    is_archive = len(parsed) > 1 or parsed[0][2]
    filename = parsed[0][1] if len(parsed) == 1 else "folder"

    # ---- optional cleaning mode -------------------------------------------
    clean = form_flag("clean")
    clean_mode = (request.form.get("cleanMode") or "auto").strip().lower()
    if clean:
        if not config.CLEAN_ENABLED:
            return fail("cleaning_disabled",
                        "Cleaning is disabled by configuration.", 409)
        if clean_mode not in ("auto", "gray", "color", "binary"):
            return fail("invalid_clean_mode",
                        "cleanMode must be auto, gray, color or binary.", 400)
    else:
        clean_mode = "auto"

    limit_mb = MAX_MB if len(parsed) == 1 else MAX_BATCH_MB
    declared_bytes = sum(int(upload.content_length or 0) for upload, _, _ in parsed)
    if declared_bytes > limit_mb * 1024 * 1024:
        return fail("file_too_large",
                    f"Maximum upload size is {limit_mb} MB.", 413)

    job_id = uuid.uuid4().hex
    work = os.path.join(WORK_DIR, job_id)
    try:
        os.makedirs(work, exist_ok=True)
        upload_paths = []
        sources = []
        for index, (upload, name, file_is_zip) in enumerate(parsed):
            upload_path = os.path.join(
                work, f"upload_{index:03d}" + (".zip" if file_is_zip else ".pdf"))
            upload.save(upload_path)
            upload_paths.append(upload_path)
            if not file_is_zip:
                sources.append((name, upload_path))

        total_bytes = sum(os.path.getsize(path) for path in upload_paths)
        if total_bytes > limit_mb * 1024 * 1024:
            shutil.rmtree(work, ignore_errors=True)
            return fail("file_too_large",
                        f"Maximum upload size is {limit_mb} MB.", 413)
    except BaseException:
        shutil.rmtree(work, ignore_errors=True)
        raise

    queued = QUEUE.would_queue()
    job = {"id": job_id, "status": "queued" if queued else "running",
           "filename": filename, "isArchive": is_archive,
           "work": work, "path": None, "createdAt": time.time(),
           "finishedAt": None, "documentCount": 0, "documentsDone": 0,
           "currentDocument": 0, "elapsedSeconds": 0, "documents": [],
           "clean": clean, "cleanMode": clean_mode,
           "stage": "cleaning" if clean else "ocr"}
    with LOCK:
        JOBS[job_id] = job

    source_list = sources or None
    upload_path = upload_paths[0] if len(parsed) == 1 else None
    QUEUE.submit(job_id, process, job_id, upload_path, is_archive, work,
                 source_list, upload_paths)
    return jsonify(public(job)), 202


@app.get("/api/jobs/<job_id>")
def get_job(job_id):
    job = JOBS.get(job_id)
    if not job:
        return fail("job_not_found", "Unknown or expired job.", 404)
    return jsonify(public(job, include_text=request.args.get("text") == "1"))


@app.get("/api/jobs/<job_id>/text")
def get_text(job_id):
    job = JOBS.get(job_id)
    if not job:
        return fail("job_not_found", "Unknown or expired job.", 404)
    if job["status"] != "done":
        return fail("not_ready", f"Job is {job['status']}.", 409)
    return jsonify({
        "id": job_id,
        "documents": [{"index": d["index"], "filename": d["filename"],
                       "text": d.get("text", "")}
                      for d in job["documents"] if d["status"] == "done"],
    })


@app.get("/api/jobs/<job_id>/file")
def get_file(job_id):
    """The single result PDF, or a ZIP of them for a multi-file batch.

    Cleaning runs before OCR, so this one file already contains the cleaned
    pages and the OCR text layer.
    """
    job = JOBS.get(job_id)
    if not job:
        return fail("job_not_found", "Unknown or expired job.", 404)
    if job["status"] != "done":
        return fail("not_ready", f"Job is {job['status']}.", 409)

    stem = os.path.splitext(job["filename"])[0]
    if job["isArchive"]:
        return send_file(job["path"], mimetype="application/zip",
                         as_attachment=request.args.get("download") == "1",
                         download_name=f"{stem}-searchable.zip")
    return send_file(job["path"], mimetype="application/pdf",
                     as_attachment=request.args.get("download") == "1",
                     download_name=f"{stem}-searchable.pdf")


@app.get("/api/jobs/<job_id>/cleaned/file")
def get_cleaned_file(job_id):
    """Alias of ``/file``: cleaning and OCR produce a single PDF.

    Cleaning runs before OCR, so a job that ran OCR already returns one PDF that
    is both cleaned and searchable — there is no second file to hand out. This
    route stays for clients written before the merge.
    """
    job = JOBS.get(job_id)
    if not job:
        return fail("job_not_found", "Unknown or expired job.", 404)
    if not job.get("clean"):
        return fail("clean_not_requested",
                    "This job was created without cleaning.", 409)
    if job["status"] != "done":
        return fail("not_ready", f"Job is {job['status']}.", 409)
    return get_file(job_id)


@app.get("/api/jobs/<job_id>/documents/<int:index>/preview")
def get_preview(job_id, index):
    """Original vs cleaned page thumbnail: ``?side=original|cleaned&page=N``."""
    job = JOBS.get(job_id)
    if not job:
        return fail("job_not_found", "Unknown or expired job.", 404)
    docs = job.get("documents", [])
    if index < 0 or index >= len(docs):
        return fail("document_not_found", "No document at that index.", 404)
    doc = docs[index]
    if doc["status"] != "done" or not doc.get("cleaning"):
        return fail("preview_unavailable",
                    "Cleaning previews are not available for this document.", 409)

    side = request.args.get("side", "cleaned")
    if side not in ("original", "cleaned"):
        return fail("invalid_side", "side must be original or cleaned.", 400)
    try:
        page = int(request.args.get("page", 1))
    except (TypeError, ValueError):
        return fail("invalid_page", "page must be an integer.", 400)

    if page not in doc["cleaning"].get("previewPages", []):
        return fail("preview_not_found",
                    "No preview stored for that page.", 404)
    path = os.path.join(job["work"], "previews", f"{index:03d}",
                        f"p{page:04d}_{side}.jpg")
    if not os.path.exists(path):
        return fail("preview_not_found", "Preview file is missing.", 404)
    return send_file(path, mimetype="image/jpeg")


@app.get("/api/jobs/<job_id>/documents/<int:index>/file")
def get_document(job_id, index):
    """One PDF out of a batch, for previewing without downloading the ZIP."""
    job = JOBS.get(job_id)
    if not job:
        return fail("job_not_found", "Unknown or expired job.", 404)
    docs = job.get("documents", [])
    if index < 0 or index >= len(docs):
        return fail("document_not_found", "No document at that index.", 404)
    doc = docs[index]
    if doc["status"] != "done":
        return fail("not_ready", f"Document is {doc['status']}.", 409)

    stem = os.path.splitext(doc["filename"])[0]
    return send_file(doc["path"], mimetype="application/pdf",
                     as_attachment=request.args.get("download") == "1",
                     download_name=f"{stem}-searchable.pdf")


@app.delete("/api/jobs/<job_id>")
def delete_job(job_id):
    with LOCK:
        job = JOBS.pop(job_id, None)
    if not job:
        return fail("job_not_found", "Unknown or expired job.", 404)
    # A job still in the queue notices it is gone before doing any work; one
    # already running notices between documents.
    shutil.rmtree(job["work"], ignore_errors=True)
    return "", 204


def _lan_addresses():
    """Local addresses other machines could reach, for the startup banner.

    Informational only: the server is already bound to every interface, this
    just names what the URL will look like from another machine on the network.
    """
    found = set()
    try:
        probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            # No packet is sent; this just asks the routing table which local
            # address would be used to leave the machine.
            probe.connect(("10.255.255.255", 1))
            found.add(probe.getsockname()[0])
        finally:
            probe.close()
    except OSError:
        pass
    return sorted(found)


if __name__ == "__main__":
    urls = [f"http://127.0.0.1:{config.PORT}"]
    if config.HOST not in ("127.0.0.1", "localhost"):
        urls += [f"http://{addr}:{config.PORT}" for addr in _lan_addresses()]
    print("OCR API on " + "   ".join(urls))
    print("CORS origins: " + (ORIGINS if ORIGINS == "*" else ", ".join(ORIGINS)))
    if DEPS["ok"]:
        print("Languages:", ", ".join(sorted(autodetect.installed_languages())))
    else:
        print("\nNOT READY:\n" + deps.explain(DEPS))

    app.run(host=config.HOST, port=config.PORT, debug=False, threaded=True)