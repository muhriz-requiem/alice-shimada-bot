# -*- coding: utf-8 -*-
"""JSON-lines phoneme worker backed by GPT-SoVITS's own text backend.

Runs inside the GPT-SoVITS venv (its python has g2p_en + nltk + the
project's corrected cmudict) so the phonemes we lip-sync against are
exactly the ones the TTS speaks.

Protocol (stdin/stdout, one JSON object per line):
    request:  {"id": <int>, "text": <str>, "lang": "en"}
    response: {"id": <int>, "ok": true, "phones": [...], "norm": <str>}
              {"id": <int>, "ok": false, "err": <str>}

On startup, after the backend import, prints:
    {"event": "ready"}   or   {"event": "error", "err": ...}

All import chatter goes to stderr. Exits cleanly on stdin EOF.
"""
import json
import sys


def emit(obj) -> None:
    sys.stdout.write(json.dumps(obj, ensure_ascii=True))
    sys.stdout.write("\n")
    sys.stdout.flush()


def main() -> None:
    sys.path.insert(0, r"D:\GPT-SoVITS\GPT_SoVITS")

    try:
        from text import english
    except Exception as exc:
        emit({"event": "error", "err": repr(exc)})
        return
    emit({"event": "ready"})

    req = {}
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            try:
                req = json.loads(line)
            except ValueError:
                req = {}
            rid = req.get("id", 0)
            text = req.get("text", "")
            lang = req.get("lang", "en")
            if not text.strip():
                raise ValueError("empty text")
            if lang != "en":
                raise ValueError("unsupported lang: %r" % lang)
            # exact parity with the server's english chain:
            norm = english.text_normalize(text)
            phones = english.g2p(norm)
            resp = {"id": rid, "ok": True, "phones": phones, "norm": norm}
        except Exception as exc:
            rid = req.get("id", 0) if isinstance(req, dict) else -1
            resp = {"id": rid, "ok": False, "err": repr(exc)}
        emit(resp)
    # stdin EOF -> clean exit


if __name__ == "__main__":
    main()
