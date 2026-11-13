"""
desktop/sovits_patch/apply.py
=============================
Applies the "lazy BERT" patch to an external GPT-SoVITS install.

Why: GPT-SoVITS eagerly loads the Chinese BERT (chinese-roberta-wwm-ext-large,
~1.35 GB fp32 on CUDA) at server startup, but `TextPreprocessor.get_bert_inf`
computes *zero tensors* for every non-"zh" language. Alice only ever speaks
English/Japanese, so on a 4 GB GPU that 1.35 GB is pure dead weight -- VRAM
that instead goes to synthesis headroom, the avatar engine, and the OS.
With the patch, BERT (+tokenizer) loads on the first zh segment; en/ja
requests never pay for it. Controlled by GSV_LAZY_BERT (default on).

The patch is idempotent: applying twice is a no-op. Originals are backed
up next to the target once (`*.alice_orig.py`). `--check` exits 0 when the
patch is present (used by prewarm to warn after a GPT-SoVITS update),
`--revert` restores the backups.

Run:  venv\\Scripts\\python.exe desktop\\sovits_patch\\apply.py [--check|--revert]
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

MARKER = "[AliceShimadaBot lazy-bert patch v1]"

# (relative path, exact original snippet, patched snippet)
EDITS: list[tuple[str, str, str]] = [
    (
        "GPT_SoVITS/TTS_infer_pack/TTS.py",
        """    def _init_models(
        self,
    ):
        self.init_t2s_weights(self.configs.t2s_weights_path)
        self.init_vits_weights(self.configs.vits_weights_path)
        self.init_bert_weights(self.configs.bert_base_path)
        self.init_cnhuhbert_weights(self.configs.cnhuhbert_base_path)""",
        """    def _init_models(
        self,
    ):
        self.init_t2s_weights(self.configs.t2s_weights_path)
        self.init_vits_weights(self.configs.vits_weights_path)
        # [AliceShimadaBot lazy-bert patch v1] The Chinese BERT (~1.35 GB fp32
        # on CUDA) is only used for "zh" segments (get_bert_inf); en/ja get
        # zero tensors. Defer the load until a zh segment asks for it so 4 GB
        # GPUs keep the VRAM. GSV_LAZY_BERT=0 restores eager loading.
        import os as _os
        if _os.environ.get("GSV_LAZY_BERT", "1").strip().lower() in ("1", "true", "yes", "on"):
            print("[lazy-bert] deferring BERT load until first zh segment")
            self.bert_model = None
            self.bert_tokenizer = None
        else:
            self.init_bert_weights(self.configs.bert_base_path)
        self.init_cnhuhbert_weights(self.configs.cnhuhbert_base_path)""",
    ),
    (
        "GPT_SoVITS/TTS_infer_pack/TTS.py",
        """        self.text_preprocessor: TextPreprocessor = TextPreprocessor(
            self.bert_model, self.bert_tokenizer, self.configs.device
        )""",
        """        # [AliceShimadaBot lazy-bert patch v1] pass an on-demand loader so
        # the preprocessor can pull BERT in when a zh segment arrives.
        self.text_preprocessor: TextPreprocessor = TextPreprocessor(
            self.bert_model, self.bert_tokenizer, self.configs.device,
            loader=lambda: self.init_bert_weights(self.configs.bert_base_path),
        )""",
    ),
    (
        "GPT_SoVITS/TTS_infer_pack/TextPreprocessor.py",
        """    def __init__(self, bert_model: AutoModelForMaskedLM, tokenizer: AutoTokenizer, device: torch.device):
        self.bert_model = bert_model
        self.tokenizer = tokenizer
        self.device = device
        self.bert_lock = threading.RLock()""",
        """    def __init__(self, bert_model: AutoModelForMaskedLM, tokenizer: AutoTokenizer, device: torch.device, loader=None):
        self.bert_model = bert_model
        self.tokenizer = tokenizer
        self.device = device
        self.bert_lock = threading.RLock()
        # [AliceShimadaBot lazy-bert patch v1] optional deferred BERT loader
        self._lazy_loader = loader""",
    ),
    (
        "GPT_SoVITS/TTS_infer_pack/TextPreprocessor.py",
        """        language = language.replace("all_", "")
        if language == "zh":
            feature = self.get_bert_feature(norm_text, word2ph).to(self.device)""",
        """        language = language.replace("all_", "")
        if language == "zh":
            # [AliceShimadaBot lazy-bert patch v1] deferred BERT load for the
            # rare zh segment (called under bert_lock, so it is race-free).
            if self.bert_model is None and self._lazy_loader is not None:
                print("[lazy-bert] loading BERT for a zh segment")
                self._lazy_loader()
            feature = self.get_bert_feature(norm_text, word2ph).to(self.device)""",
    ),
]


def _target_dir() -> Path:
    import os

    return Path(os.environ.get("VOICE_SOVITS_SERVER_DIR", r"D:\GPT-SoVITS"))


def _edit_state(text: str, old: str, new: str) -> str:
    """'applied' | 'applicable' (unpatched but anchor present) | 'drifted'."""
    if MARKER in new and MARKER in text:
        # Marker alone is ambiguous across edits; the patched snippet is the
        # authority. (Falls back to marker because snippets contain it too.)
        if new in text:
            return "applied"
        return "drifted"
    if new in text:
        return "applied"
    if old in text:
        return "applicable"
    return "drifted"


def status() -> tuple[bool, list[str]]:
    """(patched, problems) — patched means every edit is present."""
    root = _target_dir()
    problems: list[str] = []
    applied = 0
    for rel, old, new in EDITS:
        path = root / rel
        if not path.exists():
            problems.append(f"missing file: {path}")
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        state = _edit_state(text, old, new)
        if state == "applied":
            applied += 1
        elif state == "drifted":
            problems.append(f"edit not applicable (source drifted?): {rel}")
    return applied == len(EDITS), problems


def apply() -> int:
    root = _target_dir()
    if not root.exists():
        print(f"GPT-SoVITS not found at {root}")
        return 2
    patched, problems = status()
    if patched:
        print("lazy-bert patch already present; nothing to do")
        return 0
    if problems:
        for p in problems:
            print(f"WARNING: {p}")
        print("Aborting: patch anchors do not match this GPT-SoVITS build.")
        return 3
    for rel, old, _new in EDITS:
        path = root / rel
        backup = path.with_suffix(".alice_orig.py")
        if not backup.exists():
            shutil.copy2(path, backup)
            print(f"backed up {path.name} -> {backup.name}")
    for rel, old, new in EDITS:
        path = root / rel
        text = path.read_text(encoding="utf-8")
        if text.count(old) != 1:
            print(f"ERROR: anchor not unique in {rel} ({text.count(old)} hits)")
            return 4
        path.write_text(text.replace(old, new), encoding="utf-8")
        print(f"patched {rel}")
    ok, problems = status()
    print("patch applied" if ok else f"verification FAILED: {problems}")
    return 0 if ok else 5


def revert() -> int:
    root = _target_dir()
    restored = 0
    for rel, _old, _new in EDITS:
        path = root / rel
        backup = path.with_suffix(".alice_orig.py")
        if backup.exists():
            shutil.copy2(backup, path)
            restored += 1
            print(f"restored {rel} from backup")
        else:
            print(f"no backup for {rel}")
    print(f"reverted {restored} file(s); restart the GPT-SoVITS server to take effect")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description="Lazy-BERT patch for GPT-SoVITS")
    ap.add_argument("--check", action="store_true", help="exit 0 if patched, 1 if not")
    ap.add_argument("--revert", action="store_true", help="restore the .alice_orig.py backups")
    args = ap.parse_args()
    if args.revert:
        sys.exit(revert())
    if args.check:
        patched, problems = status()
        for p in problems:
            print(f"WARNING: {p}")
        print("patched" if patched else "NOT patched")
        sys.exit(0 if patched else 1)
    sys.exit(apply())


if __name__ == "__main__":
    main()
