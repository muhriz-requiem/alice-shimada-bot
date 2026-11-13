# -*- coding: utf-8 -*-
"""Mouth timeline builders.

Converts (phone list | kana text, audio seconds) into mouth events
[(preset_or_None, weight, seconds)] played via MateBridge.speak_timeline
(sticky engine-preset writes; None = closed-mouth slot).

Sync model (v2, replacing uniform per-mora timing):
- every phoneme gets a slot; vowels open a viseme and carry duration
  proportional to weight (stress-1 vowels longer), consonants are short
  closed slots so closures actually land between vowels, glides (W/Y)
  get their own shapes;
- punctuation tokens (-, . ? etc.) become pause slots; real durations
  are taken from measured energy dips in the synthesized audio when
  available (aligned by order of appearance), else from defaults;
- remaining audio time is distributed over non-pause slots
  proportionally to weight.

If the measured pauses are greedy (leftover budget < 20% of the audio)
they are shrunk so non-pause slots keep at least 20%.

Language coverage:
- English: ARPAbet phones from the phoneme worker (desktop/phonemes.py,
  GPT-SoVITS's own text backend, so the phones match what she speaks);
- Japanese: kana segmentation (small kana now take a small share of
  time instead of being skipped totally).
"""
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# ARPAbet (stress digit stripped) -> VRM viseme preset.
# Diphthongs snap to their onset shape; IY uses 'ee' so the tight vowel
# reads as distinct from 'ih'.
VOWEL_TO_VISEME: Dict[str, str] = {
    "AA": "aa", "AE": "aa", "AH": "aa",
    "AO": "oh", "AW": "aa", "AY": "aa",
    "EH": "ih", "ER": "ih", "EY": "ih",
    "IH": "ih", "IY": "ee",
    "OW": "oh", "OY": "oh",
    "UH": "ou", "UW": "ou",
}

# relative duration weight per consonant, before global scaling
CONSONANT_WEIGHT: Dict[str, float] = {
    "P": 0.35, "B": 0.35, "T": 0.30, "D": 0.30,
    "K": 0.30, "G": 0.30, "JH": 0.35, "CH": 0.35,
    "S": 0.55, "Z": 0.55, "F": 0.50, "V": 0.50,
    "TH": 0.50, "SH": 0.55, "ZH": 0.55, "DH": 0.55, "HH": 0.40,
    "W": 0.55, "Y": 0.55, "L": 0.55, "R": 0.55,
    "M": 0.50, "N": 0.50, "NG": 0.50,
}
# consonants acted with a visible mouth shape (not a closed slot)
GLIDE_VISEME: Dict[str, str] = {"W": "ou", "Y": "ih"}

DEFAULT_CONSONANT_WEIGHT = 0.45

PUNCTUATION_PAUSE: Dict[str, float] = {
    ",": 0.20, ".": 0.34, "!": 0.34, "?": 0.34, ";": 0.24, ":": 0.24,
    "\u3001": 0.20, "\u3002": 0.34, "\uff01": 0.34, "\uff1f": 0.34,
}
_STRESS_WEIGHT: Dict[int, float] = {0: 0.90, 1: 1.20, 2: 0.85}

# kind, preset, weight
Slot = Tuple[str, Optional[str], float]

Event = Tuple[Optional[str], float, float]


def _classify_phone(ph: str) -> Slot:
    """English phone -> slot descriptor (1 slot per phone, keeps timing).

    Punctuation slots carry their own token so pause lengths can be
    paired with measured dips by order of appearance.
    """
    if ph in PUNCTUATION_PAUSE:
        return ("pau", ph, 0.0)
    if len(ph) >= 2 and ph[-1].isdigit():
        base, stress = ph[:-1], int(ph[-1])
        if base in VOWEL_TO_VISEME:
            return ("vis", VOWEL_TO_VISEME[base], _STRESS_WEIGHT.get(stress, 1.0))
        if base in CONSONANT_WEIGHT or base in GLIDE_VISEME:
            return (("vis", GLIDE_VISEME[base], CONSONANT_WEIGHT[base])
                    if base in GLIDE_VISEME
                    else ("cns", None, CONSONANT_WEIGHT[base]))
        return ("cns", None, DEFAULT_CONSONANT_WEIGHT)
    if ph in GLIDE_VISEME:
        return ("vis", GLIDE_VISEME[ph], CONSONANT_WEIGHT[ph])
    if ph in CONSONANT_WEIGHT:
        return ("cns", None, CONSONANT_WEIGHT[ph])
    return ("cns", None, DEFAULT_CONSONANT_WEIGHT)


def _resolve_pauses(classified: Sequence[Slot],
                    pause_durations: Optional[Sequence[float]],
                    audio_seconds: float) -> List[float]:
    """Pause length per 'pau' slot, in order; defaults for missing."""
    pauses = list(pause_durations) if pause_durations else []
    out: List[float] = []
    i = 0
    for kind, token, _w in classified:
        if kind != "pau":
            continue
        d = 0.0
        if i < len(pauses):
            d = float(pauses[i])
        if d <= 0.0:
            d = PUNCTUATION_PAUSE.get(token, 0.20)
        out.append(max(0.06, min(d, 1.2)))
        i += 1
    total_pause = sum(out)
    if total_pause > 0.80 * audio_seconds:
        f = (0.80 * audio_seconds) / total_pause
        out = [d * f for d in out]
    return out


def _assemble(classified: Sequence[Slot],
              audio_seconds: float,
              pause_durations: Optional[Sequence[float]]) -> List[Event]:
    if not classified or audio_seconds <= 0.05:
        return []
    pause_secs = _resolve_pauses(classified, pause_durations, audio_seconds)
    total_pause = sum(pause_secs)
    budget = audio_seconds - total_pause
    body = sum(w for kind, _preset, w in classified if kind != "pau")
    if body <= 0:
        return []
    scale = budget / body

    events: List[Event] = []
    pi = 0
    for kind, preset, w in classified:
        if kind == "pau":
            events.append((None, 0.0, pause_secs[pi]))
            pi += 1
        elif kind == "vis":
            events.append((preset, min(1.0, max(0.5, w)), w * scale))
        else:
            events.append((None, 0.0, w * scale))
    return events


def build_timeline(phones: Sequence[str], audio_seconds: float,
                   pause_durations: Optional[Sequence[float]] = None,
) -> List[Event]:
    """English phone list -> mouth events over `audio_seconds`."""
    classified = [_classify_phone(ph) for ph in phones]
    return _assemble(classified, audio_seconds, pause_durations)


def _classify_kana(text: str) -> List[Slot]:
    """Kana/ASCII text -> classified slots (punct slots carry tokens)."""
    from mate_bridge import PHONEME_TO_VISEME, _KANA_SKIP

    classified: List[Slot] = []
    for ch in text:
        if ch in PUNCTUATION_PAUSE:
            classified.append(("pau", ch, 0.0))
        elif ch in _KANA_SKIP:
            classified.append(("cns", None, 0.25))  # small kana share time
        else:
            cp = ord(ch)
            if 0x3040 <= cp <= 0x309F or 0x30A0 <= cp <= 0x30FF:
                classified.append(("vis", PHONEME_TO_VISEME.get(ch, "aa"), 1.0))
            elif ch.isascii() and ch.isalpha() and \
                    ch.lower() in PHONEME_TO_VISEME:
                classified.append(("vis", PHONEME_TO_VISEME[ch.lower()], 1.0))
    return classified


def kana_timeline(text: str, audio_seconds: float,
                  pause_durations: Optional[Sequence[float]] = None,
) -> List[Event]:
    """Japanese (kana) text -> mouth events over `audio_seconds`."""
    classified = _classify_kana(text)
    return _assemble(classified, audio_seconds, pause_durations)


# ----------------------------------------------------------------- #
# Audio pause measurement                                            #
# ----------------------------------------------------------------- #

def find_pauses(audio: "np.ndarray", sr: int, min_ms: float = 90.0,
                max_ms: float = 1400.0) -> List[float]:
    """Durations (s) of interior low-energy runs, ordered by time.

    Skips 60 ms head / 200 ms tail margins so leading/trailing room tone
    is not reported as a sentence pause.
    """
    if audio is None or len(audio) < int(sr * 0.2):
        return []
    win = max(1, int(sr * 0.010))
    kernel = np.ones(win) / win
    env = np.convolve(np.abs(audio.astype("float32")), kernel, mode="same")
    peak = float(np.percentile(env, 98))
    if peak <= 0:
        return []
    quiet = env < 0.08 * peak

    head = int(sr * 0.060)
    tail = int(sr * 0.200)
    interior = quiet[head: len(quiet) - tail] if tail < len(quiet) else quiet
    min_len = int(sr * min_ms / 1000)
    max_len = int(sr * max_ms / 1000)

    durations: List[float] = []
    n = len(interior)
    i = 0
    while i < n:
        if interior[i]:
            j = i
            while j < n and interior[j]:
                j += 1
            if j - i >= min_len:
                durations.append(min(j - i, max_len) / sr)
            i = j
        else:
            i += 1
    return durations
