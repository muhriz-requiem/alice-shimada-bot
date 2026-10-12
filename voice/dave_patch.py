"""
voice/dave_patch.py
====================
Monkey-patches for discord-ext-voice-recv to support DAVE (E2EE) decryption
on incoming audio and to make the PacketRouter resilient to corrupt packets.

Applied at import time from voice/__init__.py. Survives pip upgrades because
we wrap methods rather than editing site-packages.
"""

from __future__ import annotations

from bot.logging_setup import get_logger

log = get_logger("voice.dave_patch")

_applied = False


def apply() -> None:
    """Apply all patches. Safe to call multiple times."""
    global _applied
    if _applied:
        return
    _applied = True

    _patch_packet_router()
    _patch_dave_decrypt()
    _patch_reader_callback()
    log.info("DAVE patches applied")


def _patch_packet_router() -> None:
    """Make PacketRouter catch per-packet errors instead of dying."""
    from discord.ext.voice_recv.router import PacketRouter
    from discord.ext.voice_recv.utils import MultiDataEvent  # noqa: F401 (ensures import order)

    if getattr(PacketRouter, "_alice_patched", False):
        return

    import threading

    original_do_run = PacketRouter._do_run

    def patched_do_run(self) -> None:
        errors_logged = 0
        while not self._end_thread.is_set():
            self.waiter.wait()
            with self._lock:
                for decoder in list(self.waiter.items):
                    try:
                        data = decoder.pop_data()
                        if data is not None:
                            self.sink.write(data.source, data)
                    except Exception as e:
                        errors_logged += 1
                        if errors_logged <= 5:
                            log.info(
                                "PacketRouter skip #%d (ssrc=%s): %s: %s",
                                errors_logged, decoder.ssrc, type(e).__name__, e,
                            )
                        elif errors_logged % 100 == 0:
                            log.info("PacketRouter skip total=%d", errors_logged)
                        continue

    PacketRouter._do_run = patched_do_run
    PacketRouter._alice_patched = True  # type: ignore[attr-defined]
    log.info("PacketRouter._do_run patched (per-packet error resilience)")


def _patch_dave_decrypt() -> None:
    """
    Strip the DAVE E2EE layer from incoming opus payloads before decoding.

    discord-ext-voice-recv only handles the RTP encryption layer (xsalsa20 /
    aead_xchacha20).  When Discord negotiates DAVE, the opus payload inside
    the RTP packet is additionally DAVE-encrypted, so the opus decoder sees
    garbage and raises OpusError: corrupted stream.

    We wrap PacketDecoder._decode_packet to call
    dave_session.decrypt(user_id, MediaType.audio, payload) before handing
    the payload to libopus.
    """
    from discord.ext.voice_recv.opus import PacketDecoder

    if getattr(PacketDecoder, "_alice_patched", False):
        return

    original_decode_packet = PacketDecoder._decode_packet
    _stats = {"attempted": 0, "success": 0, "skipped": 0, "failed": 0}

    def patched_decode_packet(self, packet):
        # Only touch real packets (not FakePacket / FEC placeholders).
        if packet is not None and getattr(packet, "decrypted_data", None):
            _try_dave_decrypt(self, packet, _stats)
        return original_decode_packet(self, packet)

    PacketDecoder._decode_packet = patched_decode_packet  # type: ignore[method-assign]
    PacketDecoder._alice_patched = True  # type: ignore[attr-defined]
    log.info("PacketDecoder._decode_packet patched (DAVE decrypt)")


def _try_dave_decrypt(decoder, packet, stats: dict) -> None:
    """Best-effort DAVE decryption of packet.decrypted_data in place."""
    import davey

    stats["attempted"] += 1
    n = stats["attempted"]

    try:
        vc = decoder.sink.voice_client
        if vc is None:
            stats["skipped"] += 1
            if n <= 3:
                log.info("DAVE decrypt skip #%d: no voice_client", n)
            return

        conn = getattr(vc, "_connection", None)
        dave_session = getattr(conn, "dave_session", None)
        if dave_session is None:
            stats["skipped"] += 1
            if n <= 3:
                log.info("DAVE decrypt skip #%d: no dave_session", n)
            return
        if not dave_session.ready:
            stats["skipped"] += 1
            if n <= 3:
                log.info(
                    "DAVE decrypt skip #%d: session not ready (status=%s)",
                    n, getattr(dave_session, "status", "?"),
                )
            return

        user_id = vc._get_id_from_ssrc(decoder.ssrc)
        if user_id is None:
            stats["skipped"] += 1
            if n <= 3:
                log.info("DAVE decrypt skip #%d: ssrc=%s not mapped to user", n, decoder.ssrc)
            return

        # Passthrough users send plain opus — don't try to DAVE-decrypt them
        can_passthrough = getattr(dave_session, "can_passthrough", None)
        if callable(can_passthrough):
            try:
                if can_passthrough(user_id):
                    stats["skipped"] += 1
                    if n <= 3:
                        log.info("DAVE decrypt skip #%d: user=%s passthrough (plain opus)", n, user_id)
                    return
            except Exception:
                pass  # if API differs, fall through to decrypt attempt

        packet.decrypted_data = dave_session.decrypt(
            user_id, davey.MediaType.audio, packet.decrypted_data
        )
        stats["success"] += 1
        if n <= 3 or n % 500 == 0:
            log.info(
                "DAVE decrypt OK #%d (user=%s, %d bytes -> %d bytes)",
                n, user_id,
                len(packet.decrypted_data) if packet.decrypted_data else 0,
                len(packet.decrypted_data or b""),
            )

    except Exception as e:
        stats["failed"] += 1
        if stats["failed"] <= 5:
            log.info(
                "DAVE decrypt FAIL #%d (attempted=%d): %s: %s",
                stats["failed"], n, type(e).__name__, e,
            )
        # Leave decrypted_data untouched; opus decode may fail but the
        # PacketRouter patch will skip the packet instead of crashing.


def _patch_reader_callback() -> None:
    """Log first N raw packets so we can tell 'no packets' from 'bad decode'."""
    from discord.ext.voice_recv.reader import AudioReader

    if getattr(AudioReader, "_alice_patched", False):
        return

    original_callback = AudioReader.callback
    _n = {"count": 0, "rtp": 0, "rtcp": 0, "err": 0}

    def patched_callback(self, packet_data: bytes) -> None:
        _n["count"] += 1
        n = _n["count"]
        if n <= 5 or n % 500 == 0:
            log.info(
                "AudioReader.packet #%d (len=%d, total=%d)",
                n, len(packet_data), n,
            )
        return original_callback(self, packet_data)

    AudioReader.callback = patched_callback  # type: ignore[method-assign]
    AudioReader._alice_patched = True  # type: ignore[attr-defined]
    log.info("AudioReader.callback patched (receive tracing)")
