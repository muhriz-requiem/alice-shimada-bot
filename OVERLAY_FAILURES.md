# Overlay Failure Postmortem

> **Status: historical record (superseded).** The in-app overlay route
> documented below was retired in favor of the external **MateEngine**
> renderer (`desktop/mate_bridge.py` + `desktop/engine_manager.py`). The
> retired in-house overlay/web attempts (three.js window, Qt WebEngine,
> pywebview/WebView2) were deleted in the Nov 2026 cleanup along with the
> Blender/mmd_tools PMX pipeline; the original `Shimada Arisu.pmx` source
> is preserved at `assets/models/Garupan Models/`. Current avatar docs
> live in `README.md` and `desktop/README.md`. Keep this file for the
> "why it failed" record only — do not restart the PMX → VRM route.

**Date:** 2026-11-02  
**Scope:** Alice desktop companion VRM avatar overlay (`desktop/overlay_qt.py`)

## Goal
Build a transparent, always-on-top VRM avatar overlay showing Alice (from `D:\Garupan Models\Garupan Models\Alice Shimada\Shimada Arisu v1.0\Shimada Arisu.pmx`) with lip-sync, blinking, and idle movement — see-through over games like CS2.

## What was attempted

1. **PMX → VRM conversion**
   - Downloaded `PMX2VRMConverterEXE` (GitHub: hobosore/PMX2VRMConverterEXE)
   - Extracted `Shimada Arisu.pmx` (3.19 MB, MMD format with 20 PNG textures)
   - Converted to `D:\AliceShimadaBot\.vrm` (39.4 MB)

2. **VRM overlay implementation** (`desktop/overlay_qt.py`)
   - pywebview (Edge/Chromium backend) + three.js + `@pixiv/three-vrm@3.5.5`
   - Local HTTP server to serve VRM + JS assets (Edge WebView2 blocks CDN + `file://`)
   - Win32 color-key transparency (`WS_EX_LAYERED` + `SetLayeredWindowAttributes`)
   - Click-through (`WS_EX_TRANSPARENT`)
   - Lip-sync via VRM blend shapes, blinking, idle head movement

3. **PNG fallback overlay** (working)
   - Shows `preview.png` with RMS-driven mouth animation
   - Kept as fallback after VRM approach failed

## Failures

### F1: PMX→VRM conversion dropped blend shapes
**Symptom:** `expressionManager.expressions` returns empty array; `getExpression('aa'/'oh'/'blink')` returns null.  
**Impact:** No mouth expressions → no lip-sync. No blink expressions → no blinking.  
**Root cause:** PMX2VRMConverterEXE did not preserve morph targets / blend shapes from the PMX model. The converted `.vrm` contains only the mesh, bones, and materials — no expression data.

### F2: Material rendering glitches
**Symptom:** Alice's dress straps (supposed to be black) rendered white/glitching. Model appeared vertically stretched.  
**Impact:** Unacceptable visual quality for an avatar overlay.  
**Root cause:** PMX2VRMConverterEXE material conversion issues — MToon material parameters, texture mappings, and/or scale not correctly translated from PMX to VRM.

### F3: Edge WebView2 transparency unreliable
**Symptom:** Overlay showed as a white rectangle instead of transparent. Win32 color-key (`WS_EX_LAYERED` + `SetLayeredWindowAttributes` on form BackColor) applied but did not produce true transparency in Edge WebView2.  
**Impact:** Overlay obstructs the screen instead of floating transparently over games.  
**Root cause:** Edge WebView2 (Chromium) does not reliably composite WebView2 alpha over a Win32 color-keyed layered window. The WebView2 control paints an opaque surface behind the Chromium content, defeating the color key. This is the same failure mode documented in `desktop/README.md` under "Approaches That Failed" (pywebview + WebView2 transparent mode).

### F4: ES module import issues in Edge WebView2
**Symptom:** `importmap` not supported; bare module specifiers (`from "three"`) failed; relative imports (`from "./three.module.js"`) required per-file path fixes.  
**Impact:** Development friction; three-vrm v3.x internally imports `three` as a bare specifier, requiring post-processing of the library file.  
**Root cause:** Edge WebView2 version on this system lacks full ES module importmap support.

### F5: pywebview `html=` vs `url=` module execution
**Symptom:** Passing HTML via `html=` parameter caused the `<script type="module">` block to not execute (no status output, no errors surfaced to Python). Switching to `url=` served via local HTTP server resolved module loading.  
**Impact:** Added complexity (local HTTP server, file serving, query parameter passing).  
**Root cause:** pywebview/Edge WebView2 handles `html=`-injected modules differently than file-served modules.

## What was deleted/cleaned up

| Path | Action |
|------|--------|
| `D:\AliceShimadaBot\.vrm` | Deleted (faulty 39.4 MB VRM) |
| `desktop/static/` | Deleted (three.js, three-vrm, GLTFLoader, overlay.html) |
| `desktop/utils/` | Deleted (BufferGeometryUtils.js) |
| `desktop/http_server.py` | Deleted (local HTTP server for VRM assets) |

## What remains (working)

| Component | Status |
|-----------|--------|
| `desktop/overlay_qt.py` | PNG avatar overlay — working (lip-sync via RMS, transparent background) |
| `desktop/app.py` | Tray app — working (pystray on background thread, webview on main thread) |
| `desktop/cli.py` | Headless CLI — working |
| `desktop/voice_app.py` | Shared companion engine — working |
| `desktop/local_client.py` | Local voice client — working |
| `desktop/mic_input.py` | Mic capture — working |
| `desktop/web_viewer.py` | pywebview static viewer — present but not used by overlay |

## Recommendation

**Do not retry the PMX→VRM path** without a properly authored VRM source. The PMX2VRMConverterEXE tool does not preserve blend shapes or materials correctly for this model.

**For a 3D animated Alice overlay in the future:**
1. Obtain a native VRM file (e.g., from VRoid Studio or a properly converted source) that includes:
   - Blend shapes / morph targets (for mouth, blink)
   - MToon materials with correct texture maps
   - Correct scale and orientation (faces -Z, upright)
2. Revisit `desktop/overlay_qt.py` with three.js + three-vrm
3. Address Edge WebView2 transparency (possibly via Win32 layered window color key on the form's actual BackColor, as documented in `desktop/tools/composition_bench.py`)

**Current working state:** PNG avatar overlay with RMS-driven lip-sync is functional and transparent.

## Files referencing overlay

- `desktop/app.py` — tray entry point, constructs `OverlayController`, calls `webview.start()`
- `desktop/voice_app.py` — lip-sync RMS callback calls `ov.set_mouth()`
- `desktop/README.md` — documents failed overlay approaches (Qt WebEngine, pywebview Edge, static HTML)
- `desktop/tools/` — overlay prototyping scripts (alpha_test, composition_bench, probe_live_overlay, verify_live_overlay, overlay_cycle_test)
