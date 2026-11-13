# desktop/convert — reproducible VRM build

Ports the SFW model patch pipeline from `D:\Temp\opencode\` into the repo so
the deployed avatar (`Alice.vrm`) can be rebuilt from the untouched source
(`Alice_original.vrm`).

## Files

- `patch_vrm.py` — offline GLB patcher (skin weights, alpha blend, eye
  compositing). Parameterized via env vars (see below).
- `check_patched.py` — verification of the patched VRM.
- `build_vrm.py` — entry point: runs patcher then checker.

## Usage

```bash
# defaults point at the existing D:\Temp\opencode\ inputs
python desktop/convert/build_vrm.py
```

Override inputs/outputs via env vars:

```bash
set MATE_VRM_SRC=D:\path\to\Alice_original.vrm
set MATE_VRM_DST=D:\path\to\Alice_patched.vrm
set MATE_VRM_IMGS=D:\path\to\imgs
python desktop/convert/build_vrm.py
```

## Deploy

After a successful build, deploy to MateEngine:

1. Stop MateEngineX.
2. Copy `Alice_patched.vrm` -> `Alice.vrm` (in `D:\Temp\opencode\`).
3. Start MateEngineX, wait ~32 s, run `takeover_face.py`.

## Notes

- `Alice_original.vrm` (29 MB) and the `imgs/` texture dir are external inputs,
  not stored in the repo. Keep them in `D:\Temp\opencode\` (or point the env
  vars elsewhere).
- The patcher edits skin weights directly in the GLB; no Blender round-trip.
