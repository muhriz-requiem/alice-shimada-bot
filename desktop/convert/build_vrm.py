"""Reproducible VRM build entry point.

Runs the SFW patcher (patch_vrm.py) against the source VRM and verifies
the result with check_patched.py.

Inputs (override via env vars or CLI):
  MATE_VRM_SRC   source VRM (default D:\\Temp\\opencode\\Alice_original.vrm)
  MATE_VRM_DST   output VRM (default D:\\Temp\\opencode\\Alice_patched.vrm)
  MATE_VRM_IMGS  extracted texture dir (default D:\\Temp\\opencode\\imgs)
  MATE_VRM_CHECK path to verify (defaults to MATE_VRM_DST)

Deploy (after a successful build):
  1. Stop MateEngineX
  2. Copy Alice_patched.vrm -> Alice.vrm (in D:\\Temp\\opencode\\)
  3. Start MateEngineX, wait ~32 s, run takeover_face.py
"""
from __future__ import annotations

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PYTHON = sys.executable


def main() -> int:
    src = os.environ.get("MATE_VRM_SRC", r"D:\Temp\opencode\Alice_original.vrm")
    dst = os.environ.get("MATE_VRM_DST", r"D:\Temp\opencode\Alice_patched.vrm")
    imgs = os.environ.get("MATE_VRM_IMGS", r"D:\Temp\opencode\imgs")
    check = os.environ.get("MATE_VRM_CHECK", dst)

    if not os.path.isfile(src):
        print("source VRM not found:", src)
        return 1
    if not os.path.isdir(imgs):
        print("imgs dir not found:", imgs)
        return 1

    env = dict(os.environ)
    env["MATE_VRM_SRC"] = src
    env["MATE_VRM_DST"] = dst
    env["MATE_VRM_IMGS"] = imgs
    env["MATE_VRM_CHECK"] = check

    print("== patch_vrm.py ==")
    r = subprocess.run([PYTHON, os.path.join(HERE, "patch_vrm.py")], env=env)
    if r.returncode != 0:
        print("patch failed")
        return r.returncode

    print("== check_patched.py ==")
    r2 = subprocess.run([PYTHON, os.path.join(HERE, "check_patched.py")], env=env)
    if r2.returncode != 0:
        print("check failed")
        return r2.returncode

    print("build OK ->", dst)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
