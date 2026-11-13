# -*- coding: utf-8 -*-
"""Clean, offset-safe final check of Alice_patched.vrm.

Prints per-prim top humanoid/non-hu joint usage (names escaped), plus
overall non-humanoid weight ratio. Uses the exact GLB chunk layout.
"""
import json
import struct

import numpy as np

import os as _os
PATH = _os.environ.get("MATE_VRM_CHECK", r"D:\Temp\opencode\Alice_patched.vrm")


def esc(s):
    return (s or "?").encode("unicode_escape").decode("ascii")


def load(path):
    with open(path, "rb") as f:
        assert f.read(4) == b"glTF"
        f.read(8)
        jlen, jtype = struct.unpack("<II", f.read(8))
        assert jtype == 0x4E4F534A
        gltf = json.loads(f.read(jlen).decode("utf-8"))
        # NOTE: NO stray reads here. Bin chunk header follows immediately.
        blen, btype = struct.unpack("<II", f.read(8))
        assert btype == 0x004E4942
        blob = f.read(blen)
    return gltf, blob


NCOMP = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}
CS = {5120: 1, 5121: 1, 5122: 2, 5123: 2, 5125: 4, 5126: 4}
DTS = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16,
       5125: np.uint32, 5126: np.float32}


def read(gltf, blob, idx):
    a = gltf["accessors"][idx]
    bv = gltf["bufferViews"][a.get("bufferView", 0)]
    off = bv.get("byteOffset", 0) + a.get("byteOffset", 0)
    n = a["count"] * NCOMP[a["type"]] * CS[a["componentType"]]
    return np.frombuffer(blob[off:off + n],
                         dtype=DTS[a["componentType"]]).reshape(
        -1, NCOMP[a["type"]]), a


def main():
    gltf, blob = load(PATH)
    nodes = gltf["nodes"]
    skin = gltf["skins"][0]
    joints = skin["joints"]
    ext = (gltf.get("extensions") or {}).get("VRMC_vrm") or {}
    hb = (ext.get("humanoid") or {}).get("humanBones") or {}
    hm = {e["node"]: b for b, e in hb.items()
          if isinstance(e, dict) and isinstance(e.get("node"), int)}
    hu_nodes = set(hm)
    node_to_joint = {ni: ji for ji, ni in enumerate(joints)}
    hu_joints = set(node_to_joint[n] for n in hu_nodes if n in node_to_joint)

    tex_src = [t.get("source", -1) for t in gltf.get("textures", [])]
    mats = gltf.get("materials", [])

    bad = 0.0
    total = 0.0
    for pi, prim in enumerate(gltf["meshes"][0]["primitives"]):
        attrs = prim["attributes"]
        mat = (mats[prim["material"]].get("name", "?")
               if "material" in prim else "?")
        tex = ""
        if "material" in prim:
            m = mats[prim["material"]]
            bt = (m.get("pbrMetallicRoughness") or {}).get(
                "baseColorTexture")
            tex = ("<%s>" % tex_src[bt["index"]] if bt else "NO-TEX")
        if "JOINTS_0" not in attrs:
            print("prim %2d mat=%-26s tex=%-6s NO-SKIN" % (pi, esc(mat), tex))
            continue
        J, _ = read(gltf, blob, attrs["JOINTS_0"])
        W, wacc = read(gltf, blob, attrs["WEIGHTS_0"])
        cnt = {}
        for v in range(J.shape[0]):
            for k in range(4):
                w = float(W[v, k])
                if w > 0.002:
                    j = int(J[v, k])
                    cnt[j] = cnt.get(j, 0.0) + w
                    if j not in hu_joints:
                        bad += w
                    total += w
        order = sorted(cnt.items(), key=lambda kv: -kv[1])
        names = []
        for j, w in order[:6]:
            ni = joints[j]
            names.append("%s(%.0f%s)" % (
                esc(nodes[ni].get("name", "?")), w,
                ",hu" if j in hu_joints else ",??"))
        print("prim %2d mat=%-26s tex=%-6s verts=%4d wsum=%.3f %s" % (
            pi, esc(mat), tex, J.shape[0], cnt_top_sum := sum(
                v2 for _, v2 in order[:6]), " ".join(names)))
    print("non-humanoid weight ratio: %.5f" % (bad / max(total, 1e-9)))


if __name__ == "__main__":
    main()
