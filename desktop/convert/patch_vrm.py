"""Patch Alice.vrm offline (GLB rebuild; original untouched).

v7 fixes on top of the deployed v6 build's known defects:

Phase A  targeted joint redirect: unmapped duplicate/physics chains are
         pointed at their humanoid counterparts (JOINTS_0 indices only).
         NEW: "lower-body-root" joints (下半身 / Hip_L / Hip_R / Bip01) are
         NOT blanket-mapped on body/sock/pants surfaces; those slots get a
         height-based split into hips <-> upper-leg so thigh surfaces stop
         riding the pelvis bone (the hollow-leg tears).
Phase B  remaining non-humanoid weight -> nearest humanoid joint, but with
         ZONE RESTRICTIONS:
           - |x| < 0.13 m (torso column): arm-chain bones are excluded
             (fixes chest/strap verts snapping to the upper-arm bone).
           - y > 0.90 m: leg-chain bones are excluded.
Phase B2 copy-from-skin: sock / pants / skirt-waistband vertices receive the
         exact weights of the nearest body-skin vertex, so painted clothing
         layers move in lockstep with the body (straps stay on the torso).
Phase C  negligible remnants (<0.005) dropped; weights re-normalised.
Phase D  baseColor textures re-bound by name-prefix; PLUS:
           - MMD overlay sheets (SkinHi / Mayu / SkinAlpha / Pants003)
             switched to alphaMode BLEND (they are transparent-background
             art; opaque rendering turned the transparent zones dark).
           - Face008_EyeL/EyeR textures replaced with composited copies:
             the iris art pasted over a WHITE sclera canvas and embedded
             back into the GLB (fixes black eyeballs).

SRC : D:\\Temp\\opencode\\Alice_original.vrm (untouched original)
DST : D:\\Temp\\opencode\\Alice_patched.vrm
"""
import json
import os
import os as _os
import re
import struct

import numpy as np
from PIL import Image

SRC = _os.environ.get("MATE_VRM_SRC", r"D:\Temp\opencode\Alice_original.vrm")
DST = _os.environ.get("MATE_VRM_DST", r"D:\Temp\opencode\Alice_patched.vrm")
IMGD = _os.environ.get("MATE_VRM_IMGS", r"D:\Temp\opencode\imgs")

COMP = {5120: 1, 5121: 1, 5122: 2, 5123: 2, 5125: 4, 5126: 4}
NCOMP = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}
DTNP = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16,
        5125: np.uint32, 5126: np.float32}

PHASE_A = [
    ("足D", "足"),
    ("ひざD", "ひざ"),
    ("足首D", "足首"),
    ("足先EX", "つま先"),
    ("momotwist", "足"),
    ("momotwist2", "ひざ"),
    ("momoniku", "足"),
    ("Kata_L", "肩.L"), ("Kata_R", "肩.R"),
    ("Foretwist_L", "ひじ.L"), ("Foretwist_R", "ひじ.R"),
    ("Foretwist1_L", "ひじ.L"), ("Foretwist1_R", "ひじ.R"),
    ("Uppertwist_L", "腕.L"), ("Uppertwist_R", "腕.R"),
    ("Uppertwist1_L", "腕.L"), ("Uppertwist1_R", "腕.R"),
    ("腕捻", "腕"),
    ("胸操作", "上半身2"),
    ("AH1", "上半身2"), ("AH2", "上半身2"),
    ("Bip01 Spine0a", "上半身"),
    ("Bip01 Spine1a", "上半身2"),
    ("Hair_F", "頭"),
    ("Hair_R", "頭"),
]
# these four become CONTEXTUAL on split prims, exact センター elsewhere
PHASE_A_LOWWER_ROOT = ("下半身", "Hip_L", "Hip_R", "Bip01")
PHASE_A_WILDCARD = [
    ("_yure_skirt_", "センター"),
    ("Skirt", "センター"),
    ("_yure_hair", "頭"),
    ("_yure_soft", "頭"),
    ("BoneOnepRibbon", "頭"),
]
PHASE_A_PREFIX = [
    ("Bip01 L Toe", "つま先.L"),
    ("Bip01 R Toe", "つま先.R"),
    ("Bip01 L Foot", "足首.L"),
    ("Bip01 R Foot", "足首.R"),
]
#: prims whose lower-body-root slots get the hips<->upperLeg split
SPLIT_PRIMS = {0: "body skin", 13: "socks layer", 15: "pants"}
#: prims whose verts copy weights from the body skin; value = (lo, hi)
#: smooth transition band (copy starts / reaches full weight), so no
#: hard ring at a single y ("cut in half" fix); (0, 0) = full copy.
#: 17 shirt: gated additionally by |x|<=0.12 + skin arm-freeness so only
#: the chest/strap column is de-armed, sleeves keep arm-riding.
COPY_PRIMS = {13: (0.0, 0.0), 15: (0.0, 0.0), 17: (0.86, 0.90),
              18: (0.88, 0.93)}
#: материалов that must render as alpha overlays instead of opaque sheets
BLEND_MATERIALS = (
    "Face008_SkinHi (Instance)",
    "Face008_Mayu (Instance)",
    "Face008_SkinAlpha (Instance)",
    "Pants003",
)
#: Eye materials whose textures get a white-sclera composite
EYE_COMPOSITE = {
    "Face008_EyeL (Instance)": ("06", "eyeL_comp"),
    "Face008_EyeR (Instance)": ("07", "eyeR_comp"),
}


def esc(s):
    return (s or "?").encode("unicode_escape").decode("ascii")


def smoothstep(t):
    t = min(1.0, max(0.0, t))
    return t * t * (3.0 - 2.0 * t)


def acc_range(gltf, idx):
    acc = gltf["accessors"][idx]
    bv = gltf["bufferViews"][acc.get("bufferView", 0)]
    return (bv.get("byteOffset", 0) + acc.get("byteOffset", 0),
            acc["count"] * NCOMP[acc["type"]] * COMP[acc["componentType"]],
            bv.get("byteStride"))


def read_glb(path):
    with open(path, "rb") as f:
        assert f.read(4) == b"glTF", "bad magic"
        f.read(8)
        jlen, jtype = struct.unpack("<II", f.read(8))
        assert jtype == 0x4E4F534A
        gltf = json.loads(f.read(jlen).decode("utf-8"))
        blen, btype = struct.unpack("<II", f.read(8))
        assert btype == 0x004E4942
        blob = bytearray(f.read(blen))
    return gltf, blob


def write_glb(path, gltf, blob):
    js = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    if len(js) % 4:
        js += b" " * (4 - len(js) % 4)
    pad = b"\x00" * ((4 - len(blob)) % 4)
    total = 12 + 8 + len(js) + 8 + len(blob) + len(pad)
    with open(path, "wb") as f:
        f.write(b"glTF")
        f.write(struct.pack("<II", 2, total))
        f.write(struct.pack("<II", len(js), 0x4E4F534A))
        f.write(js)
        f.write(struct.pack("<II", len(blob) + len(pad), 0x004E4942))
        f.write(blob)
        f.write(pad)


def main():
    gltf, blob = read_glb(SRC)
    nodes = gltf["nodes"]
    mats = gltf.get("materials", [])
    skin = (gltf.get("skins") or [{}])[0]
    joints = skin.get("joints", [])

    name_to_node = {}
    for i, n in enumerate(nodes):
        name_to_node.setdefault(n.get("name", ""), i)
    node_to_joint = {ni: ji for ji, ni in enumerate(joints)}

    ext = (gltf.get("extensions") or {}).get("VRMC_vrm") or {}
    hb = (ext.get("humanoid") or {}).get("humanBones") or {}
    hm = {e["node"]: bone for bone, e in hb.items()
          if isinstance(e, dict) and isinstance(e.get("node"), int)}
    hu_nodes = set(hm)
    hu_joints = sorted(node_to_joint[n] for n in hu_nodes if n in node_to_joint)
    print("humanoid nodes=%d, humanoid joints=%d" % (len(hu_nodes), len(hu_joints)))

    def joint_of(name):
        ni = name_to_node.get(name)
        return node_to_joint.get(ni) if ni is not None else None

    def hb_joint_of(bone):
        e = hb.get(bone)
        if isinstance(e, dict) and isinstance(e.get("node"), int):
            return node_to_joint.get(e["node"])
        return None

    # ---- bind-pose joint world positions (IBM column-major) ----------------
    ibm_i = skin.get("inverseBindMatrices")
    joint_pos = {}
    if ibm_i is not None:
        i0, n0, _ = acc_range(gltf, ibm_i)
        ibm = np.frombuffer(blob[i0:i0 + n0], dtype=np.float32)
        ibm = ibm.reshape(-1, 4, 4)
        for ji in range(min(len(joints), ibm.shape[0])):
            w = np.linalg.inv(ibm[ji].T)
            joint_pos[ji] = w[:3, 3].copy()
    print("joint bind positions: %d" % len(joint_pos))

    j_hips = hb_joint_of("hips")
    j_spine = hb_joint_of("spine")
    j_chest = hb_joint_of("chest")
    j_upL = hb_joint_of("leftUpperLeg")
    j_upR = hb_joint_of("rightUpperLeg")
    for tag, j in (("hips", j_hips), ("spine", j_spine), ("chest", j_chest),
                   ("upL", j_upL), ("upR", j_upR)):
        print("  %-5s joint=%s bind=%s" % (
            tag, j, joint_pos.get(j, [0, 0, 0]).round(3) if j else None))

    # contextual blend zone for thigh split, derived from real bind heights
    y_up = (joint_pos[j_upL][1] + joint_pos[j_upR][1]) * 0.5
    top_mix = max(y_up + 0.03, y_up)
    # v11.1: hips share transitions over the WIDE band (v10 profile).
    # Tightening it to the crease made the gradient steeper AT the
    # flex-sensitive buttock ring (37mm self-tear regression). The thigh
    # junk was killed by the unlimited upper<->lower re-fraction instead.
    bot_mix = y_up - 0.12
    print("  thigh split zone: hips at y>=%.3f, leg at y<=%.3f"
          % (top_mix, bot_mix))

    # ---- v10 leg-field re-proportion -------------------------------------
    # adjacent skin verts carried chaotic hips<->leg fractions (e.g.
    # upLeg:0.86/hips:0.14 next to hips:0.68) -> interleaved shear under
    # sit / dance ("thighs glitching", knee hollows). Rebuild the leg
    # mix field as smooth height functions over the SAME pair masses so
    # nearby verts always end up with coherent fractions.
    j_loL = hb_joint_of("leftLowerLeg")
    j_loR = hb_joint_of("rightLowerLeg")
    j_fol = hb_joint_of("leftFoot")
    j_for = hb_joint_of("rightFoot")
    y_knee = (0.5 * (joint_pos[j_loL][1] + joint_pos[j_loR][1])
              if j_loL is not None and j_loR is not None else None)
    y_ankle = (0.5 * (joint_pos[j_fol][1] + joint_pos[j_for][1])
               if j_fol is not None and j_for is not None else None)
    knee_lo = (y_knee - 0.052) if y_knee is not None else None
    ankle_lo = (y_ankle - 0.040) if y_ankle is not None else None

    def frac_leg(y):
        # matches the contextual split: leg fraction grows downward
        return smoothstep((top_mix - y) / max(1e-6, top_mix - bot_mix))

    def frac_ul(y):
        # thigh above the knee, shin below
        return smoothstep((y - knee_lo) / 0.104)

    def frac_ll(y):
        # shin above the ankle, foot below
        return smoothstep((y - ankle_lo) / 0.080)

    def recon_legfield(newJ, newW, P):
        nfix = 0
        for v in range(newJ.shape[0]):
            y = float(P[v, 1])
            idx = {int(newJ[v, k]): k
                   for k in range(4) if newW[v, k] > 0.0}

            def getw(jj_):
                k = idx.get(jj_)
                return float(newW[v, k]) if k is not None else 0.0

            def setw(jj_, val):
                k = idx.get(jj_)
                if k is None:
                    if val <= 1e-6:
                        return
                    for k2 in range(4):
                        if newW[v, k2] <= 1e-9:
                            newJ[v, k2] = jj_
                            newW[v, k2] = val
                            idx[jj_] = k2
                            return
                    return
                newW[v, k] = val

            # hips <-> upper leg (pelvic window keeps its hips share)
            if j_hips_j is not None and y <= top_mix + 0.08:
                side = j_upL if float(P[v, 0]) >= 0.0 else j_upR
                wh, wu = getw(j_hips_j), getw(side)
                if wh > 1e-6 or wu > 1e-6:
                    m = wh + wu
                    f = frac_leg(y)
                    setw(j_hips_j, m * (1.0 - f))
                    setw(side, m * f)
                    nfix += 1
            # upper leg <-> lower leg (knee): apply wherever any of the
            # pair is present so junk outside the narrow band is cleaned
            # too (cuff verts with stray lowerLeg mass tear at deep flex)
            if knee_lo is not None:
                for up, lo in ((j_upL, j_loL), (j_upR, j_loR)):
                    wu, wl = getw(up), getw(lo)
                    if wu > 1e-6 or wl > 1e-6:
                        m = wu + wl
                        f = frac_ul(y)
                        setw(up, m * f)
                        setw(lo, m * (1.0 - f))
                        nfix += 1
            # lower leg <-> foot (ankle): same widow treatment
            if ankle_lo is not None:
                for lo, fo in ((j_loL, j_fol), (j_loR, j_for)):
                    wl, wf = getw(lo), getw(fo)
                    if wl > 1e-6 or wf > 1e-6:
                        m = wl + wf
                        f = frac_ll(y)
                        setw(lo, m * f)
                        setw(fo, m * (1.0 - f))
                        nfix += 1
        s = newW.sum(axis=1, keepdims=True)
        s[s < 1e-6] = 1.0
        newW /= s
        return nfix

    # specials = lower-body-root source joints
    special_joints = {}
    for nm in PHASE_A_LOWWER_ROOT:
        for s in (nm, nm + ".L", nm + ".R", "Hip_L", "Hip_R"):
            ji = joint_of(s)
            if ji is not None:
                special_joints[ji] = s
    print("lower-body-root joints: %s"
          % {esc(k): 1 for k in special_joints.values()})

    # ---- Phase B candidates with exclusions -------------------------------
    arm_bones = {"leftUpperArm", "leftLowerArm", "leftHand",
                 "rightUpperArm", "rightLowerArm", "rightHand"}
    leg_bones = {"leftUpperLeg", "leftLowerLeg", "leftFoot", "leftToes",
                 "rightUpperLeg", "rightLowerLeg", "rightFoot", "rightToes"}
    node_to_hbone = {}
    for k, e in hb.items():
        if isinstance(e, dict) and isinstance(e.get("node"), int):
            node_to_hbone[e["node"]] = k         # node id -> bone name
    arm_nodes = {n for n, b in node_to_hbone.items() if b in arm_bones}
    leg_nodes = {n for n, b in node_to_hbone.items() if b in leg_bones}
    arm_joints = {node_to_joint[n] for n in arm_nodes if n in node_to_joint}
    leg_joints = {node_to_joint[n] for n in leg_nodes if n in node_to_joint}
    # idle sway animates the shoulder BONES directly: torso-core strays
    # must not land on them (chest band shear / "cut in half")
    sh_joints = {node_to_joint[n] for n, b in node_to_hbone.items()
                 if b in ("leftShoulder", "rightShoulder")
                 and n in node_to_joint}

    exc_bones = {"leftEye", "rightEye"}
    for b in list(hb):
        if any(t in b for t in ("Index", "Middle", "Ring", "Little",
                                "Thumb")):
            exc_bones.add(b)
    hu_nodes_phaseb = set(hu_nodes) - {
        n for n in hu_nodes
        if any(hm[n].startswith(sb) or hm[n] == sb for sb in exc_bones)}
    hu_joints_phaseb = sorted(
        node_to_joint[n] for n in hu_nodes_phaseb if n in node_to_joint)
    print("phase B candidate joints=%d (excluded eyes+fingers)"
          % len(hu_joints_phaseb))

    # ---- Phase A: joint redirect map (WITHOUT the lower-body roots) --------
    jmap = {}
    for src, dst in PHASE_A:
        for s in (src, src + ".L", src + ".R"):
            if s in name_to_node:
                end = ("" if s == src else
                       ".L" if s.endswith(".L") else ".R")
                ji_src = node_to_joint.get(name_to_node[s])
                ji_dst = joint_of(dst if not end else dst + end)
                if ji_src is not None and ji_dst is not None:
                    jmap[ji_src] = ji_dst
    for frag, dst in PHASE_A_WILDCARD:
        for i, n in enumerate(nodes):
            if frag in n.get("name", ""):
                ji_src = node_to_joint.get(i)
                ji_dst = joint_of(dst)
                if ji_src is not None and ji_dst is not None:
                    jmap[ji_src] = ji_dst
    for pref, dst in PHASE_A_PREFIX:
        for i, n in enumerate(nodes):
            if (n.get("name", "") or "").startswith(pref):
                ji_src = node_to_joint.get(i)
                ji_dst = joint_of(dst)
                if ji_src is not None and ji_dst is not None:
                    jmap[ji_src] = ji_dst
    # jmap_full additionally bangs lower-body roots onto hips (skirt etc.)
    j_hips_j = joint_of("センター")
    if j_hips_j is None:
        j_hips_j = j_hips
    jmap_full = dict(jmap)
    for ji in special_joints:
        jmap_full[ji] = j_hips_j

    def jname(ji):
        ni = joints[ji]
        return (nodes[ni].get("name") or "?")

    def jname2(ji):
        return jname(ji)

    for probe in (9, 123, 124, 126, 127):
        src = jmap.get(probe)
        if src is not None:
            print("JMAP probe: joint %d (%s) -> joint %d (%s)"
                  % (probe, jname(probe), src, jname(src)))
    print("Phase A: %d base redirects (+%d lower-root->hips for full) "
          % (len(jmap), len(special_joints)))

    mesh = gltf["meshes"][0]
    trace_verts = {}
    if os.environ.get("TRACE"):
        for spec in os.environ["TRACE"].split(";"):
            pi_s, vs = spec.split(":")
            trace_verts[int(pi_s)] = {int(x) for x in vs.split(",") if x}

    # ---- process body skin FIRST so cloth prims can copy its weights ------

    def process_prim(pi, prim, jmap_used, copy_from, copy_band=None,
                     no_arm=False, alt_copy=None, gate_arm=False):
        # copy_band: (lo, hi) smooth copy transition; (0, 0) = full copy
        # alt_copy: (search, J, W, y_min) - copy from a second source
        #           (e.g. the shirt) above a higher y threshold (strap zone)
        # gate_arm: scale copy by arm-freeness of samples (shirt band)
        attrs = prim["attributes"]
        if "JOINTS_0" not in attrs or "WEIGHTS_0" not in attrs:
            print("prim %2d: no skin (skip)" % pi)
            return None
        ja, wa = attrs["JOINTS_0"], attrs["WEIGHTS_0"]
        j0, jn, jstride = acc_range(gltf, ja)
        w0, wn, wstride = acc_range(gltf, wa)
        jacc, wacc = gltf["accessors"][ja], gltf["accessors"][wa]
        for acc, tag in ((jacc, "JOINTS_0"), (wacc, "WEIGHTS_0")):
            st = acc_range(gltf, attrs[tag])[2]
            tight = NCOMP[acc["type"]] * COMP[acc["componentType"]]
            if st is not None and st != tight:
                raise SystemExit("prim %d: %s strided (unsupported)" % (pi, tag))
        if jacc["componentType"] != 5121:
            raise SystemExit("JOINTS_0 not uint8 in prim %d" % pi)
        wct = wacc["componentType"]
        if wct != 5126 and wct != 5121:
            raise SystemExit("WEIGHTS_0 dtype %d unsupported" % wct)

        nvert = jacc["count"]
        J = np.frombuffer(blob[j0:j0 + jn], dtype=DTNP[5121]).reshape(nvert, 4)
        W = np.frombuffer(blob[w0:w0 + wn],
                          dtype=DTNP[wct]).reshape(nvert, 4)
        Wf = W.astype(np.float32)
        if wct == 5121:
            Wf = Wf / 255.0
        pos_acc = gltf["accessors"][attrs["POSITION"]]
        p0, pn, _ = acc_range(gltf, attrs["POSITION"])
        P = np.frombuffer(blob[p0:p0 + pn],
                          dtype=np.float32).reshape(pos_acc["count"], 3)
        assert pos_acc["count"] == nvert and pos_acc["componentType"] == 5126

        hu_set = set(hu_joints)
        newJ = np.zeros((nvert, 4), dtype=np.int64)
        newW = np.zeros((nvert, 4), dtype=np.float32)
        split_used = copied = fallback = 0
        strip_copied = 0

        # nearest-candidate machinery (zone masked)
        if hu_joints_phaseb and joint_pos:
            hu_mat = np.stack([joint_pos[j] for j in hu_joints_phaseb])
            allowed = np.ones((nvert, len(hu_joints_phaseb)), dtype=bool)
            for cj, jj in enumerate(hu_joints_phaseb):
                if jj in arm_joints:
                    allowed[:, cj] = np.abs(P[:, 0]) >= 0.13
                if jj in leg_joints:
                    allowed[:, cj] &= P[:, 1] <= 0.90
                # hips must never grab strays deep in the thigh/knee zone
                # (hollow legs) NOR at chest height (cut band): restrict
                # to the pelvic window only
                if j_hips_j is not None and jj == j_hips_j:
                    allowed[:, cj] &= (P[:, 1] >= 0.635) & (P[:, 1] <= 0.75)
                if jj in sh_joints:
                    # shoulder bones only for shoulder-cap territory
                    allowed[:, cj] &= np.abs(P[:, 0]) >= 0.09
            dist = ((P[:, None, :] - hu_mat[None, :, :]) ** 2).sum(-1)
            dist_masked = np.where(allowed, dist, np.inf)
            nearest = np.argmin(dist_masked, axis=1)
        else:
            nearest = None

        for v in range(nvert):
            _tv = trace_verts and pi in trace_verts and v in trace_verts[pi]
            if _tv:
                print("TRACE v=%d pos=%s origSlots=%s" % (
                    v, P[v].round(3),
                    [(int(J[v, k]), round(float(Wf[v, k]), 3))
                     for k in range(4) if Wf[v, k] > 0.0]))
            # ---- original pipeline: jmap + contextual split + Phase B
            agg = {}
            for k in range(4):
                if Wf[v, k] > 0.0:
                    j = int(J[v, k])
                    if j in jmap_used:
                        j = jmap_used[j]
                    if j in special_joints and jmap_used is jmap_base:
                        # contextual lower-body-root split
                        y = float(P[v, 1])
                        if y >= 0.95:
                            tgt = j_chest if y >= 1.10 else j_spine
                            if tgt is None:
                                tgt = j_hips_j
                        else:
                            s = smoothstep((top_mix - y)
                                           / max(1e-6, top_mix - bot_mix))
                            side = j_upL if P[v, 0] >= 0.0 else j_upR
                            w = float(Wf[v, k])
                            hh = w * (1.0 - s)
                            # hips share stays onセンター
                            agg[j_hips_j] = agg.get(j_hips_j, 0.0) + hh
                            if s > 1e-4:
                                agg[side] = agg.get(side, 0.0) + w * s
                            split_used += 1
                            continue
                        j = tgt
                    agg[j] = agg.get(j, 0.0) + float(Wf[v, k])
            # Phase B: strays -> zone-restricted nearest
            stray = 0.0
            for j in list(agg.keys()):
                if j not in hu_set:
                    stray += agg.pop(j)
            if stray > 0.0 and nearest is not None:
                best = hu_joints_phaseb[int(nearest[v])]
                agg[best] = agg.get(best, 0.0) + stray
                if _tv:
                    print("TRACE v=%d stray=%.3f transferred to joint %d (%s)"
                          % (v, stray, best, jname2(best)))
            # no-arm cloth rule (skirt must never ride the arm chain)
            if no_arm:
                armslot = [j for j in agg if j in arm_joints]
                for j in armslot:
                    tgt = j_chest if j_chest is not None else j_hips_j
                    agg[tgt] = agg.get(tgt, 0.0) + agg.pop(j)

            # ---- v10: smooth copy blend --------------------------------
            # copy_band=(lo,hi): below lo -> original; above hi -> full
            # copy; between -> smoothstep alpha so the switch is a BAND,
            # not a hard ring ("cut in half" fix).
            alpha = 0.0
            cagg = None
            full_copy = (copy_band is not None
                         and copy_band[0] == copy_band[1])
            if (copy_from is not None and copy_band is not None
                    and (full_copy or float(P[v, 1]) >= copy_band[0])
                    and not (gate_arm and abs(float(P[v, 0])) > 0.12)):
                if full_copy:
                    t = 1.0
                else:
                    t = smoothstep(
                        (float(P[v, 1]) - copy_band[0])
                        / max(1e-6, copy_band[1] - copy_band[0]))
                if t > 0.001:
                    samps = None
                    SJ = SW = None
                    if (alt_copy is not None
                            and float(P[v, 1]) >= alt_copy[3]):
                        samps = alt_copy[0](P[v].astype(np.float64))
                        SJ, SW = alt_copy[1], alt_copy[2]
                    if samps is None:
                        samps = copy_from(P[v].astype(np.float64))
                        SJ, SW = skin_final_j, skin_final_w
                    if not samps:
                        fallback += 1
                        if _tv:
                            print("TRACE v=%d copy FAILED -> fallback" % v)
                    else:
                        cagg = {}
                        tot = 0.0
                        for sv, dist in samps:
                            wgt = 1.0 / (dist + 1e-4)
                            tot += wgt
                            for k in range(4):
                                w = float(SW[sv, k])
                                if w <= 1e-6:
                                    continue
                                j = int(SJ[sv, k])
                                cagg[j] = cagg.get(j, 0.0) + w * wgt
                        if no_arm:
                            # dress never rides the arm chain itself
                            tgt_chest = (j_chest if j_chest is not None
                                         else j_hips_j)
                            for j in [j2 for j2 in cagg
                                      if j2 in arm_joints]:
                                atot = cagg.pop(j)
                                cagg[tgt_chest] = (cagg.get(tgt_chest, 0.0)
                                                   + atot)
                        # gate_arm copies the skin field FAITHFULLY (its
                        # collar carries legit shoulder/neck mixes):
                        # alignment with the body beats dilution, and the
                        # |x|<=0.12 column already guards the sleeves.
                        if sum(cagg.values()) <= 1e-6:
                            cagg = None
                        else:
                            alpha = t
            if cagg is not None and alpha >= 0.02:
                if alpha >= 0.995:
                    agg = dict(cagg)
                else:
                    otot = sum(agg.values()) or 1.0
                    ctot = sum(cagg.values()) or 1.0
                    merged = {}
                    for j, w in agg.items():
                        merged[j] = w / otot * (1.0 - alpha)
                    for j, w in cagg.items():
                        merged[j] = merged.get(j, 0.0) + w / ctot * alpha
                    agg = merged
                copied += 1
                strip_copied += 1
            for j in list(agg.keys()):
                if agg[j] < 0.005:
                    del agg[j]
            out = sorted(agg.items(), key=lambda kv: -kv[1])[:4]
            if not out:
                out = [(hu_joints[0], 1.0)]
            tot = sum(w for _, w in out)
            tot = tot if tot > 1e-6 else 1.0
            for k in range(4):
                if k < len(out):
                    newJ[v, k], newW[v, k] = out[k][0], out[k][1] / tot
            if _tv:
                print("TRACE v=%d agg=%s out=%s" % (
                    v, [(j, round(w, 3)) for j, w in agg.items()],
                    [(int(j), round(w, 3)) for j, w in out]))

        if pi == 0:
            nfix = recon_legfield(newJ, newW, P)
            print("  prim 0 leg-field re-proportioned verts=%d" % nfix)
        blob[j0:j0 + jn] = newJ.astype(np.uint8).tobytes()
        if wct == 5126:
            blob[w0:w0 + wn] = newW.astype(np.float32).tobytes()
        else:
            blob[w0:w0 + wn] = (newW * 255.0).round().astype(
                np.uint8).tobytes()
        print("prim %2d %-26s verts=%4d copied=%d splitSlots=%d "
              "copyFallback=%d stripCopied=%d"
              % (pi, esc(mats[prim.get("material", -1)]
                         .get("name", "?"))[:26] if
                  prim.get("material", -1) >= 0 else "?",
                  nvert, copied, split_used, fallback, strip_copied))
        return P, newJ, newW

    prims = mesh["primitives"]

    # body skin (prim 0): jmap without lower-roots + contextual split,
    # processed FIRST so cloth prims can copy its corrected weights
    jmap_base = jmap                    # mapping WITHOUT specials
    res0 = process_prim(0, prims[0], jmap_base, None)
    skin_final_j = res0[1]
    skin_final_w = res0[2]
    skin_pos = res0[0]

    def make_search(pos, k=4, rmax=0.12, fmax=0.15):
        """nearest-sample search returning up to k samples (idx, dist)
        within rmax, with widening rings + brute fallback (capped at
        fmax), so cloth copies blend a smooth field instead of one
        sharp sample. Nothing near enough => [] -> keep originals."""
        cell2 = 0.02
        grid2 = {}
        for i in range(pos.shape[0]):
            grid2.setdefault(tuple((pos[i] / cell2).astype(np.int64)),
                             []).append(i)

        def search(p):
            base = (p / cell2).astype(np.int64)
            best = []
            for R in (2, 4, 6, 10):
                cand = []
                for dx in range(-R, R + 1):
                    for dy in range(-R, R + 1):
                        for dz in range(-R, R + 1):
                            for sv in grid2.get(tuple(base + np.array(
                                    [dx, dy, dz])), ()):
                                d = float(np.sqrt(((pos[sv] - p)
                                                   ** 2).sum()))
                                cand.append((d, sv))
                if cand:
                    cand.sort()
                    best = [(sv, d) for d, sv in cand if d <= rmax][:k]
                    if len(best) >= k:
                        break
            if not best:
                d2all = ((pos - p) ** 2).sum(1)
                i0 = int(np.argmin(d2all))
                d0 = float(np.sqrt(d2all[i0]))
                if fmax is not None and d0 > fmax:
                    return []
                return [(i0, d0)]
            return best

        return search

    copy_from = make_search(skin_pos, fmax=0.15)
    # the shirt needs a TIGHT local sample radius: averaging k samples
    # across a steep neck/chest gradient smears the skin field, so
    # shirt-vs-skin adjacency drifts (minimal chest cut residue)
    shirt_copy_from = make_search(skin_pos, rmax=0.035, fmax=0.06)

    shirt_res = None
    for pi in range(1, len(prims)):
        prim = prims[pi]
        band = COPY_PRIMS.get(pi)     # None -> plain pipeline
        alt_copy = None
        if pi == 18 and shirt_res is not None:
            shP, shJ, shW = shirt_res
            # strap zone copies the SHIRT's weights: the two layers then
            # move in lockstep (no see-saw between dress and shirt)
            alt_copy = (make_search(shP.astype(np.float64), fmax=0.15),
                        shJ, shW, 0.93)
        use_search = shirt_copy_from if pi == 17 else copy_from
        res = process_prim(pi, prim, jmap_full,
                           use_search if band is not None else None,
                           copy_band=band,
                           no_arm=(pi == 18), gate_arm=(pi == 17),
                           alt_copy=alt_copy)
        if pi == 17 and res is not None:
            shirt_res = res
    # ---- Phase D: texture rebind + alpha-mode + composites -----------------
    images = gltf.get("images", [])
    textures = gltf.get("textures", [])
    tex_src = [t.get("source", -1) for t in textures]

    def image_for_name(name):
        base = re.sub(r"\.\d+$", "", name).strip()
        for probe in (base + "_(", base + ".", base + "_"):
            for ti, im in enumerate(tex_src):
                iname = images[im].get("name") or ""
                if iname.startswith(probe):
                    return ti
        return None

    bound = 0
    for mat in mats:
        name = mat.get("name", "")
        pbr = mat.setdefault("pbrMetallicRoughness", {})
        factor = pbr.get("baseColorFactor", [1.0, 1.0, 1.0, 1.0])
        alpha_ok = (mat.get("alphaMode", "OPAQUE") == "OPAQUE"
                    or len(factor) > 3 and factor[3] > 0.5)
        ti = image_for_name(name)
        if ti is not None and alpha_ok:
            pbr["baseColorTexture"] = {"index": ti}
            pbr["baseColorFactor"] = [1.0, 1.0, 1.0, 1.0]
            mat.pop("metallicFactor", None)
            mat["roughnessFactor"] = 1.0
            bound += 1
            print("bound   %-28s -> %s" % (
                esc(name), esc(images[tex_src[ti]].get("name") or "?")))
        else:
            print("LEFT    %-28s (%s alpha=%.2f)" % (
                esc(name), "no image" if ti is None else "alpha",
                factor[3] if len(factor) > 3 else 1.0))
    print("Phase D: %d/%d materials bound" % (bound, len(mats)))

    # alpha overlay modes
    for mat in mats:
        name = mat.get("name", "")
        if name in BLEND_MATERIALS:
            mat["alphaMode"] = "BLEND"
            pbr = mat.setdefault("pbrMetallicRoughness", {})
            pbr["baseColorFactor"] = [1.0, 1.0, 1.0, 1.0]
            print("blend   %-28s" % esc(name))

    # eye composites (white sclera + iris art, embedded as new textures)
    for mat in mats:
        name = mat.get("name", "")
        if name not in EYE_COMPOSITE:
            continue
        pref, tag = EYE_COMPOSITE[name]
        pbr = mat.setdefault("pbrMetallicRoughness", {})
        bti = (pbr.get("baseColorTexture") or {}).get("index", -1)
        if bti < 0:
            print("eye comp: %s has no bound tex?" % esc(name))
            continue
        src_img = tex_src[bti]
        iname = images[src_img].get("name") or ("img" + str(src_img))
        sane = "".join(c if c.isalnum() or c in "_-." else "_"
                       for c in iname.replace(" (Instance)", "")
                       .replace("/", "_"))
        cands = [
            os.path.join(IMGD, "%02d_%s.png.png" % (src_img, sane)),
            os.path.join(IMGD, "%02d_%s.png" % (src_img, sane)),
            os.path.join(IMGD, sane + ".png"),
            os.path.join(IMGD, sane + ".png.png"),
        ]
        path = next((c for c in cands if os.path.exists(c)), None)
        if path is None:
            print("eye comp FAIL: %s missing (probed %s...)"
                  % (esc(name), os.path.basename(cands[0])))
            continue
        try:
            img = Image.open(path).convert("RGBA")
        except Exception:
            print("eye comp FAIL: %s unreadable" % esc(name))
            continue
        arr = np.asarray(img).astype(np.float32)
        a = arr[:, :, 3:4] / 255.0
        white = np.ones_like(arr)
        white[:, :, :3] = 255.0
        comp = arr * a + white * (1.0 - a)      # premultiplied blend over W
        comp[:, :, :3] = np.clip(comp[:, :, :3], 0, 255)
        comp[:, :, 3] = 255.0
        out_png = os.path.join(r"D:\Temp\opencode", tag + ".png")
        Image.fromarray(comp.astype(np.uint8), "RGBA").save(out_png)
        with open(out_png, "rb") as fh:
            pngb = fh.read()
        # embed: 4-align, append buffer bytes, then bufferView + image + tex
        while len(blob) % 4:
            blob.append(0)
        off = len(blob)
        blob.extend(pngb)
        bv_idx = len(gltf["bufferViews"])
        gltf["bufferViews"].append({"buffer": 0, "byteOffset": off,
                                    "byteLength": len(pngb)})
        img_idx = len(images)
        images.append({"bufferView": bv_idx, "mimeType": "image/png",
                       "name": iname + " whitecomp"})
        tex_idx = len(textures)
        textures.append({"source": img_idx})
        pbr["baseColorTexture"] = {"index": tex_idx}
        mat["alphaMode"] = "OPAQUE"
        # render eyes full-bright: PBR shading turned white sclera grey
        mat.setdefault("extensions", {})["KHR_materials_unlit"] = {}
        mat.pop("metallicFactor", None)
        pbr.setdefault("metallicFactor", 0.0)
        pbr.setdefault("roughnessFactor", 1.0)
        mat.pop("normalTexture", None)
        print("eye comp %-26s -> new img %d tex %d (%d bytes, unlit)"
              % (esc(name), img_idx, tex_idx, len(pngb)))

    write_glb(DST, gltf, blob)
    print("written %s (%d bytes)" % (DST, os.path.getsize(DST)))


if __name__ == "__main__":
    main()
