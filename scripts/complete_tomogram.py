"""Complete membrane objects of a tomogram's membrane segmentation (e.g. MemBrain-seg) and save figures + stats.

usage:
  python complete_tomogram.py TOMO.zarr MASK.zarr OUTDIR COMP_ID[:name] [COMP_ID[:name] ...]
        [--hidden-deg 55] [--closed]

Component ids refer to the 26-connected labelling of the full-resolution mask (scipy.ndimage.label).
Each object is cropped (and binned so it fits 128^3), completed with wedgefill.complete_membrane, and plotted as
XZ / YZ / XY slices through its centre: yellow = mask, red = added membrane (or the closed surface with --closed).
With --auto N, the N largest objects not cut by the XY edges are tried; a result is "accepted" (and drawn on the
overview) only if it added >= --min-added of its surface, dropped <= --max-dropped (visible, off-mask surface; high
means not a closed membrane) and the mask is not flat (--min-flatness).
"""
import argparse
import json
import os
import time

import matplotlib
import numpy as np
import zarr
from scipy import ndimage as ndi

from wedgefill import complete_membrane, face_geometry, sample_surface

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

S = 128


def open_level0(path):
    z = zarr.open(path, mode="r")
    return z["0"] if hasattr(z, "keys") and "0" in z else z


def crop(a, c, side, fill):
    o = np.full((side,) * 3, fill, a.dtype)
    lo = [ci - side // 2 for ci in c]
    src = tuple(slice(max(0, l), min(n, l + side)) for l, n in zip(lo, a.shape))
    dst = tuple(slice(s.start - l, s.stop - l) for s, l in zip(src, lo))
    o[dst] = a[src]
    return o, np.array(lo)


def binv(a, b, how):
    if b == 1:
        return a
    s = a.reshape(S, b, S, b, S, b)
    return s.max((1, 3, 5)) if how == "max" else s.mean((1, 3, 5))


def norm(a, p=(1, 99)):
    lo, hi = np.percentile(a, p)
    return np.clip((a - lo) / (hi - lo + 1e-9), 0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tomo"); ap.add_argument("mask"); ap.add_argument("out")
    ap.add_argument("objects", nargs="*", help="COMP_ID[:name]; omit with --auto")
    ap.add_argument("--auto", type=int, default=0, help="complete the N largest suitable objects")
    ap.add_argument("--min-vox", type=int, default=3000)
    ap.add_argument("--max-bin", type=int, default=4)
    ap.add_argument("--min-added", type=float, default=0.15, help="accept: added area fraction at least this")
    ap.add_argument("--min-flatness", type=float, default=0.20,
                    help="accept: thinnest/widest principal-axis s.d. of the mask at least this (rejects flat streaks)")
    ap.add_argument("--max-dropped", type=float, default=0.10,
                    help="accept: dropped (visible, off-mask) area fraction at most this; high = not a closed membrane")
    ap.add_argument("--hidden-deg", type=float, default=55.0)
    ap.add_argument("--closed", action="store_true", help="plain closed fit (completes in every direction)")
    ap.add_argument("--device", default="cuda" if __import__("torch").cuda.is_available() else "cpu")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    tomo = np.asarray(open_level0(a.tomo)[:], np.float32)
    mask = np.asarray(open_level0(a.mask)[:]) > 0
    lab, _ = ndi.label(mask, structure=np.ones((3, 3, 3)))
    objs = ndi.find_objects(lab)
    print("loaded", round(time.time() - t0, 1), "s", flush=True)
    suffix = "_closed" if a.closed else ""
    objects = list(a.objects)
    if a.auto:
        sizes = ndi.sum(mask, lab, range(1, len(objs) + 1))
        cand = []
        for cid in np.argsort(sizes)[::-1] + 1:
            sl = objs[cid - 1]
            ext = np.array([s_.stop - s_.start for s_ in sl])
            touches_xy = any(s_.start == 0 or s_.stop == n for s_, n in zip(sl[1:], mask.shape[1:]))
            if sizes[cid - 1] < a.min_vox or touches_xy or np.ceil(ext.max() * 1.25 / S) > a.max_bin:
                continue
            cand.append(f"{cid}:obj{cid}")
            if len(cand) == a.auto:
                break
        objects = cand
        print("auto-selected", objects, flush=True)
    results, added_full = [], []
    for arg in objects:
        cid, name = (arg.split(":") + [None])[:2]
        cid = int(cid); name = name or f"comp{cid}"
        sl = objs[cid - 1]
        ext = np.array([s.stop - s.start for s in sl]); c = [(s.start + s.stop) // 2 for s in sl]
        b = int(np.ceil(ext.max() * 1.25 / S)); side = S * b
        T, lo = crop(tomo, c, side, np.float32(np.nan)); T = np.where(np.isnan(T), np.nanmean(T), T)
        L, _ = crop(lab, c, side, 0)
        comp = binv(L == cid, b, "max"); Tb = binv(T, b, "mean")
        pts = np.argwhere(comp).astype(float)
        tf = time.time()
        res = complete_membrane(pts, hidden_deg=a.hidden_deg, wedge_aware=not a.closed, device=a.device)
        fit_s = time.time() - tf
        fn, area, cen = face_geometry(res.vertices, res.faces)
        r = dict(name=name, comp_id=cid, bin=b, crop_origin_zyx=lo.tolist(), fit_s=round(fit_s, 1),
                 mask_extent_zyx_vox=np.ptp(pts, 0).tolist(),
                 completed_extent_z_vox=float(np.ptp(np.concatenate([pts[:, 0], cen[res.added, 0]]))),
                 **res.area_fractions(), visible_faces_mean_dist_vox=float(
                     np.average(res.dist_to_mask[~res.added & ~res.dropped], weights=area[~res.added & ~res.dropped])))
        ev = np.sqrt(np.clip(np.linalg.eigvalsh(np.cov((pts - pts.mean(0)).T)), 0, None))
        r["flatness"] = float(ev[0] / ev[-1])
        r["accepted"] = bool(r["added"] >= a.min_added and r["dropped"] <= a.max_dropped
                             and r["flatness"] >= a.min_flatness)
        print(json.dumps(r), flush=True)
        results.append(r)
        if res.added.any() and r["accepted"]:
            sp = sample_surface(res.vertices, res.faces[res.added], 20000, np.random.default_rng(1))
            added_full.append(lo + sp * b)  # back to full-resolution (z, y, x)
        # red overlay: thin shell sampled from the added faces
        shell = np.zeros((S, S, S), bool)
        if res.added.any():
            sp = sample_surface(res.vertices, res.faces[res.added], 40000, np.random.default_rng(0))
            ii = np.clip(np.round(sp).astype(int), 0, S - 1); shell[ii[:, 0], ii[:, 1], ii[:, 2]] = True
            shell = ndi.binary_dilation(shell)
        ctr = np.round(pts.mean(0)).astype(int)
        fig, axs = plt.subplots(2, 3, figsize=(9.5, 6.6))
        for j, (ax_n, (xl, yl)) in enumerate([(1, ("x", "z")), (2, ("y", "z")), (0, ("x", "y"))]):
            k = ctr[ax_n]
            img = norm(np.take(Tb, range(max(0, k - 2), k + 3), axis=ax_n).mean(ax_n))
            mk, sh = np.take(comp, k, axis=ax_n), np.take(shell, k, axis=ax_n)
            for i in range(2):
                axs[i, j].imshow(img, cmap="gray"); axs[i, j].set_xticks([]); axs[i, j].set_yticks([])
                axs[i, j].set_xlabel(xl, fontsize=8); axs[i, j].set_ylabel(yl, fontsize=8)
            if mk.any():
                axs[1, j].contour(mk, [0.5], colors=["#ffd000"], linewidths=0.7)
            if sh.any():
                axs[1, j].contour(sh, [0.5], colors=["#ff2d55"], linewidths=1.0)
        for k, ax in enumerate(axs.ravel()):
            ax.text(0.02, 0.97, "abcdef"[k], transform=ax.transAxes, color="w", fontsize=10, fontweight="bold",
                    va="top")
        fig.tight_layout(); fig.savefig(f"{a.out}/{name}{suffix}_slices.png", dpi=150); plt.close(fig)
    json.dump(results, open(f"{a.out}/stats{suffix}.json", "w"), indent=1)
    # overview: XZ slab through the tomogram at the y with the most added membrane
    if added_full:
        P = np.concatenate(added_full)
        y0 = int(np.bincount(np.clip(P[:, 1].astype(int), 0, mask.shape[1] - 1)).argmax())
        slab = tomo[:, max(0, y0 - 3):y0 + 4, :].mean(1)
        mk = mask[:, y0, :]
        near = np.abs(P[:, 1] - y0) < 1.5
        fig, ax = plt.subplots(figsize=(12, 12 * slab.shape[0] / slab.shape[1] + 0.4))
        ax.imshow(norm(slab), cmap="gray", aspect="equal")
        ax.contour(mk, [0.5], colors=["#ffd000"], linewidths=0.6)
        ax.scatter(P[near, 2], P[near, 0], s=0.6, c="#ff2d55", linewidths=0)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel("x"); ax.set_ylabel("z (beam)")
        fig.tight_layout(); fig.savefig(f"{a.out}/overview_xz{suffix}.png", dpi=150); plt.close(fig)
    print("done", round(time.time() - t0, 1), "s")


if __name__ == "__main__":
    main()
