"""Rotated-wedge cross-validation: hide a band of membrane that IS visible and test how well each method predicts it.

For each object (a component of the membrane mask), the band of mask voxels whose local normal lies within THETA of
the x or y axis is held out. The remaining mask is completed with the real beam-axis cone plus the held-out cone
marked as hidden. The held-out voxels are then compared against:

  wedgefill  : the fitted surface
  hull       : convex hull of the remaining voxels
  ellipsoid  : least-squares ellipsoid through the remaining voxels

Errors are distances (nm) from held-out voxels to each surface, with a "floor": the same distance for kept voxels to
the wedgefill surface (mask thickness and discretisation). Also the signed radial error (> 0 = predicted surface lies
outside the true membrane, i.e. inflated).

usage: python crossval_rotated_wedge.py MASK.zarr VOXEL_NM OUT.json COMP_ID [COMP_ID ...] [--device cuda]
"""
import argparse
import json
import time

import numpy as np
import zarr
from scipy import ndimage as ndi
from scipy.spatial import ConvexHull, cKDTree

from wedgefill import complete_membrane, sample_surface

S = 128


def open_level0(path):
    z = zarr.open(path, mode="r")
    return z["0"] if hasattr(z, "keys") and "0" in z else z


def object_points(lab, objs, cid):
    """Mask voxels of component cid, binned so the object fits S^3: (points (N, 3) in binned voxels, bin)."""
    sl = objs[cid - 1]
    ext = np.array([s.stop - s.start for s in sl])
    b = int(np.ceil(ext.max() * 1.25 / S))
    sub = lab[sl] == cid
    pts = (np.argwhere(sub) + np.array([s.start for s in sl])) / b
    pts = np.unique(np.floor(pts), axis=0)
    return pts - pts.mean(0), b


def local_normals(pts, r=4.0):
    tree = cKDTree(pts)
    n = np.zeros_like(pts)
    for i, nb in enumerate(tree.query_ball_point(pts, r)):
        q = pts[nb] - pts[nb].mean(0)
        n[i] = np.linalg.eigh(q.T @ q)[1][:, 0] if len(nb) >= 4 else 0
    return n


def ellipsoid_surface(p, n=40000, rng=None):
    """Least-squares quadric through p; returns sampled surface points or None if it is not an ellipsoid."""
    x, y, z = p.T
    A = np.stack([x * x, y * y, z * z, 2 * y * z, 2 * x * z, 2 * x * y, 2 * x, 2 * y, 2 * z], 1)
    c = np.linalg.lstsq(A, np.ones(len(p)), rcond=None)[0]
    M = np.array([[c[0], c[5], c[4]], [c[5], c[1], c[3]], [c[4], c[3], c[2]]])
    if np.any(np.linalg.eigvalsh(M) <= 0):
        return None
    ctr = -np.linalg.solve(M, c[6:9])
    k = 1 + ctr @ M @ ctr
    w, V = np.linalg.eigh(M / k)
    u = rng.normal(size=(n, 3)); u /= np.linalg.norm(u, axis=1, keepdims=True)
    return ctr + (u / np.sqrt(w)) @ V.T


def hull_surface(p, n=40000, rng=None):
    h = ConvexHull(p)
    return sample_surface(p, h.simplices, n, rng)


def errors(held, surf, centre):
    d, j = cKDTree(surf).query(held)
    rad_true = np.linalg.norm(held - centre, axis=1)
    rad_pred = np.linalg.norm(surf[j] - centre, axis=1)
    return float(d.mean()), float(np.percentile(d, 90)), float((rad_pred - rad_true).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mask"); ap.add_argument("voxel_nm", type=float); ap.add_argument("out")
    ap.add_argument("comps", nargs="+", type=int)
    ap.add_argument("--thetas", default="35,50")
    ap.add_argument("--hidden-deg", type=float, default=55.0)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    t0 = time.time()
    mask = np.asarray(open_level0(a.mask)[:]) > 0
    lab, _ = ndi.label(mask, structure=np.ones((3, 3, 3)))
    objs = ndi.find_objects(lab)
    print("loaded", round(time.time() - t0, 1), "s", flush=True)
    rows = []
    for cid in a.comps:
        pts, b = object_points(lab, objs, cid)
        nm = a.voxel_nm * b
        nrm = local_normals(pts)
        centre = pts.mean(0)
        for axis_name, axis in (("x", (0.0, 0.0, 1.0)), ("y", (0.0, 1.0, 0.0))):  # (z, y, x) order
            for th in map(float, a.thetas.split(",")):
                held = np.abs(nrm @ np.array(axis)) > np.cos(np.deg2rad(th))
                if held.sum() < 30 or (~held).sum() < 200:
                    continue
                keep = pts[~held]
                res = complete_membrane(keep, hidden_cones=[((1.0, 0.0, 0.0), a.hidden_deg), (axis, th)],
                                        device=a.device)
                surf_w = sample_surface(res.vertices, res.faces, 60000, rng)
                row = dict(comp=cid, axis=axis_name, theta=th, n_held=int(held.sum()), nm_per_vox=nm)
                floor = cKDTree(surf_w).query(keep)[0]
                row["floor_nm"] = float(floor.mean() * nm)
                for name, surf in (("wedgefill", surf_w), ("hull", hull_surface(keep, rng=rng)),
                                   ("ellipsoid", ellipsoid_surface(keep, rng=rng))):
                    if surf is None:
                        row[name] = None
                        continue
                    m, p90, bias = errors(pts[held], surf, centre)
                    row[name] = dict(mean_nm=m * nm, p90_nm=p90 * nm, radial_bias_nm=bias * nm)
                print(json.dumps(row), flush=True)
                rows.append(row)
    json.dump(rows, open(a.out, "w"), indent=1)
    print("done", round(time.time() - t0, 1), "s")


if __name__ == "__main__":
    main()
