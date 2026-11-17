"""Wedge-aware membrane completion.

A segmented membrane (e.g. a MemBrain-seg mask) loses its top and bottom in cryo-ET because the missing wedge
removes signal where the membrane normal is near the beam axis. `complete_membrane` fits a closed surface to
the mask points under a Willmore (bending-energy) prior, with two rules:

* surface whose normal points in a VISIBLE direction (more than `hidden_deg` from the beam axis) must lie on the
  data; it is pulled onto the mask during the fit and never added in the output;
* only off-mask surface whose normal lies within `hidden_deg` of the beam axis is returned as completion.

So membrane is only added where the wedge could have hidden it; openings visible in XY stay open.
Coordinates are (z, y, x) voxels; z is the beam axis.
"""
from dataclasses import dataclass

import numpy as np
import torch
from scipy.spatial import cKDTree

from .mesh import face_geometry, icosphere, vertex_normals_t, willmore_terms


@dataclass
class Completion:
    vertices: np.ndarray       # (V, 3) fitted closed surface
    faces: np.ndarray          # (F, 3)
    added: np.ndarray          # (F,) bool: faces returned as completion (off-mask, in the hidden cone)
    dropped: np.ndarray        # (F,) bool: off-mask faces in visible directions (discarded)
    dist_to_mask: np.ndarray   # (F,) distance of each face centroid to the nearest mask point (voxels)

    def area_fractions(self):
        _, area, _ = face_geometry(self.vertices, self.faces)
        t = area.sum()
        return dict(added=float(area[self.added].sum() / t), dropped=float(area[self.dropped].sum() / t))


def complete_membrane(points, hidden_deg=55.0, lam=0.05, w_vis=1.0, level=4, steps=1500, off_mask=2.0,
                      wedge_aware=True, max_points=6000, device="cpu", seed=0):
    """Fit a wedge-aware closed surface to membrane points (N, 3) and classify its faces.

    Parameters
    ----------
    points : (N, 3) array of (z, y, x) mask voxel coordinates of ONE membrane object.
    hidden_deg : normals within this angle of the beam axis may be completed. MemBrain-seg on a +-64.5 deg
        tilt series lost membrane up to about 55 deg, wider than the geometric wedge (90 - tilt).
    lam : weight of the Willmore energy relative to the data terms.
    w_vis : weight of the term that pulls visible-direction surface onto the mask (0 with wedge_aware=False).
    off_mask : faces farther than this from every mask point (voxels) count as new surface.
    wedge_aware : False gives the plain closed fit (completes in every direction; for comparison only).
    """
    rng = np.random.default_rng(seed)
    points = np.asarray(points, float)
    fit_pts = points[rng.choice(len(points), min(max_points, len(points)), replace=False)]
    v0, f = icosphere(level)
    c = fit_pts.mean(0)
    r = np.median(np.linalg.norm(fit_pts - c, axis=1))
    V = torch.tensor(c + r * v0, dtype=torch.float64, device=device, requires_grad=True)
    F = torch.tensor(f, device=device)
    P = torch.tensor(fit_pts, dtype=torch.float64, device=device)
    opt = torch.optim.Adam([V], lr=0.2)
    cos_h = float(np.cos(np.deg2rad(hidden_deg)))
    e0 = None
    for step in range(steps):
        s = 8.0 * (1 - step / steps) + 1.5          # robust scale, annealed (voxels)
        D = torch.cdist(P, V)
        d2p = D.min(dim=1).values ** 2              # every mask point close to the surface
        loss = (d2p / (d2p + s**2)).mean() * s**2
        if wedge_aware and w_vis > 0:
            n = vertex_normals_t(V, F)
            vis = torch.sigmoid((cos_h - n[:, 0].abs()) / 0.05)  # ~1 where the normal is in a visible direction
            d2v = D.min(dim=0).values ** 2                         # vertex close to some mask point
            loss = loss + w_vis * (vis.detach() * d2v / (d2v + s**2)).mean() * s**2
        W, _, farea = willmore_terms(V, F)
        if e0 is None:
            e0 = farea.mean().detach()
        loss = loss + lam * W + 0.5 * ((farea / e0 - 1) ** 2).mean()  # last term keeps triangles even
        opt.zero_grad()
        loss.backward()
        opt.step()
    Vn = V.detach().cpu().numpy()
    fn, _, cen = face_geometry(Vn, f)
    dist = cKDTree(points).query(cen)[0]
    off = dist > off_mask
    hidden = np.abs(fn[:, 0]) > cos_h
    if not wedge_aware:
        return Completion(Vn, f, off, np.zeros_like(off), dist)
    return Completion(Vn, f, off & hidden, off & ~hidden, dist)
