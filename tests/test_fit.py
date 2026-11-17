"""Behaviour tests on a sphere shell (radius 20 voxels) with parts removed."""
import numpy as np

from wedgefill import complete_membrane, face_geometry

R = 20.0


def shell(keep):
    """Points on a sphere shell (1-voxel spacing), kept where keep(unit normal) is True."""
    g = np.mgrid[-24:25, -24:25, -24:25].reshape(3, -1).T.astype(float)
    r = np.linalg.norm(g, axis=1)
    p = g[np.abs(r - R) < 0.8]
    return p[keep(p / np.linalg.norm(p, axis=1, keepdims=True))]


def test_wedge_caps_are_completed_at_the_right_height():
    # remove top and bottom caps: normals within 40 deg of the beam axis (axis 0)
    pts = shell(lambda n: np.abs(n[:, 0]) < np.cos(np.deg2rad(40)))
    c = complete_membrane(pts, hidden_deg=55, level=3, steps=600)
    fn, area, cen = face_geometry(c.vertices, c.faces)
    assert c.added.any()
    # every added face lies in the hidden cone
    assert (np.abs(fn[c.added, 0]) > np.cos(np.deg2rad(55)) - 1e-9).all()
    # the completed caps reach the true poles (+-R) within 2 voxels
    zc = cen[c.added, 0]
    assert abs(zc.max() - R) < 2.0 and abs(zc.min() + R) < 2.0


def test_visible_opening_is_not_closed():
    # wedge caps missing AND a visible hole on the +x side (normals within 30 deg of +x)
    keep = lambda n: (np.abs(n[:, 0]) < np.cos(np.deg2rad(40))) & (n[:, 2] < np.cos(np.deg2rad(30)))
    pts = shell(keep)
    c = complete_membrane(pts, hidden_deg=55, level=3, steps=600)
    _, _, cen = face_geometry(c.vertices, c.faces)
    hole = cen[:, 2] > R * np.cos(np.deg2rad(25))
    assert not (c.added & hole).any(), "membrane was added across a visible opening"
    assert c.area_fractions()["dropped"] > 0.02
