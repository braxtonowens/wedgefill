"""Triangle-mesh utilities. Coordinates are (z, y, x) voxel units; z (axis 0) is the beam axis."""
import numpy as np
import torch


def icosphere(level):
    """Unit icosphere with consistently outward-oriented faces: (vertices (V, 3), faces (F, 3))."""
    t = (1 + 5**0.5) / 2
    v = [[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0], [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
         [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]]
    f = [[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9], [5, 11, 4], [11, 10, 2],
         [10, 7, 6], [7, 1, 8], [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9], [4, 9, 5], [2, 4, 11],
         [6, 2, 10], [8, 6, 7], [9, 8, 1]]
    v = [np.array(p, float) / np.linalg.norm(p) for p in v]
    for _ in range(level):
        cache, nf = {}, []

        def mid(i, j):
            key = (min(i, j), max(i, j))
            if key not in cache:
                m = v[i] + v[j]
                v.append(m / np.linalg.norm(m))
                cache[key] = len(v) - 1
            return cache[key]

        for a, b, c in f:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            nf += [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]
        f = nf
    return np.array(v), np.array(f)


def face_geometry(v, f):
    """Per-face unit normals, areas and centroids."""
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    n = np.cross(b - a, c - a)
    area = 0.5 * np.linalg.norm(n, axis=1)
    return n / (2 * area[:, None] + 1e-12), area, (a + b + c) / 3


def area_volume(v, f):
    """Surface area and enclosed volume of a closed, consistently oriented mesh."""
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1).sum()
    vol = abs(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)
    return float(area), float(vol)


def sample_surface(v, f, n, rng):
    """n points sampled uniformly by area on the mesh."""
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    w = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    idx = rng.choice(len(f), n, p=w / w.sum())
    r1, r2 = rng.random(n), rng.random(n)
    s = np.sqrt(r1)
    return (1 - s)[:, None] * a[idx] + (s * (1 - r2))[:, None] * b[idx] + (s * r2)[:, None] * c[idx]


def voxelize(v, f, shape):
    """Inside mask of a closed mesh by ray parity along axis 0 for each (y, x) column of a grid of `shape`."""
    mask = np.zeros(shape, bool)
    hits = {}
    for tri in f:
        A, B, C = v[tri]
        ys = np.arange(int(np.ceil(min(A[1], B[1], C[1]))), int(np.floor(max(A[1], B[1], C[1]))) + 1)
        xs = np.arange(int(np.ceil(min(A[2], B[2], C[2]))), int(np.floor(max(A[2], B[2], C[2]))) + 1)
        if not len(ys) or not len(xs):
            continue
        d = (B[1] - A[1]) * (C[2] - A[2]) - (C[1] - A[1]) * (B[2] - A[2])
        if abs(d) < 1e-12:
            continue
        Y, X = np.meshgrid(ys, xs, indexing="ij")
        Y, X = Y.ravel() + 1e-4, X.ravel() + 2e-4  # nudge off vertices and edges
        l1 = ((Y - A[1]) * (C[2] - A[2]) - (C[1] - A[1]) * (X - A[2])) / d
        l2 = ((B[1] - A[1]) * (X - A[2]) - (Y - A[1]) * (B[2] - A[2])) / d
        ok = (l1 >= 0) & (l2 >= 0) & (l1 + l2 <= 1)
        zz = A[0] + l1 * (B[0] - A[0]) + l2 * (C[0] - A[0])
        for yy, xx, z in zip(Y[ok].astype(int), X[ok].astype(int), zz[ok]):
            hits.setdefault((yy, xx), []).append(z)
    zgrid = np.arange(shape[0])
    for (yy, xx), zs in hits.items():
        if not (0 <= yy < shape[1] and 0 <= xx < shape[2]):
            continue
        zs = np.sort(zs)
        for z0, z1 in zip(zs[0::2], zs[1::2]):
            mask[(zgrid >= z0) & (zgrid <= z1), yy, xx] = True
    return mask


# ------------------------------------------------------------------ differentiable (torch)
def vertex_normals_t(V, F):
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    fn = torch.linalg.cross(b - a, c - a)
    n = torch.zeros_like(V)
    for i in range(3):
        n.index_add_(0, F[:, i], fn)
    return n / (n.norm(dim=1, keepdim=True) + 1e-12)


def willmore_terms(V, F):
    """Discrete Willmore energy sum_i H_i^2 A_i (cotangent Laplacian, barycentric areas), vertex areas, face areas.

    A round sphere gives 4*pi regardless of radius.
    """
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]

    def cot(p, q, r):  # cotangent of the angle at p
        u, w = q - p, r - p
        return (u * w).sum(1) / (torch.linalg.cross(u, w).norm(dim=1) + 1e-9)

    ca, cb, cc = cot(a, b, c), cot(b, c, a), cot(c, a, b)
    S = torch.zeros_like(V)
    for i, j, w in ((1, 2, ca), (2, 0, cb), (0, 1, cc)):  # edge (i, j) is opposite the corner with cot w
        vi, vj = F[:, i], F[:, j]
        d = (V[vj] - V[vi]) * w[:, None]
        S.index_add_(0, vi, d)
        S.index_add_(0, vj, -d)
    farea = 0.5 * torch.linalg.cross(b - a, c - a).norm(dim=1)
    A = torch.zeros(len(V), dtype=V.dtype, device=V.device)
    for i in range(3):
        A.index_add_(0, F[:, i], farea / 3)
    W = ((S**2).sum(1) / (16 * A + 1e-9)).sum()
    return W, A, farea
