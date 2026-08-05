"""Marginal projections of a 3-D voxel volume, for imshow with origin='lower'."""
from __future__ import annotations


def project(V):
    """(nx, ny, nz) -> (xy, zy), each with depth y as the first axis."""
    return V.sum(2).T, V.sum(0)


def extents(g):
    ex_xy = [g.x_lo, g.x_lo + g.nx * g.x_bin, g.y_lo, g.y_lo + g.ny * g.y_bin]
    ex_zy = [g.z_lo, g.z_lo + g.nz * g.z_bin, g.y_lo, g.y_lo + g.ny * g.y_bin]
    return ex_xy, ex_zy
