"""
Vectors and rotations, shared by everything that poses a skeleton.

A `Vector` is three floats; a `Matrix` is a 3x3 rotation as rows. Both games are Y-up and build a
rotation from three angles the same way: about X first, then Y, then Z.
"""

from __future__ import annotations

import math

Vector = tuple[float, float, float]
Matrix = list[list[float]]


def add(a: Vector, b: Vector) -> Vector:
    """The sum of two vectors."""
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def sub(a: Vector, b: Vector) -> Vector:
    """The vector from `b` to `a`."""
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def dot(a: Vector, b: Vector) -> float:
    """The dot product."""
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def cross(a: Vector, b: Vector) -> Vector:
    """The cross product."""
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def unit(a: Vector) -> Vector:
    """`a` at length one; straight up if it has no length to speak of."""
    length = math.sqrt(dot(a, a))
    return (a[0] / length, a[1] / length, a[2] / length) if length > 1e-12 else (0.0, 1.0, 0.0)


def matmul(a: Matrix, b: Matrix) -> Matrix:
    """The rotation that applies `b`, then `a`."""
    return [[a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j] for j in range(3)] for i in range(3)]


def transpose(a: Matrix) -> Matrix:
    """A rotation's transpose, which is its inverse."""
    return [[a[j][i] for j in range(3)] for i in range(3)]


def rotate(m: Matrix, v: Vector) -> Vector:
    """A vector turned by a rotation."""
    return tuple(m[i][0] * v[0] + m[i][1] * v[1] + m[i][2] * v[2] for i in range(3))


def to_local(rotation: Matrix, origin: Vector, point: Vector) -> Vector:
    """A model-space point in the space of a bone at `origin` with rest rotation `rotation`."""
    d = sub(point, origin)
    return tuple(rotation[0][c] * d[0] + rotation[1][c] * d[1] + rotation[2][c] * d[2] for c in range(3))


def to_model(rotation: Matrix, origin: Vector, point: Vector) -> Vector:
    """The inverse of `to_local`: a point in a bone's space, in model space."""
    return tuple(
        origin[r] + rotation[r][0] * point[0] + rotation[r][1] * point[1] + rotation[r][2] * point[2] for r in range(3)
    )


def euler_matrix(rx: float, ry: float, rz: float) -> Matrix:
    """The rotation for angles in radians about X, then Y, then Z."""
    cx, sx, cy, sy, cz, sz = math.cos(rx), math.sin(rx), math.cos(ry), math.sin(ry), math.cos(rz), math.sin(rz)
    return [
        [cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx],
        [sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx],
        [-sy, cy * sx, cy * cx],
    ]


def between(a: Matrix, b: Matrix, share: float) -> Matrix:
    """
    The rotation `share` of the way from `a` to `b`.

    It turns about one axis, the short way round: the turn that takes `a` to `b`, cut down to that
    share of its angle.

    Returns:
        `a` at a share of 0 or less, `b` at 1 or more, and a rotation between them otherwise.
    """
    if share <= 0.0:
        return [list(row) for row in a]
    if share >= 1.0:
        return [list(row) for row in b]
    step = matmul(b, transpose(a))
    angle = math.acos(max(-1.0, min(1.0, (step[0][0] + step[1][1] + step[2][2] - 1.0) / 2.0)))
    axis = (step[2][1] - step[1][2], step[0][2] - step[2][0], step[1][0] - step[0][1])
    length = math.sqrt(sum(c * c for c in axis))
    if angle < 1e-9 or length < 1e-9:  # the same rotation; or half a turn exactly, which has no short way round
        return [list(row) for row in (a if angle < 1.0 else b)]
    x, y, z = (c / length for c in axis)
    c, s = math.cos(angle * share), math.sin(angle * share)
    part = [
        [c + x * x * (1 - c), x * y * (1 - c) - z * s, x * z * (1 - c) + y * s],
        [y * x * (1 - c) + z * s, c + y * y * (1 - c), y * z * (1 - c) - x * s],
        [z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + z * z * (1 - c)],
    ]
    return matmul(part, a)


def matrix_euler(m: Matrix, near: Vector | None = None) -> Vector:
    """
    The angles `euler_matrix` would turn into rotation `m`.

    A rotation has more than one such triple. `near` (a previous triple) picks the one closest to
    it, which keeps a baked curve from jumping a turn between two frames.

    Returns:
        The angles about X, Y and Z, in radians.
    """
    sy = max(-1.0, min(1.0, -m[2][0]))
    if abs(sy) > 1.0 - 1e-12:  # looking straight along Y: X and Z turn about one axis
        x, y, z = 0.0, math.asin(sy), math.atan2(-m[0][1], m[1][1])
    else:
        x, y, z = math.atan2(m[2][1], m[2][2]), math.asin(sy), math.atan2(m[1][0], m[0][0])
    if near is None:
        return (x, y, z)
    best = None
    for candidate in ((x, y, z), (x + math.pi, math.pi - y, z + math.pi)):
        turned = tuple(a + 2 * math.pi * round((n - a) / (2 * math.pi)) for a, n in zip(candidate, near, strict=True))
        distance = sum((a - n) ** 2 for a, n in zip(turned, near, strict=True))
        if best is None or distance < best[0]:
            best = (distance, turned)
    return best[1]
