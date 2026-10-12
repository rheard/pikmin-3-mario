"""
Draw Mario, read from a Super Mario 64 ROM, as a picture on a transparent background.

Nothing but Python is needed. It is a plain z-buffer rasteriser, slow by any other standard (a 64x64
picture takes a fraction of a second) and quick enough for a few dozen icons.

It draws what the game draws, as far as a still picture goes:

* **Shading the game's way.** The N64 lights each corner of a triangle and blends across it: the
  part's colour at full strength wherever the surface faces within 60 degrees of the light, falling
  to half at a right angle and beyond. The light sits over the viewer's right shoulder, where the
  game puts it, and turns with the camera as it does there.
* **His face as the game builds it.** Eyes, moustache, sideburns, cap badge and buttons are small
  pictures laid over the part's flat colour; `Mario.image` gives them already combined, and they
  are read here smoothed between texels, much as the game's own filter softens them. Which eyes he has is the
  `eyes` choice of `Mario` (open, half, closed, right, left, up, down, dead).
* **Edges.** Every pixel is the average of `samples` x `samples` points, and its alpha is the share
  of them he covers. Colour is kept at full strength right to the edge and carried a few pixels past
  it, so nothing dark creeps in when the picture is compressed or scaled.

`Figure(mario, pose, parts)` is him in one pose, `view(turn, tilt)` a camera, `fit(...)` a scale and
centre that put him inside a picture, and `draw(...)` the rows of RGBA.
"""

from __future__ import annotations

import math

from typing import TYPE_CHECKING

from mario.geometry import matmul, rotate

if TYPE_CHECKING:
    from mario.geometry import Matrix
    from mario.sm64.rom import Mario, Pose, Rows

LIGHT = (1 / math.sqrt(3.0),) * 3  # towards the light, in the camera's axes: right, up, at the viewer
AMBIENT = 0.5  # the game's shadowed colour is half the lit one

Bounds = tuple[float, float, float, float]  # left, right, bottom, top


def view(turn: float = 0.0, tilt: float = 0.0, roll: float = 0.0) -> Matrix:
    """
    A camera as a 3x3 matrix (rows): model space -> x right, y up, z towards the viewer.

    `turn` swings him about his upright axis (positive: he faces the viewer's right), `tilt` raises
    the camera to look down on him, `roll` leans the picture clockwise. All in degrees.

    Returns:
        The camera's rotation.
    """
    a, b, c = math.radians(turn), math.radians(tilt), math.radians(roll)
    ry = [[math.cos(a), 0.0, math.sin(a)], [0.0, 1.0, 0.0], [-math.sin(a), 0.0, math.cos(a)]]
    rx = [[1.0, 0.0, 0.0], [0.0, math.cos(b), -math.sin(b)], [0.0, math.sin(b), math.cos(b)]]
    rz = [[math.cos(c), math.sin(c), 0.0], [-math.sin(c), math.cos(c), 0.0], [0.0, 0.0, 1.0]]
    return matmul(rz, matmul(rx, ry))


class Figure:
    """
    Mario in one pose, ready to draw.

    `tris` is [(image or colour, [(position, normal, (u, v))] * 3)]: a part's flat colour as
    (r, g, b), or its picture as (width, height, rows). `parts` names the parts to keep; all of
    them if left out.
    """

    def __init__(self, mario: Mario, pose: Pose | None = None, parts: tuple[str, ...] | None = None) -> None:
        self.tris = []
        images = {}
        for material, part, corners in mario.triangles(pose):
            if parts is not None and part not in parts:
                continue
            if material not in images:
                width, height, rows = mario.image(material)
                first = rows[0][0][:3]
                flat = all(px[:3] == first for row in rows for px in row)
                images[material] = tuple(first) if flat else (width, height, rows)
            self.tris.append((images[material], corners))
        if not self.tris:
            raise ValueError(f"nothing to draw: no triangles in {parts}")

    def bounds(self, camera: Matrix) -> Bounds:
        """(left, right, bottom, top) of the figure as the camera sees it, in model units."""
        xs, ys = [], []
        for _, corners in self.tris:
            for position, _, _ in corners:
                x, y, _ = rotate(camera, position)
                xs.append(x)
                ys.append(y)
        return min(xs), max(xs), min(ys), max(ys)


def fit(
    bounds: Bounds,
    width: int,
    height: int,
    margin: float = 2.0,
    anchor: str = "centre",
) -> tuple[float, tuple[float, float]]:
    """
    (scale, centre) that put a rectangle of the camera's view inside a picture.

    `bounds` is (left, right, bottom, top); the picture keeps `margin` pixels clear all round. With
    `anchor="top"` the scale comes from the width alone and the top edge is placed at the margin:
    for a bust, whose lower part is meant to run off the bottom.

    Returns:
        Pixels per model unit, and the point of the view that lands in the middle of the picture.
    """
    left, right, bottom, top = bounds
    if anchor == "top":
        scale = (width - 2 * margin) / (right - left)
        return scale, ((left + right) / 2.0, top - (height / 2.0 - margin) / scale)
    scale = min((width - 2 * margin) / (right - left), (height - 2 * margin) / (top - bottom))
    return scale, ((left + right) / 2.0, (bottom + top) / 2.0)


def _texel(image: tuple[int, int, Rows], u: float, v: float) -> tuple[float, float, float]:
    """A picture read at a coordinate in picture widths (v down), smoothed, clamped at its edges."""
    w, h, rows = image
    x = u * w - 0.5
    y = v * h - 0.5
    if x < 0.0:
        x = 0.0
    elif x > w - 1.0:
        x = w - 1.0
    if y < 0.0:
        y = 0.0
    elif y > h - 1.0:
        y = h - 1.0
    x0, y0 = int(x), int(y)
    x1 = x0 + 1 if x0 + 1 < w else x0
    y1 = y0 + 1 if y0 + 1 < h else y0
    fx, fy = x - x0, y - y0
    a, b, c, e = rows[y0][x0], rows[y0][x1], rows[y1][x0], rows[y1][x1]
    gx, gy = 1.0 - fx, 1.0 - fy
    return (
        (a[0] * gx + b[0] * fx) * gy + (c[0] * gx + e[0] * fx) * fy,
        (a[1] * gx + b[1] * fx) * gy + (c[1] * gx + e[1] * fx) * fy,
        (a[2] * gx + b[2] * fx) * gy + (c[2] * gx + e[2] * fx) * fy,
    )


def _cover(
    figure: Figure,
    camera: Matrix,
    wide: int,
    high: int,
    scale: float,
    centre: tuple[float, float],
) -> tuple[list[int], list[tuple | None]]:
    """
    Find the triangle nearest the viewer at every point of a `wide` x `high` grid.

    Returns:
        Each point's triangle number (-1 where there is none), row after row, and for each triangle
        what shading it needs: its image, its corners on the grid as (x, y, depth, light, u, v), and
        the weights of its first two corners as a0 + b0*x + c0*y. A triangle facing away has None.
    """
    ox, oy = wide / 2.0 - centre[0] * scale, high / 2.0 + centre[1] * scale
    depth = [-1e30] * (wide * high)
    hit = [-1] * (wide * high)
    setup = []
    m = camera
    lx, ly, lz = LIGHT
    floor, ceil = math.floor, math.ceil
    for image, corners in figure.tris:
        pts = []
        for pos, nrm, uv in corners:
            x = m[0][0] * pos[0] + m[0][1] * pos[1] + m[0][2] * pos[2]
            y = m[1][0] * pos[0] + m[1][1] * pos[1] + m[1][2] * pos[2]
            z = m[2][0] * pos[0] + m[2][1] * pos[1] + m[2][2] * pos[2]
            n = rotate(m, nrm)
            lit = AMBIENT + max(0.0, n[0] * lx + n[1] * ly + n[2] * lz)
            pts.append((ox + x * scale, oy - y * scale, z, min(lit, 1.0), uv[0], uv[1]))
        (ax, ay, az), (bx, by, bz), (cx, cy, cz) = pts[0][:3], pts[1][:3], pts[2][:3]
        area = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
        if area > -1e-9:  # facing away (the screen's y runs down), or edge on
            setup.append(None)
            continue
        number = len(setup)
        inv = 1.0 / area
        # Each corner's weight at a point is a0 + b0*x + c0*y, and so is everything blended by them.
        b0, c0 = (by - cy) * inv, (cx - bx) * inv
        a0 = (bx * cy - by * cx) * inv
        b1, c1 = (cy - ay) * inv, (ax - cx) * inv
        a1 = (cx * ay - cy * ax) * inv
        setup.append((image, pts, a0, b0, c0, a1, b1, c1))
        a2, b2, c2 = 1.0 - a0 - a1, -b0 - b1, -c0 - c1
        dz = az * b0 + bz * b1 + cz * b2
        x_lo = max(0, int(floor(min(ax, bx, cx))))
        x_hi = min(wide - 1, int(ceil(max(ax, bx, cx))))
        y_lo = max(0, int(floor(min(ay, by, cy))))
        y_hi = min(high - 1, int(ceil(max(ay, by, cy))))
        for py in range(y_lo, y_hi + 1):
            mid_y = py + 0.5
            mid_x = x_lo + 0.5
            w0 = a0 + b0 * mid_x + c0 * mid_y
            w1 = a1 + b1 * mid_x + c1 * mid_y
            w2 = a2 + b2 * mid_x + c2 * mid_y
            lo, hi = 0, x_hi - x_lo
            for w, dw in ((w0, b0), (w1, b1), (w2, b2)):  # the run of the row inside all three edges
                if dw > 1e-12:
                    lo = max(lo, int(ceil(-w / dw - 1e-9)))
                elif dw < -1e-12:
                    hi = min(hi, int(floor(-w / dw + 1e-9)))
                elif w < 0.0:
                    hi = -1
            if hi < lo:
                continue
            z = az * (w0 + b0 * lo) + bz * (w1 + b1 * lo) + cz * (w2 + b2 * lo)
            i = py * wide + x_lo + lo
            for _ in range(hi - lo + 1):
                if z > depth[i]:
                    depth[i] = z
                    hit[i] = number
                i += 1
                z += dz
    return hit, setup


def draw(
    figure: Figure,
    camera: Matrix,
    width: int,
    height: int,
    scale: float,
    centre: tuple[float, float],
    samples: int = 4,
    spread: int = 4,
) -> Rows:
    """
    The figure as `height` rows of `width` (r, g, b, a), 0-255.

    `scale` is pixels per model unit and `centre` the point of the camera's view (x, y) that lands
    in the middle of the picture. `spread` is how many pixels past his outline his colour is carried
    into the transparent surround.

    Returns:
        The picture's rows, top first.
    """
    wide = width * samples
    hit, setup = _cover(figure, camera, wide, height * samples, scale * samples, centre)

    # Shade what is left showing, and average each pixel's samples.
    total = samples * samples
    out = []
    for py in range(height):
        row = []
        for px in range(width):
            r = g = b = 0.0
            n = 0
            for sy in range(py * samples, (py + 1) * samples):
                mid_y = sy + 0.5
                i = sy * wide + px * samples
                for sx in range(px * samples, (px + 1) * samples):
                    number = hit[i]
                    i += 1
                    if number < 0:
                        continue
                    image, pts, a0, b0, c0, a1, b1, c1 = setup[number]
                    mid_x = sx + 0.5
                    w0 = a0 + b0 * mid_x + c0 * mid_y
                    w1 = a1 + b1 * mid_x + c1 * mid_y
                    w2 = 1.0 - w0 - w1
                    pa, pb, pc = pts
                    lit = pa[3] * w0 + pb[3] * w1 + pc[3] * w2
                    if len(image) == 3 and not isinstance(image[2], list):
                        col = image
                    else:
                        col = _texel(image, pa[4] * w0 + pb[4] * w1 + pc[4] * w2, pa[5] * w0 + pb[5] * w1 + pc[5] * w2)
                    r += col[0] * lit
                    g += col[1] * lit
                    b += col[2] * lit
                    n += 1
            if n:
                row.append((int(r / n + 0.5), int(g / n + 0.5), int(b / n + 0.5), (n * 255 + total // 2) // total))
            else:
                row.append((0, 0, 0, 0))
        out.append(row)
    return bleed(out, spread)


def bleed(rows: Rows, passes: int) -> Rows:
    """Give transparent pixels the colour of the nearest ones that have any, `passes` pixels deep, in place."""
    h, w = len(rows), len(rows[0])
    known = [[px[3] > 0 for px in row] for row in rows]
    for _ in range(passes):
        fresh = []
        for y in range(h):
            for x in range(w):
                if known[y][x]:
                    continue
                r = g = b = n = 0
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        yy, xx = y + dy, x + dx
                        if 0 <= yy < h and 0 <= xx < w and known[yy][xx]:
                            p = rows[yy][xx]
                            r, g, b, n = r + p[0], g + p[1], b + p[2], n + 1
                if n:
                    fresh.append((y, x, (r // n, g // n, b // n, 0)))
        if not fresh:
            break
        for y, x, px in fresh:
            rows[y][x] = px
            known[y][x] = True
    return rows


def enlarge(rows: Rows, zoom: int) -> Rows:
    """A picture with every pixel repeated `zoom` times each way."""
    out = []
    for row in rows:
        wide = [px for px in row for _ in range(zoom)]
        out += [list(wide) for _ in range(zoom)]
    return out
