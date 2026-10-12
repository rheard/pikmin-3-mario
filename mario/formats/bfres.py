"""
Read a BFRES file, the Switch's model format, and edit its vertices where they lie.

A BFRES file (it starts `FRES`) holds models and the animation clips that play on them. A model
(`FMDL`) is a skeleton (`FSKL`), some shapes (`FSHP`: vertices, a triangle list and the material
that draws them) and its materials (`FMAT`). A clip (`FSKA`) has a track for each bone it moves.

What to know before reading the code:

* Every offset counts from the start of the file, except those of the vertex and index buffers,
  which count from the buffer pool that follows the main part of the file.
* A name is a 16-bit length and then the text. A list of named things comes with a dictionary
  whose entries, after a root, are in the list's order.
* A bone is placed in its parent's space by a scale, three Euler angles and a translation.
* A vertex says which bone it follows with a matrix number: an index into the skeleton's list of
  skinned bones (`Skeleton.matrix_bones`), not a bone's own number.
* A clip finds its bones by name, and gives a bone a translation only where it means to move it.
  In this game's clips that is the root alone, so models of different proportions share them.

`Shape.read` and `Shape.write` turn one attribute of every vertex into floats and back, and
`Shape.indices` and `Shape.set_indices` do the same for the triangle list. Give `Bfres` a
`bytearray` to write. Nothing here moves or resizes anything, so a file edited this way keeps every
offset it had; `mario.formats.bfres_edit` is for adding to one.

The offsets are those of BFRES version 0.8.1.1, the one Pikmin 3 Deluxe uses, and are not checked
against any other.
"""

from __future__ import annotations

import struct

from mario.geometry import Matrix, Vector, euler_matrix

# In place of a struct code: three signed 10-bit fields and a 2-bit one, in 32 bits.
PACKED = "10_10_10_2"
# Vertex attribute formats: code -> (struct code, bytes, divisor). Normalised integers come back as
# floats (0..1 or -1..1); a divisor of None means the numbers are stored as they are.
ATTR_FORMATS: dict[int, tuple[str, int, float | None]] = {
    0x0512: ("<2e", 4, None),  # two half floats
    0x0515: ("<4e", 8, None),  # four half floats
    0x0517: ("<2f", 8, None),
    0x0518: ("<3f", 12, None),
    0x0519: ("<4f", 16, None),
    0x020E: (PACKED, 4, 511.0),  # as normals are stored
    0x020B: ("<4b", 4, 127.0),
    0x010B: ("<4B", 4, 255.0),
    0x0112: ("<2H", 4, 65535.0),
    0x0212: ("<2h", 4, 32767.0),
    0x0109: ("<2B", 2, 255.0),
    0x0309: ("<2B", 2, None),  # plain bytes: the bone indices
}


class Bfres:
    """A BFRES file, read where it lies. `d` is the data it was given: a `bytearray` if anything is to be written."""

    def __init__(self, data: bytes | bytearray) -> None:
        self.d = data
        if data[:4] != b"FRES":
            raise ValueError(f"not a BFRES file: {data[:8]!r}")
        info = self.u64(0x90)
        self.pool = self.u64(info + 8) if info else 0  # where vertex and index buffer offsets count from

    def u8(self, o: int) -> int:
        """The byte at `o`."""
        return self.d[o]

    def u16(self, o: int) -> int:
        """The unsigned 16-bit number at `o`."""
        return struct.unpack_from("<H", self.d, o)[0]

    def s16(self, o: int) -> int:
        """The signed 16-bit number at `o`."""
        return struct.unpack_from("<h", self.d, o)[0]

    def u32(self, o: int) -> int:
        """The unsigned 32-bit number at `o`."""
        return struct.unpack_from("<I", self.d, o)[0]

    def u64(self, o: int) -> int:
        """The unsigned 64-bit number at `o`: every pointer in the file is one."""
        return struct.unpack_from("<Q", self.d, o)[0]

    def f32(self, o: int, n: int = 1) -> tuple[float, ...]:
        """`n` floats starting at `o`."""
        return struct.unpack_from(f"<{n}f", self.d, o)

    def s(self, o: int) -> str | None:
        """The name at `o` (a 16-bit length, then the text), or None for a null offset."""
        if not o:
            return None
        n = self.u16(o)
        return self.d[o + 2 : o + 2 + n].decode("utf-8", "replace")

    def dic(self, o: int) -> list[str | None]:
        """A dictionary's names in the order of the list it belongs to (entry 0 is the root and is skipped)."""
        if not o:
            return []
        n = self.u32(o + 4)
        return [self.s(self.u64(o + 8 + i * 16 + 8)) for i in range(1, n + 1)]

    def models(self) -> list[Model]:
        """Every model, in the file's order."""
        n, off = self.u16(0xBC), self.u64(0x28)
        return [Model(self, off + i * 0x78) for i in range(n)]

    def model(self, name: str) -> Model:
        """The model of that name; KeyError if the file has none."""
        n, off = self.u16(0xBC), self.u64(0x28)
        for i in range(n):
            if self.s(self.u64(off + i * 0x78 + 0x10)) == name:
                return Model(self, off + i * 0x78)
        raise KeyError(name)

    def skel_anims(self) -> list[SkelAnim]:
        """Every skeletal animation clip, in the file's order."""
        n, off = self.u16(0xBE), self.u64(0x38)
        return [SkelAnim(self, off + i * 0x60) for i in range(n)]


class Model:
    """One model (`FMDL`): its skeleton `skel`, its `shapes` and its materials `mats`."""

    def __init__(self, f: Bfres, o: int) -> None:
        if f.d[o : o + 4] != b"FMDL":
            raise ValueError(f"no FMDL at {o:#x}")
        self.name = f.s(f.u64(o + 0x10))
        self.skel = Skeleton(f, f.u64(o + 0x20))
        nshape, nmat = f.u16(o + 0x6A), f.u16(o + 0x6C)
        self.shapes = [Shape(f, f.u64(o + 0x30) + i * 0x70) for i in range(nshape)]
        self.mats = [Material(f, f.u64(o + 0x40) + i * 0xB8) for i in range(nmat)]


class Skeleton:
    """
    A model's bones (`FSKL`).

    `bones` holds a dict for each bone, a parent before its children: `name`, `index`, `parent` (-1
    for the root), `smooth` and `rigid` (its matrix numbers, -1 for none), `flags`, `scale`, `rot`
    (four floats, the first three of them Euler angles), `trans` and `off`, where the bone is in
    the file. A bone's flags are four bytes at `off` + 0x34 and its translation three floats at
    `off` + 0x54.
    """

    def __init__(self, f: Bfres, o: int) -> None:
        if f.d[o : o + 4] != b"FSKL":
            raise ValueError(f"no FSKL at {o:#x}")
        self.off = o
        self.nbone = f.u16(o + 0x4C)
        self.nsmooth = f.u16(o + 0x4E)
        self.nrigid = f.u16(o + 0x50)
        self.flags = f.u32(o + 0x48)  # 0x1000: rotations are Euler angles
        arr = f.u64(o + 0x18)
        self.bones = [_bone(f, arr + i * 0x60) for i in range(self.nbone)]
        # A vertex's bone index is an index into this list (smooth-skinned bones first), not a bone number.
        table = f.u64(o + 0x20)
        self.matrix_bones = [f.u16(table + 2 * i) for i in range(self.nsmooth + self.nrigid)]
        # One 3x4 matrix (12 floats, rows) per smooth-skinned bone: model space -> that bone's space.
        self.inverse_off = f.u64(o + 0x28)

    def world(self) -> list[tuple[Matrix, list[float]]]:
        """Each bone's rest pose in model space as (3x3 rotation rows, position), from the bones' own values."""
        if not self.flags & 0x1000:
            raise ValueError("skeleton does not use Euler rotations")
        out = []
        for b in self.bones:
            r = euler_matrix(*b["rot"][:3])
            r = [[r[i][j] * b["scale"][j] for j in range(3)] for i in range(3)]
            t = list(b["trans"])
            if b["parent"] >= 0:
                pr, pt = out[b["parent"]]
                t = [pt[i] + sum(pr[i][k] * t[k] for k in range(3)) for i in range(3)]
                r = [[sum(pr[i][k] * r[k][j] for k in range(3)) for j in range(3)] for i in range(3)]
            out.append((r, t))
        return out


def _bone(f: Bfres, b: int) -> dict:
    """The bone at `b`, as `Skeleton.bones` holds it."""
    return {
        "name": f.s(f.u64(b)),
        "index": f.u16(b + 0x28),
        "parent": f.s16(b + 0x2A),
        "smooth": f.s16(b + 0x2C),
        "rigid": f.s16(b + 0x2E),
        "flags": f.u32(b + 0x34),
        "scale": f.f32(b + 0x38, 3),
        "rot": f.f32(b + 0x44, 4),
        "trans": f.f32(b + 0x54, 3),
        "off": b,
    }


class Shape:
    """
    One mesh of a model (`FSHP`): vertices, a triangle list, and the number `mat` of its material.

    `attrs` maps a vertex attribute's name to (format code, buffer number, offset inside a vertex):
    `_p0` is the position, `_n0` the normal, `_t0` and `_b0` the tangent and bitangent, `_c0` a
    colour, `_u0` the texture coordinate, `_i0` the matrix numbers of the bones the vertex follows
    and `_w0` their weights. `buffers` lists (file offset, size, stride) for each vertex buffer, and
    `skinbones` the numbers of the bones the shape's vertices follow.
    """

    def __init__(self, f: Bfres, o: int) -> None:
        if f.d[o : o + 4] != b"FSHP":
            raise ValueError(f"no FSHP at {o:#x}")
        self.off = o
        self.name = f.s(f.u64(o + 0x10))
        self.mat = f.u16(o + 0x5E)
        nskin = f.u16(o + 0x64)
        sk = f.u64(o + 0x28)
        self.skinbones = [f.u16(sk + 2 * i) for i in range(nskin)] if sk else []
        self.f = f
        # Bounds: one box (centre, half-size) per sub-mesh and then one for the whole shape, and a radius.
        # For a skinned shape they are measured in bone space: every vertex in the space of the bone it
        # follows, all boxes merged.
        self.bounds_off = f.u64(o + 0x40)
        self.radius_off = f.u64(o + 0x48)
        # The vertex buffer: one attribute per buffer in these files, buffers laid end to end, 8-aligned.
        v = f.u64(o + 0x18)
        self.nvert = f.u32(v + 0x58)
        nattr, nbuf = f.u8(v + 0x54), f.u8(v + 0x55)
        sizes, strides = f.u64(v + 0x38), f.u64(v + 0x40)
        pos = f.pool + f.u32(v + 0x50)
        self.buffers = []  # (file offset, size, stride)
        for i in range(nbuf):
            pos = (pos + 7) & ~7
            self.buffers.append((pos, f.u32(sizes + i * 0x10), f.u32(strides + i * 0x10)))
            pos += self.buffers[-1][1]
        attrs = f.u64(v + 0x10)
        self.attrs = {}  # name -> (format code, buffer number, offset inside a vertex)
        for i in range(nattr):
            a = attrs + i * 0x10
            self.attrs[f.s(f.u64(a))] = (struct.unpack_from(">H", f.d, a + 8)[0], f.u16(a + 0xE), f.u16(a + 0xC))
        # The first mesh (these models have one level of detail and one sub-mesh).
        mesh = f.u64(o + 0x20)
        self.index_off = f.pool + f.u32(mesh + 0x20)
        self.index_format = f.u32(mesh + 0x28)  # 1: 16-bit
        self.nindex = f.u32(mesh + 0x2C)
        self.nsubmesh = f.u16(mesh + 0x34)

    def _attr(self, name: str) -> tuple[str, int, float | None, int, int]:
        """(struct code, bytes, divisor, where the first vertex's value is, stride) of an attribute."""
        fmt, buf, inner = self.attrs[name]
        if fmt not in ATTR_FORMATS:
            raise ValueError(f"{self.name} {name}: vertex format {fmt:#06x} is not handled")
        start, _size, stride = self.buffers[buf]
        return (*ATTR_FORMATS[fmt], start + inner, stride)

    def read(self, name: str) -> list[tuple[float, ...]]:
        """One attribute of every vertex, as tuples of floats."""
        code, _width, div, start, stride = self._attr(name)
        d, out = self.f.d, []
        for i in range(self.nvert):
            p = start + i * stride
            if code == PACKED:
                w = struct.unpack_from("<I", d, p)[0]
                v = [((w >> sh & 0x3FF) ^ 0x200) - 0x200 for sh in (0, 10, 20)]
                out.append((*(max(x / div, -1.0) for x in v), float(w >> 30)))
            else:
                v = struct.unpack_from(code, d, p)
                out.append(tuple(x / div for x in v) if div else tuple(float(x) for x in v))
        return out

    def write(self, name: str, values: list[tuple[float, ...]], first: int = 0) -> None:
        """
        Store one attribute of the vertices from `first` on, from tuples of floats as `read` gives them.

        A value with fewer numbers than the format holds is filled out with ones, as the game's own
        data has it (w = 1).

        Raises:
            ValueError: If the values run past the shape's last vertex.
        """
        code, _width, div, start, stride = self._attr(name)
        if first + len(values) > self.nvert:
            raise ValueError(f"{self.name} {name}: {first + len(values)} vertices do not fit in {self.nvert}")
        d = self.f.d
        n = 4 if code == PACKED else struct.calcsize(code) // struct.calcsize(code[-1])
        for i, value in enumerate(values):
            v = tuple(value) + (1.0,) * (n - len(value))
            p = start + (first + i) * stride
            if code == PACKED:
                x, y, z = (round(max(-1.0, min(1.0, c)) * div) & 0x3FF for c in v[:3])
                struct.pack_into("<I", d, p, x | y << 10 | z << 20 | (int(v[3]) & 3) << 30)
            elif div:
                lo, hi = (-div, div) if code[-1] in "bh" else (0, div)
                struct.pack_into(code, d, p, *(round(max(lo, min(hi, c * div))) for c in v[:n]))
            elif code[-1] == "B":
                struct.pack_into(code, d, p, *(int(c) for c in v[:n]))
            else:
                struct.pack_into(code, d, p, *v[:n])

    def indices(self) -> list[int]:
        """The triangle list: three vertex numbers for each triangle."""
        if self.index_format != 1:
            raise ValueError(f"{self.name}: index format {self.index_format} is not handled")
        return list(struct.unpack_from(f"<{self.nindex}H", self.f.d, self.index_off))

    def set_indices(self, idx: list[int]) -> None:
        """
        Replace the triangle list.

        The list keeps its length: what is left over is filled with triangles that have no area
        (all three corners on vertex 0), which draw nothing.

        Raises:
            ValueError: If there are more indices than the list has room for.
        """
        if len(idx) > self.nindex:
            raise ValueError(f"{self.name}: {len(idx)} indices do not fit in {self.nindex}")
        self.indices()  # checks the format
        padded = list(idx) + [0] * (self.nindex - len(idx))
        struct.pack_into(f"<{self.nindex}H", self.f.d, self.index_off, *padded)

    def bounds(self) -> tuple[tuple[float, ...], tuple[float, ...], float]:
        """(centre, half-size) of the box around the whole shape, and its radius."""
        o = self.bounds_off + self.nsubmesh * 24
        return self.f.f32(o, 3), self.f.f32(o + 12, 3), self.f.f32(self.radius_off)[0]

    def set_bounds(self, centre: Vector | list[float], half: Vector | list[float], radius: float) -> None:
        """Give every sub-mesh and the whole shape one box (centre, half-size), and the shape a radius."""
        for k in range(self.nsubmesh + 1):
            struct.pack_into("<6f", self.f.d, self.bounds_off + k * 24, *centre, *half)
        struct.pack_into("<f", self.f.d, self.radius_off, radius)


class Material:
    """
    A material (`FMAT`): the names of its `textures`, and in `role` which of them plays which part.

    `role` maps a sampler of the shader (`_a0` colour, `_n0` normal map, `_s0` specular mask, `_r0`
    roughness) to a texture's name. It is found the long way round: the shader's sampler, then the
    material's sampler of that name, then the texture in the same position. The order of
    `textures` alone does not say; it differs between leaders.
    """

    def __init__(self, f: Bfres, o: int) -> None:
        if f.d[o : o + 4] != b"FMAT":
            raise ValueError(f"no FMAT at {o:#x}")
        self.name = f.s(f.u64(o + 0x10))
        ntex = f.u8(o + 0xA9)
        tn = f.u64(o + 0x38)
        self.textures = [f.s(f.u64(tn + 8 * i)) for i in range(ntex)] if tn else []
        sa = f.u64(o + 0x28)  # the shader assignment: which of the material's samplers feeds each of the shader's
        slots = f.dic(f.u64(o + 0x50))
        self.role = {}
        if sa:
            given = f.u64(sa + 0x20)
            for i, shader_name in enumerate(f.dic(f.u64(sa + 0x28))[: f.u8(sa + 0x45)]):
                slot = f.s(f.u64(given + 8 * i))
                if slot in slots and slots.index(slot) < len(self.textures):
                    self.role[shader_name] = self.textures[slots.index(slot)]


class SkelAnim:
    """
    One skeletal animation clip (`FSKA`).

    `o` is where the clip is in the file, and `frames` the number of its last frame: it plays from
    frame 0 to that one.
    """

    def __init__(self, f: Bfres, o: int) -> None:
        if f.d[o : o + 4] != b"FSKA":
            raise ValueError(f"no FSKA at {o:#x}")
        self.name = f.s(f.u64(o + 0x10))
        self.f, self.o = f, o
        self.frames = struct.unpack_from("<i", f.d, o + 0x4C)[0]

    def bones(self) -> dict[str, tuple[dict[str, list[float]], list[tuple]]]:
        """
        The clip's track for each bone it moves.

        `base` holds "S", "R" and "T" (scale, rotation, translation) only where the clip sets them:
        a bone the clip gives no translation keeps its model's own. In the leaders' clips that is
        every bone but the root.

        A curve is (target, frames, keys, scale, offset, kind). The target says which number of the
        bone it drives: 4, 8 and 0xc are the scale's x, y and z, 0x10 to 0x18 the translation's and
        0x20 to 0x28 the rotation's. `frames` are the frame numbers of its keys. With t the fraction
        of the way from a key to the next, a key holds the four coefficients of a cubic in t for
        kind 0, a value and how far it goes by the next key for kind 1, and one value otherwise.
        What a key gives is multiplied by `scale` and added to `offset`.

        Returns:
            {bone name: (base, curves)}.
        """
        f, d = self.f, self.f.d
        out = {}
        arr = f.u64(self.o + 0x30)
        for i in range(f.u16(self.o + 0x58)):
            b = arr + i * 0x28
            flags, ncurve = f.u32(b + 0x18), f.u8(b + 0x1E)
            base, p = {}, f.u64(b + 0x10)
            for key, bit, n in (("S", 0x8, 3), ("R", 0x10, 4), ("T", 0x20, 3)):
                if flags & bit:
                    base[key] = list(f.f32(p, n))[:3]
                    p += 4 * n
            curves = []
            for c in range(ncurve):
                co = f.u64(b + 8) + c * 0x30
                cflags, nkey, target = f.u16(co + 0x10), f.u16(co + 0x12), f.u32(co + 0x14)
                scale, offset = f.f32(co + 0x20, 2)
                # frame numbers are stored as floats, as 16-bit counts of 1/32 of a frame, or as bytes
                numbers = struct.unpack_from(f"<{nkey}{'fhB'[cflags & 3]}", d, f.u64(co))
                frames = [x / 32.0 if cflags & 3 == 1 else float(x) for x in numbers]
                kind = cflags >> 4 & 7  # 0 cubic, 1 linear, else one value a key
                per = {0: 4, 1: 2}.get(kind, 1)
                values = struct.unpack_from(f"<{nkey * per}{'fhb'[cflags >> 2 & 3]}", d, f.u64(co + 8))
                keys = [values[k * per : (k + 1) * per] for k in range(nkey)]
                curves.append((target, frames, keys, scale, offset, kind))
            out[f.s(f.u64(b))] = (base, curves)
        return out

    def pose(self, frame: float) -> dict[str, dict[str, list[float]]]:
        """{bone name: {"S" / "R" / "T": [x, y, z]}} at a frame, holding only what the clip sets."""
        out = {}
        for name, (base, curves) in self.bones().items():
            p = {k: list(v) for k, v in base.items()}
            for target, frames, keys, scale, offset, kind in curves:
                if frame <= frames[0]:
                    i, t = 0, 0.0
                elif frame >= frames[-1]:
                    i, t = len(frames) - 1, 0.0
                else:
                    i = max(k for k in range(len(frames)) if frames[k] <= frame)
                    t = (frame - frames[i]) / (frames[i + 1] - frames[i])
                k = keys[i]
                if kind == 0:
                    v = ((k[3] * t + k[2]) * t + k[1]) * t + k[0]
                elif kind == 1:
                    v = k[0] + k[1] * t
                else:
                    v = k[0]
                group = "S" if target < 0x10 else "T" if target < 0x20 else "R"
                first = {"S": 4, "T": 0x10, "R": 0x20}[group]
                axis = (target - first) // 4
                if axis < 3:
                    p.setdefault(group, [1.0] * 3 if group == "S" else [0.0] * 3)[axis] = v * scale + offset
            out[name] = p
        return out
