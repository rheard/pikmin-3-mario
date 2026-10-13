"""
The built mod, read back the way the game will read it.

These need both games (see `conftest.py`) and are skipped without them.
"""

import struct

import pytest

from mario import icons, model, moves, voice
from mario.formats import bfres_edit, bfsar, bntx, sarc, vfxb, yaz0
from mario.formats.bfres import Bfres, Skeleton
from mario.geometry import euler_matrix, matmul, to_model
from mario.romfs import RomFs
from mario.sm64.audio import Voice
from mario.sm64.rom import Rom

LEADER_MODEL = "player" + model.LEADER
WRITTEN = {
    model.ARCHIVE,
    model.SETTINGS,
    model.EFFECTS,
    voice.ARCHIVE,
    voice.GROUP,
    voice.COMMANDS,
    *(f"lyt/{name}.szs" for name in ("common", "missionselect", "parts", "saveselect", "substoryselect")),
    *(f"lyt/{name}.szs" for name in ("vsselect", "worldranking")),
}
NEAR = 0.02  # how far, in the game's units, a part may be from where his animation puts it


def inside(files: RomFs, archive: str, name: str) -> bytes:
    """One file out of one of the game's archives."""
    return sarc.Sarc(yaz0.decompress(files.read(archive))).files()[name]


def played(skel: Skeleton, pose: dict) -> list:
    """Where every bone is under a clip's pose: its rotation, scaled, and its origin."""
    out = []
    for bone in skel.bones:
        said = pose.get(bone["name"], {})
        rotation = euler_matrix(*said.get("R", bone["rot"])[:3])
        scale = said.get("S", bone["scale"])
        rotation = [[rotation[i][j] * scale[j] for j in range(3)] for i in range(3)]
        place = said.get("T", bone["trans"])
        if bone["parent"] >= 0:
            parent_rotation, parent_origin = out[bone["parent"]]
            out.append((matmul(parent_rotation, rotation), to_model(parent_rotation, parent_origin, place)))
        else:
            out.append((rotation, tuple(place)))
    return out


def stored_dict(f: Bfres, offset: int, count: int) -> list[tuple]:
    """The nodes of a name dictionary as the file holds them."""
    return [struct.unpack_from("<IHH", f.d, offset + 8 + i * 16) for i in range(count + 1)]


@pytest.fixture(scope="module")
def stock(romfs: RomFs) -> Bfres:
    """The game's own model file."""
    return Bfres(inside(romfs, model.ARCHIVE, model.MODEL_FILE))


@pytest.fixture(scope="module")
def built(mod: RomFs) -> Bfres:
    """The mod's model file."""
    return Bfres(inside(mod, model.ARCHIVE, model.MODEL_FILE))


@pytest.fixture(scope="module")
def char(rom: Rom) -> model.Character:
    """Mario, stood in the game's units."""
    return model.Character(rom, model.HEIGHT)


def test_the_mod_is_these_files(mod: RomFs):
    """The mod replaces fourteen of the game's files and holds nothing else."""
    files = {file.relative_to(mod.root).as_posix() for file in mod.root.rglob("*") if file.is_file()}
    assert files == WRITTEN
    patches = [file.name for file in (mod.root.parent.parent / "exefs").iterdir()]
    assert patches == [f"{moves.BUILD_ID}.ips"]


# -- the model -------------------------------------------------------------------------------------


@pytest.mark.parametrize("which", ["stock", "built"])
def test_name_tables_are_the_ones_their_names_build(which: str, request: pytest.FixtureRequest):
    """Every name dictionary in the file, the game's own included, is what `build_dict` makes of its names."""
    f = request.getfixturevalue(which)
    tables = [(f.u64(0x40), [clip.name for clip in f.skel_anims()])]
    tables += [(f.u64(m.skel.off + 0x10), [bone["name"] for bone in m.skel.bones]) for m in f.models()]
    for offset, names in tables:
        assert f.dic(offset) == names
        assert stored_dict(f, offset, len(names)) == bfres_edit.build_dict(names)
        assert [bfres_edit.dict_find(f, offset, name) for name in names] == list(range(len(names)))
        assert bfres_edit.dict_find(f, offset, "no_such_name") == -1


def test_the_grown_file_holds_together(built: Bfres):
    """Every pointer of the built file is listed for the game to fix up, and lies inside its section."""
    assert bfres_edit.check(built.d) > 0
    assert bfres_edit.unlisted(built.d) == []


def test_only_one_model_changes(stock: Bfres, built: Bfres):
    """Every other leader and every Pikmin keeps the skeleton it had."""
    assert [m.name for m in built.models()] == [m.name for m in stock.models()]
    for m in stock.models():
        if m.name != LEADER_MODEL:
            assert built.model(m.name).skel.bones == m.skel.bones


def test_his_bones_hang_where_his_parts_do(stock: Bfres, built: Bfres, char: model.Character):
    """The leader's bones keep their order and sit on Mario's joints; each of his parts has a bone of its own."""
    old, new = stock.model(LEADER_MODEL).skel, built.model(LEADER_MODEL).skel
    names = [bone["name"] for bone in new.bones]
    assert names[: old.nbone] == [bone["name"] for bone in old.bones]
    assert names[old.nbone :] == [
        model.OWN + part for part in sorted(char.frames, key=lambda p: (names.index(char.stock[p]), p))
    ]
    world = new.world()
    for bone, pivot in char.pivot.items():
        assert world[names.index(bone)][1] == pytest.approx(pivot, abs=1e-4)
    for part, (_, origin) in char.frames.items():
        bone = names.index(model.OWN + part)
        assert new.bones[bone]["parent"] == names.index(char.stock[part])
        assert world[bone][1] == pytest.approx(origin, abs=1e-3)


def test_every_clip_keeps_what_it_had(stock: Bfres, built: Bfres):
    """The game's clips play as they did for everyone else, and hold his bones at rest unless they are his."""
    before = {clip.name: clip for clip in stock.skel_anims()}
    after = {clip.name: clip for clip in built.skel_anims()}
    assert list(after) == [*before, *model.MOVES, *model.STANDS]
    skel = built.model(LEADER_MODEL).skel
    own = {bone["name"]: bone for bone in skel.bones if bone["name"].startswith(model.OWN)}
    for name, clip in before.items():
        tracks = after[name].bones()
        assert after[name].frames == clip.frames
        assert all(tracks[bone] == track for bone, track in clip.bones().items())
        assert set(tracks) - set(clip.bones()) == set(own)
        if name not in model.CLIPS:
            for bone, (base, curves) in ((bone, tracks[bone]) for bone in own):
                assert not curves
                assert tuple(base["R"]) == own[bone]["rot"][:3]
                assert tuple(base["T"]) == own[bone]["trans"]


def test_his_walk_and_run_play_back(stock: Bfres, built: Bfres, char: model.Character, romfs: RomFs):
    """Played from the built file, the leaders' walk and run put every part where his own animations do."""
    loops = model.clip_loops(inside(romfs, model.SETTINGS, model.CLIP_LIST))
    skel0 = stock.model(LEADER_MODEL).skel
    _, plan = model.plan_skeleton(skel0, char)
    skel = built.model(LEADER_MODEL).skel
    names = [bone["name"] for bone in skel.bones]
    clips = {clip.name: clip for clip in built.skel_anims()}
    for name, (which, lean) in model.CLIPS.items():
        clip = next(c for c in stock.skel_anims() if c.name == name)
        _, targets = model.clip_targets(char, clip, loops[name], which, lean, skel0, plan)
        for frame, row in enumerate(targets):
            world = played(skel, clips[name].pose(float(frame)))
            for part, (_, _, _, want) in row.items():
                assert world[names.index(model.OWN + part)][1] == pytest.approx(want, abs=NEAR)


def test_his_moves_play_back(stock: Bfres, built: Bfres, char: model.Character):
    """Each new clip is as long as its pieces and, played back, is his animation to the frame."""
    skel0 = stock.model(LEADER_MODEL).skel
    _, plan = model.plan_skeleton(skel0, char)
    skel = built.model(LEADER_MODEL).skel
    names = [bone["name"] for bone in skel.bones]
    clips = {clip.name: clip for clip in built.skel_anims()}
    for name, move in model.MOVES.items():
        _, targets = model.move_targets(char, move, skel0, plan)
        sizes = model.move_swell(char.mario, move, len(targets))
        assert clips[name].frames == len(model.move_steps(move)) - 1
        assert list(clips[name].bones()) == names
        for frame, row in enumerate(targets):
            world = played(skel, clips[name].pose(float(frame)))
            for part, (_, _, _, want) in row.items():
                rotation, origin = world[names.index(model.OWN + part)]
                size = sizes[frame] if move.swell and part == move.swell[0] else 1.0
                assert origin == pytest.approx(want, abs=NEAR)
                assert sum(rotation[r][0] ** 2 for r in range(3)) ** 0.5 == pytest.approx(size, abs=1e-4)


def test_his_standing_still_plays_back(stock: Bfres, built: Bfres, char: model.Character, romfs: RomFs):
    """His standing clip is the leaders' stand, round and round, for their bones, and his own for his."""
    clip_list = inside(romfs, model.SETTINGS, model.CLIP_LIST)
    skel0 = stock.model(LEADER_MODEL).skel
    _, plan = model.plan_skeleton(skel0, char)
    skel = built.model(LEADER_MODEL).skel
    names = [bone["name"] for bone in skel.bones]
    clips = {clip.name: clip for clip in built.skel_anims()}
    for name, stand in model.STANDS.items():
        theirs = next(c for c in stock.skel_anims() if c.name == stand.like)
        period = model.stand_loop(clip_list, stand)
        poses, targets = model.stand_targets(char, stand, theirs, period, skel0, plan)
        clip = clips[name]
        assert clip.frames == len(targets) - 1
        assert clip.frames % period == 0  # a whole number of rounds of theirs, so both come round together
        assert built.u32(clip.o + 0x48) == stock.u32(theirs.o + 0x48)  # and it plays round and round, as theirs does
        assert set(clip.bones()) == set(theirs.bones()) | {model.OWN + part for part in char.frames}
        for frame, (pose, row) in enumerate(zip(poses, targets, strict=True)):
            said = clip.pose(float(frame))
            for bone, values in pose.items():  # what the stock clip says on this frame of its loop
                for key, value in values.items():
                    assert said[bone][key] == pytest.approx(value, abs=1e-4)
            world = played(skel, said)
            for part, (_, _, _, want) in row.items():
                assert world[names.index(model.OWN + part)][1] == pytest.approx(want, abs=NEAR)
        first, last = clip.pose(0.0), clip.pose(float(clip.frames))
        for bone, values in first.items():  # the last frame is the first again
            for key, value in values.items():
                assert last[bone][key] == pytest.approx(value, abs=1e-4)


def test_the_mesh_is_his(built: Bfres, char: model.Character):
    """Read back, the two shapes are his triangles; every other shape of the leader's is empty."""
    leader = built.model(LEADER_MODEL)
    body, head = model.body_and_head(leader)
    for shape, tris in zip((body, head), model.split(char), strict=True):
        positions, indices = shape.read("_p0"), shape.indices()
        assert not any(indices[len(tris) * 3 :])
        for t, tri in enumerate(tris):
            for corner, (want, *_) in enumerate(tri):
                assert positions[indices[t * 3 + corner]][:3] == pytest.approx(want, abs=NEAR)
    for shape in leader.shapes:
        if shape is not body and shape is not head:
            assert not any(shape.indices())


def test_the_clip_list_gains_his_clips(romfs: RomFs, mod: RomFs, char: model.Character):
    """The game's list is the stock one with the standing entry playing his clip and his moves on the end."""
    old, new = inside(romfs, model.SETTINGS, model.CLIP_LIST), inside(mod, model.SETTINGS, model.CLIP_LIST)
    stood = model.define_stands(old, char.mario)
    assert model.define_moves(stood, model.MOVES)[0] == new
    # Pointing an entry at his clip changes two lines of it, the clip's name and where its loop ends, and
    # moves nothing: every clip the game plays by its place in the list is where it was.
    old_lines, stood_lines = old.split(b"\r\n"), stood.split(b"\r\n")
    assert len(stood_lines) == len(old_lines)
    assert sum(a != b for a, b in zip(old_lines, stood_lines, strict=True)) == 2 * len(model.STANDS)
    assert [tag for tag, _, _ in model.clip_entries(stood)] == [tag for tag, _, _ in model.clip_entries(old)]
    for name, stand in model.STANDS.items():
        length = model.stand_length(char.mario, stand, model.stand_loop(old, stand))
        entry = next(entry for entry in model.clip_entries(new) if entry[0] == stand.tag)
        assert entry == (stand.tag, name, [(0.0, model.LOOP_START), (float(length), model.LOOP_END)])
    body = stood.partition(b"\r\n{\r\n")[2].partition(b"\r\n}\r\n")[0]
    assert new.partition(b"\r\n{\r\n")[2].startswith(body)
    assert new.partition(b"\r\n}\r\n")[2] == old.partition(b"\r\n}\r\n")[2]


def test_only_his_lights_go_out(romfs: RomFs, mod: RomFs):
    """The effect file changes in the leader's antenna lights and nowhere else."""
    old, new = inside(romfs, model.EFFECTS, model.EFFECT_FILE), inside(mod, model.EFFECTS, model.EFFECT_FILE)
    assert len(new) == len(old)
    lights = {name.format(model.LEADER) for name in model.LIGHTS}
    spans = []
    for (_, name, emitters), (_, name1, emitters1) in zip(
        vfxb.emitter_blocks(old),
        vfxb.emitter_blocks(new),
        strict=True,
    ):
        assert (name, emitters) == (name1, emitters1)
        if name in lights:
            spans += [vfxb.hide_emitter(bytearray(old), block) for _, block in emitters]
    assert len(spans) >= len(lights)
    changed = [i for i in range(len(old)) if old[i] != new[i]]
    assert changed
    assert all(any(lo <= i < hi for lo, hi in spans) for i in changed)


# -- the voice -------------------------------------------------------------------------------------


def wave_of(archive: bfsar.Archive, group: bfsar.Group, sound: int) -> dict:
    """Follow a sound to its samples the way the sound library does."""
    info = archive.sound(sound)
    item, number, _ = bfsar.read_wave_sounds(group.file(info["file"]))[info["index"]]
    holder = archive.wave_archive(item & 0xFFFFFF)
    at = next(group.body + offset for file, offset, _ in group.items if file == holder)
    offset, _ = bfsar.read_wave_archive(group.d, at)[number]
    return bfsar.read_wave(group.d, offset)


def test_nothing_the_game_had_is_changed(romfs: RomFs, mod: RomFs):
    """Every stock sound, file and wave archive entry is still there, as it was."""
    old, new = bfsar.Archive(romfs.read(voice.ARCHIVE)), bfsar.Archive(mod.read(voice.ARCHIVE))
    for table in (bfsar.SOUNDS, bfsar.FILES, bfsar.WAVE_ARCHIVES):
        assert new.count(table) > old.count(table)
        for i in range(old.count(table)):
            was = old.raw(table, i)
            assert new.raw(table, i)[: len(was)] == was
    before, after = bfsar.Group(romfs.read(voice.GROUP)), bfsar.Group(mod.read(voice.GROUP))
    assert after.items[: len(before.items)] == before.items
    assert len(after.items) == len(before.items) + 2


def test_he_speaks_and_nobody_else_does(rom: Rom, romfs: RomFs, mod: RomFs):
    """Each line's record leads the leader to the sounds made for it, and everyone else to silence."""
    said = voice.lines(Voice(rom))
    archive, group = bfsar.Archive(mod.read(voice.ARCHIVE)), bfsar.Group(mod.read(voice.GROUP))
    old_table, table = (inside(files, voice.COMMANDS, voice.TABLE) for files in (romfs, mod))
    count, _, _, names = voice.table_parts(table)
    assert count == voice.table_parts(old_table)[0] + len(said)
    assert names[-len(said) :] == [line.name.encode() for line in said]
    records = table.split(b"\t# (OBJ)")[-len(said) :]
    slots = bfsar.TYPE_SLOTS[voice.KIND]
    for line, record in zip(said, records, strict=True):
        assert b'# anim tag\r\n\t\t\t"' + line.clip.encode() + b'"' in record
        assert b"# frame\r\n\t\t\t%d" % line.frame in record
        base = int(record.split(b"# sound id\r\n\t\t\t")[1].split()[0], 16) & 0xFFFFFF
        for who, slot in slots.items():
            chosen = voice.picks(archive, base, slot)
            if who == voice.LEADER:
                assert len(chosen) == len(line.takes)
                for (sound, weight), take in zip(chosen, line.takes, strict=True):
                    wave = wave_of(archive, group, sound)
                    assert (wave["samples"], wave["rate"]) == (list(take.samples), take.rate)
                    assert weight == (take.weight if len(line.takes) > 1 else 16)
            else:
                for sound, _ in chosen:
                    assert archive.sound(sound)["volume"] == 0
                    assert not any(wave_of(archive, group, sound)["samples"])


# -- the pictures ----------------------------------------------------------------------------------


def test_the_pictures_are_redrawn_in_place(rom: Rom, romfs: RomFs, mod: RomFs):
    """Each layout archive changes inside the leader's textures and nowhere else, and they decode as drawn."""
    artist = icons.Artist(rom, samples=1)  # one point a pixel: enough to tell whose picture it is
    for name in ("common", "saveselect"):
        file = f"{icons.LAYOUTS}/{name}.szs"
        old = icons.Layout(name, romfs.read(file))
        new = icons.Layout(name, mod.read(file))
        assert new.raw[:16] == old.raw[:16]
        assert len(new.plain) == len(old.plain)
        planned = icons.plan(old)
        assert planned
        spans = [(old.where[0] + t.offsets[0], old.where[0] + t.offsets[0] + t.size) for t, *_ in planned]
        at = 0
        for lo, hi in sorted(spans):
            assert new.plain[at:lo] == old.plain[at:lo]
            at = hi
        assert new.plain[at:] == old.plain[at:]
        data = new.plain[new.where[0] : new.where[1]]
        for texture, kind, number, count in planned:
            drawn = icons.encoded(texture, artist.picture(kind, number, texture.width, texture.height, count))
            got = bntx.decode(data, new.textures[texture.name])
            coverage = 1 if texture.codec == "BC5" else 3
            off = [
                abs(a[coverage] - b[coverage])
                for ra, rb in zip(drawn, got, strict=True)
                for a, b in zip(ra, rb, strict=True)
            ]
            assert sum(off) / len(off) < 6  # the same silhouette, to within the fewer points per pixel


# -- the code --------------------------------------------------------------------------------------


def test_the_patch_is_the_program(romfs: RomFs, mod: RomFs):
    """The patch holds the code assembled for the clip numbers the built list gives his moves."""
    first = model.define_moves(inside(romfs, model.SETTINGS, model.CLIP_LIST), model.MOVES)[1]
    patch = (mod.root.parent.parent / "exefs" / f"{moves.BUILD_ID}.ips").read_bytes()
    assert moves.read_ips(patch) == moves.program(first, model.HEIGHT)
