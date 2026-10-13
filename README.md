# Mario in Pikmin 3

This builds a mod for **Pikmin 3 Deluxe** that turns Captain Alph into Mario: the Super Mario 64
one, polygon for polygon, with his own walk, his own voice, his punch-punch-kick and his triple jump.
He throws Pikmin with a "wah!", and he can still settle things with a Dwarf Bulborb the old way, by
landing on it.

**There is no Nintendo code or art in this repository.** It does not contain the mod. It contains
the program that *makes* the mod, out of two things only you can supply: your own Super Mario 64
ROM and the files of your own copy of Pikmin 3 Deluxe. Everything of Mario's is read out of your
ROM while it runs.

## What he does

- **He's Mario.** The model and its textures come straight from the ROM.
- **He moves like Mario.** His walk and run are the 1996 animations, knees and elbows included.
- **He stands like Mario.** Left alone he takes his own stance, knees and elbows bent, and looks to
  his left, to his right and ahead again.
- **He sounds like Mario.** "Wah!" on a throw, "yah, wah, hoo!" on the punches, "yahoo!" on the
  triple jump.
- **He's on the HUD.** Every picture of Alph in the game, from the leader button to the map marker
  to the menus, is painted again as Mario. All 140 of them.
- **Punch, punch, kick.** With no Pikmin following, A beside an enemy starts a punch. Press again
  for the second, again for the kick.
- **Jump, double jump, triple jump.** D-pad Down, or click the right stick. Press again as he lands
  to go higher; the third needs a run-up, exactly as it did on the N64.
- **Stomp.** Come down on an enemy and it takes a hit twice as hard as a punch, and he bounces off.
  The small ones are squashed outright.

Everybody else is left alone. Brittany, Charlie and the Pikmin share Alph's animation clips and
his sound tables, and none of them moves or sounds any different.

## Build it

You need Python 3.10 or newer and nothing else: no packages to install.

1. **A Super Mario 64 ROM**, dumped from a cartridge you own. Any of the three usual byte orders
   (`.z64`, `.n64`, `.v64`). The US release is the one this was made with.
2. **Pikmin 3 Deluxe's RomFS**, extracted from your own copy. His moves are patched into the game's
   program, which must be the one version 1.2.0 has (build ID `D467F5AD367BBEE8...`); a loader will
   not apply the patch to another.
3. Run it:

   ```bash
   git clone https://github.com/rheard/pikmin-3-mario
   cd pikmin-3-mario
   python -m mario "Super Mario 64.z64" path/to/romfs
   ```

   It takes a few minutes, most of them spent compressing one large file and rendering the menu
   pictures, and writes a folder named `Mario`.
4. Copy the `Mario` folder into the game's mod folder and start the game.

## How it works, and how it was made

The interesting part of a mod like this is everything the two games disagree about. Here are the
steps this mod was made in, and the challenges found along the way.

### Model swap

The first step was a simple model swap to get rid of Alph and bring in Mario.

**A cartridge is not a model file.** Super Mario 64 has no file called `mario.obj`. Mario is a
script that draws twenty parts with display lists, each relying on colours the list before it left
set, and he has no rest pose at all: only animations. So the reader follows the game's own level
script to his layout, runs the display lists the way the console would, and builds a T-pose out of
his joints for the new skeleton to bind to.

**Fitting him into Alph's place.** Alph's joints are moved to Mario's hips, neck and shoulders, and
his body and head are overwritten with Mario's triangles. Mario's handful of small textures and
flat colours are packed together and encoded over Alph's own textures. Everything else of Alph's
(helmet, backpack, antenna) is emptied, and the light on the tip of the antenna is put out: with
the antenna gone it floated over Mario's head.

### Giving Mario his N64 walk

With the model in, the next step was his own walk and run in place of Alph's.

**The people of Koppai have no knees.** A leader's arm is one bone. So is a leg. They bend like
rubber hoses, and Mario very much does not: his animations want an elbow and a knee. He needed
fifteen bones the game has never heard of, and this kind of model file was never meant to grow.
Every pointer in it is listed in a table at the end, names are looked up through a bit-by-bit
search tree, and the game trusts all of it. `formats/bfres_edit.py` grows the file anyway, and the
tests hold it to account by rebuilding every one of the file's own name tables byte for byte.

**Everyone shares the walk.** The clip Alph walks with is the clip every Pikmin walks with, so
replacing it was never an option. The way round it: a clip finds its bones *by name*, and skips
names a skeleton doesn't have. So Mario's fifteen bones are named like nobody else's, and the walk
gains fifteen tracks only he can hear. Alph's rubber-hose walk still plays underneath; Mario's
bones cancel it exactly and put 1996 on top. The price is that all 129 clips in the file need a
track telling each new bone to stay put, or his legs would freeze wherever his last walk left them.

### Adding Mario's throw noise

The next step was to make sure we could import audio by adding Mario's throw noise for when he
is throwing Pikmin.

**His voice had to be his alone.** The game picks a sound by who is making it, so each of his lines
is a family of sounds with Mario in Alph's slot and sixteen milliseconds of silence in everyone
else's. The samples come out of the ROM's sound bank through a small re-creation of the N64's
sequence player. "Yahoo!" is not one recording, it turns out: it is one sample played as two notes
with a slide between them.

### Replacing Alph in the HUD

Then came replacing all the instances of Alph in the HUD. This was more of a side project,
and required rendering the exported Mario model in various poses, since Super Mario 64 does not
have all the same HUD icon poses that Pikmin 3 does.

**The HUD is 140 little paintings.** No leader's picture is drawn from his model, so there is a
software renderer in here that shades Mario the way his own game does and paints every one of them
again at the right size, expression and angle. The picture on the D-pad menu is the 16-pixel head
from Super Mario 64's own HUD.

### Adding Mario's punch-punch-kick mechanic

While punching doesn't come up much in Pikmin 3, this seemed like it would serve as a good test
before implementing triple jumping.

**A leader can already punch.** With no Pikmin following, pressing A beside an enemy throws one
punch. The game's own routine is left to find the enemy and land the hit; the new code changes
which clip it plays, lets a second and a third press of A carry on into the next punch and the
kick, and makes his hits land half again as hard as a leader's and reach further. His fist and
foot swell as he strikes, as they do on the N64. How far is a table buried in Super Mario 64's
code, which the reader finds by following the MIPS instructions of the function that uses it.

**There was nowhere to put new code.** A patch can change the game's program but cannot make it
longer. Luckily this is a Wii U port, and code that served the GamePad is still in there behind
checks that can never be true again. The punches' hundred or so words of ARM went there, written by
hand and put together by an assembler written for the purpose.

### Finally, the coup de grâce: Mario's triple jump

This was by far the most complex task and required adding an entirely new mechanic to Pikmin:
jumping. It also built on all the others since that obviously required adding new animations
and sounds.

**Pikmin has no jump button, because Pikmin has no jump.** But a leader is a body in the game's
physics, with gravity and a "touching the ground" flag. Give that body an upward speed and it
jumps. The jump lives inside the game's punch routine with different answers to its questions:
which clip to play (a jump), when to look for something to hit (whenever he is coming down), where
to look (under him), and when to stop (when he lands). Heights are Super Mario 64's own, scaled to
his size here: the triple jump is nearly four times his height.

**A button nobody was using.** The punches stay on A, so the jump needed a button of its own.
D-pad Down is the one button the game maps to nothing, so that is the jump, along with a click of
the right stick.

**Landing on enemies.** The game already has a hit for something landing on an enemy from above:
it is what a thrown Pikmin's landing does, and each enemy's own numbers say whether that squashes
it. Mario's landing sends that hit first, which is what kills a Dwarf Bulborb outright. Anything it
does nothing to takes an ordinary blow twice as strong as his punch, and he bounces off.

The jump needed more room than the GamePad's leftovers had, and found it in the functions of two
kinds of mission rule that no mission in the game uses. All told, Mario's moves are 441 words of
ARM, reached through ten changed instructions of the original game.

### Standing still

With the moves done, the next step was the smallest thing he does: standing still.

**His stance is longer than theirs.** Every leader stands with one short clip, a loop of nineteen
frames, and every Pikmin stands with it too. Mario's standing still is three animations, ninety
frames in all: he looks to his left, to his right and ahead again, and each ends where the next
begins. That cannot ride inside the shared clip the way his walk does, because the clip is not
long enough to hold it.

**So the game's list is pointed at a clip of his own.** The game plays a clip by its place in a
list, and the entry a leader stands with is a line of its own in that list. That one line now names
a new clip. Every leader plays that entry, so the new clip holds the old stand for their bones,
played round five times, and Mario's ninety frames for his. Five rounds of nineteen frames is
ninety-five, so he stands a twentieth slower than he did on the N64, and nobody else stands any
differently.

## What's in here

| | |
|---|---|
| `mario/sm64/` | Reads the ROM: the model and animations, the voice, and a renderer for the pictures |
| `mario/formats/` | The Switch's file formats: archives, models, textures, sounds, effects |
| `mario/model.py` | The skeleton, the mesh, the clips |
| `mario/voice.py` | What he says, and when |
| `mario/icons.py` | The HUD and menu pictures |
| `mario/moves.py`, `mario/asm/` | The new code: the punches and the jumps, in ARM assembly |
| `mario/arm.py` | The assembler |
| `tests/` | `pytest`. Most run anywhere; set `SM64_ROM` and `PIKMIN3_ROMFS` to also build the mod and read it back the way the game will |

## The small print

This is a fan project. It is not made, endorsed or supported by Nintendo, and Mario, Pikmin and
everything about them belong to Nintendo. The code here is original and is offered under the MIT
licence; it contains nothing taken from either game. You need your own legally obtained copies of
both games to use it, and nobody here can help you get them.

## Thanks

To the people who took Super Mario 64 apart and wrote down how it works, and to everyone who has
documented Nintendo's file formats over the years. This would have been a much longer story
without them.
