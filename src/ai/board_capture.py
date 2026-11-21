"""Blackboard pages from sampled lecture frames (fixed camera, green boards).

Chalk on a board only accumulates until it is wiped, so one image taken just
before each wipe holds everything written since the previous one.  Boards in
a lecture hall slide up and down and pass behind each other, so chalk is
tracked on sheets: one per physical board, in coordinates that move with it.

  1. per frame: a board mask and a chalk mask, reduced to 4x4-px blocks, and
     the visible panels (columns of board, split at the frames between them);
  2. each panel is matched to a sheet: the one already on it while its chalk
     is still there, else a sheet whose chalk fits at some vertical offset
     (the board was slid), else a new sheet;
  3. per sheet and block: a debounced "chalk present" state.  Blocks hidden
     by the teacher, the projection screen or another board keep their state;
  4. a sheet yields a page when chalk leaves it - wiped in place, or the
     sheet is never recognised again - and one for what is left at the end.
     A page is composed per tile from the frames that show most of the
     tile's chalk with nothing in front of it;
  5. a page whose chalk is all on another page is absorbed by it.

Nothing is calibrated to a room.  Frames are not kept in memory;
``load_frame(i)`` must return sample ``i`` again when a page is composed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

BLOCK = 4                 # px per state block
TILE = 8                  # blocks per composition tile
HUE = (67, 147)           # PIL hue units, 95-208 degrees: dim green to lit teal
SAT_MIN = 21
VAL = (31, 214)
INK_LO, INK_HI = 5, 9     # gray levels above the local background
BRD, LO, HI = 1, 2, 4
PAD = 140                 # blocks a sheet may travel from where it was first seen
MIN_REG = 30              # chalk blocks needed to recognise a sheet


def frame_features(rgb: np.ndarray) -> np.ndarray:
    """Block map with bits BRD (board), LO / HI (weak / strong chalk)."""
    img = Image.fromarray(rgb)
    hsv = np.asarray(img.convert("HSV"))
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    green = ((h >= HUE[0]) & (h <= HUE[1]) & (s >= SAT_MIN)
             & (v >= VAL[0]) & (v <= VAL[1]))
    # Chalk strokes are not green.  Closing fills gaps under 5 px; the frames
    # between panels are wider and stay open.
    closed = (Image.fromarray(green.astype(np.uint8) * 255)
              .filter(ImageFilter.MaxFilter(5)).filter(ImageFilter.MinFilter(5)))
    board = np.asarray(closed) > 127
    gray = img.convert("L")
    diff = (np.asarray(gray, dtype=np.int16)
            - np.asarray(gray.filter(ImageFilter.BoxBlur(6)), dtype=np.int16))
    hb, wb = rgb.shape[0] // BLOCK, rgb.shape[1] // BLOCK

    def blocks(mask, need):
        return (mask[:hb * BLOCK, :wb * BLOCK]
                .reshape(hb, BLOCK, wb, BLOCK).sum(axis=(1, 3)) >= need)

    return (blocks(board, BLOCK * BLOCK - 1) * BRD
            + blocks((diff > INK_LO) & board, 2) * LO
            + blocks((diff > INK_HI) & board, 3) * HI).astype(np.uint8)


def _morph(mask: np.ndarray, size: int, grow: bool) -> np.ndarray:
    kind = ImageFilter.MaxFilter if grow else ImageFilter.MinFilter
    return np.asarray(Image.fromarray(mask.astype(np.uint8) * 255)
                      .filter(kind(size))) > 127


def _tile_sum(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask, ((0, -mask.shape[0] % TILE), (0, -mask.shape[1] % TILE)))
    return padded.reshape(padded.shape[0] // TILE, TILE, -1, TILE).sum(axis=(1, 3))


def _runs(flags: np.ndarray) -> list[tuple[int, int]]:
    """Half-open [start, stop) runs of True."""
    edges = np.flatnonzero(np.diff(np.concatenate(([0], flags.astype(np.int8), [0]))))
    return list(zip(edges[::2].tolist(), edges[1::2].tolist()))


def panels(board: np.ndarray, columns: np.ndarray, ink: np.ndarray,
           min_w: int = 24, min_h: int = 12) -> list[tuple]:
    """Visible panels (x0, y0, x1, y1): columns of board, split at the frames between panels.

    `columns` is the board mask of the last minute or so: someone standing in
    front of a panel must not split its column in two.  The frame between two
    panels can pass for board in part when it reflects one; it is then a
    bright line across a row that is not all board, which a line of
    handwriting never is.
    """
    out = []
    for x0, x1 in _runs(columns.sum(axis=0) >= min_h):
        if x1 - x0 < min_w:
            continue
        share = board[:, x0:x1].mean(axis=1)
        inside = (share >= 0.5) & ~((share < 0.9) & (ink[:, x0:x1].mean(axis=1) >= 0.5))
        for y0, y1 in _runs(inside):
            if y1 - y0 >= min_h and board[y0:y1, x0:x1].mean() >= 0.5:
                out.append((x0, y0, x1, y1))
    return out


def _overlap(a0, a1, b0, b1) -> float:
    """Shared length of two intervals as a share of the shorter one."""
    return max(0, min(a1, b1) - max(a0, b0)) / max(1, min(a1 - a0, b1 - b0))


class _Sheet:
    """Chalk on one physical board, in rows that move with the board."""

    def __init__(self, shape):
        self.rows = shape[0] + 2 * PAD
        self.state = np.zeros((self.rows, shape[1]), bool)
        self.cnt = np.zeros((self.rows, shape[1]), np.uint8)
        self.last = np.zeros((self.rows, shape[1]), np.int32)   # sample the chalk was last seen
        self.born = np.zeros((self.rows, shape[1]), np.int32)   # sample it appeared
        self.offset = -PAD            # frame row of sheet row 0
        self.pos: dict[int, int] = {}
        self.vis: dict[int, tuple] = {}   # sample -> its panel in sheet coordinates (r0, c0, r1, c1)
        self.events: list[tuple] = []     # (sample, rows, cols, last, born) of chalk that went away
        self.grown = np.zeros((self.rows, shape[1]), bool)      # state, one block wider
        self.ink = 0
        self.seen = -1
        self.moves = 0
        self.unsettled: set[int] = set()  # samples at which it had just been slid

    def frame_rect(self) -> tuple:
        r0, c0, r1, c1 = self.vis[self.seen]
        return c0, r0 + self.offset, c1, r1 + self.offset

    def agree(self, obs, ink, near, seg, offset, need):
        """How well the panel's chalk and this sheet's agree at `offset`, both ways.

        Each way is the share of one side's chalk found next to the other's,
        rescaled so that 0 is what a densely written board gives by chance.
        """
        x0, y0, x1, y1 = seg
        r0, r1 = y0 - offset, y1 - offset
        if r0 < 0 or r1 > self.rows:
            return None
        mine = self.state[r0:r1, x0:x1] & obs
        n, area = int(mine.sum()), int(obs.sum())
        if n < need:
            return None
        theirs = ink & obs
        m = int(theirs.sum())
        if not m:
            return 0.0
        grown = self.grown[r0:r1, x0:x1] & obs
        found, chance = (mine & near).sum() / n, (near & obs).sum() / area
        back, back_chance = (theirs & grown).sum() / m, grown.sum() / area
        return float(min((found - chance) / max(1 - chance, 1e-6),
                         (back - back_chance) / max(1 - back_chance, 1e-6)))

    def search(self, obs, ink, near, seg, need):
        """Best (agreement, offset) over vertical offsets, nearest to the current one on ties."""
        x0, y0, x1, y1 = seg
        rows = np.flatnonzero(self.state[:, x0:x1].any(axis=1))
        best = (0.0, None)
        if len(rows):
            offsets = range(y0 - int(rows[-1]), y1 - int(rows[0]))
            for off in sorted(offsets, key=lambda o: abs(o - self.offset)):
                s = self.agree(obs, ink, near, seg, off, need)
                if s is not None and s > best[0]:
                    best = (s, off)
        return best

    def update(self, i, seg, offset, obs, hi, near, confirm):
        x0, y0, x1, y1 = seg
        slid = bool(self.pos) and abs(offset - self.offset) > 2
        r0, r1 = y0 - offset, y1 - offset
        self.offset, self.seen = offset, i
        self.pos[i], self.vis[i] = offset, (r0, x0, r1, x1)
        if slid:
            # Caught mid-slide the board is blurred and its position a guess:
            # note where it is, read its chalk from the next sample on.
            self.moves += 1
            self.unsettled.add(i)
            return
        sl = (slice(r0, r1), slice(x0, x1))
        state, cnt = self.state[sl], self.cnt[sl]
        # Chalk counts as gone only if nothing is left next to it either: after a
        # slide the board rarely lands on the same 4-px grid.
        contra = obs & np.where(state, ~near, hi)
        cnt[...] = np.where(contra, np.minimum(cnt, 250) + 1, np.where(obs, 0, cnt))
        flip = cnt >= confirm
        if flip.any():
            off = flip & state
            if off.any():
                ys, xs = np.nonzero(off)
                self.events.append((i, ys + r0, xs + x0, self.last[sl][ys, xs], self.born[sl][ys, xs]))
            self.born[sl][flip & ~state] = max(0, i - confirm + 1)
            state ^= flip
            cnt[flip] = 0
            self.ink = int(self.state.sum())
            self.grown = _morph(self.state, 3, grow=True)
        self.last[sl][obs & near & state] = i


@dataclass
class Page:
    rect: tuple            # x0, y0, x1, y1 in frame pixels when the content was last seen
    first_ms: int          # when its first sizeable burst of chalk appeared
    last_ms: int           # last time its content was seen intact
    lost_ms: int | None    # when it was found gone; None: still there at the end
    image: Image.Image | None
    segments: list         # [{"start_ms", "end_ms", "tiles", "box"}], box in page pixels
    ink_blocks: int
    uncaptured: float      # share of chalk blocks no unobstructed frame showed
    frames: int = 1        # frames the image is composed from
    absorbed: list = field(default_factory=list)   # other views of this content
    # block-level bookkeeping, not part of the output
    occ: np.ndarray = field(default=None, repr=False)
    born: np.ndarray = field(default=None, repr=False)    # may inherit earlier times
    seen: np.ndarray = field(default=None, repr=False)    # when the chalk appeared on this sheet
    ref: int = 0
    sheet: int = 0
    origin: tuple = (0, 0)   # sheet row and column of the page's top-left block

    def meta(self) -> dict:
        return {"rect": list(self.rect), "first_ms": self.first_ms,
                "last_ms": self.last_ms, "lost_ms": self.lost_ms,
                "segments": self.segments, "ink_blocks": self.ink_blocks,
                "uncaptured": round(self.uncaptured, 4), "frames": self.frames,
                "absorbed": self.absorbed}


class BoardExtractor:
    def __init__(self, load_frame: Callable[[int], np.ndarray] | None = None, confirm: int = 2,
                 min_blocks: int = 40, margin: int = 3):
        self.load_frame = load_frame
        self.confirm = confirm          # contradicting observations before a block flips
        self.min_blocks = min_blocks    # chalk blocks below this are a correction, not a page
        self.margin = margin            # blocks kept clear of anything that is not board
        self.times: list[int] = []
        self.feats: list[np.ndarray] = []
        self.sheets: list[_Sheet] = []
        self.pending: list[tuple] = []  # panels no sheet claimed, with how long they have stayed
        self.recent: list[np.ndarray] = []   # board masks of the last minute

    # ── per sample ──────────────────────────────────────────────────────
    def add(self, rgb: np.ndarray, t_ms: int) -> None:
        self.shape = rgb.shape[:2]
        self.add_features(frame_features(rgb), t_ms)

    def add_features(self, f: np.ndarray, t_ms: int) -> None:
        i = len(self.times)
        brd = (f & BRD) > 0
        obs = self._observable(f)
        hi, lo = (f & HI) > 0, (f & LO) > 0
        near = _morph(lo, 3, grow=True)
        self.recent = (self.recent + [brd])[-12:]
        free = dict(enumerate(panels(brd, np.any(self.recent, axis=0), lo)))

        def crop(a, seg):
            return a[seg[1]:seg[3], seg[0]:seg[2]]

        def assign(sheet, g, offset):
            seg = free.pop(g)
            sheet.update(i, seg, offset, crop(obs, seg), crop(hi, seg), crop(near, seg), self.confirm)

        # The sheet that was on a panel stays on it while its chalk is still there.
        for s in self.sheets:
            if s.seen < i - 3:
                continue
            fx0, fy0, fx1, fy1 = s.frame_rect()
            here = [(_overlap(seg[1], seg[3], fy0, fy1), g) for g, seg in free.items()
                    if _overlap(seg[0], seg[2], fx0, fx1) >= 0.5]
            if not here or max(here)[0] < 0.5:
                continue
            g = max(here)[1]
            seg = free[g]
            sc = s.agree(crop(obs, seg), crop(lo, seg), crop(near, seg), seg, s.offset, MIN_REG)
            unmoved = abs(seg[1] - fy0) <= 2 and abs(seg[3] - fy1) <= 2
            if sc is None:
                if unmoved:
                    assign(s, g, s.offset)
                continue
            # A nudge of a block or two is followed without a search.
            offset = s.offset
            for near_by in (s.offset - 1, s.offset + 1, s.offset - 2, s.offset + 2):
                other = s.agree(crop(obs, seg), crop(lo, seg), crop(near, seg), seg, near_by, MIN_REG)
                if other is not None and other > sc + 0.05:
                    sc, offset = other, near_by
            if sc >= 0.5:
                assign(s, g, offset)

        # A panel with chalk that no sheet claimed: was a known board slid here?
        idle = [s for s in self.sheets if s.seen != i and s.ink >= MIN_REG]
        found = []
        for g, seg in free.items():
            if crop(hi & obs, seg).sum() < self.min_blocks:
                continue
            for k, s in enumerate(idle):
                c0, c1 = s.vis[s.seen][1], s.vis[s.seen][3]
                if _overlap(seg[0], seg[2], c0, c1) < 0.5:
                    continue
                score, offset = s.search(crop(obs, seg), crop(lo, seg), crop(near, seg), seg,
                                         max(MIN_REG, 0.3 * s.ink))
                if score >= 0.6:
                    found.append((score, g, k, offset))
        for score, g, k, offset in sorted(found, reverse=True):
            if g in free and idle[k].seen != i:
                assign(idle[k], g, offset)

        # Whatever else stays put for two samples is a board not met before.
        pending = []
        for g, seg in list(free.items()):
            count = 1 + max((c for old, c in self.pending
                             if abs(old[1] - seg[1]) <= 2 and abs(old[3] - seg[3]) <= 2
                             and _overlap(old[0], old[2], seg[0], seg[2]) >= 0.5), default=0)
            if count >= 2:
                self.sheets.append(_Sheet(f.shape))
                assign(self.sheets[-1], g, -PAD)
            else:
                pending.append((seg, count))
        self.pending = pending
        self.sheets = [s for s in self.sheets if s.ink or s.events or s.seen >= i - 3]
        self.feats.append(f)
        self.times.append(int(t_ms))

    def _observable(self, f: np.ndarray) -> np.ndarray:
        return _morph((f & BRD) > 0, 2 * self.margin + 1, grow=False)

    # ── at the end ──────────────────────────────────────────────────────
    def finish(self, images: bool = True) -> tuple[list[Page], dict]:
        n = len(self.times)
        if n < 2:
            return [], {"samples": n, "pages": 0, "candidates": 0, "sheets": 0,
                        "coverage": {"share": None}}
        self.F = np.stack(self.feats)
        self.feats = []
        self.dt = float(np.median(np.diff(self.times)))
        self.dropped = 0
        self._cache: dict[int, np.ndarray] = {}
        pages: list[Page] = []
        for k, s in enumerate(self.sheets):
            pages += self._sheet_pages(k, s, images)
        pages.sort(key=lambda p: (p.ref, p.rect))
        coverage = self._audit(pages)
        kept = self._absorb(pages)
        report = {"samples": n, "interval_s": round(self.dt / 1000, 2),
                  "frame": [v * BLOCK for v in self.F.shape[1:]], "pages": len(kept),
                  "candidates": len(pages),
                  "sheets": sum(1 for s in self.sheets if s.ink or s.events),
                  "slides": sum(s.moves for s in self.sheets),
                  "dropped_blocks": self.dropped, "coverage": coverage}
        self._cache = {}
        return kept, report

    def _samples(self, seconds: float) -> int:
        return max(1, round(seconds * 1000 / self.dt))

    def _sheet_pages(self, k: int, s: _Sheet, images: bool) -> list[Page]:
        """Pages of one sheet: one per wipe, one for what it still holds."""
        n = len(self.times)
        if len(s.pos) < self._samples(30):
            # Seen for a few seconds only: a board caught mid-slide, not content of its own.
            self.dropped += s.ink
            return []
        losses = []                      # [rows, cols, last, born, first sample, last sample]
        if s.events:
            count = np.zeros(n)
            for e in s.events:
                count[e[0]] += len(e[1])
            win = self._samples(15)
            active = np.convolve(count, np.ones(2 * win + 1), "same") >= self.min_blocks
            for a, b in _runs(active):
                if losses and a - losses[-1][5] <= self._samples(45):
                    a = losses.pop()[4]
                ev = [e for e in s.events if a <= e[0] < b]
                losses.append([*(np.concatenate([e[j] for e in ev]) for j in (1, 2, 3, 4)),
                               ev[0][0], ev[-1][0]])
            self.dropped += int(count[~active].sum())
        ys, xs = np.nonzero(s.state)
        if len(ys):
            # Not recognised again: wiped, or hidden until the end.
            gone = s.seen + 1 if s.seen < n - 1 - self._samples(20) else None
            rest = [ys, xs, s.last[ys, xs], s.born[ys, xs], gone, gone]
            if gone is not None and losses and gone - losses[-1][5] <= self._samples(45):
                # the same wipe went on after the sheet stopped being recognisable
                prev = losses.pop()
                rest = [*(np.concatenate([prev[j], rest[j]]) for j in range(4)), prev[4], gone]
            losses.append(rest)
        out = []
        for ys, xs, last, born, lost, _ in losses:
            _, first = np.unique(ys * s.state.shape[1] + xs, return_index=True)
            if len(first) < self.min_blocks:
                self.dropped += len(first)
                continue
            out.append(self._compose(k, s, ys[first], xs[first], last[first], born[first],
                                     None if lost is None else self.times[min(lost, n - 1)], images))
        return out

    def _compose(self, k, s, ys, xs, last, born, lost_ms, images) -> Page:
        ref = int(last.max())
        # Chalk on the sheet at `ref`: what it still holds, what went later, what goes now.
        b0 = np.full(s.state.shape, -1, np.int32)     # sample the chalk appeared
        b1 = b0.copy()                                # last sample known to show it
        alive = s.state & (s.born <= ref)
        b0[alive], b1[alive] = s.born[alive], ref
        for i, eys, exs, _, eborn in s.events:
            if i > ref:
                then = eborn <= ref
                b0[eys[then], exs[then]], b1[eys[then], exs[then]] = eborn[then], ref
        b0[ys, xs], b1[ys, xs] = born, last

        # The board's extent: rows and columns its panel covered in a fair share of its samples.
        seen = [t for t in s.vis if t <= ref]
        rows, cols = np.zeros(s.state.shape[0], int), np.zeros(s.state.shape[1], int)
        for t in seen:
            r0, c0, r1, c1 = s.vis[t]
            rows[r0:r1] += 1
            cols[c0:c1] += 1
        need = max(2, 0.1 * len(seen))
        rr, cc = np.flatnonzero(rows >= need), np.flatnonzero(cols >= need)
        if not len(rr) or not len(cc):
            rr, cc = np.flatnonzero(rows), np.flatnonzero(cols)
        r0, r1, c0, c1 = int(rr[0]), int(rr[-1]) + 1, int(cc[0]), int(cc[-1]) + 1
        h, w = r1 - r0, c1 - c0
        b0, b1 = b0[r0:r1, c0:c1], b1[r0:r1, c0:c1]
        occ = b0 >= 0

        # Candidate samples: the recent ones densely, older ones sparsely.
        seen = sorted(t for t in seen if t not in s.unsettled or t == ref)
        cand = np.array(sorted(set(seen[-120:]) | set(seen[:-120][::6])))
        hb = self.F.shape[1]
        brd, lo, hi = (np.zeros((len(cand), h, w), bool) for _ in range(3))
        for j, t in enumerate(cand):                  # each sample, where this sheet's panel was
            p = s.pos[t]
            vr0, vc0, vr1, vc1 = s.vis[t]
            a0, a1 = max(r0, vr0, -p), min(r1, vr1, hb - p)
            x0, x1 = max(c0, vc0), min(c1, vc1)
            if a0 < a1 and x0 < x1:
                f = self.F[t, a0 + p:a1 + p, x0:x1]
                at = (j, slice(a0 - r0, a1 - r0), slice(x0 - c0, x1 - c0))
                brd[at], lo[at], hi[at] = (f & BRD) > 0, (f & LO) > 0, (f & HI) > 0
        near = np.stack([_morph(m, 3, grow=True) for m in lo])
        occ_near = _morph(occ, 3, grow=True)
        # Blocks on a panel's rim flicker between board and not; only the
        # interior tells whether something stands in front of a tile.
        core = _morph(brd.mean(axis=0) >= 0.2, 5, grow=False)
        tiles = [(slice(ty, min(ty + TILE, h)), slice(tx, min(tx + TILE, w)))
                 for ty in range(0, h, TILE) for tx in range(0, w, TILE)]
        # A frame serves a tile by what it shows there: the page's chalk, and no other.
        score = np.full((len(cand), len(tiles)), -np.inf)
        hits = np.zeros(score.shape, int)
        for n, (sy, sx) in enumerate(tiles):
            wide = (slice(None), slice(max(sy.start - 2, 0), sy.stop + 2),
                    slice(max(sx.start - 2, 0), sx.stop + 2))
            here = (slice(None), sy, sx)
            clear = (~(core[wide[1:]] & ~brd[wide]).any(axis=(1, 2))
                     & (brd[here].mean(axis=(1, 2)) >= 0.9))
            hits[:, n] = (occ[sy, sx] & near[here]).sum(axis=(1, 2))
            stray = (hi[here] & ~occ_near[sy, sx]).sum(axis=(1, 2))
            score[clear, n] = (hits[:, n] - stray)[clear]
        # A frame from before a tile's last chalk appeared would lose that chalk, so
        # only later frames may stand in for the best one; without any, take the best.
        ready = np.array([b0[sy, sx][occ[sy, sx]].max(initial=-1) for sy, sx in tiles])
        late = np.isfinite(score) & (cand[:, None] >= ready[None, :])
        some = late.any(axis=0)
        pool = np.where(some[None, :], late, np.isfinite(score))
        best = np.where(pool, score, -np.inf).max(axis=0)
        good = pool & (score >= best - some)
        # As few frames as possible, so that the page looks like one photograph.
        choice = np.full(len(tiles), -1)
        left = good.any(axis=0)
        while left.any():
            serves = good[:, left].sum(axis=1)
            j = len(serves) - 1 - int(np.argmax(serves[::-1]))    # the latest of the best
            choice[left & good[j]] = j
            left &= ~good[j]
        each = np.arange(len(tiles))
        captured = int(hits[choice, each][choice >= 0].sum())
        # A tile no frame shows unobstructed: the frame that shows most of it as
        # board, or nothing if it was never much in view.
        for n in np.flatnonzero(choice < 0):
            share = brd[(slice(None), *tiles[n])].mean(axis=(1, 2))
            j = len(share) - 1 - int(np.argmax(share[::-1]))
            if share[j] >= 0.5:
                choice[n] = j
        image = Image.new("RGB", (w * BLOCK, h * BLOCK), (28, 44, 38)) if images else None
        if images:
            for j in np.unique(choice[choice >= 0]):
                t = int(cand[j])
                frame, p = self._frame(t), s.pos[t]
                for n in np.flatnonzero(choice == j):
                    sy, sx = tiles[n]
                    top, left_px = (r0 + sy.start + p) * BLOCK, (c0 + sx.start) * BLOCK
                    piece = frame[max(top, 0):top + (sy.stop - sy.start) * BLOCK,
                                  left_px:left_px + (sx.stop - sx.start) * BLOCK]
                    image.paste(Image.fromarray(piece), (sx.start * BLOCK, sy.start * BLOCK))
        frames_used = len(np.unique(choice[choice >= 0]))
        total = int(occ.sum())
        p = s.pos[ref]
        page = Page(rect=(c0 * BLOCK, (r0 + p) * BLOCK, c1 * BLOCK, (r1 + p) * BLOCK), first_ms=0,
                    last_ms=self.times[ref], lost_ms=lost_ms, image=image, segments=[],
                    ink_blocks=total, uncaptured=1 - captured / max(total, 1), frames=frames_used,
                    occ=occ, born=b0, seen=b0.copy(), ref=ref, sheet=k, origin=(r0, c0))
        self._timeline(page)
        return page

    def _frame(self, idx: int) -> np.ndarray:
        if idx not in self._cache:
            if len(self._cache) >= 6:
                self._cache.pop(next(iter(self._cache)))
            self._cache[idx] = self.load_frame(idx)
        return self._cache[idx]

    def _timeline(self, page: Page) -> None:
        """When each part of the page was written: tiles grouped into bursts.

        Single blocks are unreliable (a stroke's fringe flickers, new chalk can
        land where old chalk was), so a tile takes the median of its blocks.
        """
        h, w = page.occ.shape
        tiles = []
        for ty in range(0, h, TILE):
            for tx in range(0, w, TILE):
                sl = (slice(ty, ty + TILE), slice(tx, tx + TILE))
                born = page.born[sl][page.occ[sl]]
                if len(born) >= 3:
                    tiles.append((int(np.median(born)), tx, ty))
        tiles.sort()
        page.segments = []
        if not tiles:
            page.first_ms = page.last_ms
            return
        when = np.array([t for t, _, _ in tiles])
        for g in np.split(np.arange(len(tiles)), np.flatnonzero(np.diff(when) > self._samples(60)) + 1):
            xs, ys = [tiles[k][1] for k in g], [tiles[k][2] for k in g]
            page.segments.append({
                "start_ms": self.times[tiles[g[0]][0]], "end_ms": self.times[tiles[g[-1]][0]],
                "tiles": len(g),
                "box": [min(xs) * BLOCK, min(ys) * BLOCK,
                        min(max(xs) + TILE, w) * BLOCK, min(max(ys) + TILE, h) * BLOCK]})
        # A few tiles can keep the time of chalk they replaced; the page starts
        # with its first burst of real size.
        big = [g for g in page.segments if g["tiles"] >= max(3, 0.1 * len(tiles))]
        page.first_ms = (big or page.segments)[0]["start_ms"]

    def _absorb(self, pages: list[Page]) -> list[Page]:
        """Drop a page whose chalk another page shows, block for block.

        Handwriting is never repeated that exactly, so this is the same chalk:
        an earlier state of the board, a view of it after it was slid, or a
        board read wrongly mid-slide.  The match is corrected for what a
        densely written page gives by chance.
        """
        wb = self.F.shape[2]
        wide = []
        for p in pages:
            x0 = p.rect[0] // BLOCK
            row = np.zeros((p.occ.shape[0], wb), bool)
            row[:, x0:x0 + p.occ.shape[1]] = p.occ
            wide.append(row)
        grown = [_morph(row, 3, grow=True) for row in wide]
        gone: dict[int, int] = {}
        order = sorted(range(len(pages)), key=lambda k: pages[k].ink_blocks)
        for a in order:                         # smallest first: it is the partial view
            A, na = wide[a], pages[a].ink_blocks
            cols = np.flatnonzero(A.any(axis=0))
            best = (0.0, None, 0)
            for b in range(len(pages)):
                if b == a or b in gone or pages[b].ink_blocks < 0.9 * na:
                    continue
                B = grown[b]
                ha, hb = A.shape[0], B.shape[0]
                same = pages[b].sheet == pages[a].sheet
                at = pages[a].origin[0] - pages[b].origin[0]
                for dy in ([at] if same else range(-ha + 1, hb)):   # A's row 0 sits on B's row dy
                    lo, hi = max(0, dy), min(hb, dy + ha)
                    if lo >= hi:
                        continue
                    hit = (A[lo - dy:hi - dy] & B[lo:hi]).sum() / na
                    if hit > best[0]:
                        chance = B[lo:hi, cols[0]:cols[-1] + 1].mean()
                        if hit - chance >= 0.8 * (1 - chance):
                            best = (hit, b, dy)
            if best[0] >= 0.9:
                gone[a] = best[1]
                self._inherit(pages[a], pages[best[1]], best[2], best[0])
        return [p for k, p in enumerate(pages) if k not in gone]

    def _inherit(self, old: Page, new: Page, dy: int, match: float) -> None:
        """Carry the older view's write times over to the page that absorbs it."""
        dx = (old.rect[0] - new.rect[0]) // BLOCK
        ys, xs = np.nonzero(old.occ)
        ty, tx = ys + dy, xs + dx
        ok = (ty >= 0) & (ty < new.occ.shape[0]) & (tx >= 0) & (tx < new.occ.shape[1])
        ys, xs, ty, tx = ys[ok], xs[ok], ty[ok], tx[ok]
        ok = new.occ[ty, tx] & (old.born[ys, xs] < new.born[ty, tx])
        new.born[ty[ok], tx[ok]] = old.born[ys[ok], xs[ok]]
        new.absorbed.append({"rect": list(old.rect), "last_ms": old.last_ms,
                             "lost_ms": old.lost_ms, "match": round(match, 3)})
        new.absorbed += old.absorbed
        self._timeline(new)

    def _audit(self, pages: list[Page]) -> dict:
        """Share of strong chalk seen once a minute that some page holds.

        Uses the raw per-frame chalk, not the tracked state, so chalk the
        tracker never believed in, or on a panel it put on no sheet, counts
        against the result.
        """
        seen = held = 0
        worst = []
        for i in range(0, len(self.times), self._samples(60)):
            f = self.F[i]
            ink = self._observable(f) & ((f & HI) > 0)
            cover = np.zeros(ink.shape, bool)
            for k, s in enumerate(self.sheets):
                if i not in s.pos:
                    continue
                mine = np.zeros(s.state.shape, bool)
                for p in pages:
                    if p.sheet == k and p.ref >= i and p.seen[p.occ].min() <= i:
                        r0, c0 = p.origin
                        mine[r0:r0 + p.occ.shape[0], c0:c0 + p.occ.shape[1]] |= p.occ
                r0, c0, r1, c1 = s.vis[i]
                y0 = r0 + s.pos[i]
                cover[y0:y0 + r1 - r0, c0:c1] |= _morph(mine[r0:r1, c0:c1], 3, grow=True)
            miss = int((ink & ~cover).sum())
            seen += int(ink.sum())
            held += int(ink.sum()) - miss
            worst.append((miss, self.times[i]))
        worst.sort(reverse=True)
        return {"share": round(held / seen, 4) if seen else None, "chalk_blocks": seen,
                "worst": [{"ms": t, "missing_blocks": m} for m, t in worst[:5] if m]}


def contact_sheet(pages: list[Page], width: int = 1400) -> Image.Image:
    """All pages on one sheet for review, labelled with their time span."""
    def clock(ms):
        s = ms // 1000
        return f"{s // 3600}:{s // 60 % 60:02d}:{s % 60:02d}"

    cols = 2
    cell_w = width // cols
    tiles = []
    for k, p in enumerate(pages):
        scale = (cell_w - 10) / p.image.width
        thumb = p.image.resize((cell_w - 10, max(1, round(p.image.height * scale))))
        end = "end of lecture" if p.lost_ms is None else f"gone by {clock(p.lost_ms)}"
        tiles.append((thumb, f"#{k + 1}  written {clock(p.first_ms)}-{clock(p.last_ms)}, {end}"))
    rows = [tiles[i:i + cols] for i in range(0, len(tiles), cols)]
    heights = [max(t.height for t, _ in row) + 24 for row in rows]
    sheet = Image.new("RGB", (width, max(sum(heights), 1)), "white")
    draw = ImageDraw.Draw(sheet)
    y = 0
    for row, hgt in zip(rows, heights):
        for c, (thumb, label) in enumerate(row):
            draw.text((c * cell_w + 5, y + 5), label, fill="black")
            sheet.paste(thumb, (c * cell_w + 5, y + 22))
        y += hgt
    return sheet
