"""Blackboard pages from sampled lecture frames (fixed camera, green boards).

Chalk on a board only accumulates until it is wiped, slid away or covered, so
one image taken just before each such loss holds everything written since the
previous one.  The extractor therefore looks for losses, not for changes:

  1. per frame: a board mask and a chalk mask, reduced to 4x4-px blocks;
  2. per block: a debounced "chalk present" state.  A block that is not
     visible (teacher, projection screen, bare wall) keeps its last state;
  3. losses close in time form one wipe.  Each board panel it touches yields
     one page, composed per tile from the frame that shows most of the tile's
     chalk with nothing in front of it;
  4. a page whose chalk reappears in another page at a different height (the
     board was slid, not wiped) is absorbed by that page.

Panels are not configured: they come from an XY-cut of the board mask at the
time the content was last seen, so sliding boards and other rooms need no
calibration.  Frames are not kept in memory; ``load_frame(i)`` must return
sample ``i`` again when a page is composed.
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
BRD, LO, HI, ST = 1, 2, 4, 8


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


def xy_cut(mask: np.ndarray, min_w: int = 24, min_h: int = 12) -> list[tuple]:
    """Axis-aligned panels (x0, y0, x1, y1) of a block mask."""
    out = []

    def cut(x0, y0, x1, y1, axis):
        sub = mask[y0:y1, x0:x1]
        for step in range(2):
            ax = (axis + step) % 2            # 0: split along x, 1: along y
            line = sub.sum(axis=ax)
            if not line.max():
                return
            parts = _runs(line >= 0.3 * line.max())
            if len(parts) > 1 or parts[0] != (0, len(line)):
                for a, b in parts:
                    if ax == 0:
                        cut(x0 + a, y0, x0 + b, y1, 1)
                    else:
                        cut(x0, y0 + a, x1, y0 + b, 0)
                return
        if x1 - x0 >= min_w and y1 - y0 >= min_h and sub.mean() >= 0.5:
            out.append((x0, y0, x1, y1))

    cut(0, 0, mask.shape[1], mask.shape[0], 0)
    return out


@dataclass
class Page:
    rect: tuple            # x0, y0, x1, y1 in pixels
    first_ms: int          # earliest chalk on the page
    last_ms: int           # last time its content was seen intact
    lost_ms: int | None    # when the loss was confirmed; None: still there at the end
    image: Image.Image
    segments: list         # [{"start_ms", "end_ms", "box"}], box in page pixels
    ink_blocks: int
    uncaptured: float      # share of chalk blocks no unobstructed frame showed
    absorbed: list = field(default_factory=list)   # earlier views of this content
    # block-level bookkeeping, not part of the output
    occ: np.ndarray = field(default=None, repr=False)
    born: np.ndarray = field(default=None, repr=False)    # may inherit earlier times
    seen: np.ndarray = field(default=None, repr=False)    # when chalk appeared at this place
    ref: int = 0

    def meta(self) -> dict:
        return {"rect": list(self.rect), "first_ms": self.first_ms,
                "last_ms": self.last_ms, "lost_ms": self.lost_ms,
                "segments": self.segments, "ink_blocks": self.ink_blocks,
                "uncaptured": round(self.uncaptured, 4), "absorbed": self.absorbed}


class BoardExtractor:
    def __init__(self, load_frame: Callable[[int], np.ndarray], confirm: int = 2,
                 min_blocks: int = 40, margin: int = 3):
        self.load_frame = load_frame
        self.confirm = confirm          # contradicting observations before a block flips
        self.min_blocks = min_blocks    # chalk blocks below this are a correction, not a page
        self.margin = margin            # blocks kept clear of anything that is not board
        self.times: list[int] = []
        self.feats: list[np.ndarray] = []
        self.events: list[tuple] = []   # (sample, ys, xs, last, born) of blocks that lost chalk
        self.state = None

    # ── passes 1 and 2, one sample at a time ────────────────────────────
    def add(self, rgb: np.ndarray, t_ms: int) -> None:
        f = frame_features(rgb)
        if self.state is None:
            self.shape = rgb.shape[:2]
            self.state = np.zeros(f.shape, bool)
            self.cnt = np.zeros(f.shape, np.uint8)
            self.last = np.zeros(f.shape, np.int32)
            self.born = np.zeros(f.shape, np.int32)
        i = len(self.times)
        obs = self._observable(f)
        lo, hi = (f & LO) > 0, (f & HI) > 0
        contra = obs & np.where(self.state, ~lo, hi)
        self.cnt = np.where(contra, np.minimum(self.cnt, 250) + 1,
                            np.where(obs, 0, self.cnt)).astype(np.uint8)
        flip = self.cnt >= self.confirm
        if flip.any():
            off = flip & self.state
            if off.any():
                ys, xs = np.nonzero(off)
                self.events.append((i, ys, xs, self.last[ys, xs], self.born[ys, xs]))
                # A tile that lost most of its chalk at once was swapped for
                # other content: chalk still in it is new, not a survivor.
                before, gone = _tile_sum(self.state), _tile_sum(off)
                swapped = np.repeat(np.repeat((gone >= 0.35 * before) & (before >= 4), TILE, 0),
                                    TILE, 1)[:f.shape[0], :f.shape[1]]
                rest = swapped & self.state & ~flip
                if rest.any():
                    ys, xs = np.nonzero(rest)
                    self.events.append((i, ys, xs, np.minimum(self.last[ys, xs], i - self.confirm),
                                        self.born[ys, xs]))
                    self.born[rest] = max(0, i - self.confirm + 1)
            self.born[flip & ~self.state] = max(0, i - self.confirm + 1)
            self.state ^= flip
            self.cnt[flip] = 0
        self.last[obs & lo & self.state] = i
        self.feats.append(f | (self.state * ST).astype(np.uint8))
        self.times.append(int(t_ms))

    def _observable(self, f: np.ndarray) -> np.ndarray:
        return _morph((f & BRD) > 0, 2 * self.margin + 1, grow=False)

    # ── passes 3 and 4 ──────────────────────────────────────────────────
    def finish(self) -> tuple[list[Page], dict]:
        n = len(self.times)
        if n < 2:
            return [], {"samples": n, "pages": 0}
        self.F = np.stack(self.feats)
        self.feats = []
        self.dt = float(np.median(np.diff(self.times)))
        self.dropped = 0
        self._cache: dict[int, np.ndarray] = {}
        pages: list[Page] = []

        count = np.zeros(n)
        for i, ys, *_ in self.events:
            count[i] += len(ys)
        win = self._samples(15)
        active = np.convolve(count, np.ones(2 * win + 1), "same") >= self.min_blocks
        runs = _runs(active)
        merged = []
        for a, b in runs:
            if merged and a - merged[-1][1] <= self._samples(45):
                merged[-1] = (merged[-1][0], b)
            else:
                merged.append((a, b))
        wipes = []
        for a, b in merged:
            ev = [e for e in self.events if a <= e[0] < b]
            ys, xs, last, born = (np.concatenate([e[k] for e in ev]) for k in (1, 2, 3, 4))
            _, keep = np.unique(ys * self.F.shape[2] + xs, return_index=True)
            made = self._pages(ys[keep], xs[keep], last[keep], born[keep], self.times[ev[0][0]])
            wipes.append({"start_ms": self.times[ev[0][0]], "end_ms": self.times[ev[-1][0]],
                          "blocks": len(keep), "pages": len(made)})
            pages += made
        self.dropped += int(count[~active].sum())
        ys, xs = np.nonzero(self.state)
        pages += self._pages(ys, xs, self.last[ys, xs], self.born[ys, xs], None)

        pages.sort(key=lambda p: (p.ref, p.rect))
        kept = self._absorb(pages)
        report = {"samples": n, "interval_s": round(self.dt / 1000, 2),
                  "frame": list(self.shape), "pages": len(kept),
                  "candidates": len(pages), "wipes": wipes,
                  "dropped_blocks": self.dropped, "coverage": self._audit(pages)}
        self._cache = {}
        return kept, report

    def _samples(self, seconds: float) -> int:
        return max(1, round(seconds * 1000 / self.dt))

    def _pages(self, ys, xs, last, born, lost_ms) -> list[Page]:
        """One page per panel for chalk blocks lost together (or left at the end)."""
        pages = []
        if not len(ys):
            return pages
        order = np.argsort(last, kind="stable")
        ys, xs, last, born = ys[order], xs[order], last[order], born[order]
        # Content hidden long before it was found gone belongs to an earlier layout.
        bounds = np.flatnonzero(np.diff(last) > self._samples(120)) + 1
        for g in np.split(np.arange(len(ys)), bounds):
            if len(g) < self.min_blocks:
                self.dropped += len(g)
                continue
            ref = int(last[g].max())
            layout = (self.F[max(0, ref - self._samples(60)):ref + 1] & BRD).any(axis=0)
            used = np.zeros(len(g), bool)
            for x0, y0, x1, y1 in xy_cut(layout):
                inside = (xs[g] >= x0) & (xs[g] < x1) & (ys[g] >= y0) & (ys[g] < y1)
                if inside.sum() < self.min_blocks:
                    continue
                used |= inside
                m = g[inside]
                pages.append(self._compose((x0, y0, x1, y1), ref, layout, ys[m], xs[m],
                                           last[m], born[m], lost_ms))
            self.dropped += int((~used).sum())
        return pages

    def _compose(self, rect, ref, layout, ys, xs, last, born, lost_ms) -> Page:
        x0, y0, x1, y1 = rect
        h, w = y1 - y0, x1 - x0
        alive = (self.F[ref, y0:y1, x0:x1] & ST) > 0
        # Sample at which each block that still holds chalk at `ref` got it.
        hist = (self.F[:ref + 1, y0:y1, x0:x1] & ST) > 0
        since = ref + 1 - np.argmax(~hist[::-1], axis=0)
        since[hist.all(axis=0)] = 0
        b0 = np.where(alive, np.maximum(since - self.confirm + 1, 0), -1)
        b1 = np.where(alive, ref, -1)           # last sample known to show the chalk
        b0[ys - y0, xs - x0], b1[ys - y0, xs - x0] = born, last
        occ = b0 >= 0

        lookback = self._samples(300)
        cand = np.arange(max(0, ref - lookback), ref + 1)
        brd = (self.F[cand[0]:ref + 1] & BRD) > 0
        # Blocks on a panel's rim flicker between board and not; only the
        # interior tells whether something stands in front of a tile.
        core = _morph(layout, 5, grow=False)
        image = Image.new("RGB", (w * BLOCK, h * BLOCK))
        picks: dict[int, list] = {}
        captured = 0
        for ty in range(0, h, TILE):
            for tx in range(0, w, TILE):
                sl = (slice(ty, min(ty + TILE, h)), slice(tx, min(tx + TILE, w)))
                a, b = b0[sl][occ[sl]], b1[sl][occ[sl]]
                ay0, ay1 = max(y0 + ty - 2, 0), y0 + ty + TILE + 2
                ax0, ax1 = max(x0 + tx - 2, 0), x0 + tx + TILE + 2
                clear = ~(core[ay0:ay1, ax0:ax1] & ~brd[:, ay0:ay1, ax0:ax1]).any(axis=(1, 2))
                shown = ((a[None, :] <= cand[:, None]) & (cand[:, None] <= b[None, :])).sum(axis=1)
                score = np.where(clear, shown, -1)
                best = len(cand) - 1 - int(np.argmax(score[::-1]))   # latest of the best
                if score[best] < 0:
                    best = len(cand) - 1                              # never unobstructed
                else:
                    captured += int(score[best])
                picks.setdefault(int(cand[best]), []).append(sl)
        for idx, tiles in picks.items():
            frame = self._frame(idx)
            for sy, sx in tiles:
                box = ((x0 + sx.start) * BLOCK, (y0 + sy.start) * BLOCK,
                       (x0 + sx.stop) * BLOCK, (y0 + sy.stop) * BLOCK)
                image.paste(Image.fromarray(frame[box[1]:box[3], box[0]:box[2]]),
                            (sx.start * BLOCK, sy.start * BLOCK))
        total = int(occ.sum())
        page = Page(rect=tuple(v * BLOCK for v in rect), first_ms=0,
                    last_ms=self.times[ref], lost_ms=lost_ms, image=image, segments=[],
                    ink_blocks=total, uncaptured=1 - captured / max(total, 1),
                    occ=occ, born=b0, seen=b0.copy(), ref=ref)
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
        """Drop a page whose chalk another page shows at some vertical offset."""
        wb = self.F.shape[2]
        wide = []
        for p in pages:
            x0 = p.rect[0] // BLOCK
            row = np.zeros((p.occ.shape[0], wb), bool)
            row[:, x0:x0 + p.occ.shape[1]] = p.occ
            wide.append(row)
        gone: dict[int, int] = {}
        order = sorted(range(len(pages)), key=lambda k: pages[k].ink_blocks)
        for a in order:                         # smallest first: it is the partial view
            A, na = wide[a], pages[a].ink_blocks
            best = (0.0, None, 0)
            # The same chalk shows up elsewhere only as this view goes away (the
            # board was slid); in place, it must predate this view's loss.
            slack = self._samples(30)
            lost = (len(self.times) if pages[a].lost_ms is None
                    else self.times.index(pages[a].lost_ms))
            ya = pages[a].rect[1] // BLOCK
            for b in range(len(pages)):
                if b == a or b in gone or pages[b].ink_blocks < 0.9 * na:
                    continue
                pb = pages[b]
                x0, yb = pb.rect[0] // BLOCK, pb.rect[1] // BLOCK
                before = pb.occ & (pb.seen <= lost + slack)
                moved = before & (pb.seen >= pages[a].ref - slack)
                ha, hb = A.shape[0], pb.occ.shape[0]
                near = np.zeros((2, hb, wb), bool)
                near[0, :, x0:x0 + pb.occ.shape[1]] = moved
                near[1, :, x0:x0 + pb.occ.shape[1]] = before
                near = [_morph(m, 3, grow=True) for m in near]
                for dy in range(-ha + 1, hb):   # A's row 0 sits on B's row dy
                    B = near[abs(yb + dy - ya) <= 1]
                    lo, hi = max(0, dy), min(hb, dy + ha)
                    hit = (A[lo - dy:hi - dy] & B[lo:hi]).sum() / na
                    if hit > best[0]:
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

        Uses the raw per-frame chalk, not the tracked state, so a block the
        tracker never believed in still counts against the result.
        """
        seen = held = 0
        worst = []
        for i in range(0, len(self.times), self._samples(60)):
            f = self.F[i]
            ink = self._observable(f) & ((f & HI) > 0)
            cover = np.zeros(ink.shape, bool)
            for p in pages:
                if p.ref < i or not p.occ.any() or p.born[p.occ].min() > i:
                    continue
                x0, y0 = p.rect[0] // BLOCK, p.rect[1] // BLOCK
                cover[y0:y0 + p.occ.shape[0], x0:x0 + p.occ.shape[1]] |= _morph(p.occ, 3, grow=True)
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
