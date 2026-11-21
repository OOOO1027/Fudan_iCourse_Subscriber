"""Synthetic classroom scenes for src/ai/board_capture.py.

Run: python scripts/test_board_capture.py
Each scene renders 640x360 frames 5 s apart: green boards on a white wall,
chalk rows that appear over time, a striped teacher walking in front.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from src.ai.board_capture import BoardExtractor

W, H, STEP_MS = 640, 360, 5000
BOARD, WALL = (65, 86, 76), (236, 236, 236)


class Board:
    def __init__(self, box, seed):
        self.box = list(box)                       # x0, y0, x1, y1 in frame pixels
        self.chalk = Image.new("L", (box[2] - box[0], box[3] - box[1]))
        self.rng = np.random.default_rng(seed)
        self.row = 0

    def write(self, rows=1):
        """Add rows of handwriting-like strokes below what is already there."""
        draw = ImageDraw.Draw(self.chalk)
        for _ in range(rows):
            y = 16 + 26 * self.row
            x = 14 + int(self.rng.integers(0, 40))
            while x < self.chalk.width - 30 and y < self.chalk.height - 10:
                pts = [(x + int(self.rng.integers(0, 12)), y + int(self.rng.integers(-7, 8)))
                       for _ in range(4)]
                draw.line(pts, fill=235, width=2)
                x += int(self.rng.integers(14, 24)) * int(self.rng.integers(1, 4))
            self.row += 1

    def wipe(self, x0=0, x1=None):
        ImageDraw.Draw(self.chalk).rectangle([x0, 0, x1 or self.chalk.width, self.chalk.height], fill=0)
        if x0 == 0 and x1 is None:
            self.row = 0


def render(boards, teacher_x=None, screen=None, seed=0):
    frame = Image.new("RGB", (W, H), WALL)
    for b in boards:
        face = Image.new("RGB", b.chalk.size, BOARD)
        face.paste((238, 240, 236), mask=b.chalk)
        frame.paste(face, tuple(b.box[:2]))
    if screen:                                    # projection screen: white with dark text
        draw = ImageDraw.Draw(frame)
        draw.rectangle(screen, fill=(246, 246, 250))
        for y in range(screen[1] + 20, screen[3] - 10, 18):
            draw.line([screen[0] + 20, y, screen[2] - 40, y], fill=(40, 40, 60), width=2)
    if teacher_x is not None:
        draw = ImageDraw.Draw(frame)
        draw.ellipse([teacher_x + 4, 70, teacher_x + 30, 100], fill=(190, 150, 130))
        for k, y in enumerate(range(100, 330, 8)):
            draw.rectangle([teacher_x, y, teacher_x + 36, y + 7],
                           fill=(30, 30, 36) if k % 2 else (225, 225, 228))
    noise = np.random.default_rng(seed).normal(0, 2, (H, W, 3))
    return np.clip(np.asarray(frame.filter(ImageFilter.GaussianBlur(0.6)), float) + noise,
                   0, 255).astype(np.uint8)


def run(frames):
    ex = BoardExtractor(lambda i: frames[i])
    for i, f in enumerate(frames):
        ex.add(f, i * STEP_MS)
    return ex.finish()


def recall(page, board_box, chalk):
    """Share of the true chalk pixels that are bright in the page image."""
    truth = np.zeros((H, W), bool)
    truth[board_box[1]:board_box[3], board_box[0]:board_box[2]] = np.asarray(chalk) > 128
    got = np.zeros((H, W), bool)
    x0, y0, x1, y1 = page.rect
    got[y0:y1, x0:x1] = np.asarray(page.image.convert("L")) > 150
    # page.rect is where the board was to the nearest 4-px block
    got = np.asarray(Image.fromarray(got).filter(ImageFilter.MaxFilter(7)))
    return (truth & got).sum() / max(truth.sum(), 1)


def teacher_share(page):
    """Share of page pixels as dark as the teacher's stripes."""
    return (np.asarray(page.image.convert("L")) < 45).mean()


def walking(i):
    return 50 + (i * 37) % 300


class BoardCaptureTest(unittest.TestCase):
    def test_one_page_per_wipe_without_the_teacher(self):
        board = Board((40, 40, 360, 200), seed=1)
        frames, truths = [], []
        for i in range(105):
            if i in (30, 66):
                truths.append(board.chalk.copy())
            if 30 <= i < 34 or 66 <= i < 70:        # wipe left to right over 20 s
                k = i - (30 if i < 40 else 66)
                board.wipe(0, 80 * (k + 1))
                if k == 3:
                    board.row = 0
            elif i % 3 == 0 and board.row < 5 and i not in range(20, 30):
                board.write()
            frames.append(render([board], teacher_x=walking(i), seed=i))
        truths.append(board.chalk.copy())
        pages, report = run(frames)
        self.assertEqual(len(pages), 3, report)
        for page, truth in zip(pages, truths):
            self.assertGreater(recall(page, board.box, truth), 0.97)
            self.assertLess(teacher_share(page), 0.005)
        self.assertEqual([p.lost_ms is None for p in pages], [False, False, True])
        self.assertGreater(report["coverage"]["share"], 0.95)

    def test_slid_board_is_one_page_with_its_first_write_time(self):
        board = Board((40, 150, 360, 310), seed=2)
        frames = []
        for i in range(85):
            if i == 30:                             # pushed up by 118 px, off the block grid
                board.box[1] -= 118
                board.box[3] -= 118
            if i == 60:
                truth = board.chalk.copy()
                final_box = list(board.box)
                board.wipe()
            if (i < 9 and i % 3 == 0) or (36 <= i < 42 and i % 3 == 0):
                board.write()
            frames.append(render([board], teacher_x=walking(i) + 200, seed=i))
        pages, report = run(frames)
        self.assertEqual(len(pages), 1, report)
        self.assertGreater(recall(pages[0], final_box, truth), 0.97)
        self.assertGreaterEqual(report["slides"], 1)
        self.assertLess(pages[0].first_ms, 15000)   # written from the start, not from the slide
        self.assertEqual(pages[0].rect[1] // 8, final_box[1] // 8)

    def test_content_wiped_behind_the_screen_is_kept(self):
        board = Board((40, 40, 360, 200), seed=3)
        frames = []
        for i in range(100):
            if i < 15 and i % 3 == 0:
                board.write()
            if i == 50:
                truth = board.chalk.copy()
                board.wipe()
            screen = (20, 20, 400, 230) if 30 <= i <= 80 else None
            frames.append(render([board], teacher_x=walking(i) if i < 30 else None,
                                 screen=screen, seed=i))
        pages, report = run(frames)
        self.assertEqual(len(pages), 1, report)
        self.assertGreater(recall(pages[0], board.box, truth), 0.97)
        self.assertLessEqual(pages[0].last_ms, 29 * STEP_MS)

    def test_teacher_standing_still_is_not_a_wipe(self):
        board = Board((40, 40, 360, 200), seed=4)
        frames = []
        for i in range(75):
            if i < 15 and i % 3 == 0:
                board.write()
            frames.append(render([board], teacher_x=180 if 20 <= i < 60 else walking(i), seed=i))
        pages, report = run(frames)
        self.assertEqual(len(pages), 1, report)
        self.assertGreater(recall(pages[0], board.box, board.chalk), 0.97)
        self.assertLess(teacher_share(pages[0]), 0.005)

    def test_small_correction_is_not_a_page(self):
        board = Board((40, 40, 360, 200), seed=5)
        frames = []
        for i in range(70):
            if i < 15 and i % 3 == 0:
                board.write()
            if i == 30:                             # rub out one symbol and rewrite it
                board.wipe(100, 116)
            frames.append(render([board], teacher_x=walking(i), seed=i))
        pages, report = run(frames)
        self.assertEqual(len(pages), 1, report)

    def test_two_panels_wiped_together_are_two_pages(self):
        upper, lower = Board((40, 30, 360, 160), seed=6), Board((40, 172, 360, 300), seed=7)
        frames = []
        for i in range(70):
            if i < 12 and i % 3 == 0:
                upper.write()
                lower.write()
            if i == 40:
                truths = [upper.chalk.copy(), lower.chalk.copy()]
                upper.wipe()
            if i == 42:
                lower.wipe()
            frames.append(render([upper, lower], teacher_x=walking(i) + 250, seed=i))
        pages, report = run(frames)
        self.assertEqual(len(pages), 2, report)
        for page, board, truth in zip(pages, (upper, lower), truths):
            self.assertGreater(recall(page, board.box, truth), 0.97)

    def test_unused_board_gives_nothing(self):
        frames = [render([Board((40, 40, 360, 200), seed=8)], teacher_x=walking(i), seed=i)
                  for i in range(40)]
        pages, report = run(frames)
        self.assertEqual(pages, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
