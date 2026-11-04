import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import unittest
import numpy as np
from src.ai.board_selector import BoardSelector


class SelectorTest(unittest.TestCase):
    def setUp(self):
        self.saved = []
        self.selector = BoardSelector(lambda f,t,r,why,d,q: self.saved.append((t,why)),
            regions={'board': (0,0,1,1)}, change=.005, cooldown=10, heartbeat=30)
        self.blank = np.full((180,240,3), (45,80,65), dtype=np.uint8)
        self.written = self.blank.copy()
        self.written[20:22,20:200] = 220
        self.written[60:63,20:200] = 220

    def test_addition_and_erasure_retain_before(self):
        for t,f in [(0,self.blank),(10,self.written),(15,self.written),
                    (25,self.blank),(30,self.blank)]:
            self.selector.add(f,t)
        self.assertIn((0,'before_change'), self.saved)
        self.assertTrue(any(why=='possible_erasure_or_board_move' for _,why in self.saved))

    def test_transient_change_not_committed(self):
        for t,f in [(0,self.blank),(10,self.written),(15,self.blank)]:
            self.selector.add(f,t)
        self.assertFalse(any(why=='persistent_visual_change' for _,why in self.saved))

    def test_heartbeat_and_final_pending(self):
        self.selector.add(self.blank,0)
        self.selector.add(self.blank,30)
        self.selector.add(self.written,40)
        self.selector.finish()
        self.assertIn((30,'heartbeat'), self.saved)
        self.assertIn((40,'unconfirmed_final_change'), self.saved)

    def test_occlusion_proxy_does_not_commit_black_obstruction(self):
        blocked = self.written.copy()
        blocked[:, :100] = 0
        for t,f in [(0,self.written),(10,blocked),(15,blocked),(20,self.written)]:
            self.selector.add(f,t)
        self.assertEqual(self.saved, [(0,'initial')])

    def test_uniform_exposure_change_not_committed(self):
        brighter = (self.written.astype(np.int16)+12).clip(0,255).astype(np.uint8)
        for t,f in [(0,self.written),(10,brighter),(15,brighter)]:
            self.selector.add(f,t)
        self.assertEqual(self.saved, [(0,'initial')])

    def test_small_foreground_object_is_excluded(self):
        blocked = self.written.copy()
        blocked[90:160,80:120] = (180,90,70)
        for t,f in [(0,self.written),(10,blocked),(15,blocked)]:
            self.selector.add(f,t)
        self.assertEqual(self.saved, [(0,'initial')])


if __name__ == '__main__': unittest.main()
