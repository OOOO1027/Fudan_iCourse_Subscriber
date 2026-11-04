import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import unittest
import numpy as np
from PIL import Image, ImageFilter
from src.ai.board_curator import BoardFeatures, SIZE, curate, extract_features


def feature(ink):
    expanded = np.asarray(Image.fromarray(ink.astype(np.uint8)*255).filter(ImageFilter.MaxFilter(3))) > 127
    return [BoardFeatures(ink,expanded,1.0)]


class CuratorTests(unittest.TestCase):
    def setUp(self):
        self.a = np.zeros((SIZE,SIZE),dtype=bool)
        self.a[30:32,20:150] = True
        self.b = np.zeros_like(self.a)
        self.b[220:222,210:350] = True

    def test_identical_boards_need_one_image(self):
        r=curate([feature(self.a)]*5,list(range(5)))
        self.assertEqual(len(r['selected_indices']),1)

    def test_completed_board_replaces_partial(self):
        r=curate([feature(self.a),feature(self.a|self.b)],[0,1])
        self.assertEqual(r['selected_indices'],[1])

    def test_erased_content_stays_represented(self):
        r=curate([feature(self.a),feature(self.b)],[0,1])
        self.assertEqual(r['selected_indices'],[0,1])

    def test_complementary_views_are_both_kept(self):
        c=np.zeros_like(self.a)
        c[130:132,130:250]=True
        r=curate([feature(self.a|c),feature(self.b|c)],[0,1])
        self.assertEqual(r['selected_indices'],[0,1])

    def test_clear_view_replaces_obstructed_views(self):
        r=curate([feature(self.a),feature(self.b),feature(self.a|self.b)],[0,1,2])
        self.assertEqual(r['selected_indices'],[2])

    def test_blank_is_not_content(self):
        r=curate([feature(np.zeros_like(self.a))],[0])
        self.assertEqual(r['selected_indices'],[])
        self.assertEqual(r['frames'][0]['status'],'low_visual_evidence')

    def test_exposure_and_foreground(self):
        image=np.full((SIZE,SIZE,3),(45,80,65),dtype=np.uint8)
        image[self.a]=150
        brighter=(image.astype(np.int16)+10).clip(0,255).astype(np.uint8)
        regions={'board':(0,0,1,1)}
        a=extract_features(image,regions)[0].ink
        b=extract_features(brighter,regions)[0].ink
        self.assertGreater(a.sum(),0)
        self.assertGreater((a&b).sum()/a.sum(),.98)


if __name__=='__main__': unittest.main()
