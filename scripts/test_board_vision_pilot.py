import unittest
from board_vision_pilot import align_segments, shift_segments


class AlignmentTest(unittest.TestCase):
    def test_offsets_do_not_mutate_input(self):
        original = [{'start_ms': 100, 'end_ms': 900, 'text': 'speech'}]
        shifted = shift_segments(original, 60000)
        self.assertEqual(shifted[0]['start_ms'], 60100)
        self.assertEqual(original[0]['start_ms'], 100)
        self.assertEqual(align_segments(shifted, 60000, 1000), shifted)
        self.assertEqual(align_segments(shifted, 90000, 1000), [])

    def test_silence_does_not_invent_speech(self):
        self.assertEqual(align_segments([], 40000), [])


if __name__ == '__main__':
    unittest.main()
