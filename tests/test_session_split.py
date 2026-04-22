import unittest

from session_split import build_split_masks, fuse_split_scores


class SessionSplitTests(unittest.TestCase):
    def test_threshold_not_triggered_returns_original_masks(self):
        masks = [[1, 1, 1, 1, 0, 0]]
        front, back, triggered = build_split_masks(masks, split_threshold=8, front_ratio=0.6)
        self.assertEqual(front, masks)
        self.assertEqual(back, masks)
        self.assertEqual(triggered, [False])

    def test_long_session_splits_into_front_and_back(self):
        masks = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]]
        front, back, triggered = build_split_masks(masks, split_threshold=8, front_ratio=0.6)
        self.assertEqual(front, [[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]])
        self.assertEqual(back, [[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0]])
        self.assertEqual(triggered, [True])

    def test_fuse_split_scores_returns_full_scores_when_no_session_triggered(self):
        full_scores = [[1.0, 2.0], [3.0, 4.0]]
        split_scores = [[10.0, 20.0], [30.0, 40.0]]
        triggered = [False, False]

        fused = fuse_split_scores(full_scores, split_scores, triggered, split_lambda=0.2)

        self.assertEqual(fused, full_scores)

    def test_fuse_split_scores_only_updates_triggered_sessions(self):
        full_scores = [[1.0, 3.0], [2.0, 4.0]]
        split_scores = [[11.0, 13.0], [12.0, 14.0]]
        triggered = [True, False]

        fused = fuse_split_scores(full_scores, split_scores, triggered, split_lambda=0.1)

        expected = [
            [2.0, 4.0],
            [2.0, 4.0],
        ]
        self.assertEqual(fused, expected)


if __name__ == '__main__':
    unittest.main()
