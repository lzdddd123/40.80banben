import unittest

from tmp_preprocess.preprocess import (
    generate_examples,
    infer_num_nodes_from_sequences,
)


class PreprocessEcommerceTests(unittest.TestCase):
    def test_generate_examples_matches_sr_gnn_style_prefix_target_pairs(self):
        sequences = [[1, 2, 3], [4, 5]]

        example_seqs, labels, session_indices = generate_examples(sequences)

        self.assertEqual(example_seqs, [[1, 2], [1], [4]])
        self.assertEqual(labels, [3, 2, 5])
        self.assertEqual(session_indices, [0, 0, 1])

    def test_infer_num_nodes_uses_max_item_plus_one(self):
        sequences = [[1, 3, 8], [2, 5]]

        self.assertEqual(infer_num_nodes_from_sequences(sequences), 9)


if __name__ == '__main__':
    unittest.main()
