import unittest

from build_graph import build_cooccurrence_graph


class BuildGraphTests(unittest.TestCase):
    def test_distance_decay_one_reproduces_plain_counts(self):
        seq = [[1, 2, 3, 4]]
        adj = build_cooccurrence_graph(seq, num_nodes=6, distance_decay=1.0)

        self.assertEqual(adj[1][2], 1.0)
        self.assertEqual(adj[2][3], 1.0)
        self.assertEqual(adj[3][4], 1.0)
        self.assertEqual(adj[1][3], 1.0)
        self.assertEqual(adj[1][4], 1.0)

    def test_distance_decay_reduces_farther_hop_weight(self):
        seq = [[1, 2, 3, 4]]
        adj = build_cooccurrence_graph(seq, num_nodes=6, distance_decay=0.5)

        self.assertEqual(adj[1][2], 1.0)
        self.assertEqual(adj[1][3], 0.5)
        self.assertEqual(adj[1][4], 0.25)


if __name__ == '__main__':
    unittest.main()
