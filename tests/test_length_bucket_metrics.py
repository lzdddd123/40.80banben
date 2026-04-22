import unittest

from length_bucket_metrics import (
    bucket_name_for_length,
    format_length_bucket_report,
    summarize_length_buckets,
)


class LengthBucketMetricsTests(unittest.TestCase):
    def test_bucket_name_for_length_uses_short_medium_long_thresholds(self):
        self.assertEqual(bucket_name_for_length(1), 'SHORT')
        self.assertEqual(bucket_name_for_length(3), 'SHORT')
        self.assertEqual(bucket_name_for_length(4), 'MEDIUM')
        self.assertEqual(bucket_name_for_length(7), 'MEDIUM')
        self.assertEqual(bucket_name_for_length(8), 'LONG')

    def test_summarize_length_buckets_computes_bucket_metrics(self):
        lengths = [2, 5, 10, 9]
        hits = [1.0, 0.0, 1.0, 1.0]
        mrrs = [0.5, 0.0, 0.25, 1.0]

        summary = summarize_length_buckets(lengths, hits, mrrs)

        self.assertEqual(summary['SHORT']['count'], 1)
        self.assertAlmostEqual(summary['SHORT']['hr'], 100.0)
        self.assertAlmostEqual(summary['SHORT']['mrr'], 50.0)

        self.assertEqual(summary['MEDIUM']['count'], 1)
        self.assertAlmostEqual(summary['MEDIUM']['hr'], 0.0)
        self.assertAlmostEqual(summary['MEDIUM']['mrr'], 0.0)

        self.assertEqual(summary['LONG']['count'], 2)
        self.assertAlmostEqual(summary['LONG']['hr'], 100.0)
        self.assertAlmostEqual(summary['LONG']['mrr'], 62.5)

    def test_format_length_bucket_report_contains_expected_sections(self):
        summary = {
            'SHORT': {'count': 1, 'ratio': 25.0, 'hr': 100.0, 'mrr': 50.0},
            'MEDIUM': {'count': 1, 'ratio': 25.0, 'hr': 0.0, 'mrr': 0.0},
            'LONG': {'count': 2, 'ratio': 50.0, 'hr': 100.0, 'mrr': 62.5},
        }

        report = format_length_bucket_report(summary)

        self.assertIn('[Ablation Analysis] Session Length Buckets', report)
        self.assertIn('[SHORT] (占比: 25.0%, N=1):', report)
        self.assertIn('-> HR@20:  100.0000', report)
        self.assertIn('-> MRR@20: 62.5000', report)


if __name__ == '__main__':
    unittest.main()
