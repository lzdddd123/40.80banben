import os
import shutil
import tempfile
import unittest


class DatasetAdaptationTests(unittest.TestCase):
    def test_resolve_dataset_dir_matches_case_insensitive_directory(self):
        from dataset_utils import resolve_dataset_dir

        tmpdir = os.path.join(os.getcwd(), 'tests_tmp_dataset_adaptation')
        if os.path.exists(tmpdir):
            shutil.rmtree(tmpdir)
        try:
            datasets_root = os.path.join(tmpdir, 'datasets')
            os.makedirs(os.path.join(datasets_root, 'yoochoose1_64'))
            cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                resolved = resolve_dataset_dir('YooChoose1_64', datasets_root=datasets_root)
            finally:
                os.chdir(cwd)
        finally:
            if os.path.exists(tmpdir):
                shutil.rmtree(tmpdir)
        self.assertTrue(resolved.lower().endswith('yoochoose1_64'))

    def test_infer_num_nodes_from_sequences_uses_max_item_plus_one(self):
        from dataset_utils import infer_num_nodes_from_sequences

        self.assertEqual(infer_num_nodes_from_sequences([[1, 5, 9], [2, 4]]), 10)


if __name__ == '__main__':
    unittest.main()
