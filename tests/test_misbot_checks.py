import unittest

import pandas as pd

from src.misbot_checks import account_ids, audit_text_overlap, user_balanced_style


class MisBotChecksTest(unittest.TestCase):
    def test_overlap_counts_types_and_posts_for_each_split_pair(self):
        report = audit_text_overlap(
            ['shared', 'shared', 'train-test', '', 'shared', 'val-test', 'train-test', 'val-test'],
            ['train', 'train', 'train', 'train', 'validation', 'validation', 'test', 'test'],
        )
        self.assertEqual(report, {
            'train_vs_validation': {'shared_texts': 1, 'train_posts': 2, 'validation_posts': 1},
            'train_vs_test': {'shared_texts': 1, 'train_posts': 1, 'test_posts': 1},
            'validation_vs_test': {'shared_texts': 1, 'validation_posts': 1, 'test_posts': 1},
        })

    def test_account_ids_reject_duplicate_ids_across_json_types(self):
        with self.assertRaisesRegex(ValueError, 'duplicate account'):
            account_ids([{'uid': 12}, {'uid': '12'}])

    def test_account_ids_reject_missing_values(self):
        with self.assertRaisesRegex(ValueError, 'nonempty account ID'):
            account_ids([{'uid': ' '}])

    def test_topic_score_gives_each_user_the_same_weight(self):
        posts = pd.DataFrame({'user_id': ['a', 'a', 'a', 'b'], 'bot_style': [0.1, 0.1, 0.1, 0.9]})
        self.assertAlmostEqual(user_balanced_style(posts), 0.5)


if __name__ == '__main__':
    unittest.main()
