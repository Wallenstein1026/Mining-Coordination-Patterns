"""Regression checks for the private MisBot scoring handoff."""

import hashlib
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

import pandas as pd
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from run_analysis import style
from score_weibo_style import score_tables, write_handoff


class WeiboStyleHandoffTest(unittest.TestCase):
    def test_tables_follow_shared_topic_rules_and_weight_authors_equally(self):
        common, rare = '控件正常运行', '直播！！！@提及 #标签# 123'
        model = LogisticRegression().fit([style(common), style(rare)], [0, 1])
        posts = pd.DataFrame({
            'mid': [f'{i:03d}' for i in range(30)],
            'user_id': ['active'] * 26 + ['a', 'b', 'c', 'd'],
            'content': [common] * 26 + [rare] * 4,
            'tags': [['话题', '疫情', '人数偏少']] * 29 + [['话题', '疫情']],
        })
        post_scores, topic_scores = score_tables(posts, model)
        self.assertEqual(list(post_scores), ['mid', 'bot_style'])
        self.assertEqual(post_scores.mid.tolist(), posts.mid.tolist())
        self.assertEqual(topic_scores.topic.tolist(), ['话题'])
        row = topic_scores.iloc[0]
        self.assertEqual((row.posts, row.users), (30, 5))
        a, b = model.predict_proba([style(common), style(rare)])[:, 1]
        self.assertAlmostEqual(row.bot_style, (26 * a + 4 * b) / 30)
        self.assertAlmostEqual(row.user_balanced_style, (a + 4 * b) / 5)

    def test_handoff_records_hashes_and_can_be_retried(self):
        post_scores = pd.DataFrame({'mid': ['0001'], 'bot_style': [0.125]})
        topic_scores = pd.DataFrame(columns=['topic', 'posts', 'users', 'bot_style', 'user_balanced_style'])
        provenance = {'weibo_normalized_sha256': 'a' * 64, 'model_sha256': 'b' * 64}
        with TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = write_handoff(root, post_scores, topic_scores, provenance)
            self.assertEqual(write_handoff(root, post_scores, topic_scores, provenance), bundle)
            self.assertEqual(list(root.iterdir()), [bundle])
            receipt = json.loads((bundle / 'receipt.json').read_text())
            self.assertEqual((receipt['scored_posts'], receipt['eligible_topics']), (1, 0))
            for filename, digest in receipt['output_sha256'].items():
                self.assertEqual(hashlib.sha256((bundle / filename).read_bytes()).hexdigest(), digest)
            self.assertEqual((bundle / 'post_style_scores.csv').read_text().strip(), 'mid,bot_style\n0001,0.125')


if __name__ == '__main__':
    unittest.main()
