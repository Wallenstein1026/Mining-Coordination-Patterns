"""Score the audited Weibo-COV subset using the frozen local MisBot model."""

from hashlib import file_digest, sha256
import json
from pathlib import Path
import pickle
from tempfile import TemporaryDirectory

import pandas as pd

from run_analysis import (
    MIN_TOPIC_POSTS, MIN_TOPIC_USERS, RES, ROOT, STYLE_NAMES, dataset_input, load_weibo,
    score_post_style, topic_membership, user_balanced_style,
)


def score_tables(posts, model):
    scored = posts[['mid', 'user_id', 'tags']].assign(bot_style=score_post_style(posts.content, model))
    if not scored.bot_style.between(0, 1).all():
        raise ValueError('The model returned a non-finite or out-of-range style score')

    topics = []
    for topic, indices in topic_membership(scored).items():
        members = scored.iloc[indices]
        topics.append((topic, len(members), members.user_id.nunique(), members.bot_style.mean(),
                       user_balanced_style(members)))
    columns = ['topic', 'posts', 'users', 'bot_style', 'user_balanced_style']
    return scored[['mid', 'bot_style']], pd.DataFrame(topics, columns=columns)


def fingerprint(path):
    with path.open('rb') as handle:
        return file_digest(handle, 'sha256').hexdigest()


def write_handoff(output_dir, post_scores, topic_scores, provenance):
    output_dir.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='.building-', dir=output_dir) as temp:
        staged = Path(temp) / 'handoff'
        staged.mkdir(mode=0o700)
        tables = {'post_style_scores.csv': post_scores, 'topic_style_scores.csv': topic_scores}
        for name, table in tables.items():
            table.to_csv(staged / name, index=False)
        receipt = {
            **provenance,
            'scored_posts': len(post_scores),
            'eligible_topics': len(topic_scores),
            'output_sha256': {name: fingerprint(staged / name) for name in tables},
        }
        receipt_file = staged / 'receipt.json'
        receipt_file.write_text(json.dumps(receipt, indent=2, sort_keys=True) + '\n', encoding='utf-8')
        receipt_hash = fingerprint(receipt_file)
        final = output_dir / receipt_hash[:16]
        try:
            staged.rename(final)
        except OSError:
            expected = {**receipt['output_sha256'], 'receipt.json': receipt_hash}
            if not final.is_dir() or any(not (final / name).is_file() or fingerprint(final / name) != digest
                                         for name, digest in expected.items()):
                raise
        return final


def main():
    source = dataset_input('weibo')
    audit = json.loads((RES / 'weibo_data_audit.json').read_text())
    input_sha256 = source[1]['normalized_sha256']
    if audit['status'] != 'monthly_import_verified' or audit['normalized_sha256'] != input_sha256:
        raise ValueError('Weibo-COV input does not match the published handoff audit')

    model_metadata = json.loads((RES / 'misbot_style_model.json').read_text())
    misbot = json.loads((ROOT / 'data/misbot_manifest.json').read_text())
    if (model_metadata['input_sha256'] != misbot['sha256']
            or model_metadata['release_kind'] != misbot['release_kind']
            or model_metadata['features'] != STYLE_NAMES
            or model_metadata['positive_label'] != 1):
        raise ValueError('MisBot provenance differs from the frozen style model metadata')
    # The pickle must come from a trusted local run of run_analysis.py --stage misbot.
    serialized = (RES / 'text_style_model.pkl').read_bytes()
    if sha256(serialized).hexdigest() != model_metadata['model_sha256']:
        raise ValueError('Frozen MisBot model SHA-256 mismatch')
    model = pickle.loads(serialized)

    posts = load_weibo(source=source, write_audit=False)
    if len(posts) != audit['flow']['retained'] or posts.user_id.nunique() != audit['unique_users']:
        raise ValueError('Weibo-COV decoded row or user count differs from the data audit')
    post_scores, topic_scores = score_tables(posts, model)
    provenance = {
        'weibo_normalized_sha256': input_sha256,
        'weibo_coverage_start': audit['first'],
        'weibo_coverage_end': audit['last'],
        'weibo_users': audit['unique_users'],
        'misbot_input_sha256': model_metadata['input_sha256'],
        'misbot_release_kind': model_metadata['release_kind'],
        'model_sha256': model_metadata['model_sha256'],
        'topic_min_posts': MIN_TOPIC_POSTS,
        'topic_min_users': MIN_TOPIC_USERS,
        'scope': 'Descriptive cross-domain text-style scores for the audited Weibo-COV subset',
    }
    output = write_handoff(ROOT / '.local/misbot_weibo_style', post_scores, topic_scores, provenance)
    print(output)
    print(f'Scored {len(post_scores):,} posts and {len(topic_scores):,} eligible topics.')


if __name__ == '__main__':
    main()
