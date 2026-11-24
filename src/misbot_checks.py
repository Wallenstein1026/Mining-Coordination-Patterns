"""Aggregate checks for an account-stratified MisBot benchmark."""

from collections import Counter
from itertools import combinations


SPLITS = ('train', 'validation', 'test')


def account_ids(records):
    ids = []
    for record in records:
        uid = record.get('uid')
        if type(uid) not in (int, str) or not str(uid).strip():
            raise ValueError('MisBot records require a nonempty account ID')
        ids.append(str(uid).strip())
    if len(set(ids)) != len(ids):
        raise ValueError('MisBot contains a duplicate account ID')
    return ids


def audit_text_overlap(texts, splits):
    counts = {name: Counter() for name in SPLITS}
    for text, split in zip(texts, splits, strict=True):
        if text:
            counts[split][text] += 1

    report = {}
    for first, second in combinations(SPLITS, 2):
        shared = counts[first].keys() & counts[second].keys()
        report[f'{first}_vs_{second}'] = {
            'shared_texts': len(shared),
            f'{first}_posts': sum(counts[first][text] for text in shared),
            f'{second}_posts': sum(counts[second][text] for text in shared),
        }
    return report


def user_balanced_style(posts):
    return float(posts.groupby('user_id')['bot_style'].mean().mean())
