"""Recompute held-out MisBot metrics and check the saved model provenance."""

import json
import math
from hashlib import sha256

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

from run_analysis import RES, SEED, SPLITS, STYLE_NAMES, dataset_input, metrics


def main():
    source, manifest = dataset_input('misbot')
    report = json.loads((RES / 'misbot_benchmark.json').read_text())
    frozen = json.loads((RES / 'misbot_style_model.json').read_text())
    summary = pd.read_csv(RES / 'misbot_benchmark_summary.csv').set_index('model')
    uid_options = {'dtype': {'uid': str}, 'keep_default_na': False}
    splits = pd.read_csv(RES / 'misbot_account_splits.csv', **uid_options).set_index('uid')
    assert splits.index.is_unique and set(splits['split']) == set(SPLITS)
    assert splits['split'].value_counts().to_dict() == report['split_counts']
    with source.open(encoding='utf-8') as handle:
        source_labels = {
            str(record['uid']).strip(): int(record['label'])
            for record in map(json.loads, handle)
        }
    assert splits['label'].to_dict() == source_labels, 'Saved splits must cover the source accounts and labels'

    predictions = {}
    for name in ('validation', 'test'):
        pred = pd.read_csv(RES / f'misbot_{name}_predictions.csv', float_precision='round_trip',
                           **uid_options).set_index('uid')
        expected = splits.loc[splits['split'] == name, 'label'].sort_index()
        assert pred['label'].sort_index().equals(expected), f'{name} account IDs or labels changed'
        predictions[name] = pred

    val, test = predictions['validation'], predictions['test']
    for model, saved in report['models'].items():
        grid = np.linspace(0.1, 0.9, 81)
        val_scores = [balanced_accuracy_score(val['label'], val[model] >= threshold) for threshold in grid]
        best = int(np.argmax(val_scores))
        assert math.isclose(grid[best], saved['threshold'], abs_tol=1e-12), f'{model} validation threshold'
        assert math.isclose(val_scores[best], report['validation_balanced_accuracy'][model], abs_tol=1e-12)
        calculated = metrics(test['label'], test[model], saved['threshold'])
        assert calculated.pop('confusion_matrix') == saved['confusion_matrix'], f'{model} confusion matrix'
        for name, value in calculated.items():
            assert math.isclose(value, saved[name], abs_tol=1e-12), f'{model} {name}'
            assert math.isclose(value, summary.at[model, name], abs_tol=1e-12), f'{model} summary {name}'
        for name, count in report['split_counts'].items():
            assert summary.at[model, f'n_{name}_accounts'] == count
        assert summary.at[model, 'n_test_positive_accounts'] == test['label'].sum()
        assert summary.at[model, 'input_sha256'] == manifest['sha256']
        assert summary.at[model, 'release_kind'] == manifest['release_kind']

    assert frozen['input_sha256'] == manifest['sha256']
    assert frozen['release_kind'] == manifest['release_kind']
    assert frozen['seed'] == SEED and frozen['features'] == STYLE_NAMES
    assert frozen['source_account_threshold'] == report['models']['Text-style LR']['threshold']
    assert sha256((RES / 'text_style_model.pkl').read_bytes()).hexdigest() == frozen['model_sha256']
    print('MisBot input hash, account isolation, validation thresholds, test metrics and model hash verified.')


if __name__ == '__main__':
    main()
