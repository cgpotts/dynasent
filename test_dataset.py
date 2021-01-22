from collections import Counter, defaultdict
import itertools
import os
import pytest

from dynasent_utils import load_dataset


__author__ = 'Christopher Potts'


src_dirname = 'dynasent-v1.1'

r1_filename_template = os.path.join(
    src_dirname,
    'dynasent-v1.1-round01-yelp-{}.jsonl')

r2_filename_template = os.path.join(
    src_dirname,
    'dynasent-v1.1-round02-dynabench-{}.jsonl')

sst_filename = os.path.join(
    src_dirname,
    'sst-dev-validated.jsonl')


@pytest.fixture
def dataset():
    data = {
        'r1_train': load_dataset(r1_filename_template.format('train')),
        'r1_dev': load_dataset(r1_filename_template.format('dev')),
        'r1_test': load_dataset(r1_filename_template.format('test')),
        'r2_train': load_dataset(r2_filename_template.format('train')),
        'r2_dev': load_dataset(r2_filename_template.format('dev')),
        'r2_test': load_dataset(r2_filename_template.format('test')),
        'sst': load_dataset(sst_filename)}
    return data


@pytest.mark.parametrize('round', [1, 2])
def test_no_sentence_overlap(round, dataset):
    train_s = {d['sentence'] for d in dataset[f'r{round}_train']}
    dev_s = {d['sentence'] for d in dataset[f'r{round}_dev']}
    test_s = {d['sentence'] for d in dataset[f'r{round}_test']}
    assert len(train_s & dev_s) == 0
    assert len(dev_s & test_s) == 0
    assert len(train_s & test_s) == 0


@pytest.mark.parametrize('split', [
    'r1_train',
    'r1_dev',
    'r1_test',
    'r2_train',
    'r2_dev',
    'r2_test'
])
def test_no_repeated_sentences(split, dataset):
    sents = [d['sentence'] for d in dataset[split]]
    assert len(sents) == len(set(sents))


@pytest.mark.parametrize('split, model_key, expected', [
    ('r1_dev', 'model_0_label', 400),
    ('r1_test', 'model_0_label', 400),
    ('r2_dev', 'model_1_label', 80),
    ('r2_test', 'model_1_label', 80)
])
def test_gold_vs_model_assess(split, model_key, expected, dataset):
    dist = Counter([(d[model_key], d['gold_label']) for d in dataset[split]])
    assert all(x == expected for x in dist.values())


@pytest.mark.parametrize('split, expected', [
    ['r1_dev', 600],
    ['r1_test', 600]
])
def test_gold_vs_rating_round1_assess(split, expected, dataset):
    dist = Counter([(d['review_rating'], d['gold_label']) for d in  dataset[split]])
    assert all(x == expected for x in dist.values())


@pytest.mark.parametrize('split', [
    'r1_train',
    'r1_dev',
    'r1_test',
    'r2_train',
    'r2_dev',
    'r2_test'
])
def test_gold_label_inference(split, dataset):
    for d in dataset[split]:
        dist = [(len(v), k) for k, v in d['label_distribution'].items()]
        dist = sorted(dist)
        count, cls = dist[-1]
        if count >= 3:
            gold_label = cls
        else:
            gold_label = None
        assert d['gold_label'] == gold_label


@pytest.mark.parametrize('split', [
    'r1_train',
    'r1_dev',
    'r1_test',
    'r2_train',
    'r2_dev',
    'r2_test',
    'sst'
])
def test_unique_annotators(split, dataset):
    for d in dataset[split]:
        workers = {w for vals in d['label_distribution'].values() for w in vals}
        assert len(workers) == 5


@pytest.mark.parametrize('split', [
    'r1_train',
    'r1_dev',
    'r1_test',
    'r2_train',
    'r2_dev',
    'r2_test',
    'sst'
])
def test_no_real_mturk_ids(split, dataset):
    for d in dataset[split]:
        for workers in d['label_distribution'].values():
            for w in workers:
                assert _is_our_anonymized_mturk_id(w)
        if '2' in split:
            assert _is_our_anonymized_mturk_id(d['sentence_author'])


def _is_our_anonymized_mturk_id(s):
    return s.startswith("w") and not any(c.isupper() for c in s)


@pytest.mark.parametrize('split', [
    'r2_train',
    'r2_dev',
    'r2_test'
])
def test_no_round2_self_rating(split, dataset):
    for d in dataset[split]:
        author = d['sentence_author']
        for workers in d['label_distribution'].values():
            for w in workers:
                assert w != author


def test_expected_dataset_size(dataset):
    total = sum(len(exs) for split, exs in dataset.items() if split != 'sst')
    expected = 121634
    assert total == expected


def test_expected_sst_dev_size(dataset):
    total = len(dataset['sst'])
    expected = 1101
    assert total == expected


def test_no_round2_assess_set_repeated_prompts(dataset):
    all_prompts = defaultdict(list)
    for split in ('r2_train', 'r2_dev', 'r2_test'):
        for d in dataset[split]:
            if d['has_prompt']:
                prompt_sentence = d['prompt_data']['prompt_sentence']
                all_prompts[prompt_sentence].append(d)
    for split in ('r2_dev', 'r2_test'):
        for d in dataset[split]:
            prompt_sentence = d['prompt_data']['prompt_sentence']
            assert len(all_prompts[prompt_sentence]) == 1


def test_unique_ids(dataset):
    all_ids = []
    for split, exs in dataset.items():
        all_ids += [d['text_id'] for d in exs]
    assert len(all_ids) == len(set(all_ids))
