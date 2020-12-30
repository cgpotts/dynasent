import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from sklearn.metrics import precision_recall_fscore_support
from statsmodels.stats.inter_rater import fleiss_kappa


__author__ = 'Christopher Potts'


pd.set_option('max_colwidth', 999)
pd.set_option('display.max_rows', 999)
pd.set_option('display.max_columns', 999)


ALL_CATS = ('positive', 'negative', 'neutral', 'mixed')
TERNARY_CATS = ('positive', 'negative', 'neutral')


def load_dataset(*src_filenames, labels=None):
    data = []
    for filename in src_filenames:
        with open(filename) as f:
            for line in f:
                d = json.loads(line)
                if labels is None or d['gold_label'] in labels:
                    data.append(d)
    return data


def get_label_distribution(*splits, dist_labels=False):
    if dist_labels:
        all_labels = []
        for split in splits:
            for d in split:
                dist = d['label_distribution']
                all_labels += [label for label, ids in dist.items()
                               for _ in range(len(ids))]
        series = pd.Series(all_labels)
    else:
        df = pd.concat((pd.DataFrame(split) for split in splits))
        series = df.gold_label
        series = series.fillna("No Majority")
    dist = series.value_counts(dropna=False)
    dist['Total'] = dist.sum()
    return dist


def get_label_model_relationship(*splits, model_colname='model_0_label'):
    all_splits = sum(splits, [])
    df = pd.DataFrame(all_splits)
    return df.groupby(['gold_label', model_colname]).apply(len)


def get_adversarial_rate(*splits, model_colname='model_0_label', labels=None):
    dist = get_label_model_relationship(*splits, model_colname=model_colname)
    dist = dist.reset_index().rename(columns={0: 'examples'})
    total = dist.examples.sum()
    if labels is not None:
        dist = dist[dist.gold_label.isin(labels)]
    adversarial = dist[dist.gold_label != dist[model_colname]]
    return adversarial.examples.sum(), total


def get_label_rating_relationship(*splits):
    all_splits = sum(splits, [])
    df = pd.DataFrame(all_splits)
    return df.groupby(['gold_label', 'review_rating']).apply(len)


def get_dist_of_majority_dists(split):
    data = []
    for d in split:
        if d['gold_label']:
            dist = sorted([(len(v), k) for k, v in d['label_distribution'].items()])
            c = dist[-1][0]
            data.append(c)
    return pd.Series(data).value_counts()


def get_global_worker_dist(*splits):
    all_workers = []
    for split in splits:
        for d in split:
            for workers in d['label_distribution'].values():
                all_workers += workers
    series = pd.Series(all_workers)
    return series.value_counts()


def plot_global_worker_dist(worker_dist, output_filename=None):
    ax = worker_dist.apply(np.log).hist(bins=10, figsize=(8, 5))
    ax.set_xlabel("Examples labeled (log scale)")
    ax.set_ylabel("Workers")
    if output_filename is not None:
        plt.tight_layout()
        plt.savefig(output_filename, dpi=200)


def sample_short_examples(split, model_colname, minlength=30, maxlength=50, random_state=1):
    df = pd.DataFrame(split)
    # No cases without majorities:
    df = df[df.gold_label.isnull() == False]
    # Length restrictions:
    df = df[
        (df.sentence.str.len() > minlength) &
        (df.sentence.str.len() < maxlength)]
    # One example per combination of gold and predicted:
    samp = df.groupby(['gold_label', model_colname]).apply(
        lambda x: x.sample(1, random_state=random_state))
    # Abbreviated label distribution for display:
    samp['responses'] = samp.label_distribution.apply(_format_dist)
    # Final formatting:
    samp = samp[['sentence', model_colname, 'responses']].reset_index(drop=True)
    samp = samp.sort_values(model_colname)
    return samp


def sample_examples(split, model_colname, n=10, random_state=1):
    df = pd.DataFrame(split)
    df = df[df.gold_label.isnull() == False]
    samp = df.sample(n, random_state=random_state)
    samp['responses'] = samp.label_distribution.apply(_format_dist)
    samp = samp[['sentence', model_colname, 'responses']].reset_index(drop=True)
    return samp


def _format_dist(d):
    vals = sorted([k[: 3] for k, v in d.items() for _ in range(len(v))])
    return ", ".join(vals)


def get_fleiss_kappa(split, cats=ALL_CATS):
    mat = np.zeros((len(split), len(cats)))
    for i, d in enumerate(split):
        for j, cat in enumerate(cats):
            mat[i][j] += len(d['label_distribution'][cat])
    return fleiss_kappa(mat)


def estimate_human_precision_recall_f1(split, cats=TERNARY_CATS, random_state=1):
    random.seed(random_state)
    y = []
    all_responses = []
    for i, d in enumerate(split):
        gold_label = d['gold_label']
        if gold_label is not None:
            y.append(gold_label)
            responses = [c for c, vals in d['label_distribution'].items()
                         for _ in range(len(vals))]
            random.shuffle(responses)
            all_responses.append(responses)
    all_scores = []
    for i in range(len(all_responses[0])):
        ann = [x[i] for x in all_responses]
        scores = precision_recall_fscore_support(y, ann, labels=cats)
        scores = np.array(scores)
        all_scores.append(scores)
    mu = np.mean(all_scores, axis=0)
    df = pd.DataFrame(
        mu.T,
        index=cats,
        columns=['precision', 'recall', 'F1', 'support'])
    df.loc['Macro avg'] = df.mean(axis=0)
    return df


def get_worker_agreement(split):
    data = []
    for d in split:
        gold_label = d['gold_label']
        if gold_label is not None:
            for c, workers in d['label_distribution'].items():
                for w in workers:
                    data.append({
                        'worker': w,
                        'response': c,
                        'gold_label': gold_label})

    df = pd.DataFrame(data)

    def agr_data(grp_df):
        n_maj = grp_df[grp_df.response == grp_df.gold_label].shape[0]
        n_total = grp_df.shape[0]
        rate = n_maj / n_total
        return pd.Series({
            'n_majority': n_maj,
            'n_total': n_total,
            'rate': rate})

    return df.groupby('worker').apply(agr_data)


def plot_worker_agreement(dev, test, output_filename=None):
    dev_df = get_worker_agreement(dev)
    dev_df['split'] = "dev"
    test_df = get_worker_agreement(test)
    test_df['split'] = "test"

    df = pd.concat((dev_df, test_df))
    plotstuff = df.boxplot(
        'rate', by='split',
        figsize=(5, 4), notch=True,
        return_type='both')

    boxes = plotstuff.rate.lines
    ax = plotstuff.rate.ax

    splits_with_colors = ((dev_df, "#E24A33"), (test_df, "#8EBA42"))
    for i, (split, color) in enumerate(splits_with_colors, start=1):
        ax.scatter(
            jitter(i, len(split)),
            split['rate'].values,
            marker=".",
            alpha=0.2,
            color=color)

    ax.set_ylabel("Rate")
    ax.set_xlabel("")
    _extra_boxplot_styling(boxes)

    if output_filename:
        plt.tight_layout()
        plt.savefig(output_filename, dpi=200)


def prompt_cmp_plot(
        df, colname, ylabel, n_per_group=500,
        output_filename=None, ylim=[0, 40],
        random_state=1):

    df = df.groupby('prompt').apply(
        lambda x: x.sample(n_per_group, random_state=random_state))
    df = df.reset_index(drop=True)

    plotstuff = df.boxplot(
        colname, by='prompt',
        notch=True, figsize=(5, 5),
        return_type='both')

    boxes = plotstuff[colname].lines
    ax = plotstuff[colname].ax

    _extra_boxplot_styling(boxes)

    colors = ["#777777", "#348ABD", "#E24A33"]

    for i, x in enumerate(("No Prompt", "Prompt", "Yelp"), start=1):
        sf = df[df.prompt == x]
        ax.scatter(
            jitter(i, len(sf)),
            sf[colname].values,
            marker=".",
            alpha=0.2,
            color=colors[i-1])

    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    ax.set_ylim(ylim)

    if output_filename:
        plt.tight_layout()
        plt.savefig(output_filename, dpi=200)


def vocab_diversity_experiments(
        df, n_ex=200, n_words=5, n_experiments=500,
        output_filename=None, random_state=1):

    random.seed(random_state)

    df = df[df.token_length >= n_words]

    def func(sf):
        samp = sf.sample(n_ex)
        vocab = set()
        for toks in samp['tokens'].values:
            toks = random.sample(toks, n_words)
            vocab |= set(toks)
        return len(vocab)

    dfs = []
    for i in range(n_experiments):
        dfs.append(df.groupby('prompt').apply(func))

    results = pd.concat(dfs, axis=1).T
    results = results.to_dict(orient='records')
    data = []
    for d in results:
        for k, v in d.items():
            data.append({'prompt': k, 'value': v})
    data = pd.DataFrame(data)

    plotstuff = data.boxplot(
        "value",
        by='prompt',
        notch=True,
        figsize=(5, 5),
        return_type='both')

    boxes = plotstuff["value"].lines
    ax = plotstuff["value"].ax

    colors = ["#777777", "#348ABD", "#E24A33"]

    for i, x in enumerate(("No Prompt", "Prompt", "Yelp"), start=1):
        sf = data[data.prompt == x]
        ax.scatter(
            jitter(i, len(sf)),
            sf['value'].values,
            marker=".",
            alpha=0.2,
            color=colors[i-1])

    _extra_boxplot_styling(boxes)

    ax.set_xlabel("")
    ax.set_ylabel("Vocab size")

    if output_filename:
        plt.tight_layout()
        plt.savefig(output_filename, dpi=200)


def jitter(x, n):
    return np.random.uniform(x-0.1, x+0.1, n)


def _extra_boxplot_styling(boxes):
    plt.suptitle("")
    plt.title("")
    plt.grid(b=True, axis='y')
    # Not sure why these aren't being set by the style sheet:
    plt.setp(boxes['boxes'], color='black', fill=False)
    plt.setp(boxes['whiskers'], color='black')
    plt.setp(boxes['medians'], color='black', lw=2)
