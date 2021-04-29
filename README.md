# DynaSent: Dynamic Sentiment Analysis Dataset

DynaSent is an English-language benchmark task for ternary (positive/negative/neutral) sentiment analysis.

## Contents

* [Citation](#Citation)
* [Dataset files](#dataset-files)
* [Quick start](#quick-start)
* [Data format](#data-format)
* [Models](#models)
* [Other files](#other-files)
* [License](#license)


## Citation

[Christopher Potts](http://web.stanford.edu/~cgpotts/), [Zhengxuan Wu](http://zen-wu.social), Atticus Geiger, and [Douwe Kiela](https://douwekiela.github.io). 2020. [DynaSent: A dynamic benchmark for sentiment analysis](https://arxiv.org/abs/2012.15349). Ms., Stanford University and Facebook AI Research.

```stex
  @article{potts-etal-2020-dynasent,
    title={{DynaSent}: A Dynamic Benchmark for Sentiment Analysis},
    author={Potts, Christopher and Wu, Zhengxuan and Geiger, Atticus and Kiela, Douwe},
    journal={arXiv preprint arXiv:2012.15349},
    url={https://arxiv.org/abs/2012.15349},
    year={2020}}
```

## Dataset files

The dataset is [dynasent-v1.1.zip](dynasent-v1.1.zip), which is included in this repository. `v1.1` differs from `v1` only in that `v1.1` has proper unique ids for Round 1 and corrects a bug that led to some non-unique ids in Round 2. There are no changes to the examples or other metadata.

The dataset consists of two rounds, each with a train/dev/test split:


### Round 1: Naturally occurring sentences

* `dynasent-v1.1-round01-yelp-train.jsonl`
* `dynasent-v1.1-round01-yelp-dev.jsonl`
* `dynasent-v1.1-round01-yelp-test.jsonl`


### Round 2: Sentences crowdsourced using Dynabench

* `dynasent-v1.1-round02-dynabench-train.jsonl`
* `dynasent-v1.1-round02-dynabench-dev.jsonl`
* `dynasent-v1.1-round02-dynabench-test.jsonl`


### SST-dev revalidation

The dataset also contains a version of the [Stanford Sentiment Treebank](https://nlp.stanford.edu/sentiment/) dev set in our format with labels from our validation task:

* `sst-dev-validated.jsonl`


## Quick start

This function can be used to load any subset of the files:

```python
import json

def load_dataset(*src_filenames, labels=None):
    data = []
    for filename in src_filenames:
        with open(filename) as f:
            for line in f:
                d = json.loads(line)
                if labels is None or d['gold_label'] in labels:
                    data.append(d)
    return data
```

For example, to create a Round 1 train set restricting to examples with ternary gold labels:

```python
import os

r1_train_filename = os.path.join('dynasent-v1.1', 'dynasent-v1.1-round01-yelp-train.jsonl')

ternary_labels = ('positive', 'negative', 'neutral')

r1_train = load_dataset(r1_train_filename, labels=ternary_labels)

X_train, y_train = zip(*[(d['sentence'], d['gold_label']) for d in r1_train])
```

DynaSent rounds can also be accessed directly using the [HuggingFace Datasets](https://huggingface.co/datasets/dynabench/dynasent) library:

```python
"""
Make sure you install the Datasets library using:
pip install datasets
"""
from datasets import load_dataset

r1_dataset = load_dataset("dynabench/dynasent", "dynabench.dynasent.r1.all")

r2_dataset = load_dataset("dynabench/dynasent", "dynabench.dynasent.r2.all")
```

## Data format

### Round 1 format

```python
{'hit_ids': ['y5238'],
 'sentence': 'Roto-Rooter is always good when you need someone right away.',
 'indices_into_review_text': [0, 60],
 'model_0_label': 'positive',
 'model_0_probs': {'negative': 0.01173639390617609,
  'positive': 0.7473671436309814,
  'neutral': 0.24089649319648743},
 'text_id': 'r1-0000001',
 'review_id': 'IDHkeGo-nxhqX4Exkdr08A', 
 'review_rating': 1,
 'label_distribution': {'positive': ['w130', 'w186', 'w207', 'w264', 'w54'],
  'negative': [],
  'neutral': [],
  'mixed': []},
 'gold_label': 'positive'}
```

Details:

* `'hit_ids'`: List of Amazon Mechanical Turk Human Interface Tasks (HITs) in which this example appeared during validation. The values are anonymized but used consistently throughout the dataset.
* `'sentence'`: The example text.
* `'indices_into_review_text':` indices of `'sentence'` into the original review in the [Yelp Academic Dataset](https://www.yelp.com/dataset).
* `'model_0_label'`: prediction of Model 0 as described in the paper. The possible values are `'positive'`, `'negative'`, and `'neutral'`.
* `'model_0_probs'`: probability distribution predicted by Model 0. The keys are `('positive', 'negative', 'neutral')` and the values are floats.
* `'text_id'`: unique identifier for this entry.
* `'review_id'`: review-level identifier for the review from the [Yelp Academic Dataset](https://www.yelp.com/dataset) containing `'sentence'`.
* `'review_rating'`: review-level star-rating for the review containing `'sentence'` in the [Yelp Academic Dataset](https://www.yelp.com/dataset). The possible values are `1`, `2`, `3`, `4`, and `5`.
* `'label_distribution':` response distribution from the MTurk validation task. The keys are `('positive', 'negative', 'neutral')` and the values are lists of anonymized MTurk ids, which are used consistently throughout the dataset.
* `'gold_label'`: the label chosen by at least three of the five workers if there is one (possible values: `'positive'`, `'negative'`, '`neutral'`, and `'mixed'`), else `None`. 

Here is some code one could use to augment a dataset, as loaded by `load_dataset`, with a field giving the full review text from the [Yelp Academic Dataset](https://www.yelp.com/dataset):

```python
import json

def index_yelp_reviews(yelp_src_filename='yelp_academic_dataset_review.json'):
    index = {}
    with open(yelp_src_filename) as f:
        for line in f:
            d = json.loads(line)
            index[d['review_id']] = d['text']
    return index

yelp_index = index_yelp_reviews()

def add_review_text_round1(dataset, yelp_index):
    for d in dataset:
        review_text = yelp_index[d['text_id']]
        # Check that we can find the sentence as expected:
        start, end = d['indices_into_review_text']
        assert review_text[start: end] == d['sentence']
        d['review_text'] = review_text
    return dataset
```

### Round 2 format

```python
{'hit_ids': ['y22661'],
 'sentence': "We enjoyed our first and last meal in Toronto at Bombay Palace, and I can't think of a better way to book our journey.",
 'sentence_author': 'w250',
 'has_prompt': True,
 'prompt_data': {'indices_into_review_text': [2093, 2213],
  'review_rating': 5,
  'prompt_sentence': "Our first and last meals in Toronto were enjoyed at Bombay Palace and I can't think of a better way to bookend our trip.",
  'review_id': 'Krm4kSIb06BDHternF4_pA'},
 'model_1_label': 'positive',
 'model_1_probs': {'negative': 0.29140257835388184,
  'positive': 0.6788994669914246,
  'neutral': 0.029697999358177185},
 'text_id': 'r2-0000001',
 'label_distribution': {'positive': ['w43', 'w26', 'w155', 'w23'],
  'negative': [],
  'neutral': [],
  'mixed': ['w174']},
 'gold_label': 'positive'}
 ```

Details:

* `'hit_ids'`: List of Amazon Mechanical Turk Human Interface Tasks (HITs) in which this example appeared during validation. The values are anonymized but used consistently throughout the dataset.
* `'sentence'`: The example text.
* `'sentence_author'`:  Anonymized MTurk id of the worker who wrote `'sentence'`. These are from the same family of ids as used in `'label_distribution'`, but this id is never one of the ids in `'label_distribution'` for this example.
* `'has_prompt'`: `True` if the `'sentence'` was written with a Prompt else `False`.
* `'prompt_data'`: None if `'has_prompt'` is False, else:
  * `'indices_into_review_text'`: indices of `'prompt_sentence'` into the original review in the [Yelp Academic Dataset](https://www.yelp.com/dataset).
  * `'review_rating'`: review-level star-rating for the review containing `'sentence'` in the [Yelp Academic Dataset](https://www.yelp.com/dataset).
  * `'prompt_sentence'`: The prompt text.
  * `'review_id'`: review-level identifier for the review from the [Yelp Academic Dataset](https://www.yelp.com/dataset) containing `'prompt_sentence'`.
* `'model_1_label'`: prediction of Model 1 as described in the paper. The possible values are `'positive'`, `'negative'`, and '`neutral'`.
* `'model_1_probs'`: probability distribution predicted by Model 1. The keys are `('positive', 'negative', 'neutral')` and the values are floats.  
* `'text_id'`: unique identifier for this entry.
* `'label_distribution'`: response distribution from the MTurk validation task. The keys are  `('positive', 'negative', 'neutral')` and the values are lists of anonymized MTurk ids, which are used consistently throughout the dataset.
* `'gold_label'`: the label chosen by at least three of the five workers if there is one (possible values: `'positive'`, `'negative'`, '`neutral'`, and `'mixed'`), else `None`. 

To add the review texts to the `'prompt_data'` field, one can extend the code above for Round 1 with the following function:

```python
def add_review_text_round2(dataset, yelp_index):
    for d in dataset:
        if d['has_prompt']:
            prompt_data = d['prompt_data']
            review_text = yelp_index[prompt_data['review_id']]
            # Check that we can find the sentence as expected:
            start, end = prompt_data['indices_into_review_text']
            assert review_text[start: end] == prompt_data['prompt_sentence']
            prompt_data['review_text'] = review_text
    return dataset
```

### SST-dev format

```python
{'hit_ids': ['s20533'],
 'sentence': '-LRB- A -RRB- n utterly charming and hilarious film that reminded me of the best of the Disney comedies from the 60s.',
 'tree': '(4 (2 (1 -LRB-) (2 (2 A) (3 -RRB-))) (4 (4 (2 n) (4 (3 (2 utterly) (4 (3 (4 charming) (2 and)) (4 hilarious))) (3 (2 film) (3 (2 that) (4 (4 (2 (2 reminded) (3 me)) (4 (2 of) (4 (4 (2 the) (4 best)) (2 (2 of) (3 (2 the) (3 (3 Disney) (2 comedies))))))) (2 (2 from) (2 (2 the) (2 60s)))))))) (2 .)))',
 'text_id': 'sst-dev-validate-0000437',
 'sst_label': '4',
 'label_distribution': {'positive': ['w207', 'w3', 'w840', 'w135', 'w26'],
  'negative': [],
  'neutral': [],
  'mixed': []},
 'gold_label': 'positive'}
```

Details:

* `'hit_ids'`: List of Amazon Mechanical Turk Human Interface Tasks (HITs) in which this example appeared during validation. The values are anonymized but used consistently throughout the dataset.
* `'sentence'`: The example text.
* `'tree'`: The parsetree for the example as given in the SST distribution.
* `'text_id'`: A new identifier for this example.
* `'sst_label'`: The root-node label from the SST. Possible values `'0'`, `'1'` `'2'`, `'3'`, and `'4'`.
* `'label_distribution':` response distribution from the MTurk validation task. The keys are  `('positive', 'negative', 'neutral')` and the values are lists of anonymized MTurk ids, which are used consistently throughout the dataset.
* `'gold_label'`: the label chosen by at least three of the five workers if there is one (possible values: `'positive'`, `'negative'`, '`neutral'`, and `'mixed'`), else `None`.


## Models

Model 0 and Model 1 from the paper are available here:

https://drive.google.com/drive/folders/1dpKrjNJfAILUQcJPAFc5YOXUT51VEjKQ?usp=sharing

This repository includes a Python module `dynasent_models.py` that provides a [Hugging Face](https://huggingface.co)-based wrapper around these ([PyTorch](https://pytorch.org)) models. Simple examples:

```python
import os
from dynasent_models import DynaSentModel

# `dynasent_model0` should be downloaded from the above Google Drive link and 
# placed in the `models` directory. `dynasent_model1` works the same way.
model = DynaSentModel(os.path.join('models', 'dynasent_model0.bin'))

examples = [
    "superb",
    "They said the experience would be amazing, and they were right!",
    "They said the experience would be amazing, and they were wrong!"]

model.predict(examples)
```
This should return the list `['positive', 'positive', 'negative']`.

The `predict_proba` method provides access to the predicted distribution over the class labels; see the demo at the bottom of `dynasent_models.py` for details.

The following code uses `load_dataset` from above to reproduce the Round 2 dev-set report on Model 0 from the paper:

```python
import os
from sklearn.metrics import classification_report
from dynasent_models import DynaSentModel

dev_filename = os.path.join('dynasent-v1.1', 'dynasent-v1.1-round02-dynabench-dev.jsonl')

dev = load_dataset(dev_filename)

X_dev, y_dev = zip(*[(d['sentence'], d['gold_label']) for d in dev])

model = DynaSentModel(os.path.join('models', 'dynasent_model0.bin'))

preds = model.predict(X_dev)

print(classification_report(y_dev, preds, digits=3))
```
For a fuller report on these models, see our paper and [our model card](dynasent_modelcard.md).


## Other files


### Analysis notebooks

The following notebooks reproduce the dataset statistics, figures, and random example selections from the paper:

* `analyses_comparative.ipynb`
* `analysis_round1.ipynb`
* `analysis_round2.ipynb`
* `analysis_sst_dev_revalidate.ipynb`

The Python module `dynasent_utils.py` contains functions that support those notebooks, and `dynasent.mplstyle` helps with styling the plots.


### Datasheet

The [Datasheet](https://arxiv.org/abs/1803.09010) for our dataset:

* [dynasent_datasheet.md](dynasent_datasheet.md)


### Model Card

The [Model Card](https://arxiv.org/pdf/1810.03993.pdf) for our models:

* [dynasent_modelcard.md](dynasent_modelcard.md)


### Tests

The module `test_dataset.py` contains PyTest tests for the dataset. To use it, run

```
py.test -vv test_dataset.py
```

in the root directory of this repository.


### Validation HIT code

The file `validation-hit-contents.html` contains the HTML/Javascript used in the validation task. It could be used directly on Amazon Mechanical Turk, by simply pasting its contents into the usual HIT creation window.


## License

DynaSent has a [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/).
