# Model Card for DynaSent Model 0 and Model 1

[Christopher Potts](http://web.stanford.edu/~cgpotts/), [Zhengxuan Wu](http://zen-wu.social), Atticus Geiger, and [Douwe Kiela](https://douwekiela.github.io). 2020. DynaSent: A dynamic benchmark for sentiment analysis. Ms., Stanford University and Facebook AI Research.


## Model details

* Details on how to obtain and use these models are [here](README.md#models).

* Developed by Christopher Potts, Zhengxuan Wu, Atticus Geiger, and Douwe Kiela.

* Extend [RoBERTa-base parameters](https://github.com/pytorch/fairseq/tree/master/examples/roberta) with a three-way classifier head  for ternary (positive/negative/neutral) sentiment analysis. The model parameters are trained/fine-tuned against a cross-entropy loss.

* Implemented in [PyTorch](https://pytorch.org), and distributed with a [Hugging Face](https://huggingface.co)-based wrapper to use them for prediction: [dynasent_models.py](dynasent_models.py).


## Intended use

* The primary intended use is in sentiment analysis of the texts of product and service reviews, and this is the domain in which the model has been evaluated to date.

* We urge caution about using these models for sentiment prediction in other domains. For example, sentiment expression in medical contexts and professional evaluations can be different from sentiment expression in product/service reviews.


## Factors

* Model 0 and Model 1 are evaluated on product/service reviews from Yelp and Amazon, on sentences from Rotten Tomatoes (as given by the [Stanford Sentiment Treebank](https://nlp.stanford.edu/sentiment/)), and on the DynaSent dataset. Extensive details are given in our paper.

* The labels in the training and evaluation data are provided by workers on Amazon Mechanical Turk. For details on the work they did and how they were compensated, see our [Datasheet](dynasent_datasheet.md).


## Metrics

* Our primary evaluation metrics are precision, recall, and F1.

* We macro-average these scores across our three classes in our evaluations, some of which have imbalanced class sizes.


## Training data

* Both models were trained on product/service reviews from Yelp, reviews from Amazon, reviews from IMDB (as defined by [this dataset](https://ai.stanford.edu/~amaas/data/sentiment/)), sentences from Rotten Tomatoes (as given by the [Stanford Sentiment Treebank](https://nlp.stanford.edu/sentiment/)), the [Customer Reviews dataset](https://www.cs.uic.edu/~liub/FBS/sentiment-analysis.html), and on subsets of the DynaSent dataset. Extensive details on these datasets are included in our paper.

* Our evaluation data are described above under [Factors](#factors).


## Ethical Considerations

* Our models are trained on data that are publicly available, on commercial websites and/or as pre-packaged datasets used widely by the AI community.

* For discussion of the ethical considerations underlying the DynaSent dataset, see our [Datasheet](dynasent_datasheet.md).


## Caveats and recommendations

* We recommend careful study of how these models behave, even when they are used in the domain on which they were trained and assessed. The models are deep learning models about which it is challenging to gain full analytic command; two examples that appear synonymous to human readers can receive very different predictions from these models, in ways that are hard to anticipate or explain, and so it is crucial to do continual qualitative and quantitative evaluation as part of any deployment.

* We advise even more caution when using these models in new domains, as sentiment expression can shift in subtle (and not-so-subtle) ways across different domains, and this could lead specific phenomena to be mis-handled in ways that could have dramatic and pernicious consequences.
