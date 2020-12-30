# Datasheet for the DynaSent, version 1

[Christopher Potts](http://web.stanford.edu/~cgpotts/), [Zhengxuan Wu](http://zen-wu.social), Atticus Geiger, and [Douwe Kiela](https://douwekiela.github.io). 2020. DynaSent: A dynamic benchmark for sentiment analysis. Ms., Stanford University and Facebook AI Research.


## Motivation


### For what purpose was the dataset created?

DynaSent is an English-language benchmark task for ternary (positive/negative/neutral) sentiment analysis. It was created to challenge current models, systems, and approaches to simple evaluative sentiment analysis. We hope to have created a dataset that is of high quality and that current models struggle with, and we hope DynaSent is useful in creating the next generation of models.


### Who created the dataset and on behalf of which entity?

The dataset was created by Christopher Potts (Stanford University), Zengxuan Wu (Stanford University), Atticus Geiger (Stanford University), and Douwe Kiela (Facebook AI Research). All members of the team were functioning as independent researchers within their organizations.


### Who funded the creation of the dataset?

The effort was funded by Google and Facebook faculty research grants to Potts and by money from Stanford University.


## Composition


### What do the instances that comprise the dataset represent?

The instances are English-language sentences with labels and additional metadata. These sentences are meant to be records of acts of linguistic communication.


### How many instances are there in total?

Version 1 of the dataset has 121,634 instances.


### Does the dataset contain all possible instances or is it a sample (not necessarily random) of instances from a larger set?

Round 1 of the dataset consists of sentences sampled from the [Yelp Academic Dataset](https://www.yelp.com/dataset)  (the version with 8,021,122 reviews). These sentences were sampled randomly, but with a bias for sentences occurring in 1-star reviews that our Model 0 predicted to be Positive and sentences occurring in 5-star reviews that our Model 0 predicted to be Negative. The intention behind this biased sampling method is to find sentences that are likely to be challenging for current models.

Round 2 of the dataset consists of sentences written by Amazon Mechanical Turk workers on the [Dynabench Platform](https://dynabench.org/). All the sentences created by these workers under the rubric of our project are included in the dataset.

We think Round 1 is representative of English-language sentences, written in reviews on Yelp, that are likely to be challenging for strong present-day sentiment models. This is a very narrow slice of conceivable English-language sentences conveying evaluative sentiment. The most meaningful restriction imposed here is likely that the domain is product and service reviews, which is likely to be linguistically very different from, for example, the domain of job performance texts or medical notes. We would thus urge extreme caution if one were to assume that models trained on this dataset will generalize in those ways.

The likely biases in Round 2 are less clear to us at present. The sentences were written by a small population of crowdworkers. The majority of the sentences were created from prompt sentences extracted from the [Yelp Academic Dataset](https://www.yelp.com/dataset)  (from different reviews than those used in Round 1). We would thus expect to see the domain influences just reviewed for Round 1, but infused also with the particular attitudes, perspectives, and beliefs of our population of crowdworkers. Our paper makes the case that the prompt-derived sentences have a lot in common with the naturally occurring Yelp examples, but there are sure to be differences as well.


### What data does each instance consist of?

The dataset is released in the JSON lines (JSONL) format. Our [README.md](README.md) documents the dataset format, including glosses for all the data fields.


### Is there a label or target associated with each instance?

Yes, there are multiple labels associated with each example. The primary labels are Positive, Negative, Neutral, Mixed Sentiment, and No Majority. These are inferred from response distributions from Amazon Mechanical Turk workers performing a sentiment validation task. The distribution of such responses is also included in the dataset, using anonymized worker ids. Each example in the dataset also has a number of metadata fields that could be used as labels as well. See our [README.md](README.md) for a complete description.


### Is any information missing from individual instances?

We have tried to include as much contextual information as we can without compromising our goal of ensuring anonymity for all our workers. No text has been altered or redacted. As a result, the dataset may contain sentences that are offensive to readers.


### Are relationships between individual instances made explicit?

We think these relationships are made explicit in the data format and and described in the [README.md](README.md). We will update the [README.md](README.md) if people contact us to ask for clarification.


### Are there recommended data splits (e.g., training, development/validation, testing)?

Yes, both Round 1 and Round 2 come with predefined train/dev/test splits. The nature of these splits is important to the dataset itself; as described in the paper, we deliberately constructed splits in which our best models to date perform at chance on the dev and test sets.


### Are there any errors, sources of noise, or redundancies in the dataset?

Errors in some sense are sure to exist, but we have tried to minimize them. For example, our unit tests (included with the corpus distribution) verify that there are no sentences shared across the train/dev/test splits for either round. Some of the examples are not sentences by traditional linguistic criteria, due to a variety of factors &mdash; errors in our sentence tokenization, non-standard punctuation, etc. &mdash; but we think this is likely to be useful variation for creating robust systems.

If errors are discovered in the dataset, the distribution and documentation will be updated.


### Is the dataset self-contained, or does it link to or otherwise rely on external resources?

The dataset is self-contained, but it does include review ids and offsets for sentences that occur in the [Yelp Academic Dataset](https://www.yelp.com/dataset). This is meant to help users bring in added context from the [Yelp Academic Dataset](https://www.yelp.com/dataset) if they wish. However, we cannot guarantee that the [Yelp Academic Dataset](https://www.yelp.com/dataset) will always be available, as Yelp is within its rights to remove it from public distribution at any time. Our dataset would remain intact even if this did happen.

There are no fees associated with using our dataset. We impose no restrictions on its usage ourselves, but we are also not in a position to adjudicate the question of whether it inherits [the terms of the Yelp Academic Dataset](https://s3-media3.fl.yelpcdn.com/assets/srv0/engineering_pages/bea5c1e92bf3/assets/vendor/yelp-dataset-agreement.pdf) in virtue of using text snippets from that resources as example sentences (Round 1) and as prompt sentences (Round 2). This is a legal matter about which we have no expertise ourselves.

We ourselves have an archival version of the dataset that includes the full review texts extracted from the [Yelp Academic Dataset](https://www.yelp.com/dataset). Our [README.md](README.md) provides the code snippets that enable others to create such a version assuming they have access to the [Yelp Academic Dataset](https://www.yelp.com/dataset).


### Does the dataset contain data that might be considered confidential?

Round 1 of the dataset contains only text snippets that are derived from the [Yelp Academic Dataset](https://www.yelp.com/dataset). These are all from public reviews that Yelp decided to include in its dataset release.

Round 2 consists of original sentences, most of them derived from prompt sentenes that also come from the  [Yelp Academic Dataset](https://www.yelp.com/dataset). We assume that crowdworkers have not disclosed private information in these sentences.


### Does the dataset contain data that, if viewed directly, might be offensive, insulting, threatening, or might otherwise cause anxiety?

It is possible that the dataset contains sentences that would be offensive to some, or trigger unwelcome thoughts or emotions in them. For instance, online reviews can be critical and insensitive, and this content might be especially sensitive for people who have themselves been targeted by critical reviews and other forms of negative feedback on the internet. We have not endeavored to filter out sentences that might cause offense.


### Does the dataset relate to people?

Yes, in virtue of the fact that it contains records of communicative acts by people, and often about people, we consider this to be a dataset that relates closely to people.


### Does the dataset identify any subpopulations (e.g., by age, gender)?

It does not do this in any way that we are aware of, and it is not out intention to identify specific populations.


### Is it possible to identify individuals (i.e., one or more natural persons), either directly or indirectly (i.e., in combination with other data) from the dataset?

Yes, there is a path to such identification for Round 1 examples. One could go from our corpus entry to the review in the [Yelp Academic Dataset](https://www.yelp.com/dataset), and from there one could discover what product or service the review was about. The [Yelp Academic Dataset](https://www.yelp.com/dataset) provides additional metadata, including website links and contact information. Our dataset does not increase such identification possibilities beyond what is in the [Yelp Academic Dataset](https://www.yelp.com/dataset).


### Does the dataset contain data that might be considered sensitive in any way (e.g., data that reveals racial or ethnic origins, sexual orientations, religious beliefs, political opinions or union memberships, or locations; financial or health data; biometric or genetic data; forms of government identification, such as social security numbers; criminal history)?

We believe the answer to be "No", except insofar as the [Yelp Academic Dataset](https://www.yelp.com/dataset) might itself contain such information.


## Collection Process

###  How was the data associated with each instance acquired?

For Round 1, the examples were sampled from the [Yelp Academic Dataset](https://www.yelp.com/dataset) using a model ("Model 0" in the paper) to help find examples that we thought would be challenging for current sentiment models. These sentences come with metadata included in the  [Yelp Academic Dataset](https://www.yelp.com/dataset).

The sampled examples were then fed into a validation task in which multiple crowdworkers assigned sentiment labels to the sentences. The HTML/JavaScript interface they used for this is reproduced in the paper's supplementary materials and included with the dataset distribution. A gold label is then inferred from these response distributions: where a true majority of responses are for label *y*, the gold label is *y*; where there is no such true majority, no gold label is inferred. Our corpus distribution include the full response distributions for each example.

For Round 2, the sentences were created on the [Dynabench Platform](https://dynabench.org/) and fed into the same validation task we used for Round 1.

We make no inferences about the identities of the workers who wrote original sentences in Round 2 or worked on our validation task, and we include no information about these people in our corpus (nor did we collect such information). The worker ids included in the corpus are internally consistent within the corpus but cannot be linked to Mechanical Turk worker ids or any other identifiers.


### What mechanisms or procedures were used to collect the data?

All the information in the corpus was collected using Web applications. The validation HTML/JavaScript for the validation task is included with the corpus distribution, and the [Dynabench Platform](https://dynabench.org/) is publicly available and open source.


### If the dataset is a sample from a larger set, what was the sampling strategy?

For Round 1, sentences were sampled according to [the method described above](#composition). 

For the validation task in both rounds, we got more than 5 responses for most examples and chose the 5 that were by the workers who seemed most reliable to us in that they agreed with other workers at the highest rates for examples they rated. This allowed us to remove responses from workers who seemed to be inattentive or not fully understand the examples or the nature of the task. However, this is a potential source of bias, as it is, by definition, a process that reduces the number of perspectives represented in the responses.


### Who was involved in the data collection process (e.g., students, crowdworkers, contractors) and how were they compensated (e.g., how much were crowdworkers paid)?

The data collection was done through Amazon Mechanical Turk. For the validation task, workers were paid US$0.25 to rate 10 examples. For Round 2, workers were paid US$0.30 for each example they created, and they were given a US$0.30 bonus for each example that actually fooled the model according to our validation task.


### Over what timeframe was the data collected?

The validation task and the Round 2 sentence-creation task were both done in the time period of September 2020 to December 2020. The sentences that comprise the Round 1 dataset are derived from the [Yelp Academic Dataset](https://www.yelp.com/dataset), which includes reviews from October 2004 to December 2019.


### Were any ethical review processes conducted?

Yes, the entire project was conducted under the Stanford University IRB Protocol 'Pragmatic enrichment and contextual inference' (PI Potts). Workers were shown information about this protocol at the start of the task, including information about how to contact the PI and the IRB.


### Did you collect the data from individuals directly, or obtain it via third parties or other sources (e.g., websites)?

The Round 1 examples are derived from the [Yelp Academic Dataset](https://www.yelp.com/dataset), and the Round 2 examples are obtained directly from crowdworkers writing texts specifically for our task.


### Were the individuals in question notified about the data collection?

Yes, the data collection was the purpose of the tasks we conducted ourselves.

We did not contact the authors included in the [Yelp Academic Dataset](https://www.yelp.com/dataset) for permission to use excerpts from their reviews in our corpus. 


### Did the individuals in question consent to the collection and use of their data?

We assume that participation in our tasks constitutes consent.


### If consent was obtained, were the consenting individuals provided with a mechanism to revoke their consent in the future or for certain uses?

Workers know from our protocol information that they have the right to request that their data be removed from the dataset at any time.


### Has an analysis of the potential impact of the dataset and its use on data subjects (e.g., a data protection impact analysis) been conducted?

We have not conducted such an analysis. However, we do not consider our dataset to belong to the sort of high-risk category that would require such an analysis. We are open to reconsidering this decision at a future date based on changing regulation or new information about how our dataset is used.


## Preprocessing/cleaning/labeling


### Was any preprocessing/cleaning/labeling of the data done (e.g., discretization or bucketing, tokenization, part-of-speech tagging, SIFT feature extraction, removal of instances, processing of missing values)?

The only preprocessing we have done is sentence tokenizing of the reviews in the [Yelp Academic Dataset](https://www.yelp.com/dataset) to extract Round 1 texts.


### Was the "raw" data saved in addition to the preprocessed/cleaned/labeled data (e.g., to support unanticipated future uses)?

The only raw data in this context would be the [Yelp Academic Dataset](https://www.yelp.com/dataset). The corpus distribution includes the offsets of the extracted texts into the original reviews.


### Is the software used to preprocess/clean/label the instances avail- able?

Yes, the sentence tokenizer is the one that comes with NLTK:

```python
from nltk.tokenize import sent_tokenize
```


## Uses


### Has the dataset been used for any tasks already?

As of this writing, the data have been used only for the experiments reported in the paper.


### Is there a repository that links to any or all papers or systems that use the dataset?

The current repository links to our paper. There is not currently a website or repository for keeping track of all papers that make use of the dataset.


### What (other) tasks could the dataset be used for?

The primary intended use of the dataset is to create ternary sentiment models, and to help scientists understand this sentiment domain better. We hope also that the dataset is useful for benchmarking architectures for natural language understanding and multi-task learning, and for addressing other more general questions in NLP and machine learning.

Based on qualitative study of the corpus examples, we think the corpus is useful as a repository of examples that rely heavily on non-literal language use (e.g., hyperbole, sarcasm). We have not labeled the examples in this way, but some applications in this space would not require such labeling.

Those are the NLP applications that we can imagine. There are also industrial applications of the data and models trained on the data, and these are likely to be much more wide-ranging and hard to control.


### Is there anything about the composition of the dataset or the way it was collected and preprocessed/cleaned/labeled that might impact future uses?

Yes. For example, [we noted above](#composition) that we are unsure models trained on this dataset will generalize to high-impact domains like job performance reviews and medical records, but some applications might assume that this is possible, and harm could come of that assumption.


### Are there tasks for which the dataset should not be used?

We are not presently able to identify tasks for which the dataset should not be used, beyond the vast number of clear misapplications one can imagine. If we learn of problematic uses, we will update this section.


## Distribution


### Will the dataset be distributed to third parties outside of the entity (e.g., company, institution, organization) on behalf of which the dataset was created?

Yes, the dataset is publicly available.


### How will the dataset will be distributed?

It is distributed via the current repository.


### When will the dataset be distributed?

It is presently available.


### Will the dataset be distributed under a copyright or other intellectual property (IP) license, and/or under applicable terms of use (ToU)?

We have released the dataset under a [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/), and the associated code in this repository has an Apache 2.0 license.


### Have any third parties imposed IP-based or other restrictions on the data associated with the instances?

No.


### Do any export controls or other regulatory restrictions apply to the dataset or to individual instances?

Not that we are aware.


## Maintenance


### Who is supporting/hosting/maintaining the dataset?

The creators of the dataset are supporting and maintaining it. The dataset is hosted in a personal repository of Potts.


### How can the owner/curator/manager of the dataset be contacted (e.g., email address)?

Potts can be contacted via messaging functions built in to Github, or via contact information that is available at his website.


###  Is there an erratum?

Not at present, but we will create one as necessary and update this section at that time.


### Will the dataset be updated (e.g., to correct labeling errors, add new instances, delete instances)?

Yes. We will correct errors to the extent that we can find them, and we will document the removal of any instances that need to be removed.

Our vision for the dataset is that it be dynamic &mdash; responsive to new models, new developments, and new requirements &mdash; and thus we hope to add many new rounds of data. We are unsure of the cadence of such updates at present.

### If the dataset relates to people, are there applicable limits on the retention of the data associated with the instances (e.g., were individuals in question told that their data would be retained for a fixed period of time and then deleted)?

We are not aware of such limits.

### Will older versions of the dataset continue to be supported/hosted/maintained?

Yes, they will be version-controlled. The only exception is that any instances we are required to remove will be removed from archived versions as well.


### If others want to extend/augment/build on/contribute to the dataset, is there a mechanism for them to do so?

Yes, we are actively looking for partners in expanding and improving the dataset. We would ask people to contact us via email or by using messaging functions on Github.
