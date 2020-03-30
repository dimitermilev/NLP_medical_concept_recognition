# Medical Concept Recognition in Free Text

I use NLP methods and shallow (Naive Bayes, stochastic gradient descent classifier) and deep learning (Bidirectional LSTM) to recognize and classify medical concepts in free text hospital notes.

## Data

Data collected from the [n2c2 NLP Research Data Set](https://portal.dbmi.hms.harvard.edu/projects/n2c2-nlp/#) made available by the Harvard Medical School. The dataset comes from the NLP Relations challenge. It consists of:

* Data from 3 hospitals
* 871 hospital notes (progress notes and discharge summaries) in free text
* 35,000+ concept tags that locate problems, treatments and tests in the free text

## Methods

**Natural Language Processing Pipeline**
- Sentence parsing, Tokenization, Part of Speech tagging, Stemming

**Feature Engineering**
- Context window features: dates, digits, titles, parts of speech, upper/lower

**Shallow ML:**
- Naive Bayes and linear classifier

**Deep Learning:**
- Bidirectional LSTM and Sequence-to-sequence modeling

## Data manipulation and analysis

```
numpy
pandas
nltk
sklearn
keras
gensim
imblearn
```

### Front end and visualization

```
spaCy
Flask
```

