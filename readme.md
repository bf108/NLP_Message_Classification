# Disaster Response Classification

## Introduction
The purpose of this analysis is to develop a machine learning algorithm to
classify messages to a particular  category or categories.

Model will be developed using Natural Language Processing (NLP) techniques.

## Dataset
Data was provided by Figure Eight [Figure Eight Wiki](https://en.wikipedia.org/wiki/Figure_Eight_Inc.)
(formerly known as Dolores Lab, CrowdFlower, acquired by Appen and Five River)

Data consists of messages obtained during disaster scenarios across multiple
platforms: social media, news, direct messaging.

Total entries: 26,215 entries and 35 categories of disaster.

## ETL Pipeline
**Data Quality**
The dataset had very little data on some Disaster categories. For example
"Child Alone". Out of the entire dataset, not one message was classified with
this category. As such, this category was removed from the model. With no
training data, it made future predictions impossible.

**ETL Script**
The final ETL script can be found in process_data.py

## NLP
**NLP**
The model trained by conducting sentiment analysis on each message and comparing
it to the classifications given.
  - Multi-class classification

The natural language processing pipeline adopted was:
  - Normalization of text (Lowercase, remove punctuation and urls)
  - Tokenize sentences and then words in sentences.
  - Bag of words - Frequency of words in full corpus
  - Token frequency inverse document frequency (TD-IDF)

## Machine Learning Pipeline
**Algorithm Selection**
The original algorithm used was a RandomForest Classifier. Precision scores were
in the 70% region, but recall was only marginally better than guessing.

GridSearch of a few features marginally improved performance.

Significant performance increase was observed when an AdaBoost classifier was
used.

The script to train the final model can be found in train_classifier.py

## Final Product - Flask Application
Web app where an emergency worker can input a new message and get classification
results in several categories.

The web app also displays some visuals on the primary data from Figure Eight.

Web app can be displayed with run.py

## Running The Model

In the terminal cd to directory with cloned repo. Then run the following commands:

Step 1:
python, Filepaths for Processing Script, message data, category data, database name.
Example:
python process_data.py dis_messages.csv dis_cats.csv DisasterResponse.db

Step 2:
python, filepaths for train_classifier.py, database created in step 1, and
pkl file where model will be saved.
Example
python train_classifier.py DisasterResponse.db classifier.pkl

Step 3:
**train_classifier.py file is in same directory as run.py**
python, filepaths for run.py file, database created in step1, model file
example
python App/run.py DisasterResponse.db classifier.pkl
