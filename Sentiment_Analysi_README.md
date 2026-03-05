# Sentiment Analysis Using NLP and K-Nearest Neighbors (KNN)

## Project Overview

This project performs sentiment analysis on movie reviews using Natural Language Processing (NLP) and a K-Nearest Neighbors (KNN) machine learning algorithm. The goal is to classify movie reviews as **positive** or **negative** based on their textual content.

The analysis uses the **IMDB Movie Review Dataset**, which contains 50,000 labeled movie reviews. By applying text preprocessing and TF-IDF feature extraction, the project converts raw text into numerical vectors that can be used to train a machine learning model.

This project demonstrates how traditional machine learning techniques can be applied to natural language data for sentiment classification.

---

## Dataset

Dataset: **IMDB Movie Reviews**

Source:
https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

The dataset contains:

- 50,000 movie reviews
- Balanced classes (positive and negative sentiment)
- Two columns:
  - `review` – the text of the movie review
  - `sentiment` – label indicating positive or negative sentiment

---

## Project Workflow

The project follows the standard NLP pipeline:

1. **Data Loading**
   - Import the IMDB dataset
   - Inspect structure and sentiment distribution

2. **Text Preprocessing**
   - Convert text to lowercase
   - Remove HTML tags and punctuation
   - Remove stopwords
   - Clean and normalize text

3. **Feature Engineering**
   - Convert text into numerical vectors using **TF-IDF Vectorization**
   - Include both unigrams and bigrams

4. **Model Training**
   - Train a **K-Nearest Neighbors (KNN)** classifier

5. **Model Evaluation**
   - Accuracy score
   - Classification report
   - Confusion matrix

6. **Hyperparameter Tuning**
   - Evaluate multiple values of **K**
   - Determine the optimal number of neighbors

---

## Technologies Used

- Python
- Pandas
- Scikit-learn
- NLTK
- Matplotlib
- Google Colab

---

## Model

Algorithm used:

**K-Nearest Neighbors (KNN)**

KNN classifies a review based on the sentiment of the closest reviews in the feature space. Text similarity is measured using vector representations created through TF-IDF.
