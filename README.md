# Sentiment Analysis using Support Vector Machines (SVM)

## Overview

Sentiment analysis using Support Vector Machines (SVM) on textual data. Sentiment analysis is a natural language processing (`NLP`) technique used to determine the sentiment (positive, negative, or neutral) expressed in a piece of text.

The script utilizes the popular `scikit-learn` library for machine learning and NLP tasks. It demonstrates how to preprocess textual data, vectorize it using Term Frequency-Inverse Document Frequency (`TF-IDF`) representation, train a Linear Support Vector Machine classifier, and make predictions on new user input.

## Dependencies

Ensure you have the following Python libraries installed:

- scikit-learn
- numpy
- pandas
- nltk

You can install the required dependencies using pip:

```bash
pip install scikit-learn numpy pandas nltk
```

## Prepare your data:
Make sure you have a CSV file named `Sentimental Analysis Data.csv`
with two columns: text and sentiment. The text column should
contain the textual data, and the sentiment column should contain the
corresponding sentiment labels (positive, negative, neutral).

## Data Preprocessing
The text data undergoes thorough preprocessing before training and prediction. 
The preprocessing steps include:
- Removing punctuation, numbers and special characters. 
- Converting text to lowercase.
- Tokenizing the text.
- Applying stemming using Porter Stemmer.
These steps help to ensure that the text data is in a suitable format for the `SVM` classifier.

## Model Training and Prediction
The script uses the `TfidfVectorizer` to convert the text data into a numerical 
representation based on the `TF-IDF` algorithm. Subsequently, a Linear Support 
Vector Machine (`LinearSVC`) classifier is trained on the vectorized data to 
predict the sentiment labels.

## Evaluation
To assess the model's performance, the script evaluates its predictions using 
precision, recall, accuracy, and F1 measure. The evaluation is conducted by 
predicting the sentiment label for user input and comparing it with the actual 
label.

## License
This project is licensed under the `MIT License` - see the LICENSE file for details.
