# Email Spam Classification with PySpark

This project implements a scalable email spam classification system using PySpark. The system processes and analyzes the Enron spam dataset to classify emails as either spam or legitimate (ham) messages.

## Project Overview

The system uses natural language processing (NLP) and machine learning techniques to:
- Clean and preprocess email text data
- Extract features using TF-IDF vectorization
- Train and evaluate multiple classification models
- Compare model performance metrics

## Requirements

- Python 3.9+
- PySpark
- NLTK
- spaCy
- inflect
- pandas
- Hugging Face datasets

You'll also need to download the spaCy English language model:
```bash
python -m spacy download en_core_web_sm
```

## Dataset

The project uses the SetFit/enron_spam dataset from Hugging Face, which contains:
- 31,716 total emails
- Binary classification (spam/ham)
- Features: message_id, text, label, subject, message, date

## Project Structure

The codebase is organized into several main components:

### 1. Data Preprocessing
- `TextCleaner` class handles:
  - Text normalization
  - URL and email address removal
  - Number conversion
  - Lemmatization
  - Stop word removal

### 2. Feature Extraction
- `FeatureExtraction` class implements:
  - Text tokenization
  - TF-IDF vectorization
  - Feature combination for both subject and message content

### 3. Model Training and Evaluation
- `SparkSpamClassifier` class provides:
  - Multiple classification models:
    - Naive Bayes
    - Logistic Regression
    - Gradient Boosted Trees
  - Model training
  - Performance evaluation
  - Comparative metrics

## Results

The models achieved the following performance metrics on the test set:

| Model                | Accuracy | Precision | Recall   | F1 Score |
|---------------------|----------|-----------|----------|----------|
| Naive Bayes         | 0.969    | 0.969     | 0.969    | 0.969    |
| Logistic Regression | 0.972    | 0.972     | 0.972    | 0.972    |
| Gradient Boosted Trees | 0.922  | 0.929     | 0.922    | 0.922    |

## Key Findings

1. Data Distribution:
   - Balanced dataset with ~49% ham and ~51% spam emails
   - Over 16,000 unique words in subject lines
   - Over 137,000 unique words in message bodies

2. Model Performance:
   - Logistic Regression performed best overall
   - All models achieved >92% accuracy
   - High precision and recall across all models

## Usage

1. Initialize text cleaner:
```python
cleaner = TextCleaner(subject_col="subject", 
                     message_col="message", 
                     remove_stopwords=True)
```

2. Clean the data:
```python
cleaned_df = cleaner.clean_dataframe(df)
```

3. Extract features:
```python
feature_extractor = FeatureExtraction(
    subject_col="subject_lem_tokens",
    message_col="message_lem_tokens",
    ngram_range=(1, 2),
    max_features=1000
)
```

4. Train and evaluate models:
```python
classifier = SparkSpamClassifier(
    label_col="label", 
    feature_cols=["subject_lem_tokens_features", "message_lem_tokens_features"]
)
classifier.train(train_df)
results = classifier.evaluate(test_df)
```

## Future Improvements

1. Feature Engineering:
   - Implement n-gram features
   - Add custom spam indicators
   - Include email metadata features

2. Model Enhancements:
   - Hyperparameter tuning
   - Ensemble methods
   - Deep learning approaches

3. System Optimization:
   - Spark configuration tuning
   - Pipeline optimization
   - Real-time prediction capabilities
