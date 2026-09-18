import os
import sys
import json
import re
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import ExtraTreesClassifier, VotingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

def preprocess_text(text):
    """
    Cleans and standardizes email text while preserving high-value spam signals
    such as currency symbols, links, phone numbers, and urgency punctuation.
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    
    text = text.lower()
    # Replace URLs
    text = re.sub(r'https?://\S+|www\.\S+', ' http_url ', text)
    # Replace email addresses
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', ' email_addr ', text)
    # Replace currency symbols
    text = re.sub(r'[\$£€¥₹]', ' currency_sym ', text)
    # Replace phone numbers and numeric codes
    text = re.sub(r'\b\d{10,}\b', ' phone_num ', text)
    text = re.sub(r'\b\d{3,6}\b', ' short_code ', text)
    text = re.sub(r'\b\d+\b', ' num_val ', text)
    # Normalize urgent punctuation indicators
    text = re.sub(r'!+', ' excl_mark ', text)
    text = re.sub(r'\?+', ' quest_mark ', text)
    # Remove remaining non-alphanumeric characters except whitespace and underscores
    text = re.sub(r'[^a-zA-Z0-9_\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def train_spam_model(data_path='Dataset.csv'):
    print("=" * 60)
    print("Training High-Accuracy Email Spam Shield Model...")
    print("=" * 60)

    # 1. Load Dataset
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset file '{data_path}' not found!")
    
    df = pd.read_csv(data_path)
    df = df.rename(columns={'Category': 'label', 'Message': 'text'})
    
    # 2. Data Cleaning & Validation
    initial_count = len(df)
    df = df[df['label'].isin(['ham', 'spam'])].copy()
    df.dropna(subset=['text', 'label'], inplace=True)
    df.drop_duplicates(subset=['text'], inplace=True)
    clean_count = len(df)
    
    print(f"Dataset summary: {initial_count} raw rows -> {clean_count} clean unique samples")
    ham_count = (df['label'] == 'ham').sum()
    spam_count = (df['label'] == 'spam').sum()
    print(f"Distribution: Ham = {ham_count} ({ham_count/clean_count*100:.1f}%), Spam = {spam_count} ({spam_count/clean_count*100:.1f}%)")

    # 3. Text Preprocessing
    print("\nPreprocessing text features...")
    df['clean_text'] = df['text'].apply(preprocess_text)
    df['target'] = (df['label'] == 'spam').astype(int)

    # 4. Stratified Train / Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        df['clean_text'],
        df['target'],
        test_size=0.20,
        random_state=42,
        stratify=df['target']
    )

    # 5. Dual N-Gram Feature Extraction (Word + Character Subwords)
    print("Extracting Word & Character N-Gram TF-IDF features...")
    word_vec = TfidfVectorizer(ngram_range=(1, 3), max_features=15000, sublinear_tf=True)
    char_vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), max_features=15000, sublinear_tf=True)
    
    feature_union = FeatureUnion([
        ('word', word_vec),
        ('char', char_vec)
    ])

    X_train_vec = feature_union.fit_transform(X_train)
    X_test_vec = feature_union.transform(X_test)

    # 6. Ensemble Model Architecture
    print("Training Calibrated Soft-Voting Ensemble Classifier...")
    svc_calibrated = CalibratedClassifierCV(LinearSVC(C=1.2, random_state=42))
    lr_model = LogisticRegression(C=15.0, max_iter=1000, random_state=42)
    mnb_model = MultinomialNB(alpha=0.05)
    cnb_model = ComplementNB(alpha=0.05)
    et_model = ExtraTreesClassifier(n_estimators=200, random_state=42, n_jobs=-1)

    model = VotingClassifier(
        estimators=[
            ('svc', svc_calibrated),
            ('lr', lr_model),
            ('mnb', mnb_model),
            ('cnb', cnb_model),
            ('et', et_model)
        ],
        voting='soft',
        weights=[3, 2, 1, 1, 2]
    )

    model.fit(X_train_vec, y_train)

    # 7. Model Evaluation
    y_pred = model.predict(X_test_vec)
    y_prob = model.predict_proba(X_test_vec)[:, 1]

    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred))
    rec = float(recall_score(y_test, y_pred))
    f1 = float(f1_score(y_test, y_pred))
    roc_auc = float(roc_auc_score(y_test, y_prob))
    cm = confusion_matrix(y_test, y_pred).tolist()

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS ON TEST DATA (20% Split):")
    print("=" * 60)
    print(f"Accuracy  : {acc * 100:.2f}%")
    print(f"Precision : {prec * 100:.2f}%")
    print(f"Recall    : {rec * 100:.2f}%")
    print(f"F1-Score  : {f1 * 100:.2f}%")
    print(f"ROC-AUC   : {roc_auc * 100:.2f}%")
    print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=['Ham (0)', 'Spam (1)'], digits=4))
    print("Confusion Matrix:\n", np.array(cm))
    print(f"  - True Negatives  (Correct Ham)  : {cm[0][0]}")
    print(f"  - False Positives (Ham as Spam)  : {cm[0][1]}")
    print(f"  - False Negatives (Spam as Ham) : {cm[1][0]}")
    print(f"  - True Positives  (Correct Spam) : {cm[1][1]}")
    print("=" * 60)

    # 8. Save Model, Vectorizer, and Metadata
    print("\nSaving model artifacts...")
    joblib.dump(model, 'spam_model.pkl')
    joblib.dump(feature_union, 'vectorizer.pkl')

    metrics = {
        'accuracy': round(acc, 4),
        'precision': round(prec, 4),
        'recall': round(rec, 4),
        'f1_score': round(f1, 4),
        'roc_auc': round(roc_auc, 4),
        'confusion_matrix': cm,
        'total_samples': clean_count,
        'ham_count': int(ham_count),
        'spam_count': int(spam_count),
        'test_samples': len(y_test)
    }

    with open('model_metrics.json', 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=4)

    print("[SUCCESS] Model trained and saved successfully as 'spam_model.pkl' & 'vectorizer.pkl'!")
    return model, feature_union, metrics

if __name__ == '__main__':
    train_spam_model()
