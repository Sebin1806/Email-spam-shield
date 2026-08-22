# 📧 Email Spam Shield

A machine learning-based **Email Spam Detection System** that classifies text messages as **Spam** or **Not Spam (Ham)** using Natural Language Processing (NLP) and a Multinomial Naive Bayes classifier.

The project provides a simple **Streamlit web interface** where users can enter an email or message and instantly receive a spam classification.

---

## 🚀 Project Overview

Email spam is one of the most common problems in digital communication. Spam messages can contain advertisements, scams, phishing attempts, fraudulent offers, and other unwanted content.

**Email Spam Shield** uses machine learning and natural language processing techniques to automatically analyze the content of a message and determine whether it is:

* 🚨 **SPAM**
* ✅ **NOT SPAM**

The system preprocesses the input text, converts it into numerical features using **TF-IDF**, and passes those features to a trained **Multinomial Naive Bayes** model.

---

## ✨ Features

* 📧 Email/message spam classification
* 🤖 Machine Learning-based prediction
* 🧠 NLP-based text preprocessing
* 🔤 Text normalization and cleaning
* 🛑 Stopword removal
* 🌱 Porter stemming
* 📊 TF-IDF feature extraction
* 🧮 Multinomial Naive Bayes classification
* 💾 Pre-trained model and vectorizer included
* 🌐 Interactive Streamlit web interface
* ⚡ Real-time prediction
* 🐍 Python-based implementation

---

## 🧠 How It Works

The complete machine learning pipeline works as follows:

```text
                    User Input
                        │
                        ▼
              ┌──────────────────┐
              │   Text Cleaning  │
              └────────┬─────────┘
                       │
                       ▼
              ┌──────────────────┐
              │ Convert to Lower │
              │      Case        │
              └────────┬─────────┘
                       │
                       ▼
              ┌──────────────────┐
              │ Remove Special   │
              │    Characters    │
              └────────┬─────────┘
                       │
                       ▼
              ┌──────────────────┐
              │ Stopword Removal │
              └────────┬─────────┘
                       │
                       ▼
              ┌──────────────────┐
              │ Porter Stemming  │
              └────────┬─────────┘
                       │
                       ▼
              ┌──────────────────┐
              │  TF-IDF Vector   │
              │  Transformation  │
              └────────┬─────────┘
                       │
                       ▼
              ┌──────────────────┐
              │ Multinomial Naive│
              │      Bayes       │
              └────────┬─────────┘
                       │
              ┌────────┴────────┐
              ▼                 ▼
        🚨 SPAM            ✅ NOT SPAM
```

---

# 🔬 Machine Learning Pipeline

## 1. Dataset

The project uses a labeled SMS/email-style text dataset stored in:

```text
Dataset.csv
```

The dataset contains two primary columns:

| Column     | Description                    |
| ---------- | ------------------------------ |
| `Category` | Message class: `spam` or `ham` |
| `Message`  | Actual text message            |

The dataset contains approximately **5,574 records**.

The classes are mapped as:

```text
ham  → 0
spam → 1
```

---

## 2. Data Cleaning

Before training the model, the text is cleaned.

The preprocessing pipeline performs:

### Lowercase conversion

```python
text = text.lower()
```

This ensures that words such as:

```text
FREE
Free
free
```

are treated consistently.

### Special character removal

Characters other than alphabetic characters are removed using regular expressions.

```python
re.sub(r'[^a-z]', ' ', text)
```

### Stopword removal

Common English words that usually provide limited classification value are removed using NLTK's English stopword list.

Examples include:

```text
the
is
a
an
and
to
of
```

### Stemming

The project uses the **Porter Stemmer** to reduce words to their root form.

For example:

```text
playing
played
plays
```

can be reduced toward a common stem.

---

# 📊 Feature Extraction

Machine learning algorithms cannot directly process raw text.

Therefore, the cleaned messages are converted into numerical vectors using:

## TF-IDF

**TF-IDF** stands for:

> Term Frequency – Inverse Document Frequency

The project uses:

```python
TfidfVectorizer(max_features=3000)
```

This converts the cleaned text into numerical features while limiting the vocabulary to the top 3,000 features.

TF-IDF gives higher importance to words that are useful for distinguishing messages while reducing the importance of very common words.

---

# 🤖 Machine Learning Model

The classification algorithm used in this project is:

## Multinomial Naive Bayes

```python
MultinomialNB()
```

Multinomial Naive Bayes is particularly suitable for text classification problems because it works effectively with word-frequency and TF-IDF-based features.

The model learns patterns associated with:

```text
Spam messages
        ↓
Promotional language
Prize/offer terminology
Suspicious phrases
Unwanted advertisements
etc.

Ham messages
        ↓
Normal conversations
Personal messages
Regular communication
etc.
```

---

# 🏋️ Model Training

The dataset is divided into training and testing sets using:

```python
train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

This means approximately:

* **80%** → Training data
* **20%** → Testing data

The model is then trained:

```python
model = MultinomialNB()
model.fit(X_train, y_train)
```

The project also calculates:

* Accuracy
* Classification Report
* Confusion Matrix

using Scikit-learn.

---

# 💾 Saved Model

After training, the project saves two important files:

```text
spam_model.pkl
vectorizer.pkl
```

### `spam_model.pkl`

Contains the trained Multinomial Naive Bayes model.

### `vectorizer.pkl`

Contains the trained TF-IDF vectorizer.

This allows the Streamlit application to make predictions without retraining the model every time the application starts.

---

# 🌐 Streamlit Application

The user interface is implemented using **Streamlit**.

The application:

1. Loads the trained model.
2. Loads the trained TF-IDF vectorizer.
3. Accepts user input.
4. Preprocesses the input.
5. Converts the text into TF-IDF features.
6. Sends the features to the trained model.
7. Displays the prediction.

Example:

```text
User enters message
        ↓
Text preprocessing
        ↓
TF-IDF transformation
        ↓
Naive Bayes model
        ↓
Prediction
        ↓
SPAM / NOT SPAM
```

---

# 🛠️ Technologies Used

| Technology          | Purpose                     |
| ------------------- | --------------------------- |
| Python              | Core programming language   |
| Pandas              | Dataset processing          |
| NumPy               | Numerical operations        |
| Scikit-learn        | Machine learning and TF-IDF |
| NLTK                | NLP preprocessing           |
| Joblib              | Saving/loading ML models    |
| Streamlit           | Web application             |
| Regular Expressions | Text cleaning               |

---

# 📁 Project Structure

```text
Email-spam-shield/
│
├── Email spam shield/
│   │
│   ├── app.py
│   ├── Email_spam.py
│   ├── Dataset.csv
│   ├── spam_model.pkl
│   └── vectorizer.pkl
│
└── README.md
```

### `app.py`

Main Streamlit application.

Responsible for:

* Loading the trained model
* Loading the TF-IDF vectorizer
* Processing user input
* Generating predictions
* Displaying results

### `Email_spam.py`

Machine learning training script.

Responsible for:

* Loading the dataset
* Cleaning the data
* NLP preprocessing
* TF-IDF feature extraction
* Training the Naive Bayes model
* Evaluating the model
* Saving the trained model

### `Dataset.csv`

Training dataset containing spam and ham messages.

### `spam_model.pkl`

Serialized trained classification model.

### `vectorizer.pkl`

Serialized TF-IDF vectorizer.

---

# ⚙️ Installation

## Prerequisites

Make sure you have installed:

* Python 3.10+
* pip
* Git

Check Python:

```bash
python --version
```

Check pip:

```bash
pip --version
```

---

# 📥 Clone the Repository

Clone the repository:

```bash
git clone https://github.com/YOUR-USERNAME/Email-spam-shield.git
```

Move into the project directory:

```bash
cd Email-spam-shield
```

Then enter the application folder:

```bash
cd "Email spam shield"
```

---

# 🐍 Create a Virtual Environment

Windows:

```bash
python -m venv .venv
```

Activate it:

```bash
.venv\Scripts\activate
```

You should see:

```text
(.venv)
```

in your terminal.

---

# 📦 Install Dependencies

Install the required packages:

```bash
pip install streamlit pandas nltk scikit-learn joblib
```

Alternatively, if a `requirements.txt` file is included:

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the Application

Start the Streamlit application:

```bash
python -m streamlit run app.py
```

Alternatively:

```bash
streamlit run app.py
```

After starting the application, Streamlit will provide a local address similar to:

```text
Local URL: http://localhost:8501
```

Open the URL in your browser.

---

# 🧪 Using the Application

Once the application is running:

1. Open the Streamlit URL.
2. Enter an email or message.
3. Click **Check**.
4. The system analyzes the message.
5. The result is displayed.

### Example Spam Message

```text
Congratulations! You have won a free prize. Click here to claim your reward now!
```

Expected result:

```text
🚨 This is SPAM!
```

### Example Normal Message

```text
Hi, can you send me the project report before tomorrow's meeting?
```

Expected result:

```text
✅ This is NOT SPAM.
```

---

# 🔄 Retraining the Model

The project includes the training script:

```text
Email_spam.py
```

To retrain the model:

```bash
python Email_spam.py
```

The script will:

```text
Dataset.csv
     ↓
Data Cleaning
     ↓
Label Encoding
     ↓
NLP Preprocessing
     ↓
TF-IDF
     ↓
Train/Test Split
     ↓
Multinomial Naive Bayes
     ↓
Model Evaluation
     ↓
spam_model.pkl
vectorizer.pkl
```

After training, start the application again:

```bash
python -m streamlit run app.py
```

---

# 📈 Model Evaluation

The training script evaluates the model using:

### Accuracy

Measures the percentage of correctly classified messages.

```python
accuracy_score(y_test, y_pred)
```

### Classification Report

Provides:

* Precision
* Recall
* F1-score
* Support

```python
classification_report(y_test, y_pred)
```

### Confusion Matrix

Shows:

```text
                 Predicted
              Ham       Spam

Actual Ham      TN        FP
Actual Spam     FN        TP
```

This helps identify false positives and false negatives.

---

# 🧩 NLP Techniques Used

This project demonstrates several important NLP concepts:

### 1. Tokenization

The input text is separated into individual words.

### 2. Stopword Removal

Common words are removed to reduce unnecessary features.

### 3. Stemming

Words are reduced toward their root/stem representation.

### 4. TF-IDF

Text is converted into numerical feature vectors.

### 5. Text Classification

The extracted features are classified using Multinomial Naive Bayes.

---

# 🏗️ System Architecture

```text
                 ┌────────────────────┐
                 │      User           │
                 └─────────┬──────────┘
                           │
                           ▼
                 ┌────────────────────┐
                 │ Streamlit Web App  │
                 └─────────┬──────────┘
                           │
                           ▼
                 ┌────────────────────┐
                 │ Text Preprocessing │
                 │                    │
                 │ • Lowercase        │
                 │ • Cleaning         │
                 │ • Stopwords        │
                 │ • Stemming         │
                 └─────────┬──────────┘
                           │
                           ▼
                 ┌────────────────────┐
                 │   TF-IDF Vectorizer│
                 └─────────┬──────────┘
                           │
                           ▼
                 ┌────────────────────┐
                 │ Multinomial Naive  │
                 │ Bayes Classifier   │
                 └─────────┬──────────┘
                           │
                    ┌──────┴──────┐
                    ▼             ▼
              ┌──────────┐  ┌─────────────┐
              │   SPAM   │  │  NOT SPAM   │
              └──────────┘  └─────────────┘
```

---

# 🔐 Important Note

This project is intended as an **educational machine learning project**.

The classifier should not be considered a production-grade email security system. A real-world spam detection platform would typically require additional techniques such as:

* Deep learning
* Transformer-based NLP models
* Phishing URL detection
* Sender reputation analysis
* Header analysis
* Domain reputation
* Real-time threat intelligence
* Adversarial attack protection
* Continuous model retraining

---

# 🚀 Future Improvements

Possible improvements include:

* [ ] Upgrade the model to Logistic Regression or SVM
* [ ] Experiment with Random Forest and ensemble models
* [ ] Add word and character n-grams
* [ ] Improve dataset quality
* [ ] Add cross-validation
* [ ] Add ROC-AUC evaluation
* [ ] Add precision-recall curves
* [ ] Add probability/confidence scores
* [ ] Add email file upload
* [ ] Add `.eml` email parsing
* [ ] Detect suspicious URLs
* [ ] Detect phishing messages
* [ ] Add sender/domain analysis
* [ ] Experiment with BERT/Transformer models
* [ ] Deploy the application to Streamlit Cloud
* [ ] Add a REST API using FastAPI
* [ ] Add database storage for prediction history
* [ ] Add user authentication
* [ ] Add monitoring and model performance tracking

---

# 📚 Learning Outcomes

Through this project, you can learn:

* Python programming
* Data preprocessing
* Exploratory dataset handling
* Natural Language Processing
* Text cleaning
* Stopword removal
* Stemming
* TF-IDF
* Feature engineering
* Train/test splitting
* Naive Bayes classification
* Model evaluation
* Model serialization
* Streamlit application development
* Machine learning deployment concepts

---

# 🎯 Project Use Cases

The same concept can be extended to:

* 📧 Email spam filtering
* 💬 SMS spam detection
* 🛡️ Phishing message detection
* 📱 Chat moderation
* 📢 Advertisement filtering
* 🔒 Suspicious communication detection

---

# ⚠️ Troubleshooting

## Streamlit command not recognized

If this command:

```bash
streamlit run app.py
```

doesn't work, use:

```bash
python -m streamlit run app.py
```

---

## NLTK Stopwords Error

If you receive an error related to:

```text
Resource stopwords not found
```

run:

```bash
python -c "import nltk; nltk.download('stopwords')"
```

Then start the application again:

```bash
python -m streamlit run app.py
```

---

## Model File Not Found

If you receive:

```text
FileNotFoundError
```

make sure these files exist in the same directory as `app.py`:

```text
spam_model.pkl
vectorizer.pkl
```

---

# 👨‍💻 Author

**Sebin S**

AI & Data Science Student
Python | Machine Learning | Artificial Intelligence | Data Science

---

# ⭐ Support

If you found this project useful, consider giving the repository a ⭐ on GitHub.

---

## 📜 License

This project is intended for educational and learning purposes.

You may modify and extend the project for your own academic and personal projects.
