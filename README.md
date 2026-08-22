# 📧 Email Spam Shield

A Machine Learning and NLP-based web application that detects whether an email/message is **Spam** or **Not Spam** using **TF-IDF** and **Multinomial Naive Bayes**.

## 🚀 Features

* 📧 Spam / Ham classification
* 🧠 NLP text preprocessing
* 🔤 Stopword removal & stemming
* 📊 TF-IDF feature extraction
* 🤖 Multinomial Naive Bayes
* 🌐 Streamlit web interface
* ⚡ Real-time predictions
* 💾 Pre-trained model included

## 🛠️ Tech Stack

* **Python**
* **Pandas**
* **Scikit-learn**
* **NLTK**
* **Joblib**
* **Streamlit**

## 🧠 How It Works

```text
User Message
     ↓
Text Preprocessing
     ↓
Stopword Removal + Stemming
     ↓
TF-IDF Vectorization
     ↓
Multinomial Naive Bayes
     ↓
Spam / Not Spam
```

## 📁 Project Structure

```text
Email-spam-shield/
│
├── Email spam shield/
│   ├── app.py
│   ├── Email_spam.py
│   ├── Dataset.csv
│   ├── spam_model.pkl
│   └── vectorizer.pkl
│
└── README.md
```

| File             | Description                 |
| ---------------- | --------------------------- |
| `app.py`         | Streamlit web application   |
| `Email_spam.py`  | Model training & evaluation |
| `Dataset.csv`    | Spam/ham dataset            |
| `spam_model.pkl` | Trained Naive Bayes model   |
| `vectorizer.pkl` | Trained TF-IDF vectorizer   |

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/Email-spam-shield.git
cd Email-spam-shield
cd "Email spam shield"
```

### 2. Create virtual environment

```bash
python -m venv .venv
```

### 3. Activate it

**Windows:**

```bash
.venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install streamlit pandas nltk scikit-learn joblib
```

### 5. Run the application

```bash
python -m streamlit run app.py
```

Open:

```text
http://localhost:8501
```

## 🧪 Example

**Input:**

```text
Congratulations! You have won a free prize. Click here to claim now!
```

**Output:**

```text
🚨 SPAM
```

Normal messages are classified as:

```text
✅ NOT SPAM
```

## 🔄 Retrain the Model

To train the model again:

```bash
python Email_spam.py
```

This generates:

```text
spam_model.pkl
vectorizer.pkl
```

Then run the application again.

## 📈 Machine Learning

**Preprocessing:**

* Lowercase conversion
* Special character removal
* Stopword removal
* Porter stemming

**Feature Extraction:**

* TF-IDF (`max_features=3000`)

**Classifier:**

* Multinomial Naive Bayes

## 🚀 Future Improvements

* Transformer/BERT-based classification
* Phishing URL detection
* Confidence scores
* Email `.eml` file support
* FastAPI backend
* Cloud deployment
* Improved dataset and model evaluation

## 👨‍💻 Author

**Sebin S**

AI & Data Science | Python | Machine Learning | Artificial Intelligence

---

⭐ If you found this project useful, consider starring the repository.
