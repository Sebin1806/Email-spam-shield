import streamlit as st
import joblib
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk





nltk.download('stopwords')

# Load the saved model and vectorizer
model = joblib.load('spam_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z]', ' ', text)
    words = [stemmer.stem(word) for word in text.split() if word not in stop_words]
    return ' '.join(words)

# Streamlit UI
st.title("📧 Email Spam Classifier")
st.write("Enter a message to check if it's spam.")

user_input = st.text_area("Your Message", height=150)

if st.button("Check"):
    cleaned_text = preprocess(user_input)
    vector_input = vectorizer.transform([cleaned_text])
    prediction = model.predict(vector_input)[0]

    if prediction == 1:
        st.error("🚨 This is **SPAM**!")
    else:
        st.success("✅ This is **NOT SPAM**.")

