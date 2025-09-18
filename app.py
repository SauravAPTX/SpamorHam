import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
import os

# Download required NLTK resources

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", quiet=True)
    

ps = PorterStemmer()

# Function to preprocess the input text
def transform_text(text):
    text = text.lower()  # Convert text to lowercase
    text = nltk.word_tokenize(text)  # Tokenize the text

    y = []
    # Remove non-alphanumeric characters
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    # Remove stopwords and punctuation
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    # Apply stemming
    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


# Get the current working directory
current_dir = os.getcwd()
print("Current directory:", current_dir)

# Load the TF-IDF vectorizer
tfidf_path = os.path.join(current_dir, 'venv/vectorizer.pkl')

try:
    with open(tfidf_path, 'rb') as file:
        tfidf = pickle.load(file)
    print("TF-IDF Vectorizer loaded successfully.")
except FileNotFoundError:
    print(f"Error: 'vectorizer.pkl' not found at {tfidf_path}")
    st.stop()  # Stop the app if the vectorizer is missing

# Load the classification model
model_path = os.path.join(current_dir, 'venv/model.pkl')

try:
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    print("Model loaded successfully.")
except FileNotFoundError:
    print(f"Error: 'model.pkl' not found at {model_path}")
    st.stop()  # Stop the app if the model is missing

# Set up the Streamlit interface
st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

# Prediction button
if st.button('Predict'):

    # 1. Preprocess the input message
    transformed_sms = transform_text(input_sms)

    # 2. Vectorize the input message using the loaded TF-IDF vectorizer
    vector_input = tfidf.transform([transformed_sms])

    # 3. Predict using the loaded classification model
    result = model.predict(vector_input)[0]

    # 4. Display the result
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
