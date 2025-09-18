import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
import os

# Configure Streamlit page
st.set_page_config(
    page_title="Spam Classifier",
    page_icon="📧",
    layout="centered"
)

# Download required NLTK resources
@st.cache_resource
def download_nltk_resources():
    """Download NLTK resources with caching to avoid repeated downloads"""
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)

    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords", quiet=True)
    
    return PorterStemmer()

ps = download_nltk_resources()

# Function to preprocess the input text
def transform_text(text):
    """
    Preprocesses text for spam classification:
    1. Converts to lowercase
    2. Tokenizes
    3. Removes non-alphanumeric characters
    4. Removes stopwords and punctuation
    5. Applies stemming
    """
    if not text.strip():
        return ""
    
    text = text.lower()
    text = nltk.word_tokenize(text)

    # Remove non-alphanumeric characters
    text = [word for word in text if word.isalnum()]
    
    # Remove stopwords and punctuation
    stop_words = set(stopwords.words('english'))
    text = [word for word in text if word not in stop_words and word not in string.punctuation]
    
    # Apply stemming
    text = [ps.stem(word) for word in text]
    
    return " ".join(text)

# Load models with caching
@st.cache_resource
def load_models():
    """Load TF-IDF vectorizer and classification model with error handling"""
    current_dir = os.getcwd()
    
    # Load TF-IDF vectorizer
    tfidf_path = os.path.join(current_dir, 'vectorizer.pkl')
    try:
        with open(tfidf_path, 'rb') as file:
            tfidf = pickle.load(file)
        st.success("✅ TF-IDF Vectorizer loaded successfully")
    except FileNotFoundError:
        st.error(f"❌ Error: 'vectorizer.pkl' not found at {tfidf_path}")
        st.stop()
    
    # Load classification model
    model_path = os.path.join(current_dir, 'model.pkl')
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        st.success("✅ Model loaded successfully")
    except FileNotFoundError:
        st.error(f"❌ Error: 'model.pkl' not found at {model_path}")
        st.stop()
    
    return tfidf, model

# Load models
tfidf, model = load_models()

# Main UI
st.title("📧 Email/SMS Spam Classifier")
st.markdown("---")

# Add some information about the app
with st.expander("ℹ️ How it works"):
    st.write("""
    This app uses machine learning to classify messages as spam or not spam.
    
    **Steps:**
    1. Enter your message in the text area below
    2. Click 'Predict' to analyze the message
    3. The app will preprocess the text and make a prediction
    
    **Text preprocessing includes:**
    - Converting to lowercase
    - Removing punctuation and special characters
    - Removing common stop words
    - Stemming words to their root form
    """)

# Input section
st.subheader("📝 Enter Your Message")
input_sms = st.text_area(
    "Message to classify:",
    height=100,
    placeholder="Type or paste your email/SMS message here..."
)

# Add example messages
col1, col2 = st.columns(2)
with col1:
    if st.button("📩 Try Spam Example"):
        input_sms = "URGENT! You've won $1000! Click here now to claim your prize. Limited time offer!"
        st.rerun()

with col2:
    if st.button("📧 Try Normal Example"):
        input_sms = "Hi, how are you doing? Let's catch up over coffee this weekend."
        st.rerun()

# Prediction section
if st.button('🔍 Predict', type="primary", use_container_width=True):
    if not input_sms.strip():
        st.warning("⚠️ Please enter a message to classify.")
    else:
        with st.spinner("Analyzing message..."):
            # Preprocess the input message
            transformed_sms = transform_text(input_sms)
            
            if not transformed_sms:
                st.warning("⚠️ Message appears to contain no meaningful content after preprocessing.")
            else:
                # Vectorize the input message
                vector_input = tfidf.transform([transformed_sms])
                
                # Predict using the model
                result = model.predict(vector_input)[0]
                
                # Get prediction probability if available
                try:
                    probabilities = model.predict_proba(vector_input)[0]
                    confidence = max(probabilities) * 100
                except:
                    confidence = None
                
                # Display results
                st.markdown("---")
                st.subheader("🎯 Prediction Result")
                
                if result == 1:
                    st.error("🚨 **SPAM DETECTED**")
                    if confidence:
                        st.error(f"Confidence: {confidence:.1f}%")
                else:
                    st.success("✅ **NOT SPAM**")
                    if confidence:
                        st.success(f"Confidence: {confidence:.1f}%")
                
                # Show preprocessed text
                with st.expander("🔧 View Preprocessed Text"):
                    st.code(transformed_sms if transformed_sms else "No text after preprocessing")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "Built with Streamlit • Machine Learning Spam Classifier"
    "</div>", 
    unsafe_allow_html=True
)