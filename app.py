import streamlit as st
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the saved model
with open('tfidf.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the TfidfVectorizer with the same vocabulary used during training
with open('tfidf_vectorizer.pkl', 'rb') as file:
    tfidf_vectorizer = pickle.load(file)

# Function to preprocess the input text
def preprocess_text(text):
    # Apply any necessary preprocessing to the input text
    # For example, you can convert the text to lowercase, remove punctuation, etc.
    processed_text = text.lower()  # Placeholder pre-processing step
    return processed_text

# Function to make predictions
def predict_fake_news(text):
    # Preprocess the input text
    processed_text = preprocess_text(text)
    
    # Vectorize the preprocessed text
    text_vector = tfidf_vectorizer.transform([processed_text])
    
    # Make the prediction
    prediction = model.predict(text_vector)[0]
    
    # Return the prediction
    return prediction

# Create a Streamlit app
def main():
    # Set the app title
    st.title('Fake News Detection')
    
    # Create a textarea for user input
    user_input = st.text_area('Enter the news text:')
    
    # Add a button to trigger the prediction
    if st.button('Predict'):
        # Perform prediction only if the user has entered some text
        if user_input:
            # Make the prediction
            prediction = predict_fake_news(user_input)
            
            # Display the prediction
            if prediction == 0:
                st.error('This news is classified as FAKE.')
            else:
                st.success('This news is classified as TRUE.')

# Run the app
if __name__ == '__main__':
    main()
