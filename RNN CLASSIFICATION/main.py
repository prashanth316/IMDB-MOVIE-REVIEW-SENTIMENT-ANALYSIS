import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

word_index = imdb.get_word_index()
reverse_word_index = {value:key for key,value in word_index.items()}

import os

# Get the directory where main.py is located
base_path = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_path, "rnnmodel.h5")

# Load the model
model = load_model(model_path)



def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word,2)+3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review],maxlen=500)
    return padded_review

def  predict_sentiment(review):
    preprocessed_input = preprocess_text(review)

    prediction = model.predict(preprocessed_input)

    return sentiment, prediction[0][0]

## streamlit app
import streamlit as st

st.title('IMDB MOVIE REVIEW SENTIMENT ANALYSIS')
st.write('Enter a movie review to classify it as positive or negative')

user_input = st.text_area('Movie Review')

if st.button('Classify'):
    preprocess_input = preprocess_text(user_input)

    prediction = model.predict(preprocess_input)
    sentiment = 'Positive' if  prediction[0][0] > 0.5 else 'Negative'

    st.write(f'Sentment :  {sentiment}')
    st.write(f'prediction Score : {prediction[0][0]}')
else :
    st.write('Please enter a movie review')    
