import streamlit as st
from keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import time
def predict_class(text):

    model = load_model('Twitter_sentiment_model.h5')
    sentiment_classes = ['Negative', 'Neutral', 'Positive']
    max_len=50

    tokenizer = Tokenizer(num_words=5000, lower=True, split=' ')
    xt = tokenizer.texts_to_sequences(text)

    xt = pad_sequences(xt, padding='post', maxlen=max_len)

    yt = model.predict(xt).argmax(axis=1)

    st.write('The predicted sentiment is', sentiment_classes[yt[0]])

st.title("Twitter Sentiment Analysis")


tweet = [str(st.text_input("Enter your tweet"))]

submit = st.button('Predict')

if submit:
    start = time.time()
    predict_class(tweet)
    end = time.time()
    st.write('Prediction time taken: ', round(end-start, 2), "sec")

