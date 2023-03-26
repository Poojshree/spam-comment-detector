import streamlit as st
import pickle
from sklearn.feature_extraction.text import CountVectorizer
cv = pickle.load(open("vectorize.pkl","rb"))
model = pickle.load(open("spam_detection.pkl",'rb'))

st.title("Spam comment detector")
input_ = st.text_input("Enter the comment")
#preprocess
if st.button('Predict'):
    data = cv.transform([input_]).toarray()

    result = model.predict(data)[0]
    if result==1:
        st.header("Spam")
    else:
        st.header("Not Spam")