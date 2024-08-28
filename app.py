import streamlit as st
from transformers import pipeline
import torch

device = 0 if torch.cuda.is_available() else -1
summarizer = pipeline('summarization', device=device)

st.title("Article Summarizer")


article = st.text_area("Enter the article text:", height=300)

min_length = st.slider("Minimum length of the summary:", 30, 100, 30)
max_length = st.slider("Maximum length of the summary:", 100, 500, 130)

if st.button("Summarize"):
    if len(article.split()) > 1000:  
        st.error("The article is too long. Please enter a shorter text (less than 1000 words).")
    elif len(article.split()) < 10:  
        st.error("The article is too short. Please enter a longer text.")
    else:
        with st.spinner("Generating summary..."):
            summary = summarizer(article, max_length=max_length, min_length=min_length, do_sample=False)
            st.success("Summary generated!")
            st.write(summary[0]['summary_text'])


