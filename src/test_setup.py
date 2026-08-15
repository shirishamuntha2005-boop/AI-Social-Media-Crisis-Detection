import pandas as pd
import numpy as np
import sklearn
import nltk
import spacy
import streamlit
import plotly
import joblib

print("======================================")
print(" AI SOCIAL MEDIA CRISIS DETECTION")
print(" ENVIRONMENT TEST")
print("======================================")

print("Pandas:", pd.__version__)
print("NumPy:", np.__version__)
print("Scikit-learn:", sklearn.__version__)
print("spaCy:", spacy.__version__)
print("Streamlit:", streamlit.__version__)
print("Plotly:", plotly.__version__)

# Test spaCy
nlp = spacy.load("en_core_web_sm")

text = "Heavy flooding was reported in the city."

doc = nlp(text)

print("\nspaCy Test:")

for token in doc:
    print(token.text, "->", token.pos_)

print("\n======================================")
print("✅ All libraries are working!")
print("✅ NLP environment is ready!")
print("======================================")