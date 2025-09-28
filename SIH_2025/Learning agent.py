# Cell 1: Imports & Setup
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
import os
import joblib  # to save/load models

nlp = spacy.load("en_core_web_sm")

# persistent files
SYMPTOM_FILE = "Disease_symptoms_new.csv"
MODEL_FILE = "disease_model.pkl"
VECTORIZER_FILE = "vectorizer.pkl"


# Cell 2: Preprocessing
def preprocess(text):
    doc = nlp(str(text).lower())
    return " ".join([
        token.lemma_ for token in doc 
        if not token.is_stop and token.is_alpha
    ])


# Cell 3: Train or Load Model
def train_or_load_model():
    if os.path.exists(MODEL_FILE) and os.path.exists(VECTORIZER_FILE):
        model = joblib.load(MODEL_FILE)
        vectorizer = joblib.load(VECTORIZER_FILE)
        return model, vectorizer
    
    df = pd.read_csv(SYMPTOM_FILE)
    df['Symptoms'] = df['Symptoms'].apply(preprocess)
    
    X = df['Symptoms']
    y = df['Disease']

    vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=2)
    X_tfidf = vectorizer.fit_transform(X)

    model = LogisticRegression(max_iter=2000)
    model.fit(X_tfidf, y)

    # save for later reuse
    joblib.dump(model, MODEL_FILE)
    joblib.dump(vectorizer, VECTORIZER_FILE)

    return model, vectorizer


# Cell 4: Prediction with Top-3
def predict_top3(symptoms_list, model, vectorizer, threshold=0.7):
    cleaned = [preprocess(text) for text in symptoms_list]
    features = vectorizer.transform(cleaned)
    proba = model.predict_proba(features)
    classes = model.classes_
    
    results = []
    for raw_sym, clean_sym, probs in zip(symptoms_list, cleaned, proba):
        top_idx = np.argsort(probs)[::-1][:3]
        top_diseases = [(classes[i], float(probs[i])) for i in top_idx]
        pred = classes[top_idx[0]] if probs[top_idx[0]] >= threshold else "Uncertain"
        
        results.append({
            "Original Input": raw_sym,
            "Preprocessed": clean_sym,
            "Prediction": pred,
            "Top 3": top_diseases
        })
    return results


# Cell 5: Critic + Learning (Agent Feedback Loop)
def update_agent(symptom_text, confirmed_disease):
    """Update dataset and retrain model with new confirmed case"""
    # Append new row to CSV dataset
    df = pd.read_csv(SYMPTOM_FILE)
    df.loc[len(df)] = [confirmed_disease, symptom_text]
    df.to_csv(SYMPTOM_FILE, index=False)
    # retrain model
    if os.path.exists(MODEL_FILE): os.remove(MODEL_FILE)
    if os.path.exists(VECTORIZER_FILE): os.remove(VECTORIZER_FILE)
    return train_or_load_model()
