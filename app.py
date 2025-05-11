import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch

# Load model and tokenizer
@st.cache_resource
def load_model():
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
    return tokenizer, model

tokenizer, model = load_model()

st.title("📱 Redmi 6 Customer Feedback Sentiment Analysis")

@st.cache_data
def load_data():
    df = pd.read_csv("redmi6.csv", encoding="ISO-8859-1")
    df["text"] = df["Review Title"].fillna('') + " " + df["Comments"].fillna('')
    return df

df = load_data()

def predict_sentiment(text):
    inputs = tokenizer(str(text), return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        return torch.argmax(outputs.logits, dim=1).item()

df['sentiment'] = df['text'].apply(predict_sentiment)
df['sentiment_label'] = df['sentiment'].map({0: 'Negative', 1: 'Positive'})

st.subheader("🧁 Sentiment Distribution")
sentiment_counts = df['sentiment_label'].value_counts()
fig1, ax1 = plt.subplots()
ax1.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=90)
ax1.axis('equal')
st.pyplot(fig1)

st.subheader("☁️ Word Cloud from Reviews")
all_text = ' '.join(df['text'].dropna().astype(str).tolist())
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)
fig2, ax2 = plt.subplots()
ax2.imshow(wordcloud, interpolation='bilinear')
ax2.axis("off")
st.pyplot(fig2)

st.subheader("📝 Labeled Feedback Data")
st.dataframe(df[['Review Title', 'Comments', 'sentiment_label']])
