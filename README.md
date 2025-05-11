# Redmi6_Sentiment_Analysis_App
readme_content = """# 📱 Redmi 6 Customer Feedback Sentiment Analysis

This project applies Natural Language Processing (NLP) techniques to classify customer feedback on the **Redmi 6 smartphone** into positive or negative sentiments. It uses a fine-tuned **DistilBERT** model and visualizes the results via an interactive **Streamlit app**.

---

## 🧠 Problem Statement

Manually analyzing large volumes of customer feedback is time-consuming and inconsistent. This project automates sentiment analysis of Amazon product reviews to help brands:

- Track customer satisfaction
- Detect pain points
- Drive product improvement decisions

---

## 📊 Dataset

The dataset (`redmi6.csv`) contains 280 real-world Amazon reviews with:
- Review Title
- Comments
- Rating
- Category
- Helpfulness

For analysis, we combined `Review Title + Comments` into a single feedback text.

---

## 🧪 Model & Methodology

| Component            | Approach                                                   |
|----------------------|-------------------------------------------------------------|
| **Preprocessing**    | Lowercasing, combining title/comments, lemmatization        |
| **Model**            | `DistilBERT` from Hugging Face transformers                 |
| **Classification**   | Fine-tuned for binary sentiment classification (0 = Neg, 1 = Pos) |
| **Visualization**    | Pie chart + Word cloud using Streamlit                      |

---

## 🚀 Streamlit App Features

- Classify any single feedback input
- Upload CSV file for bulk analysis
- Pie chart of sentiment distribution
- Word cloud of frequent terms
- Downloadable results

---

## 📦 How to Run

### ▶️ Option 1: Google Colab + ngrok
1. Open `Redmi6_Sentiment_Analysis_App.ipynb` in Colab
2. Upload `redmi6.csv`
3. Add your [ngrok authtoken](https://dashboard.ngrok.com/)
4. App will generate a public link like `https://<your-app>.ngrok.io`

### ▶️ Option 2: Local
```bash
git clone https://github.com/<your-username>/Redmi6_Sentiment_Analysis_App.git
cd Redmi6_Sentiment_Analysis_App
pip install -r requirements.txt  # optional
streamlit run app.py
