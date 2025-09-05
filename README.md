# ⚡ Sentiment Shock — LSTM + Transformer + Visuals

Welcome to **Sentiment Shock**, a fun and interactive demo app for sentiment analysis! This project combines the power of **pretrained Transformers** and a **lightweight BiLSTM** to analyze text sentiment, visualize insights, and make the results visually engaging. Perfect for demos, learning, or just showing off NLP magic! 😎

---

## 🎯 Project Goal

The goal of this project is to provide a **hands-on, visual, and interactive way** to explore sentiment analysis:

- **Live demo:** Type or paste any text and instantly see the predicted sentiment.
- **Batch demo:** Upload CSV files of reviews and get predictions with confidence scores.
- **Visualizations:** Word clouds, attention heatmaps, UMAP embeddings, and confusion matrices.
- **Explainability:** Optional token-level interpretability via attention or integrated gradients.
- **Training:** Train a small BiLSTM locally for demo purposes to understand sequence learning.

This project is **educational**, visually engaging, and demonstrates the **strength of Transformers vs. LSTMs** on text sentiment tasks.

---

## 📚 Inspirations & References

This project was inspired by foundational work in NLP and deep learning:

- **BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding**  
  Devlin et al., 2019. ACL NAACL-HLT.  
  [Read Paper](https://aclanthology.org/N19-1423.pdf?utm_source=chatgpt.com)

- **Attention Is All You Need** — Vaswani et al., 2017 (The original Transformer paper)  
  [Read Paper](https://arxiv.org/pdf/1706.03762)

- **LSTM: Long Short-Term Memory** — Hochreiter & Schmidhuber, 1997  
  [Read Paper](https://www.bioinf.jku.at/publications/older/2604.pdf?utm_source=chatgpt.com)

- **Stanford CS224N Lecture Series (RNNs, LSTMs, Transformers)**  
  [Watch on YouTube](https://youtube.com/playlist?list=PLoROMvodv4rMFqRtEuo6SGjY4XbRIVRd4&si=rjSYNTZl2OzN8JkH)

---

## 🛠 Features

1. **Live Text Sentiment Prediction**  
   Type any review or sentence and see the predicted sentiment immediately.

2. **Batch Prediction**  
   Upload CSV files with a `text` column and get predictions for multiple entries. Download results as a CSV.

3. **Visualizations & Explainability**  
   - Attention heatmaps for Transformers  
   - Word clouds for input text  
   - UMAP projection of embeddings  
   - Confusion matrices for evaluation

4. **Optional BiLSTM Training**  
   Train a small BiLSTM on a local CSV for demo purposes. Not production-grade but great for learning.

---

## 📦 Requirements

Install dependencies via:

```bash
pip install -r requirements.txt

```
## How to run
### Clone the repository
```bash
git clone <your-repo-url>
cd Sentiment_analysis
```
### Create a virtual environment (recommended):

```bash

python3 -m venv venv
source venv/bin/activate   # Linux/macOS
# OR
venv\Scripts\activate      # Windows PowerShell

```
### Run the Streamlit app:
```bach

streamlit run streamlit_sentiment_app.py

```
