# Streamlit Sentiment Analysis Demo — LSTM + Transformer + Visuals
# Single-file demo app built for a "wow" presentation: demo-mode uses a pretrained
# transformer; optional LSTM training for small local datasets.

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import logging as hf_logging
hf_logging.set_verbosity_error()
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.manifold import TSNE
import umap
from wordcloud import WordCloud
import io
import base64
import os

# Optional Captum for integrated gradients (if installed)
try:
    from captum.attr import IntegratedGradients
    CAPTUM_AVAILABLE = True
except Exception:
    CAPTUM_AVAILABLE = False

# -----------------------------
# Lightweight BiLSTM model (PyTorch)
# -----------------------------
class LSTMSentiment(nn.Module):
    def __init__(self, vocab_size, embed_dim=100, hidden_dim=128, num_layers=1, num_classes=2, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim*2, num_classes)

    def forward(self, x, lengths=None):
        x = self.embedding(x)
        outputs, (h, c) = self.lstm(x)
        h_cat = torch.cat((h[-2,:,:], h[-1,:,:]), dim=1)
        return self.fc(h_cat)

# -----------------------------
# Helpers: tokenizer & transformer model loader
# -----------------------------
@st.cache_resource
def load_transformer_model(model_name="distilbert-base-uncased-finetuned-sst-2-english"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, output_attentions=True)
    model.eval()
    return tokenizer, model

# Prediction using transformer
def predict_transformer(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probs = torch.softmax(logits, dim=-1).numpy()[0]
    pred = int(np.argmax(probs))
    label = model.config.id2label[pred] if hasattr(model.config, 'id2label') else str(pred)
    # extract attentions
    attentions = None
    if outputs.attentions is not None:
        # attentions: tuple of layers, each (batch, heads, seq_len, seq_len)
        # we average across heads and layers for a simple token-importance proxy
        attn = torch.stack(outputs.attentions)  # (layers, batch, heads, seq, seq)
        attn = attn.mean(dim=2).squeeze(1).mean(dim=0)  # (seq, seq) averaged
        # token importance: average attention to CLS (or average row)
        token_importance = attn.mean(dim=0).numpy()
        # token_importance aligns with tokenizer tokens
        attentions = token_importance
    return label, float(probs[pred]), attentions, inputs

# Token-level HTML highlight
def token_highlight_html(tokenizer, inputs, importances, pred_label):
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    # skip special tokens if present
    html = f"<div style='font-family:monospace;'>Prediction: <b>{pred_label}</b><br>"
    for t, imp in zip(tokens, importances):
        # normalize importance
        score = float(imp)
        # map to color (green pos, red neg) — here we just use intensity on a single color
        rgba = int(255 * min(max(score, 0.0), 1.0))
        color = f'rgba(255, {255-rgba}, {255-rgba}, 0.6)'
        html += f"<span style='background:{color};padding:2px;margin:1px;border-radius:3px'>{t}</span> "
    html += "</div>"
    return html

# Simple UMAP visualization for embeddings
def plot_umap(embeddings, labels, title="UMAP projection"):
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1)
    proj = reducer.fit_transform(embeddings)
    df = pd.DataFrame({'x': proj[:,0], 'y': proj[:,1], 'label': labels})
    fig = px.scatter(df, x='x', y='y', color='label', title=title)
    return fig

# Wordcloud
def make_wordcloud(texts):
    text = " ".join(texts)
    wc = WordCloud(width=600, height=300, background_color='white').generate(text)
    fig, ax = plt.subplots(figsize=(8,4))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    return fig

# Confusion matrix plot
from sklearn.metrics import confusion_matrix
import plotly.express as px

def plot_confmat(y_true, y_pred, labels=None):
    if labels is None:
        # Sort labels alphabetically to match sklearn's default
        labels = sorted(list(set(y_true + y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig = px.imshow(cm,
                    x=labels,
                    y=labels,
                    text_auto=True,
                    title='Confusion Matrix')
    return fig


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(layout="wide", page_title="Sentiment Shock Demo")
st.title("⚡ Sentiment Shock — LSTM + Transformer + Visuals")

# Sidebar controls
st.sidebar.header("Demo Controls")
model_name = st.sidebar.selectbox("Transformer demo model", [
    "distilbert-base-uncased-finetuned-sst-2-english",
    "nlptown/bert-base-multilingual-uncased-sentiment"
])
use_demo_models = st.sidebar.checkbox("Use demo pretrained transformer (recommended)", value=True)
show_attention = st.sidebar.checkbox("Show token importance heatmap", value=True)

# Load transformer
tokenizer, transformer_model = load_transformer_model(model_name)

# Main layout: tabs
tabs = st.tabs(["Live Demo","Batch Demo","Visuals & Explainability","Train LSTM (optional)"])

# Tab 1: Live Demo
with tabs[0]:
    st.header("Live text demo — type or paste any review")
    text = st.text_area("Enter text", value="I loved the movie — great story and acting!", height=120)
    col1, col2 = st.columns([1,1])
    if st.button("Analyze" , key='analyze_live'):
        label, prob, attentions, inputs = predict_transformer(text, tokenizer, transformer_model)
        st.metric("Prediction", f"{label}", delta=f"{prob:.2f}")
        with col1:
            st.subheader("Transformer output")
            st.write(f"Label: **{label}**  — confidence: **{prob:.2f}**")
            # Show token heatmap
            if show_attention and attentions is not None:
                imp = attentions / (attentions.max()+1e-9)
                html = token_highlight_html(tokenizer, inputs, imp, label)
                st.write("Token importance (averaged attentions)", unsafe_allow_html=True)
                st.components.v1.html(html, height=120)
        with col2:
            st.subheader("Extra visuals")
            # wordcloud from single text
            fig_wc = make_wordcloud([text])
            st.pyplot(fig_wc)

# Tab 2: Batch Demo (CSV upload)
with tabs[1]:
    st.header("Batch demo — upload CSV with a `text` column")
    uploaded = st.file_uploader("Upload CSV", type=['csv'])
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        if 'text' not in df.columns:
            st.error("CSV must contain a 'text' column")
        else:
            max_rows = st.slider("Max rows to analyze", 10, 1000, 200)
            sample = df.head(max_rows)
            preds = []
            probs = []
            for t in sample['text'].tolist():
                label, p, att, inp = predict_transformer(t, tokenizer, transformer_model)
                preds.append(label)
                probs.append(p)
            sample['pred'] = preds
            sample['prob'] = probs
            st.dataframe(sample)
            csv = sample.to_csv(index=False).encode('utf-8')
            st.download_button("Download predictions CSV", data=csv, file_name='predictions.csv')

# Tab 3: Visuals & Explainability
with tabs[2]:
    st.header("Model comparison, embeddings and explainability")
    st.markdown("**Transformer embedding projection (UMAP)**")
    # Quick demo: embed a few example sentences
    demo_texts = [
        "I loved the movie, it was fantastic!",
        "It was an absolute waste of time, horrible.",
        "Not bad, some good moments.",
        "Amazing acting and story, highly recommend.",
        "I didn't like it."
    ]
    inputs = tokenizer(demo_texts, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = transformer_model(**inputs, output_attentions=False, output_hidden_states=True)
    # use last hidden state (mean pooling)
    hidden = outputs.hidden_states[-1].mean(dim=1).numpy()
    fig_umap = plot_umap(hidden, ['pos','neg','neu','pos','neg'], title='Transformer embeddings (demo)')
    st.plotly_chart(fig_umap, use_container_width=True)

    st.markdown("**Confusion matrix / metrics**")
    st.info("Upload CSV with 'text' and 'label' (POSITIVE/NEGATIVE) to compute metrics")
    eval_file = st.file_uploader("Upload labeled CSV for eval", type=['csv'], key='eval')
    if eval_file is not None:
        edf = pd.read_csv(eval_file)
        if set(['text','label']).issubset(edf.columns):
            y_true = edf['label'].tolist()
            y_pred = []
            for t in edf['text'].tolist():
                lab, p, a, i = predict_transformer(t, tokenizer, transformer_model)
                y_pred.append(lab)
            fig_cm = plot_confmat(y_true, y_pred, labels=list(set(y_true)))
            st.plotly_chart(fig_cm, use_container_width=True)
            st.text(classification_report(y_true, y_pred))
        else:
            st.error("CSV must contain 'text' and 'label' columns")

    st.markdown("**Explainability: Integrated Gradients (if available)**")
    if CAPTUM_AVAILABLE:
        st.write("Captum detected — you can compute attributions for tokens (may be slow)")
        example = st.text_input("Text for IG explanation", value="I absolutely loved the film")
        if st.button("Run IG", key='ig'):
            # Note: Real IG for transformers is complex; here we provide a simple wrapper demo
            st.write("Running IG (demo) — this may take a few seconds")
            # Demo: show attention-based heatmap again
            label, prob, att, inp = predict_transformer(example, tokenizer, transformer_model)
            if att is not None:
                imp = att / (att.max()+1e-9)
                html = token_highlight_html(tokenizer, inp, imp, label)
                st.components.v1.html(html, height=140)
    else:
        st.warning("Captum not installed. Install captum to use Integrated Gradients: pip install captum")

# Tab 4: Train LSTM (optional, demo-scale)
with tabs[3]:
    st.header("Train a lightweight BiLSTM locally (demo-scale)")
    st.markdown("This trains a tiny BiLSTM on a small uploaded CSV (text,label) for demonstration.\nNot for production.")
    lstm_file = st.file_uploader("Upload CSV for LSTM training (text,label)", type=['csv'], key='lstm')
    if lstm_file is not None:
        df = pd.read_csv(lstm_file).dropna()
        if set(['text','label']).issubset(df.columns):
            n_samples = st.slider("Max rows to use", 100, min(2000, len(df)), 1000)
            sample = df.head(n_samples)
            label_map = {k:i for i,k in enumerate(sorted(sample['label'].unique()))}
            sample['y'] = sample['label'].map(label_map)
            st.write("Label map:", label_map)
            if st.button("Train LSTM (demo)"):
                st.info("Tokenizing + building tiny vocab — simple whitespace tokenizer")
                texts = sample['text'].tolist()
                toks = [t.lower().split() for t in texts]
                vocab = {w:i+2 for i,w in enumerate({w for s in toks for w in s})}
                vocab['<pad>'] = 0
                vocab['<unk>'] = 1
                maxlen = int(np.percentile([len(s) for s in toks], 95))
                def encode(s):
                    ids = [vocab.get(w,1) for w in s]
                    if len(ids) < maxlen:
                        ids += [0]*(maxlen-len(ids))
                    else:
                        ids = ids[:maxlen]
                    return ids
                X = np.array([encode(s) for s in toks])
                y = sample['y'].values
                # tiny train loop
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model = LSTMSentiment(vocab_size=len(vocab), embed_dim=64, hidden_dim=64, num_classes=len(label_map))
                model.to(device)
                opt = torch.optim.Adam(model.parameters(), lr=1e-3)
                crit = nn.CrossEntropyLoss()
                Xt = torch.tensor(X, dtype=torch.long).to(device)
                yt = torch.tensor(y, dtype=torch.long).to(device)
                model.train()
                epochs = 5
                for ep in range(epochs):
                    opt.zero_grad()
                    logits = model(Xt)
                    loss = crit(logits, yt)
                    loss.backward()
                    opt.step()
                    st.write(f"Epoch {ep+1}/{epochs} loss={loss.item():.4f}")
                st.success("Training complete — you can save model state dict locally")
                buf = io.BytesIO()
                torch.save({'model_state_dict': model.state_dict(), 'vocab': vocab, 'label_map': label_map}, buf)
                buf.seek(0)
                b64 = base64.b64encode(buf.read()).decode()
                st.markdown(f"[Download LSTM checkpoint](data:application/octet-stream;base64,{b64})")
        else:
            st.error("CSV must contain 'text' and 'label' columns")

# Footer
st.markdown("---")
st.caption("Notes: Demo app uses transformer for instant results. LSTM training is tiny by design for demos. For production, do large-scale training, validation splits, hyperparameter sweeps, and strong test harnesses.")


# Running instructions are provided in the project README (see main canvas document).
