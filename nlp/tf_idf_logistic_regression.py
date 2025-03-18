import numpy as np
import re

# Sample dataset
news_data = [
    ("The team won the tournament after scoring the final goal.", "Sports"),
    ("Government announces new economic policy for development.", "Politics"),
    ("The match ended in a thrilling draw between the top teams.", "Sports"),
    ("Presidential election results will be announced tomorrow.", "Politics"),
    ("The player scored an amazing goal to win the game!", "Sports"),
    ("New tax laws proposed by the finance minister.", "Politics"),
    ("The striker netted a last-minute goal to secure victory.", "Sports"),
    ("Prime minister delivers speech on foreign policy.", "Politics"),
]

# Preprocess text
def preprocess_text(text):
    """Convert text to lowercase and tokenize."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
    return text.split()

# Build vocabulary
corpus = [preprocess_text(text) for text, _ in news_data]
vocab = list(set(word for doc in corpus for word in doc))

# Convert text to TF-IDF vectors
def compute_tf_idf(corpus, vocab):
    """Convert text corpus into TF-IDF feature vectors."""
    doc_count = len(corpus)
    tf_matrix = np.zeros((doc_count, len(vocab)))
    
    # Compute TF
    for i, doc in enumerate(corpus):
        for word in doc:
            if word in vocab:
                tf_matrix[i, vocab.index(word)] += 1
        tf_matrix[i] /= len(doc)  # Normalize
    
    # Compute IDF
    df = np.sum(tf_matrix > 0, axis=0)
    idf = np.log(doc_count / (df + 1))  # Avoid division by zero
    
    # Compute TF-IDF
    tfidf_matrix = tf_matrix * idf
    return tfidf_matrix

X = compute_tf_idf(corpus, vocab)
y = np.array([1 if label == "Sports" else 0 for _, label in news_data])  # 1 for Sports, 0 for Politics

# Train Logistic Regression (Gradient Descent)
weights = np.zeros(X.shape[1])  # Initialize weights
bias = 0
learning_rate = 0.1
epochs = 1000

for _ in range(epochs):
    linear_model = np.dot(X, weights) + bias
    y_pred = 1 / (1 + np.exp(-linear_model))  # Sigmoid function
    
    # Compute gradients
    dw = np.dot(X.T, (y_pred - y)) / len(y)
    db = np.sum(y_pred - y) / len(y)
    
    # Update weights
    weights -= learning_rate * dw
    bias -= learning_rate * db

# Prediction function
def predict_news(text):
    """Predict if news is Sports or Politics using Logistic Regression."""
    words = preprocess_text(text)
    features = np.zeros(len(vocab))
    
    for word in words:
        if word in vocab:
            features[vocab.index(word)] += 1
    features /= len(words)  # Normalize
    
    linear_model = np.dot(features, weights) + bias
    probability = 1 / (1 + np.exp(-linear_model))  # Sigmoid
    return "Sports" if probability > 0.5 else "Politics"

# Test predictions
test_news = [
    "The championship match was thrilling with a last-minute goal!",
    "The president discusses new foreign trade policies.",
    "The striker scored an incredible goal to win the game.",
    "A new healthcare policy was introduced by the government."
]

for news in test_news:
    print(f"News: {news}")
    print(f"Predicted Category: {predict_news(news)}\n")
