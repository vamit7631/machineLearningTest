import math
from collections import Counter

def compute_tf(text):
    word_counts = Counter(text.split())
    total_words = len(text.split())
    return {word: count / total_words for word, count in word_counts.items()}

def compute_idf(corpus):
    num_documents = len(corpus)
    word_document_counts = Counter(word for sentence in corpus for word in set(sentence.split()))
    return {word: math.log(num_documents / (count + 1)) for word, count in word_document_counts.items()}

def compute_tfidf(corpus):
    idf_values = compute_idf(corpus)
    tfidf_vectors = []
    
    for sentence in corpus:
        tf_values = compute_tf(sentence)
        tfidf_vector = {word: tf_values.get(word, 0) * idf_values[word] for word in idf_values}
        tfidf_vectors.append(tfidf_vector)

    return tfidf_vectors

corpus = ["this is a sample", "this is another example sample"]
# Compute TF-IDF
tfidf_matrix = compute_tfidf(corpus)

print("TF-IDF Matrix:", tfidf_matrix)
