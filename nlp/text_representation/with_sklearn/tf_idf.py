from sklearn.feature_extraction.text import TfidfVectorizer

corpus = ["this is a sample", "this is another example sample"]
# Initialize TF-IDF Vectorizer
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(corpus)

# Convert to array and print
print("Feature Names:", vectorizer.get_feature_names_out())
print("TF-IDF Matrix:\n", tfidf_matrix.toarray())
