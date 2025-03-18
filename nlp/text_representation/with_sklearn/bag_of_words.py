from sklearn.feature_extraction.text import CountVectorizer

corpus = ["Hello world", "Hello NLP", "NLP is fun"]

vectorizer = CountVectorizer()
bow_matrix = vectorizer.fit_transform(corpus)

print("Feature names : ", vectorizer.get_feature_names_out())
print("Bow matrix: \n", bow_matrix.toarray())