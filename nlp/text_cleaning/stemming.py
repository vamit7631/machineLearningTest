from nltk.stem import PorterStemmer

stemmer = PorterStemmer()
words = ["running", "played", "flies", "easily", "studies"]
stemmed_word = [stemmer.stem(word) for word in words]
print(stemmed_word)

#  output : ['run', 'play', 'fli', 'easili', 'studi']

