corpus = ["Hello world", "Hello NLP", "NLP is fun"]
word_set = set(word.lower() for sentence in corpus for word in sentence.split())

bow = [{word: sentence.lower().split().count(word) for word in word_set} for sentence in corpus]
print(bow)

# output

# [{'fun': 0, 'world': 1, 'hello': 1, 'nlp': 0, 'is': 0},
#  {'fun': 0, 'world': 0, 'hello': 1, 'nlp': 1, 'is': 0}, 
#  {'fun': 1, 'world': 0, 'hello': 0, 'nlp': 1, 'is': 1}]