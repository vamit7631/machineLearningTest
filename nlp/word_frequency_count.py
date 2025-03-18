from collections import Counter

text = "Hello world! Hello everyone. Welcome to the world of NLP."

words = text.lower().split()
word_freq = Counter(words)
print(word_freq) 

# output : Counter({'hello': 2, 'world!': 1, 'everyone.': 1, 'welcome': 1, 'to': 1, 'the': 1, 'world': 1, 'of': 1, 'nlp.': 1})
