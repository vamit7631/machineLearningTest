text = "Hello! How are You?"
stopwords = {"is", "the", "in", "and", "are", "to", "a"}

words = text.split()
filtered_words = [word for word in words if word.lower() not in stopwords]
print(filtered_words)

# output : ['Hello!', 'How', 'You?']

