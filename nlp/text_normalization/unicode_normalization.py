import unicodedata

def unicodeNormalization(text):
    return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')



text = "Café is amazing!"
normalizeText = unicodeNormalization(text)
print(f"Unicode normalize text : {normalizeText}")

# Unicode normalize text : Cafe is amazing!