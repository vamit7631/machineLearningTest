from textblob import TextBlob 

def correctSpelling(text):
    return str(TextBlob(text).correct())



text = "Teh coffee iz amzing!"
correctText = correctSpelling(text)
print("Corrected Text:", correctText)