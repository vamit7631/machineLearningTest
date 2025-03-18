import re

positive_words = {"happy", "love", "excellent", "good", "great", "fantastic", "awesome", "enjoy"}
negative_words = {"sad", "hate", "terrible", "bad", "awful", "worst", "poor", "disappointed"}


def processText(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    words = text.split()
    return words


def classify_sentiment(text):
    words = processText(text)
    pos_count = sum(1 for word in words if word in positive_words)
    neg_count = sum(1 for word in words if word in negative_words)

    if pos_count > neg_count:
        return "POSITIVE"
    
    elif neg_count > pos_count:
        return "NEGATIVE"
    
    else:
        return "NEUTRAL"

sentences = [
    "I love this movie, it was fantastic!",
    "This is the worst book I have ever read.",
    "The food was okay, nothing great or bad.",
    "What an excellent day, I feel awesome!",
    "I am so disappointed with this product."
]

# Classify each sentence
for sentence in sentences:
    print(f"Sentence: {sentence}")
    print(f"Class: {classify_sentiment(sentence)}\n")