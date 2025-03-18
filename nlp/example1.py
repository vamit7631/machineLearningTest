import re 

# Sample dataset (Manually created)
positive_words = {"good", "great", "awesome", "fantastic", "love", "excellent", "happy", "nice"}
negative_words = {"bad", "terrible", "awful", "hate", "worst", "poor", "sad", "horrible"}

def processText(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    words = text.split()
    return words


def classify_text(text):
    words = processText(text)

    pos_count = sum(1 for word in words if word in positive_words)
    neg_count = sum(1 for word in words if word in negative_words)

    if pos_count > neg_count:
        return "Positive"
    elif neg_count > pos_count:
        return "Negative"
    else:
        return "Neutral"
    

test_sentences = [
    "I love this product, it is fantastic!",
    "This is the worst experience ever. I hate it!",
    "The movie was okay, nothing special.",
    "What a great day! I feel awesome!",
    "The service was poor and the food was terrible."
]

for sentence in test_sentences:
    print(f"Sentence : {sentence}")
    print(f"Class: {classify_text(sentence)}\n")