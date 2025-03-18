import re

# Define spam keywords
spam_keywords = {"win", "prize", "free", "money", "offer", "lottery", "click", "buy", "subscribe"}

def preprocess_text(text):
    """Convert text to lowercase, remove punctuation, and tokenize."""
    text = text.lower()  # Convert to lowercase
    text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
    words = text.split()  # Tokenize
    return words

def classify_message(text):
    """Classify message as Spam or Not Spam based on keyword occurrences."""
    words = preprocess_text(text)
    spam_count = sum(1 for word in words if word in spam_keywords)
    
    return "Spam" if spam_count > 1 else "Not Spam"

# Test messages
messages = [
    "You have won a FREE prize! Click here to claim.",
    "Hello, how are you doing today?",
    "Limited time offer! Buy now and save money.",
    "Your subscription is about to expire, renew now!",
    "Let's meet for lunch tomorrow."
]

# Classify each message
for message in messages:
    print(f"Message: {message}")
    print(f"Class: {classify_message(message)}\n")
