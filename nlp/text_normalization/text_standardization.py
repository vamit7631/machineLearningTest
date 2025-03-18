import contractions

def standardize_text(text):
    return contractions.fix(text)

# Example Usage
text = "u gotta try it. I'm gonna go now."
standardized_text = standardize_text(text)
print("Standardized Text:", standardized_text)
