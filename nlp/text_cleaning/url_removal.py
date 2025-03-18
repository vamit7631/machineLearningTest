text = "Visit https://example.com or contact me at email@example.com."
clean_text = re.sub(r'https?://\S+|www\.\S+|[\w\.-]+@[\w\.-]+', '', text)
print(clean_text)  # "Visit  or contact me at ."
