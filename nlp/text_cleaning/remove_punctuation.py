import re 

text = "Hello! How are you doing today?"
clean_text = re.sub(r'[^\w\s]', '', text)
print(clean_text)

# output : Hello How are you doing today