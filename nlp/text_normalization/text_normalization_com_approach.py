import contractions
import inflect
import wordninja
from symspellpy import SymSpell

# Load Dictionary
dictionary_path = "frequency_dictionary_en_82_765.txt"
sym_spell = SymSpell()
if not sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1):
    print(f"Error: Could not load dictionary from {dictionary_path}.")
    exit()  # Stop execution if the dictionary is missing

p = inflect.engine()

def normalize_text(text):
    text = contractions.fix(text)
    text = " ".join([p.number_to_words(word) if word.isdigit() else word for word in text.split()])
    
    # Spelling correction
    suggestions = sym_spell.lookup_compound(text, max_edit_distance=2)
    corrected_text = suggestions[0].term if suggestions else text
    
    # Word segmentation
    segmented_text = " ".join(wordninja.split(corrected_text))

    return segmented_text

# Test
text = "u gotta try teh 2 newyork pizzas!"
normalized_text = normalize_text(text)
print(normalized_text)


# output :  you got to try the two new york pizzas !