import nltk
from fuzzywuzzy import process
from nltk import word_tokenize, pos_tag

# Download necessary NLTK resources (only once needed)
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')

# Known animal list
animal_list = [
    "cat", "dog", "elephant", "lion", "tiger", "giraffe", "zebra",
    "kangaroo", "panda", "bear", "wolf", "fox", "monkey", "rabbit",
    "cow", "goat", "sheep","horse"
]

def fuzzy_match(word, choices, threshold=80):
    match, score = process.extractOne(word, choices)
    return match if score >= threshold else None

def guess_animal(text):
    tokens = word_tokenize(text)
    pos_tags = pos_tag(tokens)
    l=[]
    for word, tag in pos_tags:
        # NN = singular noun, NNS = plural noun, etc.
        if tag.startswith("NN"):
            lower_word = word.lower()
            match = fuzzy_match(lower_word, animal_list)
            if match:
                l.append(lower_word)
    return l
