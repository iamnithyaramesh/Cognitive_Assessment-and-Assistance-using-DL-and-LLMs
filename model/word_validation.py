import random
import string
import nltk
from nltk.corpus import stopwords

# Downloads (run once)
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

# Get the list of English stopwords
stop_words = set(stopwords.words('english'))

def get_random_letter():
    return random.choice(string.ascii_uppercase)

def is_proper_noun(word: str) -> bool:
  
    tagged = nltk.pos_tag([word])

    return tagged[0][1] in ("NNP", "NNPS")

def validate_word(word: str, letter: str) -> dict:
    word_clean = word.strip()

    if not word_clean:
        return {"valid": False, "reason": "Empty word"}

    if not word_clean[0].isalpha() or not word_clean.lower().startswith(letter.lower()):
        return {"valid": False, "reason": "Word does not start with the correct letter"}

    if is_proper_noun(word_clean):
        return {"valid": False, "reason": "Proper nouns not allowed"}

    if word_clean.lower() in stop_words:
        return {"valid": False, "reason": "Stop words not allowed"}

    if not word_clean.isalpha():
        return {"valid": False, "reason": "Only alphabetic characters allowed"}

    return {"valid": True} 


