import nltk
import numpy as np
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer

stemming = PorterStemmer()

def tokenize(sentence):
    tokens = word_tokenize(sentence)
    lower_tokens = [t.lower() for t in tokens]
    return lower_tokens

def español(spanish):
    stop_words_sp= set(stopwords.words('spanish'))
    texto=[w for w in spanish if not w in stop_words_sp]
    return texto

def stemm(word):
    return stemming.stem(word)

def bolsa_palabras(tokenize_sentence, todo):
    tokenize_sentence=[stemm(w) for w in tokenize_sentence]
    bag=np.zeros(len(todo), dtype=np.float32)
    for idx, w in enumerate(todo):
        if w in tokenize_sentence:
            bag[idx]=1.0
    return bag