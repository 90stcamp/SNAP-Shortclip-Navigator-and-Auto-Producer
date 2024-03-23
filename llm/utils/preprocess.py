from nltk.tokenize import sent_tokenize, word_tokenize


## nltk.download('punkt') 안했으면 terminal에서 설치하기

def get_sententence_num(text):
    return len(sent_tokenize(text))

def get_token_num(text):
    return len(text.encode('utf-8') )

