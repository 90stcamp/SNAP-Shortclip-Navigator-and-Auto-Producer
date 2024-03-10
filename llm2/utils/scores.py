import numpy as np
from tqdm import tqdm
from rouge_score import rouge_scorer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def make_candidates(timestamps,shorts_time=60, interval=1):
    candidates = []
    n = len(timestamps)
    for i in range(0, n, interval):
        start, end = timestamps[i]['timestamp']
        temptext = timestamps[i]['text']
        for j in range(i+1, n):
            if end - start > shorts_time:
                break
            end = timestamps[j]['timestamp'][1]
            temptext += ' ' + timestamps[j]['text']
        candidates.append([temptext, start, end])
    return candidates

def calculate_cosine_similarity(str1, str2):
    vectorizer = TfidfVectorizer().fit_transform([str1, str2])
    vectors = vectorizer.toarray()
    csim = cosine_similarity(vectors)
    return csim[0][1]


def retrieve_top_k_rouge(text, candidates, k=1, score='rouge'):
    retrieve=[]
    scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
    for candidate in candidates:
        scores = scorer.score(text, candidate[0])    #우리가 요약한 문장과 쇼츠 후보와 rouge1점수 비교
        retrieve.append([*candidate,scores['rouge1'][2]])#[60초동안의 문장, 시작시간,끝나는시간, rouge1_F1점수]
    retrieve.sort(reverse = True, key = lambda x : x[3])#rouge1_F1점수로 sort
    return retrieve[0]

def retrieve_top_k_cosine(text, candidates, k=1):
    retrieve=[]
    for candidate in candidates:
        retrieve.append([*candidate,calculate_cosine_similarity(text,candidate[0])])
    retrieve.sort(reverse = True, key = lambda x : x[3])
    return retrieve[0]
