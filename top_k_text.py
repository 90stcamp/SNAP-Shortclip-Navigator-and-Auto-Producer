from rouge_score import rouge_scorer

#text는 [timestamps, summarized text] 형태
def top_k_text(text, shorts_time =60,k = 3 ,interval = 1):
    candidates = [] #쇼츠 후보군들
    # shorts_time = 60 #쇼츠후보들의 초수
    # k = 3            #몇개의 후보를 선정할건지
    # interval = 1     #몇개의 문장을 기준으로 나눌껀지

    for i in range(0,len(text[0]),interval):
        start = text[0][i][1]       #시작시간
        temptext = text[0][i][0]    #첫문장
        for j in range(i+1,len(text[0])):
            end = text[0][j][2]     #끝나는시간
            temptext= ' '.join([temptext,text[0][j][0]])
            if end-start>shorts_time:
                break               #계속 합치다가 60초가 넘어가면 break
        candidates.append([temptext,start,end]) #60초동안의 문장,시작시간,끝나는시간 저장
        if j == len(text[0])-1:     #끝나는시간이 영상끝이라면 끝
            break
        
    scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
    for i, candidate in enumerate(candidates):
        scores = scorer.score(text[1], candidate[0])    #우리가 요약한 문장과 쇼츠 후보와 rouge1점수 비교
        candidates[i] = [*candidate,scores['rouge1'][2]]#[60초동안의 문장, 시작시간,끝나는시간, rouge1_F1점수]

    candidates.sort(reverse = True, key = lambda x : x[3])#rouge1_F1점수로 sort

    final_candidates = candidates[:k]
    return final_candidates
