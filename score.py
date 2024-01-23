from rouge import Rouge


def get_Rouge_score(list_generated,list_abstract):
    rouge = Rouge()
    dict_score=rouge.get_scores(list_generated,list_abstract,avg=True)
    return dict_score
