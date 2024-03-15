def prompt_extsum_basic1():
    """
    Give role to LLM
    """
    template = """
    You are an abstractive summarizer that follows the output pattern.
    Please revise the extracted summary based on the document. The revised summary should include the information in the extracted summary. Document: {document} Extractive Summary: [Extractive Summary] [Format Instruction].
    """
    return template

def prompt_extsum_basic2():
    """
    slightly different with basic1. Does not give role
    """
    template = """
    script: {document} /n/n

    Please extract summary based on the document. The summary of the document should be 
    extracted to senteces inside the document. Document: [{document}] Summary: [Summary]

    """
    return template

def prompt_extsum_basic2_num():
    """
    Give instruction to generate within # sentences.
    To balance with original abstract data.
    
    num_sen: get_sententence_num(list_abstract[-1]) 
    """
    template = """
    script: {document} /n/n

    Please extract summary based on the document. The summary of the document should be 
    extracted to senteces inside the document within {num_sen} sentences. Document: [{document}] Summary: [Summary]

    """
    return template

def prompt_extsum_paper():
    """
    based on paper Extractive Summarization via ChatGPT for Faithful Summary Generation
    """
    template="""
    <s>[INST]<>You are an extractive summarizer that follows the output pattern.
    The following examples are successful extractive summarization instances: 

    Example Document: 'the three methods of assesing glycemic control , hba1c , smbg and cgms provide distinct information , yet complementary at the same time . hba1c does not equally reflect the glycemic values over the three months that forego its determination . \n hba1c assesses the average glycemic exposure in time without being able to differentiate between preprandial and postprandial glycemia , possible hypo and hyper glycemia . \n this method is able to identify both hypoglycemic and hyperglycemic episodes allowing immediate therapeutic decisions and therefore a glycemic balance closer to normal . \n introduction of cgms in the assessment of the glycemic status represents a great technological advance . \n this glucose monitoring method clears glycemic balance abnormalities in an otherwise impossible to obtain manner , evaluating both therapeutic efficiency and glycemic control . even if cgm systems are far from being implemented at a large scale in current practice , they are about to change the diabetes management by providing an optimal glycemic control .'
    Example Summary: 'type 2 diabetes is a chronic disease and maintaining a tight glycemic control is essential to prevent both microvascular and macrovascular complications , as demonstrated in previous studies . \n it is essential to monitor the glucose levels in order to achieve the targets . \n the blood glucose monitoring can be done by different methods : glycated haemoglobin a1c , self - monitoring of blood glucose ( before and after meals ) with a glucometer and continuous glucose monitoring with a system that measures interstitial glucose concentrations . even though glycated haemoglobin a1c is considered the  gold standard  of diabetes care \n , it does not provide complete information about the magnitude of the glycemic disequilibrium . \n therefore the self - monitoring and continuous monitoring of blood glucose are considered an important adjunct for achieving and maintaining optimal glycemic control . \n the three methods of assessing glycemic control : hba1c , smbg and cgms provide distinct but at the same time complementary information ,'
    
    Please summarize the following document.The summary should contain 3 sentences.
    Original Document: {document}<>[/INST]<\s>.
    """
    return template


def prompt_extsum_paper2():
    """
    based on paper Extractive Summarization via ChatGPT for Faithful Summary Generation
    Change example with shorter one and satisfied extsum
    """
    template="""
    <s>[INST]<>You are an extractive summarizer that follows the output pattern.
    The following examples are successful extractive summarization instances: 

    Example Document: 'Local communities have seen a dramatic increase in recycling rates over the past year, thanks to the implementation of new, user-friendly recycling programs. These programs include simplified sorting systems and more accessible drop-off points, making it easier for residents to participate. Additionally, the city has launched an educational campaign to raise awareness about the benefits of recycling and how it contributes to environmental sustainability. As a result, there has been a notable decrease in landfill waste, contributing to a reduction in greenhouse gas emissions and a positive impact on the local environment. Experts believe that if these initiatives are adopted on a wider scale, they could play a significant role in addressing global environmental challenges.'
    Example Summary: 'Local communities have seen a dramatic increase in recycling rates over the past year, thanks to the implementation of new, user-friendly recycling programs. As a result, there has been a notable decrease in landfill waste, contributing to a reduction in greenhouse gas emissions and a positive impact on the local environment. Experts believe that if these initiatives are adopted on a wider scale, they could play a significant role in addressing global environmental challenges.'
    
    Please summarize the following document.The summary should contain 3 sentences.
    Original Document: {document}<>[/INST]<\s>.
    """
    return template

def prompt_extsum_genbygpt():
    """
    extractive summarization prompts generated by GPT-4
    """
    template="""
    <s>[INST]<>Choose 3 sentences in 100 tokens that concisely capture the document's main argument, key evidence, and conclusion.

    Document: {document}<>[/INST]<\s>.
    """
    return template

def prompt_entertain_pick():
    """
    Domain Summary for entertainment
    """
    template="""
    <s>[INST]<> Find the entertaining moment in this script and return it.

    Document: {document}<>[/INST]<\s>.
    """
    return template

def prompt_comedy_pick():
    """
    Domain Summary for comedy
    """
    template="""
    <s>[INST]<> Find the funniest moment in this script and return it.

    Document: {document}<>[/INST]<\s>.
    """
    return template

def prompt_sports_pick():
    """
    Domain Summary for sports
    """
    template="""
    <s>[INST]<> Find the highlight scene of the match in this script and return it.

    Document: {document}<>[/INST]<\s>.
    """
    return template

def prompt_game_pick():
    """
    Domain Summary for game
    """
    template="""
    <s>[INST]<> Find the exciting scene of the gameplay in this script and return it.

    Document: {document}<>[/INST]<\s>.
    """
    return template

def prompt_retrieval():
    """
    Domain Summary for entertainment
    """
    template="""
    <s>[INST]<> Find the most appropriate parts in document with given explanation.
    Return original sentences of document.

    Document: {document} explanation:{pick}<>[/INST]<\s>.
    """
    return template

def prompt_reduce_common_pick():
    """
    Reduce prompt for all common domains to pick top scenes
    """
    template = """
    <s>[INST]<>Please pick out the top five scenes that could be the most important moments or 'hot clips' from various parts of the script.
    Important Moments: {ext_sum}<>[/INST]<\s>.
    """
    return template


def prompt_reduce_entertain_pick():
    """
    Reduce prompt for entertainment to pick top scenes
    """
    template = """
    <s>[INST]<>Please pick out the top five scenes that could be the most entertaining moments or 'hot clips' from various parts of the script.
    Entertaining Moments: {ext_sum}<>[/INST]<\s>.
    """
    return template

def prompt_reduce_news_pick():
    """
    Reduce prompt for news and politics to pick top scenes
    """
    template = """
    <s>[INST]<>Please pick out the top five scenes that could be the most informative moments or 'hot clips' from various parts of the script.
    Important Moments: {ext_sum}<>[/INST]<\s>.
    """
    return template

def prompt_reduce_comedy_pick():
    """
    Reduce prompt for comedy to pick top scenes
    """
    template = """
    <s>[INST]<>Please pick out the top five scenes that could be the most funniest moments or 'hot clips' from various parts of the script.
    Important Moments: {ext_sum}<>[/INST]<\s>.
    """
    return template

def prompt_reduce_sci_pick():
    """
    Reduce prompt for science and technology to pick top scenes
    """
    template = """
    <s>[INST]<>Please pick out the top five scenes that could be the most interesting scientific moments or 'hot clips' from various parts of the script.
    Important Moments: {ext_sum}<>[/INST]<\s>.
    """
    return template

def prompt_reduce_travel_pick():
    """
    Reduce prompt for travel and events to pick top scenes
    """
    template = """
    <s>[INST]<>Please pick out the top five scenes that could be the most fasntastic moments on travel or 'hot clips' from various parts of the script.
    Important Moments: {ext_sum}<>[/INST]<\s>.
    """
    return template
