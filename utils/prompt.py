def prompt_basic1(document):
    """
    Give role to LLM
    """
    template = """
    You are an abstractive summarizer that follows the output pattern.
    Please revise the extracted summary based on the document. The revised summary should include the information in the extracted summary. Document: {document} Extractive Summary: [Extractive Summary] [Format Instruction].
    """
    return template

def prompt_basic2(document):
    """
    slightly different with basic1. Does not give role
    """
    template = f"""
    script: {document} /n/n

    Please extract summary based on the document. The summary of the document should be 
    extracted to senteces inside the document. Document: [{document}] Summary: [Summary]

    """
    return template

def prompt_basic2_num(document, num_sen):
    """
    Give instruction to generate within # sentences.
    To balance with original abstract data.
    
    num_sen: get_sententence_num(list_abstract[-1]) 
    """
    template = f"""
    script: {document} /n/n

    Please extract summary based on the document. The summary of the document should be 
    extracted to senteces inside the document within {num_sen} sentences. Document: [{document}] Summary: [Summary]

    """
    return template




