import os
import torch
from transformers import CLIPTokenizer, CLIPTextModel

class CLIPTextEmbedding:
    '''
    summarize된 문장을 하나씩 넣어 각각의 임베딩을 뽑도록 하는 클래스
    
    input : 처리된 summarize문장. -> 처리된 문장 ? ex) ["sentence1","sentence2","sentence3"]
    output : 각 문장의 임베딩이 담겨있는 list (torch.tensor가 len(input)만큼 존재)
    '''
    def __init__(self, model_name="Searchium-ai/clip4clip-webvid150k", cache_dir=os.getcwd()):
        self.model = CLIPTextModel.from_pretrained(model_name, cache_dir=cache_dir)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name, cache_dir=cache_dir)

    def get_text_embedding(self, sentences):
        texts_tensor = []
        for text in sentences:
            inputs = self.tokenizer(text=text, return_tensors="pt")
            outputs = self.model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
            final_output = outputs[1] / outputs[1].norm(dim=-1, keepdim=True)
            texts_tensor.append(final_output)
        return texts_tensor

