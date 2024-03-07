import torch

from video_emb import CLIPVideoEmbedding
from text_emb import CLIPTextEmbedding


def get_topk(texts_tensor,video_embedding, k=20, frame_rate=1.0):
    top_idx = []
    mul = 1/frame_rate
    for st in texts_tensor:
        dot_product_result = torch.matmul(st, video_embedding.t())
        _, top_indices = torch.topk(dot_product_result, k=k)
        top_indices = top_indices.squeeze(dim=0)
        top_idx.append(top_indices*int(mul))
    return top_idx

def process_and_output(tensor):
    result = []
    tensor = tensor.tolist()  
    
    while tensor:
        current_value = tensor[0]
        end_range = current_value + 60
        
        current_range_values = [num for num in range(current_value, end_range + 1) if num in tensor]
        result.append(current_range_values)
        
        tensor = [num for num in tensor if num not in current_range_values]
    
    return result

def get_score(origin_tensor, sorted_tensor, k=20):
    origin_tensor = origin_tensor.tolist()
    score_list = []
    for sort in sorted_tensor:
        score = [k-origin_tensor.index(x) for x in sort]
        score_list.append(sum(score)/len(score) + len(score))
    return score_list

def return_timestamp(filtered_output, filtered_score, get=2):
    top_timestamp = []
    for _ in range(get):
        max_score = max(filtered_score)
        max_idx = filtered_score.index(max_score)
        max_output = filtered_output[max_idx]
        top_timestamp.append((max_output[0],max_output[-1]))
        
        filtered_output.remove(max_output)
        filtered_score.remove(max_score)
    
    return top_timestamp

def output(top_idx):
    output_list = []
    for idx in top_idx:
        sorted_tensor = torch.sort(idx).values.flatten()
        output = process_and_output(sorted_tensor)
        score_output = get_score(idx,output)
        
        filtered_output = [sublist for sublist in output if len(sublist) > 1 and sublist[-1] -sublist[0] > 3]
        filtered_score = [score for sublist, score in zip(output, score_output) if len(sublist) > 1 and sublist[-1] -sublist[0] > 3]
        
        output = return_timestamp(filtered_output,filtered_score)
        output_list+=output
    
    output_list = sorted(output_list, key = lambda x: x[0])
    
    return output_list


if __name__ == '__main__':
    sentences = ["Mack's encounter with the thousand spiders: This scene is a classic fear-based challenge that is sure to entertain audiences as Mack faces his arachnophobia in a high-pressure situation. The suspense builds as he tries to remain calm and complete the challenge, making it a crowd-pleaser.",
             "Mack's trust fall: In this scene, Mack's friendship with his partner is put to the test as he must trust him to catch him during a blind jump off a plank. The potential for disaster adds to the tension and excitement, making it an entertaining moment.",
             "Mack's struggle to save the duffel bags: This action-packed scene is full of suspense as Mack races against the clock to save as many bags of money as possible from a sinking car. The danger and difficulty of the task make it an exciting and entertaining moment.",
             "Mack's reaction to the Feastables bars: This lighthearted scene provides a nice contrast to the more intense challenges as Mack tries a new product and shares his thoughts on the new formula and flavors. The unexpected twist and Mack's honest reactions make it an entertaining moment.",
             "Mack's experience being buried alive: This nerve-wracking and entertaining moment tests Mack's mental strength as he faces his fear of being buried alive and struggles to keep track of time and deal with his claustrophobia. The psychological aspect of the challenge adds an extra layer of intrigue and suspense."]
    
    text_embedder = CLIPTextEmbedding()
    text_embedding = text_embedder.get_text_embedding(sentences)

    video_embedder = CLIPVideoEmbedding()
    video_path = "/root/Youtube-Short-Generator/videos/example.mp4"
    video_embedding = video_embedder.process_video_and_embedding(video_path)
    
    top_idx = get_topk(text_embedding,video_embedding)
    output_list = output(top_idx)
    print(output_list)
    

