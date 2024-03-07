import os
import torch
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm
from transformers import CLIPVisionModelWithProjection
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, InterpolationMode

class CLIPVideoEmbedding:
    '''
    video를 받아 각 frame의 임베딩을 뽑는 클래스
    
    input : video path
    output : video frame embedding tensor
    '''
    def __init__(self, model_name="Searchium-ai/clip4clip-webvid150k", cache_dir=os.getcwd()):
        self.model = CLIPVisionModelWithProjection.from_pretrained(model_name, cache_dir=cache_dir)

    def preprocess(self, size, n_px):
        return Compose([
            Resize(size, interpolation=InterpolationMode.BICUBIC),            
            CenterCrop(size),
            lambda image: image.convert("RGB"),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])(n_px)

    def process_video_and_embedding(self, video_path, frame_rate=1.0, size=224, segment_size=1500):
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = round(float(cap.get(cv2.CAP_PROP_FPS)))
        segment_size = segment_size * fps

        if fps < 1:
            print("ERROR: problem reading video file: ", video_path)
            return None

        total_duration = (frame_count + fps - 1) // fps
        interval = fps / frame_rate

        all_video_embeds = []
        
        for start_sec in tqdm(np.arange(0, total_duration, segment_size/fps),desc="processing..."):
            end_sec = min(start_sec + segment_size / fps, total_duration)

            frames_idx = np.floor(np.arange(start_sec * fps, end_sec * fps, interval))
            ret = True
            images = np.zeros([len(frames_idx), 3, size, size], dtype=np.float32)

            for i, idx in enumerate(frames_idx):
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if not ret: 
                    break

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                last_frame = i
                images[i, :, :, :] = self.preprocess(size, Image.fromarray(frame).convert("RGB"))

            images = images[:last_frame + 1]
            video_segment = torch.tensor(images)

            model = self.model.eval()
            torch.cuda.empty_cache()

            with torch.no_grad():
                visual_output = model(video_segment)

            visual_output = visual_output["image_embeds"]
            visual_output = visual_output / visual_output.norm(dim=-1, keepdim=True)
            all_video_embeds.append(visual_output)

        cap.release()
        return torch.cat(all_video_embeds)

