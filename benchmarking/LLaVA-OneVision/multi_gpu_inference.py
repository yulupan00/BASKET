from operator import attrgetter
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle

import torch
import cv2
import numpy as np
from PIL import Image
import requests
import copy
import warnings
from decord import VideoReader, cpu

import re
from tqdm import tqdm

warnings.filterwarnings("ignore")

import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)

pretrained = "your_llava-ov_checkpoint"

model_name = "llava_qwen"
llava_model_args = {
    "multimodal": True,
}

def load_model(device):
    """Initialize model on a specific GPU."""
    tokenizer, model, image_processor, max_length = load_pretrained_model(
        pretrained, None, model_name, device_map={"": device}, attn_implementation="sdpa", **llava_model_args
    )
    model.eval()
    return tokenizer, model, image_processor

def load_video(video_path, max_frames_num):
    """Function to extract frames from video"""
    if isinstance(video_path, str):
        vr = VideoReader(video_path, ctx=cpu(0))
    else:
        vr = VideoReader(video_path[0], ctx=cpu(0))
    total_frame_num = len(vr)
    uniform_sampled_frames = np.linspace(0, total_frame_num - 1, max_frames_num, dtype=int)
    frame_idx = uniform_sampled_frames.tolist()
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    return spare_frames

def split_list(lst, num_splits):
    """Split a list into evenly sized chunks"""
    chunk_size = len(lst) // num_splits
    remainder = len(lst) % num_splits
    chunks = []
    start = 0
    for i in range(num_splits):
        end = start + chunk_size + (1 if i < remainder else 0)
        chunks.append(lst[start:end])
        start = end
    return chunks

def process_chunk(gpu_id, chunk, results):
    """Process a chunk of data on a specific GPU."""
    device = f"cuda:{gpu_id}"
    tokenizer, model, image_processor = load_model(device)
    model = model.to(device)

    res_l = []
    for sample in tqdm(chunk, desc=f"GPU {gpu_id}"):
        try:
            # Load and process video
            video_path = sample[0]
            video_frames = load_video(video_path, 32)
            frames = image_processor.preprocess(video_frames, return_tensors="pt")["pixel_values"].half().to(device)

            # Prepare conversation input
            conv_template = "qwen_1_5"
            
            question = f"{DEFAULT_IMAGE_TOKEN}\nAnalyze the video content and evaluate the performance for the following 20 skills. Assign a ranking to each skill on a scale of 0 to 4, where 0 represents no impact or poor performance and 4 represents excellent performance. Follow the format: `Skill Name: Ranking` (e.g., `Free Throw: 4`). The skills to evaluate are: Free Throw, 2-PTs, 3-PTs, Contested-shots, Overall Shooting, Rebounds, Defensive Rebounds, Offensive Rebounds, Steals, Turnovers, Points-allowed, Defensive Consistency, Assists, Passing Accuracy, Fouls, Contribution, Offensive Consistency, Teamwork, Impact, Efficiency"
            
            conv = copy.deepcopy(conv_templates[conv_template])
            conv.append_message(conv.roles[0], question)
            conv.append_message(conv.roles[1], None)
            prompt_question = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
            image_sizes = [frame.size for frame in video_frames]

            # Generate response
            cont = model.generate(
                input_ids,
                images=[frames],
                image_sizes=image_sizes,
                do_sample=False,
                temperature=0,
                max_new_tokens=4096,
                modalities=["video"],
            )
            text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)

            src_txt = text_outputs[0].replace('\n', ',')

            src_num = re.findall(r':\s*(\d+)', src_txt)
            src_num = [int(num) for num in src_num]
            if len(src_num) != 20:
                print(f'output skill num mismatch! {len(src_num)}')

            tgt_num = [int(x) for x in sample[1:]]

            matches = sum(1 for x, y in zip(src_num, tgt_num) if x == y) / len(tgt_num)
            res_l.append(matches)

        except Exception as e:
            print(f"Error on GPU {gpu_id}: {e}")
            continue

    results[gpu_id] = res_l

if __name__ == "__main__":
    test_path = 'your_BASKET_test_path'

    test_list = []
    with open(test_path, 'r') as f:
        for line in f.readlines()[1:-2]:
            test_list.append(line.strip().split(','))

    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("No GPUs available")

    test_chunks = split_list(test_list, num_gpus)

    manager = mp.Manager()
    results = manager.dict()
    processes = []

    for gpu_id, chunk in enumerate(test_chunks):
        p = mp.Process(target=process_chunk, args=(gpu_id, chunk, results))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    # Aggregate results
    all_results = []
    for res in results.values():
        all_results.extend(res)

    average = sum(all_results) / len(all_results)
    print(f'Average accuracy out of 20: {average}')
