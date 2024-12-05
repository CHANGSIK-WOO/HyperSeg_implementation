import argparse
import torch
import torch.nn as nn
import os
import json
from tqdm import tqdm
import torch.distributed as distributed

import shortuuid

import numpy as np
from transformers import StoppingCriteria, StoppingCriteriaList
from transformers import SiglipImageProcessor
from torch.utils.data import Dataset, DataLoader
import math
import copy
from typing import Dict, Optional, Sequence, List
from dataclasses import dataclass, field
import transformers

from hyperseg.utils.builder import load_pretrained_model

from hyperseg.utils import conversation as conversation_lib


from hyperseg.eval.eval_dataset.eval_datasets import VQA_Dataset



@dataclass
class DataArguments:

    local_rank: int = 0

    lora_enable: bool = True
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"    

    vision_tower: str = "../pretrained_model/siglip-so400m-patch14-384"
    vision_tower_mask: str = "../pretrained_model/mask2former/maskformer2_swin_base_IN21k_384_bs16_50ep.pkl"
    
    lazy_preprocess: bool = False
    is_multimodal: bool = True
    mm_use_im_start_end: bool = field(default=False)
    
    model_path: Optional[str] = field(default="../model/HyperSeg-3B")
    mask_config: Optional[str] = field(default="../hyperseg/model/mask_decoder/mask_config/maskformer2_swin_base_384_bs16_50ep.yaml")
    image_aspect_ratio: str = 'square'
    image_grid_pinpoints: Optional[str] = field(default=None)


    json_path: str = '../dataset/vqa_dataset/vqa_v2/bunny_vqav2_mscoco_test-dev2015.jsonl'
    image_folder: Optional[str] = field(default='../dataset/vqa_dataset/vqa_v2/test_imgs/test2015')

    model_map_name: str = 'HyperSeg'
    version: str = 'llava_phi'
    answers_file: str = '../output/vqav2.jsonl'

    num_chunks: int = 1
    chunk_idx: int = 0

    segmentation: bool = True
    eval_batch_size: int = 1
    dataloader_num_workers: int = 8
    
    eval_dataset: Optional[str] = field(default="vqa")

    enable_mgvp_seg_query: bool = field(default=True)




class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        last_token = input_ids[0][-1]
        for stop in self.stops:
            if stop == last_token:
                return True
        return False


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def init_distributed_mode(para):
    para.distributed = True
    if torch.cuda.device_count() <= 1:
        para.distributed = False
        para.local_rank = 0
        para.world_size = 1

    if para.distributed:
         # Init distributed environment
        distributed.init_process_group(backend="nccl")

        local_rank = distributed.get_rank()
        world_size = distributed.get_world_size()
        torch.cuda.set_device(local_rank)
        print('I am rank %d in this world of size %d!' % (local_rank, world_size))
        para.local_rank = local_rank
        para.world_size = world_size

def evaluation():
    parser = transformers.HfArgumentParser(DataArguments)
    data_args = parser.parse_args_into_dataclasses()[0]

    init_distributed_mode(data_args)
    model_path = os.path.expanduser(data_args.model_path)

    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, mask_config=data_args.mask_config, model_args=data_args)

    device = torch.device(data_args.local_rank if torch.cuda.is_available() else "cpu") 
    model.to(dtype=torch.float32, device=device)

    
    data_args.image_processor = image_processor


    answers_file = os.path.expanduser(data_args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    clip_image_processor = SiglipImageProcessor.from_pretrained(data_args.vision_tower)

    conversation_lib.default_conversation = conversation_lib.conv_templates[data_args.version]
    eval_dataset = VQA_Dataset(json_path=data_args.json_path, tokenizer=tokenizer, clip_image_processor=clip_image_processor,
                                                                    data_args=data_args)

    
    dataloader_params = {
        "batch_size": data_args.eval_batch_size,
        "num_workers": data_args.dataloader_num_workers,
    }
    if data_args.distributed:
        val_sampler = torch.utils.data.distributed.DistributedSampler(eval_dataset, shuffle=False, drop_last=False)
    else:
        val_sampler = None
    
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=dataloader_params['batch_size'],
        shuffle=False,
        num_workers=dataloader_params['num_workers'],
        pin_memory=False,
        sampler=val_sampler,)



    device = 'cuda' if torch.cuda.is_available() else 'cpu'


    model.eval()
    with torch.no_grad():
        for k, inputs in tqdm(enumerate(eval_dataloader), total=len(eval_dataloader)):
            idx = int(inputs["question_id"][0])
            cur_prompt = inputs["text"]

            inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}
            images_clip=inputs['images_clip'].float()
            if len(images_clip.shape) < 3:
                images_clip = None
            output_ids = model.eval_vqa(
                do_sample=False,
                temperature=0.,
                num_beams=1,
                max_new_tokens=64,
                eos_token_id = tokenizer.eos_token_id,
                use_cache=True,
                input_ids=inputs['input_ids'],
                images_clip=images_clip,
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            input_token_len = inputs['input_ids'].shape[1]
            n_diff_input_output = (inputs['input_ids'] != output_ids[:, :input_token_len]).sum().item()
            if n_diff_input_output > 0:
                print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
            outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
            outputs = outputs.strip()
            # outputs = outputs.replace('\n:', '').replace('\n', '')
            if '[SEG]' in outputs:
                outputs = outputs.replace('[SEG]', '')
            # if data_args.local_rank == 0:
            print(outputs, '\n')

            ans_id = shortuuid.uuid()
            ans_file.write(json.dumps({"question_id": idx,
                                    "prompt": cur_prompt,
                                    "text": outputs,
                                    "answer_id": ans_id,
                                    "model_id": model_name,
                                    "metadata": {}}) + "\n")
    
    ans_file.close()



    return None



if __name__ == "__main__":
    evaluation()

