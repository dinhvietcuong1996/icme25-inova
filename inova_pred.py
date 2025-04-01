#!/usr/bin/env python
# coding: utf-8
import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from llava.constants import IGNORE_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IMAGE_TOKEN_INDEX
from typing import Dict, Optional, Sequence, List
import transformers
import re

from PIL import Image
import math

multichoice_letters = ["a", "b", "c", "d", "e", "f", "g", "h"]
multichoice_letters = [c.upper() for c in multichoice_letters]

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def preprocess_qwen(sources, tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False, max_len=2048, system_message: str = "You are a helpful assistant.") -> Dict:
    roles = {"human": "<|im_start|>user", "gpt": "<|im_start|>assistant"}

    im_start, im_end = tokenizer.additional_special_tokens_ids
    nl_tokens = tokenizer("\n").input_ids
    _system = tokenizer("system").input_ids + nl_tokens
    _user = tokenizer("user").input_ids + nl_tokens
    _assistant = tokenizer("assistant").input_ids + nl_tokens

    # Apply prompt templates
    input_ids, targets = [], []

    source = sources
    if roles[source[0]["from"]] != roles["human"]:
        source = source[1:]

    input_id, target = [], []
    system = [im_start] + _system + tokenizer(system_message).input_ids + [im_end] + nl_tokens
    input_id += system
    target += [im_start] + [IGNORE_INDEX] * (len(system) - 3) + [im_end] + nl_tokens
    assert len(input_id) == len(target)
    for j, sentence in enumerate(source):
        role = roles[sentence["from"]]
        if has_image and sentence["value"] is not None and "<image>" in sentence["value"]:
            num_image = len(re.findall(DEFAULT_IMAGE_TOKEN, sentence["value"]))
            texts = sentence["value"].split('<image>')
            _input_id = tokenizer(role).input_ids + nl_tokens 
            for i,text in enumerate(texts):
                _input_id += tokenizer(text).input_ids 
                if i<len(texts)-1:
                    _input_id += [IMAGE_TOKEN_INDEX] + nl_tokens
            _input_id += [im_end] + nl_tokens
            assert sum([i==IMAGE_TOKEN_INDEX for i in _input_id])==num_image
        else:
            if sentence["value"] is None:
                _input_id = tokenizer(role).input_ids + nl_tokens
            else:
                _input_id = tokenizer(role).input_ids + nl_tokens + tokenizer(sentence["value"]).input_ids + [im_end] + nl_tokens
        input_id += _input_id
        if role == "<|im_start|>user":
            _target = [im_start] + [IGNORE_INDEX] * (len(_input_id) - 3) + [im_end] + nl_tokens
        elif role == "<|im_start|>assistant":
            _target = [im_start] + [IGNORE_INDEX] * len(tokenizer(role).input_ids) + _input_id[len(tokenizer(role).input_ids) + 1 : -2] + [im_end] + nl_tokens
        else:
            raise NotImplementedError
        target += _target

    input_ids.append(input_id)
    targets.append(target)
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)
    return input_ids

## load data func
def load_data(args):
    data_folder = args.data_folder
    questions = []
    for dataset_name in os.listdir(data_folder):
        if os.path.isfile(os.path.join(data_folder, dataset_name)): continue
        dataset_folder = os.path.join(data_folder, dataset_name, "full")
        file_name = "valid.json" if args.data_set == "valid" else "train.json"
        data_file_path = os.path.join(dataset_folder, file_name)
    
        with open(data_file_path, "rt") as infile:
            data = json.load(infile)

        cur_questions = data['data']
        # cur_questions = get_chunk(data['data'], args.num_chunks, args.chunk_idx)
        for ques in cur_questions:
            ques['task_instruction'] =  data['metadata']['task_instruction'][ques["task_instruction_id"]]
            ques['metadata'] = {}
            ques['metadata']['dataset'] = dataset_name
            ques['metadata']['question_type'] = data['metadata']['question_type']
            
            ques['task_instance']['images_path'] = [os.path.join(dataset_folder, "images", image_file_name) 
                                       for image_file_name in ques['task_instance']['images_path']]
        questions.extend(cur_questions)
    return questions

def predict_single_question(model, line, ans_file):
    model_name, tokenizer, model, image_processor, context_len = model

    idx = line["sample_id"]
    dataset_name = line["metadata"]["dataset"]
    ground_truth = line["response"]
    question_type = line['metadata']['question_type']

    image_files = line['task_instance']['images_path']
    ques_text = line['task_instruction']
    ques_text += "\n" + line['task_instance']['context']
    if "choice_list" in line['task_instance']:
        ques_text+= "\nChoice List:\n"
        for choice_letter, choice_text in zip(multichoice_letters, line['task_instance']['choice_list']):
            ques_text += choice_letter + ": " + choice_text + "\n"
            # if choice_text == line["response"]:
            #     ground_truth = choice_letter
        
        ques_text += "Answer with the option's letter from the given choices directly."

    ques_text = re.sub(r'\{image#\d+\}', '<image>', ques_text)
    args.conv_mode = "qwen_1_5"
    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], ques_text)
    conv.append_message(conv.roles[1], None)
    
    cur_prompt = conv.get_prompt()

    input_ids = preprocess_qwen([{'from': 'human','value': ques_text},
                                    {'from': 'gpt','value': None}], 
                                tokenizer, has_image=True).cuda()
    img_num = list(input_ids.squeeze()).count(IMAGE_TOKEN_INDEX)

    try:
        image_tensors = []
        for image_file in image_files:
            image = Image.open(image_file)
            image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values']
            image_tensors.append(image_tensor.half().cuda())
        # image_tensors = torch.cat(image_tensors, dim=0)
    except:
        return

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensors,
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            # no_repeat_ngram_size=3,
            max_new_tokens=1024,
            use_cache=True)

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[:-len(stop_str)]
    outputs = outputs.strip()

    if "choice_list" in line['task_instance']:
        choice_list = line['task_instance']['choice_list']
        if outputs in multichoice_letters:
            outputs = choice_list[multichoice_letters.index(outputs.upper())]



    ans_id = shortuuid.uuid()
    ans_file.write(json.dumps({
                            "dataset": dataset_name,
                            "sample_id": idx,
                            "prompt": cur_prompt,
                            "pred_response": outputs,
                            "gt_response": ground_truth,
                            "shortuuid": ans_id,
                            "model_id": model_name,
                            "question_type": question_type,
                            }) + "\n")
    ans_file.flush()

    del input_ids
    del image_tensors
    del output_ids
    torch.cuda.empty_cache()

def eval_model(args):
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)
    
    questions = load_data(args)

    
    directory = os.path.dirname(args.answers_file)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

    print(f"Saving answers to {args.answers_file}")
    ans_file = open(args.answers_file, "w")
    
    for line in tqdm(questions):
        predict_single_question((model_name, tokenizer, model, image_processor, context_len), line, ans_file)
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    default_model_path_05b = "/home/nnguyen/spinning-storage/nnguyen/inova_icme25/foundation_models/llava-next-interleave-qwen-0.5b"
    default_model_path_7b = "/home/nnguyen/spinning-storage/nnguyen/inova_icme25/foundation_models/llava-next-interleave-qwen-7b"
    default_data_folder = "/home/nnguyen/spinning-storage/nnguyen/inova_icme25/Comprehension/"
    parser.add_argument("--model-path", type=str, default=default_model_path_7b)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--data-folder", type=str, default=default_data_folder)
    parser.add_argument("--data-set", type=str, default="valid")
    parser.add_argument("--extra-prompt", type=str, default="")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="qwen_1_5")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--test_size", type=int, default=10000000)
    args = parser.parse_args()

    eval_model(args)


