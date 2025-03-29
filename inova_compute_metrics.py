#!/usr/bin/env python
# coding: utf-8
import re
from rouge import Rouge
import argparse
import os
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class Evaluator:
    def __init__(self):
        self.periodStrip = re.compile("(?!<=\d)(\.)(?!\d)")
        self.commaStrip = re.compile("(\d)(\,)(\d)")
        self.punct = [
            ";",
            r"/",
            "[",
            "]",
            '"',
            "{",
            "}",
            "(",
            ")",
            "=",
            "+",
            "\\",
            "_",
            "-",
            ">",
            "<",
            "@",
            "`",
            ",",
            "?",
            "!",
        ]
        
    def processPunctuation(self, inText):
        outText = inText
        for p in self.punct:
            if (p + " " in inText or " " + p in inText) or (
                re.search(self.commaStrip, inText) != None
            ):
                outText = outText.replace(p, "")
            else:
                outText = outText.replace(p, " ")
        outText = self.periodStrip.sub("", outText, re.UNICODE)
        return outText
    
    def process(self, answer):
        answer = answer.replace("\n", " ")
        answer = answer.replace("\t", " ")
        answer = answer.strip()
        answer = self.processPunctuation(answer)
        answer = answer.strip('\'')
        answer = answer.strip('\"')
        answer = answer.strip(')')
        answer = answer.strip('(')
        answer = answer.strip().lower()
        return answer

    def evaluate_rouge(self,preds):
        rouge = Rouge()
        acc = {'f': []}
        eval_list = []
        for i, res in enumerate(preds):
            sample_id = res['sample_id']
            # print(sample_id)
            gt_ans = self.process(res["gt_response"])
            pred_ans = self.process(res["pred_response"])
            # assert gt_ans != ''

            if gt_ans == '':
                continue
            
            if pred_ans == '':
                s = 0
            else:
                if len(pred_ans) > 512:
                    pred_ans = pred_ans[0: 512]
                s = rouge.get_scores(pred_ans, gt_ans)[0]['rouge-l']['f']
            acc['f'].append(s)
            eval_list.append({'id':str(sample_id),'score':str(round(s,3))})
        results = {'Rouge-L f': np.mean(acc['f'])}
        return results,eval_list


    def judge_multi_choice(self,sample):
        sample_id = sample['sample_id']
        gt_ans = sample["gt_response"]
        pred_ans = sample["pred_response"]

        if ":" in pred_ans:
            a_list = pred_ans.split(":")
            a_list = [a.strip() for a in a_list ]
            for a in a_list:
                if len(a) == 1 and a[-1] in ["a", "b", "c", "d", "e", "f", "g", "h"]:
                    pred_ans = a

        if pred_ans == gt_ans:
            return 1
        else:
            return 0

    def process_sample(self,sample):
        sample["gt_response"] = self.process(sample["gt_response"])
        sample["pred_response"] = self.process(sample["pred_response"])

    def evaluate_multichoice(self, preditions):
        correct = 0
        eval_list = []
        for i, sample in enumerate(preditions):
            self.process_sample(sample)
            score = self.judge_multi_choice(sample)
            sample_id = sample['sample_id']
            sample['result'] = score
            eval_list.append({'id':str(sample_id),'score':str(score)})
            correct+=score
        return {'Accuracy':correct/len(preditions)},eval_list

    def evaluate_multi_choice_image(self,preditions):
        correct = 0
        eval_list = []
        for i,sample in enumerate(preditions):
            gt_ans = self.process(sample["gt_response"])
            pred_ans = self.process(sample["pred_response"])
            sample_id = sample['sample_id']

            if ":" in pred_ans:
                a_list = pred_ans.split(":")
                a_list = [a.strip() for a in a_list ]
                for a in a_list:
                    if len(a) == 1 and a[-1] in ["a", "b", "c", "d", "e", "f", "g", "h"]:
                        pred_ans = a

            if gt_ans == pred_ans:
                score = 1
            else:
                score = 0
            sample_id = sample['sample_id']
            sample['result'] = score
            eval_list.append({'id':str(sample_id),'score':str(score)})
            correct+=score
        return {'Accuracy':correct/len(preditions)},eval_list




# spot_the_diff = ["Spot-the-Diff", "Birds-to-Words", "CLEVR-Change"]
# image_edit_instruct = ["IEdit", "HQ-Edit", "MagicBrush"]
# visual_story_telling = ["AESOP", "FlintstonesSV", "PororoSV", "VIST"]
# visual_cloze = ["COMICS_Dialogue", "RecipeQA_VisualCloze"]
# text_rich_vqa = ["WebQA", "TQA", "OCR-VQA", "DocVQA"]
# multi_image_vqa = ["MIT-States_StateCoherence", "MIT-States_PropertyCoherence", "VISION", "RecipeQA_ImageCoherence"]


def read_jsonl_file(file_path):
    records = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue  # skip empty lines
            # Parse the JSON object on the current line
            data = json.loads(line)
            records.append(data)
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--answers-file', type=str, required=True)
    parser.add_argument('--result-dir', type=str, required=True)

    args = parser.parse_args()

    args.result_dir = "result_dir"
    if not os.path.exists(args.result_dir):
        os.mkdir(args.result_dir)
    
    preds_all = read_jsonl_file(args.answers_file)
    
    preds_all_dict = dict()
    for pred in preds_all:
        if pred["dataset"] not in preds_all_dict:
            preds_all_dict[pred["dataset"]] = list()
        preds_all_dict[pred["dataset"]].append(pred)
    
    
    
    evaluator = Evaluator()
    
    eval_result_list = dict()
    eval_result_list_detail = dict()
    for dataset in preds_all_dict:
        preds = preds_all_dict[dataset]
        question_type = preds[0]["question_type"]
    
        if question_type == 'multi-choice':
            eval_result, eval_list = evaluator.evaluate_multichoice(preds)
        elif question_type == 'open-ended':
            eval_result, eval_list = evaluator.evaluate_rouge(preds)
        print(dataset, end = ':  ')
        print(eval_result)
    
        eval_result_list[dataset] = eval_result
        eval_result_list_detail[dataset] = eval_list
    
    os.makedirs(args.result_dir, exist_ok=True)
    with open(os.path.join(args.result_dir, 'eval_dataset.json'), 'w') as f:
        json.dump(eval_result_list, f, indent=4)
    
    with open(os.path.join(args.result_dir,'eval_dataset_details.json'), 'w') as f:
        json.dump(eval_result_list_detail, f, indent=4)



