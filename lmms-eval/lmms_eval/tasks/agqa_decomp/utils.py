import datetime
import json
import os
import re
import sys
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Union

import cv2
import datasets
import numpy as np
import yaml
from loguru import logger as eval_logger

from lmms_eval.llm_judge import ServerConfig, get_server, Request

with open(Path(__file__).parent / "agqa_decomp.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        # remove function definition since yaml load cannot handle it
        if "!function" not in line:
            safe_data.append(line)
cache_dir = yaml.safe_load("".join(safe_data))["dataset_kwargs"]["cache_dir"]


def agqa_doc_to_visual(doc):
    video_path = doc["video_id"] + ".mp4"
    video_path = os.path.join(cache_dir, "Charades_v1", video_path)
    if os.path.exists(video_path):
        video_path = video_path
    elif os.path.exists(video_path.replace("mp4", "MP4")):
        video_path = video_path.replace("mp4", "MP4")
    elif os.path.exists(video_path.replace("mp4", "mkv")):
        video_path = video_path.replace("mp4", "mkv")
    else:
        sys.exit(f"video path:{video_path} does not exist, please check")
    return [video_path]


def agqa_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    # pre_prompt = lmms_eval_specific_kwargs["pre_prompt"] if "pre_prompt" in lmms_eval_specific_kwargs else \
    #     "Give a short answer to the question based on the video. Respond with a simple word or phrase."
    # question = doc["question"]
    # post_prompt = lmms_eval_specific_kwargs["post_prompt"] if "post_prompt" in lmms_eval_specific_kwargs else "The best answer is:"
    # model_input = {
    #     'preprompt': pre_prompt,
    #     'postprompt': post_prompt,
    #     'question': question,
    #     "hierarchy": json.loads(doc["hierarchy"]) if isinstance(doc["hierarchy"], str) else doc["hierarchy"],
    #     'subquestions': json.loads(doc["subquestions"]) if isinstance(doc["subquestions"], str) else doc["subquestions"],
    #     'answer': doc["answer"],
    #     'video_id': doc["video_id"],
    #     'question_id': doc["question_id"],
    # }
    # return model_input
    return 'DUMMY_TEXT'

def extract_characters_regex(s):
    s = s.strip()
    answer_prefixes = [
        "The best answer is",
        "The correct answer is",
        "The answer is",
        "The answer",
        "The best option is" "The correct option is",
        "Best answer:" "Best option:",
    ]
    for answer_prefix in answer_prefixes:
        s = s.replace(answer_prefix, "")
    return s


matrices = [
    'Accuracy', 'Score', 'CP', 'CR', 'CF1', 'NCP', 'NCR', 'NCF1',
    'Sub-Accuracy', 'Sub-Score', 'Sub-CP', 'Sub-CR', 'Sub-CF1', 'Sub-NCP', 'Sub-NCR', 'Sub-NCF1',
    'Main-Accuracy', 'Main-Score', 'Main-CP', 'Main-CR', 'Main-CF1', 'Main-NCP', 'Main-NCR', 'Main-NCF1',
    'Binary-Accuracy', 'Binary-Score', 'Binary-CP', 'Binary-CR', 'Binary-CF1', 'Binary-NCP', 'Binary-NCR', 'Binary-NCF1',
    'OE-Accuracy', 'OE-Score', 'OE-CP', 'OE-CR', 'OE-CF1', 'OE-NCP', 'OE-NCR', 'OE-NCF1',
]


config = ServerConfig(
    model_name='gemini-2.5-pro',
    temperature=0.0,
    max_tokens=2048,
    judge_type='general',
    response_format='json',
    num_retries=20,
    max_concurrent=20,
)
server = get_server(server_name='openai', config=config)
def eval_answer(question, model_output, gt):
    response = server.evaluate(
        request=Request(
            messages=[
                {"role": "system", "content": "You're a helpful assistant. Please evaluate if the following two answers to the questions are the same."},
                {
                    "role": "user",
                    "content": (
                        f"Question: {question}\nAnswer1: {model_output}\nAnswer2: {gt}"
                        f"Respond with the following JSON format:"
                        "{"
                        "    'result': '1' if the two answers are the same else '0',"
                        "    'score': an integer form 0 to 5, where 0 means the two answers are most different, 5 means the two answers are the most similar."
                        "}"
                    )
                },
            ],
            config=config,
        )
    )
    try:
        return {
            k: eval(v) if isinstance(v, str) else v
            for k, v in json.loads(response.content).items()
        } if response.success else {"result": -1, "score": -1}
    except Exception as e:
        print(f"Error evaluation server response: \n {response}\nError: \n", e, '\n' + '-' * 100)
        return {'result": -1, "score": -1'}



def agqa_process_results(doc, results):
    """
    Args:
        doc: a instance of the eval dataset
        results: [pred]
    Returns:
        a dictionary with key: metric name (in this case agqa score), value: metric value
    """
    metrics = defaultdict(list)


    pred = results[0]
    # gt_ans = doc["answer"].lower().strip().replace(".", "")

    main_answer = doc['answer']
    eval = eval_answer(doc['question'], pred, main_answer)

    sub_answers: Dict[str,str] = {k: v['answer'] for k, v in json.loads(doc['subquestions']).items()}

    rk1_graph: Dict[str, List[str]] = doc['rank1_graph']
    main_pred, sub_preds = pred['main_answer'], pred['sub_answers']
    main_pred = extract_characters_regex(main_pred)
    sub_preds = {k: extract_characters_regex(v) for k, v in sub_preds.items()}
    eval_results = {
        doc['main_key']: eval_answer(doc['question'], main_pred, main_answer),
        **{
            k: 1 if eval_answer(doc['question'], v, sub_answers[k]) else 0
            for k, v in sub_preds.items()
        }
    }
    accs = {k: v['result'] for k, v in eval_results.items()}
    scores = {k: v['score'] for k, v in eval_results.items()}

    metrics['main_acc'] = [accs[doc['main_key']]]
    metrics['main_score'] = [scores[doc['main_key']]]
    metrics['sub_acc'] = [v for k, v in accs.items() if k != doc['main_key']]
    metrics['sub_score'] = [v for k, v in scores.items() if k != doc['main_key']]

    for root_key, leaf_keys in rk1_graph.items():
        is_main = root_key == doc["main_key"]
        prefix_ms = "Main-" if is_main else "Sub-"
        is_binary = doc['answers'][root_key] in {'after', 'before', 'yes', 'no'}
        prefix_bo = "Binary-" if is_binary else "OE-"
        prefixes = [prefix_ms, prefix_bo, '']

        root_acc = accs[root_key]
        root_score = scores[root_key]
        if leaf_keys is not None:
            sub_acc = all(accs[k] for k in leaf_keys)
            if root_acc and sub_acc:
                postfixes =['CP', 'CR']
                value = 1
            elif root_acc and not sub_acc:
                postfixes = ['CP', 'NCR']
                value = 0
            elif not root_acc and sub_acc:
                postfixes = ['NCP', 'CR']
                value = 0
            elif not root_acc and not sub_acc:
                postfixes = ['NCP', 'NCR']
                value = 1
            else:
                raise RuntimeError("root_acc and sub_acc error. This line should never be triggered.")
        else:
            postfixes, value = [], None
        for prefix in prefixes:
            for postfix in postfixes:
                metrics[f"{prefix}{postfix}"].append(value)
            metrics[f'{prefix}Accuracy'].append(root_acc)
            metrics[f'{prefix}Score'].append(root_score)

    return {f"agqa_score": metrics}


def agqa_aggregate_results(results):
    """
    Aggregate AGQA results across the dataset

    Args:
        results: List of metric dictionaries from process_results

    Returns:
        Aggregated scores
    """
    # Flatten all metrics
    all_metrics = defaultdict(list)
    for result in results:
        if "agqa_score" in result:
            for metric_name, values in result["agqa_score"].items():
                all_metrics[metric_name].extend(values)

    # Calculate averages
    aggregated = {}
    for metric_name, values in all_metrics.items():
        if values:
            # Filter out invalid values (-1)
            valid_values = [v for v in values if v != -1]
            if valid_values:
                aggregated[metric_name] = sum(valid_values) / len(valid_values)
            else:
                aggregated[metric_name] = 0.0

    # Calculate overall score (average of main metrics)
    main_metrics = ['Accuracy', 'Score', 'CF1']
    overall_scores = [aggregated.get(m, 0.0) for m in main_metrics if m in aggregated]
    if overall_scores:
        aggregated['overall'] = sum(overall_scores) / len(overall_scores)

    return aggregated

