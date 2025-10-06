import concurrent.futures
from collections import Counter
from typing import List

from verl.utils.reward_score.ttrl.qwen.grader import \
    math_equal as qwen_math_equal
from verl.utils.reward_score.ttrl.qwen.math_grade import grade_answer
from verl.utils.reward_score.ttrl.qwen.qwen_math_parser import extract_answer

import numpy as np
import re

def parse_bbox(text_or_bbox):
    """
    Parses bounding box from string or returns directly if already a bbox tuple/list.
    Returns tuple (x1, y1, x2, y2) or None if invalid.
    """
    # If it's already a list or tuple of 4 numbers, return it directly
    if isinstance(text_or_bbox, (tuple, list)) and len(text_or_bbox) == 4:
        try:
            # Ensure all elements are numeric
            x1, y1, x2, y2 = map(float, text_or_bbox)
            return (x1, y1, x2, y2)
        except (TypeError, ValueError):
            return None

    # Otherwise treat as string and attempt to parse
    if not isinstance(text_or_bbox, str):
        return None

    cleaned = re.sub(r"[^\d.,\s\-.]", "", text_or_bbox.replace(" ", ""))
    parts = re.split(r"[,\s]+", cleaned)

    if len(parts) < 4:
        return None

    try:
        x1, y1, x2, y2 = map(float, parts[:4])
        return (x1, y1, x2, y2)
    except (ValueError, TypeError):
        return None


def reward_spatial(pred_box, consensus_box):
    """
    Spatial reward function based on Euclidean distance between corners.
    Returns higher reward when boxes are closer.
    """
    if pred_box is None or consensus_box is None:
        return 0.0  # Or -1.0 for stronger penalty on invalid output

    x1a, y1a, x2a, y2a = pred_box
    x1b, y1b, x2b, y2b = consensus_box

    d1 = np.sqrt((x1a - x1b)**2 + (y1a - y1b)**2)
    d2 = np.sqrt((x2a - x2b)**2 + (y2a - y2b)**2)
    return 1 / (1 + d1 + d2)  # Higher reward if closer

def reward_iou(pred_box, consensus_box):
    if pred_box is None or consensus_box is None:
        return 0.0

    x1a, y1a, x2a, y2a = pred_box
    x1b, y1b, x2b, y2b = consensus_box

    # Ensure coordinates are ordered properly
    x1i = max(x1a, x1b)
    y1i = max(y1a, y1b)
    x2i = min(x2a, x2b)
    y2i = min(y2a, y2b)

    inter_area = max(0, x2i - x1i) * max(0, y2i - y1i)

    area_a = (x2a - x1a) * (y2a - y1a)
    area_b = (x2b - x1b) * (y2b - y1b)

    union_area = area_a + area_b - inter_area
    iou = inter_area / union_area if union_area > 0 else 0.0

    return iou

def qwen_reward_fn_spatial(generated_text, golden_answer):
    """
    Reward function for bounding box prediction tasks.
    Handles both string and tuple/list formats for input.
    Returns 0.0 if either cannot be parsed into a valid bbox.
    """
    pred_box = parse_bbox(generated_text)
    consensus_box = parse_bbox(golden_answer)

    return reward_iou(pred_box, consensus_box)

def qwen_reward_fn(generated_text, golden_answer, task="math"):
    model_answer = extract_answer(generated_text, task)
    accuracy = 1.0 if grade_answer(model_answer, golden_answer) else 0.0 #-0.5
    # if "boxed" not in generated_text:
    #     accuracy = -1.0
    return accuracy


def qwen_reward_fn_gpqa(generated_text, golden_answer, task="gpqa"):
    model_answer = extract_answer(generated_text, task)
    accuracy = 1.0 if grade_answer(model_answer, golden_answer) else 0.0 #-0.5
    # if "boxed" not in generated_text:
    #     accuracy = -1.0
    return accuracy



# ------------------------------------------------------------
# import math

# def match_answer(response):
#     is_matched = False
#     ans_marker = 'The answer is: '
#     ans_idx = response.lower().rfind(ans_marker)
#     if ans_idx != -1:
#         is_matched = True
#         response = response[ans_idx + len(ans_marker):].strip()
#         if response.endswith("\n"):
#             response = response[:-2]

#     ans_marker = 'answer:\n'
#     ans_idx = response.lower().rfind(ans_marker)
#     if ans_idx != -1:
#         is_matched = True
#         response = response[ans_idx + len(ans_marker):].strip()
#         if response.endswith("\n"):
#             response = response[:-2]

#     ans_marker = 'answer: '
#     ans_idx = response.lower().rfind(ans_marker)
#     if ans_idx != -1:
#         is_matched = True
#         response = response[ans_idx + len(ans_marker):].strip()
#         if response.endswith("\n"):
#             response = response[:-2]

#     # Find boxed
#     ans_boxed = _last_boxed_only_string(response)
#     if ans_boxed:
#         is_matched = True
#         response = ans_boxed

#     # Grade
#     return is_matched, response


# def _last_boxed_only_string(string):
#     idx = string.rfind("\\boxed")
#     if idx < 0:
#         idx = string.rfind("\\fbox")
#         if idx < 0:
#             return None

#     i = idx
#     left_brace_idx = None
#     right_brace_idx = None
#     num_left_braces_open = 0
#     while i < len(string):
#         if string[i] == "{":
#             num_left_braces_open += 1
#             if left_brace_idx is None:
#                 left_brace_idx = i
#         elif string[i] == "}":
#             num_left_braces_open -= 1
#             if num_left_braces_open == 0:
#                 right_brace_idx = i
#                 break

#         i += 1

#     if left_brace_idx is None or right_brace_idx is None:
#         return None

#     return string[left_brace_idx + 1: right_brace_idx].strip()

# def qwen_reward_fn(generated_text, golden_answer, task="math"):
#     if golden_answer in ["A", "B", "C", "D"]:
#         task = "gpqa"
#         model_answer = extract_answer(generated_text, task)
#         accuracy = 1.0 if grade_answer(model_answer, golden_answer) else 0.0 #-0.5
#         # if "boxed" not in generated_text:
#         #     accuracy = -1.0
#         return accuracy

#     answer = golden_answer.lstrip('0') 
#     is_matched, model_output = match_answer(generated_text)
#     model_output = model_output.strip("The final answer is ").strip(". I hope it is correct.")
#     try:
#         if "\pi" in model_output or "\pi" in golden_answer:
#             equivs = []
#             for pi in [math.pi, 3.14]:
#                 equivs.append(math_equal(model_output, answer, timeout=True, pi=pi))
#             equiv = any(equivs)
#         else:
#             equiv = math_equal(model_output, answer, timeout=True)
#     except:
#         equiv = False

#     if equiv:
#         return 1.0
#     else:
#         return 0.0

# ------------------------------------------------------------

def majority_vote(
    solutions: List[str],
    ground_truth: str,
    task="math"
):
    model_answers = [extract_answer(generated_text, task) for generated_text in solutions]
    model_answers = [answer for answer in model_answers if answer is not None]

    if len(model_answers) == 0:
        return 0
    
    counter = Counter(model_answers)
    
    majority_answer, _ = counter.most_common(1)[0]
    accuracy = 1.0 if grade_answer(majority_answer, ground_truth) else 0
    # if "boxed" not in generated_text:
    #     accuracy = -1.0
    return accuracy

def qwen_math_equal_subprocess(prediction, reference, timeout_seconds=10):
    with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
        future = executor.submit(qwen_math_equal, prediction=prediction, reference=reference, timeout=False)
        try:
            return future.result(timeout=timeout_seconds)
        except concurrent.futures.TimeoutError:
            return False

def simplerl_reward_fn(generated_text, golden_answer):
    model_answer = extract_answer(generated_text, "math")
    accuracy = 1.0 if qwen_math_equal_subprocess(prediction=model_answer, reference=golden_answer) else -0.5
    if "boxed" not in generated_text:
        accuracy = -1.0
    return accuracy