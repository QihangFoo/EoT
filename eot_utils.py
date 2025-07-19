# -*- coding: utf-8 -*-
# @Time     :2024/10/21 21:26
# @Author   :qihangfu

import json
import logging
import random
from tqdm import tqdm
from multiprocessing.pool import ThreadPool

logger = logging.getLogger(__name__)


def create_user_prompt(statement, options, answer, num_options=None, exclude_options=None, shuffle=False):
    # Extract necessary information from the input data
    correct_answer_ids = answer.copy()
    options = options.copy()

    # Remove the specified options if exclude_options are provided and num_options is not specified
    if exclude_options and not num_options:
        for opt in exclude_options:
            if opt in options:
                del options[opt]

    # Ensure that we have the required number of options, including the correct one, if num_options is provided
    elif num_options and not exclude_options:
        if len(options) > num_options:
            # Keep the correct answers and randomly remove others until we have num_options left
            correct_set = set(correct_answer_ids)
            remaining_options = [key for key in options.keys() if key not in correct_set]
            while len(correct_set) + len(remaining_options) > num_options:
                remaining_options.pop()
            options = {key: options[key] for key in sorted(list(correct_set) + remaining_options)}

    # Update the correct answer ID list based on remaining options
    correct_answer_ids = [key for key in correct_answer_ids if key in options]

    # Shuffle the options if needed
    if shuffle:
        shuffled_items = list(options.items())
        random.shuffle(shuffled_items)
        options = dict(shuffled_items)

    # Reassign new keys in alphabetical order after shuffling
    new_keys = sorted(options.keys())
    updated_options = {}
    key_mapping = {}
    for i, (old_key, value) in enumerate(options.items()):
        new_key = chr(65 + i)  # 'A', 'B', 'C', ...
        updated_options[new_key] = value
        key_mapping[old_key] = new_key

    # Update the correct answer ID list to reflect new keys
    correct_answer_ids = [key_mapping[old_key] for old_key in correct_answer_ids if old_key in key_mapping]

    # Generate the chat message in the required format
    question_text = f"{statement}\n"
    for key, value in updated_options.items():
        question_text += f"{key}: {value}\n"

    # Prepare the chat message structure
    chat_message = [
        {"role": "user", "content": f"Question: {statement.strip()}\nOptions:\n" + \
                                    "\n".join([f"{option_id}. {value}".strip()
                                               for option_id, value in updated_options.items()]) + \
                                    "\n" + 'Answer:'},
        {"role": "assistant", "content": ''.join(correct_answer_ids)},
    ]

    # Return the chat message, the correct answer ID, and the current options in the new shuffled order
    return chat_message, correct_answer_ids, updated_options


def create_system_prompt(sys_prompt_type=None, category=None):
    # Standard Prompting
    standard_sys_msg = ('The following are multiple choice questions about {}.' + \
              'Your output should only include options, and nothing else.', "")
    cot_sys_msg = ('The following are multiple choice questions about {},You should reason in a step-by-step manner as to get the right answer.', "Let's think step by step:")
    zero_shot_cot_sys_msg = ('Answer the following multiple-choice question about {}. Think step by step before giving your final answer.', "Let's think through this carefully:")
    complex_cot_sys_msg = ("You'll solve this multiple-choice questions about {}. \n"+\
                            "First, You'll break down the question and identify key information.\n" +\
                            "Next, You'll evaluate each option using relevant knowledge and logical reasoning.\n" +\
                            "For each option, You'll determine whether it's correct or incorrect and explain why.\n",
                            "Finally, You'll select the option ID that represents the correct answer.")
    msg = ''
    if sys_prompt_type == 'standard':
        msg = standard_sys_msg
    elif sys_prompt_type == 'cot':
        msg = cot_sys_msg
    elif sys_prompt_type == 'zero_shot_cot':
        msg = zero_shot_cot_sys_msg
    elif sys_prompt_type == 'complex_cot':
        msg = complex_cot_sys_msg
    if not msg:
        raise SystemExit('Not set prompt_type!')
    sys_msg = msg[0].format(category)
    prompt = [
        {"role": "system", "content": sys_msg},
    ]
    return prompt, msg[1]


def get_accuracy(data):
    results = [int(e['data']["correct"]) for e in data if e['type'] == 'result']
    num_correct = sum(results)
    num_total = len(results)
    if num_total == 0:
        return float("nan")
    else:
        return num_correct / num_total


def save_results(save_file_path, results: list, metrics: dict = None):
    while True:
        try:
            with open(save_file_path, 'w') as f:
                for result in results + ([metrics] if metrics is not None else []):
                    f.write(json.dumps(result) + '\n')
            break
        except OSError as e:
            logger.info(f"OSError: {e}. Retrying.")
            continue


def _index_samples(samples, max_num_samples=None, remaining=False):
    indices = list(range(len(samples)))
    random.Random(123).shuffle(indices)
    if max_num_samples is not None:
        if remaining:
            indices = indices[max_num_samples:]
        else:
            indices = indices[:max_num_samples]
    logger.info(f"Evaluating {len(indices)} samples")
    work_items = [(idx, samples[idx]) for idx in sorted(indices)]
    return work_items


def eval_all_samples(eval_fn, eval_samples, name=None, threads=5):
    work_items = _index_samples(eval_samples)
    def eval_sample(args):
        """
        Evaluate a single sample.
        """
        # idx, _ = args
        # seed = f"{idx}:20230101".encode("utf-8")
        # rng = random.Random(seed)
        return eval_fn(args)

    while True:
        try:
            with ThreadPool(threads) as pool:
                if threads > 1:
                    logger.info(f"Running in threaded mode with {threads} threads!")
                    iter = pool.imap_unordered(eval_sample, work_items)
                else:
                    logger.info(f"Running in sequential mode!")
                    iter = map(eval_sample, work_items)
                results = list(tqdm(iter, total=len(work_items), dynamic_ncols=True, desc=name))
            break

        except RuntimeError as e:
            if threads > 1:
                threads = threads // 2
                logger.info(f"RuntimeError: {e}. Retrying with {threads} threads.")
                # torch.cuda.empty_cache()
                continue
            else:
                raise e

    results = sorted(results, key=lambda x: x['data']['idx'])
    return results
