# -*- coding: utf-8 -*-
# @Time     :2024/10/21 17:19
# @Author   :qihangfu

import logging
import copy
from eot_utils import create_system_prompt, create_user_prompt


logger = logging.getLogger(__name__)

option_ids_idx = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L']

opts2ids = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'J': 9}
cot_answer_msgs = "\nGiven all of the above, the answer of the question is:"


def gain_probability(model_res: any):
    if len(model_res) == 5:
        return sorted([model_res[0][0], model_res[1][0], model_res[2][0], model_res[3][0]])
    if len(model_res) == 4:
        return sorted([model_res[0][0], model_res[1][0], model_res[2][0]])
    if len(model_res) == 3:
        return sorted([model_res[0][0], model_res[1][0]])
    if len(model_res) == 2:
        return [model_res[0][0]]


def prepare_eval_fn_base(args, api, few_shot_samples, num_few_shot):
    def eval_fn(sample):
        idx, (question, options, ideal, category) = sample
        candidate_opt = copy.deepcopy(options)
        temp_ideal = [copy.deepcopy(ideal),]
        option_ids = copy.deepcopy(args.option_ids_header)

        messages, _ = create_system_prompt(category=category)
        for s in few_shot_samples[category][
                 (len(options) - len(candidate_opt)) * 5: num_few_shot + (len(options) - len(candidate_opt)) * 5]:
            messages += s
        # def create_user_prompt(statement, options, answer, num_options=None, exclude_options=None, shuffle=False):
        prompt, temp_ideal, candidate_opt = create_user_prompt(statement=question, options=candidate_opt,
                                                               answer=temp_ideal, num_options=len(candidate_opt))
        messages.append(prompt[0])

        # def generate(self, messages, max_tokens, do_sample=True, temperature=0.0, top_p=0.1):
        result = api(messages=messages, candidate=option_ids[:len(candidate_opt)])
        res_probs_sort = sorted(result, key=lambda x: float(x[1]), reverse=True)
        return {
            'type': 'result',
            'data': {
                'idx': idx,
                'options': options,
                'sampled': res_probs_sort[0][0],
                'ideal': ideal,
                'original_p': res_probs_sort,
                'correct': res_probs_sort[0][0] == ideal,
            },
        }
    return eval_fn


def prepare_eval_fn_cot(args, api, few_shot_samples, num_few_shot):
    def eval_fn(sample):
        idx, (question, options, ideal, category) = sample
        candidate_opt = copy.deepcopy(options)
        temp_ideal = [copy.deepcopy(ideal),]
        option_ids = copy.deepcopy(args.option_ids_header)

        messages, cot_msgs = create_system_prompt(args.prompt_type, category=category)

        if args.prompt_type in ['cot']:
            for s in few_shot_samples[category][
                     (len(options) - len(candidate_opt)) * 5: num_few_shot + (len(options) - len(candidate_opt)) * 5]:
                messages += s

        prompt, temp_ideal, candidate_opt = create_user_prompt(statement=question, options=candidate_opt,
                                                               answer=temp_ideal, num_options=len(candidate_opt))
        messages.append(prompt[0])

        messages += [{'content': cot_msgs[0], 'role': 'assistant', }]
        cot = api.generate(messages=messages, max_tokens=512, do_sample=False, temperature=0.0)
        messages += [{'content': cot, 'role': 'assistant', }]
        messages += [{'content': cot_answer_msgs, 'role': 'user', }]

        result = api(messages=messages, candidate=option_ids[:len(candidate_opt)])

        res_probs_sort = sorted(result, key=lambda x: float(x[1]), reverse=True)
        return {
            'type': 'result',
            'data': {
                'idx': idx,
                'options': options,
                'sampled': res_probs_sort[0][0],
                'ideal': ideal,
                'original_p': res_probs_sort,
                'correct': res_probs_sort[0][0] == ideal,
            },
        }
    return eval_fn


def prepare_eval_fn_eot_base(args, api, few_shot_samples, num_few_shot, threshold, threshold_max=0.8, rounds=3, p=2):
    def eval_fn(sample):
        idx, (question, options, ideal, category) = sample
        candidate_opt = copy.deepcopy(options)
        temp_ideal = [copy.deepcopy(ideal), ]
        option_ids = copy.deepcopy(args.option_ids_header)
        count = copy.deepcopy(rounds)

        original_p = []
        initial_num_options = len(candidate_opt)  # 初始选项数量

        def calculate_dynamic_threshold(current_num_options):
            return threshold + (threshold_max - threshold) * ((1 - current_num_options / initial_num_options) ** p)

        LLMs_ans_history = []
        while True:
            messages, _ = create_system_prompt(args.prompt_type, category=category)
            # adjust the few-shot example based on the number of remaining options
            for s in few_shot_samples[category][
                     (len(options) - len(candidate_opt)) * 5: num_few_shot + (len(options) - len(candidate_opt)) * 5]:
                messages += s

            # construct user prompt
            prompt, temp_ideal, candidate_opt = create_user_prompt(statement=question, options=candidate_opt,
                                                                   answer=temp_ideal, num_options=len(candidate_opt))
            messages.append(prompt[0])

            # get the probability distribution returned by the model
            result = api(messages=messages, candidate=option_ids[:len(candidate_opt)])
            res_probs_sort = sorted(result, key=lambda x: float(x[1]), reverse=True)  # sort by probability in descending order
            if not original_p:
                original_p = res_probs_sort

            LLMs_ans_history.append((temp_ideal, res_probs_sort,))
            delta = float(res_probs_sort[0][1]) - float(res_probs_sort[1][1])
            # calculate the current dynamic threshold
            dynamic_threshold = calculate_dynamic_threshold(len(candidate_opt))

            if delta >= dynamic_threshold:
                return {
                    'type': 'result',
                    'data': {
                        'idx': idx,
                        'options': options,
                        'history_answers': LLMs_ans_history,
                        'correct': res_probs_sort[0][0] == temp_ideal[0],
                    },
                }

            candidate_opt.pop(res_probs_sort[-1][0])
            # speed up the elimination process
            if args.dataset in ['MMLU-PRO']:
                candidate_opt.pop(res_probs_sort[-2][0])

            if len(candidate_opt) == 1 or temp_ideal[0] not in candidate_opt.keys() or count <= 0:
                return {
                    'type': 'result',
                    'data': {
                        'idx': idx,
                        'options': options,
                        'history_answers': LLMs_ans_history,
                        'correct': res_probs_sort[0][0] == temp_ideal[0],
                    },
                }

            count -= 1

    return eval_fn


def prepare_eval_fn_eot_cot(args, api, few_shot_samples, num_few_shot, threshold, threshold_max=0.8, rounds=3, p=2):
    def eval_fn(sample):
        idx, (question, options, ideal, category) = sample
        candidate_opt = copy.deepcopy(options)
        temp_ideal = [copy.deepcopy(ideal), ]
        option_ids = copy.deepcopy(args.option_ids_header)
        count = copy.deepcopy(rounds)

        original_p = []
        initial_num_options = len(candidate_opt)

        def calculate_dynamic_threshold(current_num_options):
            return threshold + (threshold_max - threshold) * ((1 - current_num_options / initial_num_options) ** p)

        LLMs_ans_history = []
        while True:
            messages, cot_msgs = create_system_prompt(args.prompt_type, category=category)

            if args.prompt_type in ['cot']:
                for s in few_shot_samples[category][
                         (len(options) - len(candidate_opt)) * 5: num_few_shot + (len(options) - len(candidate_opt)) * 5]:
                    messages += s

            prompt, temp_ideal, candidate_opt = create_user_prompt(statement=question, options=candidate_opt,
                                                                   answer=temp_ideal, num_options=len(candidate_opt))
            messages.append(prompt[0])

            messages += [{'content': cot_msgs[0], 'role': 'assistant', }]
            cot = api.generate(messages=messages, max_tokens=512, do_sample=False, temperature=0.0)
            messages += [{'content': cot, 'role': 'assistant', }]
            messages += [{'content': cot_answer_msgs, 'role': 'user', }]

            result = api(messages=messages, candidate=option_ids[:len(candidate_opt)])

            res_probs_sort = sorted(result, key=lambda x: float(x[1]), reverse=True)
            if not original_p:
                original_p = res_probs_sort

            LLMs_ans_history.append((temp_ideal, res_probs_sort,))
            delta = float(res_probs_sort[0][1]) - float(res_probs_sort[1][1])

            dynamic_threshold = calculate_dynamic_threshold(len(candidate_opt))

            if delta >= dynamic_threshold:
                return {
                    'type': 'result',
                    'data': {
                        'idx': idx,
                        # 'options': options,
                        'history_answers': LLMs_ans_history,
                        'correct': res_probs_sort[0][0] == temp_ideal[0],
                    },
                }

            candidate_opt.pop(res_probs_sort[-1][0])
            # speed up the elimination process
            if args.dataset in ['MMLU-PRO']:
                candidate_opt.pop(res_probs_sort[-2][0])

            if len(candidate_opt) == 1 or temp_ideal[0] not in candidate_opt.keys() or count <= 0:
                return {
                    'type': 'result',
                    'data': {
                        'idx': idx,
                        # 'options': options,
                        'history_answers': LLMs_ans_history,
                        'correct': res_probs_sort[0][0] == temp_ideal[0],
                    },
                }

            count -= 1
    return eval_fn
