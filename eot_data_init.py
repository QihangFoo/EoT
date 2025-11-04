# -*- coding: utf-8 -*-
# @Time     :2024/10/21 17:18
# @Author   :qihangfu

import os
import json
from eot_utils import create_user_prompt
from functools import partial
from eot_process import (
    prepare_eval_fn_base,
    prepare_eval_fn_eot_base,
    prepare_eval_fn_eot_cot,
    prepare_eval_fn_cot
)


def prepare_eval(args):
    # task and setting
    args.task = task = args.task.lower()
    args.num_few_shot = num_few_shot = args.num_few_shot
    args.setting = setting = args.setting

    # save_path
    save_path = f'results_{task}/{setting}/{task}'
    if setting is not None:
        save_path += f'_{setting}_{args.model_name}_{args.threshold}'
    args.save_path = save_path
    os.makedirs(args.save_path, exist_ok=True)

    # set options
    if task in ['csqa', 'aqua']:
        args.option_ids_header = ['A', 'B', 'C', 'D', 'E']
    elif task in ['mmlu_pro']:
        args.option_ids_header = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
    else:
        args.option_ids_header = ['A', 'B', 'C', 'D']

    # check dataset
    data_path = f'data/data_{task}'
    subjects = sorted([f.split("_test.jsonl")[0] for f in os.listdir(f'{data_path}/test') if "_test.jsonl" in f])

    def prepare_few_shot_samples_exclude(subject):
        dev_samples = []
        with open(f'{data_path}/dev/{subject}_dev.jsonl', 'r', encoding='utf-8') as fp:
            for line in fp:
                temp = json.loads(line)
                dev_samples.append(temp)

        # Build missing options example
        option_len = len(args.option_ids_header)
        class_category = {}
        if 'category' in dev_samples[0]:
            for sample in dev_samples:
                if sample['category'] not in class_category:
                    class_category[sample['category']] = [sample, ]
                else:
                    class_category[sample['category']].append(sample)
        else:
            class_category[subject] = dev_samples

        few_shot_samples = {}
        for i in range(option_len, 1, -1):
            for class_sample, dev_sample in class_category.items():
                for sample in dev_sample[:5]:
                    prompt, _, _ = create_user_prompt(statement=sample['statement'], options=sample['option_list'],
                                                      answer=sample['answer'], num_options=i)
                    if class_sample not in few_shot_samples:
                        few_shot_samples[class_sample] = [prompt]
                    else:
                        few_shot_samples[class_sample].append(prompt)

        return few_shot_samples

    # prepare_eval_samples
    def prepare_eval_samples(subject):
        options, ideals, question, categorys = [], [], [], []
        with open(f'{data_path}/test/{subject}_test.jsonl', 'r', encoding='utf-8') as fp:
            for one in fp.readlines():
                if len(one) > 10:
                    temp = json.loads(one)
                    question.append(temp['statement'])
                    ideals.append(temp['answer'][0])
                    options.append(temp['option_list'])
                    categorys.append(temp['category'] if 'category' in temp else subject)

        return list(zip(question, options, ideals, categorys))

    if args.setting in ['base']:
        if args.prompt_type in ['standard']:
            prepare_eval_fn = partial(prepare_eval_fn_base, num_few_shot=num_few_shot)
        else:
            prepare_eval_fn = partial(prepare_eval_fn_cot, num_few_shot=num_few_shot)
    elif args.setting in ['eot']:
        if args.prompt_type in ['standard']:
            prepare_eval_fn = partial(prepare_eval_fn_eot_base, num_few_shot=num_few_shot, threshold=args.threshold)
        else:
            prepare_eval_fn = partial(prepare_eval_fn_eot_cot, num_few_shot=num_few_shot, threshold=args.threshold)
    else:
        raise NotImplementedError

    return subjects, prepare_few_shot_samples_exclude, prepare_eval_samples, prepare_eval_fn
